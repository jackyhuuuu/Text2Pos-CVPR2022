import numpy as np
import os
import os.path as osp
import cv2
from easydict import EasyDict

import torch
from torch.utils.data import DataLoader
import time

from training.plots import plot_metrics
from training.losses import calc_pose_error
from evaluation.args import parse_arguments
from evaluation.utils import calc_sample_accuracies, print_accuracies

from dataloading.kitti360.cells import Kitti360CoarseDataset, Kitti360CoarseDatasetMulti
# from dataloading.kitti360.poses import Kitti360PoseReferenceDatasetMulti, Kitti360PoseReferenceDataset
from dataloading.kitti360.eval import Kitti360TopKDataset

from datapreparation.kitti360.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_TEST

from training.coarse import eval_epoch as eval_epoch_retrieval
from models.superglue_matcher import get_pos_in_cell

import torch_geometric.transforms as T 

'''
TODO:
- Fine: which objects to cut-off? Just pad_size=32 not different.
- How to handle orientation predictions?
'''

@torch.no_grad()
def run_coarse(model, dataloader, args):
    """Run text-to-cell retrieval to obtain the top-cells and coarse pose accuracies.  

    Args:
        model: retrieval model
        dataloader: retrieval dataset
        args: global arguments

    Returns:
        [List]: retrievals as [(cell_indices_i_0, cell_indices_i_1, ...), (cell_indices_i+1, ...), ...] with i ∈ [0, len(poses)-1], j ∈ [0, max(top_k)-1]
        [Dict]: accuracies
    """
    model.eval()

    # Run retrieval model to obtain top-cells
    retrieval_accuracies, retrievals = eval_epoch_retrieval(model, dataloader, args)
    retrievals = [retrievals[idx] for idx in range(len(retrievals))] # Dict -> list
    print('Retrieval Accs:')
    print(retrieval_accuracies)
    assert len(retrievals) == len(dataloader.dataset.all_poses)

    all_cells_dict = {cell.id: cell for cell in dataloader.dataset.all_cells}

    # Gather the accuracies for each sample
    accuracies = {k: {t: [] for t in args.threshs} for k in args.top_k}
    for i_sample in range(len(retrievals)):
        pose = dataloader.dataset.all_poses[i_sample]
        top_cells = [all_cells_dict[cell_id] for cell_id in retrievals[i_sample]]
        pos_in_cells = 0.5 * np.ones((len(top_cells), 2)) # Predict cell-centers
        accs = calc_sample_accuracies(pose, top_cells, pos_in_cells, args.top_k, args.threshs)

        for k in args.top_k:
            for t in args.threshs:
                accuracies[k][t].append(accs[k][t])

    for k in args.top_k:
        for t in args.threshs:
            accuracies[k][t] = np.mean(accuracies[k][t])

    return retrievals, accuracies

def get_confidences(P):
    assert len(P.shape) == 3 # [batch_size, objects, hints]
    return np.sum(P[:, 0:-1, 0:-1], axis=(1,2))
  
@torch.no_grad()
def eval_conf(model, dataset):
    scores = []
    for i_sample in range(10):
        confs = []
        idx = np.random.randint(len(dataset))
        data = Kitti360CoarseDataset.collate_fn([dataset[idx], ])
        output = model(data['objects'], data['debug_hint_descriptions'], data['object_points'])
        print(output.P[0])
        conf = get_confidences(output.P.detach().cpu().numpy())
        assert len(conf) == 1
        confs.append(conf[0])

        for idx in np.random.randint(len(dataset), size=4):
            data = Kitti360CoarseDataset.collate_fn([dataset[idx], ])
            output = model(data['objects'], data['debug_hint_descriptions'], data['object_points'])
            print(output.P[0])
            conf = get_confidences(output.P.detach().cpu().numpy())
            assert len(conf) == 1
            confs.append(conf[0])

        print(confs)
        print('\n --- \n')
        

        scores.append(np.argmax(confs) == 0)
    print('Score:', np.mean(scores))


@torch.no_grad()
def run_fine(model, retrievals, dataloader, args):
    # A batch in this dataset contains max(top_k) times the pose vs. each of the max(top_k) top-cells.
    dataset_topk = Kitti360TopKDataset(dataloader.dataset.all_poses, dataloader.dataset.all_cells, retrievals, transform, args)
    dataloader_topk = DataLoader(dataset_topk, batch_size=6, collate_fn=Kitti360TopKDataset.collate_extend)

    num_samples = max(args.top_k)

    t0 = time.time()
    # Obtain the matches, offsets and confidences for each pose vs. its top-cells
    # TODO: Speed this up!
    matches = []
    offsets = []
    confidences = []
    cell_ids = []
    poses_w = []
    if not args.use_batching:
        for i_sample, sample in enumerate(dataset_topk):
            output = model(sample['objects'], sample['hint_descriptions'], sample['object_points'])
            matches.append(output.matches0.detach().cpu().numpy())
            offsets.append(output.offsets.detach().cpu().numpy())
            confs = get_confidences(output.P.detach().cpu().numpy())
            assert len(confs) == num_samples
            confidences.append(confs)
            
            cell_ids.append([cell.id for cell in sample['cells']])
            poses_w.append(sample['poses'][0].pose_w)
    else:
        for i_batch, batch in enumerate(dataloader_topk):
            output = model(batch['objects'], batch['hint_descriptions'], batch['object_points'])
            batch_size = len(batch['poses'])
            assert batch_size % num_samples == 0
            for i_sample in range(batch_size // num_samples):
                matches.append(output.matches0.detach().cpu().numpy()[i_sample * num_samples : (i_sample + 1) * num_samples])
                offsets.append(output.offsets.detach().cpu().numpy()[i_sample * num_samples : (i_sample + 1) * num_samples])

                cell_ids.append([cell.id for cell in batch['cells'][i_sample * num_samples : (i_sample + 1) * num_samples]])
                poses_w.append(batch['poses'][i_sample * num_samples].pose_w)

    assert len(matches) == len(offsets) == len(retrievals)
    cell_ids = np.array(cell_ids)

    t1 = time.time()
    print('ela:', t1-t0)

    all_cells_dict = {cell.id: cell for cell in dataloader.dataset.all_cells}

    # Gather the accuracies for each sample
    accuracies_mean = {k: {t: [] for t in args.threshs} for k in args.top_k}
    accuracies_offset = {k: {t: [] for t in args.threshs} for k in args.top_k}
    accuracies_mean_conf = {1: {t: [] for t in args.threshs}}
    for i_sample in range(len(retrievals)):
        pose = dataloader.dataset.all_poses[i_sample]
        top_cells = [all_cells_dict[cell_id] for cell_id in retrievals[i_sample]]
        sample_matches = matches[i_sample]
        sample_offsets = offsets[i_sample]
        sample_confidences = confidences[i_sample]

        if not np.all(np.array([cell.id for cell in top_cells]) == cell_ids[i_sample]):
            print()
            print([cell.id for cell in top_cells])
            print(cell_ids[i_sample])

        assert np.all(np.array([cell.id for cell in top_cells]) == cell_ids[i_sample])
        assert np.allclose(pose.pose_w, poses_w[i_sample])
        
        # Get objects, matches and offsets for each of the top-cells
        pos_in_cells_mean = []
        pos_in_cells_offsets = []
        for i_cell in range(len(top_cells)):
            cell = top_cells[i_cell]
            cell_matches = sample_matches[i_cell]
            cell_offsets = sample_offsets[i_cell]
            pos_in_cells_mean.append(get_pos_in_cell(cell.objects, cell_matches, np.zeros_like(cell_offsets)))
            pos_in_cells_offsets.append(get_pos_in_cell(cell.objects, cell_matches, cell_offsets))
        pos_in_cells_mean = np.array(pos_in_cells_mean)
        pos_in_cells_offsets = np.array(pos_in_cells_offsets)

        accs_mean = calc_sample_accuracies(pose, top_cells, pos_in_cells_mean, args.top_k, args.threshs)
        accs_offsets = calc_sample_accuracies(pose, top_cells, pos_in_cells_offsets, args.top_k, args.threshs)
        
        conf_idx = np.argmax(sample_confidences)
        accs_mean_conf = calc_sample_accuracies(pose, top_cells[conf_idx : conf_idx+1], pos_in_cells_mean[conf_idx : conf_idx+1], top_k=[1,], threshs=args.threshs)

        for k in args.top_k:
            for t in args.threshs:
                accuracies_mean[k][t].append(accs_mean[k][t])        
                accuracies_offset[k][t].append(accs_offsets[k][t])
                accuracies_mean_conf[1][t].append(accs_mean_conf[1][t])

    for k in args.top_k:
        for t in args.threshs:
            accuracies_mean[k][t] = np.mean(accuracies_mean[k][t])
            accuracies_offset[k][t] = np.mean(accuracies_offset[k][t])
            accuracies_mean_conf[1][t] = np.mean(accuracies_mean_conf[1][t])

    return accuracies_mean, accuracies_offset, accuracies_mean_conf



@torch.no_grad()
def depr_run_fine(model, dataloader):
    raise Exception("Not udpated yet!")
    offsets = []
    matches0 = []    
    for i_batch, batch in enumerate(dataloader):
        output = model(batch['objects'], batch['hint_descriptions'], batch['object_points'])
        offsets.append(output.offsets.detach().cpu().numpy())
        matches0.append(output.matches0.detach().cpu().numpy())
    return np.vstack((offsets)), np.vstack((matches0))        

'''
- Eval accuracies directly in run_matching(), rename run_coarse(), run_fine()
- Use TopK-Dataset, for now no DataLoader
'''

if __name__ == '__main__':
    args = parse_arguments()
    print(str(args).replace(',','\n'), '\n')

    # Load datasets
    transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])
    # dataset_retrieval = Kitti360CellDatasetMulti(args.base_path, SCENE_NAMES_TEST, transform, split=None)
    # dataset_matching = Kitti360PoseReferenceDatasetMulti(args.base_path, SCENE_NAMES_TEST, transform, args, split=None)
    # assert len(dataset_retrieval) == len(dataset_matching) # If poses and cells become separate, this will need dedicated handling
    
    # dataloader_retrieval = DataLoader(dataset_retrieval, batch_size=args.batch_size, collate_fn=Kitti360CellDataset.collate_fn)
    # dataloader_matching = DataLoader(dataset_matching, batch_size=args.batch_size, collate_fn=Kitti360PoseReferenceDataset.collate_fn)

    # dataset_retrieval = Kitti360CoarseDatasetMulti(args.base_path, SCENE_NAMES_TEST, transform, shuffle_hints=False, flip_poses=False)
    dataset_retrieval = Kitti360CoarseDatasetMulti(args.base_path, ['2013_05_28_drive_0003_sync', ], transform, shuffle_hints=False, flip_poses=False)
    dataloader_retrieval = DataLoader(dataset_retrieval, batch_size = args.batch_size, collate_fn=Kitti360CoarseDataset.collate_fn)

    # dataset_cell_only = dataset_retrieval.get_cell_dataset()

    # Load models
    model_retrieval = torch.load(args.path_coarse)
    model_matching = torch.load(args.path_fine)

    eval_conf(model_matching, dataset_retrieval)
    quit()

    # Run coarse
    retrievals, coarse_accuracies = run_coarse(model_retrieval, dataloader_retrieval, args)
    print_accuracies(coarse_accuracies, "Coarse")

    # Run fine
    accuracies_mean, accuracies_offsets, accuracies_mean_conf = run_fine(model_matching, retrievals, dataloader_retrieval, args)
    print_accuracies(accuracies_mean, "Fine (mean)")
    print_accuracies(accuracies_offsets, "Fine (offsets)")
    print_accuracies(accuracies_mean_conf, "Fine (mean-conf)")

    quit()
    # OLD

    # Run retrieval
    retrievals = run_retrieval(model_retrieval, dataloader_retrieval)
    pos_in_cell = [np.array((0.5, 0.5)) for i in range(len(dataset_retrieval))] # Estimate middle of the cell for each retrieval
    accuracies = eval_pose_accuracies(dataset_retrieval, retrievals, pos_in_cell, top_k=args.top_k, threshs=threshs)
    print_accuracies(accuracies)

    # Run matching
    offsets, matches0 = run_matching(model_matching, dataloader_matching)

    # Without offsets
    pos_in_cell = [get_pos_in_cell(dataset_matching[i]['objects'], matches0[i], np.zeros_like(offsets[i])) for i in range(len(dataset_matching))] # Zero-offsets to just take mean of objects
    accuracies = eval_pose_accuracies(dataset_retrieval, retrievals, pos_in_cell, top_k=args.top_k, threshs=threshs)
    print_accuracies(accuracies)

    # With offsets
    pos_in_cell = [get_pos_in_cell(dataset_matching[i]['objects'], matches0[i], offsets[i]) for i in range(len(dataset_matching))] # Using actual offset-vectors
    accuracies = eval_pose_accuracies(dataset_retrieval, retrievals, pos_in_cell, top_k=args.top_k, threshs=threshs)    
    print_accuracies(accuracies)
    
