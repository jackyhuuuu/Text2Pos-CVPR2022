import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

from models.cell_retrieval import CellRetrievalNetwork

from dataloading.semantic3d.semantic3d_poses import Semantic3dPosesDataset, Semantic3dPosesDatasetMulti
from datapreparation.semantic3d.imports import COMBINED_SCENE_NAMES as SCENE_NAMES_S3D
from datapreparation.semantic3d.drawing import draw_retrieval

from datapreparation.kitti360.utils import SCENE_NAMES as SCENE_NAMES_K360
from dataloading.kitti360.cells import Kitti360CellDataset, Kitti360CellDatasetMulti

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, PairwiseRankingLoss, HardestRankingLoss

'''
TODO:
- try all-to-all edges?
- does it generalize? some form of augmentation? K-Fold?
- max-dist for descriptions?
- remove "identical negative" (currently does not occur in Kitti)
'''

def train_epoch(model, dataloader, args):
    model.train()
    epoch_losses = []   

    for i_batch, batch in enumerate(dataloader):
        if args.max_batches is not None and i_batch >= args.max_batches:
            break  

        batch_size = len(batch['texts'])
 
        optimizer.zero_grad()
        anchor = model.encode_text(batch['texts']) 
        cell_objects = [cell.objects for cell in batch['cells']]
        positive = model.encode_objects(cell_objects)

        if args.ranking_loss == 'triplet':
            negative_cell_objects = [cell.objects for cell in batch['negative_cells']]
            negative = model.encode_objects(negative_cell_objects)
            loss = criterion(anchor, positive, negative)
        else:
            loss = criterion(anchor, positive)

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    return np.mean(epoch_losses)

printed = False

# TODO: possibly update this to also handle S3D
@torch.no_grad()
def eval_epoch(model, dataloader, args):
    """Top-k retrieval for each pose against all cells in the dataset.

    Args:
        model ([type]): The model
        dataloader ([type]): Train or test dataloader
        args ([type]): Global

    Returns:
        Dict: Top-k accuracies
        Dict: Top retrievals
    """
    global printed

    cell_encodings = np.zeros((len(dataloader.dataset), model.embed_dim))
    text_encodings = np.zeros((len(dataloader.dataset), model.embed_dim))
    index_offset = 0
    for batch in dataloader:
        cell_objects = [cell.objects for cell in batch['cells']]
        cell_enc = model.encode_objects(cell_objects)
        text_enc = model.encode_text(batch['texts'])

        batch_size = len(cell_enc)
        cell_encodings[index_offset : index_offset + batch_size, :] = cell_enc.cpu().detach().numpy()
        text_encodings[index_offset : index_offset + batch_size, :] = text_enc.cpu().detach().numpy()
        index_offset += batch_size

    accuracies = {k: [] for k in args.top_k}
    top_retrievals = {} # Top retrievals as {query_idx: sorted_indices}
    for query_idx in range(len(text_encodings)):
        if args.ranking_loss == 'triplet':
            dists = np.linalg.norm(cell_encodings[:] - text_encodings[query_idx], axis=1)
            sorted_indices = np.argsort(dists) # Low->high
        else:
            scores = cell_encodings[:] @ text_encodings[query_idx]
            sorted_indices = np.argsort(-1.0 * scores) # Sort high->low
        
        for k in args.top_k:
            accuracies[k].append(query_idx in sorted_indices[0:k])
        top_retrievals[query_idx] = sorted_indices

    for k in args.top_k:
        accuracies[k] = np.mean(accuracies[k])
    
    return accuracies, top_retrievals

if __name__ == "__main__":
    args = parse_arguments()
    print(args, "\n")

    # WEITER: Colors, compare features
    
    '''
    Create data loaders
    '''    
    if args.dataset == 'S3D':
        scene_names = args.scene_names #['sg27_station2_intensity_rgb', 'sg27_station4_intensity_rgb', 'sg27_station5_intensity_rgb'] #'sg27_station4_intensity_rgb','sg27_station5_intensity_rgb','sg27_station9_intensity_rgb','sg28_station4_intensity_rgb']
        dataset_train = Semantic3dPosesDatasetMulti('./data/numpy_merged/', './data/semantic3d', scene_names, args.cell_size, args.cell_stride, split='train')
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=Semantic3dPosesDataset.collate_fn, shuffle=args.shuffle)
        dataset_val = Semantic3dPosesDatasetMulti('./data/numpy_merged/', './data/semantic3d', scene_names, args.cell_size, args.cell_stride, split='test')
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=Semantic3dPosesDataset.collate_fn, shuffle=False)
        print('Scene names:', scene_names)

        dataset_val = Semantic3dPosesDataset('./data/numpy_merged/', './data/semantic3d', "sg27_station2_intensity_rgb", args.cell_size, args.cell_stride, split='test')
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=Semantic3dPosesDataset.collate_fn, shuffle=False)
        
        print("\t\t Stats: ", args.cell_size, args.cell_stride, dataset_train.gather_stats())

    if args.dataset == 'K360':
        scene_names = ['2013_05_28_drive_0000_sync', ]
        dataset_train = Kitti360CellDatasetMulti('./data/kitti360', scene_names, split='train')
        # dataset_train = Kitti360CellDataset('./data/kitti360', scene_name, split='train')
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=Kitti360CellDataset.collate_fn, shuffle=args.shuffle)
        # dataset_val = Kitti360CellDataset('./data/kitti360', scene_name, split='test')
        dataset_val = Kitti360CellDatasetMulti('./data/kitti360', scene_names, split='test')
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=Kitti360CellDataset.collate_fn, shuffle=False)    

    train_words = dataset_train.get_known_words()
    for word in dataset_val.get_known_words():
        assert word in train_words

    assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes())

    data = dataset_train[0]        
    batch = next(iter(dataloader_train))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)
    torch.autograd.set_detect_anomaly(True)     

    learning_rates = np.logspace(-2, -4, 5)
    dict_loss = {lr: [] for lr in learning_rates}
    dict_acc = {k: {lr: [] for lr in learning_rates} for k in args.top_k}
    dict_acc_val = {k: {lr: [] for lr in learning_rates} for k in args.top_k}    

    # ACC_TARGET = 'all'
    for lr in learning_rates:
        model = CellRetrievalNetwork(dataset_train.get_known_classes(), dataset_train.get_known_words(), args.embed_dim, k=args.k, use_features=args.use_features)
        model.to(device)    

        optimizer = optim.Adam(model.parameters(), lr=lr)
        if args.ranking_loss == 'pairwise':
            criterion = PairwiseRankingLoss(margin=args.margin)
        if args.ranking_loss == 'hardest':
            criterion = HardestRankingLoss(margin=args.margin)
        if args.ranking_loss == 'triplet':
            criterion = nn.TripletMarginLoss(margin=args.margin)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,args.lr_gamma)

        for epoch in range(args.epochs):
            loss = train_epoch(model, dataloader_train, args)
            train_acc, train_retrievals = eval_epoch(model, dataloader_train, args)
            val_acc, val_retrievals = eval_epoch(model, dataloader_val, args)

            key = lr
            dict_loss[key].append(loss)
            for k in args.top_k:
                dict_acc[k][key].append(train_acc[k])
                dict_acc_val[k][key].append(val_acc[k])

            scheduler.step()
            print(f'\t lr {lr:0.4} loss {loss:0.2f} epoch {epoch} train-acc: ', end="")
            for k, v in train_acc.items():
                print(f'{k}-{v:0.2f} ', end="")
            print('val-acc: ', end="")
            for k, v in val_acc.items():
                print(f'{k}-{v:0.2f} ', end="")            
            print()        

    '''
    Save plots
    '''
    # scene_name = scene_name.split('_')[-2]
    # plot_name = f'Cells-{args.dataset}_bs{args.batch_size}_mb{args.max_batches}_e{args.embed_dim}_l-{args.ranking_loss}_m{args.margin}_c{int(args.cell_size)}-{int(args.cell_stride)}_f{"-".join(args.use_features)}.png'
    plot_name = f'Cells-{args.dataset}_bs{args.batch_size}_mb{args.max_batches}_e{args.embed_dim}_l-{args.ranking_loss}_m{args.margin}_f{"-".join(args.use_features)}.png'

    train_accs = {f'train-acc-{k}': dict_acc[k] for k in args.top_k}
    val_accs = {f'val-acc-{k}': dict_acc_val[k] for k in args.top_k}
    metrics = {
        'train-loss': dict_loss,
        **train_accs,
        **val_accs
    }
    plot_metrics(metrics, './plots/'+plot_name)    

    # show = 'val'
    # retrievals = val_retrievals if show=='val' else train_retrievals
    # dataset = dataset_val if show=='val' else dataset_train
    # for pose_idx in retrievals.keys():
    #     img = draw_retrieval(dataset, pose_idx, retrievals[pose_idx])          
    #     cv2.imwrite(f"retrievals_{show}_{pose_idx:02.0f}.png", img)

# DEPRECATED / Old function for S3D
# @torch.no_grad()
# def eval_epoch(model, dataloader, args, targets='all'):
#     """Top-k retrieval for each pose against all cells in the dataset.
#     Model might not have seen cells in pairwise-ranking-loss training.

#     Args:
#         model : The model
#         dataloader : The dataloader
#         args : Global arguments
#         targets: <all> or <poses> to use all available cells as targets or only those matching a pose
#     """
#     global print_targets

#     assert targets in ('all', 'poses')
#     dataset = dataloader.dataset
    
#     #Encode all the cells
#     cell_encodings = np.zeros((0, model.embed_dim))
#     for i in range(0, len(dataset.cells), args.batch_size):
#         cells = dataset.cells[i : i+args.batch_size]
#         cell_objects = [cell.objects for cell in cells]
#         encodings = model.encode_objects(cell_objects)
#         cell_encodings = np.vstack((cell_encodings, encodings.cpu().detach().numpy()))

#     #Encode all the poses
#     pose_encodings = np.zeros((0, model.embed_dim))
#     correct_indices = []
#     for i_batch, batch in enumerate(dataloader):
#         if args.max_batches is not None and i_batch >= args.max_batches:
#             break 

#         encodings = model.encode_text(batch['texts'])
#         pose_encodings = np.vstack((pose_encodings, encodings.cpu().detach().numpy()))
#         correct_indices.extend(batch['cell_indices'])
#     assert len(correct_indices) == len(pose_encodings)

#     if targets == 'poses': # Remove all the cells that are not the target of a pose
#         for idx in range(len(cell_encodings)):
#             if idx not in correct_indices:
#                 cell_encodings[idx, :] = np.inf

#     if print_targets:
#         print('# targets: ', len(np.unique(correct_indices)))
#         print_targets = False

#     accuracies = {k: [] for k in args.top_k}
#     top_retrievals = {} # Top retrievals as {query_pose_idx: sorted_indices}
#     for i in range(len(pose_encodings)):
#         if args.ranking_loss == 'triplet':
#             dists = np.linalg.norm(cell_encodings[:] - pose_encodings[i], axis=1)
#             sorted_indices = np.argsort(dists) #Low->high
#         else:
#             scores = cell_encodings[:] @ pose_encodings[i]
#             sorted_indices = np.argsort(-1.0 * scores) #Sort high->low
            
#         for k in args.top_k:
#             accuracies[k].append(correct_indices[i] in sorted_indices[0:k])

#         top_retrievals[i] = sorted_indices
    
#     for k in args.top_k:
#         accuracies[k] = np.mean(accuracies[k])
#     return accuracies, top_retrievals    