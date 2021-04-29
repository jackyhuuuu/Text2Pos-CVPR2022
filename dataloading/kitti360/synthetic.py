from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2
from easydict import EasyDict
import time

import torch
from torch.utils.data import Dataset, DataLoader

from datapreparation.kitti360.utils import CLASS_TO_LABEL, LABEL_TO_CLASS, CLASS_TO_MINPOINTS, CLASS_TO_INDEX
from datapreparation.kitti360.utils import COLORS, COLOR_NAMES, SCENE_NAMES_TRAIN
from datapreparation.kitti360.descriptions import describe_cell
from datapreparation.kitti360.imports import Object3d, Cell
from datapreparation.kitti360.drawing import show_pptk, show_objects, plot_cell
from dataloading.kitti360.base import Kitti360BaseDataset
from dataloading.kitti360.objects import Kitti360ObjectsDatasetMulti


'''
TODO:
- Add transforms for copied objects?
- Sample class first or not?
- Explicitly move an object to on-top first or not?

- use combined describe_cell method from descriptions
- make create_hint_descriptions() in Base static and use here
- (cell normed right away would be ok)
- try to use for cells and poses, refactor later if necessary

'''

class Kitti360PoseReferenceMockDatasetPoints(Dataset):
    def __init__(self, base_path, scene_names, args, length=1024, fixed_seed=False):
        self.cell_size = 30 # CARE: Match to K360 prepare
        
        # Create an objects dataset to copy the objects from in synthetic cell creation
        objects_dataset = Kitti360ObjectsDatasetMulti(base_path, scene_names) # Transform of this dataset is ignored
        self.objects_dict = {c: [] for c in CLASS_TO_INDEX.keys()}
        for obj in objects_dataset.objects:
            self.objects_dict[obj.label].append(obj)
        # CARE: some classes might be empty because of the train/test split

        self.pad_size = args.pad_size
        self.num_mentioned = args.num_mentioned
        self.length = length
        self.fixed_seed = fixed_seed
        self.colors = COLORS
        self.color_names = COLOR_NAMES        

    def __getitem__(self, idx):
        if self.fixed_seed:
            np.random.seed(idx)

        pose = np.random.rand(3)
        num_distractors = np.random.randint(self.pad_size - self.num_mentioned) if self.pad_size > self.num_mentioned else 0

        # Copy over random objects from the real dataset to random positions
        # Note that the objects are already clustered and normed: taken from cells (not scene) in Kitti360ObjectsDataset
        # Closest points are re-set here
        objects = []
        for i in range(self.num_mentioned + num_distractors):
            obj_class = np.random.choice([k for k, v in self.objects_dict.items() if len(v) > 0])
            obj = np.random.choice(self.objects_dict[obj_class])
            assert np.max(np.abs(obj.xyz)) < 2.1

            # Shift the object center to a random position ∈ [0, 1] in x-y-plane, z is kept
            obj.xyz[:, 0:2] -= np.mean(obj.xyz[:, 0:2], axis=0)
            obj.xyz[:, 0:2] += np.random.rand(2)

            # Possibly shift the object so that it is not out-of-bounds
            for dim in (0, 1):
                if np.min(obj.xyz[:, dim]) < 0: obj.xyz[:, dim] -= np.min(obj.xyz[:, dim])
                if np.max(obj.xyz[:, dim]) > 1: obj.xyz[:, dim] -= np.max(obj.xyz[:, dim]) - 1          

            objects.append(obj)
            # TODO: possibly apply transforms

        # Random shift an object close to the pose for on-top
        if np.random.choice((True, False)):
            idx = np.random.randint(len(objects))
            obj.xyz[:, 0:2] -= np.mean(obj.xyz[:, 0:2], axis=0)
            obj.xyz[:, 0:2] += np.array(pose[0:2] + np.random.randn(2)*0.01).reshape((1,2))

        # Create the cell description object
        cell = describe_cell(np.array([0,0,0,1,1,1]), objects, pose, "Mock-Scene")
        assert cell is not None
        padded_objects = cell.objects # CARE: objects in cell now also padded (all by reference)

        # Add padding objects
        while len(padded_objects) < self.pad_size:
            xyz = np.zeros((1, 3))
            label = 'pad'
            rgb = np.zeros((1, 3))            
            padded_objects.append(Object3d(xyz, rgb, label, None))

        # Get the cell text
        hints = Kitti360BaseDataset.create_hint_description(cell)

        # Build matches
        matches = np.array(cell.matches)
        all_matches = [(i,j) for (i, j) in matches]
        for obj_idx, _ in enumerate(padded_objects):
            if obj_idx not in matches[:, 0]: # If the object at this index is not mentioned (distractor / pad) ...
                all_matches.append((obj_idx, self.num_mentioned)) # ... then match it to the hints-side bin.
        
        matches, all_matches = np.array(matches), np.array(all_matches)
        assert len(matches) == self.num_mentioned, matches
        assert len(all_matches) == len(padded_objects), f'#dist: {num_distractors}, all_matches: \n {all_matches}'
        assert np.sum(all_matches[:, 1] == self.num_mentioned) == self.pad_size - self.num_mentioned, f'#dist: {num_distractors}, all_matches: \n {all_matches}'

        return {
            'objects': padded_objects,
            'hint_descriptions': hints,
            'texts': ' '.join(hints),
            'num_mentioned': self.num_mentioned,
            'num_distractors': num_distractors,
            'matches': matches,
            'all_matches': all_matches,
            'poses': cell.pose,
            'offsets': cell.offsets,
            'cells': cell
        }


    def __len__(self):
        return self.length

    def collate_fn(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch      

    def get_known_classes(self):
        return list(self.objects_dict.keys())

    def get_known_words(self):
        known_words = []
        for i in range(50):
            data = self[i]
            for hint in data['hint_descriptions']:
                known_words.extend(hint.replace('.','').replace(',','').lower().split())
        return list(np.unique(known_words))  

if __name__ == '__main__':
    base_path = './data/kitti360'
    
    args = EasyDict(pad_size=8, num_mentioned=6)
    dataset = Kitti360PoseReferenceMockDatasetPoints(base_path, ['2013_05_28_drive_0000_sync',], args)
    data = dataset[0]
    print(data['hint_descriptions'])
    cv2.imshow("", plot_cell(data['cells'])); cv2.waitKey()

    words = dataset.get_known_words()
    print(words)
