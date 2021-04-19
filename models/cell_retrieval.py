import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch.utils.data import Dataset, DataLoader

import time
import numpy as np
import os
import pickle
from easydict import EasyDict

from models.modules import get_mlp, LanguageEncoder
from dataloading.semantic3d.semantic3d import Semantic3dCellRetrievalDataset
from dataloading.semantic3d.semantic3d_poses import Semantic3dPosesDataset

'''
TODO:
- global mean or max?
- more layers?
- generally other architecture?
'''

class CellRetrievalNetwork(torch.nn.Module):
    def __init__(self, known_classes, known_words, args):
        super(CellRetrievalNetwork, self).__init__()
        self.embed_dim = args.embed_dim
        self.use_features = args.use_features
        self.variation = args.variation
        embed_dim = self.embed_dim

        '''
        Object path
        '''
        # Set idx=0 for padding
        self.known_classes = {c: (i+1) for i,c in enumerate(known_classes)}
        self.known_classes['<unk>'] = 0
        self.class_embedding = nn.Embedding(len(self.known_classes), embed_dim, padding_idx=0)

        self.pos_embedding = get_mlp([3, 64, embed_dim]) #OPTION: pos_embedding layers
        self.color_embedding = get_mlp([3, 64, embed_dim]) #OPTION: color_embedding layers

        self.mlp_merge = get_mlp([len(self.use_features)*embed_dim, embed_dim])

        # CARE: possibly handle variation in forward()!
        if self.variation == 0:
            self.graph1 = gnn.DynamicEdgeConv(get_mlp([2 * embed_dim, embed_dim, embed_dim], add_batchnorm=True), k=8, aggr='max') # Originally: k=4
            # self.graph2 = gnn.DynamicEdgeConv(get_mlp([2 * embed_dim, embed_dim, embed_dim], add_batchnorm=True), k=k, aggr='max')            
            # self.lin1 = nn.Sequential(get_mlp([2*embed_dim, embed_dim]), nn.ReLU())
            self.lin = get_mlp([embed_dim, embed_dim, embed_dim])
        if self.variation == 1:
            self.graph1 = gnn.DynamicEdgeConv(get_mlp([2 * embed_dim, embed_dim, embed_dim], add_batchnorm=True), k=8, aggr='max') # Originally: k=4
            self.lin = get_mlp([embed_dim, embed_dim, embed_dim])            
        if self.variation == 2:
            self.graph1 = gnn.DynamicEdgeConv(get_mlp([2 * embed_dim, embed_dim, embed_dim], add_batchnorm=True), k=8, aggr='max') # Originally: k=4
            self.graph2 = gnn.DynamicEdgeConv(get_mlp([2 * embed_dim, embed_dim, embed_dim], add_batchnorm=True), k=8, aggr='max')            
            self.lin_combine = get_mlp([2*embed_dim, embed_dim])
            self.lin = get_mlp([embed_dim, embed_dim, embed_dim])

        '''
        Textual path
        '''
        self.language_encoder = LanguageEncoder(known_words, embed_dim, bi_dir=True)                   

        print(f'CellRetrievalNetwork variation: {self.variation}, dim: {embed_dim}, features: {self.use_features}')

        self.printed = False

    def encode_text(self, descriptions):
        batch_size = len(descriptions)
        description_encodings = self.language_encoder(descriptions) # [B, DIM]

        description_encodings = F.normalize(description_encodings)

        return description_encodings

    def encode_objects(self, objects):
        '''
        Process the objects in a flattened way to allow for the processing of batches with uneven samples
        '''
        batch_size = len(objects)

        class_indices = []
        batch = [] #Batch tensor to send into PyG
        for i_batch, objects_sample in enumerate(objects):
            for obj in objects_sample:
                class_idx = self.known_classes.get(obj.label, 0)
                class_indices.append(class_idx)
                batch.append(i_batch)
        batch = torch.tensor(batch, dtype=torch.long, device=self.device)

        embeddings = []
        if 'class' in self.use_features:
            class_embedding = self.class_embedding(torch.tensor(class_indices, dtype=torch.long, device=self.device))
            embeddings.append(F.normalize(class_embedding, dim=-1))
        if 'color' in self.use_features:
            colors = []
            for objects_sample in objects:
                colors.extend([obj.get_color_rgb() for obj in objects_sample])
            color_embedding = self.color_embedding(torch.tensor(colors, dtype=torch.float, device=self.device))
            embeddings.append(F.normalize(color_embedding, dim=-1))
        if 'position' in self.use_features:
            positions = []
            for objects_sample in objects:
                # positions.extend([obj.center_in_cell for obj in objects_sample])
                positions.extend([obj.closest_point for obj in objects_sample])
            pos_embedding = self.pos_embedding(torch.tensor(positions, dtype=torch.float, device=self.device))
            embeddings.append(F.normalize(pos_embedding, dim=-1))

        if len(embeddings) > 1:
            embeddings = self.mlp_merge(torch.cat(embeddings, dim=-1))
        else:
            embeddings = embeddings[0]
        # embeddings = torch.sum(torch.stack(embeddings), dim=0)

        if self.variation == 0:
            x = self.graph1(embeddings, batch)
            x = gnn.global_max_pool(x, batch)
            x = self.lin(x)
        if self.variation == 1:
            x = self.graph1(embeddings, batch)
            x = gnn.global_mean_pool(x, batch)
            x = self.lin(x)   
        if self.variation == 2:
            x1 = self.graph1(embeddings, batch)
            x2 = self.graph1(x1, batch)
            x = torch.cat((x1, x2), dim=-1)
            x = self.lin_combine(x)
            x = gnn.global_max_pool(x, batch)
            x = self.lin(x)                     
        
        x = F.normalize(x)
        return x

    def forward(self):
        raise Exception('Not implemented.')

    @property
    def device(self):
        return next(self.pos_embedding.parameters()).device   

if __name__ == "__main__":
    model = CellRetrievalNetwork(['high vegetation', 'low vegetation', 'buildings', 'hard scape', 'cars'], 'a b c d e'.split(), embed_dim=32, k=2)      

    dataset = Semantic3dPosesDataset('./data/numpy_merged/', './data/semantic3d')
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=Semantic3dPosesDataset.collate_fn)
    data = dataset[0]        
    batch = next(iter(dataloader))

    # dataset = Semantic3dCellRetrievalDataset('./data/numpy_merged/', './data/semantic3d', ['class', 'color', 'position'])
    # dataloader = DataLoader(dataset, batch_size=2, collate_fn=Semantic3dCellRetrievalDataset.collate_fn)
    # data = dataset[0]
    # batch = next(iter(dataloader))

    # text = model.encode_text(batch['descriptions'])
    # objects = model.encode_objects(batch['objects'])

    