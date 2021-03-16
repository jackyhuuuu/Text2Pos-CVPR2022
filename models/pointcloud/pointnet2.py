import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric.nn as gnn
import torch_geometric.transforms as T

import numpy as np

from easydict import EasyDict

# from models.modules import get_mlp

def get_mlp(channels, add_batchnorm=True):
    return nn.Sequential(*[
        # nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i]))
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.BatchNorm1d(channels[i]), nn.ReLU())
        for i in range(1, len(channels))
    ])

class SetAbstractionLayer(nn.Module):
    def __init__(self, ratio, radius, mlp):
        super(SetAbstractionLayer, self).__init__()
        self.ratio = ratio
        self.radius = radius
        self.point_conv = gnn.PointConv(local_nn=mlp)

    def forward(self, x, pos, batch):
        subset_indices = gnn.fps(pos, batch, self.ratio)

        sparse_indices, dense_indices = gnn.radius(pos, pos[subset_indices], self.radius, batch_x=batch, batch_y=batch[subset_indices])
        edge_index = torch.stack((dense_indices, sparse_indices), dim=0) #TODO/CARE: Indices are propagated internally? Care edge direction: a->b <=> a is in N(b)

        x = self.point_conv(x, (pos, pos[subset_indices]), edge_index)

        return x, pos[subset_indices], batch[subset_indices]

class GlobalAbstractionLayer(nn.Module):
    def __init__(self, mlp):
        super(GlobalAbstractionLayer, self).__init__()
        self.mlp = mlp

    def forward(self, x, pos, batch):
        x = torch.cat((x, pos), dim=1)
        x = self.mlp(x)
        x = gnn.global_max_pool(x, batch)
        return x

class PointNet2(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2, self).__init__()
        # self.sa1 = SetAbstractionLayer(0.5, 0.2, get_mlp([3 + 3, 16, 32], add_batchnorm=True))
        # self.sa2 = SetAbstractionLayer(0.25, 0.4, get_mlp([32 + 3, 64, 128], add_batchnorm=True))
        # self.ga = GlobalAbstractionLayer(get_mlp([128 + 3, 256, 512], add_batchnorm=True))

        # self.lin1 = nn.Linear(512, 256)
        # self.lin2 = nn.Linear(256, num_classes)

        self.sa1 = SetAbstractionLayer(0.5, 0.2, get_mlp([3 + 3, 32, 64], add_batchnorm=True))
        self.sa2 = SetAbstractionLayer(0.25, 0.4, get_mlp([64 + 3, 128, 256], add_batchnorm=True))
        self.ga = GlobalAbstractionLayer(get_mlp([256 + 3, 512, 1024], add_batchnorm=True))

        self.lin1 = nn.Linear(1024, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, num_classes)

    def forward(self, data):
        data.to(self.device)

        x, pos, batch = self.sa1(data.x, data.pos, data.batch)
        x, pos, batch = self.sa2(x, pos, batch)
        x = self.ga(x, pos, batch)

        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(self.lin2(x))
        # x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin3(x)

        return x

    @property
    def device(self):
        return next(self.lin1.parameters()).device

if __name__ == "__main__":
    transform = T.Compose([T.NormalizeScale(), T.FixedPoints(5)])
    pos = torch.rand(10, 3)
    print(pos)
    print(transform(EasyDict(pos=pos, num_nodes=10)).pos)

    quit()


    x = torch.rand(10, 3)
    pos = torch.rand(10, 3)
    batch = torch.zeros(10, dtype=torch.long)

    model = PointNet2(10)

    out = model(EasyDict(x=x, pos=pos, batch=batch))
