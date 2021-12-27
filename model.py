from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.nn.conv import MessagePassing


class ResGCN(torch.nn.Module, ABC):
    def __init__(self, num_feature, num_label):
        super(ResGCN, self).__init__()
        self.GCN1 = GCNConv(num_feature, 64, cached=True)
        self.GCN2 = GCNConv(64, num_label, cached=True)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.LP = torch.nn.Linear(num_feature, 64)
        self.ln = torch.nn.LayerNorm([213, 64], elementwise_affine=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        res_x = self.LP(x)
        x = self.GCN1(x, edge_index)
        x=res_x+x
        x=fun.silu(x)
        x=self.ln(x)
        x = self.GCN2(x, edge_index)
        return x