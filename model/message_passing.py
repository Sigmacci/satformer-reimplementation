# Reimplemntation of NeuroSAT message passing in PyG
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class BipartiteMessagePassing(MessagePassing):
    def __init__(self, hidden_dim: int):
        super().__init__(aggr='add')
        
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), # nn.LeakyReLU(0.1) to be tested
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.gru = nn.GRUCell(hidden_dim, hidden_dim) # LSTMCell(hidden_dim, hidden_dim) to be tested

    def forward(self, x_src, x_dst, edge_index):
        agg_msgs = self.propagate(
            edge_index,
            x=(x_src, x_dst),
            size=(x_src.size(0), x_dst.size(0))
        )
        return self.gru(agg_msgs, x_dst)

    def message(self, x_j):
        return self.msg_mlp(x_j)