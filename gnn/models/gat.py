import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4, dropout=0.2):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

    # expose first-layer attention for explanations
    @torch.no_grad()
    def attention(self, x, edge_index):
        out, (ei, alpha) = self.conv1(x, edge_index, return_attention_weights=True)
        # alpha shape: [num_edges, num_heads] -> average heads
        alpha = alpha.mean(dim=-1)
        return ei, alpha
