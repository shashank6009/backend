import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

def device():
    return torch.device("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available()
                        else "cpu")

class TinySAGE(nn.Module):
    def __init__(self, in_dim=16, hid=32, out_dim=2, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid)
        self.conv2 = SAGEConv(hid, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

def make_toy_graph():
    x = torch.randn(6, 16)
    edge_index = torch.tensor([
        [0,1,1,2,2,3,3,4,4,5,5,0],
        [1,0,2,1,3,2,4,3,5,4,0,5]
    ], dtype=torch.long)
    y = torch.tensor([0,0,0,1,1,1], dtype=torch.long)
    train_mask = torch.tensor([1,1,1,1,0,0], dtype=torch.bool)
    test_mask  = torch.tensor([0,0,0,0,1,1], dtype=torch.bool)
    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

def train_one_epoch(model, data, opt):
    model.train()
    opt.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    opt.step()
    return float(loss.item())

@torch.no_grad()
def eval_acc(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    correct = (pred[mask] == data.y[mask]).sum().item()
    total = int(mask.sum())
    return correct / max(total, 1)

def main():
    dev = device()
    print(f"[smoke] Torch {torch.__version__}, PyG ok, device: {dev}")
    data = make_toy_graph().to(dev)
    model = TinySAGE(in_dim=16, hid=32, out_dim=2).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(1, 51):
        loss = train_one_epoch(model, data, opt)
        if epoch % 10 == 0 or epoch == 1:
            acc = eval_acc(model, data, data.test_mask)
            print(f"epoch {epoch:03d} | loss {loss:.4f} | test_acc {acc:.2f}")

if __name__ == "__main__":
    main()
