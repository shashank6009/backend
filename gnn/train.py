# backend/gnn/train.py
import argparse, random, numpy as np, torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from models.sage import GraphSAGE
from models.gat import GAT
from json_graph_dataset import JsonGraphDataset


def get_device():
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split_dataset(dataset, val_split=0.15, test_split=0.15, seed=42):
    n = len(dataset)
    n_test = int(n * test_split)
    n_val  = int(n * val_split)
    n_train = n - n_val - n_test
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_val, n_test], generator=gen)
    return train_ds, val_ds, test_ds

def forward_graph_logits(model, batch):
    # model outputs node-level logits; pool to graph-level logits
    node_logits = model(batch.x, batch.edge_index)
    graph_logits = global_mean_pool(node_logits, batch.batch)  # [B, out_dim]
    return graph_logits

def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        logits = forward_graph_logits(model, batch)
        loss = F.cross_entropy(logits, batch.y)  # y shape: [B]
        loss.backward()
        opt.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = forward_graph_logits(model, batch)
        preds = logits.argmax(dim=-1).detach().cpu().tolist()
        y_true.extend(batch.y.detach().cpu().tolist())
        y_pred.extend(preds)
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to .json or .jsonl")
    parser.add_argument("--model", type=str, default="sage", choices=["sage","gat"])
    parser.add_argument("--save_path", type=str, default="backend/gnn/experiments/best_model.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()

    # Load dataset (embeddings are created inside)
    dataset = JsonGraphDataset(args.dataset)
    in_dim = dataset[0].x.size(1)
    out_dim = 2  # binary: 0 = hallucinated, 1 = supported

    train_ds, val_ds, test_ds = split_dataset(dataset, args.val_split, args.test_split, args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size)

    if args.model == "sage":
        model = GraphSAGE(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=out_dim, dropout=args.dropout).to(device)
    else:
        model = GAT(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=out_dim, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    print(f"[info] device={device} | N={len(dataset)} (train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}) | in_dim={in_dim}")

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, opt, device)
        val_metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch:03d} | loss {loss:.4f} | val_acc {val_metrics['acc']:.3f} | val_f1 {val_metrics['f1']:.3f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Test with best model
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({
            "state_dict": best_state,
            "in_dim": in_dim,
            "hidden_dim": args.hidden_dim,
            "out_dim": out_dim,
            "model": args.model
        }, args.save_path)
        print(f"[saved] {args.save_path}")

    test_metrics = evaluate(model, test_loader, device)
    print(f"[test] acc {test_metrics['acc']:.3f} | precision {test_metrics['precision']:.3f} | recall {test_metrics['recall']:.3f} | f1 {test_metrics['f1']:.3f}")

if __name__ == "__main__":
    main()