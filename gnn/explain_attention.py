import argparse, torch
from json_graph_dataset import JsonGraphDataset
from models.sage import GraphSAGE
from models.gat import GAT
from torch_geometric.nn import global_mean_pool

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_type = ckpt["model"]
    if model_type == "gat":
        model = GAT(ckpt["in_dim"], ckpt["hidden_dim"], ckpt["out_dim"])
    else:
        model = GraphSAGE(ckpt["in_dim"], ckpt["hidden_dim"], ckpt["out_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, model_type

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index", type=int, default=0, help="which sample to explain")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    ds = JsonGraphDataset(args.dataset)
    g = ds[args.index]  # single graph
    texts = g.node_texts  # [Q, doc1..docN, answer]

    model, mtype = load_model(args.ckpt)
    
    # Move data to same device as model
    device = next(model.parameters()).device
    g = g.to(device)

    # prediction (graph-level)
    with torch.no_grad():
        node_logits = model(g.x, g.edge_index)
        graph_logits = global_mean_pool(node_logits, torch.zeros(g.x.size(0), dtype=torch.long, device=device))
        pred = int(graph_logits.argmax(dim=-1))
        print(f"[prediction] label={pred} (0=hallucinated, 1=supported)")

    if mtype != "gat":
        print("Model is not GAT; no attention to show. Train with --model gat to enable.")
        return

    # attention from first GAT layer
    ei, alpha = model.attention(g.x, g.edge_index)  # ei: [2, E], alpha: [E]
    src, dst = ei[0], ei[1]

    # show edges from the question node (index 0) to others
    mask = (src == 0)
    scores = alpha[mask]
    dst_nodes = dst[mask].tolist()
    pairs = list(zip(dst_nodes, scores.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)

    print("\n[top attention from QUESTION → node]")
    for i, (dst_i, score) in enumerate(pairs[:args.topk], 1):
        kind = "ANSWER" if dst_i == (len(texts)-1) else ("DOC" if dst_i > 0 else "QUESTION")
        print(f"{i}. {kind} node #{dst_i} | attn={score:.4f}")
        print(f"   {texts[dst_i][:180]}{'...' if len(texts[dst_i])>180 else ''}")

if __name__ == "__main__":
    main()
