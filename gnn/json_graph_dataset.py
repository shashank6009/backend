import json
import torch
from torch_geometric.data import Data, InMemoryDataset
from utils.embeddings import EmbeddingEncoder

class JsonGraphDataset(InMemoryDataset):
    def __init__(self, path, transform=None, pre_transform=None):
        self.path = path
        self.encoder = EmbeddingEncoder()  # NEW: load embedding model once
        super().__init__(None, transform, pre_transform)
        self.data, self.slices = self.load()

    def load(self):
        data_list = []

        # detect format
        if self.path.endswith(".jsonl"):
            with open(self.path, "r") as f:
                entries = [json.loads(line) for line in f]
        else:
            with open(self.path, "r") as f:
                entries = json.load(f)

        for entry in entries:
            question = entry.get("question", "")
            retrieved = entry.get("retrieved", entry.get("context", []))
            answer = entry.get("answer", "")

            # unify nodes
            nodes = [question] + (retrieved if isinstance(retrieved, list) else [retrieved]) + [answer]

            # 🔹 REAL EMBEDDINGS
            x = self.encoder.encode(nodes)  # shape: (num_nodes, embedding_dim)

            # trivial edges: Q → docs, Q → answer
            edge_index = []
            for i in range(1, len(nodes)-1):
                edge_index.append([0, i])
            edge_index.append([0, len(nodes)-1])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            # Use hallucination label if available, otherwise default to 1
            label = 0 if entry.get("is_hallucination", False) else 1
            y = torch.tensor([label], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)
            data.node_texts = nodes  # <— keep the original node texts for explanations
            data_list.append(data)

        return self.collate(data_list)
