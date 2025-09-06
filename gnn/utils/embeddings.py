from sentence_transformers import SentenceTransformer
import torch

class EmbeddingEncoder:
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        self.model = SentenceTransformer(model_name)
        self.device = device if device else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)
        return embeddings
