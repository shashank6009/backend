import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
import sys
import os

# Add gnn module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gnn'))

from models.gat import GAT
from .faiss_retriever import FAISSRetriever
from .llm_generator import LLMGenerator
from .openrouter_generator import OpenRouterGenerator

# -------------------------
# Load pretrained models
# -------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Sentence encoder
encoder = SentenceTransformer("all-MiniLM-L6-v2").to(DEVICE)

# 2. Load GAT checkpoint (trained earlier)
CKPT_PATH = os.path.join(os.path.dirname(__file__), "..", "gnn", "experiments", "best_gat.pt")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model = GAT(ckpt["in_dim"], ckpt["hidden_dim"], ckpt["out_dim"])
model.load_state_dict(ckpt["state_dict"])
model = model.to(DEVICE)
model.eval()

# 3. Initialize FAISS retriever with production dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "gnn", "production_dataset.json")
retriever = FAISSRetriever(documents_path=DATASET_PATH)

# 4. Initialize LLM generator (try OpenRouter first, then OpenAI)
if os.getenv("OPENROUTER_API_KEY"):
    llm_generator = OpenRouterGenerator()
    print("✅ Using OpenRouter for LLM generation")
elif os.getenv("OPENAI_API_KEY"):
    llm_generator = LLMGenerator()
    print("✅ Using OpenAI for LLM generation")
else:
    llm_generator = OpenRouterGenerator()  # Will use mock mode
    print("⚠️  No API keys found, using mock responses")

# -------------------------
# Real retrieval + LLM (FAISS + GPT)
# -------------------------
def retrieve_docs(question: str, k: int = 3):
    """
    Retrieve relevant documents using FAISS vector search
    
    Args:
        question: User question
        k: Number of documents to retrieve
        
    Returns:
        List of retrieved document texts
    """
    results = retriever.search(question, k=k)
    return [result["text"] for result in results]

def generate_answer(question: str, docs: list):
    """
    Generate answer using LLM with retrieved context
    
    Args:
        question: User question
        docs: List of retrieved document texts
        
    Returns:
        Generated answer string
    """
    result = llm_generator.generate_answer(question, docs)
    return result["answer"]

# -------------------------
# Build graph + run GNN
# -------------------------
def build_graph(question: str, docs: list, answer: str):
    nodes = [question] + docs + [answer]
    x = encoder.encode(nodes, convert_to_tensor=True).to(DEVICE)

    edge_index = []
    for i in range(1, len(nodes)-1):
        edge_index.append([0, i])
    edge_index.append([0, len(nodes)-1])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(DEVICE)

    y = torch.tensor([0])  # dummy, not used in inference
    g = Data(x=x, edge_index=edge_index, y=y)
    g.node_texts = nodes
    return g

def run_gnn(data):
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        graph_logits = global_mean_pool(logits, torch.zeros(data.x.size(0), dtype=torch.long, device=DEVICE))
        probs = torch.softmax(graph_logits, dim=-1).cpu().numpy().tolist()[0]
        pred = int(graph_logits.argmax(dim=-1))
    return pred, probs

# -------------------------
# Pipeline
# -------------------------
def pipeline(question: str, k: int = 3):
    """
    Complete pipeline: FAISS retrieval + LLM + GNN trust scoring
    
    Args:
        question: User question
        k: Number of documents to retrieve
        
    Returns:
        Dictionary with question, answer, retrieval results, and trust score
    """
    # Retrieve relevant documents using FAISS
    retrieval_results = retriever.search(question, k=k)
    docs = [result["text"] for result in retrieval_results]
    
    # Generate answer using LLM
    llm_result = llm_generator.generate_answer(question, docs)
    answer = llm_result["answer"]
    
    # Build graph and run GNN
    g = build_graph(question, docs, answer)
    pred, probs = run_gnn(g)

    return {
        "question": question,
        "answer": answer,
        "retrieved": docs,
        "retrieval_scores": [result["score"] for result in retrieval_results],
        "retrieval_types": [result["type"] for result in retrieval_results],
        "prediction": pred,            # 0 = hallucinated, 1 = supported
        "trust_score": probs[1],       # probability of supported
        "raw_probs": probs,
        "evidence": g.node_texts[1:-1], # retrieved docs
        "retriever_stats": retriever.get_stats(),
        "llm_metadata": {
            "model": llm_result["model"],
            "method": llm_result["method"],
            "context_used": llm_result["context_used"],
            "usage": llm_result["usage"]
        }
    }
