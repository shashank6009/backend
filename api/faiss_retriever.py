import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class FAISSRetriever:
    def __init__(self, documents_path: str = None, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize FAISS retriever with document embeddings
        
        Args:
            documents_path: Path to JSON file with documents
            model_name: SentenceTransformer model name
        """
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.index = None
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension
        self.index_built = False
        
        # Check for cached index
        if documents_path and os.path.exists(documents_path):
            self.index_cache_path = documents_path.replace('.json', '.faiss.idx')
            if os.path.exists(self.index_cache_path):
                print(f"Loading cached FAISS index → {self.index_cache_path}")
                self.index = faiss.read_index(self.index_cache_path)
                self.index_built = True
                # Still need to load documents for metadata
                self.load_documents(documents_path)
            else:
                self.load_documents(documents_path)
                self.build_index()
    
    def load_documents(self, documents_path: str):
        """Load documents from JSON file"""
        print(f"Loading documents from {documents_path}...")
        
        with open(documents_path, 'r') as f:
            data = json.load(f)
        
        # Extract documents from our dataset format
        for entry in data:
            question = entry.get("question", "")
            retrieved = entry.get("retrieved", entry.get("context", []))
            answer = entry.get("answer", "")
            
            # Add question-answer pairs as documents
            self.documents.append({
                "text": f"Q: {question}\nA: {answer}",
                "type": "qa_pair",
                "question": question,
                "answer": answer
            })
            
            # Add retrieved context as separate documents
            if isinstance(retrieved, list):
                for doc in retrieved:
                    if doc.strip():  # Skip empty documents
                        self.documents.append({
                            "text": doc,
                            "type": "context",
                            "question": question
                        })
            elif retrieved.strip():
                self.documents.append({
                    "text": retrieved,
                    "type": "context", 
                    "question": question
                })
        
        print(f"Loaded {len(self.documents)} documents")
    
    def build_index(self):
        """Build FAISS index from document embeddings with batching"""
        if not self.documents:
            print("No documents to index")
            return
        
        if self.index_built:
            print("Index already built")
            return
            
        print("Building FAISS index (batched)...")
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        texts = [doc["text"] for doc in self.documents]
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            # Encode batch
            batch_embeddings = self.encoder.encode(batch_texts, convert_to_tensor=False)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(batch_embeddings)
            
            # Add to index
            self.index.add(batch_embeddings.astype('float32'))
        
        # Save index to cache
        faiss.write_index(self.index, self.index_cache_path)
        print(f"Saved FAISS index cache → {self.index_cache_path}")
        print(f"FAISS index built with {self.index.ntotal} vectors")
        self.index_built = True
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for most similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        if self.index is None:
            print("No index available")
            return []
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_tensor=False)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    "text": self.documents[idx]["text"],
                    "score": float(score),
                    "type": self.documents[idx]["type"],
                    "metadata": {k: v for k, v in self.documents[idx].items() 
                               if k not in ["text", "type"]}
                })
        
        return results
    
    def add_documents(self, documents: List[str]):
        """Add new documents to the index"""
        for doc_text in documents:
            self.documents.append({
                "text": doc_text,
                "type": "new_document"
            })
        
        # Rebuild index
        self.build_index()
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": self.dimension
        }
