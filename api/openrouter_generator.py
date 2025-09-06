import requests
import json
import os
from typing import List, Dict, Optional

class OpenRouterGenerator:
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "openai/gpt-3.5-turbo",
                 max_tokens: int = 500,
                 temperature: float = 0.7):
        """
        Initialize OpenRouter generator for answer generation
        
        Args:
            api_key: OpenRouter API key (if None, will try to get from environment)
            model: Model to use (e.g., "openai/gpt-3.5-turbo", "anthropic/claude-3-haiku")
            max_tokens: Maximum tokens for response
            temperature: Response creativity (0.0 = deterministic, 1.0 = creative)
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = "https://openrouter.ai/api/v1"
        
        # Set up API key
        if api_key:
            self.api_key = api_key
        else:
            # Try to get from environment variable
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            if not self.api_key:
                print("⚠️  No OpenRouter API key found. Set OPENROUTER_API_KEY environment variable.")
                print("   For now, using mock responses.")
                self.use_mock = True
                return
        
        self.use_mock = False
        print(f"✅ OpenRouter Generator initialized with model: {model}")
    
    def generate_answer(self, question: str, retrieved_docs: List[str]) -> Dict[str, any]:
        """
        Generate answer using OpenRouter with retrieved context
        
        Args:
            question: User question
            retrieved_docs: List of retrieved document texts
            
        Returns:
            Dictionary with answer, usage stats, and metadata
        """
        if self.use_mock:
            return self._generate_mock_answer(question, retrieved_docs)
        
        try:
            # Prepare context from retrieved documents
            context = self._prepare_context(retrieved_docs)
            
            # Create prompt
            prompt = self._create_prompt(question, context)
            
            # Call OpenRouter API
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8000",  # Optional: for tracking
                    "X-Title": "Trustworthy LLM API"  # Optional: for tracking
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. If the context doesn't contain enough information to answer the question, say so."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data["choices"][0]["message"]["content"].strip()
                
                # Extract usage info if available
                usage = data.get("usage", {})
                
                return {
                    "answer": answer,
                    "model": self.model,
                    "usage": usage,
                    "context_used": len(retrieved_docs),
                    "method": "openrouter_api",
                    "provider": "openrouter"
                }
            else:
                print(f"❌ OpenRouter API error: {response.status_code} - {response.text}")
                return self._generate_mock_answer(question, retrieved_docs)
                
        except Exception as e:
            print(f"❌ Error calling OpenRouter API: {e}")
            print("   Falling back to mock response")
            return self._generate_mock_answer(question, retrieved_docs)
    
    def _prepare_context(self, retrieved_docs: List[str]) -> str:
        """Prepare context string from retrieved documents"""
        if not retrieved_docs:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            # Truncate very long documents
            doc_text = doc[:500] + "..." if len(doc) > 500 else doc
            context_parts.append(f"Context {i}: {doc_text}")
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for LLM"""
        return f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question accurately, please say so.

Context:
{context}

Question: {question}

Answer:"""
    
    def _generate_mock_answer(self, question: str, retrieved_docs: List[str]) -> Dict[str, any]:
        """Generate mock answer when API is not available"""
        # Simple mock that incorporates some context
        if retrieved_docs:
            # Extract some keywords from retrieved docs
            context_words = []
            for doc in retrieved_docs[:2]:  # Use first 2 docs
                words = doc.split()[:10]  # First 10 words
                context_words.extend(words)
            
            context_hint = " ".join(context_words[:5])  # First 5 words
            answer = f"Based on the context about {context_hint}, here's an answer to: {question}"
        else:
            answer = f"Generated answer to: {question}"
        
        return {
            "answer": answer,
            "model": "mock",
            "usage": {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
            "context_used": len(retrieved_docs),
            "method": "mock_response",
            "provider": "mock"
        }
    
    def test_connection(self) -> bool:
        """Test if API connection works"""
        if self.use_mock:
            return False
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10
                },
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"❌ OpenRouter connection test failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        if self.use_mock:
            return ["mock"]
        
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return [model["id"] for model in data.get("data", [])]
            return []
        except Exception as e:
            print(f"❌ Error fetching models: {e}")
            return []
