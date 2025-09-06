import openai
import os
from typing import List, Dict, Optional
import json

class LLMGenerator:
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 max_tokens: int = 500,
                 temperature: float = 0.7):
        """
        Initialize LLM generator for answer generation
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: OpenAI model to use
            max_tokens: Maximum tokens for response
            temperature: Response creativity (0.0 = deterministic, 1.0 = creative)
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Set up API key
        if api_key:
            openai.api_key = api_key
        else:
            # Try to get from environment variable
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                openai.api_key = api_key
            else:
                print("⚠️  No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
                print("   For now, using mock responses.")
                self.use_mock = True
                return
        
        self.use_mock = False
        print(f"✅ LLM Generator initialized with model: {model}")
    
    def generate_answer(self, question: str, retrieved_docs: List[str]) -> Dict[str, any]:
        """
        Generate answer using LLM with retrieved context
        
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
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. If the context doesn't contain enough information to answer the question, say so."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "answer": answer,
                "model": self.model,
                "usage": response.usage,
                "context_used": len(retrieved_docs),
                "method": "openai_api"
            }
            
        except Exception as e:
            print(f"❌ Error calling OpenAI API: {e}")
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
            "method": "mock_response"
        }
    
    def test_connection(self) -> bool:
        """Test if API connection works"""
        if self.use_mock:
            return False
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            print(f"❌ API connection test failed: {e}")
            return False
