# Example: How to use the API with real OpenAI integration

## 1. Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

## 2. Start the API server
python3 -m uvicorn api.main:app --reload --port 8000

## 3. Test with curl
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'

## 4. Expected response with real LLM:
{
  "question": "What is the capital of France?",
  "answer": "Based on the context provided, Paris is the capital of France.",
  "retrieved": ["Q: What is Paris's country? A: France", ...],
  "retrieval_scores": [0.716, 0.716, 0.716],
  "retrieval_types": ["qa_pair", "qa_pair", "qa_pair"],
  "prediction": 1,
  "trust_score": 0.850,
  "raw_probs": [0.150, 0.850],
  "evidence": ["Q: What is Paris's country? A: France", ...],
  "retriever_stats": {"total_documents": 6415, "index_size": 6415, "dimension": 384},
  "llm_metadata": {
    "model": "gpt-3.5-turbo",
    "method": "openai_api",
    "context_used": 3,
    "usage": {"total_tokens": 150, "prompt_tokens": 120, "completion_tokens": 30}
  }
}

## 5. Interactive API docs
Visit: http://localhost:8000/docs
