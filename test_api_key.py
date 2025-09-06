#!/usr/bin/env python3

"""
Test script to verify LLM API key setup (OpenAI or OpenRouter)
Run this after setting your API key to make sure it works
"""

import os
import sys

# Add current directory to path
sys.path.append('.')

def test_api_keys():
    """Test if any LLM API key is working"""
    print("🔑 Testing LLM API Key Setup")
    print("=" * 40)
    
    # Check for OpenRouter first (preferred)
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if openrouter_key:
        print(f"✅ OpenRouter API key found: {openrouter_key[:10]}...")
        return test_openrouter()
    elif openai_key:
        print(f"✅ OpenAI API key found: {openai_key[:10]}...")
        return test_openai()
    else:
        print("❌ No LLM API keys found")
        print("\nTo set up:")
        print("• OpenRouter (recommended): ./setup_openrouter.sh")
        print("• OpenAI: ./setup_api_key.sh")
        return False

def test_openrouter():
    """Test OpenRouter API"""
    try:
        from api.openrouter_generator import OpenRouterGenerator
        
        print("\n🌐 Testing OpenRouter Generator...")
        generator = OpenRouterGenerator()
        
        if generator.use_mock:
            print("⚠️  Using mock mode (API key not working)")
            return False
        
        # Test connection
        print("🔗 Testing OpenRouter API connection...")
        if generator.test_connection():
            print("✅ OpenRouter API connection successful!")
            
            # Test actual generation
            print("\n🤖 Testing answer generation...")
            result = generator.generate_answer(
                "What is the capital of France?", 
                ["Paris is the capital of France"]
            )
            
            print(f"✅ Generated answer: {result['answer'][:100]}...")
            print(f"✅ Model: {result['model']}")
            print(f"✅ Provider: {result['provider']}")
            
            return True
        else:
            print("❌ OpenRouter API connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing OpenRouter API: {e}")
        return False

def test_openai():
    """Test OpenAI API"""
    try:
        from api.llm_generator import LLMGenerator
        
        print("\n🤖 Testing OpenAI Generator...")
        generator = LLMGenerator()
        
        if generator.use_mock:
            print("⚠️  Using mock mode (API key not working)")
            return False
        
        # Test connection
        print("🔗 Testing OpenAI API connection...")
        if generator.test_connection():
            print("✅ OpenAI API connection successful!")
            
            # Test actual generation
            print("\n🤖 Testing answer generation...")
            result = generator.generate_answer(
                "What is the capital of France?", 
                ["Paris is the capital of France"]
            )
            
            print(f"✅ Generated answer: {result['answer'][:100]}...")
            print(f"✅ Model: {result['model']}")
            
            return True
        else:
            print("❌ OpenAI API connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing OpenAI API: {e}")
        return False

def test_complete_pipeline():
    """Test the complete FAISS + LLM + GNN pipeline"""
    print("\n🚀 Testing Complete Pipeline")
    print("=" * 40)
    
    try:
        from api.inference import pipeline
        
        result = pipeline("What is the capital of France?", k=2)
        
        print(f"✅ Question: {result['question']}")
        print(f"✅ Answer: {result['answer'][:100]}...")
        print(f"✅ Trust Score: {result['trust_score']:.3f}")
        print(f"✅ LLM Model: {result['llm_metadata']['model']}")
        print(f"✅ Context Used: {result['llm_metadata']['context_used']} docs")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 LLM API Key Test Suite")
    print("=" * 50)
    
    # Test API key
    api_works = test_api_keys()
    
    if api_works:
        # Test complete pipeline
        pipeline_works = test_complete_pipeline()
        
        if pipeline_works:
            print("\n🎉 All tests passed! Your API key is working correctly.")
            print("\n🚀 Next steps:")
            print("1. Start the API server: python3 -m uvicorn api.main:app --reload --port 8000")
            print("2. Visit: http://localhost:8000/docs")
            print("3. Test the /ask endpoint with real LLM responses!")
        else:
            print("\n⚠️  API key works but pipeline has issues")
    else:
        print("\n❌ API key setup failed")
        print("\n📋 Troubleshooting:")
        print("1. Make sure you have a valid API key")
        print("2. Check your internet connection")
        print("3. Verify the key has sufficient credits")
        print("4. Run: ./setup_openrouter.sh for OpenRouter (recommended)")
        print("5. Run: ./setup_api_key.sh for OpenAI")
