#!/bin/bash

# OpenAI API Key Setup Script
# Run this script to help set up your OpenAI API key

echo "🔑 OpenAI API Key Setup"
echo "======================"
echo ""

# Check if API key is already set
if [ ! -z "$OPENAI_API_KEY" ]; then
    echo "✅ OPENAI_API_KEY is already set!"
    echo "Current key: ${OPENAI_API_KEY:0:10}..."
    echo ""
    echo "To test if it works, run:"
    echo "python3 -c \"from api.llm_generator import LLMGenerator; print('API Key Status:', 'Working' if LLMGenerator().test_connection() else 'Not Working')\""
    exit 0
fi

echo "❌ OPENAI_API_KEY is not set"
echo ""
echo "📋 Steps to get your API key:"
echo "1. Visit: https://platform.openai.com/api-keys"
echo "2. Sign in to your OpenAI account"
echo "3. Click 'Create new secret key'"
echo "4. Copy the key (starts with 'sk-')"
echo ""

read -p "Do you have your API key ready? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🔧 Choose setup method:"
    echo "1) Temporary (current session only)"
    echo "2) Permanent (add to ~/.zshrc)"
    echo "3) Manual setup"
    echo ""
    
    read -p "Enter choice (1-3): " choice
    
    case $choice in
        1)
            echo ""
            read -p "Enter your API key: " api_key
            export OPENAI_API_KEY="$api_key"
            echo "✅ API key set for current session!"
            echo "To test: python3 -c \"from api.llm_generator import LLMGenerator; print('Working!' if LLMGenerator().test_connection() else 'Not working')\""
            ;;
        2)
            echo ""
            read -p "Enter your API key: " api_key
            echo "export OPENAI_API_KEY=\"$api_key\"" >> ~/.zshrc
            echo "✅ API key added to ~/.zshrc"
            echo "Run 'source ~/.zshrc' or restart terminal to activate"
            ;;
        3)
            echo ""
            echo "Manual setup instructions:"
            echo "1. Run: export OPENAI_API_KEY='your-key-here'"
            echo "2. Or add to ~/.zshrc: echo 'export OPENAI_API_KEY=\"your-key-here\"' >> ~/.zshrc"
            ;;
        *)
            echo "Invalid choice"
            ;;
    esac
else
    echo ""
    echo "📝 Manual setup instructions:"
    echo ""
    echo "1. Get your API key from: https://platform.openai.com/api-keys"
    echo "2. Set it temporarily:"
    echo "   export OPENAI_API_KEY='your-key-here'"
    echo ""
    echo "3. Or set it permanently:"
    echo "   echo 'export OPENAI_API_KEY=\"your-key-here\"' >> ~/.zshrc"
    echo "   source ~/.zshrc"
    echo ""
    echo "4. Test it works:"
    echo "   python3 -c \"from api.llm_generator import LLMGenerator; print('Working!' if LLMGenerator().test_connection() else 'Not working')\""
fi

echo ""
echo "🚀 Once set up, start the API server:"
echo "python3 -m uvicorn api.main:app --reload --port 8000"
echo ""
echo "Then visit: http://localhost:8000/docs"
