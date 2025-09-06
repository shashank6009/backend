#!/bin/bash

# OpenRouter API Key Setup Script
# OpenRouter provides access to multiple LLMs with better pricing and free tiers

echo "🌐 OpenRouter API Key Setup"
echo "=========================="
echo ""

# Check if API key is already set
if [ ! -z "$OPENROUTER_API_KEY" ]; then
    echo "✅ OPENROUTER_API_KEY is already set!"
    echo "Current key: ${OPENROUTER_API_KEY:0:10}..."
    echo ""
    echo "To test if it works, run:"
    echo "python3 -c \"from api.openrouter_generator import OpenRouterGenerator; print('API Key Status:', 'Working' if OpenRouterGenerator().test_connection() else 'Not Working')\""
    exit 0
fi

echo "❌ OPENROUTER_API_KEY is not set"
echo ""
echo "📋 Steps to get your OpenRouter API key:"
echo "1. Visit: https://openrouter.ai/keys"
echo "2. Sign up for a free account (no credit card required)"
echo "3. Click 'Create Key'"
echo "4. Copy the key (starts with 'sk-or-')"
echo ""
echo "🎁 OpenRouter Benefits:"
echo "• Free tier: 1M tokens/month"
echo "• Access to GPT-3.5, GPT-4, Claude, and more"
echo "• Better pricing than direct OpenAI"
echo "• No credit card required for free tier"
echo ""

read -p "Do you have your OpenRouter API key ready? (y/n): " -n 1 -r
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
            read -p "Enter your OpenRouter API key: " api_key
            export OPENROUTER_API_KEY="$api_key"
            echo "✅ API key set for current session!"
            echo "To test: python3 -c \"from api.openrouter_generator import OpenRouterGenerator; print('Working!' if OpenRouterGenerator().test_connection() else 'Not working')\""
            ;;
        2)
            echo ""
            read -p "Enter your OpenRouter API key: " api_key
            echo "export OPENROUTER_API_KEY=\"$api_key\"" >> ~/.zshrc
            echo "✅ API key added to ~/.zshrc"
            echo "Run 'source ~/.zshrc' or restart terminal to activate"
            ;;
        3)
            echo ""
            echo "Manual setup instructions:"
            echo "1. Run: export OPENROUTER_API_KEY='your-key-here'"
            echo "2. Or add to ~/.zshrc: echo 'export OPENROUTER_API_KEY=\"your-key-here\"' >> ~/.zshrc"
            ;;
        *)
            echo "Invalid choice"
            ;;
    esac
else
    echo ""
    echo "📝 Manual setup instructions:"
    echo ""
    echo "1. Get your API key from: https://openrouter.ai/keys"
    echo "2. Set it temporarily:"
    echo "   export OPENROUTER_API_KEY='your-key-here'"
    echo ""
    echo "3. Or set it permanently:"
    echo "   echo 'export OPENROUTER_API_KEY=\"your-key-here\"' >> ~/.zshrc"
    echo "   source ~/.zshrc"
    echo ""
    echo "4. Test it works:"
    echo "   python3 -c \"from api.openrouter_generator import OpenRouterGenerator; print('Working!' if OpenRouterGenerator().test_connection() else 'Not working')\""
fi

echo ""
echo "🚀 Once set up, start the API server:"
echo "python3 -m uvicorn api.main:app --reload --port 8000"
echo ""
echo "Then visit: http://localhost:8000/docs"
echo ""
echo "💡 OpenRouter will be used automatically if API key is set!"
