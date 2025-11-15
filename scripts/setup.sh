#!/bin/bash

# Setup script for my-llm project

set -e  # Exit on error

echo "🚀 Setting up my-llm project..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Python 3.10+ is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "⚙️ Creating .env file from template..."
    cp .env.example .env
    echo "✏️ Please edit .env file with your configuration"
fi

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/{datasets,checkpoints,logs,models,memory}

# Download base model (optional)
read -p "🤔 Do you want to download the base model now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "⬇️ Downloading TinyLlama model..."
    python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
print(f'Downloading {model_name}...')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.save_pretrained('./data/models/tinyllama')
model.save_pretrained('./data/models/tinyllama')
print('Model downloaded successfully!')
"
fi

# Install package in development mode
echo "🔧 Installing my-llm in development mode..."
pip install -e .

# Run tests
echo "🧪 Running tests..."
pytest tests/ -v || echo "⚠️ Some tests failed. This is normal for a new setup."

echo ""
echo "✨ Setup complete!"
echo ""
echo "To get started:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Edit the .env file with your configuration"
echo "3. Start the API server: python api/main.py"
echo "4. Visit http://localhost:8000/docs for API documentation"
echo ""
echo "Happy coding! 🎉"
