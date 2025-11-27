# THAU - Self-Learning AI Framework

<p align="center">
  <strong>An experimental framework for building self-learning language models with cognitive age progression</strong>
</p>

<p align="center">
  <em>Dedicated to Thomas and Aurora - watching you learn and grow inspired this project</em>
</p>

<p align="center">
  <a href="#features">Features</a> |
  <a href="#installation">Installation</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#current-status">Status</a> |
  <a href="#contributing">Contributing</a> |
  <a href="#support-this-project">Support</a>
</p>

---

## What is THAU?

THAU (Thinking, Heuristic, Autonomous, Understanding) is an **experimental** framework exploring how AI models can learn progressively, similar to human cognitive development.

> **Important**: This is a learning/research project. The models are small (~15M to ~400M parameters) and cannot compete with production LLMs like GPT or Claude. The value is in the *concepts* and *approach*, not the model quality.

### What it actually does:

- **Self-questions** to generate its own training data (2,800+ Q&A pairs generated so far)
- **Trains progressively** through "cognitive ages" with increasing model complexity
- **Integrates with Ollama** for answer generation during self-questioning
- **Exports to GGUF** format for use with Ollama

### What it does NOT do (yet):

- Does not produce production-quality responses
- Does not truly "understand" - it's pattern matching like any LLM
- Live learning and external research features are experimental/incomplete

## Current Status

### Trained Models (Real Data)

| Age | Parameters | Final Loss | Status |
|-----|------------|------------|--------|
| 0 | ~15M | 7.88 | Trained |
| 1 | ~15M | 5.99 | Trained |
| 3 | ~15M | 4.50 | Trained |
| 6 | ~50M | 3.50 | Trained |
| 11 | ~230M | 5.59 | Trained |
| 12 | ~367M | 4.76 | Trained |

### Training Data

- **2,872 Q&A pairs** generated via self-questioning
- Categories: Python, JavaScript, DevOps, Databases, Architecture, etc.
- Answer sources: Ollama (llama3.1, deepseek-coder, mistral)

## Features

### Self-Questioning System (Working)

The core innovation - the model generates questions and uses external LLMs to get answers:

```bash
# This actually works and generates training data
python scripts/intensive_learning.py --questions 50 --model ollama
```

### Live Learning (Experimental)

Exists but not fully tested:
- `thau_trainer/live_learning.py` - learns from conversations
- Needs more development

### External Learning (Experimental)

Code exists but not production-ready:
- `thau_trainer/external_learning.py` - web/PDF scraping
- May have bugs

### Multi-Model Integration

- **Ollama**: Local models (Llama, Mistral, DeepSeek, etc.)
- **Gemini CLI**: Google's Gemini for search/research
- **Custom models**: Your own trained THAU models

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- Ollama (optional, for local models)
- Gemini CLI (optional, for research)

### Setup

```bash
# Clone the repository
git clone https://github.com/luepow/thau.git
cd thau

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env

# (Optional) Install Ollama models
ollama pull llama3.1:8b
ollama pull mistral:latest
```

## Quick Start

### 1. Self-Questioning Session

Generate Q&A pairs for training:

```bash
# Generate 50 questions across all categories
python scripts/intensive_learning.py --questions 50 --age 12

# Focus on a specific category
python scripts/intensive_learning.py --category python_advanced --questions 100

# Use a specific model for answers
python scripts/intensive_learning.py --questions 50 --model ollama
```

### 2. Train a Model

```bash
# Train Age 3 model (starter)
python scripts/train_phase1.py --epochs 3 --batch-size 4

# Progress to Age 6
python train_age_6.py --steps 500 --batch 4

# Continue to higher ages...
python train_age_12.py --steps 600 --batch 4
```

### 3. Interactive Chat

```bash
# Start the API server
python api/main.py

# Or use the CLI
python thau_trainer/live_learning.py --chat
```

### 4. Learn from External Sources

```bash
# Learn from a URL
python thau_trainer/external_learning.py --url "https://docs.python.org/3/tutorial/" --topic python

# Learn from a PDF
python thau_trainer/external_learning.py --pdf /path/to/book.pdf --topic programming

# Research a topic with Gemini
python thau_trainer/external_learning.py --research "machine learning basics" --depth deep
```

## Architecture

```
thau/
├── api/                    # REST API (FastAPI)
│   ├── main.py
│   └── routes/
├── core/                   # Core ML components
│   ├── models/            # Neural network architectures
│   ├── training/          # Training infrastructure
│   ├── inference/         # Text generation
│   └── tokenizer/         # Tokenization
├── memory/                 # Memory systems
│   ├── short_term.py      # Conversation context
│   ├── long_term.py       # Vector store (ChromaDB)
│   └── episodic.py        # Temporal experiences
├── thau_trainer/          # Self-learning systems
│   ├── self_questioning.py
│   ├── live_learning.py
│   ├── external_learning.py
│   └── own_model_manager.py
├── adapters/              # Cross-platform support
│   └── device_manager.py  # MPS/CUDA/CPU detection
├── scripts/               # Training & utility scripts
└── data/                  # Data storage
    ├── self_questioning/  # Generated Q&A pairs
    ├── knowledge/         # Knowledge base
    └── checkpoints/       # Model checkpoints
```

## Configuration

Key settings in `.env`:

```bash
# Model
MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
DEVICE=auto  # auto, cuda, mps, cpu

# Training
BATCH_SIZE=4
LEARNING_RATE=2e-5
MAX_LENGTH=1024

# Generation
TEMPERATURE=0.7
TOP_P=0.9
```

## API Reference

### REST Endpoints

```bash
# Chat
POST /chat/message
{
  "message": "What is Python?",
  "context": []
}

# Training
POST /train/interaction
{
  "user_input": "What is a lambda?",
  "assistant_response": "A lambda is..."
}

# Memory
POST /memory/store
{
  "content": "Important fact",
  "importance": 8
}
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black .
isort .
```

## Roadmap

- [ ] Multi-language support
- [ ] Distributed training
- [ ] Model quantization (INT4/INT8)
- [ ] Web UI for training monitoring
- [ ] Plugin system for custom learning sources

## Support This Project

If you find THAU useful or interesting, consider supporting its development:

<p align="center">
  <a href="https://buymeacoffee.com/luepow"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>
  <a href="https://github.com/sponsors/luepow"><img src="https://img.shields.io/badge/GitHub%20Sponsors-EA4AAA?style=for-the-badge&logo=github-sponsors&logoColor=white" alt="GitHub Sponsors"></a>
  <a href="https://paypal.me/luepow"><img src="https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white" alt="PayPal"></a>
</p>

Your support helps cover compute costs and keeps this project alive!

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on [TinyLlama](https://github.com/jzhang38/TinyLlama)
- Uses [Transformers](https://huggingface.co/transformers) by Hugging Face
- Inspired by cognitive development research

## Contact

- **Author**: Luis Perez
- **Email**: luepow@hotmail.com
- **GitHub**: [@luepow](https://github.com/luepow)

---

<p align="center">
  Made with curiosity and code
</p>
