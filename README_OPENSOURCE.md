# THAU - Self-Learning AI Framework

<p align="center">
  <strong>An experimental framework for building self-learning language models with cognitive age progression</strong>
</p>

<p align="center">
  <a href="#features">Features</a> |
  <a href="#installation">Installation</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#architecture">Architecture</a> |
  <a href="#contributing">Contributing</a>
</p>

---

## What is THAU?

THAU (Thinking, Heuristic, Autonomous, Understanding) is an experimental framework for building AI models that can:

- **Learn continuously** from interactions (like a living being)
- **Self-question** to generate training data autonomously
- **Research topics** from the web and PDFs
- **Progress through cognitive ages** (0-15 years) with increasing complexity

> **Note**: This is an educational/experimental project. The trained models are not meant to compete with production LLMs like GPT or Claude, but rather to explore novel approaches to continuous learning.

## Features

### Cognitive Age System
Models progress through developmental stages, each with increasing complexity:

| Age | Parameters | Description |
|-----|------------|-------------|
| 0-3 | ~15M | Basic language understanding |
| 6 | ~50M | Simple reasoning |
| 9 | ~150M | Complex patterns |
| 12 | ~400M | Advanced reasoning |
| 15 | ~2B | Full cognitive capacity |

### Self-Learning Systems

1. **Self-Questioning** (`thau_trainer/self_questioning.py`)
   - Generates its own Q&A pairs for training
   - Covers multiple knowledge categories
   - Uses external models (Ollama, Gemini) for answers

2. **Live Learning** (`thau_trainer/live_learning.py`)
   - Learns from each user interaction
   - Short-term memory for context
   - Auto-researches unknown topics

3. **External Learning** (`thau_trainer/external_learning.py`)
   - Scrapes and learns from web pages
   - Extracts knowledge from PDFs
   - Research mode for deep topic investigation

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
