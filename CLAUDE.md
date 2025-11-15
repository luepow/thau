# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**my-llm** is a modular framework for developing Large Language Models with incremental learning capabilities. The project is built with Python 3.10+ and uses PyTorch, Transformers, and FastAPI.

### Key Technologies
- **Python 3.10+**: Main language
- **PyTorch 2.0+**: Deep learning framework
- **Transformers**: HuggingFace library for LLMs
- **FastAPI**: REST API framework
- **ChromaDB**: Vector database for RAG
- **SQLite**: Episodic memory storage

## Commands

### Development

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=core --cov=memory --cov-report=html

# Format code
black .
isort .

# Type checking
mypy core/ memory/ api/
```

### Running the Application

```bash
# Start API server
python api/main.py

# Or using uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Run training scripts
python scripts/train_phase1.py --epochs 3 --batch-size 4
python scripts/train_phase2.py --epochs 3

# Run benchmarks
python scripts/benchmark.py
```

### Testing Individual Modules

```bash
# Test models
python core/models/base_transformer.py

# Test tokenizer
python core/tokenizer/tokenizer.py

# Test memory manager
python memory/manager.py

# Test device manager
python adapters/device_manager.py
```

## Architecture

### Core Components

1. **config/**: Configuration management with environment variable support
   - `base_config.py`: Global configuration with defaults from `.env`
   - `model_configs.py`: Model architecture configurations (Tiny, Small, Medium)
   - `training_configs.py`: Training hyperparameters and data loaders

2. **core/models/**: Neural network architectures
   - `base_transformer.py`: Main TinyLLM model with generation capabilities
   - `attention.py`: Multi-head attention with RoPE support
   - `layers.py`: Transformer blocks, FFN, positional encoding

3. **core/tokenizer/**: Text processing
   - `tokenizer.py`: Wrapper around HuggingFace tokenizers
   - `vocab.py`: Custom vocabulary management utilities

4. **core/training/**: Training infrastructure
   - `trainer.py`: Base trainer with evaluation and checkpointing
   - `incremental_trainer.py`: LoRA-based incremental learning from interactions
   - `optimizer.py`: Optimizer and scheduler utilities

5. **core/inference/**: Text generation
   - `generator.py`: Text generation with various decoding strategies
   - `sampler.py`: Sampling algorithms (top-k, nucleus, beam search)

6. **memory/**: Multi-level memory system
   - `manager.py`: Unified memory manager coordinating all systems
   - `short_term.py`: Conversation context buffer (FIFO)
   - `long_term.py`: Vector store for RAG (ChromaDB)
   - `episodic.py`: Temporal experiences (SQLite)
   - `vector_store.py`: ChromaDB wrapper for semantic search

7. **reasoning/**: Reasoning capabilities
   - `chain_of_thought.py`: CoT prompting and step-by-step reasoning
   - `planning.py`: Task decomposition and planning
   - `reflection.py`: Self-reflection on outputs

8. **adapters/**: Cross-platform abstractions
   - `device_manager.py`: Auto-detection and optimization for MPS/CUDA/CPU
   - `model_adapter.py`: Model loading, quantization, and LoRA setup

9. **api/**: REST API
   - `main.py`: FastAPI application with CORS and lifecycle management
   - `routes/chat.py`: Chat endpoints (message, stream, history)
   - `routes/training.py`: Training endpoints (interaction, stats, checkpoint)
   - `routes/memory.py`: Memory endpoints (store, recall, stats)
   - `schemas/models.py`: Pydantic models for request/response validation

### Data Flow

#### Chat Request Flow:
1. Request arrives at `/chat/message`
2. `ChatRouter` retrieves conversation history from `MemoryManager`
3. `TextGenerator` formats prompt with context
4. Model generates response
5. `MemoryManager` updates short-term and episodic memory
6. Response returned to client

#### Incremental Learning Flow:
1. Request arrives at `/train/interaction`
2. `IncrementalTrainer` formats training text
3. Tokenizes and creates input tensors
4. Runs gradient descent for specified steps
5. Updates LoRA adapters
6. Optionally saves checkpoint

#### Memory Recall Flow:
1. Request arrives at `/memory/recall`
2. `MemoryManager.recall()` searches:
   - Short-term buffer for recent context
   - Long-term vector store for semantic matches
   - Episodic database for relevant experiences
3. Combines and returns results

## Important Patterns

### Device Management
All tensor operations use `DeviceManager.to_device()` for cross-platform compatibility:

```python
from adapters.device_manager import get_device_manager

device_manager = get_device_manager()
tensor = device_manager.to_device(tensor)
```

### Configuration Access
Always use the global config instance:

```python
from config.base_config import get_config

config = get_config()
model_name = config.MODEL_NAME
```

### Model Loading with Quantization
Use `ModelAdapter` for consistent model loading:

```python
from adapters.model_adapter import ModelAdapter
from config.model_configs import QUANT_8BIT

adapter = ModelAdapter(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    use_quantization=True,
    quantization_config=QUANT_8BIT,
)
adapter.load_model()
adapter.load_tokenizer()
```

### Memory Operations
Use `MemoryManager` for all memory operations:

```python
from memory.manager import MemoryManager

memory = MemoryManager()

# Store
memory_id = memory.remember("Important fact", importance=8)

# Recall
results = memory.recall("search query", n_results=5)

# Update context
memory.update_context("user", "message content")
```

## File Organization Conventions

- All `__init__.py` files are present to make packages importable
- Test files mirror the source structure: `tests/test_models.py` tests `core/models/`
- Configuration uses environment variables with defaults
- All modules have standalone `if __name__ == "__main__"` sections for testing

## Development Workflow

### Adding a New Feature

1. **Define Configuration**: Add parameters to `config/base_config.py` or relevant config file
2. **Implement Core Logic**: Add implementation in appropriate module
3. **Add API Endpoint**: If needed, create route in `api/routes/`
4. **Write Tests**: Add tests in `tests/test_*.py`
5. **Update Documentation**: Update README.md with usage examples

### Modifying the Model

1. **Architecture Changes**: Edit `core/models/base_transformer.py` or related files
2. **Configuration**: Update `config/model_configs.py` with new parameters
3. **Testing**: Run `python core/models/base_transformer.py` to verify
4. **Integration**: Update `ModelAdapter` if loading logic changes

### Adding Memory Features

1. **Core Implementation**: Add to relevant memory module
2. **Integration**: Update `memory/manager.py` to coordinate
3. **API**: Add endpoints in `api/routes/memory.py`
4. **Persistence**: Ensure data is saved/loaded correctly

## Common Tasks

### Running a Quick Test
```bash
# Test model forward pass
python -c "from core.models.base_transformer import TinyLLM; from config.model_configs import TINY_CONFIG; m = TinyLLM(TINY_CONFIG); print('Model OK')"

# Test device detection
python -c "from adapters.device_manager import DeviceManager; dm = DeviceManager(); print(dm.get_device_info())"

# Test tokenizer
python -c "from core.tokenizer.tokenizer import Tokenizer; t = Tokenizer(); print(t)"
```

### Debugging Memory Issues
```bash
# Check ChromaDB
python -c "from memory.vector_store import VectorStore; vs = VectorStore(); print(f'Documents: {vs.count()}')"

# Check episodic database
python -c "from memory.episodic import EpisodicMemory; em = EpisodicMemory(); print(f'Episodes: {em.get_count()}')"

# View memory stats
curl http://localhost:8000/memory/stats
```

### Checking Model Performance
```bash
# Run benchmarks
python scripts/benchmark.py

# Check device info via API
curl http://localhost:8000/health
```

## Environment Variables

Key variables in `.env`:

- `MODEL_NAME`: HuggingFace model identifier
- `DEVICE`: Device selection (auto/cuda/mps/cpu)
- `BATCH_SIZE`: Training/inference batch size
- `MAX_LENGTH`: Maximum sequence length
- `TEMPERATURE`: Generation temperature
- `USE_QUANTIZATION`: Enable 8-bit quantization
- `MEMORY_DB_PATH`: ChromaDB storage path
- `API_PORT`: API server port

## Troubleshooting

### Import Errors
- Ensure virtual environment is activated: `source venv/bin/activate`
- Install in development mode: `pip install -e .`
- Check Python version: `python --version` (must be 3.10+)

### CUDA/MPS Issues
- Check device detection: Run `adapters/device_manager.py`
- Set explicit device: `DEVICE=cpu` in `.env`
- Clear PyTorch cache: `torch.cuda.empty_cache()` or `torch.mps.empty_cache()`

### API Not Starting
- Check port availability: `lsof -i :8000`
- Review logs in `data/logs/my-llm.log`
- Test without reload: `uvicorn api.main:app --host 0.0.0.0 --port 8000`

### Memory/ChromaDB Issues
- Delete and recreate: `rm -rf data/memory/chroma_db`
- Check disk space: `df -h`
- Review ChromaDB logs

## Key Design Decisions

1. **Modular Architecture**: Each component (models, memory, reasoning) is independent
2. **Device Agnostic**: Automatic detection and optimization for MPS/CUDA/CPU
3. **Memory Efficient**: Quantization and LoRA for resource-constrained environments
4. **Incremental Learning**: Fine-tune from interactions without full retraining
5. **Multi-Level Memory**: Short-term, long-term, and episodic for different use cases
6. **Production Ready**: FastAPI with proper error handling, logging, and monitoring

## Performance Considerations

- **Model Size**: TinyLlama-1.1B is default (adjust in config for larger models)
- **Quantization**: 8-bit recommended for < 16GB RAM, 4-bit for < 8GB RAM
- **Batch Size**: Auto-adjusted by DeviceManager based on available memory
- **Context Length**: Reduce MAX_LENGTH if memory constrained
- **Flash Attention**: Enable for compatible GPUs to reduce memory

## Testing Strategy

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **API Tests**: Test endpoints with various inputs
- **Performance Tests**: Benchmark generation speed and throughput
- **Manual Tests**: Use `if __name__ == "__main__"` sections

Run tests frequently during development to catch issues early.
