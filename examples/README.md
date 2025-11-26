# THAU Examples

This directory contains example scripts demonstrating how to use THAU.

## Available Examples

### 1. Simple Chat (`simple_chat.py`)

Interactive chat with a trained THAU model.

```bash
python examples/simple_chat.py
```

**Features:**
- Loads pre-trained model
- Interactive chat loop
- Clean exit with Ctrl+C

### 2. Custom Training (`train_custom.py`)

Train THAU with your own dataset.

```bash
python examples/train_custom.py
```

**Features:**
- Custom dataset definition
- Training loop with progress
- Checkpoint saving
- Generation testing

### 3. API Usage (`api_client.py`)

Use THAU through REST API.

```bash
# Start server first
python api/thau_code_server.py

# Then run client
python examples/api_client.py
```

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model first
python train_ages_simple.py
```

### Running Examples

All examples are standalone scripts that can be run directly:

```bash
# Make executable
chmod +x examples/*.py

# Run any example
python examples/simple_chat.py
```

## Creating Your Own Examples

Base template:

```python
#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from thau_trainer.own_model_manager import ThauOwnModelManager

def main():
    # Your code here
    manager = ThauOwnModelManager()
    manager.initialize_model(cognitive_age=0)
    # ... rest of your code

if __name__ == "__main__":
    main()
```

## Need Help?

- Check the [documentation](../docs/)
- Open an [issue](https://github.com/luepow/thau/issues)
- Join [discussions](https://github.com/luepow/thau/discussions)
