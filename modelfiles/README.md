# THAU Modelfiles for Ollama

This directory contains Modelfiles for creating THAU models in Ollama.

## Available Models

### Production Models (Recommended)

| Modelfile | Size | Description | Command |
|-----------|------|-------------|---------|
| `Modelfile_thau_7b_final` | 15 GB | THAU 7B - Best quality, cognitive reasoning | `ollama create thau-7b -f modelfiles/Modelfile_thau_7b_final` |
| `Modelfile_thau_1.1b` | 637 MB | THAU 1.1B - Lightweight, fast | `ollama create thau -f modelfiles/Modelfile_thau_1.1b` |

### Specialized Models

| Modelfile | Base | Specialty |
|-----------|------|-----------|
| `Modelfile_thau_advanced` | Qwen2.5-3B | Advanced reasoning + SVG |
| `Modelfile_thau_developer` | Qwen2.5-3B | Code generation focus |
| `Modelfile_thau_unified` | Qwen2.5-3B | Combined training |
| `Modelfile_reasoning` | Qwen2.5-3B | Chain of thought |
| `Modelfile_contable` | Qwen2.5-3B | Accounting (Spanish) |

### Legacy/Experimental

| Modelfile | Notes |
|-----------|-------|
| `Modelfile_agi_v2` | AGI experiment v2 |
| `Modelfile_agi_v3` | AGI experiment v3 |
| `Modelfile_spanish` | Spanish focus |
| `Modelfile_thau_books` | Trained on books |

## Usage

### Create a model from Modelfile:

```bash
# From project root
ollama create thau-7b -f modelfiles/Modelfile_thau_7b_final

# Run the model
ollama run thau-7b
```

### Push to Ollama Hub:

```bash
# Tag with your username
ollama cp thau-7b username/thau-7b

# Push
ollama push username/thau-7b
```

## Requirements

- Ollama installed (`brew install ollama` or https://ollama.ai)
- GGUF model files in `data/models/` directory
- For 7B model: 16GB+ RAM recommended

## Model Files Location

The GGUF files referenced in Modelfiles should be in:
- `data/models/thau-7b.gguf` - THAU 7B (15GB)
- `export/models/thau-f16.gguf` - THAU 1.1B F16

## Creating Custom Modelfiles

Template:

```
FROM ./path/to/model.gguf

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

SYSTEM """Your custom system prompt here."""
```
