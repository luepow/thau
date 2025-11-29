# THAU v2.0 - Self-Learning Language Model

A self-learning language model with specialized training in tool calling, reasoning, and Spanish.

## Overview

**THAU** (Thinking, Helpful, Autonomous, Understanding) is a lightweight language model built on TinyLlama-1.1B, fine-tuned with LoRA using specialized training data. Version 2.0 includes enhanced tool calling, chain-of-thought reasoning, and improved Spanish language support.

## What's New in v2.0

- **Enhanced Tool Calling**: 112 specialized examples for reliable function invocation
- **Chain of Thought**: Step-by-step reasoning for complex problems
- **Image Generation**: Prompt engineering for image generation tools
- **Spanish Fluency**: Natural and technical Spanish conversations
- **Lower Loss**: Trained to 0.43 loss (down from 0.94)

## Features

- **Self-Learning**: Learns from interactions and self-generated Q&A
- **Tool Calling**: Native support for function calling with structured JSON format
- **Bilingual**: Trained primarily in Spanish with English support
- **Lightweight**: ~1.1B parameters, runs efficiently on CPU or GPU
- **Context Window**: 2048 tokens

## Quick Start

```bash
# Pull the model
ollama pull luepow/thau

# Run interactive chat
ollama run luepow/thau

# Run with a prompt
ollama run luepow/thau "Hola, que puedes hacer?"
```

## Tool Calling

THAU supports tool calling with the following format:

```
<tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>
```

### Available Tools

| Tool | Description |
|------|-------------|
| `get_current_time` | Get current date and time |
| `web_search` | Search for information online |
| `execute_python` | Execute Python code |
| `generate_image` | Generate an image from a prompt |

### Example

**User**: What time is it?

**THAU**:
```
<tool_call>{"name": "get_current_time", "arguments": {}}</tool_call>
```

## API Usage

### With curl

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "luepow/thau",
  "prompt": "Explica que es machine learning",
  "stream": false
}'
```

### With Python

```python
import requests

response = requests.post('http://localhost:11434/api/generate', json={
    'model': 'luepow/thau',
    'prompt': 'Hola, como estas?',
    'stream': False
})
print(response.json()['response'])
```

### Chat API

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "luepow/thau",
  "messages": [
    {"role": "user", "content": "Hola!"}
  ],
  "stream": false
}'
```

## Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.7 | Controls randomness (0-2) |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `top_k` | 40 | Top-k sampling |
| `repeat_penalty` | 1.1 | Penalize repetitions |
| `num_ctx` | 2048 | Context window size |

### Customize Parameters

```bash
ollama run luepow/thau --temperature 0.5 --num-ctx 4096
```

## Training System

THAU uses a progressive "cognitive age" training approach:

| Age Range | Focus Areas |
|-----------|-------------|
| 0-3 | Basic language, simple patterns |
| 4-6 | Grammar, vocabulary expansion |
| 7-9 | Reasoning, logic |
| 10-12 | Advanced topics, programming |
| 13-15 | Specialized knowledge, tool use |

### Training Data Sources

- Self-generated Q&A pairs via teacher models
- Programming tutorials (Python, JavaScript, C++, etc.)
- Tool calling examples (MCP format)
- General knowledge across multiple domains

## Architecture

| Component | Value |
|-----------|-------|
| Base Model | TinyLlama-1.1B-Chat-v1.0 |
| Parameters | ~1.1B |
| Hidden Size | 2048 |
| Layers | 22 |
| Vocabulary | 32,000 tokens |
| Format | GGUF F16 (2.2 GB) |
| Training | LoRA Fine-tuning |
| Final Loss | 0.43 |

## Training Data (v2.0)

| Category | Examples |
|----------|----------|
| Tool Calling | 112 |
| Spanish Natural/Technical | 52 |
| Image Generation | 30 |
| Conversational Spanish | 20 |
| Chain of Thought Reasoning | 20 |
| Programming | 30+ |
| **Total** | **297 specialized** |

## Limitations

- Model size limits complex multi-step reasoning
- May hallucinate on topics outside training data
- Tool calling accuracy varies by complexity
- Spanish is the primary language; English is secondary
- Best for simple to moderate complexity tasks

## Use Cases

- **Conversational AI**: General chat and Q&A
- **Code Assistance**: Simple programming help
- **Tool Integration**: Automated task execution
- **Learning Assistant**: Educational explanations
- **Prototyping**: Quick AI integration testing

## Ethical Guidelines

This model should NOT be used for:
- Generating harmful or misleading content
- Impersonating real individuals
- Making critical decisions without human oversight
- Any illegal or unethical purposes

## Links

- **HuggingFace**: [luepow/thau](https://huggingface.co/luepow/thau)
- **License**: Apache 2.0

## Citation

```bibtex
@misc{thau2024,
  title={THAU v2.0: A Self-Learning Language Model},
  author={Luis Perez (luepow)},
  year={2024},
  url={https://ollama.com/luepow/thau}
}
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v2.0 | Nov 2024 | Specialized training: tool calling, reasoning, Spanish |
| v1.0 | Nov 2024 | Initial release with cognitive age progression |

---

*THAU v2.0 - Built with incremental learning and specialized training*

*Dedicated to Thomas & Aurora*
