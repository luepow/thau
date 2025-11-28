# THAU - Self-Learning Language Model

A lightweight, self-learning language model with tool calling capabilities.

> *Dedicated to Thomas and Aurora - watching you learn and grow inspired this project*

## The Story Behind THAU

**THAU** was born from a simple question: *"Can an AI learn progressively, like a child does?"*

As a developer and father, I (Luis Perez) was fascinated by how my children Thomas and Aurora learn - starting with basic concepts and gradually building more complex understanding. This inspired me to create a framework that mimics this cognitive progression in AI.

### Why I Built This

- **Curiosity**: I wanted to understand how LLMs work from the inside out
- **Experimentation**: To test if progressive "cognitive age" training could improve model quality
- **Learning**: Building something hands-on is the best way to learn
- **Open Source**: To share the journey with others who are curious about AI

### Built With Claude

This entire project was developed in collaboration with **Claude** (Anthropic's AI assistant). From architecture decisions to code implementation, debugging, and documentation - Claude has been my pair programming partner throughout this journey. It's a testament to what human-AI collaboration can achieve.

## Overview

**THAU** (Thinking, Helpful, Autonomous, Understanding) is a language model built on TinyLlama-1.1B, fine-tuned using a unique "cognitive age" progression system. It supports native tool calling and runs efficiently on consumer hardware.

## Features

- **Self-Learning**: Learns from interactions and self-generated Q&A pairs
- **Tool Calling**: Native JSON-based function calling support
- **Bilingual**: Spanish primary, English secondary
- **Lightweight**: ~1.1B parameters, 2.2GB model size
- **Fast**: Optimized for quick inference on CPU/GPU

## Quick Start

```bash
# Install
ollama pull luepow/thau

# Run
ollama run luepow/thau

# With prompt
ollama run luepow/thau "Hola, que puedes hacer?"
```

## Tool Calling

THAU supports structured tool calling:

```
<tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>
```

### Built-in Tools

| Tool | Description |
|------|-------------|
| `get_current_time` | Get current date and time |
| `web_search` | Search the web |
| `execute_python` | Run Python code |
| `generate_image` | Generate images from prompts |

### Example

**User**: What time is it?

**THAU**:
```
<tool_call>{"name": "get_current_time", "arguments": {}}</tool_call>
```

## API Usage

### REST API

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "luepow/thau",
  "prompt": "Explica que es machine learning",
  "stream": false
}'
```

### Chat API

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "luepow/thau",
  "messages": [{"role": "user", "content": "Hola!"}],
  "stream": false
}'
```

### Python

```python
import requests

response = requests.post('http://localhost:11434/api/generate', json={
    'model': 'luepow/thau',
    'prompt': 'Hola, como estas?',
    'stream': False
})
print(response.json()['response'])
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.7 | Randomness (0-2) |
| `top_p` | 0.9 | Nucleus sampling |
| `top_k` | 40 | Top-k sampling |
| `repeat_penalty` | 1.1 | Repetition penalty |
| `num_ctx` | 2048 | Context window |

### Customize

```bash
ollama run luepow/thau --temperature 0.5 --num-ctx 4096
```

## Training System

THAU uses progressive "cognitive age" training:

| Age | Focus |
|-----|-------|
| 0-3 | Basic language, patterns |
| 4-6 | Grammar, vocabulary |
| 7-9 | Reasoning, logic |
| 10-12 | Programming, advanced topics |
| 13-15 | Specialization, tool use |

## Architecture

| Component | Value |
|-----------|-------|
| Base Model | TinyLlama-1.1B-Chat |
| Parameters | ~1.1B |
| Hidden Size | 2048 |
| Layers | 22 |
| Vocabulary | 32,000 |
| Format | GGUF F16 |

## Limitations

- Limited complex reasoning due to model size
- May hallucinate on unfamiliar topics
- Spanish-first, English secondary
- Best for simple to moderate tasks

## Links

- [HuggingFace](https://huggingface.co/luepow/thau)
- [GitHub](https://github.com/luepow/thau)

## About the Author

**Luis Perez** - Software developer, father, and AI enthusiast.

- GitHub: [@luepow](https://github.com/luepow)
- Email: luepow@hotmail.com

## Acknowledgments

- **Thomas & Aurora** - My children, whose learning journey inspired this project
- **Claude (Anthropic)** - AI pair programming partner throughout development
- **TinyLlama Team** - For the excellent base model
- **Hugging Face** - For the transformers library and model hosting
- **Ollama Team** - For making local LLM deployment accessible

## Support This Project

If you find THAU interesting or useful, consider supporting its development:

- [Buy Me a Coffee](https://buymeacoffee.com/luepowg)
- [GitHub Sponsors](https://github.com/sponsors/luepow)
- [PayPal](https://paypal.me/luepow)

Your support helps cover compute costs and keeps this project alive!

## License

Apache 2.0

---

*THAU - Built with curiosity, love, and a lot of help from Claude*

*"The best way to learn is to build something"*
