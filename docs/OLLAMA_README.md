# THAU 7B - Cognitive AI Assistant

**Thinking Human-like Artificial Understanding**

THAU is a fine-tuned AI assistant specialized in cognitive reasoning, code generation, and multi-step problem solving.

## Quick Start

```bash
# Pull the model
ollama pull luepow/thau-7b

# Run interactive chat
ollama run luepow/thau-7b
```

## Model Variants

| Model | Size | RAM | Use Case |
|-------|------|-----|----------|
| `luepow/thau-7b` | 15 GB | 16GB+ | Best quality, complex tasks |
| `luepow/thau` | 637 MB | 4GB+ | Fast, lightweight |

## Capabilities

### Code Generation
- Python, JavaScript, TypeScript, Java, Rust, Go, SQL
- Clean Architecture, SOLID principles
- FastAPI, React, Spring Boot

### Cognitive Reasoning
- Step-by-step problem solving (Chain of Thought)
- Task decomposition and planning
- Multi-step reasoning

### Tool Calling
Native JSON tool invocation:
```json
<tool_call>{"name": "execute_python", "arguments": {"code": "print(2+2)"}}</tool_call>
```

### Accounting & Finance
- Double-entry bookkeeping
- Financial statements
- IFRS/GAAP compliance

### Bilingual
- Full Spanish and English support
- Technical documentation in both languages

## Example Prompts

**Code Generation:**
```
Create a FastAPI endpoint for user authentication with JWT tokens
```

**Reasoning:**
```
Explain step by step how to implement a binary search tree in Python
```

**Accounting:**
```
Record a journal entry for a $10,000 equipment purchase paid with cash
```

**SVG Generation:**
```
Create an animated loading spinner SVG
```

## System Requirements

- **RAM**: 16GB minimum (32GB recommended)
- **Disk**: 15GB free space
- **Platform**: macOS, Linux, Windows

## Base Model

- **Architecture**: Qwen2.5-7B-Instruct
- **Fine-tuning**: LoRA (r=16, alpha=32)
- **Context Length**: 4096 tokens
- **Training**: 677 specialized examples

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| temperature | 0.7 | Creativity level |
| top_p | 0.9 | Nucleus sampling |
| num_ctx | 4096 | Context window |

## Custom System Prompt

```bash
ollama run luepow/thau-7b "Your question" --system "You are a Python expert"
```

## API Usage

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "luepow/thau-7b",
  "prompt": "Explain recursion",
  "stream": false
}'
```

## License

Apache 2.0

## Links

- **GitHub**: https://github.com/luepow/thau
- **HuggingFace**: https://huggingface.co/luepow/thau-7b
- **Author**: Luis Perez (luepow@hotmail.com)

---

*THAU - Built with curiosity, love, and collaboration*
