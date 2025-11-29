#!/usr/bin/env python3
"""
Publica el modelo THAU en Hugging Face Hub

Uso:
    # Login primero
    huggingface-cli login

    # Publicar
    python scripts/publish_to_huggingface.py --repo tu-usuario/thau
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

from huggingface_hub import HfApi, create_repo, upload_folder


def create_model_card(model_dir: Path, repo_id: str) -> str:
    """Crea el README.md (Model Card) para el modelo"""

    # Leer config si existe
    config_path = model_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)

    hidden_size = config.get("hidden_size", "Unknown")
    num_layers = config.get("num_hidden_layers", "Unknown")
    vocab_size = config.get("vocab_size", "Unknown")
    model_type = config.get("model_type", "llama")

    model_card = f"""---
language:
  - es
  - en
license: apache-2.0
tags:
  - llm
  - conversational
  - text-generation
  - thau
  - self-learning
  - tool-calling
library_name: transformers
pipeline_tag: text-generation
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
---

# THAU - Self-Learning Language Model

<img src="https://img.shields.io/badge/THAU-LLM-blue" alt="THAU LLM">

## Model Description

**THAU** (Thinking, Helpful, Autonomous, Understanding) is a self-learning language model with incremental training capabilities. Built on top of TinyLlama, THAU has been fine-tuned using a unique "cognitive age" progression system.

### Key Features

- **Self-Learning**: Learns from interactions and self-generated Q&A
- **Tool Calling**: Supports MCP (Model Context Protocol) for tool invocation
- **Bilingual**: Trained primarily in Spanish with English support
- **Lightweight**: ~{hidden_size}M parameters, runs on consumer hardware

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Hidden Size | {hidden_size} |
| Layers | {num_layers} |
| Vocabulary Size | {vocab_size} |
| Model Type | {model_type} |
| Base Model | TinyLlama-1.1B-Chat |

## Training

THAU uses a progressive "cognitive age" training system:

- **Age 0-3**: Basic language, simple patterns
- **Age 4-6**: Grammar, vocabulary expansion
- **Age 7-9**: Reasoning, logic
- **Age 10-12**: Advanced topics, programming
- **Age 13-15**: Specialized knowledge, tool use

### Training Data

- Self-generated Q&A pairs via Ollama teachers
- Programming tutorials (Python, JavaScript, C++, etc.)
- Tool calling examples (MCP format)
- General knowledge across multiple domains

## Usage

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

prompt = "Hola, que puedes hacer?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### With Ollama

```bash
# Download and convert
ollama pull {repo_id.split('/')[-1]}

# Or create from GGUF
ollama create thau -f Modelfile
```

### Tool Calling Format

THAU supports tool calling with this format:

```
<tool_call>{{"name": "tool_name", "arguments": {{"param": "value"}}}}</tool_call>
```

Example tools: `get_current_time`, `web_search`, `execute_python`, `generate_image`

## Limitations

- Model size limits complex reasoning
- May hallucinate on topics outside training data
- Tool calling accuracy depends on training quality
- Spanish-primary, English secondary

## Ethical Considerations

This model was trained on self-generated data and open datasets. It should not be used for:
- Generating harmful or misleading content
- Impersonating real individuals
- Making critical decisions without human oversight

## Citation

```bibtex
@misc{{thau2024,
  title={{THAU: A Self-Learning Language Model}},
  author={{THAU Team}},
  year={{2024}},
  url={{https://huggingface.co/{repo_id}}}
}}
```

## License

Apache 2.0

---

*THAU - Built with incremental learning and cognitive progression*
"""
    return model_card


def publish_model(
    model_dir: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = None
):
    """Publica el modelo a Hugging Face Hub"""

    model_path = Path(model_dir)

    if not model_path.exists():
        print(f"Error: Directorio no encontrado: {model_path}")
        return False

    # Verificar archivos necesarios
    required_files = ["config.json", "tokenizer.json"]
    model_files = ["model.safetensors", "pytorch_model.bin"]

    for f in required_files:
        if not (model_path / f).exists():
            print(f"Advertencia: Falta {f}")

    has_model = any((model_path / f).exists() for f in model_files)
    if not has_model:
        print("Error: No se encontro archivo de modelo (safetensors o bin)")
        return False

    # Crear Model Card
    print("\n1. Creando Model Card...")
    readme_path = model_path / "README.md"
    model_card = create_model_card(model_path, repo_id)
    with open(readme_path, 'w') as f:
        f.write(model_card)
    print(f"   Creado: {readme_path}")

    # Crear repo
    print(f"\n2. Creando repositorio: {repo_id}")
    api = HfApi()

    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        print(f"   Repositorio creado/verificado")
    except Exception as e:
        print(f"   Error creando repo: {e}")
        return False

    # Subir archivos
    print(f"\n3. Subiendo archivos a {repo_id}...")

    if commit_message is None:
        commit_message = f"Upload THAU model - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    try:
        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            ignore_patterns=["*.pyc", "__pycache__", ".git*", "*.log"]
        )
        print(f"   Archivos subidos exitosamente")
    except Exception as e:
        print(f"   Error subiendo: {e}")
        return False

    # Resumen
    print("\n" + "=" * 60)
    print("  PUBLICACION EXITOSA")
    print("=" * 60)
    print(f"\n  URL: https://huggingface.co/{repo_id}")
    print(f"  Tipo: {'Privado' if private else 'Publico'}")
    print(f"\n  Uso:")
    print(f"    from transformers import AutoModelForCausalLM")
    print(f"    model = AutoModelForCausalLM.from_pretrained('{repo_id}')")
    print("\n" + "=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description="Publicar THAU en Hugging Face")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./export/merged/thau-tool-calling",
        help="Directorio del modelo"
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="ID del repositorio (usuario/nombre)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Crear repositorio privado"
    )
    parser.add_argument(
        "--message",
        type=str,
        default=None,
        help="Mensaje de commit"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  THAU - Publicacion en Hugging Face")
    print("=" * 60)

    # Verificar login
    print("\nVerificando autenticacion...")
    try:
        api = HfApi()
        user = api.whoami()
        print(f"   Autenticado como: {user['name']}")
    except Exception as e:
        print(f"\n   No estas autenticado. Ejecuta primero:")
        print(f"   huggingface-cli login")
        print(f"\n   O configura HF_TOKEN en tu entorno")
        return

    # Publicar
    success = publish_model(
        model_dir=args.model_dir,
        repo_id=args.repo,
        private=args.private,
        commit_message=args.message
    )

    if not success:
        print("\n   Error en la publicacion")
        exit(1)


if __name__ == "__main__":
    main()
