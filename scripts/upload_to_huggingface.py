#!/usr/bin/env python3
"""
Script para subir modelos THAU a HuggingFace Hub

Uso:
    1. Obtén un token de HuggingFace: https://huggingface.co/settings/tokens
    2. Ejecuta: python scripts/upload_to_huggingface.py --token TU_TOKEN

    O configura la variable de entorno:
    export HF_TOKEN=tu_token
    python scripts/upload_to_huggingface.py

Modelos disponibles:
    --model thau-7b     : THAU 7B (Qwen2.5-7B-Instruct + LoRA)
    --model thau-1.1b   : THAU 1.1B (TinyLlama base)
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configuración de modelos disponibles
MODELS = {
    "thau-7b": {
        "dir": "data/models/thau-7b-merged",
        "repo": "luepow/thau-7b",
        "description": "THAU 7B - Cognitive AI Assistant based on Qwen2.5-7B-Instruct",
        "commit_msg": "THAU 7B - Fine-tuned with LoRA for cognitive reasoning"
    },
    "thau-1.1b": {
        "dir": "export/models/hf_model/TinyLlama_TinyLlama-1.1B-Chat-v1.0",
        "repo": "luepow/thau",
        "description": "THAU 1.1B - Unified AI Assistant",
        "commit_msg": "THAU 1.1B - Unified model upload"
    }
}


def upload_model(token: str, model_name: str, repo_override: str = None):
    """Sube el modelo a HuggingFace Hub"""
    from huggingface_hub import HfApi, create_repo

    if model_name not in MODELS:
        print(f"Error: Modelo '{model_name}' no reconocido")
        print(f"Modelos disponibles: {', '.join(MODELS.keys())}")
        sys.exit(1)

    config = MODELS[model_name]
    api = HfApi(token=token)

    # Directorio del modelo
    model_dir = project_root / config["dir"]
    repo_id = repo_override or config["repo"]

    if not model_dir.exists():
        print(f"Error: No se encontró el directorio del modelo: {model_dir}")
        sys.exit(1)

    print("=" * 60)
    print(f"Subiendo {model_name} a HuggingFace")
    print("=" * 60)
    print(f"Directorio: {model_dir}")
    print(f"Repositorio: {repo_id}")
    print(f"Descripción: {config['description']}")

    # Crear repositorio si no existe
    try:
        create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
        print(f"\nRepositorio {repo_id} listo")
    except Exception as e:
        print(f"Error creando repositorio: {e}")
        sys.exit(1)

    # Subir archivos
    print("\nSubiendo archivos...")

    try:
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=config["commit_msg"]
        )
        print(f"\n{'=' * 60}")
        print(f"Modelo subido exitosamente!")
        print(f"{'=' * 60}")
        print(f"URL: https://huggingface.co/{repo_id}")

    except Exception as e:
        print(f"Error subiendo archivos: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Subir modelos THAU a HuggingFace Hub")
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Token de HuggingFace (o usa HF_TOKEN env var)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="thau-7b",
        choices=list(MODELS.keys()),
        help="Modelo a subir (default: thau-7b)"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Override del repositorio destino"
    )

    args = parser.parse_args()

    if not args.token:
        print("Error: Se requiere un token de HuggingFace")
        print("\nOpciones:")
        print("  1. Pasa el token: python scripts/upload_to_huggingface.py --token TU_TOKEN")
        print("  2. O configura: export HF_TOKEN=tu_token")
        print("\nObtén tu token en: https://huggingface.co/settings/tokens")
        sys.exit(1)

    upload_model(args.token, args.model, args.repo)


if __name__ == "__main__":
    main()
