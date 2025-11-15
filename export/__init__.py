"""
THAU Model Export Module
Export THAU to Ollama and LMStudio compatible formats
"""

__version__ = "1.0.0"
__author__ = "THAU Team"

from pathlib import Path

# Module paths
EXPORT_DIR = Path(__file__).parent
MODELS_DIR = EXPORT_DIR / "models"
LLAMA_CPP_DIR = EXPORT_DIR / "llama.cpp"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

__all__ = [
    "EXPORT_DIR",
    "MODELS_DIR",
    "LLAMA_CPP_DIR",
]
