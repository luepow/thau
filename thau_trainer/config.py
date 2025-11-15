"""
Configuración del sistema de entrenamiento autónomo THAU
"""

from dataclasses import dataclass
from typing import Literal
import os


@dataclass
class ThauConfig:
    """Configuración del modelo THAU"""

    # Identificación del modelo
    model_name: str = "thau"
    model_size: Literal["1.5b", "3b", "7b", "13b"] = "1.5b"
    current_version: int = 1

    # Base model
    base_model_map = {
        "1.5b": "qwen2.5-coder:1.5b-base",
        "3b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Placeholder
        "7b": "meta-llama/Llama-2-7b-chat-hf",
        "13b": "meta-llama/Llama-2-13b-chat-hf",
    }

    # Rutas
    data_dir: str = "./data"
    checkpoints_dir: str = "./data/checkpoints"
    datasets_dir: str = "./data/datasets"
    logs_dir: str = "./data/logs"
    training_queue_dir: str = "./data/training_queue"

    # Entrenamiento autónomo
    auto_train_enabled: bool = True
    auto_train_interval_hours: int = 24  # Entrenar cada 24 horas
    min_new_examples: int = 10  # Mínimo de ejemplos nuevos para entrenar
    batch_size: int = 2
    epochs_per_training: int = 3
    learning_rate: float = 2e-4

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64

    # Recursos
    max_context_length: int = 2048
    use_quantization: bool = True

    def get_model_identifier(self) -> str:
        """Retorna el identificador completo del modelo"""
        return f"{self.model_name}-{self.model_size}-v{self.current_version}"

    def get_checkpoint_path(self, version: int = None) -> str:
        """Retorna la ruta del checkpoint"""
        v = version or self.current_version
        return os.path.join(
            self.checkpoints_dir,
            f"{self.model_name}-{self.model_size}-v{v}"
        )

    def get_base_model(self) -> str:
        """Retorna el modelo base según el tamaño"""
        return self.base_model_map.get(self.model_size)

    def increment_version(self):
        """Incrementa la versión del modelo"""
        self.current_version += 1
        return self.current_version


# Singleton global
_config = None


def get_config() -> ThauConfig:
    """Obtiene la configuración global"""
    global _config
    if _config is None:
        _config = ThauConfig()
    return _config


def set_model_size(size: Literal["1.5b", "3b", "7b", "13b"]):
    """Establece el tamaño del modelo"""
    config = get_config()
    config.model_size = size
    return config
