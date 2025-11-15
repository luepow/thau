"""Base configuration for my-llm project."""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Main configuration class for my-llm project.

    All configuration parameters can be set via environment variables or
    loaded from a JSON file. Environment variables take precedence.
    """

    # Model Configuration
    MODEL_NAME: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    MODEL_PATH: str = field(default_factory=lambda: os.getenv("MODEL_PATH", "./data/models"))
    DEVICE: str = field(default_factory=lambda: os.getenv("DEVICE", "auto"))

    # Training Configuration
    BATCH_SIZE: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "4")))
    LEARNING_RATE: float = field(default_factory=lambda: float(os.getenv("LEARNING_RATE", "2e-5")))
    MAX_LENGTH: int = field(default_factory=lambda: int(os.getenv("MAX_LENGTH", "2048")))
    NUM_EPOCHS: int = field(default_factory=lambda: int(os.getenv("NUM_EPOCHS", "3")))
    GRADIENT_ACCUMULATION_STEPS: int = field(default_factory=lambda: int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4")))
    WARMUP_STEPS: int = field(default_factory=lambda: int(os.getenv("WARMUP_STEPS", "100")))

    # Generation Parameters
    TEMPERATURE: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))
    TOP_P: float = field(default_factory=lambda: float(os.getenv("TOP_P", "0.9")))
    TOP_K: int = field(default_factory=lambda: int(os.getenv("TOP_K", "50")))
    MAX_NEW_TOKENS: int = field(default_factory=lambda: int(os.getenv("MAX_NEW_TOKENS", "512")))

    # Memory Configuration
    MEMORY_DB_PATH: str = field(default_factory=lambda: os.getenv("MEMORY_DB_PATH", "./data/memory/chroma_db"))
    EPISODIC_DB_PATH: str = field(default_factory=lambda: os.getenv("EPISODIC_DB_PATH", "./data/memory/episodic.db"))
    SHORT_TERM_MEMORY_SIZE: int = field(default_factory=lambda: int(os.getenv("SHORT_TERM_MEMORY_SIZE", "4096")))
    MAX_LONG_TERM_MEMORIES: int = field(default_factory=lambda: int(os.getenv("MAX_LONG_TERM_MEMORIES", "10000")))

    # API Configuration
    API_HOST: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    API_PORT: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    API_WORKERS: int = field(default_factory=lambda: int(os.getenv("API_WORKERS", "1")))
    CORS_ORIGINS: List[str] = field(default_factory=lambda: json.loads(os.getenv("CORS_ORIGINS", '["http://localhost:3000", "http://localhost:8000"]')))

    # Logging
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    LOG_FILE: str = field(default_factory=lambda: os.getenv("LOG_FILE", "./data/logs/my-llm.log"))

    # Optional: Weights & Biases
    WANDB_PROJECT: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_PROJECT"))
    WANDB_ENTITY: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_ENTITY"))
    WANDB_API_KEY: Optional[str] = field(default_factory=lambda: os.getenv("WANDB_API_KEY"))

    # Optional: HuggingFace Token
    HF_TOKEN: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN"))

    # Performance
    USE_FLASH_ATTENTION: bool = field(default_factory=lambda: os.getenv("USE_FLASH_ATTENTION", "false").lower() == "true")
    USE_QUANTIZATION: bool = field(default_factory=lambda: os.getenv("USE_QUANTIZATION", "true").lower() == "true")
    QUANTIZATION_BITS: int = field(default_factory=lambda: int(os.getenv("QUANTIZATION_BITS", "8")))

    def __post_init__(self):
        """Validate and create necessary directories after initialization."""
        # Create necessary directories
        Path(self.MODEL_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.MEMORY_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(self.EPISODIC_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(self.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

        # Validate ranges
        assert 0.0 <= self.TEMPERATURE <= 2.0, "Temperature must be between 0.0 and 2.0"
        assert 0.0 <= self.TOP_P <= 1.0, "Top-p must be between 0.0 and 1.0"
        assert self.TOP_K > 0, "Top-k must be positive"
        assert self.BATCH_SIZE > 0, "Batch size must be positive"
        assert self.LEARNING_RATE > 0, "Learning rate must be positive"

    @classmethod
    def load_from_file(cls, filepath: str) -> "Config":
        """Load configuration from a JSON file.

        Args:
            filepath: Path to the JSON configuration file.

        Returns:
            Config instance with loaded parameters.
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def save(self, filepath: str) -> None:
        """Save configuration to a JSON file.

        Args:
            filepath: Path where to save the configuration.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return asdict(self)

    def __repr__(self) -> str:
        """String representation of the configuration."""
        config_str = "Config(\n"
        for key, value in asdict(self).items():
            # Hide sensitive information
            if any(sensitive in key.lower() for sensitive in ["token", "key", "password"]):
                value = "***" if value else None
            config_str += f"  {key}={value},\n"
        config_str += ")"
        return config_str


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance.

    Returns:
        The global Config instance.
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance.

    Args:
        config: The Config instance to set globally.
    """
    global _config
    _config = config


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print(config)

    # Test saving and loading
    test_path = "./data/test_config.json"
    config.save(test_path)
    loaded_config = Config.load_from_file(test_path)
    print("\nLoaded config:")
    print(loaded_config)
