"""Training-specific configurations."""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Training parameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_scheduler_type: str = "cosine"  # Options: linear, cosine, constant
    warmup_steps: int = 100
    warmup_ratio: float = 0.0  # Alternative to warmup_steps

    # Optimization
    optimizer_type: str = "adamw"  # Options: adamw, adam, sgd
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Data loading
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    max_seq_length: int = 2048

    # Checkpointing
    save_steps: int = 500
    save_total_limit: int = 3  # Keep only last N checkpoints
    checkpoint_dir: str = "./data/checkpoints"

    # Evaluation
    eval_steps: int = 500
    eval_strategy: str = "steps"  # Options: no, steps, epoch
    per_device_eval_batch_size: Optional[int] = None  # Defaults to batch_size

    # Logging
    logging_steps: int = 10
    logging_dir: str = "./data/logs"
    report_to: List[str] = None  # Options: wandb, tensorboard, none

    # Mixed precision
    fp16: bool = False
    bf16: bool = False  # Better for modern GPUs

    # Gradient checkpointing (saves memory)
    gradient_checkpointing: bool = False

    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if self.per_device_eval_batch_size is None:
            self.per_device_eval_batch_size = self.batch_size

        if self.report_to is None:
            self.report_to = []

        # Validate parameters
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0.0 <= self.weight_decay <= 1.0, "weight_decay must be between 0 and 1"

    def to_transformers_args(self) -> dict:
        """Convert to HuggingFace Transformers TrainingArguments format.

        Returns:
            Dictionary compatible with TrainingArguments.
        """
        return {
            "num_train_epochs": self.num_epochs,
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_steps": self.warmup_steps,
            "warmup_ratio": self.warmup_ratio,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "output_dir": self.checkpoint_dir,
            "evaluation_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
            "logging_dir": self.logging_dir,
            "report_to": self.report_to,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
        }


@dataclass
class DatasetConfig:
    """Configuration for dataset handling."""

    # Dataset paths
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None

    # HuggingFace datasets
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_split: str = "train"

    # Processing
    max_samples: Optional[int] = None  # Limit dataset size for testing
    shuffle: bool = True
    seed: int = 42

    # Text column names
    text_column: str = "text"
    prompt_column: Optional[str] = None
    response_column: Optional[str] = None

    # Preprocessing
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False

    def __post_init__(self):
        """Validate configuration."""
        has_files = any([self.train_file, self.validation_file, self.test_file])
        has_hf_dataset = self.dataset_name is not None

        assert has_files or has_hf_dataset, (
            "Must specify either local files (train_file, etc.) or "
            "HuggingFace dataset (dataset_name)"
        )


# Predefined training configurations
QUICK_TRAIN = TrainingConfig(
    num_epochs=1,
    batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    warmup_steps=50,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
)

STANDARD_TRAIN = TrainingConfig(
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_steps=100,
    save_steps=500,
    eval_steps=500,
    logging_steps=10,
    gradient_checkpointing=True,
)

INTENSIVE_TRAIN = TrainingConfig(
    num_epochs=10,
    batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    warmup_steps=500,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=50,
    gradient_checkpointing=True,
    early_stopping_patience=3,
)
