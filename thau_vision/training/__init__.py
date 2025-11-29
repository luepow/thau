"""
THAU-Vision Training Module
===========================

Training infrastructure for vision-language models:
- VisionDataset: Dataset for image-text pairs
- VisionTrainer: Training loop with LoRA support
- DataCollator: Batch collation for vision training
"""

from .dataset import VisionDataset, VisionDataCollator
from .train_vision import VisionTrainer

__all__ = ["VisionDataset", "VisionDataCollator", "VisionTrainer"]
