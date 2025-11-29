"""
THAU-Vision: Advanced Vision-Language Model System
====================================================

A complete Vision-Language Model (VLM) system that enables THAU to:
- Understand images and describe them
- Answer questions about visual content
- Learn from images with labels
- Process camera input in real-time
- OCR and document understanding
- Object detection and recognition

Architecture:
- Vision Encoder: SigLIP/CLIP for image embeddings
- Projection: MLP to map vision -> LLM space
- LLM: TinyLlama backbone
- Multi-modal fusion for image+text understanding

Author: Luis Perez (with Claude)
License: Apache 2.0
"""

__version__ = "1.0.0"
__author__ = "Luis Perez"

from .models import THAUVisionModel, VisionEncoder, ProjectionLayer
from .inference import ImageQA, CameraProcessor
from .training import VisionTrainer, VisionDataset

__all__ = [
    "THAUVisionModel",
    "VisionEncoder",
    "ProjectionLayer",
    "ImageQA",
    "CameraProcessor",
    "VisionTrainer",
    "VisionDataset",
]
