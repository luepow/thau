"""
THAU-Vision Inference Module
============================

Real-time inference capabilities:
- ImageQA: Question answering about images
- CameraProcessor: Real-time camera processing
- ObjectDetector: Object detection and recognition
"""

from .image_qa import ImageQA
from .camera import CameraProcessor

__all__ = ["ImageQA", "CameraProcessor"]
