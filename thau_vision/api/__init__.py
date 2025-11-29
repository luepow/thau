"""
THAU-Vision API Module
======================

REST API for vision-language capabilities.
"""

from .server import create_vision_api, VisionAPI

__all__ = ["create_vision_api", "VisionAPI"]
