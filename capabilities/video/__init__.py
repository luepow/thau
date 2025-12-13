"""
THAU Video Module
Módulo de generación de video y animaciones para THAU
"""

from .svg_animator import SVGAnimator, SVGElement, AnimationConfig, create_animation

__all__ = [
    "SVGAnimator",
    "SVGElement",
    "AnimationConfig",
    "create_animation",
]
