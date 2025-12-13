"""
THAU Voice Module
Módulo de síntesis de voz para THAU
"""

from .tts_service import ThauVoice, thau_voice, speak, speak_aloud, SPANISH_VOICES

__all__ = [
    "ThauVoice",
    "thau_voice",
    "speak",
    "speak_aloud",
    "SPANISH_VOICES",
]
