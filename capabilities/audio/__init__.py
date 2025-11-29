"""
THAU Audio Capabilities
- Speech Recognition (mic to text)
- Text to Speech (coming soon)
"""

from .speech_recognition import ThauSpeechRecognition, create_speech_tool

__all__ = ["ThauSpeechRecognition", "create_speech_tool"]
