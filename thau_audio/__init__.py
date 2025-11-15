"""
THAU Audio Module
Capacidades de procesamiento de audio, speech-to-text y text-to-speech
"""

from thau_audio.asr.transcriber import AudioTranscriber
from thau_audio.tts.synthesis import TextToSpeech
from thau_audio.processing.audio_editor import AudioProcessor

__all__ = [
    'AudioTranscriber',
    'TextToSpeech',
    'AudioProcessor'
]

__version__ = '0.1.0'
