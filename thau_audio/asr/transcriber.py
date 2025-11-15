"""
Audio Transcriber using Whisper
Converts speech to text with high accuracy
"""

from typing import Dict, Optional, Union
import os
from datetime import datetime
import numpy as np

try:
    import whisper
    import librosa
    import soundfile as sf
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  Whisper not available. Install with: pip install openai-whisper librosa soundfile")


class AudioTranscriber:
    """
    Transcriptor de audio usando Whisper de OpenAI

    Convierte audio (archivos o bytes) a texto con alta precisión
    """

    def __init__(self, model_size: str = "base", device: str = "cpu", language: Optional[str] = None):
        """
        Inicializa el transcriptor

        Args:
            model_size: Tamaño del modelo Whisper (tiny, base, small, medium, large)
            device: Dispositivo para inferencia (cpu, cuda, mps)
            language: Idioma esperado (None para detección automática)
        """
        self.model_size = model_size
        self.device = device
        self.language = language
        self.model = None
        self.transcription_history = []

        if WHISPER_AVAILABLE:
            self._load_model()
        else:
            print("⚠️  Whisper no disponible. Usando modo simulación.")

    def _load_model(self):
        """Carga el modelo de Whisper"""
        try:
            print(f"📦 Cargando modelo Whisper ({self.model_size})...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            print(f"✅ Modelo Whisper cargado: {self.model_size}")
        except Exception as e:
            print(f"❌ Error cargando Whisper: {e}")
            self.model = None

    def transcribe_file(self, audio_path: str, **kwargs) -> Dict:
        """
        Transcribe un archivo de audio

        Args:
            audio_path: Ruta al archivo de audio
            **kwargs: Argumentos adicionales para Whisper

        Returns:
            Dict con transcripción y metadata
        """
        if not os.path.exists(audio_path):
            return {
                "success": False,
                "error": f"Archivo no encontrado: {audio_path}"
            }

        if not WHISPER_AVAILABLE or self.model is None:
            # Modo simulación
            return self._simulate_transcription(audio_path)

        try:
            # Transcribir con Whisper
            result = self.model.transcribe(
                audio_path,
                language=self.language,
                **kwargs
            )

            transcription = {
                "success": True,
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "audio_path": audio_path,
                "model": self.model_size,
                "timestamp": datetime.now().isoformat(),
                "duration": self._get_audio_duration(audio_path)
            }

            self.transcription_history.append(transcription)

            return transcription

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "audio_path": audio_path
            }

    def transcribe_array(self, audio_array: np.ndarray, sample_rate: int = 16000, **kwargs) -> Dict:
        """
        Transcribe un array de audio

        Args:
            audio_array: Array numpy con datos de audio
            sample_rate: Frecuencia de muestreo
            **kwargs: Argumentos adicionales

        Returns:
            Dict con transcripción
        """
        if not WHISPER_AVAILABLE or self.model is None:
            return self._simulate_transcription(None)

        try:
            # Whisper espera audio a 16kHz
            if sample_rate != 16000:
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=sample_rate,
                    target_sr=16000
                )

            # Normalizar audio
            audio_array = audio_array.astype(np.float32)
            audio_array = audio_array / np.max(np.abs(audio_array))

            result = self.model.transcribe(
                audio_array,
                language=self.language,
                **kwargs
            )

            return {
                "success": True,
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "model": self.model_size,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _get_audio_duration(self, audio_path: str) -> float:
        """Obtiene la duración del audio en segundos"""
        try:
            if WHISPER_AVAILABLE:
                y, sr = librosa.load(audio_path, sr=None)
                return len(y) / sr
            return 0.0
        except:
            return 0.0

    def _simulate_transcription(self, audio_path: Optional[str]) -> Dict:
        """Simulación cuando Whisper no está disponible"""
        return {
            "success": True,
            "text": "Transcripción simulada: Este es un texto de prueba generado automáticamente.",
            "language": self.language or "es",
            "segments": [
                {
                    "start": 0.0,
                    "end": 3.0,
                    "text": "Transcripción simulada:"
                },
                {
                    "start": 3.0,
                    "end": 7.0,
                    "text": "Este es un texto de prueba generado automáticamente."
                }
            ],
            "audio_path": audio_path,
            "model": self.model_size,
            "timestamp": datetime.now().isoformat(),
            "simulated": True
        }

    def get_stats(self) -> Dict:
        """Obtiene estadísticas de transcripciones"""
        return {
            "total_transcriptions": len(self.transcription_history),
            "model_size": self.model_size,
            "device": self.device,
            "whisper_available": WHISPER_AVAILABLE,
            "language": self.language
        }


if __name__ == "__main__":
    # Testing
    print("🎤 Testing Audio Transcriber\n")

    transcriber = AudioTranscriber(model_size="base", language="es")

    # Simulación de transcripción
    result = transcriber.transcribe_file("test_audio.wav")

    print(f"Success: {result['success']}")
    print(f"Text: {result['text']}")
    print(f"Language: {result.get('language', 'N/A')}")
    print(f"\nStats: {transcriber.get_stats()}")
