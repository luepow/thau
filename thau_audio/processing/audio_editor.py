"""
Audio Processor
Herramientas para procesamiento y modificación de audio
"""

from typing import Dict, Optional, Tuple
import numpy as np
from datetime import datetime

try:
    import librosa
    import soundfile as sf
    from scipy import signal
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    print("ℹ️  Audio processing libraries not available")


class AudioProcessor:
    """
    Procesador de audio para THAU

    Permite modificar, filtrar y analizar audio
    """

    def __init__(self):
        """Inicializa el procesador de audio"""
        self.processing_history = []
        print(f"🎛️  AudioProcessor inicializado (libs available: {AUDIO_PROCESSING_AVAILABLE})")

    def load_audio(self, path: str, sample_rate: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Carga un archivo de audio

        Args:
            path: Ruta al archivo
            sample_rate: Frecuencia de muestreo deseada

        Returns:
            Tupla (audio_array, sample_rate)
        """
        if not AUDIO_PROCESSING_AVAILABLE:
            return np.zeros(16000), 16000

        try:
            audio, sr = librosa.load(path, sr=sample_rate)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return np.zeros(16000), sample_rate or 16000

    def save_audio(self, audio: np.ndarray, path: str, sample_rate: int = 16000) -> Dict:
        """
        Guarda audio a archivo

        Args:
            audio: Array de audio
            path: Ruta de salida
            sample_rate: Frecuencia de muestreo

        Returns:
            Dict con resultado
        """
        if not AUDIO_PROCESSING_AVAILABLE:
            return {"success": False, "error": "Libraries not available"}

        try:
            sf.write(path, audio, sample_rate)
            return {
                "success": True,
                "path": path,
                "duration": len(audio) / sample_rate,
                "sample_rate": sample_rate
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def change_pitch(self, audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
        """
        Cambia el pitch del audio

        Args:
            audio: Audio input
            sample_rate: Frecuencia de muestreo
            semitones: Semitonos a cambiar (+/-)

        Returns:
            Audio modificado
        """
        if not AUDIO_PROCESSING_AVAILABLE:
            return audio

        try:
            return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones)
        except Exception as e:
            print(f"Error changing pitch: {e}")
            return audio

    def change_speed(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """
        Cambia la velocidad del audio

        Args:
            audio: Audio input
            rate: Factor de velocidad (1.0 = normal, 2.0 = doble velocidad)

        Returns:
            Audio modificado
        """
        if not AUDIO_PROCESSING_AVAILABLE:
            return audio

        try:
            return librosa.effects.time_stretch(audio, rate=rate)
        except Exception as e:
            print(f"Error changing speed: {e}")
            return audio

    def add_reverb(self, audio: np.ndarray, sample_rate: int, room_scale: float = 0.5) -> np.ndarray:
        """
        Añade reverberación al audio

        Args:
            audio: Audio input
            sample_rate: Frecuencia de muestreo
            room_scale: Escala del espacio (0.0-1.0)

        Returns:
            Audio con reverb
        """
        if not AUDIO_PROCESSING_AVAILABLE:
            return audio

        try:
            # Implementación simplificada de reverb usando convolución
            impulse_length = int(sample_rate * room_scale)
            impulse = np.random.randn(impulse_length) * np.exp(-np.linspace(0, 5, impulse_length))
            impulse = impulse / np.max(np.abs(impulse))

            reverbed = signal.convolve(audio, impulse, mode='same')
            reverbed = reverbed / np.max(np.abs(reverbed))

            # Mix con audio original
            mix = 0.7 * audio + 0.3 * reverbed
            return mix / np.max(np.abs(mix))

        except Exception as e:
            print(f"Error adding reverb: {e}")
            return audio

    def normalize_audio(self, audio: np.ndarray, target_level: float = 0.9) -> np.ndarray:
        """
        Normaliza el nivel de audio

        Args:
            audio: Audio input
            target_level: Nivel objetivo (0.0-1.0)

        Returns:
            Audio normalizado
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio * (target_level / max_val)
        return audio

    def trim_silence(self, audio: np.ndarray, sample_rate: int, threshold_db: float = -40) -> np.ndarray:
        """
        Elimina silencios al inicio y final

        Args:
            audio: Audio input
            sample_rate: Frecuencia de muestreo
            threshold_db: Umbral de silencio en dB

        Returns:
            Audio recortado
        """
        if not AUDIO_PROCESSING_AVAILABLE:
            return audio

        try:
            trimmed, _ = librosa.effects.trim(audio, top_db=-threshold_db)
            return trimmed
        except Exception as e:
            print(f"Error trimming silence: {e}")
            return audio

    def extract_features(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """
        Extrae características del audio

        Args:
            audio: Audio input
            sample_rate: Frecuencia de muestreo

        Returns:
            Dict con características
        """
        if not AUDIO_PROCESSING_AVAILABLE:
            return {"duration": len(audio) / sample_rate}

        try:
            # Características básicas
            duration = len(audio) / sample_rate
            rms = np.sqrt(np.mean(audio**2))

            # Características espectrales
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
            zero_crossings = librosa.zero_crossings(audio, pad=False)

            # MFCC
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)

            return {
                "duration": duration,
                "rms_energy": float(rms),
                "spectral_centroid_mean": float(np.mean(spectral_centroids)),
                "spectral_centroid_std": float(np.std(spectral_centroids)),
                "zero_crossing_rate": float(np.sum(zero_crossings) / len(audio)),
                "mfcc_mean": mfccs.mean(axis=1).tolist(),
                "mfcc_std": mfccs.std(axis=1).tolist()
            }

        except Exception as e:
            print(f"Error extracting features: {e}")
            return {"duration": len(audio) / sample_rate, "error": str(e)}

    def process_audio(self, audio: np.ndarray, sample_rate: int, operations: Dict) -> Dict:
        """
        Procesa audio con múltiples operaciones

        Args:
            audio: Audio input
            sample_rate: Frecuencia de muestreo
            operations: Dict de operaciones a aplicar

        Returns:
            Dict con audio procesado y metadata
        """
        processed = audio.copy()
        applied_operations = []

        try:
            if operations.get("normalize"):
                processed = self.normalize_audio(processed)
                applied_operations.append("normalize")

            if operations.get("trim_silence"):
                processed = self.trim_silence(processed, sample_rate)
                applied_operations.append("trim_silence")

            if "pitch_shift" in operations:
                processed = self.change_pitch(processed, sample_rate, operations["pitch_shift"])
                applied_operations.append(f"pitch_shift_{operations['pitch_shift']}")

            if "speed_change" in operations:
                processed = self.change_speed(processed, operations["speed_change"])
                applied_operations.append(f"speed_change_{operations['speed_change']}")

            if "reverb" in operations:
                processed = self.add_reverb(processed, sample_rate, operations.get("reverb", 0.5))
                applied_operations.append("reverb")

            result = {
                "success": True,
                "audio": processed,
                "sample_rate": sample_rate,
                "operations_applied": applied_operations,
                "duration_original": len(audio) / sample_rate,
                "duration_processed": len(processed) / sample_rate,
                "timestamp": datetime.now().isoformat()
            }

            self.processing_history.append({
                "operations": applied_operations,
                "timestamp": result["timestamp"]
            })

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "operations_attempted": applied_operations
            }

    def get_stats(self) -> Dict:
        """Obtiene estadísticas del procesador"""
        return {
            "total_processings": len(self.processing_history),
            "libraries_available": AUDIO_PROCESSING_AVAILABLE
        }


if __name__ == "__main__":
    # Testing
    print("🎛️  Testing Audio Processor\n")

    processor = AudioProcessor()

    # Generar audio de prueba
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # A440 Hz

    # Procesar
    result = processor.process_audio(
        audio,
        sample_rate,
        {
            "normalize": True,
            "pitch_shift": 2,
            "speed_change": 1.2
        }
    )

    print(f"Success: {result['success']}")
    print(f"Operations: {result.get('operations_applied', [])}")
    print(f"\nStats: {processor.get_stats()}")
