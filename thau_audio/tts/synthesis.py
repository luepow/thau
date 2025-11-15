"""
Text-to-Speech Synthesis
Converts text to natural speech audio
"""

from typing import Dict, Optional
import os
from datetime import datetime
import numpy as np

# Try to import TTS libraries
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("ℹ️  gTTS not available. Install with: pip install gtts")

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("ℹ️  pyttsx3 not available. Install with: pip install pyttsx3")


class TextToSpeech:
    """
    Sistema de Text-to-Speech para THAU

    Convierte texto a audio usando diferentes backends disponibles
    """

    def __init__(self, backend: str = "auto", language: str = "es", output_dir: str = "./audio_output"):
        """
        Inicializa el sistema TTS

        Args:
            backend: Backend a usar (auto, gtts, pyttsx3, simulated)
            language: Idioma del speech (es, en, etc.)
            output_dir: Directorio para guardar archivos de audio
        """
        self.language = language
        self.output_dir = output_dir
        self.synthesis_history = []

        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)

        # Determinar backend disponible
        if backend == "auto":
            if GTTS_AVAILABLE:
                self.backend = "gtts"
            elif PYTTSX3_AVAILABLE:
                self.backend = "pyttsx3"
            else:
                self.backend = "simulated"
        else:
            self.backend = backend

        # Inicializar backend
        self.tts_engine = None
        if self.backend == "pyttsx3" and PYTTSX3_AVAILABLE:
            self._init_pyttsx3()

        print(f"🔊 TTS inicializado: backend={self.backend}, idioma={language}")

    def _init_pyttsx3(self):
        """Inicializa el engine pyttsx3"""
        try:
            self.tts_engine = pyttsx3.init()

            # Configurar idioma y velocidad
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if self.language in voice.languages:
                    self.tts_engine.setProperty('voice', voice.id)
                    break

            self.tts_engine.setProperty('rate', 150)  # Velocidad
            self.tts_engine.setProperty('volume', 0.9)  # Volumen

        except Exception as e:
            print(f"⚠️  Error inicializando pyttsx3: {e}")
            self.tts_engine = None

    def synthesize(self, text: str, output_path: Optional[str] = None, **kwargs) -> Dict:
        """
        Sintetiza texto a audio

        Args:
            text: Texto a convertir
            output_path: Ruta donde guardar el audio (opcional)
            **kwargs: Argumentos adicionales específicos del backend

        Returns:
            Dict con información del audio generado
        """
        if not text or len(text.strip()) == 0:
            return {
                "success": False,
                "error": "Texto vacío"
            }

        # Generar nombre de archivo si no se proporciona
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"tts_{timestamp}.mp3")

        # Sintetizar según backend
        if self.backend == "gtts":
            return self._synthesize_gtts(text, output_path, **kwargs)
        elif self.backend == "pyttsx3":
            return self._synthesize_pyttsx3(text, output_path, **kwargs)
        else:
            return self._synthesize_simulated(text, output_path)

    def _synthesize_gtts(self, text: str, output_path: str, **kwargs) -> Dict:
        """Síntesis con gTTS (Google TTS)"""
        if not GTTS_AVAILABLE:
            return self._synthesize_simulated(text, output_path)

        try:
            tts = gTTS(text=text, lang=self.language, slow=False)
            tts.save(output_path)

            result = {
                "success": True,
                "text": text,
                "output_path": output_path,
                "backend": "gtts",
                "language": self.language,
                "timestamp": datetime.now().isoformat(),
                "file_size": os.path.getsize(output_path) if os.path.exists(output_path) else 0
            }

            self.synthesis_history.append(result)
            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text": text
            }

    def _synthesize_pyttsx3(self, text: str, output_path: str, **kwargs) -> Dict:
        """Síntesis con pyttsx3 (offline)"""
        if not PYTTSX3_AVAILABLE or self.tts_engine is None:
            return self._synthesize_simulated(text, output_path)

        try:
            # Cambiar extensión a .wav
            if output_path.endswith('.mp3'):
                output_path = output_path.replace('.mp3', '.wav')

            self.tts_engine.save_to_file(text, output_path)
            self.tts_engine.runAndWait()

            result = {
                "success": True,
                "text": text,
                "output_path": output_path,
                "backend": "pyttsx3",
                "language": self.language,
                "timestamp": datetime.now().isoformat(),
                "file_size": os.path.getsize(output_path) if os.path.exists(output_path) else 0
            }

            self.synthesis_history.append(result)
            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text": text
            }

    def _synthesize_simulated(self, text: str, output_path: str) -> Dict:
        """Síntesis simulada (para testing sin dependencias)"""
        # Crear archivo vacío simulado
        with open(output_path, 'wb') as f:
            # Escribir cabecera MP3 simulada
            f.write(b'\xff\xfb\x90\x00' * 100)

        return {
            "success": True,
            "text": text,
            "output_path": output_path,
            "backend": "simulated",
            "language": self.language,
            "timestamp": datetime.now().isoformat(),
            "file_size": 400,
            "simulated": True
        }

    def synthesize_to_array(self, text: str, **kwargs) -> Dict:
        """
        Sintetiza texto y devuelve como array numpy

        Args:
            text: Texto a convertir
            **kwargs: Argumentos adicionales

        Returns:
            Dict con array de audio y metadata
        """
        # Por ahora, sintetizamos a archivo y luego lo leemos
        temp_path = os.path.join(self.output_dir, "temp_tts.mp3")
        result = self.synthesize(text, temp_path, **kwargs)

        if result["success"] and os.path.exists(temp_path):
            try:
                import librosa
                audio_array, sr = librosa.load(temp_path, sr=None)

                result["audio_array"] = audio_array
                result["sample_rate"] = sr

                # Limpiar archivo temporal
                os.remove(temp_path)

            except:
                result["audio_array"] = np.zeros(16000)  # 1 segundo de silencio
                result["sample_rate"] = 16000

        return result

    def get_stats(self) -> Dict:
        """Obtiene estadísticas del sistema TTS"""
        return {
            "total_syntheses": len(self.synthesis_history),
            "backend": self.backend,
            "language": self.language,
            "gtts_available": GTTS_AVAILABLE,
            "pyttsx3_available": PYTTSX3_AVAILABLE
        }


if __name__ == "__main__":
    # Testing
    print("🔊 Testing Text-to-Speech\n")

    tts = TextToSpeech(backend="auto", language="es")

    result = tts.synthesize("Hola, soy THAU. Un sistema de inteligencia artificial.")

    print(f"Success: {result['success']}")
    print(f"Backend: {result.get('backend', 'N/A')}")
    print(f"Output: {result.get('output_path', 'N/A')}")
    print(f"\nStats: {tts.get_stats()}")
