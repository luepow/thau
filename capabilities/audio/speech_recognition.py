#!/usr/bin/env python3
"""
Reconocimiento de Voz para THAU Agent
Permite que THAU escuche y entienda comandos por micrófono

Backends soportados:
1. Whisper (OpenAI) - Local, alta calidad
2. Google Speech Recognition - Online, gratis
3. Vosk - Local, ligero
"""

import os
import wave
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime

# Intentar importar dependencias opcionales
try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False


class ThauSpeechRecognition:
    """
    Sistema de reconocimiento de voz para THAU
    Convierte audio del micrófono a texto
    """

    def __init__(
        self,
        backend: str = "google",  # "whisper", "google", "vosk"
        whisper_model: str = "base",  # tiny, base, small, medium, large
        language: str = "es",  # Idioma principal
        energy_threshold: int = 300,  # Sensibilidad del micrófono
        pause_threshold: float = 0.8,  # Segundos de silencio para terminar
    ):
        self.backend = backend
        self.whisper_model = whisper_model
        self.language = language
        self.energy_threshold = energy_threshold
        self.pause_threshold = pause_threshold

        self.recognizer = None
        self.whisper = None
        self.microphone = None

        self._setup()

    def _setup(self):
        """Configura el sistema de reconocimiento"""
        if not HAS_SR:
            print("⚠️  speech_recognition no instalado. Instala con:")
            print("   pip install SpeechRecognition")
            return

        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = self.energy_threshold
        self.recognizer.pause_threshold = self.pause_threshold

        # Configurar Whisper si se usa
        if self.backend == "whisper":
            if HAS_WHISPER:
                print(f"🎤 Cargando modelo Whisper '{self.whisper_model}'...")
                self.whisper = whisper.load_model(self.whisper_model)
                print("   ✅ Whisper listo")
            else:
                print("⚠️  Whisper no instalado. Usando Google Speech.")
                print("   Instala con: pip install openai-whisper")
                self.backend = "google"

        # Verificar micrófono
        if HAS_PYAUDIO:
            try:
                self.microphone = sr.Microphone()
                print("🎤 Micrófono detectado")
            except Exception as e:
                print(f"⚠️  Error con micrófono: {e}")
        else:
            print("⚠️  PyAudio no instalado. Instala con:")
            print("   brew install portaudio && pip install pyaudio")

    def list_microphones(self) -> list:
        """Lista los micrófonos disponibles"""
        if not HAS_SR:
            return []

        mics = sr.Microphone.list_microphone_names()
        return mics

    def calibrate(self, duration: float = 2.0):
        """Calibra el umbral de ruido ambiente"""
        if not self.recognizer or not self.microphone:
            return {"success": False, "error": "Sistema no inicializado"}

        print(f"🔇 Calibrando ruido ambiente ({duration}s)... No hables.")

        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=duration)

            print(f"   ✅ Umbral ajustado a: {self.recognizer.energy_threshold}")
            return {
                "success": True,
                "energy_threshold": self.recognizer.energy_threshold
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def listen_once(self, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Escucha una vez y devuelve el texto reconocido

        Args:
            timeout: Máximo tiempo de espera en segundos

        Returns:
            Dict con texto reconocido o error
        """
        if not self.recognizer or not self.microphone:
            return {"success": False, "error": "Sistema no inicializado"}

        print("🎤 Escuchando...")

        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout)

            return self._transcribe(audio)

        except sr.WaitTimeoutError:
            return {"success": False, "error": "Timeout - no se detectó audio"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _transcribe(self, audio) -> Dict[str, Any]:
        """Transcribe audio usando el backend configurado"""
        start_time = datetime.now()

        try:
            if self.backend == "whisper" and self.whisper:
                # Guardar audio temporal
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio.get_wav_data())
                    temp_path = f.name

                # Transcribir con Whisper
                result = self.whisper.transcribe(
                    temp_path,
                    language=self.language,
                    fp16=False
                )
                text = result["text"].strip()

                # Limpiar
                os.unlink(temp_path)

            elif self.backend == "google":
                # Google Speech Recognition (gratis, online)
                text = self.recognizer.recognize_google(
                    audio,
                    language=f"{self.language}-{self.language.upper()}"
                )

            else:
                return {"success": False, "error": f"Backend '{self.backend}' no soportado"}

            elapsed = (datetime.now() - start_time).total_seconds()

            return {
                "success": True,
                "text": text,
                "backend": self.backend,
                "language": self.language,
                "processing_time": elapsed
            }

        except sr.UnknownValueError:
            return {"success": False, "error": "No se pudo entender el audio"}
        except sr.RequestError as e:
            return {"success": False, "error": f"Error de servicio: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def listen_continuous(
        self,
        callback: Callable[[str], None],
        stop_word: str = "detener",
        max_iterations: int = 100
    ):
        """
        Escucha continuamente y llama callback con cada frase

        Args:
            callback: Función a llamar con el texto reconocido
            stop_word: Palabra para detener la escucha
            max_iterations: Máximo número de iteraciones
        """
        if not self.recognizer or not self.microphone:
            print("❌ Sistema no inicializado")
            return

        print("\n🎤 Escucha continua activada")
        print(f"   Di '{stop_word}' para terminar")
        print("=" * 40)

        for i in range(max_iterations):
            result = self.listen_once()

            if result["success"]:
                text = result["text"]
                print(f"\n👤 Escuché: {text}")

                # Verificar palabra de parada
                if stop_word.lower() in text.lower():
                    print("\n🛑 Escucha detenida")
                    break

                # Llamar callback
                callback(text)
            else:
                if "Timeout" not in result.get("error", ""):
                    print(f"   ⚠️ {result['error']}")

        print("\n✅ Sesión de escucha finalizada")

    def get_status(self) -> Dict[str, Any]:
        """Devuelve el estado del sistema"""
        return {
            "initialized": self.recognizer is not None,
            "backend": self.backend,
            "language": self.language,
            "has_microphone": self.microphone is not None,
            "has_whisper": self.whisper is not None,
            "energy_threshold": self.recognizer.energy_threshold if self.recognizer else None,
            "dependencies": {
                "speech_recognition": HAS_SR,
                "whisper": HAS_WHISPER,
                "pyaudio": HAS_PYAUDIO
            }
        }


def create_speech_tool(speech: ThauSpeechRecognition):
    """Crea herramienta de escucha para el agente THAU"""

    def listen_to_user() -> Dict[str, Any]:
        """
        Escucha al usuario por el micrófono y devuelve el texto.

        Returns:
            Dict con el texto reconocido
        """
        result = speech.listen_once(timeout=15)
        return result

    return listen_to_user


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  THAU Speech Recognition - Test")
    print("=" * 60)

    # Verificar dependencias
    print("\n1. Verificando dependencias...")
    deps = {
        "speech_recognition": HAS_SR,
        "whisper": HAS_WHISPER,
        "pyaudio": HAS_PYAUDIO
    }

    for dep, installed in deps.items():
        status = "✅" if installed else "❌"
        print(f"   {status} {dep}")

    if not HAS_SR:
        print("\n⚠️  Instala dependencias con:")
        print("   pip install SpeechRecognition")
        print("   brew install portaudio && pip install pyaudio")
        print("   pip install openai-whisper  # Opcional, para Whisper local")
        exit(1)

    # Inicializar
    print("\n2. Inicializando reconocimiento de voz...")
    speech = ThauSpeechRecognition(
        backend="google",  # Cambia a "whisper" si tienes GPU
        language="es"
    )

    # Listar micrófonos
    print("\n3. Micrófonos disponibles:")
    mics = speech.list_microphones()
    for i, mic in enumerate(mics[:5]):
        print(f"   [{i}] {mic}")

    # Calibrar
    print("\n4. Calibrando...")
    speech.calibrate(duration=2)

    # Test de escucha
    print("\n5. Test de escucha (di algo)...")
    result = speech.listen_once(timeout=10)

    if result["success"]:
        print(f"\n   ✅ Texto reconocido: '{result['text']}'")
        print(f"   Backend: {result['backend']}")
        print(f"   Tiempo: {result['processing_time']:.2f}s")
    else:
        print(f"\n   ❌ Error: {result['error']}")

    print("\n" + "=" * 60)
