#!/usr/bin/env python3
"""
THAU Text-to-Speech Service
Servicio de síntesis de voz para THAU usando pyttsx3 (offline) y edge-tts (online)
"""

import os
import re
import hashlib
import base64
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import pyttsx3

# Configuración
AUDIO_OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "audio"
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Voces predeterminadas para THAU (macOS)
SPANISH_VOICES = {
    # Voces masculinas en español
    "eddy_mx": "com.apple.eloquence.es-MX.Eddy",
    "eddy_es": "com.apple.eloquence.es-ES.Eddy",
    "rocko_mx": "com.apple.eloquence.es-MX.Rocko",
    "rocko_es": "com.apple.eloquence.es-ES.Rocko",
    "reed_mx": "com.apple.eloquence.es-MX.Reed",
    "grandpa_mx": "com.apple.eloquence.es-MX.Grandpa",

    # Voces femeninas en español
    "paulina": "com.apple.voice.compact.es-MX.Paulina",
    "monica": "com.apple.voice.compact.es-ES.Monica",
    "sandy_mx": "com.apple.eloquence.es-MX.Sandy",
    "sandy_es": "com.apple.eloquence.es-ES.Sandy",
    "shelley_mx": "com.apple.eloquence.es-MX.Shelley",
    "flo_mx": "com.apple.eloquence.es-MX.Flo",
}

# Voz predeterminada para THAU - voz masculina con buen sonido
DEFAULT_VOICE = "com.apple.eloquence.es-MX.Eddy"


class ThauVoice:
    """Servicio de voz para THAU usando pyttsx3 (offline)"""

    def __init__(self, voice: str = None, rate: int = 180, volume: float = 1.0):
        """
        Inicializa el servicio de voz

        Args:
            voice: Identificador de la voz (nombre corto o ID completo)
            rate: Velocidad de habla (palabras por minuto, default 180)
            volume: Volumen (0.0 a 1.0)
        """
        self.engine = pyttsx3.init()

        # Resolver nombre corto a ID completo
        voice_id = SPANISH_VOICES.get(voice, voice) if voice else DEFAULT_VOICE
        self.voice = self._find_voice(voice_id)
        self.rate = rate
        self.volume = volume
        self.cache_enabled = True

        # Configurar motor
        if self.voice:
            self.engine.setProperty('voice', self.voice)
        self.engine.setProperty('rate', self.rate)
        self.engine.setProperty('volume', self.volume)

    def _find_voice(self, voice_id: str) -> str:
        """Encuentra una voz que coincida con el ID dado"""
        voices = self.engine.getProperty('voices')

        # Buscar coincidencia exacta
        for v in voices:
            if v.id == voice_id:
                return v.id

        # Buscar voz en español como fallback
        for v in voices:
            if 'es-ES' in v.id or 'es-MX' in v.id:
                return v.id

        # Usar primera voz disponible
        return voices[0].id if voices else None

    def _clean_text(self, text: str) -> str:
        """Limpia el texto para síntesis de voz"""
        # Remover bloques de código
        text = re.sub(r'```[\s\S]*?```', 'Código omitido.', text)
        text = re.sub(r'`[^`]+`', '', text)

        # Remover markdown
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
        text = re.sub(r'#{1,6}\s*', '', text)           # Headers
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links

        # Remover caracteres especiales
        text = re.sub(r'[<>{}|\[\]]', '', text)

        # Limpiar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _get_cache_path(self, text: str) -> Path:
        """Genera ruta de caché basada en hash del texto"""
        text_hash = hashlib.md5(f"{text}_{self.voice}_{self.rate}".encode()).hexdigest()
        return AUDIO_OUTPUT_DIR / f"cache_{text_hash}.wav"

    def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        """
        Sintetiza texto a audio

        Args:
            text: Texto a convertir en voz
            output_path: Ruta de salida (opcional, se genera automáticamente)

        Returns:
            Ruta del archivo de audio generado
        """
        # Limpiar texto
        clean_text = self._clean_text(text)

        if not clean_text:
            raise ValueError("No hay texto válido para sintetizar")

        # Verificar caché
        if self.cache_enabled:
            cache_path = self._get_cache_path(clean_text)
            if cache_path.exists():
                return str(cache_path)

        # Determinar ruta de salida
        if output_path:
            audio_path = Path(output_path)
        elif self.cache_enabled:
            audio_path = cache_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_path = AUDIO_OUTPUT_DIR / f"thau_voice_{timestamp}.wav"

        # Generar audio
        self.engine.save_to_file(clean_text, str(audio_path))
        self.engine.runAndWait()

        return str(audio_path)

    def synthesize_to_base64(self, text: str) -> str:
        """
        Sintetiza texto a audio y retorna como base64

        Args:
            text: Texto a convertir

        Returns:
            String base64 del audio WAV
        """
        audio_path = self.synthesize(text)
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        return base64.b64encode(audio_data).decode('utf-8')

    def speak(self, text: str):
        """
        Reproduce texto en voz alta inmediatamente

        Args:
            text: Texto a hablar
        """
        clean_text = self._clean_text(text)
        if clean_text:
            self.engine.say(clean_text)
            self.engine.runAndWait()

    @staticmethod
    def list_voices(filter_language: str = "es") -> List[Dict]:
        """
        Lista las voces disponibles

        Args:
            filter_language: Código de idioma para filtrar (ej: "es", "en")

        Returns:
            Lista de voces disponibles
        """
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        result = []

        for v in voices:
            voice_info = {
                "id": v.id,
                "name": v.name,
                "languages": getattr(v, 'languages', []),
            }

            # Filtrar por idioma si se especifica
            if filter_language:
                if filter_language.lower() in v.id.lower() or filter_language.lower() in v.name.lower():
                    result.append(voice_info)
            else:
                result.append(voice_info)

        return result

    def set_voice(self, voice: str):
        """Cambia la voz actual"""
        voice_id = SPANISH_VOICES.get(voice, voice)
        self.voice = self._find_voice(voice_id)
        if self.voice:
            self.engine.setProperty('voice', self.voice)

    def set_rate(self, rate: int):
        """Cambia la velocidad de habla (palabras por minuto)"""
        self.rate = rate
        self.engine.setProperty('rate', rate)

    def set_volume(self, volume: float):
        """Cambia el volumen (0.0 a 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        self.engine.setProperty('volume', self.volume)


# Instancia global para uso rápido
thau_voice = ThauVoice()


def speak(text: str, voice: str = None) -> str:
    """
    Función de conveniencia para sintetizar voz

    Args:
        text: Texto a hablar
        voice: Voz a usar (opcional)

    Returns:
        Ruta del archivo de audio
    """
    tts = ThauVoice(voice) if voice else thau_voice
    return tts.synthesize(text)


def speak_aloud(text: str, voice: str = None):
    """
    Reproduce texto en voz alta inmediatamente

    Args:
        text: Texto a hablar
        voice: Voz a usar (opcional)
    """
    tts = ThauVoice(voice) if voice else thau_voice
    tts.speak(text)


# Test
if __name__ == "__main__":
    print("🎤 THAU Voice Service - Test\n")

    # Listar voces disponibles
    print("📋 Voces en español disponibles:")
    voices = ThauVoice.list_voices("es")
    for v in voices[:10]:
        print(f"   - {v['name']}: {v['id']}")

    print(f"\n🗣️ Voz predeterminada: {DEFAULT_VOICE}")

    # Sintetizar un mensaje de prueba
    tts = ThauVoice()
    test_text = """
    Hola, soy THAU, tu asistente de inteligencia artificial.
    Estoy aquí para ayudarte con programación, análisis y mucho más.
    ¿En qué puedo ayudarte hoy?
    """

    print(f"\n🔊 Sintetizando mensaje de prueba...")
    audio_path = tts.synthesize(test_text)
    print(f"   ✅ Audio guardado en: {audio_path}")

    # Probar diferentes voces
    print("\n🎭 Probando diferentes voces:")
    for name in ["eddy_mx", "paulina", "rocko_mx"]:
        voice_id = SPANISH_VOICES.get(name)
        if voice_id:
            tts = ThauVoice(voice=name, rate=190)
            path = tts.synthesize(f"Hola, soy la voz {name}")
            print(f"   - {name}: {path}")

    print("\n✅ Test completado!")
