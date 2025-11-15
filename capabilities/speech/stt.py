"""Speech-to-Text placeholder (to be implemented with STT library)."""

from loguru import logger


class SpeechToText:
    """Speech-to-Text interface (placeholder for future implementation)."""

    def __init__(self):
        """Initialize STT."""
        logger.info("SpeechToText initialized (placeholder)")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio to text.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text

        Note:
            This is a placeholder. Implement with an STT library like:
            - OpenAI Whisper
            - Google Speech-to-Text
            - AssemblyAI
        """
        logger.warning("STT not implemented yet")
        return ""
