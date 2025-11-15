"""Self-reflection and meta-cognitive capabilities."""

from typing import Dict
from loguru import logger


class Reflector:
    """Self-reflection module for improving outputs."""

    def __init__(self, reasoning_engine=None):
        """Initialize reflector."""
        from reasoning.chain_of_thought import ReasoningEngine
        self.reasoning_engine = reasoning_engine or ReasoningEngine()

        logger.info("Reflector initialized")

    def reflect_on_response(self, response: str, criteria: str = "quality") -> Dict:
        """Reflect on a response.

        Args:
            response: Response to reflect on
            criteria: Reflection criteria

        Returns:
            Reflection results
        """
        reflection = self.reasoning_engine.reflect(f"Response: {response}\nCriteria: {criteria}")

        return {
            "original_response": response,
            "reflection": reflection,
            "criteria": criteria,
        }
