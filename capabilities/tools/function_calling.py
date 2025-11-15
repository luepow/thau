"""Function calling capabilities."""

from typing import List, Dict, Callable, Any
from loguru import logger
import json


class FunctionRegistry:
    """Registry for callable functions."""

    def __init__(self):
        """Initialize function registry."""
        self.functions: Dict[str, Callable] = {}
        self.schemas: Dict[str, Dict] = {}

        logger.info("FunctionRegistry initialized")

    def register(
        self,
        name: str,
        func: Callable,
        schema: Dict[str, Any],
    ) -> None:
        """Register a function.

        Args:
            name: Function name
            func: Callable function
            schema: Function schema (OpenAI format)
        """
        self.functions[name] = func
        self.schemas[name] = schema

        logger.info(f"Registered function: {name}")

    def call(self, name: str, **kwargs) -> Any:
        """Call a registered function.

        Args:
            name: Function name
            **kwargs: Function arguments

        Returns:
            Function result
        """
        if name not in self.functions:
            raise ValueError(f"Function not found: {name}")

        return self.functions[name](**kwargs)

    def get_schemas(self) -> List[Dict]:
        """Get all function schemas.

        Returns:
            List of schemas
        """
        return list(self.schemas.values())
