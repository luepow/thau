"""
Tool Registry for THAU
Manages available tools/capabilities that THAU can use
"""

import re
from typing import Dict, Callable, List, Optional
from dataclasses import dataclass
from enum import Enum


class ToolType(Enum):
    """Types of tools available to THAU"""
    IMAGE_GENERATION = "image_generation"
    CODE_EXECUTION = "code_execution"
    WEB_SEARCH = "web_search"
    CALCULATION = "calculation"


@dataclass
class Tool:
    """Represents a tool that THAU can use"""
    name: str
    tool_type: ToolType
    description: str
    function: Callable
    trigger_patterns: List[str]  # Regex patterns that trigger this tool
    examples: List[str]


class ToolRegistry:
    """
    Registry of all tools available to THAU

    Analiza las peticiones del usuario y decide qué herramienta usar
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        print("🔧 Tool Registry inicializado")

    def register_tool(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.name] = tool
        print(f"   ✅ Tool registrado: {tool.name}")

    def detect_tool_needed(self, user_input: str) -> Optional[Tool]:
        """
        Detect if user input requires a specific tool

        Args:
            user_input: User's message

        Returns:
            Tool to use or None
        """
        user_input_lower = user_input.lower()

        # Check each tool's trigger patterns
        for tool in self.tools.values():
            for pattern in tool.trigger_patterns:
                if re.search(pattern, user_input_lower, re.IGNORECASE):
                    print(f"🎯 Tool detectado: {tool.name}")
                    return tool

        return None

    def extract_parameters(self, user_input: str, tool: Tool) -> Dict:
        """
        Extract parameters from user input for a specific tool

        Args:
            user_input: User's message
            tool: Tool to extract parameters for

        Returns:
            Dictionary of parameters
        """
        params = {}

        if tool.tool_type == ToolType.IMAGE_GENERATION:
            # Extract image description
            # Remove trigger phrases
            description = user_input
            for pattern in tool.trigger_patterns:
                description = re.sub(pattern, "", description, flags=re.IGNORECASE)

            # Clean up
            description = description.strip()

            # Remove common prefixes
            prefixes = ["una imagen de", "un dibujo de", "una foto de", "de", "sobre"]
            for prefix in prefixes:
                if description.lower().startswith(prefix):
                    description = description[len(prefix):].strip()

            params['prompt'] = description or "a beautiful scene"

        return params

    def list_tools(self) -> List[str]:
        """Get list of available tools"""
        return [f"{name} ({tool.tool_type.value})" for name, tool in self.tools.items()]

    def get_tool_help(self, tool_name: str) -> str:
        """Get help text for a specific tool"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        tool = self.tools[tool_name]

        help_text = f"""
🔧 **{tool.name}**

**Descripción**: {tool.description}

**Tipo**: {tool.tool_type.value}

**Cómo usarlo**:
{chr(10).join(f"  - {example}" for example in tool.examples)}

**Patrones que lo activan**:
{chr(10).join(f"  - {pattern}" for pattern in tool.trigger_patterns)}
"""
        return help_text.strip()


# Global registry instance
_global_registry = None


def get_tool_registry() -> ToolRegistry:
    """Get global tool registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
        _register_default_tools()
    return _global_registry


def _register_default_tools():
    """Register default tools on first access"""
    registry = _global_registry

    # Image Generation Tool
    from capabilities.vision.image_generator import ThauImageGenerator

    image_gen = ThauImageGenerator()

    def generate_image_tool(**kwargs):
        return image_gen.generate_image(**kwargs)

    image_tool = Tool(
        name="generate_image",
        tool_type=ToolType.IMAGE_GENERATION,
        description="Genera imágenes a partir de descripciones en texto usando IA",
        function=generate_image_tool,
        trigger_patterns=[
            r"generar?\s+(una\s+)?imagen",
            r"crear?\s+(una\s+)?imagen",
            r"dibuj(a|ar)\s+(me|una)?",
            r"muestra(me)?\s+(una\s+)?imagen",
            r"haz\s+(me\s+)?(una\s+)?imagen",
            r"imagen\s+de",
        ],
        examples=[
            "Genera una imagen de un gato espacial",
            "Crea una imagen de un paisaje futurista",
            "Dibuja un robot aprendiendo a programar",
            "Muéstrame una imagen de un atardecer en las montañas",
        ]
    )

    registry.register_tool(image_tool)


# Testing
if __name__ == "__main__":
    print("="*70)
    print("🧪 Testing Tool Registry")
    print("="*70)

    registry = get_tool_registry()

    # Test inputs
    test_inputs = [
        "Genera una imagen de un gato espacial",
        "¿Qué es Python?",
        "Crea una imagen de un paisaje futurista",
        "Dibuja un robot",
        "Hola, ¿cómo estás?",
        "Muéstrame una imagen de un atardecer",
    ]

    print("\n🧪 Testing tool detection:\n")

    for user_input in test_inputs:
        print(f"Input: '{user_input}'")

        tool = registry.detect_tool_needed(user_input)

        if tool:
            print(f"  → Tool: {tool.name}")

            params = registry.extract_parameters(user_input, tool)
            print(f"  → Params: {params}")
        else:
            print(f"  → No tool needed (normal conversation)")

        print()

    # List all tools
    print("\n📋 Available tools:")
    for tool_name in registry.list_tools():
        print(f"  - {tool_name}")

    print("\n" + "="*70)
