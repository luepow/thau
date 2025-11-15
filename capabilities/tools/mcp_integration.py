#!/usr/bin/env python3
"""
THAU MCP Integration - Model Context Protocol

MCP es el estándar para que modelos LLM invoquen herramientas.
Similar a cómo Claude/OpenAI invocan herramientas.

THAU ahora puede:
1. Registrar herramientas en formato MCP
2. Invocar herramientas siguiendo el protocolo
3. Ser compatible con ecosistema MCP
"""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
from pathlib import Path


class MCPToolType(Enum):
    """Tipos de herramientas MCP"""
    FUNCTION = "function"
    API = "api"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    WEB = "web"
    CUSTOM = "custom"


@dataclass
class MCPParameter:
    """Parámetro de una herramienta MCP"""
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None


@dataclass
class MCPTool:
    """
    Herramienta en formato MCP

    Estándar compatible con Claude/OpenAI function calling
    """
    name: str
    description: str
    parameters: List[MCPParameter]
    returns: Dict[str, Any]
    function: Callable
    tool_type: MCPToolType = MCPToolType.FUNCTION
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPToolCall:
    """Invocación de herramienta MCP"""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MCPToolResult:
    """Resultado de invocación MCP"""
    call_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None


class MCPRegistry:
    """
    Registro de Herramientas MCP

    Gestiona herramientas en formato Model Context Protocol
    """

    def __init__(self):
        self.tools: Dict[str, MCPTool] = {}
        print("🔌 MCP Registry inicializado")
        print("   Model Context Protocol compatible")

    def register_tool(self, tool: MCPTool):
        """
        Registra herramienta en formato MCP

        Args:
            tool: Herramienta MCP
        """
        self.tools[tool.name] = tool
        print(f"   ✅ Tool registrado: {tool.name}")

    def register_function(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: List[MCPParameter],
        returns: Dict[str, Any],
        examples: Optional[List[Dict]] = None
    ):
        """
        Registra función como herramienta MCP

        Args:
            name: Nombre de la herramienta
            description: Descripción
            function: Función callable
            parameters: Lista de parámetros
            returns: Descripción del retorno
            examples: Ejemplos de uso
        """
        tool = MCPTool(
            name=name,
            description=description,
            function=function,
            parameters=parameters,
            returns=returns,
            examples=examples or []
        )

        self.register_tool(tool)

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Obtiene herramienta por nombre"""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Lista herramientas en formato MCP

        Retorna formato compatible con OpenAI/Claude function calling
        """
        tools_list = []

        for tool in self.tools.values():
            # Formato OpenAI/Claude
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }

            # Parámetros
            for param in tool.parameters:
                tool_def["function"]["parameters"]["properties"][param.name] = {
                    "type": param.type,
                    "description": param.description
                }

                if param.enum:
                    tool_def["function"]["parameters"]["properties"][param.name]["enum"] = param.enum

                if param.required:
                    tool_def["function"]["parameters"]["required"].append(param.name)

            tools_list.append(tool_def)

        return tools_list

    def invoke_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        call_id: Optional[str] = None
    ) -> MCPToolResult:
        """
        Invoca herramienta

        Args:
            tool_name: Nombre de la herramienta
            arguments: Argumentos
            call_id: ID de la llamada

        Returns:
            Resultado MCP
        """
        import time
        import uuid

        if not call_id:
            call_id = str(uuid.uuid4())

        if tool_name not in self.tools:
            return MCPToolResult(
                call_id=call_id,
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found"
            )

        tool = self.tools[tool_name]

        print(f"\n🔧 Invocando tool: {tool_name}")
        print(f"   Arguments: {arguments}")

        # Valida parámetros requeridos
        required_params = [p.name for p in tool.parameters if p.required]
        missing = [p for p in required_params if p not in arguments]

        if missing:
            return MCPToolResult(
                call_id=call_id,
                success=False,
                result=None,
                error=f"Missing required parameters: {', '.join(missing)}"
            )

        # Ejecuta
        try:
            start_time = time.time()
            result = tool.function(**arguments)
            execution_time = (time.time() - start_time) * 1000  # ms

            print(f"   ✅ Éxito ({execution_time:.2f}ms)")

            return MCPToolResult(
                call_id=call_id,
                success=True,
                result=result,
                execution_time_ms=execution_time
            )

        except Exception as e:
            print(f"   ❌ Error: {e}")

            return MCPToolResult(
                call_id=call_id,
                success=False,
                result=None,
                error=str(e)
            )

    def export_schema(self, filepath: str = "data/mcp/tools_schema.json"):
        """
        Exporta schema de herramientas en formato MCP

        Args:
            filepath: Ruta de salida
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        schema = {
            "mcp_version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "total_tools": len(self.tools),
            "tools": self.list_tools()
        }

        with open(filepath, 'w') as f:
            json.dump(schema, f, indent=2)

        print(f"💾 Schema exportado: {filepath}")


class MCPServer:
    """
    Servidor MCP

    Permite a THAU exponer herramientas vía MCP
    Similar a cómo Claude Desktop usa MCP servers
    """

    def __init__(self, registry: MCPRegistry):
        self.registry = registry
        self.active_sessions: Dict[str, Dict] = {}

        print("🖥️  MCP Server inicializado")

    def create_session(self, session_id: str) -> Dict[str, Any]:
        """Crea sesión MCP"""
        session = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "tools": self.registry.list_tools(),
            "calls": []
        }

        self.active_sessions[session_id] = session

        print(f"📡 Sesión MCP creada: {session_id}")
        print(f"   Tools disponibles: {len(session['tools'])}")

        return session

    def handle_tool_call(
        self,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> MCPToolResult:
        """
        Maneja llamada a herramienta

        Args:
            session_id: ID de sesión
            tool_name: Nombre de herramienta
            arguments: Argumentos

        Returns:
            Resultado MCP
        """
        if session_id not in self.active_sessions:
            return MCPToolResult(
                call_id="error",
                success=False,
                result=None,
                error=f"Session '{session_id}' not found"
            )

        # Invoca tool
        result = self.registry.invoke_tool(tool_name, arguments)

        # Registra en sesión
        self.active_sessions[session_id]["calls"].append({
            "call_id": result.call_id,
            "tool": tool_name,
            "arguments": arguments,
            "success": result.success,
            "timestamp": datetime.now().isoformat()
        })

        return result


def create_default_mcp_tools() -> MCPRegistry:
    """
    Crea herramientas MCP por defecto de THAU

    Returns:
        Registry con herramientas
    """
    registry = MCPRegistry()

    # Tool 1: Generar Imagen (THAU Visual)
    def generate_image_tool(prompt: str, num_images: int = 1) -> Dict[str, Any]:
        """Genera imágenes desde THAU Visual"""
        # En producción, llamaría a thau_visual_inference
        return {
            "status": "success",
            "images_generated": num_images,
            "prompt": prompt,
            "message": "Imágenes generadas con THAU VAE"
        }

    registry.register_function(
        name="generate_image",
        description="Genera imágenes desde la imaginación de THAU usando VAE propio",
        function=generate_image_tool,
        parameters=[
            MCPParameter(
                name="prompt",
                type="string",
                description="Descripción de la imagen a generar",
                required=True
            ),
            MCPParameter(
                name="num_images",
                type="number",
                description="Número de imágenes a generar",
                required=False,
                default=1
            )
        ],
        returns={"type": "object", "description": "Resultado de generación"},
        examples=[
            {
                "prompt": "un robot aprendiendo",
                "num_images": 3
            }
        ]
    )

    # Tool 2: Crear Evento de Calendario
    def create_calendar_event(
        title: str,
        start_time: str,
        end_time: str,
        description: str = ""
    ) -> Dict[str, Any]:
        """Crea evento en calendario"""
        from datetime import datetime

        return {
            "status": "success",
            "event_id": "evt_123",
            "title": title,
            "start": start_time,
            "end": end_time,
            "message": "Evento creado exitosamente"
        }

    registry.register_function(
        name="create_calendar_event",
        description="Crea un evento en el calendario de THAU",
        function=create_calendar_event,
        parameters=[
            MCPParameter(
                name="title",
                type="string",
                description="Título del evento",
                required=True
            ),
            MCPParameter(
                name="start_time",
                type="string",
                description="Hora de inicio (ISO 8601)",
                required=True
            ),
            MCPParameter(
                name="end_time",
                type="string",
                description="Hora de fin (ISO 8601)",
                required=True
            ),
            MCPParameter(
                name="description",
                type="string",
                description="Descripción del evento",
                required=False,
                default=""
            )
        ],
        returns={"type": "object", "description": "Evento creado"}
    )

    # Tool 3: Llamar API REST
    def call_rest_api(
        url: str,
        method: str = "GET",
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Llama API REST"""
        import requests

        response = requests.request(
            method=method,
            url=url,
            json=data,
            headers=headers or {},
            timeout=30
        )

        return {
            "status_code": response.status_code,
            "success": response.status_code < 400,
            "data": response.json() if response.content else None
        }

    registry.register_function(
        name="call_rest_api",
        description="Llama a una API REST externa",
        function=call_rest_api,
        parameters=[
            MCPParameter(
                name="url",
                type="string",
                description="URL del API",
                required=True
            ),
            MCPParameter(
                name="method",
                type="string",
                description="Método HTTP",
                required=False,
                enum=["GET", "POST", "PUT", "DELETE", "PATCH"],
                default="GET"
            ),
            MCPParameter(
                name="data",
                type="object",
                description="Datos JSON para enviar",
                required=False
            ),
            MCPParameter(
                name="headers",
                type="object",
                description="Headers HTTP",
                required=False
            )
        ],
        returns={"type": "object", "description": "Respuesta del API"}
    )

    return registry


if __name__ == "__main__":
    print("="*70)
    print("🧪 Testing THAU MCP Integration")
    print("="*70)

    # Crear registry con tools por defecto
    registry = create_default_mcp_tools()

    # Test 1: Listar tools en formato MCP
    print("\n" + "="*70)
    print("Test 1: Tools en Formato MCP")
    print("="*70)

    tools = registry.list_tools()
    print(f"\n📋 {len(tools)} tools disponibles:")

    for tool in tools:
        print(f"\n{tool['function']['name']}")
        print(f"  Description: {tool['function']['description']}")
        print(f"  Parameters:")
        for param_name, param_info in tool['function']['parameters']['properties'].items():
            required = param_name in tool['function']['parameters']['required']
            print(f"    - {param_name} ({param_info['type']}){' *' if required else ''}: {param_info['description']}")

    # Test 2: Invocar herramienta
    print("\n" + "="*70)
    print("Test 2: Invocar Herramienta MCP")
    print("="*70)

    result = registry.invoke_tool(
        "generate_image",
        {"prompt": "un robot con capacidades de agente", "num_images": 5}
    )

    print(f"\n📊 Resultado:")
    print(f"  Success: {result.success}")
    print(f"  Result: {result.result}")
    print(f"  Execution time: {result.execution_time_ms:.2f}ms")

    # Test 3: Crear sesión MCP
    print("\n" + "="*70)
    print("Test 3: MCP Server Session")
    print("="*70)

    server = MCPServer(registry)
    session = server.create_session("session_123")

    # Simular tool calls
    result1 = server.handle_tool_call(
        "session_123",
        "create_calendar_event",
        {
            "title": "Reunión THAU Agent System",
            "start_time": "2025-01-16T10:00:00",
            "end_time": "2025-01-16T11:00:00",
            "description": "Revisar capacidades de agentes"
        }
    )

    print(f"\n📅 Evento creado: {result1.result}")

    # Test 4: Exportar schema
    print("\n" + "="*70)
    print("Test 4: Exportar Schema MCP")
    print("="*70)

    registry.export_schema()

    print("\n" + "="*70)
    print("✅ Tests Completados")
    print("="*70)
    print("\n💡 THAU ahora es compatible con Model Context Protocol!")
    print("   Puede interactuar con herramientas como Claude/OpenAI")
