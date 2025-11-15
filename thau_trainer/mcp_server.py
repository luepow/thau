"""
Servidor MCP (Model Context Protocol) para THAU
Permite que THAU se comunique con herramientas externas usando el estándar de Anthropic
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
import inspect


class MCPTool:
    """Representa una herramienta MCP"""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function: Callable
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function

    def to_schema(self) -> Dict:
        """Convierte a schema MCP"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.parameters,
                "required": [
                    key for key, value in self.parameters.items()
                    if value.get("required", False)
                ]
            }
        }

    async def execute(self, arguments: Dict[str, Any]) -> Dict:
        """Ejecuta la herramienta"""
        try:
            # Verificar si la función es async
            if inspect.iscoroutinefunction(self.function):
                result = await self.function(**arguments)
            else:
                result = self.function(**arguments)

            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class MCPResource:
    """Representa un recurso MCP"""

    def __init__(
        self,
        uri: str,
        name: str,
        description: str,
        mime_type: str = "text/plain"
    ):
        self.uri = uri
        self.name = name
        self.description = description
        self.mime_type = mime_type
        self._content = None

    def to_schema(self) -> Dict:
        """Convierte a schema MCP"""
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type
        }

    def set_content(self, content: str):
        """Establece contenido del recurso"""
        self._content = content

    def get_content(self) -> str:
        """Obtiene contenido del recurso"""
        return self._content or ""


class MCPServer:
    """
    Servidor MCP para THAU
    Expone herramientas y recursos siguiendo el protocolo MCP
    """

    def __init__(self, name: str = "thau-mcp-server", version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}

        # Registrar herramientas por defecto
        self._register_default_tools()

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function: Callable
    ):
        """Registra una nueva herramienta"""
        tool = MCPTool(name, description, parameters, function)
        self.tools[name] = tool
        print(f"🔧 Herramienta MCP registrada: {name}")

    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        content: str = "",
        mime_type: str = "text/plain"
    ):
        """Registra un nuevo recurso"""
        resource = MCPResource(uri, name, description, mime_type)
        resource.set_content(content)
        self.resources[uri] = resource
        print(f"📄 Recurso MCP registrado: {name} ({uri})")

    def _register_default_tools(self):
        """Registra herramientas por defecto de THAU"""

        # 1. Web Search
        self.register_tool(
            name="web_search",
            description="Busca información en la web usando un motor de búsqueda",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Término de búsqueda",
                    "required": True
                },
                "num_results": {
                    "type": "integer",
                    "description": "Número de resultados a devolver",
                    "required": False,
                    "default": 5
                }
            },
            function=self._web_search
        )

        # 2. Code Execution
        self.register_tool(
            name="execute_python",
            description="Ejecuta código Python en un entorno seguro",
            parameters={
                "code": {
                    "type": "string",
                    "description": "Código Python a ejecutar",
                    "required": True
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout en segundos",
                    "required": False,
                    "default": 10
                }
            },
            function=self._execute_python
        )

        # 3. Memory Recall
        self.register_tool(
            name="recall_memory",
            description="Recupera información de la memoria vectorizada",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Consulta para buscar en memoria",
                    "required": True
                },
                "k": {
                    "type": "integer",
                    "description": "Número de resultados",
                    "required": False,
                    "default": 3
                }
            },
            function=self._recall_memory
        )

        # 4. Learn Word
        self.register_tool(
            name="learn_word",
            description="Aprende una nueva palabra en un idioma",
            parameters={
                "word": {
                    "type": "string",
                    "description": "Palabra a aprender",
                    "required": True
                },
                "language": {
                    "type": "string",
                    "description": "Código de idioma (es, en, fr, etc.)",
                    "required": True
                },
                "definition": {
                    "type": "string",
                    "description": "Definición de la palabra",
                    "required": True
                },
                "examples": {
                    "type": "array",
                    "description": "Ejemplos de uso",
                    "required": False
                }
            },
            function=self._learn_word
        )

        # 5. Generate Dataset
        self.register_tool(
            name="generate_dataset",
            description="Genera un dataset de entrenamiento para un tópico específico",
            parameters={
                "topic": {
                    "type": "string",
                    "description": "Tópico del dataset",
                    "required": True
                },
                "num_examples": {
                    "type": "integer",
                    "description": "Número de ejemplos",
                    "required": False,
                    "default": 5
                }
            },
            function=self._generate_dataset
        )

    # Implementaciones de herramientas

    def _web_search(self, query: str, num_results: int = 5) -> Dict:
        """Busca en la web (simulado por ahora)"""
        # TODO: Integrar con API de búsqueda real
        return {
            "query": query,
            "results": [
                {
                    "title": f"Result {i+1} for '{query}'",
                    "url": f"https://example.com/{i}",
                    "snippet": f"This is a result about {query}..."
                }
                for i in range(num_results)
            ]
        }

    def _execute_python(self, code: str, timeout: int = 10) -> Dict:
        """Ejecuta código Python"""
        try:
            # Entorno restringido
            allowed_globals = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "range": range,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                }
            }

            # Capturar output
            import io
            import sys

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                exec(code, allowed_globals)
                output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

            return {
                "status": "success",
                "output": output
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _recall_memory(self, query: str, k: int = 3) -> Dict:
        """Recupera de memoria vectorizada"""
        # TODO: Integrar con vector_memory.py
        return {
            "query": query,
            "results": [
                {
                    "text": f"Memory result {i+1} for '{query}'",
                    "score": 0.9 - i * 0.1
                }
                for i in range(k)
            ]
        }

    def _learn_word(
        self,
        word: str,
        language: str,
        definition: str,
        examples: List[str] = None
    ) -> Dict:
        """Aprende nueva palabra"""
        # TODO: Integrar con language_learning.py
        return {
            "word": word,
            "language": language,
            "status": "learned",
            "phonetic": "placeholder_ipa"
        }

    def _generate_dataset(self, topic: str, num_examples: int = 5) -> Dict:
        """Genera dataset"""
        # TODO: Integrar con self_learning.py
        return {
            "topic": topic,
            "examples_generated": num_examples,
            "status": "success"
        }

    # Métodos del protocolo MCP

    async def handle_initialize(self, params: Dict) -> Dict:
        """Maneja solicitud de inicialización MCP"""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {}
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        }

    async def handle_tools_list(self) -> Dict:
        """Lista todas las herramientas disponibles"""
        return {
            "tools": [tool.to_schema() for tool in self.tools.values()]
        }

    async def handle_tools_call(self, name: str, arguments: Dict) -> Dict:
        """Ejecuta una herramienta"""
        if name not in self.tools:
            return {
                "isError": True,
                "content": [{
                    "type": "text",
                    "text": f"Tool '{name}' not found"
                }]
            }

        tool = self.tools[name]
        result = await tool.execute(arguments)

        if result["success"]:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(result["result"], indent=2)
                }]
            }
        else:
            return {
                "isError": True,
                "content": [{
                    "type": "text",
                    "text": f"Error: {result['error']}"
                }]
            }

    async def handle_resources_list(self) -> Dict:
        """Lista todos los recursos disponibles"""
        return {
            "resources": [resource.to_schema() for resource in self.resources.values()]
        }

    async def handle_resources_read(self, uri: str) -> Dict:
        """Lee un recurso"""
        if uri not in self.resources:
            return {
                "isError": True,
                "content": [{
                    "type": "text",
                    "text": f"Resource '{uri}' not found"
                }]
            }

        resource = self.resources[uri]

        return {
            "contents": [{
                "uri": resource.uri,
                "mimeType": resource.mime_type,
                "text": resource.get_content()
            }]
        }

    async def handle_message(self, message: Dict) -> Dict:
        """Maneja mensaje MCP"""
        method = message.get("method")

        if method == "initialize":
            return await self.handle_initialize(message.get("params", {}))
        elif method == "tools/list":
            return await self.handle_tools_list()
        elif method == "tools/call":
            params = message.get("params", {})
            return await self.handle_tools_call(
                params.get("name"),
                params.get("arguments", {})
            )
        elif method == "resources/list":
            return await self.handle_resources_list()
        elif method == "resources/read":
            params = message.get("params", {})
            return await self.handle_resources_read(params.get("uri"))
        else:
            return {
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }


# Cliente MCP para THAU
class MCPClient:
    """Cliente para que THAU llame a servidores MCP externos"""

    def __init__(self):
        self.servers: Dict[str, Dict] = {}

    def add_server(self, name: str, endpoint: str):
        """Añade servidor MCP externo"""
        self.servers[name] = {
            "endpoint": endpoint,
            "tools": [],
            "resources": []
        }

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict) -> Dict:
        """Llama herramienta en servidor externo"""
        # TODO: Implementar llamada HTTP/stdio al servidor
        return {
            "result": "placeholder - would call external MCP server"
        }


# Testing
async def test_mcp_server():
    """Prueba el servidor MCP"""
    print("🧪 Probando Servidor MCP de THAU\n")

    server = MCPServer()

    # Test 1: Initialize
    print("1. Inicialización")
    init_response = await server.handle_message({
        "method": "initialize",
        "params": {}
    })
    print(json.dumps(init_response, indent=2))

    # Test 2: List tools
    print("\n2. Listado de herramientas")
    tools_response = await server.handle_message({
        "method": "tools/list"
    })
    print(f"Herramientas disponibles: {len(tools_response['tools'])}")
    for tool in tools_response['tools']:
        print(f"  - {tool['name']}: {tool['description']}")

    # Test 3: Call tool
    print("\n3. Ejecutar herramienta (web_search)")
    call_response = await server.handle_message({
        "method": "tools/call",
        "params": {
            "name": "web_search",
            "arguments": {
                "query": "Python programming",
                "num_results": 3
            }
        }
    })
    print(call_response["content"][0]["text"])

    # Test 4: Execute Python
    print("\n4. Ejecutar código Python")
    code_response = await server.handle_message({
        "method": "tools/call",
        "params": {
            "name": "execute_python",
            "arguments": {
                "code": "print('Hello from THAU!')\nresult = 2 + 2\nprint(f'2 + 2 = {result}')"
            }
        }
    })
    print(call_response["content"][0]["text"])

    # Test 5: Learn word
    print("\n5. Aprender palabra")
    learn_response = await server.handle_message({
        "method": "tools/call",
        "params": {
            "name": "learn_word",
            "arguments": {
                "word": "computadora",
                "language": "es",
                "definition": "Máquina electrónica para procesar datos",
                "examples": ["Uso la computadora para trabajar"]
            }
        }
    })
    print(call_response["content"][0]["text"])

    print("\n✅ Pruebas completadas")


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
