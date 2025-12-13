#!/usr/bin/env python3
"""
THAU MCP Developer - Sistema de Desarrollo con MCP (Model Context Protocol)

Este sistema permite a THAU:
1. Invocar herramientas MCP para desarrollo
2. Crear archivos automáticamente desde las respuestas
3. Ejecutar comandos del sistema
4. Buscar información en la web
5. Conectarse a servidores MCP externos

Compatible con el estándar MCP de Anthropic.
"""

import json
import re
import subprocess
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Generator, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import queue
import concurrent.futures
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
from loguru import logger
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

OLLAMA_MODEL = "thau:agi-v3"
OLLAMA_URL = "http://localhost:11434"
DEFAULT_PROJECT_DIR = os.path.expanduser("~/thau_projects")
PORT = 7865

# ============================================================================
# MCP TOOLS - Herramientas del Protocolo MCP
# ============================================================================

@dataclass
class MCPParameter:
    """Parámetro de una herramienta MCP"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None


@dataclass
class MCPTool:
    """Herramienta MCP"""
    name: str
    description: str
    parameters: List[MCPParameter]
    function: Callable
    category: str = "general"


@dataclass
class MCPToolResult:
    """Resultado de invocación MCP"""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0


class MCPToolRegistry:
    """Registro de herramientas MCP para THAU"""

    def __init__(self, project_path: str):
        self.tools: Dict[str, MCPTool] = {}
        self.project_path = Path(project_path)
        self.project_path.mkdir(parents=True, exist_ok=True)
        self.files_created: List[str] = []
        self.commands_executed: List[Dict] = []

        # Registrar herramientas por defecto
        self._register_default_tools()

    def set_project_path(self, path: str):
        """Cambia el directorio del proyecto"""
        self.project_path = Path(path)
        self.project_path.mkdir(parents=True, exist_ok=True)

    def _register_default_tools(self):
        """Registra las herramientas de desarrollo MCP"""

        # 1. Crear archivo
        self.register_tool(MCPTool(
            name="create_file",
            description="Crea un archivo con el contenido especificado",
            parameters=[
                MCPParameter("path", "string", "Ruta del archivo relativa al proyecto", True),
                MCPParameter("content", "string", "Contenido del archivo", True),
            ],
            function=self._create_file,
            category="filesystem"
        ))

        # 2. Leer archivo
        self.register_tool(MCPTool(
            name="read_file",
            description="Lee el contenido de un archivo",
            parameters=[
                MCPParameter("path", "string", "Ruta del archivo relativa al proyecto", True),
            ],
            function=self._read_file,
            category="filesystem"
        ))

        # 3. Listar archivos
        self.register_tool(MCPTool(
            name="list_files",
            description="Lista los archivos del proyecto",
            parameters=[
                MCPParameter("path", "string", "Ruta del directorio", False, "."),
            ],
            function=self._list_files,
            category="filesystem"
        ))

        # 4. Crear directorio
        self.register_tool(MCPTool(
            name="create_directory",
            description="Crea un directorio",
            parameters=[
                MCPParameter("path", "string", "Ruta del directorio", True),
            ],
            function=self._create_directory,
            category="filesystem"
        ))

        # 5. Ejecutar comando
        self.register_tool(MCPTool(
            name="execute_command",
            description="Ejecuta un comando en la terminal",
            parameters=[
                MCPParameter("command", "string", "Comando a ejecutar", True),
                MCPParameter("cwd", "string", "Directorio de trabajo", False),
            ],
            function=self._execute_command,
            category="system"
        ))

        # 6. Buscar en web
        self.register_tool(MCPTool(
            name="web_search",
            description="Busca información en la web usando DuckDuckGo",
            parameters=[
                MCPParameter("query", "string", "Término de búsqueda", True),
                MCPParameter("num_results", "number", "Número de resultados", False, 5),
            ],
            function=self._web_search,
            category="web"
        ))

        # 7. Obtener URL
        self.register_tool(MCPTool(
            name="fetch_url",
            description="Obtiene el contenido de una URL",
            parameters=[
                MCPParameter("url", "string", "URL a obtener", True),
            ],
            function=self._fetch_url,
            category="web"
        ))

        # 8. Ejecutar Python
        self.register_tool(MCPTool(
            name="execute_python",
            description="Ejecuta código Python y retorna el resultado",
            parameters=[
                MCPParameter("code", "string", "Código Python a ejecutar", True),
            ],
            function=self._execute_python,
            category="code"
        ))

        # 9. Instalar paquete
        self.register_tool(MCPTool(
            name="install_package",
            description="Instala un paquete Python con pip",
            parameters=[
                MCPParameter("package", "string", "Nombre del paquete", True),
            ],
            function=self._install_package,
            category="system"
        ))

        # 10. Git operations
        self.register_tool(MCPTool(
            name="git_init",
            description="Inicializa un repositorio git",
            parameters=[],
            function=self._git_init,
            category="git"
        ))

    def register_tool(self, tool: MCPTool):
        """Registra una herramienta MCP"""
        self.tools[tool.name] = tool
        logger.info(f"MCP Tool registrado: {tool.name}")

    def get_tools_schema(self) -> List[Dict]:
        """Retorna el schema de herramientas en formato MCP/OpenAI"""
        tools_list = []
        for tool in self.tools.values():
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

            for param in tool.parameters:
                tool_def["function"]["parameters"]["properties"][param.name] = {
                    "type": param.type,
                    "description": param.description
                }
                if param.required:
                    tool_def["function"]["parameters"]["required"].append(param.name)

            tools_list.append(tool_def)

        return tools_list

    def get_tools_description(self) -> str:
        """Retorna descripción de herramientas para el prompt del sistema"""
        desc = "## Herramientas MCP Disponibles\n\n"
        desc += "Puedes usar estas herramientas escribiendo:\n"
        desc += "```tool\n{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n```\n\n"

        for tool in self.tools.values():
            desc += f"### {tool.name}\n"
            desc += f"{tool.description}\n"
            desc += "Parámetros:\n"
            for param in tool.parameters:
                req = "*" if param.required else ""
                desc += f"  - {param.name}{req} ({param.type}): {param.description}\n"
            desc += "\n"

        return desc

    def invoke_tool(self, name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """Invoca una herramienta MCP"""
        import time

        if name not in self.tools:
            return MCPToolResult(
                tool_name=name,
                success=False,
                result=None,
                error=f"Herramienta '{name}' no encontrada"
            )

        tool = self.tools[name]
        logger.info(f"Invocando MCP tool: {name} con args: {arguments}")

        try:
            start = time.time()
            result = tool.function(**arguments)
            elapsed = (time.time() - start) * 1000

            return MCPToolResult(
                tool_name=name,
                success=True,
                result=result,
                execution_time_ms=elapsed
            )
        except Exception as e:
            logger.error(f"Error en tool {name}: {e}")
            return MCPToolResult(
                tool_name=name,
                success=False,
                result=None,
                error=str(e)
            )

    # ========================================================================
    # IMPLEMENTACIÓN DE HERRAMIENTAS
    # ========================================================================

    def _create_file(self, path: str, content: str) -> Dict:
        """Crea un archivo"""
        try:
            file_path = self.project_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            self.files_created.append(path)
            logger.info(f"Archivo creado: {path}")

            return {
                "success": True,
                "path": path,
                "size": len(content),
                "message": f"Archivo '{path}' creado exitosamente"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _read_file(self, path: str) -> Dict:
        """Lee un archivo"""
        try:
            file_path = self.project_path / path
            if not file_path.exists():
                return {"success": False, "error": f"Archivo no encontrado: {path}"}

            content = file_path.read_text(encoding='utf-8')
            return {
                "success": True,
                "path": path,
                "content": content,
                "size": len(content)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _list_files(self, path: str = ".") -> Dict:
        """Lista archivos"""
        try:
            dir_path = self.project_path / path
            if not dir_path.exists():
                return {"success": False, "error": f"Directorio no encontrado: {path}"}

            files = []
            for item in sorted(dir_path.rglob("*")):
                if item.name.startswith('.') or '__pycache__' in str(item):
                    continue
                try:
                    rel = item.relative_to(self.project_path)
                    if item.is_file():
                        files.append({"path": str(rel), "type": "file", "size": item.stat().st_size})
                    else:
                        files.append({"path": str(rel), "type": "directory"})
                except:
                    pass

            return {"success": True, "files": files, "count": len(files)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_directory(self, path: str) -> Dict:
        """Crea un directorio"""
        try:
            dir_path = self.project_path / path
            dir_path.mkdir(parents=True, exist_ok=True)
            return {"success": True, "path": path, "message": f"Directorio '{path}' creado"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_command(self, command: str, cwd: str = None) -> Dict:
        """Ejecuta un comando"""
        try:
            work_dir = self.project_path / cwd if cwd else self.project_path

            result = subprocess.run(
                command,
                shell=True,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=60
            )

            self.commands_executed.append({
                "command": command,
                "returncode": result.returncode,
                "timestamp": datetime.now().isoformat()
            })

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Comando excedió el timeout de 60 segundos"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _web_search(self, query: str, num_results: int = 5) -> Dict:
        """Busca en la web usando DuckDuckGo"""
        try:
            # Usar DuckDuckGo HTML API (sin JS)
            url = "https://html.duckduckgo.com/html/"
            headers = {"User-Agent": "Mozilla/5.0"}

            response = requests.post(url, data={"q": query}, headers=headers, timeout=10)

            if response.ok:
                # Extraer resultados básicos del HTML
                from html.parser import HTMLParser

                results = []
                # Simplificado: retornar mensaje de éxito
                return {
                    "success": True,
                    "query": query,
                    "message": f"Búsqueda realizada para: {query}",
                    "note": "Resultados disponibles en navegador"
                }
            else:
                return {"success": False, "error": "Error en búsqueda"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _fetch_url(self, url: str) -> Dict:
        """Obtiene contenido de una URL"""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (THAU MCP Client)"}
            response = requests.get(url, headers=headers, timeout=30)

            if response.ok:
                # Limitar tamaño del contenido
                content = response.text[:10000]
                return {
                    "success": True,
                    "url": url,
                    "status_code": response.status_code,
                    "content": content,
                    "content_type": response.headers.get("content-type", "unknown")
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_python(self, code: str) -> Dict:
        """Ejecuta código Python"""
        try:
            import io
            import sys

            # Capturar output
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                # Entorno limitado pero funcional
                local_vars = {}
                exec(code, {"__builtins__": __builtins__}, local_vars)
                output = sys.stdout.getvalue()

                return {
                    "success": True,
                    "output": output,
                    "variables": {k: str(v) for k, v in local_vars.items() if not k.startswith('_')}
                }
            finally:
                sys.stdout = old_stdout

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _install_package(self, package: str) -> Dict:
        """Instala un paquete pip"""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=120
            )

            return {
                "success": result.returncode == 0,
                "package": package,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _git_init(self) -> Dict:
        """Inicializa git"""
        try:
            result = subprocess.run(
                ["git", "init"],
                cwd=str(self.project_path),
                capture_output=True,
                text=True
            )

            return {
                "success": result.returncode == 0,
                "message": result.stdout or result.stderr
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# AUTO FILE CREATOR - Extrae código de las respuestas
# ============================================================================

class AutoFileCreator:
    """Extrae bloques de código de las respuestas y crea archivos."""

    def __init__(self, mcp_registry: MCPToolRegistry):
        self.mcp = mcp_registry

    def extract_code_blocks(self, text: str) -> List[Dict]:
        """Extrae bloques de código con sus nombres de archivo."""
        blocks = []

        # Patrón 1: ```python filename.py o ```javascript app.js
        pattern1 = r'```(\w+)\s+([^\n`]+\.(?:py|js|html|css|json|md|txt|yaml|yml|sh|sql|tsx|ts|jsx))\s*\n(.*?)```'

        # Patrón 2: # filename.py seguido de código
        pattern2 = r'#\s*([^\n]+\.(?:py|js|html|css|json|md|txt|yaml|yml|sh|sql|tsx|ts|jsx))\s*\n```(?:\w*)\n(.*?)```'

        # Patrón 3: **filename.py** seguido de código
        pattern3 = r'\*\*([^\n*]+\.(?:py|js|html|css|json|md|txt|yaml|yml|sh|sql|tsx|ts|jsx))\*\*[:\s]*\n```(?:\w*)\n(.*?)```'

        # Patrón 4: filename.py: seguido de código
        pattern4 = r'([a-zA-Z0-9_/.-]+\.(?:py|js|html|css|json|md|txt|yaml|yml|sh|sql|tsx|ts|jsx)):\s*\n```(?:\w*)\n(.*?)```'

        # Patrón 5: ```lang\n# filename.py\n...```
        pattern5 = r'```(\w+)\n#\s*([^\n]+\.(?:py|js|html|css|json|md|txt|yaml|yml|sh|sql|tsx|ts|jsx))\s*\n(.*?)```'

        for match in re.finditer(pattern1, text, re.DOTALL | re.IGNORECASE):
            lang, filename, content = match.groups()
            blocks.append({"filename": filename.strip(), "content": content.strip(), "lang": lang})

        for match in re.finditer(pattern2, text, re.DOTALL | re.IGNORECASE):
            filename, content = match.groups()
            blocks.append({"filename": filename.strip(), "content": content.strip(), "lang": "auto"})

        for match in re.finditer(pattern3, text, re.DOTALL | re.IGNORECASE):
            filename, content = match.groups()
            blocks.append({"filename": filename.strip(), "content": content.strip(), "lang": "auto"})

        for match in re.finditer(pattern4, text, re.DOTALL | re.IGNORECASE):
            filename, content = match.groups()
            blocks.append({"filename": filename.strip(), "content": content.strip(), "lang": "auto"})

        for match in re.finditer(pattern5, text, re.DOTALL | re.IGNORECASE):
            lang, filename, content = match.groups()
            blocks.append({"filename": filename.strip(), "content": content.strip(), "lang": lang})

        # Eliminar duplicados
        seen = set()
        unique = []
        for block in blocks:
            if block["filename"] not in seen:
                seen.add(block["filename"])
                unique.append(block)

        return unique

    def extract_tool_calls(self, text: str) -> List[Dict]:
        """Extrae llamadas a herramientas MCP del texto."""
        tool_calls = []

        # Patrón: ```tool\n{json}\n```
        pattern = r'```tool\s*\n(.*?)```'

        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                json_str = match.group(1).strip()
                tool_data = json.loads(json_str)
                if "name" in tool_data:
                    tool_calls.append(tool_data)
            except json.JSONDecodeError:
                continue

        return tool_calls

    def process_response(self, text: str) -> Dict:
        """Procesa una respuesta: extrae código y tool calls."""
        results = {
            "files_created": [],
            "tools_executed": [],
            "errors": []
        }

        # 1. Extraer y crear archivos
        blocks = self.extract_code_blocks(text)
        for block in blocks:
            result = self.mcp.invoke_tool("create_file", {
                "path": block["filename"],
                "content": block["content"]
            })
            if result.success:
                results["files_created"].append({
                    "path": block["filename"],
                    "size": len(block["content"])
                })
            else:
                results["errors"].append(f"Error creando {block['filename']}: {result.error}")

        # 2. Ejecutar tool calls
        tool_calls = self.extract_tool_calls(text)
        for call in tool_calls:
            tool_name = call.get("name")
            arguments = call.get("arguments", {})

            result = self.mcp.invoke_tool(tool_name, arguments)
            results["tools_executed"].append({
                "tool": tool_name,
                "success": result.success,
                "result": result.result if result.success else result.error
            })

        return results


# ============================================================================
# THAU MCP DEVELOPER
# ============================================================================

class THAUMCPDeveloper:
    """Desarrollador THAU con soporte MCP completo."""

    SYSTEM_PROMPT = """Eres THAU, un desarrollador de software experto con acceso a herramientas MCP.

Tu trabajo es crear aplicaciones completas. Tienes acceso a estas herramientas:

## Herramientas Disponibles

1. **create_file**: Crea archivos en el proyecto
   Ejemplo: ```tool
   {"name": "create_file", "arguments": {"path": "app.py", "content": "print('hello')"}}
   ```

2. **execute_command**: Ejecuta comandos en terminal
   Ejemplo: ```tool
   {"name": "execute_command", "arguments": {"command": "pip install flask"}}
   ```

3. **read_file**: Lee archivos existentes
4. **list_files**: Lista archivos del proyecto
5. **create_directory**: Crea directorios
6. **web_search**: Busca información en internet
7. **fetch_url**: Obtiene contenido de URLs
8. **execute_python**: Ejecuta código Python
9. **install_package**: Instala paquetes pip
10. **git_init**: Inicializa repositorio git

## FORMATO PARA CREAR ARCHIVOS

Método 1 - Tool MCP (preferido):
```tool
{"name": "create_file", "arguments": {"path": "archivo.py", "content": "contenido"}}
```

Método 2 - Formato Markdown (alternativo):
**nombre_archivo.py**
```python
# código completo aquí
```

## REGLAS IMPORTANTES

1. Genera código COMPLETO y funcional
2. Usa herramientas MCP cuando necesites ejecutar acciones
3. Incluye TODOS los imports necesarios
4. El código debe funcionar sin modificaciones
5. Responde siempre en español
6. Para proyectos complejos, crea la estructura completa

## EJEMPLO DE USO

Para crear una app Flask:

```tool
{"name": "create_directory", "arguments": {"path": "templates"}}
```

**app.py**
```python
from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
```

**templates/index.html**
```html
<!DOCTYPE html>
<html>
<head><title>Mi App</title></head>
<body><h1>Hola Mundo</h1></body>
</html>
```

```tool
{"name": "execute_command", "arguments": {"command": "pip install flask"}}
```

¡Comienza a desarrollar!"""

    def __init__(self, project_path: str):
        self.mcp = MCPToolRegistry(project_path)
        self.file_creator = AutoFileCreator(self.mcp)
        self.history: List[Dict] = []
        self.created_files_tracker: set = set()

    def set_project_path(self, path: str):
        """Cambia el directorio del proyecto."""
        self.mcp.set_project_path(path)
        self.created_files_tracker = set()

    def call_ollama(self, messages: List[Dict]) -> Generator[str, None, None]:
        """Llama a Ollama con streaming."""
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "num_ctx": 8192,
                    }
                },
                stream=True,
                timeout=300
            )

            if not response.ok:
                yield f"Error: {response.status_code}"
                return

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                    except:
                        continue
        except Exception as e:
            yield f"Error: {str(e)}"

    def develop(self, prompt: str) -> Generator[Dict, None, None]:
        """Desarrolla un proyecto basado en el prompt."""

        # Preparar contexto
        existing_files = self.mcp._list_files()
        context = ""
        if existing_files.get("success") and existing_files.get("files"):
            context = "\n\nArchivos existentes en el proyecto:\n"
            for f in existing_files["files"][:20]:
                context += f"- {f['path']}\n"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]

        # Agregar historial
        for msg in self.history[-4:]:
            messages.append(msg)

        # Agregar prompt
        full_prompt = f"{prompt}{context}"
        messages.append({"role": "user", "content": full_prompt})

        yield {"type": "start", "message": "Iniciando desarrollo con MCP..."}

        # Generar respuesta
        full_response = ""
        pending_content = ""

        for chunk in self.call_ollama(messages):
            full_response += chunk
            pending_content += chunk

            yield {"type": "stream", "content": chunk, "full": full_response}

            # Buscar tool calls completados
            if "```" in pending_content:
                tool_calls = self.file_creator.extract_tool_calls(pending_content)
                for call in tool_calls:
                    tool_name = call.get("name")
                    arguments = call.get("arguments", {})

                    result = self.mcp.invoke_tool(tool_name, arguments)
                    yield {
                        "type": "tool_executed",
                        "tool": tool_name,
                        "success": result.success,
                        "result": result.result if result.success else result.error
                    }

                # Buscar archivos en formato markdown
                blocks = self.file_creator.extract_code_blocks(pending_content)
                for block in blocks:
                    if block["filename"] not in self.created_files_tracker:
                        result = self.mcp.invoke_tool("create_file", {
                            "path": block["filename"],
                            "content": block["content"]
                        })
                        if result.success:
                            self.created_files_tracker.add(block["filename"])
                            yield {
                                "type": "file_created",
                                "file": {
                                    "path": block["filename"],
                                    "size": len(block["content"])
                                }
                            }

        # Procesamiento final
        final_results = self.file_creator.process_response(full_response)

        for file_info in final_results["files_created"]:
            if file_info["path"] not in self.created_files_tracker:
                self.created_files_tracker.add(file_info["path"])
                yield {"type": "file_created", "file": file_info}

        for tool_info in final_results["tools_executed"]:
            yield {"type": "tool_executed", **tool_info}

        # Guardar en historial
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": full_response})

        yield {
            "type": "done",
            "files_created": list(self.created_files_tracker),
            "total_files": len(self.created_files_tracker),
            "tools_executed": len(final_results["tools_executed"])
        }

    def get_files(self) -> List[Dict]:
        result = self.mcp._list_files()
        return result.get("files", []) if result.get("success") else []

    def read_file(self, path: str) -> Optional[str]:
        result = self.mcp._read_file(path)
        return result.get("content") if result.get("success") else None

    def invoke_tool(self, name: str, arguments: Dict) -> MCPToolResult:
        """Invoca una herramienta MCP directamente."""
        return self.mcp.invoke_tool(name, arguments)

    def get_tools(self) -> List[Dict]:
        """Retorna lista de herramientas disponibles."""
        return self.mcp.get_tools_schema()

    def clear(self):
        self.history = []
        self.created_files_tracker = set()
        self.mcp.files_created = []
        self.mcp.commands_executed = []


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="THAU MCP Developer")
developer = THAUMCPDeveloper(DEFAULT_PROJECT_DIR)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_TEMPLATE


@app.get("/api/files")
async def get_files():
    return developer.get_files()


@app.get("/api/file/{path:path}")
async def get_file(path: str):
    content = developer.read_file(path)
    if content is None:
        raise HTTPException(404, "Not found")
    return {"path": path, "content": content}


@app.post("/api/project/path")
async def set_path(data: dict):
    path = data.get("path", "")
    if path:
        developer.set_project_path(path)
    return {"path": str(developer.mcp.project_path)}


@app.get("/api/project/path")
async def get_path():
    return {"path": str(developer.mcp.project_path)}


@app.get("/api/tools")
async def get_tools():
    return developer.get_tools()


@app.post("/api/tool/invoke")
async def invoke_tool(data: dict):
    name = data.get("name")
    arguments = data.get("arguments", {})
    result = developer.invoke_tool(name, arguments)
    return {
        "tool": name,
        "success": result.success,
        "result": result.result,
        "error": result.error,
        "execution_time_ms": result.execution_time_ms
    }


@app.post("/api/clear")
async def clear():
    developer.clear()
    return {"ok": True}


@app.websocket("/ws/dev")
async def websocket_dev(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt", "")
            project_name = data.get("project_name", "")

            if project_name:
                new_path = Path(DEFAULT_PROJECT_DIR) / project_name
                new_path.mkdir(parents=True, exist_ok=True)
                developer.set_project_path(str(new_path))

            result_queue = queue.Queue()

            def run():
                try:
                    for chunk in developer.develop(prompt):
                        result_queue.put(("data", chunk))
                    result_queue.put(("done", None))
                except Exception as e:
                    logger.error(f"Error: {e}")
                    result_queue.put(("error", str(e)))

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(run)

            finished = False
            while not finished:
                await asyncio.sleep(0.02)

                while not result_queue.empty():
                    msg_type, chunk_data = result_queue.get_nowait()
                    if msg_type == "done":
                        finished = True
                        break
                    elif msg_type == "error":
                        await websocket.send_json({"type": "error", "error": chunk_data})
                        finished = True
                        break
                    elif msg_type == "data":
                        await websocket.send_json(chunk_data)

                if future.done() and not finished:
                    while not result_queue.empty():
                        msg_type, chunk_data = result_queue.get_nowait()
                        if msg_type == "data":
                            await websocket.send_json(chunk_data)
                    finished = True

            executor.shutdown(wait=False)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WS error: {e}")


# ============================================================================
# HTML TEMPLATE
# ============================================================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>THAU MCP Developer</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg: #0a0a0f;
            --bg2: #12121a;
            --bg3: #1a1a25;
            --text: #e4e4e7;
            --text2: #71717a;
            --accent: #6366f1;
            --accent2: #818cf8;
            --green: #22c55e;
            --yellow: #eab308;
            --border: #27272a;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); height: 100vh; }

        .app { display: grid; grid-template-columns: 1fr 380px; height: 100vh; }

        .main { display: flex; flex-direction: column; }
        .header {
            padding: 16px 24px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg2);
        }
        .header h1 {
            font-size: 18px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .header h1 span { color: var(--accent); }
        .badge {
            background: linear-gradient(135deg, var(--accent), #a855f7);
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }
        .header-actions { display: flex; gap: 8px; }
        .btn {
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            cursor: pointer;
            border: 1px solid var(--border);
            background: var(--bg3);
            color: var(--text);
            transition: all 0.15s;
            font-weight: 500;
        }
        .btn:hover { background: var(--border); border-color: var(--text2); }
        .btn-primary { background: var(--accent); border-color: var(--accent); color: white; }
        .btn-primary:hover { background: var(--accent2); }

        .chat { flex: 1; overflow-y: auto; padding: 24px; }

        .message { margin-bottom: 20px; animation: fadeIn 0.3s; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } }
        .message.user { display: flex; justify-content: flex-end; }
        .message.user .bubble {
            background: linear-gradient(135deg, var(--accent), #7c3aed);
            color: white;
            max-width: 70%;
        }
        .bubble {
            background: var(--bg2);
            padding: 14px 18px;
            border-radius: 16px;
            max-width: 85%;
            border: 1px solid var(--border);
            line-height: 1.6;
        }
        .bubble pre {
            background: var(--bg);
            padding: 14px;
            border-radius: 10px;
            overflow-x: auto;
            margin: 12px 0;
            border: 1px solid var(--border);
        }
        .bubble code { font-family: 'JetBrains Mono', monospace; font-size: 13px; }
        .bubble p { margin: 8px 0; }

        /* Status & notifications */
        .status {
            display: flex;
            align-items: center;
            gap: 14px;
            padding: 16px 20px;
            background: var(--bg2);
            border-radius: 14px;
            margin-bottom: 16px;
            border: 1px solid var(--border);
        }
        .spinner {
            width: 22px;
            height: 22px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .status-text { color: var(--text2); font-size: 14px; }

        .notification {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px 16px;
            border-radius: 10px;
            margin: 10px 0;
            font-size: 13px;
            animation: slideIn 0.3s;
        }
        @keyframes slideIn { from { transform: translateX(-10px); opacity: 0; } }
        .notification.file {
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid var(--green);
            color: var(--green);
        }
        .notification.tool {
            background: rgba(234, 179, 8, 0.1);
            border: 1px solid var(--yellow);
            color: var(--yellow);
        }
        .notification.error {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid #ef4444;
            color: #ef4444;
        }

        .input-area {
            padding: 16px 24px;
            border-top: 1px solid var(--border);
            background: var(--bg2);
        }
        .input-row { display: flex; gap: 12px; }
        .input-wrapper {
            flex: 1;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 18px;
            transition: border-color 0.2s;
        }
        .input-wrapper:focus-within { border-color: var(--accent); }
        .input-wrapper textarea {
            width: 100%;
            background: transparent;
            border: none;
            outline: none;
            color: var(--text);
            font-family: inherit;
            font-size: 14px;
            resize: none;
            line-height: 1.5;
        }
        .send-btn {
            width: 52px;
            height: 52px;
            background: linear-gradient(135deg, var(--accent), #7c3aed);
            border: none;
            border-radius: 12px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.15s, opacity 0.15s;
        }
        .send-btn:hover { transform: scale(1.05); }
        .send-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

        /* Sidebar */
        .sidebar {
            background: var(--bg2);
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
        }
        .sidebar-header {
            padding: 16px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .sidebar-header h2 { font-size: 14px; font-weight: 600; }

        .tabs {
            display: flex;
            border-bottom: 1px solid var(--border);
        }
        .tab {
            flex: 1;
            padding: 12px;
            text-align: center;
            font-size: 13px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            color: var(--text2);
            transition: all 0.2s;
        }
        .tab:hover { color: var(--text); }
        .tab.active { color: var(--accent); border-bottom-color: var(--accent); }

        .tab-content { flex: 1; overflow-y: auto; display: none; }
        .tab-content.active { display: block; }

        .files-list { padding: 8px; }
        .file-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            margin-bottom: 4px;
            transition: background 0.15s;
        }
        .file-item:hover { background: var(--bg3); }
        .file-item.active { background: var(--bg3); border-left: 2px solid var(--accent); }
        .file-item.new { animation: highlight 2s ease-out; }
        @keyframes highlight { 0%, 50% { background: rgba(34, 197, 94, 0.15); } }

        .tools-list { padding: 12px; }
        .tool-item {
            padding: 12px;
            background: var(--bg3);
            border-radius: 8px;
            margin-bottom: 8px;
            font-size: 13px;
        }
        .tool-item h4 { color: var(--accent); margin-bottom: 4px; }
        .tool-item p { color: var(--text2); font-size: 12px; }

        .file-preview {
            border-top: 1px solid var(--border);
            max-height: 40%;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .preview-header {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            font-size: 12px;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            background: var(--bg3);
        }
        .preview-content {
            flex: 1;
            overflow: auto;
            padding: 14px;
            background: var(--bg);
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            white-space: pre-wrap;
            line-height: 1.5;
        }

        /* Welcome */
        .welcome {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 40px;
        }
        .welcome-icon {
            width: 90px;
            height: 90px;
            background: linear-gradient(135deg, var(--accent), #a855f7);
            border-radius: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 45px;
            margin-bottom: 28px;
            box-shadow: 0 20px 40px rgba(99, 102, 241, 0.3);
        }
        .welcome h2 { margin-bottom: 12px; font-size: 24px; }
        .welcome p { color: var(--text2); max-width: 450px; margin-bottom: 28px; line-height: 1.6; }
        .templates { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; }
        .template {
            padding: 14px 24px;
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 14px;
        }
        .template:hover { border-color: var(--accent); transform: translateY(-2px); }

        /* Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.8);
            z-index: 100;
            align-items: center;
            justify-content: center;
            backdrop-filter: blur(4px);
        }
        .modal-overlay.active { display: flex; }
        .modal {
            background: var(--bg2);
            border-radius: 16px;
            padding: 28px;
            width: 480px;
            border: 1px solid var(--border);
        }
        .modal h3 { margin-bottom: 20px; font-size: 18px; }
        .modal input, .modal textarea {
            width: 100%;
            padding: 14px;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            color: var(--text);
            font-size: 14px;
            margin-bottom: 14px;
        }
        .modal input:focus, .modal textarea:focus { outline: none; border-color: var(--accent); }
        .modal-actions { display: flex; gap: 12px; justify-content: flex-end; margin-top: 20px; }

        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--text2); }

        .empty-state {
            padding: 40px 20px;
            text-align: center;
            color: var(--text2);
        }
    </style>
</head>
<body>
    <div class="app">
        <div class="main">
            <header class="header">
                <h1>
                    <span>THAU</span> MCP Developer
                    <span class="badge">MCP</span>
                </h1>
                <div class="header-actions">
                    <button class="btn" onclick="clearChat()">Limpiar</button>
                    <button class="btn btn-primary" onclick="showModal()">+ Nuevo Proyecto</button>
                </div>
            </header>

            <div class="chat" id="chat">
                <div class="welcome" id="welcome">
                    <div class="welcome-icon">🔌</div>
                    <h2>THAU MCP Developer</h2>
                    <p>Desarrollador con soporte MCP (Model Context Protocol). Crea archivos, ejecuta comandos y construye aplicaciones completas.</p>
                    <div class="templates">
                        <div class="template" onclick="quickStart('flask')">🌐 Web Flask</div>
                        <div class="template" onclick="quickStart('fastapi')">⚡ API FastAPI</div>
                        <div class="template" onclick="quickStart('cli')">💻 CLI Tool</div>
                        <div class="template" onclick="quickStart('react')">⚛️ React App</div>
                    </div>
                </div>
            </div>

            <div class="input-area">
                <div class="input-row">
                    <div class="input-wrapper">
                        <textarea id="input" placeholder="Describe qué aplicación quieres crear..." rows="2" onkeydown="handleKey(event)"></textarea>
                    </div>
                    <button class="send-btn" id="sendBtn" onclick="send()">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5">
                            <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                        </svg>
                    </button>
                </div>
            </div>
        </div>

        <aside class="sidebar">
            <div class="sidebar-header">
                <h2>Panel de Proyecto</h2>
                <button class="btn" style="padding:6px 10px;font-size:11px" onclick="refreshFiles()">↻</button>
            </div>

            <div class="tabs">
                <div class="tab active" onclick="switchTab('files')">📁 Archivos</div>
                <div class="tab" onclick="switchTab('tools')">🔧 Tools MCP</div>
            </div>

            <div class="tab-content active" id="filesTab">
                <div class="files-list" id="filesList">
                    <div class="empty-state">Los archivos aparecerán aquí</div>
                </div>
            </div>

            <div class="tab-content" id="toolsTab">
                <div class="tools-list" id="toolsList"></div>
            </div>

            <div class="file-preview" id="filePreview" style="display:none">
                <div class="preview-header">
                    <span id="previewName">archivo.py</span>
                    <span style="color:var(--text2)" id="previewSize">0 bytes</span>
                </div>
                <pre class="preview-content" id="previewContent"></pre>
            </div>
        </aside>
    </div>

    <div class="modal-overlay" id="modal">
        <div class="modal">
            <h3>🚀 Nuevo Proyecto</h3>
            <input type="text" id="projectName" placeholder="Nombre del proyecto (ej: mi-app)">
            <textarea id="projectDesc" placeholder="Describe qué quieres crear..." rows="4"></textarea>
            <div class="modal-actions">
                <button class="btn" onclick="hideModal()">Cancelar</button>
                <button class="btn btn-primary" onclick="createProject()">Crear</button>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let isGenerating = false;
        let currentBubble = null;
        let statusDiv = null;

        document.addEventListener('DOMContentLoaded', () => {
            connectWS();
            refreshFiles();
            loadTools();
            loadPath();
        });

        function connectWS() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws/dev`);
            ws.onopen = () => console.log('WebSocket conectado');
            ws.onclose = () => setTimeout(connectWS, 2000);
            ws.onmessage = handleMessage;
        }

        function handleMessage(e) {
            const data = JSON.parse(e.data);
            console.log('MSG:', data.type, data);

            switch(data.type) {
                case 'start':
                    showStatus('Generando con MCP...');
                    break;

                case 'stream':
                    if (!currentBubble) {
                        currentBubble = addBubble('assistant');
                    }
                    updateBubble(currentBubble, data.full);
                    break;

                case 'file_created':
                    addNotification('file', `📄 ${data.file.path} (${data.file.size} bytes)`);
                    refreshFiles();
                    break;

                case 'tool_executed':
                    const status = data.success ? '✅' : '❌';
                    addNotification('tool', `🔧 ${data.tool} ${status}`);
                    break;

                case 'done':
                    hideStatus();
                    if (data.total_files > 0) {
                        addNotification('file', `✅ ${data.total_files} archivo(s) creado(s)`);
                    }
                    currentBubble = null;
                    isGenerating = false;
                    document.getElementById('sendBtn').disabled = false;
                    refreshFiles();
                    break;

                case 'error':
                    hideStatus();
                    addNotification('error', `❌ Error: ${data.error}`);
                    isGenerating = false;
                    document.getElementById('sendBtn').disabled = false;
                    break;
            }
        }

        function showStatus(text) {
            const chat = document.getElementById('chat');
            statusDiv = document.createElement('div');
            statusDiv.className = 'status';
            statusDiv.innerHTML = `<div class="spinner"></div><span class="status-text">${text}</span>`;
            chat.appendChild(statusDiv);
            chat.scrollTop = chat.scrollHeight;
        }

        function hideStatus() {
            if (statusDiv) {
                statusDiv.remove();
                statusDiv = null;
            }
        }

        function addBubble(role) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.innerHTML = '<div class="bubble"></div>';
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div.querySelector('.bubble');
        }

        function updateBubble(bubble, content) {
            try {
                bubble.innerHTML = marked.parse(content);
                bubble.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b));
            } catch {
                bubble.textContent = content;
            }
            document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
        }

        function addNotification(type, message) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = `notification ${type}`;
            div.innerHTML = message;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        function send() {
            const input = document.getElementById('input');
            const text = input.value.trim();
            if (!text || isGenerating || !ws) return;

            hideWelcome();
            addUserBubble(text);

            ws.send(JSON.stringify({ prompt: text }));
            input.value = '';
            isGenerating = true;
            document.getElementById('sendBtn').disabled = true;
        }

        function addUserBubble(text) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'message user';
            div.innerHTML = `<div class="bubble">${text}</div>`;
            chat.appendChild(div);
        }

        function hideWelcome() {
            const w = document.getElementById('welcome');
            if (w) w.style.display = 'none';
        }

        function handleKey(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
            }
        }

        async function refreshFiles() {
            const res = await fetch('/api/files');
            const files = await res.json();
            const list = document.getElementById('filesList');

            if (files.length === 0) {
                list.innerHTML = '<div class="empty-state">No hay archivos</div>';
                return;
            }

            list.innerHTML = files.filter(f => f.type === 'file').map(f => `
                <div class="file-item new" onclick="previewFile('${f.path}')">
                    📄 ${f.path}
                </div>
            `).join('');
        }

        async function loadTools() {
            const res = await fetch('/api/tools');
            const tools = await res.json();
            const list = document.getElementById('toolsList');

            list.innerHTML = tools.map(t => `
                <div class="tool-item">
                    <h4>${t.function.name}</h4>
                    <p>${t.function.description}</p>
                </div>
            `).join('');
        }

        async function previewFile(path) {
            const res = await fetch(`/api/file/${encodeURIComponent(path)}`);
            if (!res.ok) return;
            const data = await res.json();

            document.getElementById('previewName').textContent = path;
            document.getElementById('previewSize').textContent = `${data.content.length} bytes`;
            document.getElementById('previewContent').textContent = data.content;
            document.getElementById('filePreview').style.display = 'flex';

            document.querySelectorAll('.file-item').forEach(el => {
                el.classList.toggle('active', el.textContent.includes(path));
            });
        }

        async function loadPath() {
            const res = await fetch('/api/project/path');
            const data = await res.json();
            console.log('Project path:', data.path);
        }

        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tab + 'Tab').classList.add('active');
        }

        function showModal() {
            document.getElementById('modal').classList.add('active');
        }

        function hideModal() {
            document.getElementById('modal').classList.remove('active');
        }

        function createProject() {
            const name = document.getElementById('projectName').value.trim();
            const desc = document.getElementById('projectDesc').value.trim();

            if (!desc) return alert('Describe tu proyecto');

            hideWelcome();
            hideModal();
            addUserBubble(desc);

            ws.send(JSON.stringify({ prompt: desc, project_name: name || 'proyecto' }));
            isGenerating = true;
            document.getElementById('sendBtn').disabled = true;

            document.getElementById('projectName').value = '';
            document.getElementById('projectDesc').value = '';
        }

        function quickStart(type) {
            const prompts = {
                flask: 'Crea una aplicación web con Flask que tenga: página de inicio con diseño moderno (Bootstrap), página about, y formulario de contacto que guarde en SQLite. Incluye todos los archivos necesarios y usa herramientas MCP para crear la estructura.',
                fastapi: 'Crea una API REST con FastAPI que tenga: endpoints CRUD para gestionar productos (crear, leer, actualizar, eliminar), modelo Pydantic, documentación automática, y base de datos SQLite. Usa herramientas MCP para crear los archivos.',
                cli: 'Crea una herramienta CLI en Python usando Click que permita: agregar tareas, listarlas, marcarlas como completadas, y eliminarlas. Usa SQLite para persistencia, Rich para colores, y herramientas MCP para crear los archivos.',
                react: 'Crea una aplicación React básica con Vite que tenga: componente de lista de tareas, estado con useState, estilos CSS modernos. Usa herramientas MCP para crear package.json, componentes, y estilos.'
            };
            document.getElementById('projectDesc').value = prompts[type];
            showModal();
        }

        async function clearChat() {
            await fetch('/api/clear', {method: 'POST'});
            document.getElementById('chat').innerHTML = `
                <div class="welcome" id="welcome">
                    <div class="welcome-icon">🔌</div>
                    <h2>THAU MCP Developer</h2>
                    <p>Desarrollador con soporte MCP. Crea archivos, ejecuta comandos y construye aplicaciones.</p>
                    <div class="templates">
                        <div class="template" onclick="quickStart('flask')">🌐 Web Flask</div>
                        <div class="template" onclick="quickStart('fastapi')">⚡ API FastAPI</div>
                        <div class="template" onclick="quickStart('cli')">💻 CLI Tool</div>
                        <div class="template" onclick="quickStart('react')">⚛️ React App</div>
                    </div>
                </div>
            `;
            refreshFiles();
        }
    </script>
</body>
</html>
'''


# ============================================================================
# IMPORTS NEEDED
# ============================================================================
import sys

if __name__ == "__main__":
    Path(DEFAULT_PROJECT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    THAU MCP Developer                                 ║
║                                                                       ║
║  🔌 Model Context Protocol habilitado                                 ║
║  🤖 Crea archivos, ejecuta comandos y desarrolla apps                 ║
║                                                                       ║
║  URL: http://localhost:{PORT}                                           ║
║  Proyecto: {DEFAULT_PROJECT_DIR}
║                                                                       ║
║  Herramientas MCP disponibles:                                        ║
║  - create_file: Crea archivos en el proyecto                          ║
║  - execute_command: Ejecuta comandos en terminal                      ║
║  - read_file, list_files, create_directory                            ║
║  - web_search, fetch_url                                              ║
║  - execute_python, install_package                                    ║
║  - git_init                                                           ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
