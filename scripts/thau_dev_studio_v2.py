#!/usr/bin/env python3
"""
THAU Development Studio v2 - Sistema Multi-Agente para Desarrollo de Aplicaciones.

Mejoras:
- Indicador animado de "pensando"
- Pensamiento expandible/colapsable
- Mejor parsing de herramientas
- Creación real de archivos
"""

import json
import re
import subprocess
import os
import signal
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import time
import queue
import concurrent.futures

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
from loguru import logger
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

OLLAMA_MODEL = "thau:agi-v3"
OLLAMA_URL = "http://localhost:11434"
DEFAULT_PROJECT_DIR = os.path.expanduser("~/thau_projects")
PORT = 7863

# ============================================================================
# DATA MODELS
# ============================================================================

class AgentRole(Enum):
    PMO = "pmo"
    ARCHITECT = "architect"
    BACKEND = "backend"
    FRONTEND = "frontend"
    UX = "ux"
    QA = "qa"
    DEVOPS = "devops"


@dataclass
class ToolResult:
    success: bool
    output: str
    error: Optional[str] = None


# ============================================================================
# TOOL EXECUTOR - Ejecuta herramientas reales
# ============================================================================

class ToolExecutor:
    """Ejecutor de herramientas del sistema."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.project_path.mkdir(parents=True, exist_ok=True)
        self.files_created: List[str] = []
        self.running_process = None

    def _safe_path(self, path: str) -> Path:
        """Asegurar ruta dentro del proyecto."""
        if path.startswith("/"):
            full_path = Path(path)
        else:
            full_path = (self.project_path / path).resolve()
        return full_path

    def create_file(self, path: str, content: str) -> ToolResult:
        """Crear un archivo."""
        try:
            file_path = self._safe_path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            rel_path = str(file_path.relative_to(self.project_path))
            self.files_created.append(rel_path)
            logger.info(f"Archivo creado: {rel_path}")
            return ToolResult(True, f"✅ Archivo creado: {rel_path} ({len(content)} bytes)")
        except Exception as e:
            logger.error(f"Error creando archivo: {e}")
            return ToolResult(False, "", str(e))

    def create_directory(self, path: str) -> ToolResult:
        """Crear directorio."""
        try:
            dir_path = self._safe_path(path)
            dir_path.mkdir(parents=True, exist_ok=True)
            rel_path = str(dir_path.relative_to(self.project_path))
            logger.info(f"Directorio creado: {rel_path}")
            return ToolResult(True, f"✅ Directorio creado: {rel_path}")
        except Exception as e:
            return ToolResult(False, "", str(e))

    def read_file(self, path: str) -> ToolResult:
        """Leer archivo."""
        try:
            file_path = self._safe_path(path)
            if not file_path.exists():
                return ToolResult(False, "", f"Archivo no encontrado: {path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return ToolResult(True, content)
        except Exception as e:
            return ToolResult(False, "", str(e))

    def list_files(self, path: str = ".") -> ToolResult:
        """Listar archivos."""
        try:
            dir_path = self._safe_path(path)
            if not dir_path.exists():
                return ToolResult(True, "(vacío)")

            items = []
            for item in sorted(dir_path.rglob("*")):
                if item.name.startswith('.') or '__pycache__' in str(item):
                    continue
                try:
                    rel = item.relative_to(self.project_path)
                    if item.is_dir():
                        items.append(f"📁 {rel}/")
                    else:
                        size = item.stat().st_size
                        items.append(f"📄 {rel} ({size:,} bytes)")
                except:
                    pass

            return ToolResult(True, "\n".join(items) if items else "(vacío)")
        except Exception as e:
            return ToolResult(False, "", str(e))

    def run_command(self, command: str, timeout: int = 60) -> ToolResult:
        """Ejecutar comando."""
        dangerous = ['rm -rf /', 'mkfs', '> /dev', 'dd if=']
        for d in dangerous:
            if d in command:
                return ToolResult(False, "", f"Comando bloqueado: {command}")

        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=str(self.project_path)
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            return ToolResult(result.returncode == 0, output or "(sin salida)",
                            result.stderr if result.returncode != 0 else None)
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", f"Timeout ({timeout}s)")
        except Exception as e:
            return ToolResult(False, "", str(e))

    def execute(self, name: str, args: Dict) -> ToolResult:
        """Ejecutar herramienta por nombre."""
        logger.info(f"Ejecutando herramienta: {name} con args: {args}")

        tools = {
            "create_file": lambda: self.create_file(args.get("path", ""), args.get("content", "")),
            "create_directory": lambda: self.create_directory(args.get("path", "")),
            "read_file": lambda: self.read_file(args.get("path", "")),
            "list_files": lambda: self.list_files(args.get("path", ".")),
            "run_command": lambda: self.run_command(args.get("command", "")),
        }

        if name not in tools:
            return ToolResult(False, "", f"Herramienta desconocida: {name}")

        result = tools[name]()
        logger.info(f"Resultado: success={result.success}, output={result.output[:100] if result.output else ''}")
        return result


# ============================================================================
# AGENT PROMPTS - Simplificados para mejor parsing
# ============================================================================

SYSTEM_PROMPT_TEMPLATE = """Eres {role_name}, un agente de desarrollo del equipo THAU.

HERRAMIENTAS DISPONIBLES:
Usa este formato EXACTO para ejecutar herramientas:

Para crear un archivo:
```tool
create_file
path: ruta/al/archivo.py
content:
contenido del archivo aquí
```

Para crear directorio:
```tool
create_directory
path: ruta/directorio
```

Para listar archivos:
```tool
list_files
path: .
```

Para ejecutar comando:
```tool
run_command
command: pip install flask
```

IMPORTANTE:
- Usa las herramientas para CREAR archivos reales
- Crea código COMPLETO y funcional
- Después de crear archivos, usa list_files para verificar
- Responde siempre en español

{role_specific}
"""

ROLE_PROMPTS = {
    AgentRole.PMO: """Tu rol es PMO (Project Manager):
- Coordina el proyecto y el equipo
- Define fases y tareas
- Delega trabajo a otros agentes con @mentions
- Crea estructura inicial del proyecto""",

    AgentRole.ARCHITECT: """Tu rol es Arquitecto de Software:
- Diseña la arquitectura del sistema
- Define estructura de carpetas
- Crea archivos de configuración (package.json, requirements.txt, etc.)
- Documenta decisiones técnicas""",

    AgentRole.BACKEND: """Tu rol es Desarrollador Backend:
- Implementa APIs y servicios
- Crea modelos y endpoints
- Maneja base de datos
- Escribe código Python/Node.js completo""",

    AgentRole.FRONTEND: """Tu rol es Desarrollador Frontend:
- Implementa interfaces de usuario
- Crea componentes HTML/CSS/JS
- Conecta con APIs
- Asegura diseño responsive""",

    AgentRole.UX: """Tu rol es Diseñador UX/UI:
- Diseña experiencia de usuario
- Crea estilos CSS
- Define paleta de colores
- Asegura accesibilidad""",

    AgentRole.QA: """Tu rol es QA Engineer:
- Crea pruebas automatizadas
- Valida funcionalidad
- Identifica bugs
- Escribe tests""",

    AgentRole.DEVOPS: """Tu rol es DevOps Engineer:
- Crea Dockerfiles
- Configura CI/CD
- Escribe scripts de deployment
- Documenta setup""",
}

AGENT_STYLES = {
    AgentRole.PMO: {"color": "#6366f1", "emoji": "📊", "name": "PMO"},
    AgentRole.ARCHITECT: {"color": "#8b5cf6", "emoji": "🏗️", "name": "Arquitecto"},
    AgentRole.BACKEND: {"color": "#10b981", "emoji": "⚙️", "name": "Backend"},
    AgentRole.FRONTEND: {"color": "#f59e0b", "emoji": "🎨", "name": "Frontend"},
    AgentRole.UX: {"color": "#ec4899", "emoji": "✨", "name": "UX/UI"},
    AgentRole.QA: {"color": "#14b8a6", "emoji": "🔍", "name": "QA"},
    AgentRole.DEVOPS: {"color": "#f97316", "emoji": "🚀", "name": "DevOps"},
}


# ============================================================================
# DEVELOPMENT AGENT
# ============================================================================

class DevelopmentAgent:
    """Agente de desarrollo con herramientas."""

    def __init__(self, role: AgentRole, tool_executor: ToolExecutor):
        self.role = role
        self.tool_executor = tool_executor
        self.style = AGENT_STYLES[role]
        self.history: List[Dict] = []

        # Build system prompt
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            role_name=self.style["name"],
            role_specific=ROLE_PROMPTS[role]
        )

    @property
    def name(self) -> str:
        return f"THAU-{self.style['name']}"

    def call_ollama(self, messages: List[Dict], stream: bool = True) -> Generator[str, None, None]:
        """Llamar a Ollama API."""
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": stream,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_ctx": 4096,
                    }
                },
                stream=stream,
                timeout=180
            )

            if not response.ok:
                yield f"Error Ollama: {response.status_code}"
                return

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Ollama error: {e}")
            yield f"Error: {str(e)}"

    def parse_tools(self, response: str) -> List[Dict]:
        """Extraer llamadas a herramientas con nuevo formato."""
        tool_calls = []

        # Buscar bloques ```tool ... ```
        pattern = r'```tool\s*\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            lines = match.strip().split('\n')
            if not lines:
                continue

            tool_name = lines[0].strip()
            args = {}

            current_key = None
            current_value = []

            for line in lines[1:]:
                if ':' in line and not current_key == 'content':
                    # Si tenemos un valor anterior, guardarlo
                    if current_key:
                        args[current_key] = '\n'.join(current_value).strip()

                    # Nueva clave
                    parts = line.split(':', 1)
                    current_key = parts[0].strip()
                    current_value = [parts[1].strip()] if len(parts) > 1 and parts[1].strip() else []
                elif current_key:
                    current_value.append(line)

            # Guardar último valor
            if current_key:
                args[current_key] = '\n'.join(current_value).strip()

            if tool_name and args:
                tool_calls.append({"name": tool_name, "args": args})
                logger.info(f"Tool parsed: {tool_name} -> {list(args.keys())}")

        # También buscar formato JSON antiguo como fallback
        json_pattern = r'<tool>\s*(\{.*?\})\s*</tool>'
        json_matches = re.findall(json_pattern, response, re.DOTALL)

        for match in json_matches:
            try:
                tool_call = json.loads(match.strip())
                if "name" in tool_call:
                    tool_calls.append(tool_call)
            except:
                pass

        return tool_calls

    def respond(self, message: str, context: str = "", max_iterations: int = 5) -> Generator[Dict, None, None]:
        """Generar respuesta con ejecución de herramientas."""

        messages = [
            {"role": "system", "content": self.system_prompt + (f"\n\nCONTEXTO:\n{context}" if context else "")},
        ]

        for msg in self.history[-4:]:
            messages.append(msg)

        messages.append({"role": "user", "content": message})

        iteration = 0
        while iteration < max_iterations:
            iteration += 1

            # Enviar estado de "pensando"
            yield {
                "type": "thinking",
                "iteration": iteration,
                "content": f"Iteración {iteration}/{max_iterations}...",
                "agent": self.style
            }

            # Generar respuesta
            full_response = ""
            for chunk in self.call_ollama(messages):
                full_response += chunk
                yield {
                    "type": "stream",
                    "content": chunk,
                    "full_content": full_response,
                    "agent": self.style
                }

            # Buscar herramientas
            tool_calls = self.parse_tools(full_response)

            if tool_calls:
                yield {
                    "type": "tools_found",
                    "count": len(tool_calls),
                    "tools": [tc["name"] for tc in tool_calls],
                    "agent": self.style
                }

                # Ejecutar herramientas
                results = []
                for tc in tool_calls:
                    name = tc.get("name", "")
                    args = tc.get("args", {})

                    yield {
                        "type": "tool_executing",
                        "tool": name,
                        "args": args,
                        "agent": self.style
                    }

                    result = self.tool_executor.execute(name, args)
                    results.append(f"{name}: {result.output if result.success else result.error}")

                    yield {
                        "type": "tool_result",
                        "tool": name,
                        "success": result.success,
                        "output": result.output if result.success else result.error,
                        "agent": self.style
                    }

                # Agregar al contexto y continuar
                messages.append({"role": "assistant", "content": full_response})
                results_str = "\n".join(results)
                messages.append({"role": "user", "content": f"Resultados de herramientas:\n{results_str}\n\nContinúa con el siguiente paso."})
            else:
                # No hay herramientas, terminado
                break

        # Guardar en historial
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": full_response})

        yield {
            "type": "done",
            "files_created": self.tool_executor.files_created.copy(),
            "agent": self.style
        }


# ============================================================================
# TEAM ORCHESTRATOR
# ============================================================================

class DevTeamOrchestrator:
    """Orquestador del equipo de desarrollo."""

    def __init__(self, project_path: str = DEFAULT_PROJECT_DIR):
        self.project_path = Path(project_path)
        self.project_path.mkdir(parents=True, exist_ok=True)
        self.tool_executor = ToolExecutor(str(self.project_path))
        self.agents: Dict[AgentRole, DevelopmentAgent] = {
            role: DevelopmentAgent(role, self.tool_executor)
            for role in AgentRole
        }
        logger.info(f"DevTeamOrchestrator initialized: {self.project_path}")

    def set_project_path(self, path: str):
        """Cambiar directorio del proyecto."""
        self.project_path = Path(path)
        self.project_path.mkdir(parents=True, exist_ok=True)
        self.tool_executor = ToolExecutor(str(self.project_path))
        for agent in self.agents.values():
            agent.tool_executor = self.tool_executor

    def detect_agent(self, message: str) -> AgentRole:
        """Detectar qué agente debe responder."""
        patterns = {
            AgentRole.PMO: r'@(pmo|pm|manager)',
            AgentRole.ARCHITECT: r'@(architect|arquitecto)',
            AgentRole.BACKEND: r'@(backend|back|api)',
            AgentRole.FRONTEND: r'@(frontend|front)',
            AgentRole.UX: r'@(ux|design)',
            AgentRole.QA: r'@(qa|test)',
            AgentRole.DEVOPS: r'@(devops|ops)',
        }
        for role, pattern in patterns.items():
            if re.search(pattern, message.lower()):
                return role
        return AgentRole.PMO

    def get_project_context(self) -> str:
        """Obtener contexto del proyecto."""
        files = self.tool_executor.list_files(".")
        return f"PROYECTO EN: {self.project_path}\nARCHIVOS:\n{files.output}"

    def send_message(self, message: str, to_agent: Optional[str] = None) -> Generator[Dict, None, None]:
        """Enviar mensaje y obtener respuesta."""
        if to_agent:
            try:
                role = AgentRole(to_agent)
            except ValueError:
                role = self.detect_agent(message)
        else:
            role = self.detect_agent(message)

        agent = self.agents[role]
        context = self.get_project_context()

        yield {"type": "start", "agent": agent.style, "role": role.value}

        for chunk in agent.respond(message, context):
            yield chunk

    def start_project(self, description: str, name: str = "") -> Generator[Dict, None, None]:
        """Iniciar un nuevo proyecto."""
        if name:
            project_dir = self.project_path / name
            project_dir.mkdir(parents=True, exist_ok=True)
            self.set_project_path(str(project_dir))

        prompt = f"""NUEVO PROYECTO: {description}

INSTRUCCIONES:
1. Analiza los requerimientos
2. USA LAS HERRAMIENTAS para crear la estructura de carpetas
3. USA LAS HERRAMIENTAS para crear los archivos del proyecto
4. Verifica con list_files que se crearon correctamente

Directorio: {self.project_path}

IMPORTANTE: Debes usar ```tool para crear archivos REALES. Comienza ahora."""

        for chunk in self.send_message(prompt, "pmo"):
            yield chunk

    def get_files(self) -> List[Dict]:
        """Obtener lista de archivos del proyecto."""
        result = self.tool_executor.list_files(".")
        if not result.success:
            return []

        files = []
        for line in result.output.split("\n"):
            if "📄" in line:
                parts = line.split(" ", 1)
                if len(parts) >= 2:
                    path = parts[1].split(" ")[0]
                    files.append({"path": path, "type": "file"})
            elif "📁" in line:
                parts = line.split(" ", 1)
                if len(parts) >= 2:
                    path = parts[1].rstrip("/").split(" ")[0]
                    files.append({"path": path, "type": "directory"})
        return files

    def read_file(self, path: str) -> Optional[str]:
        """Leer contenido de un archivo."""
        result = self.tool_executor.read_file(path)
        return result.output if result.success else None

    def clear(self):
        """Limpiar estado."""
        for agent in self.agents.values():
            agent.history = []
        self.tool_executor.files_created = []


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="THAU Development Studio v2")
orchestrator = DevTeamOrchestrator()


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_TEMPLATE


@app.get("/api/agents")
async def get_agents():
    return [{"role": role.value, "name": f"THAU-{style['name']}",
             "emoji": style["emoji"], "color": style["color"]}
            for role, style in AGENT_STYLES.items()]


@app.get("/api/files")
async def get_files():
    return orchestrator.get_files()


@app.get("/api/file/{path:path}")
async def get_file(path: str):
    content = orchestrator.read_file(path)
    if content is None:
        raise HTTPException(404, "File not found")
    return {"path": path, "content": content}


@app.post("/api/project/path")
async def set_project_path(data: dict):
    path = data.get("path", "")
    if path:
        orchestrator.set_project_path(path)
    return {"path": str(orchestrator.project_path)}


@app.get("/api/project/path")
async def get_project_path():
    return {"path": str(orchestrator.project_path)}


@app.post("/api/clear")
async def clear():
    orchestrator.clear()
    return {"status": "ok"}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket para chat en tiempo real."""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            agent = data.get("agent")
            is_project = data.get("is_project", False)
            project_name = data.get("project_name", "")

            result_queue = queue.Queue()

            def run_generator():
                try:
                    if is_project:
                        gen = orchestrator.start_project(message, project_name)
                    else:
                        gen = orchestrator.send_message(message, agent)

                    for chunk in gen:
                        result_queue.put(("data", chunk))
                    result_queue.put(("done", None))
                except Exception as e:
                    logger.error(f"Generator error: {e}")
                    result_queue.put(("error", str(e)))

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(run_generator)

            finished = False
            while not finished:
                try:
                    await asyncio.sleep(0.02)

                    while not result_queue.empty():
                        msg_type, chunk_data = result_queue.get_nowait()

                        if msg_type == "done":
                            finished = True
                            break
                        elif msg_type == "error":
                            await websocket.send_json({"type": "error", "content": chunk_data})
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

                except Exception as e:
                    logger.error(f"Queue error: {e}")
                    finished = True

            executor.shutdown(wait=False)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# ============================================================================
# HTML TEMPLATE - Con indicador de pensamiento y expansión
# ============================================================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>THAU Development Studio</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg-primary: #0a0a0a;
            --bg-secondary: #141414;
            --bg-tertiary: #1a1a1a;
            --bg-hover: #252525;
            --text-primary: #ffffff;
            --text-secondary: #a0a0a0;
            --text-muted: #666666;
            --border-color: #2a2a2a;
            --accent: #6366f1;
            --accent-hover: #818cf8;
            --success: #10b981;
            --error: #ef4444;
            --warning: #f59e0b;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            overflow: hidden;
        }

        .app {
            display: grid;
            grid-template-columns: 260px 1fr 300px;
            height: 100vh;
        }

        /* Sidebar */
        .sidebar {
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
        }

        .sidebar-header {
            padding: 20px;
            border-bottom: 1px solid var(--border-color);
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--accent), #a855f7);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }

        .logo h1 { font-size: 18px; }
        .logo p { font-size: 11px; color: var(--text-secondary); }

        .agents-section { padding: 16px; flex: 1; overflow-y: auto; }
        .section-title { font-size: 11px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 12px; }

        .agent-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 12px;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.15s;
        }
        .agent-item:hover { background: var(--bg-hover); }
        .agent-item.active { background: var(--bg-tertiary); }
        .agent-avatar { width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 16px; }
        .agent-name { font-size: 13px; font-weight: 500; }
        .agent-role { font-size: 11px; color: var(--text-secondary); }

        .project-section { padding: 16px; border-top: 1px solid var(--border-color); }
        .project-path {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 12px;
            background: var(--bg-tertiary);
            border-radius: 8px;
            font-size: 12px;
            color: var(--text-secondary);
            cursor: pointer;
        }

        /* Main */
        .main { display: flex; flex-direction: column; height: 100vh; }

        .chat-header {
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .btn {
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            border: none;
            transition: all 0.15s;
        }
        .btn-primary { background: var(--accent); color: white; }
        .btn-primary:hover { background: var(--accent-hover); }
        .btn-secondary { background: var(--bg-tertiary); color: var(--text-primary); border: 1px solid var(--border-color); }

        .messages { flex: 1; overflow-y: auto; padding: 24px; }

        .message {
            display: flex;
            gap: 16px;
            margin-bottom: 24px;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message-avatar {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
        }

        .message-content { flex: 1; min-width: 0; }
        .message-header { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
        .message-author { font-size: 14px; font-weight: 600; }
        .message-time { font-size: 11px; color: var(--text-muted); }
        .message-text { font-size: 14px; line-height: 1.6; }
        .message-text pre { background: var(--bg-tertiary); border-radius: 8px; padding: 16px; overflow-x: auto; margin: 12px 0; }
        .message-text code { font-family: 'JetBrains Mono', monospace; font-size: 13px; }

        /* Thinking indicator */
        .thinking-container {
            background: var(--bg-tertiary);
            border-radius: 12px;
            padding: 16px;
            margin: 12px 0;
        }

        .thinking-header {
            display: flex;
            align-items: center;
            gap: 12px;
            cursor: pointer;
            user-select: none;
        }

        .thinking-spinner {
            width: 24px;
            height: 24px;
            border: 3px solid var(--border-color);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .thinking-title {
            font-weight: 500;
            color: var(--text-secondary);
        }

        .thinking-toggle {
            margin-left: auto;
            color: var(--text-muted);
            font-size: 12px;
        }

        .thinking-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: var(--text-muted);
            white-space: pre-wrap;
        }

        .thinking-container.expanded .thinking-content {
            max-height: 400px;
            overflow-y: auto;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid var(--border-color);
        }

        /* Tool execution */
        .tool-execution {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border-left: 3px solid var(--accent);
            border-radius: 0 8px 8px 0;
            padding: 12px 16px;
            margin: 8px 0;
            font-size: 13px;
        }

        .tool-name {
            color: var(--accent);
            font-weight: 600;
        }

        .tool-result {
            margin-top: 8px;
            padding: 8px 12px;
            background: rgba(0,0,0,0.2);
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
        }

        .tool-result.success { border-left: 2px solid var(--success); }
        .tool-result.error { border-left: 2px solid var(--error); }

        /* Files created badge */
        .files-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: var(--success);
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
            margin-top: 12px;
        }

        /* Input */
        .input-area { padding: 16px 24px 24px; border-top: 1px solid var(--border-color); }
        .quick-actions { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
        .quick-action {
            padding: 6px 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            font-size: 12px;
            color: var(--text-secondary);
            cursor: pointer;
        }
        .quick-action:hover { background: var(--bg-hover); border-color: var(--accent); }

        .input-container { display: flex; gap: 12px; align-items: flex-end; }
        .input-wrapper {
            flex: 1;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 12px 16px;
        }
        .input-wrapper:focus-within { border-color: var(--accent); }
        .input-wrapper textarea {
            width: 100%;
            background: transparent;
            border: none;
            outline: none;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 14px;
            resize: none;
            max-height: 150px;
        }

        .send-btn {
            width: 44px;
            height: 44px;
            background: var(--accent);
            border: none;
            border-radius: 10px;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .send-btn:hover { background: var(--accent-hover); }
        .send-btn:disabled { background: var(--bg-tertiary); cursor: not-allowed; }

        /* Files panel */
        .files-panel {
            background: var(--bg-secondary);
            border-left: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
        }
        .files-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .files-header h3 { font-size: 14px; font-weight: 600; }
        .files-list { flex: 1; overflow-y: auto; padding: 12px; }
        .file-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
        }
        .file-item:hover { background: var(--bg-hover); }
        .file-item.active { background: var(--bg-tertiary); }
        .file-item.new { animation: pulse 2s ease-in-out; }

        @keyframes pulse {
            0%, 100% { background: transparent; }
            50% { background: rgba(99, 102, 241, 0.2); }
        }

        .file-preview {
            border-top: 1px solid var(--border-color);
            padding: 16px;
            max-height: 300px;
            overflow-y: auto;
        }
        .file-preview-content {
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            white-space: pre-wrap;
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
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, var(--accent), #a855f7);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 40px;
            margin-bottom: 24px;
        }
        .welcome h2 { font-size: 24px; margin-bottom: 8px; }
        .welcome p { color: var(--text-secondary); max-width: 400px; margin-bottom: 32px; }
        .welcome-actions { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; }

        /* Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.7);
            z-index: 100;
            align-items: center;
            justify-content: center;
        }
        .modal-overlay.active { display: flex; }
        .modal {
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 24px;
            width: 500px;
            max-width: 90%;
        }
        .modal h2 { font-size: 18px; margin-bottom: 16px; }
        .modal-input {
            width: 100%;
            padding: 12px 16px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 14px;
            margin-bottom: 12px;
        }
        .modal-input:focus { outline: none; border-color: var(--accent); }
        .modal-actions { display: flex; gap: 12px; justify-content: flex-end; margin-top: 16px; }

        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
    </style>
</head>
<body>
    <div class="app">
        <aside class="sidebar">
            <div class="sidebar-header">
                <div class="logo">
                    <div class="logo-icon">🤖</div>
                    <div>
                        <h1>THAU Studio</h1>
                        <p>Development Team</p>
                    </div>
                </div>
            </div>
            <div class="agents-section">
                <div class="section-title">Equipo</div>
                <div id="agentList"></div>
            </div>
            <div class="project-section">
                <div class="section-title">Proyecto</div>
                <div class="project-path" onclick="changeProjectPath()">
                    📁 <span id="currentPath">~/thau_projects</span>
                </div>
            </div>
        </aside>

        <main class="main">
            <header class="chat-header">
                <span style="font-size: 16px; font-weight: 600;">THAU Development Studio</span>
                <div style="display: flex; gap: 8px;">
                    <button class="btn btn-secondary" onclick="clearChat()">Limpiar</button>
                    <button class="btn btn-primary" onclick="showNewProjectModal()">+ Nuevo Proyecto</button>
                </div>
            </header>

            <div class="messages" id="messages">
                <div class="welcome" id="welcome">
                    <div class="welcome-icon">🚀</div>
                    <h2>THAU Development Studio</h2>
                    <p>Tu equipo de desarrollo con IA. Describe tu proyecto y THAU creará los archivos reales.</p>
                    <div class="welcome-actions">
                        <button class="btn btn-primary" onclick="showNewProjectModal()">Crear Proyecto</button>
                        <button class="btn btn-secondary" onclick="startQuickProject('web')">App Web</button>
                        <button class="btn btn-secondary" onclick="startQuickProject('api')">API REST</button>
                    </div>
                </div>
            </div>

            <div class="input-area">
                <div class="quick-actions">
                    <span class="quick-action" onclick="insertMention('@pmo')">@PMO</span>
                    <span class="quick-action" onclick="insertMention('@architect')">@Arquitecto</span>
                    <span class="quick-action" onclick="insertMention('@backend')">@Backend</span>
                    <span class="quick-action" onclick="insertMention('@frontend')">@Frontend</span>
                </div>
                <div class="input-container">
                    <div class="input-wrapper">
                        <textarea id="messageInput" placeholder="Describe lo que quieres desarrollar..." rows="1" onkeydown="handleKeydown(event)"></textarea>
                    </div>
                    <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                        </svg>
                    </button>
                </div>
            </div>
        </main>

        <aside class="files-panel">
            <div class="files-header">
                <h3>📁 Archivos</h3>
                <button style="background:transparent;border:none;color:var(--text-secondary);cursor:pointer;" onclick="refreshFiles()">🔄</button>
            </div>
            <div class="files-list" id="filesList">
                <div style="padding:20px;text-align:center;color:var(--text-muted);">Inicia un proyecto</div>
            </div>
            <div class="file-preview" id="filePreview" style="display:none;">
                <div style="font-size:12px;font-weight:600;margin-bottom:12px;" id="previewTitle"></div>
                <pre class="file-preview-content" id="previewContent"></pre>
            </div>
        </aside>
    </div>

    <div class="modal-overlay" id="newProjectModal">
        <div class="modal">
            <h2>🚀 Nuevo Proyecto</h2>
            <input type="text" class="modal-input" id="projectName" placeholder="Nombre (ej: mi-app)">
            <textarea class="modal-input" id="projectDescription" placeholder="Describe tu proyecto..." rows="4"></textarea>
            <div class="modal-actions">
                <button class="btn btn-secondary" onclick="closeModal()">Cancelar</button>
                <button class="btn btn-primary" onclick="createProject()">Crear</button>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let agents = [];
        let currentMessageDiv = null;
        let thinkingDiv = null;
        let thinkingContent = "";
        let isExpanded = false;

        document.addEventListener('DOMContentLoaded', () => {
            loadAgents();
            loadProjectPath();
            connectWebSocket();
        });

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);
            ws.onopen = () => console.log('Connected');
            ws.onclose = () => setTimeout(connectWebSocket, 2000);
            ws.onmessage = handleMessage;
        }

        function handleMessage(event) {
            const data = JSON.parse(event.data);
            console.log('Message:', data.type, data);

            switch(data.type) {
                case 'start':
                    hideWelcome();
                    currentMessageDiv = createMessageDiv(data.agent);
                    thinkingContent = "";
                    break;

                case 'thinking':
                    showThinking(data.content);
                    break;

                case 'stream':
                    updateMessageContent(data.full_content);
                    thinkingContent += data.content;
                    updateThinkingContent();
                    break;

                case 'tools_found':
                    addToolsFound(data.count, data.tools);
                    break;

                case 'tool_executing':
                    addToolExecuting(data.tool);
                    break;

                case 'tool_result':
                    addToolResult(data.tool, data.success, data.output);
                    break;

                case 'done':
                    hideThinking();
                    if (data.files_created && data.files_created.length > 0) {
                        addFilesCreated(data.files_created);
                    }
                    refreshFiles();
                    document.getElementById('sendBtn').disabled = false;
                    break;
            }
        }

        function createMessageDiv(agent) {
            const messages = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = 'message';
            div.innerHTML = `
                <div class="message-avatar" style="background:${agent.color}20;color:${agent.color}">${agent.emoji}</div>
                <div class="message-content">
                    <div class="message-header">
                        <span class="message-author">THAU-${agent.name}</span>
                        <span class="message-time">${new Date().toLocaleTimeString('es', {hour:'2-digit',minute:'2-digit'})}</span>
                    </div>
                    <div class="message-text"></div>
                </div>
            `;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
            return div;
        }

        function showThinking(text) {
            if (!currentMessageDiv) return;

            const content = currentMessageDiv.querySelector('.message-content');
            if (!thinkingDiv) {
                thinkingDiv = document.createElement('div');
                thinkingDiv.className = 'thinking-container';
                thinkingDiv.innerHTML = `
                    <div class="thinking-header" onclick="toggleThinking()">
                        <div class="thinking-spinner"></div>
                        <span class="thinking-title">Pensando...</span>
                        <span class="thinking-toggle">▶ Click para ver</span>
                    </div>
                    <div class="thinking-content"></div>
                `;
                content.appendChild(thinkingDiv);
            }
            thinkingDiv.querySelector('.thinking-title').textContent = text;
        }

        function toggleThinking() {
            if (!thinkingDiv) return;
            isExpanded = !isExpanded;
            thinkingDiv.classList.toggle('expanded', isExpanded);
            thinkingDiv.querySelector('.thinking-toggle').textContent = isExpanded ? '▼ Click para ocultar' : '▶ Click para ver';
        }

        function updateThinkingContent() {
            if (!thinkingDiv) return;
            const contentDiv = thinkingDiv.querySelector('.thinking-content');
            contentDiv.textContent = thinkingContent;
            if (isExpanded) {
                contentDiv.scrollTop = contentDiv.scrollHeight;
            }
        }

        function hideThinking() {
            if (thinkingDiv) {
                const spinner = thinkingDiv.querySelector('.thinking-spinner');
                if (spinner) {
                    spinner.style.animation = 'none';
                    spinner.style.borderTopColor = 'var(--success)';
                }
                thinkingDiv.querySelector('.thinking-title').textContent = 'Completado';
            }
            thinkingDiv = null;
        }

        function updateMessageContent(content) {
            if (!currentMessageDiv) return;
            const textDiv = currentMessageDiv.querySelector('.message-text');
            try {
                textDiv.innerHTML = marked.parse(content);
                textDiv.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
            } catch(e) {
                textDiv.textContent = content;
            }
            document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
        }

        function addToolsFound(count, tools) {
            if (!currentMessageDiv) return;
            const content = currentMessageDiv.querySelector('.message-content');
            const div = document.createElement('div');
            div.className = 'tool-execution';
            div.innerHTML = `🔧 Encontradas <strong>${count}</strong> herramientas: ${tools.join(', ')}`;
            content.appendChild(div);
        }

        function addToolExecuting(tool) {
            if (!currentMessageDiv) return;
            const content = currentMessageDiv.querySelector('.message-content');
            const div = document.createElement('div');
            div.className = 'tool-execution';
            div.innerHTML = `⚡ Ejecutando: <span class="tool-name">${tool}</span>`;
            content.appendChild(div);
        }

        function addToolResult(tool, success, output) {
            if (!currentMessageDiv) return;
            const content = currentMessageDiv.querySelector('.message-content');
            const div = document.createElement('div');
            div.className = `tool-result ${success ? 'success' : 'error'}`;
            div.innerHTML = `<strong>${tool}:</strong> ${output}`;
            content.appendChild(div);
        }

        function addFilesCreated(files) {
            if (!currentMessageDiv) return;
            const content = currentMessageDiv.querySelector('.message-content');
            const div = document.createElement('div');
            div.className = 'files-badge';
            div.innerHTML = `📁 ${files.length} archivo(s) creado(s)`;
            content.appendChild(div);
        }

        async function loadAgents() {
            const response = await fetch('/api/agents');
            agents = await response.json();
            document.getElementById('agentList').innerHTML = agents.map(a => `
                <div class="agent-item" onclick="selectAgent('${a.role}')">
                    <div class="agent-avatar" style="background:${a.color}20;color:${a.color}">${a.emoji}</div>
                    <div>
                        <div class="agent-name">${a.name}</div>
                        <div class="agent-role">${a.role}</div>
                    </div>
                </div>
            `).join('');
        }

        async function loadProjectPath() {
            const response = await fetch('/api/project/path');
            const data = await response.json();
            document.getElementById('currentPath').textContent = data.path.replace(/\\/Users\\/[^\\/]+/, '~');
        }

        async function refreshFiles() {
            const response = await fetch('/api/files');
            const files = await response.json();
            const list = document.getElementById('filesList');
            if (files.length === 0) {
                list.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text-muted);">No hay archivos</div>';
                return;
            }
            list.innerHTML = files.map(f => `
                <div class="file-item new" onclick="previewFile('${f.path}')">
                    <span>${f.type === 'directory' ? '📁' : '📄'}</span>
                    <span>${f.path}</span>
                </div>
            `).join('');
        }

        async function previewFile(path) {
            const response = await fetch(`/api/file/${encodeURIComponent(path)}`);
            if (!response.ok) return;
            const data = await response.json();
            document.getElementById('previewTitle').textContent = path;
            document.getElementById('previewContent').textContent = data.content;
            document.getElementById('filePreview').style.display = 'block';
        }

        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message || !ws) return;

            hideWelcome();
            addUserMessage(message);
            ws.send(JSON.stringify({ message }));
            input.value = '';
            document.getElementById('sendBtn').disabled = true;
        }

        function addUserMessage(text) {
            const messages = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = 'message';
            div.innerHTML = `
                <div class="message-avatar" style="background:#6366f120;color:#6366f1">👤</div>
                <div class="message-content">
                    <div class="message-header">
                        <span class="message-author">Tú</span>
                    </div>
                    <div class="message-text">${text}</div>
                </div>
            `;
            messages.appendChild(div);
        }

        function hideWelcome() {
            const w = document.getElementById('welcome');
            if (w) w.style.display = 'none';
        }

        function insertMention(m) {
            const input = document.getElementById('messageInput');
            input.value += m + ' ';
            input.focus();
        }

        function handleKeydown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        function showNewProjectModal() {
            document.getElementById('newProjectModal').classList.add('active');
        }

        function closeModal() {
            document.getElementById('newProjectModal').classList.remove('active');
        }

        function createProject() {
            const name = document.getElementById('projectName').value.trim();
            const description = document.getElementById('projectDescription').value.trim();
            if (!description) return alert('Describe tu proyecto');

            hideWelcome();
            addUserMessage(description);
            ws.send(JSON.stringify({ message: description, is_project: true, project_name: name || 'proyecto' }));
            closeModal();
            document.getElementById('sendBtn').disabled = true;
        }

        function startQuickProject(type) {
            const templates = {
                'web': 'Crea una aplicación web con Flask que tenga: página de inicio con diseño moderno, página about, y formulario de contacto. Usa Bootstrap 5.',
                'api': 'Crea una API REST con FastAPI que tenga: endpoints CRUD para usuarios, documentación automática, y tests básicos.'
            };
            document.getElementById('projectDescription').value = templates[type];
            showNewProjectModal();
        }

        function changeProjectPath() {
            const path = prompt('Nueva ruta del proyecto:', document.getElementById('currentPath').textContent);
            if (path) {
                fetch('/api/project/path', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({path})
                }).then(() => {
                    loadProjectPath();
                    refreshFiles();
                });
            }
        }

        async function clearChat() {
            await fetch('/api/clear', {method:'POST'});
            document.getElementById('messages').innerHTML = `
                <div class="welcome" id="welcome">
                    <div class="welcome-icon">🚀</div>
                    <h2>THAU Development Studio</h2>
                    <p>Tu equipo de desarrollo con IA.</p>
                    <div class="welcome-actions">
                        <button class="btn btn-primary" onclick="showNewProjectModal()">Crear Proyecto</button>
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
# MAIN
# ============================================================================

if __name__ == "__main__":
    Path(DEFAULT_PROJECT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║              THAU Development Studio v2                        ║
║                                                               ║
║  🤖 Equipo de agentes con creación REAL de archivos           ║
║                                                               ║
║  URL: http://localhost:{PORT}                                   ║
║  Proyecto: {DEFAULT_PROJECT_DIR}
╚═══════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
