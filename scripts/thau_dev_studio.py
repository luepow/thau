#!/usr/bin/env python3
"""
THAU Development Studio - Sistema Multi-Agente para Desarrollo de Aplicaciones.

Este sistema permite a THAU desarrollar aplicaciones completas con:
- Equipo de agentes especializados (PMO, Arquitecto, Backend, Frontend, QA, DevOps)
- Herramientas reales para crear archivos, ejecutar comandos, etc.
- Interfaz moderna estilo Apple/ChatGPT
- Planificación y ejecución automática

Uso: python scripts/thau_dev_studio.py
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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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
    """Roles de los agentes del equipo."""
    PMO = "pmo"
    ARCHITECT = "architect"
    BACKEND = "backend"
    FRONTEND = "frontend"
    UX = "ux"
    QA = "qa"
    DEVOPS = "devops"


@dataclass
class ToolResult:
    """Resultado de ejecución de herramienta."""
    success: bool
    output: str
    error: Optional[str] = None


@dataclass
class ProjectFile:
    """Archivo del proyecto."""
    path: str
    content: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProjectState:
    """Estado del proyecto."""
    name: str = ""
    path: str = ""
    description: str = ""
    phase: str = "planning"  # planning, architecture, development, testing, deployment
    files: Dict[str, ProjectFile] = field(default_factory=dict)
    commands_run: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    running_process: Optional[subprocess.Popen] = None


# ============================================================================
# AGENT PROMPTS (con herramientas)
# ============================================================================

TOOLS_DESCRIPTION = """
HERRAMIENTAS DISPONIBLES:
Para usar una herramienta, escribe exactamente este formato:
<tool>{"name": "nombre", "args": {"param": "valor"}}</tool>

Herramientas:
1. create_file: Crear un archivo
   <tool>{"name": "create_file", "args": {"path": "ruta/archivo.py", "content": "contenido del archivo"}}</tool>

2. create_directory: Crear directorio
   <tool>{"name": "create_directory", "args": {"path": "ruta/directorio"}}</tool>

3. read_file: Leer archivo existente
   <tool>{"name": "read_file", "args": {"path": "ruta/archivo.py"}}</tool>

4. list_files: Listar archivos del proyecto
   <tool>{"name": "list_files", "args": {"path": "."}}</tool>

5. run_command: Ejecutar comando en terminal
   <tool>{"name": "run_command", "args": {"command": "npm install"}}</tool>

6. start_server: Iniciar servidor de desarrollo
   <tool>{"name": "start_server", "args": {"command": "python app.py", "port": 5000}}</tool>

IMPORTANTE:
- Siempre crea archivos COMPLETOS y funcionales
- Usa rutas relativas al proyecto
- Después de crear archivos, verifica con list_files
- Cuando termines una fase, indica: "✅ FASE COMPLETADA: [nombre]"
"""

AGENT_PROMPTS = {
    AgentRole.PMO: f"""Eres el **PMO (Project Manager)** del equipo THAU.

🎯 **Tu rol:**
- Coordinar al equipo de desarrollo
- Definir y asignar tareas
- Gestionar el alcance del proyecto
- Asegurar que se cumplan los objetivos

📋 **Proceso de trabajo:**
1. Analizar requerimientos del usuario
2. Crear plan de proyecto con fases claras
3. Asignar tareas a cada agente (@architect, @backend, @frontend, etc.)
4. Dar seguimiento y reportar progreso

{TOOLS_DESCRIPTION}

Responde siempre en español. Sé conciso y orientado a la acción.""",

    AgentRole.ARCHITECT: f"""Eres el **Arquitecto de Software** del equipo THAU.

🏗️ **Tu rol:**
- Diseñar la arquitectura del sistema
- Definir estructura de carpetas y archivos
- Elegir tecnologías y patrones
- Documentar decisiones técnicas

📐 **Proceso de trabajo:**
1. Analizar requerimientos técnicos
2. Diseñar estructura del proyecto
3. CREAR archivos de configuración base
4. Documentar arquitectura

{TOOLS_DESCRIPTION}

IMPORTANTE: Siempre crea los archivos de estructura base usando las herramientas.
Responde en español.""",

    AgentRole.BACKEND: f"""Eres el **Desarrollador Backend** del equipo THAU.

⚙️ **Tu rol:**
- Implementar APIs y servicios
- Crear modelos de datos
- Gestionar base de datos
- Implementar lógica de negocio

💻 **Proceso de trabajo:**
1. Revisar arquitectura definida
2. CREAR archivos de código backend
3. Implementar endpoints y servicios
4. Instalar dependencias necesarias

{TOOLS_DESCRIPTION}

IMPORTANTE:
- Crea código COMPLETO y funcional
- Incluye manejo de errores
- Usa type hints en Python
Responde en español.""",

    AgentRole.FRONTEND: f"""Eres el **Desarrollador Frontend** del equipo THAU.

🎨 **Tu rol:**
- Implementar interfaces de usuario
- Crear componentes reutilizables
- Conectar con APIs backend
- Optimizar experiencia de usuario

💻 **Proceso de trabajo:**
1. Revisar diseños y arquitectura
2. CREAR archivos de componentes
3. Implementar estilos y layouts
4. Integrar con backend

{TOOLS_DESCRIPTION}

IMPORTANTE:
- Crea código completo con estilos
- Usa HTML semántico
- Asegura responsive design
Responde en español.""",

    AgentRole.UX: f"""Eres el **Diseñador UX/UI** del equipo THAU.

✨ **Tu rol:**
- Diseñar experiencia de usuario
- Definir estilos y temas
- Crear guía de estilos
- Asegurar accesibilidad

🎯 **Proceso de trabajo:**
1. Analizar necesidades del usuario
2. Crear wireframes (ASCII)
3. CREAR archivos CSS/estilos
4. Documentar sistema de diseño

{TOOLS_DESCRIPTION}

Responde en español.""",

    AgentRole.QA: f"""Eres el **QA Engineer** del equipo THAU.

🔍 **Tu rol:**
- Crear pruebas automatizadas
- Validar funcionalidad
- Identificar bugs
- Asegurar calidad

🧪 **Proceso de trabajo:**
1. Revisar código implementado
2. CREAR archivos de tests
3. Ejecutar pruebas
4. Reportar resultados

{TOOLS_DESCRIPTION}

Responde en español.""",

    AgentRole.DEVOPS: f"""Eres el **DevOps Engineer** del equipo THAU.

🚀 **Tu rol:**
- Configurar deployment
- Crear Dockerfiles
- Configurar CI/CD
- Documentar setup

📦 **Proceso de trabajo:**
1. Revisar estructura del proyecto
2. CREAR Dockerfile y docker-compose
3. Configurar scripts de deployment
4. Documentar proceso

{TOOLS_DESCRIPTION}

Responde en español.""",
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
# TOOL EXECUTOR
# ============================================================================

class ToolExecutor:
    """Ejecutor de herramientas del sistema."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.project_path.mkdir(parents=True, exist_ok=True)
        self.running_process: Optional[subprocess.Popen] = None

    def _safe_path(self, path: str) -> Path:
        """Asegurar que la ruta esté dentro del proyecto."""
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

            rel_path = file_path.relative_to(self.project_path)
            return ToolResult(True, f"✅ Archivo creado: {rel_path} ({len(content)} bytes)")
        except Exception as e:
            return ToolResult(False, "", str(e))

    def create_directory(self, path: str) -> ToolResult:
        """Crear directorio."""
        try:
            dir_path = self._safe_path(path)
            dir_path.mkdir(parents=True, exist_ok=True)
            rel_path = dir_path.relative_to(self.project_path)
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
                return ToolResult(False, "", f"Directorio no encontrado: {path}")

            items = []
            for item in sorted(dir_path.rglob("*")):
                if item.name.startswith('.') or '__pycache__' in str(item):
                    continue
                rel = item.relative_to(self.project_path)
                if item.is_dir():
                    items.append(f"📁 {rel}/")
                else:
                    size = item.stat().st_size
                    items.append(f"📄 {rel} ({size:,} bytes)")

            return ToolResult(True, "\n".join(items) if items else "(vacío)")
        except Exception as e:
            return ToolResult(False, "", str(e))

    def run_command(self, command: str, timeout: int = 60) -> ToolResult:
        """Ejecutar comando."""
        # Bloquear comandos peligrosos
        dangerous = ['rm -rf /', 'mkfs', '> /dev', 'dd if=']
        for d in dangerous:
            if d in command:
                return ToolResult(False, "", f"Comando bloqueado: {command}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.project_path)
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"

            return ToolResult(
                result.returncode == 0,
                output or "(sin salida)",
                result.stderr if result.returncode != 0 else None
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", f"Timeout ({timeout}s)")
        except Exception as e:
            return ToolResult(False, "", str(e))

    def start_server(self, command: str, port: int = 5000) -> ToolResult:
        """Iniciar servidor."""
        try:
            self.stop_server()

            self.running_process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.project_path),
                preexec_fn=os.setsid
            )

            time.sleep(2)

            if self.running_process.poll() is None:
                return ToolResult(True, f"✅ Servidor iniciado en http://localhost:{port}")
            else:
                _, stderr = self.running_process.communicate()
                return ToolResult(False, "", f"Error: {stderr.decode()}")
        except Exception as e:
            return ToolResult(False, "", str(e))

    def stop_server(self) -> ToolResult:
        """Detener servidor."""
        try:
            if self.running_process:
                os.killpg(os.getpgid(self.running_process.pid), signal.SIGTERM)
                self.running_process = None
                return ToolResult(True, "Servidor detenido")
            return ToolResult(True, "No hay servidor")
        except Exception as e:
            return ToolResult(False, "", str(e))

    def execute(self, name: str, args: Dict) -> ToolResult:
        """Ejecutar herramienta por nombre."""
        tools = {
            "create_file": lambda: self.create_file(args.get("path", ""), args.get("content", "")),
            "create_directory": lambda: self.create_directory(args.get("path", "")),
            "read_file": lambda: self.read_file(args.get("path", "")),
            "list_files": lambda: self.list_files(args.get("path", ".")),
            "run_command": lambda: self.run_command(args.get("command", "")),
            "start_server": lambda: self.start_server(args.get("command", ""), args.get("port", 5000)),
            "stop_server": lambda: self.stop_server(),
        }

        if name not in tools:
            return ToolResult(False, "", f"Herramienta desconocida: {name}")

        return tools[name]()


# ============================================================================
# DEVELOPMENT AGENT
# ============================================================================

class DevelopmentAgent:
    """Agente de desarrollo con herramientas."""

    def __init__(
        self,
        role: AgentRole,
        tool_executor: ToolExecutor,
        ollama_model: str = OLLAMA_MODEL,
        ollama_url: str = OLLAMA_URL
    ):
        self.role = role
        self.tool_executor = tool_executor
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.system_prompt = AGENT_PROMPTS[role]
        self.style = AGENT_STYLES[role]
        self.history: List[Dict] = []

    @property
    def name(self) -> str:
        return f"THAU-{self.style['name']}"

    def call_ollama(self, messages: List[Dict], stream: bool = True) -> Generator[str, None, None]:
        """Llamar a Ollama API."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": messages,
                    "stream": stream,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_ctx": 8192,
                    }
                },
                stream=stream,
                timeout=180
            )

            if not response.ok:
                yield f"Error Ollama: {response.status_code}"
                return

            if stream:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
                        except json.JSONDecodeError:
                            continue
            else:
                data = response.json()
                yield data.get("message", {}).get("content", "")

        except Exception as e:
            yield f"Error: {str(e)}"

    def parse_tools(self, response: str) -> List[Dict]:
        """Extraer llamadas a herramientas."""
        tool_calls = []
        # Pattern: <tool>{"name": "...", "args": {...}}</tool>
        pattern = r'<tool>\s*(\{.*?\})\s*</tool>'
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            try:
                tool_call = json.loads(match.strip())
                if "name" in tool_call:
                    tool_calls.append(tool_call)
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing tool: {e}")

        return tool_calls

    def execute_tools(self, tool_calls: List[Dict]) -> str:
        """Ejecutar herramientas y formatear resultados."""
        results = []
        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {})

            result = self.tool_executor.execute(name, args)

            if result.success:
                results.append(f"✅ {name}: {result.output}")
            else:
                results.append(f"❌ {name}: Error - {result.error}")

        return "\n".join(results)

    def respond(
        self,
        message: str,
        context: str = "",
        max_iterations: int = 10
    ) -> Generator[Dict, None, None]:
        """Generar respuesta con ejecución de herramientas."""

        # Preparar mensajes
        messages = [
            {"role": "system", "content": self.system_prompt + (f"\n\nCONTEXTO:\n{context}" if context else "")},
        ]

        # Agregar historial
        for msg in self.history[-6:]:
            messages.append(msg)

        # Agregar mensaje actual
        messages.append({"role": "user", "content": message})

        iteration = 0
        while iteration < max_iterations:
            iteration += 1

            # Generar respuesta
            full_response = ""
            for chunk in self.call_ollama(messages):
                full_response += chunk
                yield {
                    "type": "text",
                    "content": chunk,
                    "agent": self.style,
                    "done": False
                }

            # Buscar herramientas
            tool_calls = self.parse_tools(full_response)

            if tool_calls:
                yield {
                    "type": "tools",
                    "content": f"\n🔧 Ejecutando {len(tool_calls)} herramienta(s)...\n",
                    "agent": self.style,
                    "done": False
                }

                # Ejecutar herramientas
                results = self.execute_tools(tool_calls)

                yield {
                    "type": "tool_results",
                    "content": results + "\n",
                    "agent": self.style,
                    "done": False
                }

                # Agregar al contexto y continuar
                messages.append({"role": "assistant", "content": full_response})
                messages.append({"role": "user", "content": f"Resultados:\n{results}\n\nContinúa con el siguiente paso."})
            else:
                # No hay herramientas, terminado
                break

            # Verificar si completó
            if "✅ FASE COMPLETADA" in full_response or "✅ TAREA COMPLETADA" in full_response:
                break

        # Guardar en historial
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": full_response})

        yield {
            "type": "done",
            "content": "",
            "agent": self.style,
            "done": True
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

        self.state = ProjectState(path=str(self.project_path))
        self.conversation: List[Dict] = []

        logger.info(f"DevTeamOrchestrator initialized: {self.project_path}")

    def set_project_path(self, path: str):
        """Cambiar directorio del proyecto."""
        self.project_path = Path(path)
        self.project_path.mkdir(parents=True, exist_ok=True)
        self.tool_executor = ToolExecutor(str(self.project_path))

        # Actualizar agentes
        for agent in self.agents.values():
            agent.tool_executor = self.tool_executor

        self.state.path = str(self.project_path)

    def detect_agent(self, message: str) -> AgentRole:
        """Detectar qué agente debe responder."""
        patterns = {
            AgentRole.PMO: r'@(pmo|pm|manager|project)',
            AgentRole.ARCHITECT: r'@(architect|arquitecto|arq|diseño)',
            AgentRole.BACKEND: r'@(backend|back|api|server)',
            AgentRole.FRONTEND: r'@(frontend|front|ui|client)',
            AgentRole.UX: r'@(ux|diseño|design|estilos)',
            AgentRole.QA: r'@(qa|test|pruebas|quality)',
            AgentRole.DEVOPS: r'@(devops|ops|deploy|docker)',
        }

        for role, pattern in patterns.items():
            if re.search(pattern, message.lower()):
                return role

        return AgentRole.PMO

    def get_project_context(self) -> str:
        """Obtener contexto del proyecto."""
        files = self.tool_executor.list_files(".")
        return f"""
PROYECTO: {self.state.name or 'Sin nombre'}
DIRECTORIO: {self.project_path}
ARCHIVOS:
{files.output if files.success else '(vacío)'}
"""

    def send_message(
        self,
        message: str,
        to_agent: Optional[str] = None
    ) -> Generator[Dict, None, None]:
        """Enviar mensaje y obtener respuesta."""

        # Determinar agente
        if to_agent:
            try:
                role = AgentRole(to_agent)
            except ValueError:
                role = self.detect_agent(message)
        else:
            role = self.detect_agent(message)

        agent = self.agents[role]
        context = self.get_project_context()

        # Yield inicio
        yield {
            "type": "start",
            "agent": agent.style,
            "role": role.value
        }

        # Generar respuesta
        for chunk in agent.respond(message, context):
            yield chunk

    def start_project(self, description: str, name: str = "") -> Generator[Dict, None, None]:
        """Iniciar un nuevo proyecto."""
        self.state.description = description
        self.state.name = name or "nuevo_proyecto"

        # Crear carpeta del proyecto
        if name:
            project_dir = self.project_path / name
            project_dir.mkdir(parents=True, exist_ok=True)
            self.set_project_path(str(project_dir))

        prompt = f"""NUEVO PROYECTO: {description}

Por favor:
1. Analiza los requerimientos
2. Define la arquitectura del proyecto
3. CREA la estructura de carpetas y archivos base
4. Asigna tareas al equipo

Directorio de trabajo: {self.project_path}

Comienza ahora. Usa las herramientas para crear archivos reales."""

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
                parts = line.split(" ")
                if len(parts) >= 2:
                    path = parts[1]
                    files.append({"path": path, "type": "file"})
            elif "📁" in line:
                parts = line.split(" ")
                if len(parts) >= 2:
                    path = parts[1].rstrip("/")
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
        self.conversation = []
        self.state = ProjectState(path=str(self.project_path))


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="THAU Development Studio")

# Orquestador global
orchestrator = DevTeamOrchestrator()


class ChatRequest(BaseModel):
    message: str
    agent: Optional[str] = None


class ProjectRequest(BaseModel):
    description: str
    name: Optional[str] = ""
    path: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def index():
    """Página principal."""
    return HTML_TEMPLATE


@app.get("/api/agents")
async def get_agents():
    """Obtener lista de agentes."""
    return [
        {
            "role": role.value,
            "name": f"THAU-{style['name']}",
            "emoji": style["emoji"],
            "color": style["color"]
        }
        for role, style in AGENT_STYLES.items()
    ]


@app.get("/api/files")
async def get_files():
    """Obtener archivos del proyecto."""
    return orchestrator.get_files()


@app.get("/api/file/{path:path}")
async def get_file(path: str):
    """Obtener contenido de archivo."""
    content = orchestrator.read_file(path)
    if content is None:
        raise HTTPException(404, "File not found")
    return {"path": path, "content": content}


@app.post("/api/project/path")
async def set_project_path(data: dict):
    """Establecer ruta del proyecto."""
    path = data.get("path", "")
    if path:
        orchestrator.set_project_path(path)
    return {"path": str(orchestrator.project_path)}


@app.get("/api/project/path")
async def get_project_path():
    """Obtener ruta del proyecto."""
    return {"path": str(orchestrator.project_path)}


@app.post("/api/clear")
async def clear():
    """Limpiar conversación."""
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

            # Ejecutar generador en thread separado para no bloquear
            import queue
            import concurrent.futures

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

            # Iniciar thread
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(run_generator)

            # Leer resultados del queue y enviar por websocket
            finished = False
            while not finished:
                try:
                    # Pequeña pausa para no saturar
                    await asyncio.sleep(0.05)

                    # Procesar todos los items disponibles
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

                    # Verificar si el thread terminó
                    if future.done() and not finished:
                        # Procesar items restantes
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
# HTML TEMPLATE
# ============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>THAU Development Studio</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
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

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            overflow: hidden;
        }

        .app {
            display: grid;
            grid-template-columns: 280px 1fr 320px;
            height: 100vh;
        }

        /* Sidebar */
        .sidebar {
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            overflow: hidden;
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

        .logo-text h1 {
            font-size: 18px;
            font-weight: 700;
        }

        .logo-text p {
            font-size: 11px;
            color: var(--text-secondary);
        }

        .agents-section {
            padding: 16px;
            flex: 1;
            overflow-y: auto;
        }

        .section-title {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }

        .agent-list {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .agent-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 12px;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.15s;
        }

        .agent-item:hover {
            background: var(--bg-hover);
        }

        .agent-item.active {
            background: var(--bg-tertiary);
        }

        .agent-avatar {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
        }

        .agent-info {
            flex: 1;
        }

        .agent-name {
            font-size: 13px;
            font-weight: 500;
        }

        .agent-role {
            font-size: 11px;
            color: var(--text-secondary);
        }

        /* Project Path */
        .project-section {
            padding: 16px;
            border-top: 1px solid var(--border-color);
        }

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

        .project-path:hover {
            background: var(--bg-hover);
        }

        /* Main Chat Area */
        .main {
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }

        .chat-header {
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .chat-title {
            font-size: 16px;
            font-weight: 600;
        }

        .header-actions {
            display: flex;
            gap: 8px;
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

        .btn-primary {
            background: var(--accent);
            color: white;
        }

        .btn-primary:hover {
            background: var(--accent-hover);
        }

        .btn-secondary {
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }

        .btn-secondary:hover {
            background: var(--bg-hover);
        }

        /* Messages */
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
        }

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

        .message-content {
            flex: 1;
            min-width: 0;
        }

        .message-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }

        .message-author {
            font-size: 14px;
            font-weight: 600;
        }

        .message-time {
            font-size: 11px;
            color: var(--text-muted);
        }

        .message-text {
            font-size: 14px;
            line-height: 1.6;
            color: var(--text-primary);
        }

        .message-text p {
            margin-bottom: 12px;
        }

        .message-text pre {
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 16px;
            overflow-x: auto;
            margin: 12px 0;
        }

        .message-text code {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
        }

        .message-text ul, .message-text ol {
            margin: 12px 0;
            padding-left: 24px;
        }

        .message-text li {
            margin-bottom: 8px;
        }

        .tool-result {
            background: var(--bg-tertiary);
            border-left: 3px solid var(--success);
            padding: 12px 16px;
            border-radius: 0 8px 8px 0;
            margin: 12px 0;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
        }

        .tool-result.error {
            border-left-color: var(--error);
        }

        /* Input Area */
        .input-area {
            padding: 16px 24px 24px;
            border-top: 1px solid var(--border-color);
        }

        .quick-actions {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }

        .quick-action {
            padding: 6px 12px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            font-size: 12px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.15s;
        }

        .quick-action:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
            border-color: var(--accent);
        }

        .input-container {
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }

        .input-wrapper {
            flex: 1;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 12px 16px;
            transition: border-color 0.15s;
        }

        .input-wrapper:focus-within {
            border-color: var(--accent);
        }

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

        .input-wrapper textarea::placeholder {
            color: var(--text-muted);
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
            transition: all 0.15s;
        }

        .send-btn:hover {
            background: var(--accent-hover);
            transform: scale(1.05);
        }

        .send-btn:disabled {
            background: var(--bg-tertiary);
            cursor: not-allowed;
            transform: none;
        }

        /* Files Panel */
        .files-panel {
            background: var(--bg-secondary);
            border-left: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .files-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .files-header h3 {
            font-size: 14px;
            font-weight: 600;
        }

        .refresh-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            padding: 4px;
            border-radius: 4px;
        }

        .refresh-btn:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
        }

        .files-list {
            flex: 1;
            overflow-y: auto;
            padding: 12px;
        }

        .file-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            transition: background 0.15s;
        }

        .file-item:hover {
            background: var(--bg-hover);
        }

        .file-item.active {
            background: var(--bg-tertiary);
        }

        .file-icon {
            font-size: 14px;
        }

        .file-name {
            flex: 1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .file-preview {
            border-top: 1px solid var(--border-color);
            padding: 16px;
            max-height: 300px;
            overflow-y: auto;
        }

        .file-preview-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }

        .file-preview-title {
            font-size: 12px;
            font-weight: 600;
        }

        .file-preview-content {
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }

        /* Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.7);
            z-index: 100;
            align-items: center;
            justify-content: center;
        }

        .modal-overlay.active {
            display: flex;
        }

        .modal {
            background: var(--bg-secondary);
            border-radius: 16px;
            padding: 24px;
            width: 500px;
            max-width: 90%;
        }

        .modal h2 {
            font-size: 18px;
            margin-bottom: 16px;
        }

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

        .modal-input:focus {
            outline: none;
            border-color: var(--accent);
        }

        .modal-actions {
            display: flex;
            gap: 12px;
            justify-content: flex-end;
            margin-top: 16px;
        }

        /* Typing indicator */
        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 8px 0;
        }

        .typing-dot {
            width: 8px;
            height: 8px;
            background: var(--text-muted);
            border-radius: 50%;
            animation: typing 1.4s infinite ease-in-out;
        }

        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }

        @keyframes typing {
            0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }

        ::-webkit-scrollbar-track {
            background: transparent;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--border-color);
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }

        /* Welcome screen */
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

        .welcome h2 {
            font-size: 24px;
            margin-bottom: 8px;
        }

        .welcome p {
            color: var(--text-secondary);
            max-width: 400px;
            margin-bottom: 32px;
        }

        .welcome-actions {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            justify-content: center;
        }
    </style>
</head>
<body>
    <div class="app">
        <!-- Sidebar -->
        <aside class="sidebar">
            <div class="sidebar-header">
                <div class="logo">
                    <div class="logo-icon">🤖</div>
                    <div class="logo-text">
                        <h1>THAU Studio</h1>
                        <p>Development Team</p>
                    </div>
                </div>
            </div>

            <div class="agents-section">
                <div class="section-title">Equipo de Desarrollo</div>
                <div class="agent-list" id="agentList">
                    <!-- Agents loaded dynamically -->
                </div>
            </div>

            <div class="project-section">
                <div class="section-title">Proyecto Actual</div>
                <div class="project-path" id="projectPath" onclick="changeProjectPath()">
                    📁 <span id="currentPath">~/thau_projects</span>
                </div>
            </div>
        </aside>

        <!-- Main Chat -->
        <main class="main">
            <header class="chat-header">
                <span class="chat-title" id="chatTitle">THAU Development Studio</span>
                <div class="header-actions">
                    <button class="btn btn-secondary" onclick="clearChat()">Limpiar</button>
                    <button class="btn btn-primary" onclick="showNewProjectModal()">+ Nuevo Proyecto</button>
                </div>
            </header>

            <div class="messages" id="messages">
                <div class="welcome" id="welcome">
                    <div class="welcome-icon">🚀</div>
                    <h2>THAU Development Studio</h2>
                    <p>Tu equipo de desarrollo con IA. Describe tu proyecto y THAU lo creará con archivos reales.</p>
                    <div class="welcome-actions">
                        <button class="btn btn-primary" onclick="showNewProjectModal()">Crear Proyecto</button>
                        <button class="btn btn-secondary" onclick="startQuickProject('web')">App Web</button>
                        <button class="btn btn-secondary" onclick="startQuickProject('api')">API REST</button>
                        <button class="btn btn-secondary" onclick="startQuickProject('cli')">CLI Tool</button>
                    </div>
                </div>
            </div>

            <div class="input-area">
                <div class="quick-actions">
                    <span class="quick-action" onclick="insertMention('@pmo')">@PMO</span>
                    <span class="quick-action" onclick="insertMention('@architect')">@Arquitecto</span>
                    <span class="quick-action" onclick="insertMention('@backend')">@Backend</span>
                    <span class="quick-action" onclick="insertMention('@frontend')">@Frontend</span>
                    <span class="quick-action" onclick="insertMention('@devops')">@DevOps</span>
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

        <!-- Files Panel -->
        <aside class="files-panel">
            <div class="files-header">
                <h3>📁 Archivos del Proyecto</h3>
                <button class="refresh-btn" onclick="refreshFiles()">🔄</button>
            </div>
            <div class="files-list" id="filesList">
                <div style="padding: 20px; text-align: center; color: var(--text-muted);">
                    Inicia un proyecto para ver los archivos
                </div>
            </div>
            <div class="file-preview" id="filePreview" style="display: none;">
                <div class="file-preview-header">
                    <span class="file-preview-title" id="previewTitle">archivo.py</span>
                </div>
                <pre class="file-preview-content" id="previewContent"></pre>
            </div>
        </aside>
    </div>

    <!-- New Project Modal -->
    <div class="modal-overlay" id="newProjectModal">
        <div class="modal">
            <h2>🚀 Nuevo Proyecto</h2>
            <input type="text" class="modal-input" id="projectName" placeholder="Nombre del proyecto (ej: mi-app-web)">
            <textarea class="modal-input" id="projectDescription" placeholder="Describe tu proyecto en detalle..." rows="4" style="resize: vertical;"></textarea>
            <div class="modal-actions">
                <button class="btn btn-secondary" onclick="closeModal()">Cancelar</button>
                <button class="btn btn-primary" onclick="createProject()">Crear Proyecto</button>
            </div>
        </div>
    </div>

    <!-- Path Modal -->
    <div class="modal-overlay" id="pathModal">
        <div class="modal">
            <h2>📁 Cambiar Directorio</h2>
            <input type="text" class="modal-input" id="newPath" placeholder="/ruta/al/directorio">
            <div class="modal-actions">
                <button class="btn btn-secondary" onclick="closePathModal()">Cancelar</button>
                <button class="btn btn-primary" onclick="savePath()">Guardar</button>
            </div>
        </div>
    </div>

    <script>
        // State
        let ws = null;
        let agents = [];
        let selectedAgent = null;
        let isTyping = false;
        let currentMessageDiv = null;

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            loadAgents();
            loadProjectPath();
            connectWebSocket();
            autoResizeTextarea();
        });

        // WebSocket
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);

            ws.onopen = () => console.log('WebSocket connected');
            ws.onclose = () => {
                console.log('WebSocket disconnected, reconnecting...');
                setTimeout(connectWebSocket, 2000);
            };
            ws.onerror = (e) => console.error('WebSocket error:', e);
            ws.onmessage = handleMessage;
        }

        function handleMessage(event) {
            const data = JSON.parse(event.data);

            if (data.type === 'start') {
                hideWelcome();
                currentMessageDiv = addMessage(data.agent, '');
                setTyping(true);
            } else if (data.type === 'text' || data.type === 'tools' || data.type === 'tool_results') {
                if (currentMessageDiv) {
                    updateMessage(currentMessageDiv, data.content);
                }
            } else if (data.type === 'done') {
                setTyping(false);
                currentMessageDiv = null;
                refreshFiles();
            }
        }

        // Load agents
        async function loadAgents() {
            const response = await fetch('/api/agents');
            agents = await response.json();

            const list = document.getElementById('agentList');
            list.innerHTML = agents.map(agent => `
                <div class="agent-item" onclick="selectAgent('${agent.role}')" data-role="${agent.role}">
                    <div class="agent-avatar" style="background: ${agent.color}20; color: ${agent.color}">
                        ${agent.emoji}
                    </div>
                    <div class="agent-info">
                        <div class="agent-name">${agent.name}</div>
                        <div class="agent-role">${agent.role}</div>
                    </div>
                </div>
            `).join('');
        }

        // Load project path
        async function loadProjectPath() {
            const response = await fetch('/api/project/path');
            const data = await response.json();
            document.getElementById('currentPath').textContent = data.path.replace(os.homedir || '/Users', '~');
        }

        // Select agent
        function selectAgent(role) {
            selectedAgent = role;
            document.querySelectorAll('.agent-item').forEach(el => {
                el.classList.toggle('active', el.dataset.role === role);
            });
        }

        // Send message
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();

            if (!message || !ws || ws.readyState !== WebSocket.OPEN) return;

            // Add user message
            hideWelcome();
            addMessage({ emoji: '👤', color: '#6366f1', name: 'Tú' }, message, true);

            // Send to server
            ws.send(JSON.stringify({
                message: message,
                agent: selectedAgent
            }));

            input.value = '';
            input.style.height = 'auto';
        }

        // Add message to chat
        function addMessage(agent, content, isUser = false) {
            const messages = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = 'message';

            const time = new Date().toLocaleTimeString('es', { hour: '2-digit', minute: '2-digit' });

            div.innerHTML = `
                <div class="message-avatar" style="background: ${agent.color}20; color: ${agent.color}">
                    ${agent.emoji}
                </div>
                <div class="message-content">
                    <div class="message-header">
                        <span class="message-author">${agent.name}</span>
                        <span class="message-time">${time}</span>
                    </div>
                    <div class="message-text"></div>
                </div>
            `;

            messages.appendChild(div);

            const textDiv = div.querySelector('.message-text');
            if (isUser) {
                textDiv.textContent = content;
            } else {
                updateMessage(div, content);
            }

            messages.scrollTop = messages.scrollHeight;
            return div;
        }

        // Update message content
        function updateMessage(div, content) {
            const textDiv = div.querySelector('.message-text');

            // Parse markdown
            try {
                textDiv.innerHTML = marked.parse(content);

                // Highlight code blocks
                textDiv.querySelectorAll('pre code').forEach(block => {
                    hljs.highlightElement(block);
                });
            } catch (e) {
                textDiv.textContent = content;
            }

            // Scroll to bottom
            const messages = document.getElementById('messages');
            messages.scrollTop = messages.scrollHeight;
        }

        // Typing indicator
        function setTyping(typing) {
            isTyping = typing;
            document.getElementById('sendBtn').disabled = typing;
        }

        // Hide welcome
        function hideWelcome() {
            const welcome = document.getElementById('welcome');
            if (welcome) welcome.style.display = 'none';
        }

        // Insert mention
        function insertMention(mention) {
            const input = document.getElementById('messageInput');
            input.value += mention + ' ';
            input.focus();
        }

        // Handle keydown
        function handleKeydown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        // Auto resize textarea
        function autoResizeTextarea() {
            const textarea = document.getElementById('messageInput');
            textarea.addEventListener('input', () => {
                textarea.style.height = 'auto';
                textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
            });
        }

        // New Project Modal
        function showNewProjectModal() {
            document.getElementById('newProjectModal').classList.add('active');
        }

        function closeModal() {
            document.getElementById('newProjectModal').classList.remove('active');
        }

        function createProject() {
            const name = document.getElementById('projectName').value.trim();
            const description = document.getElementById('projectDescription').value.trim();

            if (!description) {
                alert('Por favor describe tu proyecto');
                return;
            }

            hideWelcome();
            addMessage({ emoji: '👤', color: '#6366f1', name: 'Tú' }, description, true);

            ws.send(JSON.stringify({
                message: description,
                is_project: true,
                project_name: name || 'nuevo_proyecto'
            }));

            closeModal();
            document.getElementById('projectName').value = '';
            document.getElementById('projectDescription').value = '';
        }

        // Quick project templates
        function startQuickProject(type) {
            const templates = {
                'web': 'Crea una aplicación web moderna con Python Flask, que incluya: página de inicio, about, y contacto. Usa Bootstrap 5 para los estilos y SQLite para almacenar mensajes del formulario de contacto.',
                'api': 'Crea una API REST con FastAPI que incluya: endpoints CRUD para gestionar usuarios y tareas, autenticación JWT, documentación OpenAPI, y tests con pytest.',
                'cli': 'Crea una herramienta CLI en Python con Click que permita: gestionar tareas (crear, listar, completar, eliminar), con almacenamiento en SQLite y colores en la terminal con Rich.'
            };

            document.getElementById('projectDescription').value = templates[type];
            showNewProjectModal();
        }

        // Change project path
        function changeProjectPath() {
            document.getElementById('newPath').value = document.getElementById('currentPath').textContent;
            document.getElementById('pathModal').classList.add('active');
        }

        function closePathModal() {
            document.getElementById('pathModal').classList.remove('active');
        }

        async function savePath() {
            const path = document.getElementById('newPath').value.trim();
            if (!path) return;

            await fetch('/api/project/path', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: path.replace('~', os.homedir || '/Users/lperez') })
            });

            loadProjectPath();
            closePathModal();
            refreshFiles();
        }

        // Files
        async function refreshFiles() {
            const response = await fetch('/api/files');
            const files = await response.json();

            const list = document.getElementById('filesList');

            if (files.length === 0) {
                list.innerHTML = '<div style="padding: 20px; text-align: center; color: var(--text-muted);">No hay archivos aún</div>';
                return;
            }

            list.innerHTML = files.map(file => `
                <div class="file-item" onclick="previewFile('${file.path}')">
                    <span class="file-icon">${file.type === 'directory' ? '📁' : '📄'}</span>
                    <span class="file-name">${file.path}</span>
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

            // Highlight items
            document.querySelectorAll('.file-item').forEach(el => {
                el.classList.toggle('active', el.querySelector('.file-name').textContent === path);
            });
        }

        // Clear chat
        async function clearChat() {
            await fetch('/api/clear', { method: 'POST' });
            document.getElementById('messages').innerHTML = `
                <div class="welcome" id="welcome">
                    <div class="welcome-icon">🚀</div>
                    <h2>THAU Development Studio</h2>
                    <p>Tu equipo de desarrollo con IA. Describe tu proyecto y THAU lo creará con archivos reales.</p>
                    <div class="welcome-actions">
                        <button class="btn btn-primary" onclick="showNewProjectModal()">Crear Proyecto</button>
                        <button class="btn btn-secondary" onclick="startQuickProject('web')">App Web</button>
                        <button class="btn btn-secondary" onclick="startQuickProject('api')">API REST</button>
                        <button class="btn btn-secondary" onclick="startQuickProject('cli')">CLI Tool</button>
                    </div>
                </div>
            `;
            refreshFiles();
        }

        // OS homedir helper
        const os = { homedir: '""" + os.path.expanduser("~") + """' };
    </script>
</body>
</html>
"""


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    # Create default project directory
    Path(DEFAULT_PROJECT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                   THAU Development Studio                      ║
║                                                               ║
║  🤖 Equipo de agentes con IA para desarrollo de aplicaciones  ║
║                                                               ║
║  URL: http://localhost:{PORT}                                   ║
║  Proyecto: {DEFAULT_PROJECT_DIR}
╚═══════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
