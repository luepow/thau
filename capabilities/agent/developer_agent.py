"""THAU Developer Agent - Agente autónomo para desarrollo de aplicaciones.

Este agente puede:
1. Planificar aplicaciones completas
2. Crear estructura de proyectos
3. Generar código
4. Ejecutar y probar aplicaciones
5. Iterar basado en errores
"""

import json
import re
import subprocess
import os
import signal
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

try:
    import requests
except ImportError:
    requests = None


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: str
    error: Optional[str] = None


@dataclass
class ProjectPlan:
    """Plan for a development project."""
    name: str
    type: str  # web, api, cli, mobile, etc.
    description: str
    technologies: List[str]
    structure: Dict[str, Any]  # Directory tree
    steps: List[str]
    files_to_create: List[str]


@dataclass
class AgentState:
    """Current state of the developer agent."""
    project_path: str = ""
    project_plan: Optional[ProjectPlan] = None
    current_step: int = 0
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    commands_run: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    running_process: Optional[subprocess.Popen] = None
    completed: bool = False


class DeveloperAgent:
    """Agente de desarrollo autónomo usando THAU via Ollama."""

    SYSTEM_PROMPT = """Eres THAU, un agente de desarrollo autónomo experto en crear aplicaciones.

CAPACIDADES:
- Planificar y diseñar aplicaciones completas
- Crear estructura de proyectos
- Generar código de alta calidad
- Ejecutar comandos del sistema
- Probar y depurar aplicaciones

HERRAMIENTAS DISPONIBLES:
Para usar una herramienta, escribe:
<tool_call>{"name": "tool_name", "arguments": {"arg": "value"}}</tool_call>

Herramientas:
1. create_directory: Crear directorio
   Args: {"path": "ruta/del/directorio"}

2. write_file: Crear/escribir archivo
   Args: {"path": "ruta/archivo.ext", "content": "contenido"}

3. read_file: Leer archivo existente
   Args: {"path": "ruta/archivo.ext"}

4. bash: Ejecutar comando bash
   Args: {"command": "comando a ejecutar"}

5. list_directory: Listar contenido de directorio
   Args: {"path": "ruta"}

6. start_server: Iniciar servidor de desarrollo
   Args: {"command": "python app.py", "port": 5000}

7. stop_server: Detener servidor en ejecución
   Args: {}

PROCESO DE DESARROLLO:
1. PLANIFICACIÓN: Analiza el requerimiento y diseña la arquitectura
2. ESTRUCTURA: Crea los directorios necesarios
3. IMPLEMENTACIÓN: Genera el código archivo por archivo
4. DEPENDENCIAS: Instala las dependencias necesarias
5. PRUEBA: Ejecuta la aplicación para verificar
6. ITERACIÓN: Corrige errores si los hay

IMPORTANTE:
- Siempre crea código completo y funcional
- Incluye manejo de errores
- Usa buenas prácticas de programación
- Documenta el código cuando sea necesario
- Responde siempre en español

Cuando termines una tarea, indica claramente: "✅ TAREA COMPLETADA"
Si encuentras un error, indica: "❌ ERROR: [descripción]" y propón una solución."""

    def __init__(
        self,
        working_dir: str = ".",
        ollama_model: str = "thau:agi-v3",
        ollama_url: str = "http://localhost:11434",
        max_iterations: int = 30,
        callback: Optional[Callable[[str], None]] = None
    ):
        """Initialize the developer agent.

        Args:
            working_dir: Working directory for projects
            ollama_model: Ollama model to use
            ollama_url: Ollama API URL
            max_iterations: Maximum iterations
            callback: Callback for streaming output
        """
        self.working_dir = Path(working_dir).resolve()
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.max_iterations = max_iterations
        self.callback = callback
        self.state = AgentState(project_path=str(self.working_dir))

        # Ensure working directory exists
        self.working_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DeveloperAgent initialized: dir={self.working_dir}, model={ollama_model}")

    def _log(self, message: str):
        """Log message and call callback if set."""
        logger.info(message)
        if self.callback:
            self.callback(message + "\n")

    def _safe_path(self, path: str) -> Path:
        """Ensure path is within working directory."""
        if path.startswith("/"):
            full_path = Path(path)
        else:
            full_path = (self.working_dir / path).resolve()

        # Allow paths in working dir or subdirectories
        try:
            full_path.relative_to(self.working_dir)
        except ValueError:
            # Path is outside working dir, but allow if it's the working dir itself
            if str(full_path) != str(self.working_dir):
                raise ValueError(f"Path {path} is outside working directory")

        return full_path

    # ============ TOOLS ============

    def tool_create_directory(self, path: str) -> ToolResult:
        """Create a directory."""
        try:
            dir_path = self._safe_path(path)
            dir_path.mkdir(parents=True, exist_ok=True)
            return ToolResult(True, f"✓ Directorio creado: {path}")
        except Exception as e:
            return ToolResult(False, "", str(e))

    def tool_write_file(self, path: str, content: str) -> ToolResult:
        """Write content to a file."""
        try:
            file_path = self._safe_path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            self.state.files_created.append(path)
            return ToolResult(True, f"✓ Archivo creado: {path} ({len(content)} bytes)")
        except Exception as e:
            return ToolResult(False, "", str(e))

    def tool_read_file(self, path: str) -> ToolResult:
        """Read a file."""
        try:
            file_path = self._safe_path(path)

            if not file_path.exists():
                return ToolResult(False, "", f"Archivo no encontrado: {path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return ToolResult(True, content)
        except Exception as e:
            return ToolResult(False, "", str(e))

    def tool_bash(self, command: str, timeout: int = 60) -> ToolResult:
        """Execute a bash command."""
        # Block dangerous commands
        dangerous = ['rm -rf /', 'mkfs', '> /dev', 'dd if=']
        for d in dangerous:
            if d in command:
                return ToolResult(False, "", f"Comando bloqueado por seguridad: {command}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.working_dir)
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"

            self.state.commands_run.append(command)

            return ToolResult(
                result.returncode == 0,
                output or "(sin salida)",
                result.stderr if result.returncode != 0 else None
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", f"Comando excedió timeout de {timeout}s")
        except Exception as e:
            return ToolResult(False, "", str(e))

    def tool_list_directory(self, path: str = ".") -> ToolResult:
        """List directory contents."""
        try:
            dir_path = self._safe_path(path)

            if not dir_path.exists():
                return ToolResult(False, "", f"Directorio no encontrado: {path}")

            items = []
            for item in sorted(dir_path.iterdir()):
                if item.name.startswith('.'):
                    continue
                if item.is_dir():
                    items.append(f"📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    items.append(f"📄 {item.name} ({size:,} bytes)")

            return ToolResult(True, "\n".join(items) if items else "(directorio vacío)")
        except Exception as e:
            return ToolResult(False, "", str(e))

    def tool_start_server(self, command: str, port: int = 5000) -> ToolResult:
        """Start a development server."""
        try:
            # Stop any existing server
            self.tool_stop_server()

            # Start new server
            self.state.running_process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.working_dir),
                preexec_fn=os.setsid  # Create new process group
            )

            # Wait a bit for server to start
            time.sleep(2)

            # Check if still running
            if self.state.running_process.poll() is None:
                return ToolResult(True, f"✓ Servidor iniciado en puerto {port}\nURL: http://localhost:{port}")
            else:
                stdout, stderr = self.state.running_process.communicate()
                return ToolResult(False, "", f"Servidor falló al iniciar:\n{stderr.decode()}")

        except Exception as e:
            return ToolResult(False, "", str(e))

    def tool_stop_server(self) -> ToolResult:
        """Stop the running server."""
        try:
            if self.state.running_process:
                os.killpg(os.getpgid(self.state.running_process.pid), signal.SIGTERM)
                self.state.running_process = None
                return ToolResult(True, "✓ Servidor detenido")
            return ToolResult(True, "No hay servidor en ejecución")
        except Exception as e:
            return ToolResult(False, "", str(e))

    def execute_tool(self, name: str, arguments: Dict) -> ToolResult:
        """Execute a tool by name."""
        tools = {
            "create_directory": lambda: self.tool_create_directory(arguments.get("path", ".")),
            "write_file": lambda: self.tool_write_file(
                arguments.get("path", ""),
                arguments.get("content", "")
            ),
            "read_file": lambda: self.tool_read_file(arguments.get("path", "")),
            "bash": lambda: self.tool_bash(
                arguments.get("command", ""),
                arguments.get("timeout", 60)
            ),
            "list_directory": lambda: self.tool_list_directory(arguments.get("path", ".")),
            "start_server": lambda: self.tool_start_server(
                arguments.get("command", ""),
                arguments.get("port", 5000)
            ),
            "stop_server": lambda: self.tool_stop_server(),
        }

        if name not in tools:
            return ToolResult(False, "", f"Herramienta desconocida: {name}")

        return tools[name]()

    # ============ OLLAMA INTEGRATION ============

    def call_ollama(self, messages: List[Dict], stream: bool = True) -> Generator[str, None, None]:
        """Call Ollama API with streaming."""
        if not requests:
            yield "Error: requests library not installed"
            return

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
                        "num_ctx": 4096,
                    }
                },
                stream=stream,
                timeout=120
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

        except requests.exceptions.ConnectionError:
            yield "❌ Error: No se puede conectar a Ollama. Asegúrate de que esté corriendo."
        except Exception as e:
            yield f"❌ Error: {str(e)}"

    def call_ollama_full(self, messages: List[Dict]) -> str:
        """Call Ollama and get full response."""
        return "".join(self.call_ollama(messages, stream=False))

    # ============ AGENT LOGIC ============

    def parse_tool_calls(self, response: str) -> List[Dict]:
        """Extract tool calls from response."""
        tool_calls = []

        # Pattern: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            try:
                # Clean up the JSON
                clean_match = match.strip()
                tool_call = json.loads(clean_match)
                if "name" in tool_call:
                    tool_calls.append(tool_call)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool call: {match[:100]}... Error: {e}")

        return tool_calls

    def format_tool_results(self, results: List[tuple]) -> str:
        """Format tool results for the model."""
        formatted = []
        for tool_call, result in results:
            name = tool_call.get("name", "unknown")
            if result.success:
                formatted.append(f"<tool_result name=\"{name}\" success=\"true\">\n{result.output}\n</tool_result>")
            else:
                formatted.append(f"<tool_result name=\"{name}\" success=\"false\">\n❌ Error: {result.error}\n</tool_result>")
        return "\n\n".join(formatted)

    def run(self, task: str) -> Generator[str, None, Dict]:
        """Run the agent on a development task.

        Args:
            task: Task description

        Yields:
            Progress updates

        Returns:
            Final result dictionary
        """
        self.state = AgentState(project_path=str(self.working_dir))

        yield f"🚀 Iniciando proyecto en: {self.working_dir}\n"
        yield f"📋 Tarea: {task}\n"
        yield "=" * 50 + "\n\n"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"""TAREA: {task}

DIRECTORIO DE TRABAJO: {self.working_dir}

Por favor:
1. Analiza la tarea y planifica la implementación
2. Crea la estructura del proyecto
3. Implementa el código necesario
4. Configura las dependencias
5. Prueba la aplicación

Comienza ahora:"""}
        ]

        iteration = 0
        full_response = ""

        while iteration < self.max_iterations and not self.state.completed:
            iteration += 1
            yield f"\n--- Iteración {iteration}/{self.max_iterations} ---\n"

            # Get response from model
            response_chunks = []
            for chunk in self.call_ollama(messages):
                response_chunks.append(chunk)
                yield chunk

            full_response = "".join(response_chunks)

            # Check for completion
            if "✅ TAREA COMPLETADA" in full_response:
                self.state.completed = True
                yield "\n\n🎉 ¡Proyecto completado exitosamente!\n"
                break

            # Parse and execute tool calls
            tool_calls = self.parse_tool_calls(full_response)

            if not tool_calls:
                # No tool calls, might be done or asking a question
                messages.append({"role": "assistant", "content": full_response})

                # Prompt to continue
                messages.append({
                    "role": "user",
                    "content": "Continúa con la implementación. Usa las herramientas disponibles."
                })
                continue

            # Execute tools
            yield f"\n\n🔧 Ejecutando {len(tool_calls)} herramienta(s)...\n"

            results = []
            for tc in tool_calls:
                name = tc.get("name", "")
                args = tc.get("arguments", {})

                yield f"  → {name}: "
                result = self.execute_tool(name, args)
                results.append((tc, result))

                if result.success:
                    yield f"✓\n"
                else:
                    yield f"✗ ({result.error})\n"
                    self.state.errors.append(f"{name}: {result.error}")

            # Add results to conversation
            tool_results_str = self.format_tool_results(results)
            messages.append({"role": "assistant", "content": full_response})
            messages.append({"role": "user", "content": f"Resultados:\n\n{tool_results_str}\n\nContinúa con el siguiente paso."})

        # Final report
        yield "\n" + "=" * 50 + "\n"
        yield "📊 RESUMEN DEL PROYECTO\n"
        yield f"  • Iteraciones: {iteration}\n"
        yield f"  • Archivos creados: {len(self.state.files_created)}\n"
        for f in self.state.files_created[:10]:
            yield f"    - {f}\n"
        if len(self.state.files_created) > 10:
            yield f"    ... y {len(self.state.files_created) - 10} más\n"
        yield f"  • Comandos ejecutados: {len(self.state.commands_run)}\n"
        if self.state.errors:
            yield f"  • Errores encontrados: {len(self.state.errors)}\n"
        yield f"  • Completado: {'✅ Sí' if self.state.completed else '⏸️ Parcial'}\n"

        return {
            "completed": self.state.completed,
            "iterations": iteration,
            "files_created": self.state.files_created,
            "files_modified": self.state.files_modified,
            "commands_run": self.state.commands_run,
            "errors": self.state.errors,
            "project_path": str(self.working_dir)
        }

    def plan_project(self, description: str) -> Generator[str, None, ProjectPlan]:
        """Generate a project plan.

        Args:
            description: Project description

        Yields:
            Plan generation progress

        Returns:
            ProjectPlan object
        """
        yield "📝 Generando plan del proyecto...\n\n"

        planning_prompt = f"""Analiza este proyecto y genera un plan detallado:

PROYECTO: {description}

Genera un plan en el siguiente formato JSON:
```json
{{
    "name": "nombre-del-proyecto",
    "type": "web|api|cli|mobile|desktop",
    "description": "Descripción breve",
    "technologies": ["tech1", "tech2"],
    "structure": {{
        "carpeta1/": ["archivo1.py", "archivo2.py"],
        "carpeta2/": ["archivo3.js"]
    }},
    "steps": [
        "Paso 1: ...",
        "Paso 2: ..."
    ],
    "files_to_create": [
        "ruta/archivo1.py",
        "ruta/archivo2.js"
    ]
}}
```

Sé específico y práctico. El plan debe ser ejecutable."""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": planning_prompt}
        ]

        response = ""
        for chunk in self.call_ollama(messages):
            response += chunk
            yield chunk

        # Parse JSON from response
        try:
            # Find JSON block
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(1))
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{[^{}]*"name"[^{}]*\}', response, re.DOTALL)
                if json_match:
                    plan_data = json.loads(json_match.group(0))
                else:
                    raise ValueError("No JSON found in response")

            plan = ProjectPlan(
                name=plan_data.get("name", "proyecto"),
                type=plan_data.get("type", "unknown"),
                description=plan_data.get("description", description),
                technologies=plan_data.get("technologies", []),
                structure=plan_data.get("structure", {}),
                steps=plan_data.get("steps", []),
                files_to_create=plan_data.get("files_to_create", [])
            )

            self.state.project_plan = plan

            yield "\n\n✅ Plan generado exitosamente\n"
            return plan

        except Exception as e:
            logger.error(f"Error parsing plan: {e}")
            yield f"\n\n⚠️ Error parseando plan: {e}\n"

            # Return a basic plan
            return ProjectPlan(
                name="proyecto",
                type="unknown",
                description=description,
                technologies=[],
                structure={},
                steps=["Implementar proyecto"],
                files_to_create=[]
            )


# Templates for different project types
PROJECT_TEMPLATES = {
    "flask_web": {
        "description": "Aplicación web Flask",
        "structure": {
            "app/": ["__init__.py", "routes.py", "models.py"],
            "app/templates/": ["base.html", "index.html"],
            "app/static/css/": ["style.css"],
            "app/static/js/": ["main.js"],
        },
        "files": {
            "requirements.txt": "flask>=2.0\nflask-sqlalchemy\npython-dotenv",
            "run.py": "from app import create_app\n\napp = create_app()\n\nif __name__ == '__main__':\n    app.run(debug=True)",
        }
    },
    "fastapi_api": {
        "description": "API REST con FastAPI",
        "structure": {
            "app/": ["__init__.py", "main.py", "dependencies.py"],
            "app/routers/": ["__init__.py", "items.py", "users.py"],
            "app/models/": ["__init__.py", "item.py", "user.py"],
            "app/schemas/": ["__init__.py", "item.py", "user.py"],
        },
        "files": {
            "requirements.txt": "fastapi\nuvicorn[standard]\npydantic\nsqlalchemy",
        }
    },
    "react_frontend": {
        "description": "Aplicación React",
        "structure": {
            "src/": ["index.js", "App.js", "App.css"],
            "src/components/": ["Header.js", "Footer.js"],
            "public/": ["index.html"],
        }
    },
    "python_cli": {
        "description": "CLI en Python",
        "structure": {
            "src/": ["__init__.py", "main.py", "commands.py", "utils.py"],
        },
        "files": {
            "requirements.txt": "click\nrich",
            "setup.py": "from setuptools import setup\n\nsetup(name='cli-app', packages=['src'])",
        }
    }
}


if __name__ == "__main__":
    # Test the agent
    print("Testing DeveloperAgent...")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        agent = DeveloperAgent(working_dir=tmpdir, ollama_model="thau:agi-v3")

        # Test tools
        print("\n1. Testing tools...")

        result = agent.tool_create_directory("test_project")
        print(f"create_directory: {result}")

        result = agent.tool_write_file("test_project/hello.py", "print('Hello!')")
        print(f"write_file: {result}")

        result = agent.tool_list_directory("test_project")
        print(f"list_directory: {result}")

        result = agent.tool_read_file("test_project/hello.py")
        print(f"read_file: {result}")

        result = agent.tool_bash("echo 'Test command'")
        print(f"bash: {result}")

        print("\n✅ All tools working!")

        # Test plan generation (if Ollama is available)
        print("\n2. Testing plan generation...")
        try:
            for output in agent.plan_project("Una aplicación web simple con Flask"):
                print(output, end="", flush=True)
        except Exception as e:
            print(f"Ollama not available: {e}")
