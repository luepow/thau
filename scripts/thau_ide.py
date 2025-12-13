#!/usr/bin/env python3
"""
THAU IDE - Entorno de Desarrollo Integrado con THAU

Características:
1. Selector de carpeta de trabajo
2. Base de datos SQLite para proyectos
3. Detección de tipo de proyecto
4. Ejecución automática con preview en vivo
5. Templates preconfigurados (React, Flask, FastAPI, etc.)
6. Terminal integrada
7. Hot reload para desarrollo
"""

import json
import re
import subprocess
import os
import sys
import asyncio
import sqlite3
import signal
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Generator, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import queue
import threading
import time
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from loguru import logger
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

OLLAMA_MODEL = "thau:agi-v3"
OLLAMA_URL = "http://localhost:11434"
DEFAULT_WORKSPACE = os.path.expanduser("~/thau_workspace")
DB_PATH = os.path.expanduser("~/.thau_ide/projects.db")
PORT = 7867
GENERATION_TIMEOUT = 180

# ============================================================================
# DATABASE - Gestión de proyectos
# ============================================================================

class ProjectDatabase:
    """Base de datos SQLite para proyectos"""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Inicializa la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                path TEXT NOT NULL UNIQUE,
                type TEXT DEFAULT 'unknown',
                framework TEXT,
                run_command TEXT,
                port INTEGER,
                created_at TEXT,
                updated_at TEXT,
                description TEXT,
                settings TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS project_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT,
                prompt TEXT,
                response TEXT,
                files_created TEXT,
                timestamp TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        ''')

        conn.commit()
        conn.close()
        logger.info(f"Base de datos inicializada: {self.db_path}")

    def create_project(self, name: str, path: str, project_type: str = "unknown",
                       framework: str = None, run_command: str = None, port: int = None,
                       description: str = "") -> Dict:
        """Crea un nuevo proyecto"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        project_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        try:
            cursor.execute('''
                INSERT INTO projects (id, name, path, type, framework, run_command, port, created_at, updated_at, description, settings)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (project_id, name, path, project_type, framework, run_command, port, now, now, description, '{}'))

            conn.commit()

            # Crear directorio si no existe
            Path(path).mkdir(parents=True, exist_ok=True)

            return {
                "id": project_id,
                "name": name,
                "path": path,
                "type": project_type,
                "framework": framework,
                "run_command": run_command,
                "port": port,
                "created_at": now
            }
        except sqlite3.IntegrityError:
            # Proyecto ya existe, actualizar
            cursor.execute('''
                UPDATE projects SET name=?, type=?, framework=?, run_command=?, port=?, updated_at=?, description=?
                WHERE path=?
            ''', (name, project_type, framework, run_command, port, now, description, path))
            conn.commit()

            cursor.execute('SELECT * FROM projects WHERE path=?', (path,))
            row = cursor.fetchone()
            return self._row_to_dict(row)
        finally:
            conn.close()

    def get_project(self, project_id: str = None, path: str = None) -> Optional[Dict]:
        """Obtiene un proyecto por ID o path"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if project_id:
            cursor.execute('SELECT * FROM projects WHERE id=?', (project_id,))
        elif path:
            cursor.execute('SELECT * FROM projects WHERE path=?', (path,))
        else:
            return None

        row = cursor.fetchone()
        conn.close()

        return self._row_to_dict(row) if row else None

    def list_projects(self) -> List[Dict]:
        """Lista todos los proyectos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM projects ORDER BY updated_at DESC')
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(row) for row in rows]

    def update_project(self, project_id: str, **kwargs) -> bool:
        """Actualiza un proyecto"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        kwargs['updated_at'] = datetime.now().isoformat()

        set_clause = ', '.join([f'{k}=?' for k in kwargs.keys()])
        values = list(kwargs.values()) + [project_id]

        cursor.execute(f'UPDATE projects SET {set_clause} WHERE id=?', values)
        conn.commit()
        conn.close()

        return cursor.rowcount > 0

    def delete_project(self, project_id: str) -> bool:
        """Elimina un proyecto de la BD (no los archivos)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM projects WHERE id=?', (project_id,))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()

        return success

    def add_history(self, project_id: str, prompt: str, response: str, files_created: List[str]):
        """Agrega entrada al historial"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO project_history (project_id, prompt, response, files_created, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (project_id, prompt, response, json.dumps(files_created), datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def _row_to_dict(self, row) -> Dict:
        """Convierte una fila a diccionario"""
        if not row:
            return None
        return {
            "id": row[0],
            "name": row[1],
            "path": row[2],
            "type": row[3],
            "framework": row[4],
            "run_command": row[5],
            "port": row[6],
            "created_at": row[7],
            "updated_at": row[8],
            "description": row[9],
            "settings": json.loads(row[10]) if row[10] else {}
        }


# ============================================================================
# PROJECT TEMPLATES
# ============================================================================

PROJECT_TEMPLATES = {
    "react": {
        "name": "React App",
        "icon": "⚛️",
        "type": "frontend",
        "framework": "react",
        "init_commands": [
            "npm create vite@latest . -- --template react",
            "npm install"
        ],
        "run_command": "npm run dev",
        "port": 5173,
        "description": "Aplicación React moderna con Vite"
    },
    "react-ts": {
        "name": "React + TypeScript",
        "icon": "⚛️",
        "type": "frontend",
        "framework": "react",
        "init_commands": [
            "npm create vite@latest . -- --template react-ts",
            "npm install"
        ],
        "run_command": "npm run dev",
        "port": 5173,
        "description": "React con TypeScript y Vite"
    },
    "vue": {
        "name": "Vue.js App",
        "icon": "💚",
        "type": "frontend",
        "framework": "vue",
        "init_commands": [
            "npm create vite@latest . -- --template vue",
            "npm install"
        ],
        "run_command": "npm run dev",
        "port": 5173,
        "description": "Aplicación Vue.js con Vite"
    },
    "nextjs": {
        "name": "Next.js App",
        "icon": "▲",
        "type": "fullstack",
        "framework": "nextjs",
        "init_commands": [
            "npx create-next-app@latest . --typescript --tailwind --eslint --app --src-dir --import-alias '@/*' --use-npm"
        ],
        "run_command": "npm run dev",
        "port": 3000,
        "description": "Next.js con App Router y Tailwind"
    },
    "flask": {
        "name": "Flask API",
        "icon": "🌶️",
        "type": "backend",
        "framework": "flask",
        "init_commands": [
            "python -m venv venv",
            "source venv/bin/activate && pip install flask"
        ],
        "run_command": "source venv/bin/activate && python app.py",
        "port": 5000,
        "description": "API con Flask y Python"
    },
    "fastapi": {
        "name": "FastAPI",
        "icon": "⚡",
        "type": "backend",
        "framework": "fastapi",
        "init_commands": [
            "python -m venv venv",
            "source venv/bin/activate && pip install fastapi uvicorn"
        ],
        "run_command": "source venv/bin/activate && uvicorn main:app --reload --port 8000",
        "port": 8000,
        "description": "API moderna con FastAPI"
    },
    "express": {
        "name": "Express.js",
        "icon": "🚂",
        "type": "backend",
        "framework": "express",
        "init_commands": [
            "npm init -y",
            "npm install express"
        ],
        "run_command": "node index.js",
        "port": 3000,
        "description": "API con Express.js"
    },
    "html": {
        "name": "HTML/CSS/JS",
        "icon": "📄",
        "type": "frontend",
        "framework": "vanilla",
        "init_commands": [],
        "run_command": "python -m http.server 8080",
        "port": 8080,
        "description": "Página web estática"
    },
    "python-cli": {
        "name": "Python CLI",
        "icon": "🐍",
        "type": "cli",
        "framework": "python",
        "init_commands": [
            "python -m venv venv",
            "source venv/bin/activate && pip install click rich"
        ],
        "run_command": "source venv/bin/activate && python main.py",
        "port": None,
        "description": "Herramienta de línea de comandos"
    }
}


# ============================================================================
# PROJECT RUNNER - Ejecuta proyectos
# ============================================================================

class ProjectRunner:
    """Ejecuta y gestiona procesos de proyectos"""

    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.outputs: Dict[str, List[str]] = {}

    def start(self, project_id: str, path: str, command: str, port: int = None) -> Dict:
        """Inicia un proyecto"""
        if project_id in self.processes:
            proc = self.processes[project_id]
            if proc.poll() is None:
                return {"status": "already_running", "port": port}

        try:
            # Preparar comando
            full_command = f"cd {path} && {command}"

            logger.info(f"Iniciando proyecto {project_id}: {command}")

            proc = subprocess.Popen(
                full_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid
            )

            self.processes[project_id] = proc
            self.outputs[project_id] = []

            # Thread para capturar output
            def capture_output():
                for line in proc.stdout:
                    self.outputs[project_id].append(line)
                    if len(self.outputs[project_id]) > 500:
                        self.outputs[project_id] = self.outputs[project_id][-250:]

            thread = threading.Thread(target=capture_output, daemon=True)
            thread.start()

            # Esperar un poco para ver si arranca
            time.sleep(2)

            if proc.poll() is not None:
                return {"status": "failed", "error": "Proceso terminó inmediatamente"}

            return {"status": "started", "port": port, "pid": proc.pid}

        except Exception as e:
            logger.error(f"Error iniciando proyecto: {e}")
            return {"status": "error", "error": str(e)}

    def stop(self, project_id: str) -> Dict:
        """Detiene un proyecto"""
        if project_id not in self.processes:
            return {"status": "not_running"}

        proc = self.processes[project_id]

        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
            except:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except:
                    pass

        del self.processes[project_id]
        return {"status": "stopped"}

    def get_output(self, project_id: str, last_n: int = 50) -> List[str]:
        """Obtiene output de un proyecto"""
        return self.outputs.get(project_id, [])[-last_n:]

    def is_running(self, project_id: str) -> bool:
        """Verifica si un proyecto está corriendo"""
        if project_id not in self.processes:
            return False
        return self.processes[project_id].poll() is None

    def stop_all(self):
        """Detiene todos los proyectos"""
        for project_id in list(self.processes.keys()):
            self.stop(project_id)


# ============================================================================
# CODE EXTRACTOR
# ============================================================================

class CodeExtractor:
    """Extractor de código de respuestas de THAU"""

    EXTENSIONS = r'py|js|jsx|ts|tsx|html|css|json|md|txt|yaml|yml|sh|sql|xml|toml|ini|cfg|env|vue|svelte'

    @staticmethod
    def extract_all_code_blocks(text: str) -> List[Dict]:
        blocks = []
        seen = set()

        patterns = [
            (r'```(\w+)\s+([^\n`]+\.(?:' + CodeExtractor.EXTENSIONS + r'))\s*\n(.*?)```',
             lambda m: (m.group(2).strip(), m.group(3).strip(), m.group(1))),
            (r'\*\*([^\n*]+\.(?:' + CodeExtractor.EXTENSIONS + r'))\*\*[:\s]*\n```(?:\w*)\n(.*?)```',
             lambda m: (m.group(1).strip(), m.group(2).strip(), 'auto')),
            (r'#\s*([^\n]+\.(?:' + CodeExtractor.EXTENSIONS + r'))\s*\n```(?:\w*)\n(.*?)```',
             lambda m: (m.group(1).strip(), m.group(2).strip(), 'auto')),
            (r'([a-zA-Z0-9_/.-]+\.(?:' + CodeExtractor.EXTENSIONS + r')):\s*\n```(?:\w*)\n(.*?)```',
             lambda m: (m.group(1).strip(), m.group(2).strip(), 'auto')),
            (r'```(\w+)\n#\s*([^\n]+\.(?:' + CodeExtractor.EXTENSIONS + r'))\s*\n(.*?)```',
             lambda m: (m.group(2).strip(), m.group(3).strip(), m.group(1))),
            (r'`([^\n`]+\.(?:' + CodeExtractor.EXTENSIONS + r'))`[:\s]*\n```(?:\w*)\n(.*?)```',
             lambda m: (m.group(1).strip(), m.group(2).strip(), 'auto')),
        ]

        for pattern, extractor in patterns:
            for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
                try:
                    filename, content, lang = extractor(match)
                    filename = filename.strip('`*# \t')
                    content_hash = hash(content[:200])
                    if content_hash not in seen and len(content) > 10:
                        seen.add(content_hash)
                        blocks.append({"filename": filename, "content": content, "lang": lang})
                except:
                    continue

        return blocks


# ============================================================================
# THAU DEVELOPER
# ============================================================================

class THAUDeveloper:
    """Desarrollador THAU integrado"""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.project_path.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict] = []
        self.created_files: set = set()

    def set_project_path(self, path: str):
        self.project_path = Path(path)
        self.project_path.mkdir(parents=True, exist_ok=True)
        self.created_files = set()

    def create_file(self, path: str, content: str) -> Dict:
        try:
            file_path = self.project_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.created_files.add(path)
            logger.info(f"✅ Archivo creado: {path}")
            return {"success": True, "path": path, "size": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_files(self) -> List[Dict]:
        files = []
        try:
            for item in sorted(self.project_path.rglob("*")):
                if any(x in str(item) for x in ['.git', '__pycache__', 'node_modules', '.next', 'venv']):
                    continue
                if item.name.startswith('.'):
                    continue
                try:
                    rel = item.relative_to(self.project_path)
                    if item.is_file():
                        files.append({"path": str(rel), "type": "file", "size": item.stat().st_size})
                except:
                    pass
        except:
            pass
        return files

    def read_file(self, path: str) -> Optional[str]:
        try:
            return (self.project_path / path).read_text(encoding='utf-8')
        except:
            return None

    def get_system_prompt(self, template: Dict = None) -> str:
        base = """Eres THAU, un desarrollador de software experto.

Tu trabajo es crear código completo y funcional.

FORMATO PARA ARCHIVOS - Usa EXACTAMENTE este formato:

**nombre_archivo.ext**
```lenguaje
código completo aquí
```

EJEMPLO:

**app.py**
```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello World'

if __name__ == '__main__':
    app.run(debug=True)
```

**index.html**
```html
<!DOCTYPE html>
<html>
<head><title>Mi App</title></head>
<body><h1>Hola</h1></body>
</html>
```

REGLAS:
- Genera código COMPLETO y funcional
- Incluye TODOS los imports
- El código debe ejecutarse sin cambios
- Responde en español"""

        if template:
            base += f"\n\nEstás creando un proyecto {template.get('name', '')}."
            base += f"\nFramework: {template.get('framework', 'N/A')}"

        return base

    def call_ollama(self, messages: List[Dict]) -> Generator[str, None, None]:
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": 0.7, "num_ctx": 4096, "num_predict": 4000}
                },
                stream=True,
                timeout=GENERATION_TIMEOUT
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
                        if data.get("done"):
                            break
                    except:
                        continue
        except Exception as e:
            yield f"\n\n[Error: {str(e)}]"

    def develop(self, prompt: str, template: Dict = None) -> Generator[Dict, None, None]:
        existing = self.list_files()
        context = ""
        if existing:
            context = "\n\nArchivos existentes:\n" + "\n".join([f"- {f['path']}" for f in existing[:15]])

        messages = [
            {"role": "system", "content": self.get_system_prompt(template)},
        ]

        for msg in self.history[-2:]:
            messages.append(msg)

        messages.append({"role": "user", "content": f"{prompt}{context}"})

        yield {"type": "start"}

        full_response = ""
        last_check = 0

        for chunk in self.call_ollama(messages):
            full_response += chunk
            yield {"type": "stream", "content": chunk, "full": full_response}

            if len(full_response) - last_check > 500:
                last_check = len(full_response)
                blocks = CodeExtractor.extract_all_code_blocks(full_response)
                for block in blocks:
                    if block["filename"] not in self.created_files:
                        result = self.create_file(block["filename"], block["content"])
                        if result.get("success"):
                            yield {"type": "file_created", "file": {"path": block["filename"], "size": len(block["content"])}}

        # Final
        blocks = CodeExtractor.extract_all_code_blocks(full_response)
        for block in blocks:
            if block["filename"] not in self.created_files:
                result = self.create_file(block["filename"], block["content"])
                if result.get("success"):
                    yield {"type": "file_created", "file": {"path": block["filename"], "size": len(block["content"])}}

        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": full_response})

        yield {"type": "done", "files_created": list(self.created_files), "total": len(self.created_files)}

    def clear(self):
        self.history = []
        self.created_files = set()


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="THAU IDE")
db = ProjectDatabase()
runner = ProjectRunner()
developer = THAUDeveloper(DEFAULT_WORKSPACE)


@app.on_event("shutdown")
def shutdown():
    runner.stop_all()


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_TEMPLATE


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api/projects")
async def list_projects():
    return db.list_projects()


@app.post("/api/projects")
async def create_project(data: dict):
    name = data.get("name", "nuevo-proyecto")
    path = data.get("path", str(Path(DEFAULT_WORKSPACE) / name))
    template_id = data.get("template")
    description = data.get("description", "")

    template = PROJECT_TEMPLATES.get(template_id, {})

    project = db.create_project(
        name=name,
        path=path,
        project_type=template.get("type", "unknown"),
        framework=template.get("framework"),
        run_command=template.get("run_command"),
        port=template.get("port"),
        description=description
    )

    # Crear directorio
    Path(path).mkdir(parents=True, exist_ok=True)

    # Ejecutar comandos de inicialización si hay template
    if template and template.get("init_commands"):
        for cmd in template["init_commands"]:
            try:
                subprocess.run(cmd, shell=True, cwd=path, timeout=120, capture_output=True)
            except:
                pass

    developer.set_project_path(path)

    return project


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    project = db.get_project(project_id=project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return project


@app.put("/api/projects/{project_id}")
async def update_project(project_id: str, data: dict):
    db.update_project(project_id, **data)
    return db.get_project(project_id=project_id)


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    db.delete_project(project_id)
    return {"ok": True}


@app.get("/api/templates")
async def get_templates():
    return PROJECT_TEMPLATES


@app.post("/api/projects/{project_id}/select")
async def select_project(project_id: str):
    project = db.get_project(project_id=project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    developer.set_project_path(project["path"])
    return project


@app.post("/api/projects/{project_id}/run")
async def run_project(project_id: str):
    project = db.get_project(project_id=project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    result = runner.start(
        project_id,
        project["path"],
        project.get("run_command", "echo 'No run command'"),
        project.get("port")
    )

    return result


@app.post("/api/projects/{project_id}/stop")
async def stop_project(project_id: str):
    return runner.stop(project_id)


@app.get("/api/projects/{project_id}/status")
async def project_status(project_id: str):
    return {
        "running": runner.is_running(project_id),
        "output": runner.get_output(project_id, 30)
    }


@app.get("/api/projects/{project_id}/output")
async def project_output(project_id: str):
    return {"output": runner.get_output(project_id, 100)}


@app.get("/api/files")
async def get_files():
    return developer.list_files()


@app.get("/api/files/{path:path}")
async def get_file(path: str):
    content = developer.read_file(path)
    if content is None:
        raise HTTPException(404, "File not found")
    return {"path": path, "content": content}


@app.post("/api/files")
async def create_file(data: dict):
    path = data.get("path")
    content = data.get("content", "")
    return developer.create_file(path, content)


@app.post("/api/workspace")
async def set_workspace(data: dict):
    path = data.get("path")
    if path:
        developer.set_project_path(path)
    return {"path": str(developer.project_path)}


@app.get("/api/workspace")
async def get_workspace():
    return {"path": str(developer.project_path)}


@app.websocket("/ws/dev")
async def websocket_dev(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket conectado")

    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt", "")
            template_id = data.get("template")

            template = PROJECT_TEMPLATES.get(template_id) if template_id else None

            result_queue = queue.Queue()

            def run_generator():
                try:
                    for chunk in developer.develop(prompt, template):
                        result_queue.put(("data", chunk))
                    result_queue.put(("done", None))
                except Exception as e:
                    result_queue.put(("error", str(e)))

            thread = threading.Thread(target=run_generator, daemon=True)
            thread.start()

            finished = False
            while not finished:
                await asyncio.sleep(0.05)

                while not result_queue.empty():
                    try:
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
                    except:
                        break

                if not thread.is_alive() and result_queue.empty():
                    finished = True

    except WebSocketDisconnect:
        logger.info("WebSocket desconectado")


# ============================================================================
# HTML TEMPLATE
# ============================================================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>THAU IDE</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg: #09090b;
            --bg2: #18181b;
            --bg3: #27272a;
            --text: #fafafa;
            --text2: #a1a1aa;
            --accent: #8b5cf6;
            --accent2: #a78bfa;
            --green: #22c55e;
            --red: #ef4444;
            --yellow: #eab308;
            --border: #3f3f46;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); height: 100vh; overflow: hidden; }

        .app { display: grid; grid-template-columns: 280px 1fr 320px; height: 100vh; }

        /* Sidebar izquierdo - Proyectos */
        .sidebar {
            background: var(--bg2);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
        }
        .sidebar-header {
            padding: 16px;
            border-bottom: 1px solid var(--border);
        }
        .sidebar-header h1 {
            font-size: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .sidebar-header h1 .icon {
            width: 28px; height: 28px;
            background: linear-gradient(135deg, var(--accent), #ec4899);
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
        }

        .project-list { flex: 1; overflow-y: auto; padding: 8px; }
        .project-item {
            padding: 12px;
            border-radius: 8px;
            cursor: pointer;
            margin-bottom: 4px;
            border: 1px solid transparent;
        }
        .project-item:hover { background: var(--bg3); }
        .project-item.active { background: var(--bg3); border-color: var(--accent); }
        .project-item h3 { font-size: 13px; margin-bottom: 4px; }
        .project-item p { font-size: 11px; color: var(--text2); }
        .project-item .status {
            display: inline-block;
            width: 8px; height: 8px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .project-item .status.running { background: var(--green); }
        .project-item .status.stopped { background: var(--text2); }

        .btn {
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
            border: 1px solid var(--border);
            background: var(--bg3);
            color: var(--text);
            font-weight: 500;
        }
        .btn:hover { background: var(--border); }
        .btn-primary { background: var(--accent); border-color: var(--accent); }
        .btn-sm { padding: 4px 8px; font-size: 11px; }
        .btn-success { background: var(--green); border-color: var(--green); }
        .btn-danger { background: var(--red); border-color: var(--red); }

        .new-project-btn {
            margin: 8px;
            padding: 12px;
            text-align: center;
        }

        /* Main area */
        .main { display: flex; flex-direction: column; }

        .main-header {
            padding: 12px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg2);
        }
        .main-header h2 { font-size: 14px; }
        .main-header .actions { display: flex; gap: 8px; }

        .chat { flex: 1; overflow-y: auto; padding: 20px; }
        .message { margin-bottom: 16px; animation: fadeIn 0.2s; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } }
        .message.user { display: flex; justify-content: flex-end; }
        .message.user .bubble { background: var(--accent); max-width: 70%; }
        .bubble {
            background: var(--bg2);
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 85%;
            border: 1px solid var(--border);
            font-size: 14px;
            line-height: 1.5;
        }
        .bubble pre { background: var(--bg); padding: 10px; border-radius: 6px; margin: 8px 0; overflow-x: auto; }
        .bubble code { font-family: 'JetBrains Mono', monospace; font-size: 12px; }

        .status-bar {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px 16px;
            background: var(--bg2);
            border-radius: 10px;
            border: 1px solid var(--border);
        }
        .spinner { width: 18px; height: 18px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.7s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }

        .file-notification {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 12px;
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid var(--green);
            border-radius: 6px;
            font-size: 12px;
            color: var(--green);
        }

        .input-area {
            padding: 12px 20px;
            border-top: 1px solid var(--border);
            background: var(--bg2);
        }
        .input-row { display: flex; gap: 8px; }
        .input-wrapper {
            flex: 1;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 14px;
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
        }
        .send-btn {
            width: 44px; height: 44px;
            background: var(--accent);
            border: none;
            border-radius: 10px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .send-btn:disabled { opacity: 0.5; }

        /* Panel derecho - Files & Preview */
        .right-panel {
            background: var(--bg2);
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
        }
        .panel-tabs {
            display: flex;
            border-bottom: 1px solid var(--border);
        }
        .panel-tab {
            flex: 1;
            padding: 10px;
            text-align: center;
            font-size: 12px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            color: var(--text2);
        }
        .panel-tab:hover { color: var(--text); }
        .panel-tab.active { color: var(--accent); border-bottom-color: var(--accent); }

        .panel-content { flex: 1; overflow: hidden; display: none; flex-direction: column; }
        .panel-content.active { display: flex; }

        .files-list { flex: 1; overflow-y: auto; padding: 8px; }
        .file-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 10px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
        }
        .file-item:hover { background: var(--bg3); }
        .file-item.active { background: var(--bg3); }

        .preview-frame {
            flex: 1;
            background: white;
            border-radius: 8px;
            margin: 8px;
        }
        .preview-frame iframe { width: 100%; height: 100%; border: none; border-radius: 8px; }

        .terminal {
            flex: 1;
            background: var(--bg);
            margin: 8px;
            border-radius: 8px;
            padding: 10px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            overflow-y: auto;
            white-space: pre-wrap;
            color: var(--text2);
        }

        .file-preview-panel {
            border-top: 1px solid var(--border);
            max-height: 40%;
            display: flex;
            flex-direction: column;
        }
        .file-preview-header {
            padding: 8px 12px;
            background: var(--bg3);
            font-size: 12px;
            display: flex;
            justify-content: space-between;
        }
        .file-preview-content {
            flex: 1;
            overflow: auto;
            padding: 10px;
            background: var(--bg);
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            white-space: pre-wrap;
        }

        /* Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.8);
            z-index: 100;
            align-items: center;
            justify-content: center;
        }
        .modal-overlay.active { display: flex; }
        .modal {
            background: var(--bg2);
            border-radius: 12px;
            padding: 24px;
            width: 500px;
            max-height: 80vh;
            overflow-y: auto;
            border: 1px solid var(--border);
        }
        .modal h3 { margin-bottom: 16px; }
        .modal input, .modal textarea, .modal select {
            width: 100%;
            padding: 10px 12px;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            font-size: 14px;
            margin-bottom: 12px;
        }
        .modal-actions { display: flex; gap: 8px; justify-content: flex-end; margin-top: 16px; }

        .template-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-bottom: 16px;
        }
        .template-card {
            padding: 14px;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            cursor: pointer;
            text-align: center;
        }
        .template-card:hover { border-color: var(--accent); }
        .template-card.selected { border-color: var(--accent); background: rgba(139, 92, 246, 0.1); }
        .template-card .icon { font-size: 24px; margin-bottom: 6px; }
        .template-card .name { font-size: 13px; font-weight: 500; }
        .template-card .desc { font-size: 11px; color: var(--text2); margin-top: 4px; }

        .welcome {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 40px;
        }
        .welcome-icon { font-size: 60px; margin-bottom: 20px; }
        .welcome h2 { margin-bottom: 10px; }
        .welcome p { color: var(--text2); max-width: 350px; }

        .empty-state { padding: 30px; text-align: center; color: var(--text2); font-size: 13px; }
    </style>
</head>
<body>
    <div class="app">
        <!-- Sidebar - Proyectos -->
        <aside class="sidebar">
            <div class="sidebar-header">
                <h1><span class="icon">🚀</span> THAU IDE</h1>
            </div>
            <div class="project-list" id="projectList">
                <div class="empty-state">No hay proyectos</div>
            </div>
            <div class="new-project-btn">
                <button class="btn btn-primary" style="width:100%" onclick="showNewProjectModal()">+ Nuevo Proyecto</button>
            </div>
        </aside>

        <!-- Main - Chat -->
        <main class="main">
            <header class="main-header">
                <h2 id="currentProjectName">Selecciona un proyecto</h2>
                <div class="actions">
                    <button class="btn btn-sm btn-success" id="runBtn" onclick="runProject()" style="display:none">▶ Run</button>
                    <button class="btn btn-sm btn-danger" id="stopBtn" onclick="stopProject()" style="display:none">■ Stop</button>
                </div>
            </header>

            <div class="chat" id="chat">
                <div class="welcome">
                    <div class="welcome-icon">💻</div>
                    <h2>THAU IDE</h2>
                    <p>Crea un nuevo proyecto o selecciona uno existente para comenzar a desarrollar.</p>
                </div>
            </div>

            <div class="input-area">
                <div class="input-row">
                    <div class="input-wrapper">
                        <textarea id="input" placeholder="Describe qué quieres crear..." rows="2" onkeydown="handleKey(event)"></textarea>
                    </div>
                    <button class="send-btn" id="sendBtn" onclick="send()">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5">
                            <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                        </svg>
                    </button>
                </div>
            </div>
        </main>

        <!-- Panel derecho -->
        <aside class="right-panel">
            <div class="panel-tabs">
                <div class="panel-tab active" onclick="switchPanel('files')">📁 Files</div>
                <div class="panel-tab" onclick="switchPanel('preview')">👁 Preview</div>
                <div class="panel-tab" onclick="switchPanel('terminal')">⌨ Terminal</div>
            </div>

            <div class="panel-content active" id="filesPanel">
                <div class="files-list" id="filesList">
                    <div class="empty-state">Los archivos aparecerán aquí</div>
                </div>
                <div class="file-preview-panel" id="filePreview" style="display:none">
                    <div class="file-preview-header">
                        <span id="previewFileName">archivo.py</span>
                        <span id="previewFileSize" style="color:var(--text2)">0 bytes</span>
                    </div>
                    <pre class="file-preview-content" id="previewFileContent"></pre>
                </div>
            </div>

            <div class="panel-content" id="previewPanel">
                <div class="preview-frame" id="previewFrame">
                    <iframe id="previewIframe" src="about:blank"></iframe>
                </div>
            </div>

            <div class="panel-content" id="terminalPanel">
                <div class="terminal" id="terminalOutput">$ Esperando proyecto...</div>
            </div>
        </aside>
    </div>

    <!-- Modal Nuevo Proyecto -->
    <div class="modal-overlay" id="newProjectModal">
        <div class="modal">
            <h3>🚀 Nuevo Proyecto</h3>

            <label style="font-size:12px;color:var(--text2);margin-bottom:6px;display:block">Template</label>
            <div class="template-grid" id="templateGrid"></div>

            <input type="text" id="newProjectName" placeholder="Nombre del proyecto">
            <textarea id="newProjectDesc" placeholder="Describe qué quieres crear (opcional)..." rows="3"></textarea>
            <input type="text" id="newProjectPath" placeholder="Ruta (opcional)">

            <div class="modal-actions">
                <button class="btn" onclick="hideModal('newProjectModal')">Cancelar</button>
                <button class="btn btn-primary" onclick="createProject()">Crear Proyecto</button>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let currentProject = null;
        let templates = {};
        let selectedTemplate = null;
        let isGenerating = false;
        let currentBubble = null;
        let statusDiv = null;
        let terminalInterval = null;

        document.addEventListener('DOMContentLoaded', () => {
            connectWS();
            loadProjects();
            loadTemplates();
        });

        function connectWS() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws/dev`);
            ws.onopen = () => console.log('✅ Connected');
            ws.onclose = () => setTimeout(connectWS, 2000);
            ws.onmessage = handleMessage;
        }

        function handleMessage(e) {
            const data = JSON.parse(e.data);

            switch(data.type) {
                case 'start':
                    showStatus('Generando código...');
                    break;
                case 'stream':
                    if (!currentBubble) currentBubble = addBubble('assistant');
                    updateBubble(currentBubble, data.full);
                    break;
                case 'file_created':
                    addFileNotification(data.file);
                    refreshFiles();
                    break;
                case 'done':
                    hideStatus();
                    if (data.total > 0) addSystemMessage(`✅ ${data.total} archivo(s) creado(s)`);
                    finishGeneration();
                    break;
                case 'error':
                    hideStatus();
                    addSystemMessage(`❌ ${data.error}`);
                    finishGeneration();
                    break;
            }
        }

        async function loadProjects() {
            const res = await fetch('/api/projects');
            const projects = await res.json();
            const list = document.getElementById('projectList');

            if (!projects.length) {
                list.innerHTML = '<div class="empty-state">Crea tu primer proyecto</div>';
                return;
            }

            list.innerHTML = projects.map(p => `
                <div class="project-item ${currentProject?.id === p.id ? 'active' : ''}" onclick="selectProject('${p.id}')">
                    <h3><span class="status ${p.id && false ? 'running' : 'stopped'}"></span>${p.name}</h3>
                    <p>${p.framework || p.type || 'Proyecto'}</p>
                </div>
            `).join('');
        }

        async function loadTemplates() {
            const res = await fetch('/api/templates');
            templates = await res.json();

            const grid = document.getElementById('templateGrid');
            grid.innerHTML = Object.entries(templates).map(([id, t]) => `
                <div class="template-card" data-id="${id}" onclick="selectTemplate('${id}')">
                    <div class="icon">${t.icon}</div>
                    <div class="name">${t.name}</div>
                    <div class="desc">${t.description}</div>
                </div>
            `).join('');
        }

        function selectTemplate(id) {
            selectedTemplate = id;
            document.querySelectorAll('.template-card').forEach(el => {
                el.classList.toggle('selected', el.dataset.id === id);
            });
        }

        async function selectProject(id) {
            const res = await fetch(`/api/projects/${id}`);
            currentProject = await res.json();

            await fetch(`/api/projects/${id}/select`, { method: 'POST' });

            document.getElementById('currentProjectName').textContent = currentProject.name;
            document.getElementById('runBtn').style.display = currentProject.run_command ? 'block' : 'none';

            loadProjects();
            refreshFiles();
            checkProjectStatus();

            // Limpiar chat
            document.getElementById('chat').innerHTML = '';
        }

        async function createProject() {
            const name = document.getElementById('newProjectName').value.trim() || 'nuevo-proyecto';
            const desc = document.getElementById('newProjectDesc').value.trim();
            const path = document.getElementById('newProjectPath').value.trim();

            const res = await fetch('/api/projects', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name,
                    description: desc,
                    path: path || undefined,
                    template: selectedTemplate
                })
            });

            const project = await res.json();
            currentProject = project;

            hideModal('newProjectModal');
            loadProjects();

            document.getElementById('currentProjectName').textContent = project.name;
            document.getElementById('runBtn').style.display = project.run_command ? 'block' : 'none';
            document.getElementById('chat').innerHTML = '';

            refreshFiles();

            // Si hay descripción, generar código
            if (desc) {
                send(desc);
            }

            // Reset form
            document.getElementById('newProjectName').value = '';
            document.getElementById('newProjectDesc').value = '';
            document.getElementById('newProjectPath').value = '';
            selectedTemplate = null;
            document.querySelectorAll('.template-card').forEach(el => el.classList.remove('selected'));
        }

        async function runProject() {
            if (!currentProject) return;

            const res = await fetch(`/api/projects/${currentProject.id}/run`, { method: 'POST' });
            const result = await res.json();

            if (result.status === 'started' || result.status === 'already_running') {
                document.getElementById('runBtn').style.display = 'none';
                document.getElementById('stopBtn').style.display = 'block';

                addSystemMessage(`🚀 Proyecto iniciado en puerto ${result.port || 'N/A'}`);

                if (result.port) {
                    setTimeout(() => {
                        document.getElementById('previewIframe').src = `http://localhost:${result.port}`;
                        switchPanel('preview');
                    }, 3000);
                }

                startTerminalPolling();
            } else {
                addSystemMessage(`❌ Error: ${result.error || 'No se pudo iniciar'}`);
            }
        }

        async function stopProject() {
            if (!currentProject) return;

            await fetch(`/api/projects/${currentProject.id}/stop`, { method: 'POST' });

            document.getElementById('runBtn').style.display = 'block';
            document.getElementById('stopBtn').style.display = 'none';

            addSystemMessage('⏹ Proyecto detenido');
            stopTerminalPolling();
        }

        async function checkProjectStatus() {
            if (!currentProject) return;

            const res = await fetch(`/api/projects/${currentProject.id}/status`);
            const status = await res.json();

            if (status.running) {
                document.getElementById('runBtn').style.display = 'none';
                document.getElementById('stopBtn').style.display = 'block';
                startTerminalPolling();
            } else {
                document.getElementById('runBtn').style.display = currentProject.run_command ? 'block' : 'none';
                document.getElementById('stopBtn').style.display = 'none';
            }
        }

        function startTerminalPolling() {
            if (terminalInterval) return;
            terminalInterval = setInterval(updateTerminal, 1000);
        }

        function stopTerminalPolling() {
            if (terminalInterval) {
                clearInterval(terminalInterval);
                terminalInterval = null;
            }
        }

        async function updateTerminal() {
            if (!currentProject) return;

            const res = await fetch(`/api/projects/${currentProject.id}/output`);
            const data = await res.json();

            const terminal = document.getElementById('terminalOutput');
            terminal.textContent = data.output.join('') || '$ Esperando output...';
            terminal.scrollTop = terminal.scrollHeight;
        }

        async function refreshFiles() {
            const res = await fetch('/api/files');
            const files = await res.json();
            const list = document.getElementById('filesList');

            if (!files.length) {
                list.innerHTML = '<div class="empty-state">No hay archivos</div>';
                return;
            }

            list.innerHTML = files.filter(f => f.type === 'file').map(f => `
                <div class="file-item" onclick="previewFile('${f.path}')">📄 ${f.path}</div>
            `).join('');
        }

        async function previewFile(path) {
            const res = await fetch(`/api/files/${encodeURIComponent(path)}`);
            const data = await res.json();

            document.getElementById('previewFileName').textContent = path;
            document.getElementById('previewFileSize').textContent = `${data.content.length} bytes`;
            document.getElementById('previewFileContent').textContent = data.content;
            document.getElementById('filePreview').style.display = 'flex';
        }

        function switchPanel(panel) {
            document.querySelectorAll('.panel-tab').forEach((el, i) => {
                el.classList.toggle('active', ['files', 'preview', 'terminal'][i] === panel);
            });
            document.querySelectorAll('.panel-content').forEach(el => {
                el.classList.toggle('active', el.id === panel + 'Panel');
            });
        }

        function send(text = null) {
            const input = document.getElementById('input');
            const prompt = text || input.value.trim();
            if (!prompt || isGenerating || !ws || !currentProject) return;

            addUserBubble(prompt);
            ws.send(JSON.stringify({ prompt, template: currentProject.framework }));

            if (!text) input.value = '';
            isGenerating = true;
            document.getElementById('sendBtn').disabled = true;
        }

        function handleKey(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
            }
        }

        function showStatus(text) {
            if (statusDiv) return;
            const chat = document.getElementById('chat');
            statusDiv = document.createElement('div');
            statusDiv.className = 'status-bar';
            statusDiv.innerHTML = `<div class="spinner"></div><span>${text}</span>`;
            chat.appendChild(statusDiv);
            chat.scrollTop = chat.scrollHeight;
        }

        function hideStatus() {
            if (statusDiv) { statusDiv.remove(); statusDiv = null; }
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

        function addUserBubble(text) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'message user';
            div.innerHTML = `<div class="bubble">${text}</div>`;
            chat.appendChild(div);
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

        function addFileNotification(file) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'file-notification';
            div.innerHTML = `📄 <strong>${file.path}</strong> (${file.size} bytes)`;
            chat.appendChild(div);
        }

        function addSystemMessage(text) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'message';
            div.innerHTML = `<div class="bubble" style="background:var(--bg3)">${text}</div>`;
            chat.appendChild(div);
        }

        function finishGeneration() {
            currentBubble = null;
            isGenerating = false;
            document.getElementById('sendBtn').disabled = false;
            refreshFiles();
        }

        function showNewProjectModal() {
            document.getElementById('newProjectModal').classList.add('active');
        }

        function hideModal(id) {
            document.getElementById(id).classList.remove('active');
        }
    </script>
</body>
</html>
'''

if __name__ == "__main__":
    Path(DEFAULT_WORKSPACE).mkdir(parents=True, exist_ok=True)

    print(f"""
╔════════════════════════════════════════════════════════════════════════╗
║                           THAU IDE                                      ║
║                                                                        ║
║  🚀 Entorno de desarrollo integrado con THAU                           ║
║  📁 Gestión de proyectos con SQLite                                    ║
║  🎨 Templates: React, Vue, Next.js, Flask, FastAPI, Express            ║
║  ▶️  Ejecución y preview en vivo                                        ║
║  ⌨️  Terminal integrada                                                 ║
║                                                                        ║
║  URL: http://localhost:{PORT}                                            ║
║  Workspace: {DEFAULT_WORKSPACE}
║  Database: {DB_PATH}
╚════════════════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
