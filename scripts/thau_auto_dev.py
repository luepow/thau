#!/usr/bin/env python3
"""
THAU Auto Developer - Sistema que interpreta las intenciones de THAU
y ejecuta las acciones automáticamente.

En lugar de depender de que THAU genere formatos específicos de herramientas,
este sistema analiza la respuesta y extrae bloques de código para crear archivos.
"""

import json
import re
import subprocess
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Generator
from dataclasses import dataclass
from datetime import datetime
import queue
import concurrent.futures

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
PORT = 7864

# ============================================================================
# FILE CREATOR - Extrae código y crea archivos automáticamente
# ============================================================================

class AutoFileCreator:
    """Extrae bloques de código de las respuestas y crea archivos."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.project_path.mkdir(parents=True, exist_ok=True)
        self.files_created: List[str] = []

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

        # Eliminar duplicados por filename
        seen = set()
        unique_blocks = []
        for block in blocks:
            if block["filename"] not in seen:
                seen.add(block["filename"])
                unique_blocks.append(block)

        return unique_blocks

    def create_file(self, filename: str, content: str) -> Dict:
        """Crea un archivo en el proyecto."""
        try:
            file_path = self.project_path / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            self.files_created.append(filename)
            logger.info(f"Archivo creado: {filename} ({len(content)} bytes)")

            return {"success": True, "path": filename, "size": len(content)}
        except Exception as e:
            logger.error(f"Error creando {filename}: {e}")
            return {"success": False, "path": filename, "error": str(e)}

    def create_directory(self, dirname: str) -> Dict:
        """Crea un directorio."""
        try:
            dir_path = self.project_path / dirname
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directorio creado: {dirname}")
            return {"success": True, "path": dirname}
        except Exception as e:
            return {"success": False, "path": dirname, "error": str(e)}

    def process_response(self, text: str) -> List[Dict]:
        """Procesa una respuesta y crea los archivos encontrados."""
        results = []
        blocks = self.extract_code_blocks(text)

        for block in blocks:
            result = self.create_file(block["filename"], block["content"])
            results.append(result)

        return results

    def list_files(self) -> List[Dict]:
        """Lista archivos del proyecto."""
        files = []
        try:
            for item in sorted(self.project_path.rglob("*")):
                if item.name.startswith('.') or '__pycache__' in str(item):
                    continue
                try:
                    rel = item.relative_to(self.project_path)
                    if item.is_file():
                        files.append({
                            "path": str(rel),
                            "type": "file",
                            "size": item.stat().st_size
                        })
                    else:
                        files.append({"path": str(rel), "type": "directory"})
                except:
                    pass
        except:
            pass
        return files

    def read_file(self, path: str) -> Optional[str]:
        """Lee un archivo."""
        try:
            file_path = self.project_path / path
            if file_path.exists():
                return file_path.read_text(encoding='utf-8')
        except:
            pass
        return None


# ============================================================================
# AUTO DEVELOPER
# ============================================================================

class AutoDeveloper:
    """Desarrollador automático que interpreta y ejecuta."""

    SYSTEM_PROMPT = """Eres THAU, un desarrollador de software experto.

Tu trabajo es crear aplicaciones completas. Cuando te pidan crear un proyecto:

1. PLANIFICA: Describe brevemente qué vas a crear
2. ESTRUCTURA: Lista las carpetas y archivos necesarios
3. CÓDIGO: Genera el código completo para CADA archivo

IMPORTANTE - FORMATO DE CÓDIGO:
Para cada archivo, usa EXACTAMENTE este formato:

**nombre_archivo.py**
```python
# código completo aquí
```

Ejemplos:

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

**templates/index.html**
```html
<!DOCTYPE html>
<html>
<head><title>Mi App</title></head>
<body><h1>Bienvenido</h1></body>
</html>
```

**requirements.txt**
```
flask==2.0.0
```

REGLAS:
- Genera código COMPLETO, no fragmentos
- Usa el formato **nombre.ext** antes de cada bloque de código
- Incluye TODOS los imports necesarios
- El código debe funcionar sin modificaciones
- Responde siempre en español

Comienza ahora con el proyecto solicitado."""

    def __init__(self, project_path: str):
        self.file_creator = AutoFileCreator(project_path)
        self.history: List[Dict] = []

    def set_project_path(self, path: str):
        """Cambia el directorio del proyecto."""
        self.file_creator = AutoFileCreator(path)

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
                        "num_ctx": 4096,
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
        existing_files = self.file_creator.list_files()
        context = ""
        if existing_files:
            context = "\n\nArchivos existentes en el proyecto:\n"
            context += "\n".join([f"- {f['path']}" for f in existing_files[:20]])

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]

        # Agregar historial
        for msg in self.history[-4:]:
            messages.append(msg)

        # Agregar prompt
        full_prompt = f"{prompt}{context}"
        messages.append({"role": "user", "content": full_prompt})

        yield {"type": "start", "message": "Iniciando desarrollo..."}

        # Generar respuesta
        full_response = ""
        accumulated = ""

        for chunk in self.call_ollama(messages):
            full_response += chunk
            accumulated += chunk

            yield {"type": "stream", "content": chunk, "full": full_response}

            # Buscar archivos completados mientras se genera
            # (esto permite crear archivos a medida que aparecen)
            if "```" in accumulated:
                # Verificar si hay bloques completos
                blocks = self.file_creator.extract_code_blocks(accumulated)
                for block in blocks:
                    # Verificar que este archivo no se haya creado ya
                    if block["filename"] not in [f["path"] for f in self.file_creator.files_created if isinstance(f, dict)] and block["filename"] not in self.file_creator.files_created:
                        result = self.file_creator.create_file(block["filename"], block["content"])
                        yield {
                            "type": "file_created",
                            "file": result
                        }

        # Procesar archivos finales que pudieron haberse perdido
        final_blocks = self.file_creator.extract_code_blocks(full_response)
        for block in final_blocks:
            if block["filename"] not in self.file_creator.files_created:
                result = self.file_creator.create_file(block["filename"], block["content"])
                yield {"type": "file_created", "file": result}

        # Guardar en historial
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": full_response})

        yield {
            "type": "done",
            "files_created": self.file_creator.files_created.copy(),
            "total": len(self.file_creator.files_created)
        }

    def get_files(self) -> List[Dict]:
        return self.file_creator.list_files()

    def read_file(self, path: str) -> Optional[str]:
        return self.file_creator.read_file(path)

    def clear(self):
        self.history = []
        self.file_creator.files_created = []


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="THAU Auto Developer")
developer = AutoDeveloper(DEFAULT_PROJECT_DIR)


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
    return {"path": str(developer.file_creator.project_path)}


@app.get("/api/project/path")
async def get_path():
    return {"path": str(developer.file_creator.project_path)}


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
    <title>THAU Auto Developer</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg: #0d1117;
            --bg2: #161b22;
            --bg3: #21262d;
            --text: #c9d1d9;
            --text2: #8b949e;
            --accent: #58a6ff;
            --green: #3fb950;
            --border: #30363d;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); height: 100vh; }

        .app { display: grid; grid-template-columns: 1fr 350px; height: 100vh; }

        .main { display: flex; flex-direction: column; }
        .header {
            padding: 16px 24px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header h1 { font-size: 20px; display: flex; align-items: center; gap: 10px; }
        .header-actions { display: flex; gap: 8px; }
        .btn {
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            border: 1px solid var(--border);
            background: var(--bg2);
            color: var(--text);
            transition: all 0.15s;
        }
        .btn:hover { background: var(--bg3); }
        .btn-primary { background: var(--accent); border-color: var(--accent); color: #000; }
        .btn-primary:hover { opacity: 0.9; }

        .chat { flex: 1; overflow-y: auto; padding: 24px; }

        .message { margin-bottom: 24px; animation: fadeIn 0.3s; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } }
        .message.user { text-align: right; }
        .message.user .bubble { background: var(--accent); color: #000; display: inline-block; text-align: left; }
        .bubble {
            background: var(--bg2);
            padding: 16px;
            border-radius: 12px;
            max-width: 90%;
            display: inline-block;
        }
        .bubble pre { background: var(--bg); padding: 12px; border-radius: 8px; overflow-x: auto; margin: 8px 0; }
        .bubble code { font-family: 'JetBrains Mono', monospace; font-size: 13px; }

        /* Status indicator */
        .status {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 16px;
            background: var(--bg2);
            border-radius: 12px;
            margin-bottom: 16px;
        }
        .spinner {
            width: 24px;
            height: 24px;
            border: 3px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .status-text { color: var(--text2); }

        /* File created notification */
        .file-created {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 14px;
            background: rgba(63, 185, 80, 0.1);
            border: 1px solid var(--green);
            border-radius: 8px;
            margin: 8px 0;
            font-size: 13px;
            color: var(--green);
        }

        .input-area {
            padding: 16px 24px;
            border-top: 1px solid var(--border);
        }
        .input-row { display: flex; gap: 12px; }
        .input-wrapper {
            flex: 1;
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px 16px;
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
            width: 48px;
            height: 48px;
            background: var(--accent);
            border: none;
            border-radius: 10px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .send-btn:disabled { background: var(--bg3); cursor: not-allowed; }

        /* Files panel */
        .files-panel {
            background: var(--bg2);
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
        }
        .files-header {
            padding: 16px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .files-header h2 { font-size: 14px; }
        .files-list { flex: 1; overflow-y: auto; padding: 8px; }
        .file-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            margin-bottom: 2px;
        }
        .file-item:hover { background: var(--bg3); }
        .file-item.active { background: var(--bg3); border-left: 2px solid var(--accent); }
        .file-item.new { animation: highlight 2s ease-out; }
        @keyframes highlight {
            0%, 50% { background: rgba(63, 185, 80, 0.2); }
        }

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
        }
        .preview-content {
            flex: 1;
            overflow: auto;
            padding: 12px;
            background: var(--bg);
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
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
        }
        .welcome-icon {
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, #58a6ff, #a855f7);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 40px;
            margin-bottom: 24px;
        }
        .welcome h2 { margin-bottom: 8px; }
        .welcome p { color: var(--text2); max-width: 400px; margin-bottom: 24px; }
        .templates { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; }
        .template {
            padding: 12px 20px;
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.15s;
        }
        .template:hover { border-color: var(--accent); background: var(--bg3); }

        /* Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.7);
            z-index: 100;
            align-items: center;
            justify-content: center;
        }
        .modal-overlay.active { display: flex; }
        .modal {
            background: var(--bg2);
            border-radius: 12px;
            padding: 24px;
            width: 450px;
        }
        .modal h3 { margin-bottom: 16px; }
        .modal input, .modal textarea {
            width: 100%;
            padding: 12px;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            font-size: 14px;
            margin-bottom: 12px;
        }
        .modal input:focus, .modal textarea:focus { outline: none; border-color: var(--accent); }
        .modal-actions { display: flex; gap: 12px; justify-content: flex-end; margin-top: 16px; }

        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
    </style>
</head>
<body>
    <div class="app">
        <div class="main">
            <header class="header">
                <h1>🤖 THAU Auto Developer</h1>
                <div class="header-actions">
                    <button class="btn" onclick="clearChat()">Limpiar</button>
                    <button class="btn btn-primary" onclick="showModal()">+ Nuevo Proyecto</button>
                </div>
            </header>

            <div class="chat" id="chat">
                <div class="welcome" id="welcome">
                    <div class="welcome-icon">🚀</div>
                    <h2>THAU Auto Developer</h2>
                    <p>Describe tu proyecto y THAU generará todo el código. Los archivos se crean automáticamente.</p>
                    <div class="templates">
                        <div class="template" onclick="quickStart('flask')">🌐 Web Flask</div>
                        <div class="template" onclick="quickStart('fastapi')">⚡ API FastAPI</div>
                        <div class="template" onclick="quickStart('cli')">💻 CLI Python</div>
                        <div class="template" onclick="quickStart('html')">📄 HTML/CSS/JS</div>
                    </div>
                </div>
            </div>

            <div class="input-area">
                <div class="input-row">
                    <div class="input-wrapper">
                        <textarea id="input" placeholder="Describe qué aplicación quieres crear..." rows="2" onkeydown="handleKey(event)"></textarea>
                    </div>
                    <button class="send-btn" id="sendBtn" onclick="send()">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                            <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                        </svg>
                    </button>
                </div>
            </div>
        </div>

        <aside class="files-panel">
            <div class="files-header">
                <h2>📁 Archivos del Proyecto</h2>
                <button class="btn" style="padding:4px 8px;font-size:12px" onclick="refreshFiles()">🔄</button>
            </div>
            <div class="files-list" id="filesList">
                <div style="padding:20px;text-align:center;color:var(--text2)">Los archivos aparecerán aquí</div>
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
            loadPath();
        });

        function connectWS() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws/dev`);
            ws.onopen = () => console.log('Connected');
            ws.onclose = () => setTimeout(connectWS, 2000);
            ws.onmessage = handleMessage;
        }

        function handleMessage(e) {
            const data = JSON.parse(e.data);
            console.log('MSG:', data.type);

            switch(data.type) {
                case 'start':
                    showStatus('Generando código...');
                    break;

                case 'stream':
                    if (!currentBubble) {
                        currentBubble = addBubble('assistant');
                    }
                    updateBubble(currentBubble, data.full);
                    break;

                case 'file_created':
                    addFileNotification(data.file);
                    refreshFiles();
                    break;

                case 'done':
                    hideStatus();
                    if (data.total > 0) {
                        addSystemMessage(`✅ ${data.total} archivo(s) creado(s)`);
                    }
                    currentBubble = null;
                    isGenerating = false;
                    document.getElementById('sendBtn').disabled = false;
                    break;

                case 'error':
                    hideStatus();
                    addSystemMessage(`❌ Error: ${data.error}`);
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

        function addFileNotification(file) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'file-created';
            div.innerHTML = `📄 <strong>${file.path}</strong> creado (${file.size || 0} bytes)`;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
        }

        function addSystemMessage(text) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = 'message';
            div.innerHTML = `<div class="bubble" style="background:var(--bg3)">${text}</div>`;
            chat.appendChild(div);
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
                list.innerHTML = '<div style="padding:20px;text-align:center;color:var(--text2)">No hay archivos</div>';
                return;
            }

            list.innerHTML = files.filter(f => f.type === 'file').map(f => `
                <div class="file-item new" onclick="previewFile('${f.path}')">
                    📄 ${f.path}
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

        function showModal() {
            document.getElementById('modal').classList.add('active');
        }

        function hideModal() {
            document.getElementById('modal').classList.remove('active');
        }

        function createProject() {
            const name = document.getElementById('projectName').value.trim();
            const desc = document.getElementById('projectDescription')?.value.trim() ||
                         document.getElementById('projectDesc').value.trim();

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
                flask: 'Crea una aplicación web con Flask que tenga: página de inicio con diseño moderno (usando Bootstrap), página about, y un formulario de contacto que guarde en SQLite. Incluye todos los archivos necesarios.',
                fastapi: 'Crea una API REST con FastAPI que tenga: endpoints CRUD para gestionar tareas (crear, leer, actualizar, eliminar), modelo Pydantic, y documentación automática. Incluye requirements.txt.',
                cli: 'Crea una herramienta CLI en Python usando Click que permita: agregar tareas, listarlas, marcarlas como completadas, y eliminarlas. Usa SQLite para persistencia y Rich para colores en terminal.',
                html: 'Crea una página web moderna con HTML, CSS y JavaScript que tenga: landing page con hero section, sección de características, galería de imágenes con lightbox, formulario de contacto, y footer. Usa diseño responsive.'
            };
            document.getElementById('projectDesc').value = prompts[type];
            showModal();
        }

        async function clearChat() {
            await fetch('/api/clear', {method: 'POST'});
            document.getElementById('chat').innerHTML = `
                <div class="welcome" id="welcome">
                    <div class="welcome-icon">🚀</div>
                    <h2>THAU Auto Developer</h2>
                    <p>Describe tu proyecto y THAU generará todo el código.</p>
                    <div class="templates">
                        <div class="template" onclick="quickStart('flask')">🌐 Web Flask</div>
                        <div class="template" onclick="quickStart('fastapi')">⚡ API FastAPI</div>
                        <div class="template" onclick="quickStart('cli')">💻 CLI Python</div>
                        <div class="template" onclick="quickStart('html')">📄 HTML/CSS/JS</div>
                    </div>
                </div>
            `;
            refreshFiles();
        }
    </script>
</body>
</html>
'''


if __name__ == "__main__":
    Path(DEFAULT_PROJECT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║               THAU Auto Developer                              ║
║                                                               ║
║  🤖 Genera código y crea archivos automáticamente             ║
║                                                               ║
║  URL: http://localhost:{PORT}                                   ║
║  Proyecto: {DEFAULT_PROJECT_DIR}
╚═══════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
