#!/usr/bin/env python3
"""
THAU Developer Final - Sistema robusto de desarrollo con MCP

Versión optimizada que:
1. Extrae código de CUALQUIER formato que THAU genere
2. Crea archivos automáticamente sin depender de formato específico
3. Tiene timeout handling para evitar cuelgues
4. Incluye herramientas MCP para desarrollo
"""

import json
import re
import subprocess
import os
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional, Generator, Any
from dataclasses import dataclass
from datetime import datetime
import queue
import concurrent.futures
import threading
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn
from loguru import logger
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

OLLAMA_MODEL = "thau:unified"
OLLAMA_URL = "http://localhost:11434"
DEFAULT_PROJECT_DIR = os.path.expanduser("~/thau_projects")
PORT = 7866
GENERATION_TIMEOUT = 180  # 3 minutos máximo

# ============================================================================
# MCP TOOLS
# ============================================================================

class MCPTools:
    """Herramientas MCP para desarrollo"""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.project_path.mkdir(parents=True, exist_ok=True)
        self.files_created: List[str] = []

    def set_project_path(self, path: str):
        self.project_path = Path(path)
        self.project_path.mkdir(parents=True, exist_ok=True)
        self.files_created = []

    def create_file(self, path: str, content: str) -> Dict:
        """Crea un archivo"""
        try:
            file_path = self.project_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            self.files_created.append(path)
            logger.info(f"✅ Archivo creado: {path} ({len(content)} bytes)")

            return {"success": True, "path": path, "size": len(content)}
        except Exception as e:
            logger.error(f"❌ Error creando {path}: {e}")
            return {"success": False, "error": str(e)}

    def create_directory(self, path: str) -> Dict:
        """Crea un directorio"""
        try:
            dir_path = self.project_path / path
            dir_path.mkdir(parents=True, exist_ok=True)
            return {"success": True, "path": path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_files(self) -> List[Dict]:
        """Lista archivos del proyecto"""
        files = []
        try:
            for item in sorted(self.project_path.rglob("*")):
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
        except:
            pass
        return files

    def read_file(self, path: str) -> Optional[str]:
        """Lee un archivo"""
        try:
            file_path = self.project_path / path
            if file_path.exists():
                return file_path.read_text(encoding='utf-8')
        except:
            pass
        return None

    def execute_command(self, command: str) -> Dict:
        """Ejecuta un comando"""
        try:
            result = subprocess.run(
                command, shell=True, cwd=str(self.project_path),
                capture_output=True, text=True, timeout=30
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout[:2000],
                "stderr": result.stderr[:500]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# CODE EXTRACTOR - Extrae código de CUALQUIER formato
# ============================================================================

class CodeExtractor:
    """Extractor agresivo de código que funciona con cualquier formato"""

    # Extensiones soportadas
    EXTENSIONS = r'py|js|jsx|ts|tsx|html|css|json|md|txt|yaml|yml|sh|sql|xml|toml|ini|cfg|env'

    @staticmethod
    def extract_all_code_blocks(text: str) -> List[Dict]:
        """Extrae TODOS los bloques de código posibles"""
        blocks = []
        seen_content = set()

        # Lista de patrones ordenados por especificidad
        patterns = [
            # 1. ```lang filename.ext\ncontent```
            (r'```(\w+)\s+([^\n`]+\.(?:' + CodeExtractor.EXTENSIONS + r'))\s*\n(.*?)```',
             lambda m: (m.group(2).strip(), m.group(3).strip(), m.group(1))),

            # 2. **filename.ext**\n```\ncontent```
            (r'\*\*([^\n*]+\.(?:' + CodeExtractor.EXTENSIONS + r'))\*\*[:\s]*\n```(?:\w*)\n(.*?)```',
             lambda m: (m.group(1).strip(), m.group(2).strip(), 'auto')),

            # 3. # filename.ext\n```\ncontent```
            (r'#\s*([^\n]+\.(?:' + CodeExtractor.EXTENSIONS + r'))\s*\n```(?:\w*)\n(.*?)```',
             lambda m: (m.group(1).strip(), m.group(2).strip(), 'auto')),

            # 4. filename.ext:\n```\ncontent```
            (r'([a-zA-Z0-9_/.-]+\.(?:' + CodeExtractor.EXTENSIONS + r')):\s*\n```(?:\w*)\n(.*?)```',
             lambda m: (m.group(1).strip(), m.group(2).strip(), 'auto')),

            # 5. ```lang\n# filename.ext\ncontent```
            (r'```(\w+)\n#\s*([^\n]+\.(?:' + CodeExtractor.EXTENSIONS + r'))\s*\n(.*?)```',
             lambda m: (m.group(2).strip(), m.group(3).strip(), m.group(1))),

            # 6. `filename.ext`\n```\ncontent```
            (r'`([^\n`]+\.(?:' + CodeExtractor.EXTENSIONS + r'))`[:\s]*\n```(?:\w*)\n(.*?)```',
             lambda m: (m.group(1).strip(), m.group(2).strip(), 'auto')),

            # 7. Archivo: filename.ext\n```\ncontent```
            (r'(?:Archivo|File|Crear|Create)[:\s]+([^\n]+\.(?:' + CodeExtractor.EXTENSIONS + r'))\s*\n```(?:\w*)\n(.*?)```',
             lambda m: (m.group(1).strip(), m.group(2).strip(), 'auto')),
        ]

        for pattern, extractor in patterns:
            for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
                try:
                    filename, content, lang = extractor(match)
                    # Limpiar filename
                    filename = filename.strip('`*# \t')
                    # Evitar duplicados por contenido
                    content_hash = hash(content[:200])
                    if content_hash not in seen_content and len(content) > 10:
                        seen_content.add(content_hash)
                        blocks.append({
                            "filename": filename,
                            "content": content,
                            "lang": lang
                        })
                except:
                    continue

        # Si no encontramos archivos con nombre pero hay código, crear archivos genéricos
        if not blocks:
            blocks = CodeExtractor._extract_unnamed_blocks(text)

        return blocks

    @staticmethod
    def _extract_unnamed_blocks(text: str) -> List[Dict]:
        """Extrae bloques sin nombre y les asigna nombres automáticos"""
        blocks = []

        # Buscar bloques ```lang\ncontent```
        pattern = r'```(\w+)\n(.*?)```'

        lang_to_ext = {
            'python': 'py', 'py': 'py',
            'javascript': 'js', 'js': 'js',
            'typescript': 'ts', 'ts': 'ts',
            'html': 'html',
            'css': 'css',
            'json': 'json',
            'yaml': 'yaml', 'yml': 'yaml',
            'bash': 'sh', 'sh': 'sh', 'shell': 'sh',
            'sql': 'sql',
        }

        counters = {}

        for match in re.finditer(pattern, text, re.DOTALL):
            lang = match.group(1).lower()
            content = match.group(2).strip()

            if len(content) < 20:  # Ignorar bloques muy pequeños
                continue

            ext = lang_to_ext.get(lang, 'txt')

            # Intentar inferir nombre del contenido
            filename = CodeExtractor._infer_filename(content, ext)

            if not filename:
                # Generar nombre genérico
                counters[ext] = counters.get(ext, 0) + 1
                if counters[ext] == 1:
                    filename = f"main.{ext}" if ext in ['py', 'js'] else f"file.{ext}"
                else:
                    filename = f"file_{counters[ext]}.{ext}"

            blocks.append({
                "filename": filename,
                "content": content,
                "lang": lang
            })

        return blocks

    @staticmethod
    def _infer_filename(content: str, ext: str) -> Optional[str]:
        """Intenta inferir el nombre del archivo del contenido"""

        # Para Python: buscar if __name__ == "__main__" o class/def principal
        if ext == 'py':
            if 'if __name__' in content:
                # Buscar nombre del módulo en imports o definiciones
                if match := re.search(r'class\s+(\w+)', content):
                    return f"{match.group(1).lower()}.py"
                if 'Flask' in content or 'flask' in content:
                    return 'app.py'
                if 'FastAPI' in content or 'fastapi' in content:
                    return 'main.py'
                return 'main.py'

        # Para HTML: buscar title
        if ext == 'html':
            if '<title>' in content.lower():
                if match := re.search(r'<title>([^<]+)</title>', content, re.IGNORECASE):
                    name = match.group(1).lower().replace(' ', '_')[:20]
                    return f"{name}.html"
            return 'index.html'

        # Para CSS
        if ext == 'css':
            return 'styles.css'

        # Para JSON
        if ext == 'json':
            if '"name"' in content and '"version"' in content:
                return 'package.json'
            if '"dependencies"' in content or '"devDependencies"' in content:
                return 'package.json'
            return 'data.json'

        return None


# ============================================================================
# THAU DEVELOPER
# ============================================================================

class THAUDeveloper:
    """Desarrollador THAU con extracción robusta de código"""

    SYSTEM_PROMPT = """Eres THAU, un desarrollador de software experto.

Tu trabajo es crear aplicaciones completas. Cuando te pidan crear un proyecto:

1. PLANIFICA brevemente qué vas a crear
2. GENERA el código completo para CADA archivo

FORMATO PARA ARCHIVOS - Usa este formato exacto:

**nombre_archivo.py**
```python
# código completo aquí
```

**carpeta/otro_archivo.js**
```javascript
// código aquí
```

EJEMPLO COMPLETO:

**app.py**
```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

**templates/index.html**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Mi App</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <h1>Bienvenido</h1>
</body>
</html>
```

**static/style.css**
```css
body {
    font-family: Arial, sans-serif;
    margin: 40px;
}
```

**requirements.txt**
```
flask==2.0.0
```

REGLAS:
- Genera código COMPLETO y funcional
- Incluye TODOS los imports
- El código debe ejecutarse sin cambios
- Responde en español"""

    def __init__(self, project_path: str):
        self.mcp = MCPTools(project_path)
        self.history: List[Dict] = []
        self.created_files: set = set()

    def set_project_path(self, path: str):
        self.mcp.set_project_path(path)
        self.created_files = set()

    def call_ollama_with_timeout(self, messages: List[Dict], timeout: int = GENERATION_TIMEOUT) -> Generator[str, None, None]:
        """Llama a Ollama con timeout"""
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
                        "num_predict": 4000,  # Limitar tokens
                    }
                },
                stream=True,
                timeout=timeout
            )

            if not response.ok:
                logger.error(f"Ollama error: {response.status_code}")
                yield f"Error de conexión con Ollama: {response.status_code}"
                return

            last_chunk_time = time.time()

            for line in response.iter_lines():
                # Check for stall
                if time.time() - last_chunk_time > 30:  # 30 segundos sin respuesta
                    logger.warning("Generación detenida por timeout de chunk")
                    break

                if line:
                    last_chunk_time = time.time()
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.Timeout:
            logger.error("Timeout en la generación")
            yield "\n\n[Generación interrumpida por timeout]"
        except Exception as e:
            logger.error(f"Error en Ollama: {e}")
            yield f"\n\n[Error: {str(e)}]"

    def develop(self, prompt: str) -> Generator[Dict, None, None]:
        """Desarrolla un proyecto"""

        # Contexto de archivos existentes
        existing_files = self.mcp.list_files()
        context = ""
        if existing_files:
            context = "\n\nArchivos existentes:\n" + "\n".join([f"- {f['path']}" for f in existing_files[:15]])

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]

        # Historial limitado
        for msg in self.history[-2:]:
            messages.append(msg)

        messages.append({"role": "user", "content": f"{prompt}{context}"})

        yield {"type": "start", "message": "Iniciando desarrollo..."}

        full_response = ""
        last_process_length = 0

        for chunk in self.call_ollama_with_timeout(messages):
            full_response += chunk
            yield {"type": "stream", "content": chunk, "full": full_response}

            # Procesar archivos cada 500 caracteres nuevos
            if len(full_response) - last_process_length > 500:
                last_process_length = len(full_response)

                # Extraer y crear archivos
                blocks = CodeExtractor.extract_all_code_blocks(full_response)
                for block in blocks:
                    if block["filename"] not in self.created_files:
                        result = self.mcp.create_file(block["filename"], block["content"])
                        if result.get("success"):
                            self.created_files.add(block["filename"])
                            yield {
                                "type": "file_created",
                                "file": {"path": block["filename"], "size": len(block["content"])}
                            }

        # Procesamiento final
        blocks = CodeExtractor.extract_all_code_blocks(full_response)
        for block in blocks:
            if block["filename"] not in self.created_files:
                result = self.mcp.create_file(block["filename"], block["content"])
                if result.get("success"):
                    self.created_files.add(block["filename"])
                    yield {
                        "type": "file_created",
                        "file": {"path": block["filename"], "size": len(block["content"])}
                    }

        # Guardar historial
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": full_response})

        yield {
            "type": "done",
            "files_created": list(self.created_files),
            "total": len(self.created_files)
        }

    def get_files(self) -> List[Dict]:
        return self.mcp.list_files()

    def read_file(self, path: str) -> Optional[str]:
        return self.mcp.read_file(path)

    def clear(self):
        self.history = []
        self.created_files = set()


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="THAU Developer")
developer = THAUDeveloper(DEFAULT_PROJECT_DIR)


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


@app.post("/api/clear")
async def clear():
    developer.clear()
    return {"ok": True}


@app.websocket("/ws/dev")
async def websocket_dev(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket conectado")

    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt", "")
            project_name = data.get("project_name", "")

            logger.info(f"Recibido prompt: {prompt[:100]}...")

            if project_name:
                new_path = Path(DEFAULT_PROJECT_DIR) / project_name
                new_path.mkdir(parents=True, exist_ok=True)
                developer.set_project_path(str(new_path))
                logger.info(f"Proyecto: {new_path}")

            result_queue = queue.Queue()
            stop_event = threading.Event()

            def run_generator():
                try:
                    for chunk in developer.develop(prompt):
                        if stop_event.is_set():
                            break
                        result_queue.put(("data", chunk))
                    result_queue.put(("done", None))
                except Exception as e:
                    logger.error(f"Error en generador: {e}")
                    result_queue.put(("error", str(e)))

            # Ejecutar en thread separado
            thread = threading.Thread(target=run_generator, daemon=True)
            thread.start()

            finished = False
            timeout_counter = 0
            max_timeout = GENERATION_TIMEOUT * 10  # En unidades de 100ms

            while not finished and timeout_counter < max_timeout:
                await asyncio.sleep(0.1)
                timeout_counter += 1

                # Procesar cola
                messages_processed = 0
                while not result_queue.empty() and messages_processed < 50:
                    messages_processed += 1
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
                            timeout_counter = 0  # Reset timeout on activity
                    except queue.Empty:
                        break

                # Verificar si el thread terminó
                if not thread.is_alive() and result_queue.empty():
                    finished = True

            # Si timeout, enviar mensaje
            if not finished:
                stop_event.set()
                await websocket.send_json({
                    "type": "error",
                    "error": "Generación interrumpida por timeout"
                })
                logger.warning("Generación interrumpida por timeout")

    except WebSocketDisconnect:
        logger.info("WebSocket desconectado")
    except Exception as e:
        logger.error(f"Error WS: {e}")


# ============================================================================
# HTML TEMPLATE
# ============================================================================

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>THAU Developer</title>
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
            --border: #3f3f46;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); height: 100vh; overflow: hidden; }

        .app { display: grid; grid-template-columns: 1fr 360px; height: 100vh; }

        .main { display: flex; flex-direction: column; height: 100vh; }

        .header {
            padding: 14px 24px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg2);
            flex-shrink: 0;
        }
        .header h1 {
            font-size: 18px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .header h1 .icon {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, var(--accent), #ec4899);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .btn {
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 13px;
            cursor: pointer;
            border: 1px solid var(--border);
            background: var(--bg3);
            color: var(--text);
            font-weight: 500;
            transition: all 0.15s;
        }
        .btn:hover { background: var(--border); }
        .btn-primary { background: var(--accent); border-color: var(--accent); }
        .btn-primary:hover { background: var(--accent2); }

        .chat {
            flex: 1;
            overflow-y: auto;
            padding: 20px 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .message { animation: fadeIn 0.25s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } }

        .message.user { display: flex; justify-content: flex-end; }
        .message.user .bubble {
            background: linear-gradient(135deg, var(--accent), #7c3aed);
            max-width: 70%;
        }

        .bubble {
            background: var(--bg2);
            padding: 14px 18px;
            border-radius: 16px;
            max-width: 85%;
            border: 1px solid var(--border);
            line-height: 1.6;
            font-size: 14px;
        }
        .bubble pre {
            background: var(--bg);
            padding: 12px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 10px 0;
            border: 1px solid var(--border);
        }
        .bubble code { font-family: 'JetBrains Mono', monospace; font-size: 12px; }

        .status {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 14px 18px;
            background: var(--bg2);
            border-radius: 12px;
            border: 1px solid var(--border);
        }
        .spinner {
            width: 20px; height: 20px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.7s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        .file-notification {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 14px;
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid var(--green);
            border-radius: 8px;
            font-size: 13px;
            color: var(--green);
            animation: slideIn 0.3s ease-out;
        }
        @keyframes slideIn { from { transform: translateX(-10px); opacity: 0; } }

        .input-area {
            padding: 16px 24px;
            border-top: 1px solid var(--border);
            background: var(--bg2);
            flex-shrink: 0;
        }
        .input-row { display: flex; gap: 10px; }
        .input-wrapper {
            flex: 1;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 12px;
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
            line-height: 1.5;
        }
        .send-btn {
            width: 48px; height: 48px;
            background: linear-gradient(135deg, var(--accent), #7c3aed);
            border: none;
            border-radius: 12px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }

        /* Sidebar */
        .sidebar {
            background: var(--bg2);
            border-left: 1px solid var(--border);
            display: flex;
            flex-direction: column;
        }
        .sidebar-header {
            padding: 14px 16px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .sidebar-header h2 { font-size: 14px; }

        .files-list { flex: 1; overflow-y: auto; padding: 8px; }
        .file-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 12px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            margin-bottom: 4px;
        }
        .file-item:hover { background: var(--bg3); }
        .file-item.active { background: var(--bg3); border-left: 2px solid var(--accent); }
        .file-item.new { animation: highlight 2s ease-out; }
        @keyframes highlight { 0%, 40% { background: rgba(34, 197, 94, 0.2); } }

        .file-preview {
            border-top: 1px solid var(--border);
            max-height: 45%;
            display: flex;
            flex-direction: column;
        }
        .preview-header {
            padding: 10px 14px;
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
            padding: 12px;
            background: var(--bg);
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            white-space: pre-wrap;
            line-height: 1.5;
        }

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
            width: 80px; height: 80px;
            background: linear-gradient(135deg, var(--accent), #ec4899);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 40px;
            margin-bottom: 24px;
        }
        .welcome h2 { margin-bottom: 10px; }
        .welcome p { color: var(--text2); max-width: 400px; margin-bottom: 24px; }
        .templates { display: flex; gap: 10px; flex-wrap: wrap; justify-content: center; }
        .template {
            padding: 12px 20px;
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: 10px;
            cursor: pointer;
            font-size: 14px;
        }
        .template:hover { border-color: var(--accent); }

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
            border-radius: 16px;
            padding: 24px;
            width: 450px;
            border: 1px solid var(--border);
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
        .modal-actions { display: flex; gap: 10px; justify-content: flex-end; margin-top: 16px; }

        .empty-state { padding: 30px 20px; text-align: center; color: var(--text2); font-size: 13px; }

        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    </style>
</head>
<body>
    <div class="app">
        <div class="main">
            <header class="header">
                <h1>
                    <span class="icon">🚀</span>
                    THAU Developer
                </h1>
                <div style="display:flex;gap:8px">
                    <button class="btn" onclick="clearChat()">Limpiar</button>
                    <button class="btn btn-primary" onclick="showModal()">+ Proyecto</button>
                </div>
            </header>

            <div class="chat" id="chat">
                <div class="welcome" id="welcome">
                    <div class="welcome-icon">💻</div>
                    <h2>THAU Developer</h2>
                    <p>Describe tu proyecto y THAU creará todos los archivos automáticamente.</p>
                    <div class="templates">
                        <div class="template" onclick="quickStart('flask')">🌐 Flask Web</div>
                        <div class="template" onclick="quickStart('fastapi')">⚡ FastAPI</div>
                        <div class="template" onclick="quickStart('cli')">💻 CLI Tool</div>
                        <div class="template" onclick="quickStart('html')">📄 HTML/CSS</div>
                    </div>
                </div>
            </div>

            <div class="input-area">
                <div class="input-row">
                    <div class="input-wrapper">
                        <textarea id="input" placeholder="Describe qué quieres crear..." rows="2" onkeydown="handleKey(event)"></textarea>
                    </div>
                    <button class="send-btn" id="sendBtn" onclick="send()">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5">
                            <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                        </svg>
                    </button>
                </div>
            </div>
        </div>

        <aside class="sidebar">
            <div class="sidebar-header">
                <h2>📁 Archivos</h2>
                <button class="btn" style="padding:4px 8px;font-size:11px" onclick="refreshFiles()">↻</button>
            </div>
            <div class="files-list" id="filesList">
                <div class="empty-state">Los archivos aparecerán aquí cuando THAU los cree</div>
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
        let reconnectAttempts = 0;

        document.addEventListener('DOMContentLoaded', () => {
            connectWS();
            refreshFiles();
        });

        function connectWS() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws/dev`);

            ws.onopen = () => {
                console.log('✅ Conectado');
                reconnectAttempts = 0;
            };

            ws.onclose = () => {
                console.log('❌ Desconectado');
                reconnectAttempts++;
                setTimeout(connectWS, Math.min(2000 * reconnectAttempts, 10000));
            };

            ws.onerror = (e) => console.error('WS Error:', e);
            ws.onmessage = handleMessage;
        }

        function handleMessage(e) {
            const data = JSON.parse(e.data);
            console.log('📨', data.type);

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
                    finishGeneration();
                    break;

                case 'error':
                    hideStatus();
                    addSystemMessage(`❌ ${data.error}`);
                    finishGeneration();
                    break;
            }
        }

        function finishGeneration() {
            currentBubble = null;
            isGenerating = false;
            document.getElementById('sendBtn').disabled = false;
            refreshFiles();
        }

        function showStatus(text) {
            if (statusDiv) return;
            const chat = document.getElementById('chat');
            statusDiv = document.createElement('div');
            statusDiv.className = 'status';
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
            if (!text || isGenerating || !ws || ws.readyState !== WebSocket.OPEN) return;

            hideWelcome();

            const chat = document.getElementById('chat');
            const userDiv = document.createElement('div');
            userDiv.className = 'message user';
            userDiv.innerHTML = `<div class="bubble">${text}</div>`;
            chat.appendChild(userDiv);

            ws.send(JSON.stringify({ prompt: text }));
            input.value = '';
            isGenerating = true;
            document.getElementById('sendBtn').disabled = true;
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
            try {
                const res = await fetch('/api/files');
                const files = await res.json();
                const list = document.getElementById('filesList');

                if (!files.length) {
                    list.innerHTML = '<div class="empty-state">Los archivos aparecerán aquí</div>';
                    return;
                }

                list.innerHTML = files.filter(f => f.type === 'file').map(f => `
                    <div class="file-item new" onclick="previewFile('${f.path}')">
                        📄 ${f.path}
                    </div>
                `).join('');
            } catch (e) {
                console.error('Error refreshing files:', e);
            }
        }

        async function previewFile(path) {
            try {
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
            } catch (e) {
                console.error('Error previewing file:', e);
            }
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

            const chat = document.getElementById('chat');
            const userDiv = document.createElement('div');
            userDiv.className = 'message user';
            userDiv.innerHTML = `<div class="bubble">${desc}</div>`;
            chat.appendChild(userDiv);

            ws.send(JSON.stringify({ prompt: desc, project_name: name || 'proyecto' }));
            isGenerating = true;
            document.getElementById('sendBtn').disabled = true;

            document.getElementById('projectName').value = '';
            document.getElementById('projectDesc').value = '';
        }

        function quickStart(type) {
            const prompts = {
                flask: 'Crea una aplicación web con Flask que tenga página de inicio, página about, y formulario de contacto. Usa Bootstrap para el diseño. Incluye app.py, templates/ y requirements.txt.',
                fastapi: 'Crea una API REST con FastAPI para gestionar tareas (CRUD completo). Incluye modelos Pydantic, endpoints para crear, leer, actualizar y eliminar tareas, y SQLite como base de datos.',
                cli: 'Crea una herramienta de línea de comandos en Python usando Click para gestionar una lista de tareas. Debe poder agregar, listar, completar y eliminar tareas. Guarda en un archivo JSON.',
                html: 'Crea una página web moderna con HTML, CSS y JavaScript. Incluye header con navegación, hero section, sección de características con cards, y footer. Diseño responsive y moderno.'
            };
            document.getElementById('projectDesc').value = prompts[type];
            showModal();
        }

        async function clearChat() {
            await fetch('/api/clear', {method: 'POST'});
            document.getElementById('chat').innerHTML = `
                <div class="welcome" id="welcome">
                    <div class="welcome-icon">💻</div>
                    <h2>THAU Developer</h2>
                    <p>Describe tu proyecto y THAU creará todos los archivos.</p>
                    <div class="templates">
                        <div class="template" onclick="quickStart('flask')">🌐 Flask Web</div>
                        <div class="template" onclick="quickStart('fastapi')">⚡ FastAPI</div>
                        <div class="template" onclick="quickStart('cli')">💻 CLI Tool</div>
                        <div class="template" onclick="quickStart('html')">📄 HTML/CSS</div>
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
╔═══════════════════════════════════════════════════════════════════╗
║                      THAU Developer                                ║
║                                                                   ║
║  🚀 Sistema robusto de desarrollo de aplicaciones                 ║
║  📁 Extracción automática de código de cualquier formato          ║
║  ⏱️  Timeout handling para evitar cuelgues                         ║
║                                                                   ║
║  URL: http://localhost:{PORT}                                       ║
║  Proyecto: {DEFAULT_PROJECT_DIR}
╚═══════════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
