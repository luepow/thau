#!/usr/bin/env python3
"""
THAU Multi-Agent Development System
Sistema de desarrollo con 7 agentes especializados que colaboran en proyectos

🤖 El Equipo THAU:
1. 🌟 Nova (PMO) - Project Orchestrator: Planifica, coordina y supervisa
2. 🏛️ Atlas (Architect) - System Architect: Diseña arquitectura y estructura
3. 🎨 Pixel (Frontend) - Frontend Wizard: React, Vue, Next.js, Tailwind
4. 🔧 Forge (Backend) - Backend Architect: APIs, lógica, bases de datos
5. 💫 Aura (UX) - Experience Designer: Diseño de experiencia de usuario
6. 🛡️ Sentinel (QA) - Quality Guardian: Testing y validación
7. 🚀 Rocket (DevOps) - Launch Engineer: Deployment, CI/CD, infraestructura
"""

import json
import re
import subprocess
import os
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import queue
import threading
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from loguru import logger
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

OLLAMA_MODEL = "thau:unified"
OLLAMA_URL = "http://localhost:11434"
DEFAULT_PROJECT_DIR = os.path.expanduser("~/thau_projects")
PORT = 7868
GENERATION_TIMEOUT = 180

# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

class AgentRole(Enum):
    PMO = "pmo"
    ARCHITECT = "architect"
    FRONTEND = "frontend"
    BACKEND = "backend"
    UX = "ux"
    QA = "qa"
    DEVOPS = "devops"


AGENT_CONFIGS = {
    AgentRole.PMO: {
        "name": "Nova",
        "title": "Project Orchestrator",
        "emoji": "🌟",
        "color": "#4A90D9",
        "system_prompt": """Eres el PMO (Project Manager) del equipo THAU.

Tu rol es:
1. Recibir requerimientos del usuario y entenderlos completamente
2. Crear un plan de proyecto con tareas específicas
3. Asignar tareas a cada agente especializado
4. Coordinar la comunicación entre agentes
5. Asegurar que el proyecto se complete exitosamente

Cuando recibas un proyecto:
1. Analiza los requerimientos
2. Crea un plan detallado con fases
3. Lista las tareas para cada agente:
   - @arquitecto: tareas de diseño y arquitectura
   - @frontend: tareas de UI y componentes
   - @backend: tareas de API y lógica
   - @ux: tareas de diseño visual
   - @qa: tareas de testing
   - @devops: tareas de deployment

Responde siempre de forma estructurada y profesional."""
    },

    AgentRole.ARCHITECT: {
        "name": "Atlas",
        "title": "System Architect",
        "emoji": "🏛️",
        "color": "#9B59B6",
        "system_prompt": """Eres el Arquitecto de Software del equipo THAU.

Tu rol es:
1. Diseñar la arquitectura del sistema
2. Definir la estructura de carpetas y archivos
3. Elegir tecnologías y patrones de diseño
4. Crear diagramas y documentación técnica
5. Establecer estándares de código

Cuando diseñes arquitectura:
1. Usa Clean Architecture o patrones similares
2. Define claramente capas y responsabilidades
3. Especifica estructura de carpetas
4. Lista los archivos principales a crear

Formato de respuesta para estructura:
```
proyecto/
├── src/
│   ├── components/
│   ├── pages/
│   ├── services/
│   └── utils/
├── public/
├── package.json
└── README.md
```"""
    },

    AgentRole.FRONTEND: {
        "name": "Pixel",
        "title": "Frontend Wizard",
        "emoji": "🎨",
        "color": "#3498DB",
        "system_prompt": """Eres el Desarrollador Frontend del equipo THAU.

Tu rol es:
1. Desarrollar interfaces de usuario
2. Crear componentes React/Vue/Next.js
3. Implementar estilos con Tailwind CSS
4. Manejar estado y navegación
5. Integrar con APIs del backend

Tecnologías que dominas:
- React 18+ con hooks
- Next.js 14 (App Router)
- Vue 3 con Composition API
- Tailwind CSS
- TypeScript

Cuando generes código:
1. Usa componentes funcionales
2. Aplica Tailwind para estilos
3. Incluye responsividad (mobile-first)
4. Añade accesibilidad básica

**Ejemplo de componente:**
```jsx
// components/Hero.jsx
export default function Hero() {
  return (
    <section className="min-h-screen bg-gradient-to-br from-blue-600 to-purple-700 flex items-center justify-center">
      <div className="text-center text-white px-4">
        <h1 className="text-5xl md:text-7xl font-bold mb-6">
          Bienvenido
        </h1>
        <p className="text-xl md:text-2xl mb-8 opacity-90">
          Tu solución digital
        </p>
        <button className="bg-white text-blue-600 px-8 py-3 rounded-full font-semibold hover:bg-opacity-90 transition">
          Comenzar
        </button>
      </div>
    </section>
  );
}
```"""
    },

    AgentRole.BACKEND: {
        "name": "Forge",
        "title": "Backend Architect",
        "emoji": "🔧",
        "color": "#27AE60",
        "system_prompt": """Eres el Desarrollador Backend del equipo THAU.

Tu rol es:
1. Desarrollar APIs RESTful
2. Implementar lógica de negocio
3. Diseñar bases de datos
4. Manejar autenticación y seguridad
5. Integrar servicios externos

Tecnologías que dominas:
- Python (FastAPI, Django, Flask)
- Node.js (Express, NestJS)
- Bases de datos (PostgreSQL, MongoDB)
- Docker y microservicios

Cuando generes código:
1. Sigue principios SOLID
2. Incluye validación de datos
3. Maneja errores correctamente
4. Documenta endpoints

**Ejemplo de API:**
```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Mi API")

class Item(BaseModel):
    name: str
    price: float

@app.get("/")
def root():
    return {"message": "API funcionando"}

@app.post("/items")
def create_item(item: Item):
    return {"item": item, "status": "created"}
```"""
    },

    AgentRole.UX: {
        "name": "Aura",
        "title": "Experience Designer",
        "emoji": "💫",
        "color": "#E74C3C",
        "system_prompt": """Eres el Diseñador UX/UI del equipo THAU.

Tu rol es:
1. Diseñar experiencias de usuario intuitivas
2. Crear sistemas de diseño coherentes
3. Definir paletas de colores y tipografías
4. Diseñar componentes reutilizables
5. Asegurar accesibilidad

Cuando diseñes:
1. Piensa en mobile-first
2. Usa espaciado consistente (8px grid)
3. Define jerarquía visual clara
4. Incluye estados (hover, active, disabled)

**Ejemplo de sistema de diseño:**
```css
/* Paleta de colores */
:root {
  --primary: #3B82F6;
  --secondary: #8B5CF6;
  --accent: #10B981;
  --background: #F8FAFC;
  --text: #1E293B;
}

/* Tipografía */
--font-heading: 'Inter', sans-serif;
--font-body: 'Inter', sans-serif;

/* Espaciado */
--space-xs: 4px;
--space-sm: 8px;
--space-md: 16px;
--space-lg: 24px;
--space-xl: 32px;
```"""
    },

    AgentRole.QA: {
        "name": "Sentinel",
        "title": "Quality Guardian",
        "emoji": "🛡️",
        "color": "#F39C12",
        "system_prompt": """Eres el QA Tester del equipo THAU.

Tu rol es:
1. Crear planes de prueba
2. Escribir tests unitarios y de integración
3. Realizar pruebas manuales
4. Reportar bugs y problemas
5. Verificar calidad del código

Tecnologías que dominas:
- Jest, Vitest para testing
- Cypress, Playwright para E2E
- Testing Library para React
- Pytest para Python

Cuando crees tests:
1. Cubre casos positivos y negativos
2. Prueba edge cases
3. Verifica accesibilidad
4. Documenta resultados

**Ejemplo de test:**
```javascript
// Hero.test.jsx
import { render, screen } from '@testing-library/react';
import Hero from './Hero';

describe('Hero Component', () => {
  it('renders welcome message', () => {
    render(<Hero />);
    expect(screen.getByText(/bienvenido/i)).toBeInTheDocument();
  });

  it('has call-to-action button', () => {
    render(<Hero />);
    expect(screen.getByRole('button')).toBeInTheDocument();
  });
});
```"""
    },

    AgentRole.DEVOPS: {
        "name": "Rocket",
        "title": "Launch Engineer",
        "emoji": "🚀",
        "color": "#1ABC9C",
        "system_prompt": """Eres el Ingeniero DevOps del equipo THAU.

Tu rol es:
1. Configurar pipelines CI/CD
2. Crear Dockerfiles y docker-compose
3. Configurar servidores y deployment
4. Monitorear y optimizar rendimiento
5. Gestionar infraestructura

Tecnologías que dominas:
- Docker y Kubernetes
- GitHub Actions, GitLab CI
- Nginx, Apache
- AWS, Vercel, Railway

Cuando configures deployment:
1. Usa multi-stage builds en Docker
2. Configura variables de entorno
3. Implementa health checks
4. Documenta proceso de deploy

**Ejemplo Dockerfile:**
```dockerfile
# Dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
EXPOSE 3000
CMD ["npm", "start"]
```"""
    },
}


# ============================================================================
# AGENT CLASS
# ============================================================================

@dataclass
class Agent:
    """Representa un agente especializado"""
    role: AgentRole
    config: Dict
    conversation_history: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        self.name = self.config["name"]
        self.emoji = self.config["emoji"]
        self.system_prompt = self.config["system_prompt"]

    def get_response(self, message: str, context: str = "") -> str:
        """Obtiene respuesta del agente"""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Añadir contexto del proyecto si existe
        if context:
            messages.append({
                "role": "system",
                "content": f"Contexto del proyecto:\n{context}"
            })

        # Añadir historial de conversación
        messages.extend(self.conversation_history[-10:])  # Últimos 10 mensajes

        # Añadir mensaje actual
        messages.append({"role": "user", "content": message})

        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 2048
                    }
                },
                timeout=GENERATION_TIMEOUT
            )

            if response.status_code == 200:
                result = response.json()
                assistant_message = result.get("message", {}).get("content", "")

                # Guardar en historial
                self.conversation_history.append({"role": "user", "content": message})
                self.conversation_history.append({"role": "assistant", "content": assistant_message})

                return assistant_message
            else:
                return f"Error: {response.status_code}"

        except Exception as e:
            return f"Error de comunicación: {str(e)}"

    def clear_history(self):
        """Limpia el historial de conversación"""
        self.conversation_history = []


# ============================================================================
# TEAM COORDINATOR
# ============================================================================

class TeamCoordinator:
    """Coordina el equipo de agentes"""

    def __init__(self):
        self.agents: Dict[AgentRole, Agent] = {}
        self.project_context = ""
        self.project_plan = []
        self.created_files = []

        # Inicializar agentes
        for role, config in AGENT_CONFIGS.items():
            self.agents[role] = Agent(role=role, config=config)

    def get_agent(self, role: AgentRole) -> Agent:
        """Obtiene un agente por rol"""
        return self.agents.get(role)

    def set_project_context(self, context: str):
        """Establece el contexto del proyecto"""
        self.project_context = context

    def start_project(self, requirements: str) -> Dict:
        """Inicia un nuevo proyecto con el PMO"""
        pmo = self.get_agent(AgentRole.PMO)

        prompt = f"""El usuario quiere crear el siguiente proyecto:

{requirements}

Por favor:
1. Analiza los requerimientos
2. Crea un plan de proyecto detallado
3. Lista las tareas específicas para cada agente del equipo
4. Indica el orden de ejecución

Usa el formato:
## Plan de Proyecto
[descripción general]

## Tareas por Agente
@arquitecto: [tareas]
@frontend: [tareas]
@backend: [tareas]
@ux: [tareas]
@qa: [tareas]
@devops: [tareas]
"""

        response = pmo.get_response(prompt)
        self.project_context = f"Proyecto: {requirements}\n\nPlan:\n{response}"

        return {
            "agent": "PMO",
            "response": response,
            "plan_created": True
        }

    def delegate_task(self, role: AgentRole, task: str) -> Dict:
        """Delega una tarea a un agente específico"""
        agent = self.get_agent(role)

        if not agent:
            return {"error": f"Agente {role} no encontrado"}

        response = agent.get_response(task, self.project_context)

        return {
            "agent": agent.name,
            "emoji": agent.emoji,
            "response": response
        }

    def extract_code_files(self, text: str) -> List[Dict]:
        """Extrae archivos de código del texto"""
        files = []

        # Patrón: **filename.ext** seguido de ```lang
        pattern = r'\*\*([^*]+\.\w+)\*\*\s*\n```(\w+)\n(.*?)```'  # noqa: W605

        for match in re.finditer(pattern, text, re.DOTALL):
            filename = match.group(1).strip()
            lang = match.group(2)
            content = match.group(3).strip()

            files.append({
                "filename": filename,
                "language": lang,
                "content": content
            })

        # Patrón alternativo: ```lang filename
        pattern2 = r'```(\w+)\s+([^\n]+\.(?:jsx?|tsx?|py|html|css|json))\n(.*?)```'

        for match in re.finditer(pattern2, text, re.DOTALL):
            lang = match.group(1)
            filename = match.group(2).strip()
            content = match.group(3).strip()

            if not any(f["filename"] == filename for f in files):
                files.append({
                    "filename": filename,
                    "language": lang,
                    "content": content
                })

        return files

    def save_files(self, files: List[Dict], project_path: str) -> List[str]:
        """Guarda los archivos extraídos"""
        saved = []

        for file_info in files:
            filepath = Path(project_path) / file_info["filename"]
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(file_info["content"])

            saved.append(str(filepath))
            self.created_files.append(file_info["filename"])
            logger.info(f"✅ Archivo creado: {file_info['filename']}")

        return saved


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="THAU Multi-Agent System")
coordinator = TeamCoordinator()


@app.get("/", response_class=HTMLResponse)
async def home():
    """Página principal con interfaz de chat multi-agente"""
    template_path = Path(__file__).parent / "templates" / "multi_agent.html"
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


# Mantener compatibilidad - HTML antiguo comentado
_OLD_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>THAU Multi-Agent System</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
        }}

        .container {{
            display: grid;
            grid-template-columns: 280px 1fr 300px;
            height: 100vh;
        }}

        /* Sidebar - Agentes */
        .sidebar {{
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-right: 1px solid rgba(255,255,255,0.1);
            overflow-y: auto;
        }}

        .sidebar h2 {{
            color: #fff;
            margin-bottom: 20px;
            font-size: 1.2rem;
        }}

        .agent-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.3s;
            border: 2px solid transparent;
        }}

        .agent-card:hover {{
            background: rgba(255,255,255,0.1);
            transform: translateX(5px);
        }}

        .agent-card.selected {{
            border-color: #4A90D9;
            background: rgba(74, 144, 217, 0.2);
        }}

        .agent-emoji {{
            font-size: 24px;
            margin-right: 10px;
        }}

        .agent-name {{
            font-weight: bold;
            display: block;
        }}

        .agent-title {{
            font-size: 0.8rem;
            opacity: 0.7;
        }}

        /* Main Chat Area */
        .main {{
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}

        .header {{
            background: rgba(0,0,0,0.3);
            padding: 15px 20px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}

        .header h1 {{
            font-size: 1.5rem;
            color: #fff;
        }}

        .chat-area {{
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }}

        .message {{
            margin-bottom: 15px;
            animation: fadeIn 0.3s;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .message.user {{
            text-align: right;
        }}

        .message.user .bubble {{
            background: #4A90D9;
            display: inline-block;
            padding: 12px 18px;
            border-radius: 18px 18px 4px 18px;
            max-width: 70%;
        }}

        .message.agent {{
            text-align: left;
        }}

        .message.agent .agent-info {{
            margin-bottom: 5px;
        }}

        .message.agent .bubble {{
            background: rgba(255,255,255,0.1);
            display: inline-block;
            padding: 12px 18px;
            border-radius: 18px 18px 18px 4px;
            max-width: 85%;
        }}

        .message pre {{
            background: rgba(0,0,0,0.3);
            padding: 10px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 10px 0;
        }}

        .message code {{
            font-family: 'Fira Code', monospace;
            font-size: 0.9rem;
        }}

        /* Input Area */
        .input-area {{
            background: rgba(0,0,0,0.3);
            padding: 15px 20px;
            border-top: 1px solid rgba(255,255,255,0.1);
        }}

        .input-row {{
            display: flex;
            gap: 10px;
        }}

        #messageInput {{
            flex: 1;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 25px;
            padding: 12px 20px;
            color: #fff;
            font-size: 1rem;
        }}

        #messageInput:focus {{
            outline: none;
            border-color: #4A90D9;
        }}

        .send-btn {{
            background: linear-gradient(135deg, #4A90D9, #8B5CF6);
            border: none;
            border-radius: 25px;
            padding: 12px 25px;
            color: #fff;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }}

        .send-btn:hover {{
            transform: scale(1.05);
        }}

        /* Project Panel */
        .project-panel {{
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-left: 1px solid rgba(255,255,255,0.1);
            overflow-y: auto;
        }}

        .project-panel h3 {{
            margin-bottom: 15px;
        }}

        .file-list {{
            list-style: none;
        }}

        .file-item {{
            background: rgba(255,255,255,0.05);
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 5px;
            font-size: 0.9rem;
        }}

        .quick-actions {{
            margin-top: 20px;
        }}

        .quick-btn {{
            width: 100%;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            padding: 10px;
            color: #fff;
            cursor: pointer;
            margin-bottom: 10px;
            transition: background 0.2s;
        }}

        .quick-btn:hover {{
            background: rgba(255,255,255,0.2);
        }}

        /* Project Input */
        .project-input {{
            width: 100%;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            padding: 10px;
            color: #fff;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Sidebar con Agentes -->
        <div class="sidebar">
            <h2>🤖 Equipo THAU</h2>
            {agents_html}
        </div>

        <!-- Área Principal de Chat -->
        <div class="main">
            <div class="header">
                <h1>💬 THAU Multi-Agent Development</h1>
                <p style="opacity:0.7">Sistema de desarrollo colaborativo con 7 agentes especializados</p>
            </div>

            <div class="chat-area" id="chatArea">
                <div class="message agent">
                    <div class="agent-info">📋 PMO</div>
                    <div class="bubble">
                        ¡Hola! Soy el Project Manager del equipo THAU. Describe tu proyecto y coordinaré al equipo para desarrollarlo.
                        <br><br>
                        Puedes hablar conmigo o seleccionar un agente específico del panel izquierdo.
                    </div>
                </div>
            </div>

            <div class="input-area">
                <div class="input-row">
                    <input type="text" id="messageInput" placeholder="Describe tu proyecto o haz una pregunta..." />
                    <button class="send-btn" onclick="sendMessage()">Enviar</button>
                </div>
            </div>
        </div>

        <!-- Panel de Proyecto -->
        <div class="project-panel">
            <h3>📁 Proyecto</h3>

            <input type="text" class="project-input" id="projectName" placeholder="Nombre del proyecto" value="mi_proyecto" />

            <div class="quick-actions">
                <button class="quick-btn" onclick="startProject()">🚀 Iniciar Proyecto</button>
                <button class="quick-btn" onclick="askArchitect()">🏗️ Diseñar Arquitectura</button>
                <button class="quick-btn" onclick="askFrontend()">🎨 Crear Frontend</button>
                <button class="quick-btn" onclick="askBackend()">⚙️ Crear Backend</button>
                <button class="quick-btn" onclick="runAllAgents()">⚡ Ejecutar Todo</button>
            </div>

            <h3 style="margin-top:20px">📄 Archivos Creados</h3>
            <ul class="file-list" id="fileList">
                <li class="file-item" style="opacity:0.5">Sin archivos aún...</li>
            </ul>
        </div>
    </div>

    <script>
        let selectedAgent = 'pmo';
        let ws = null;
        let reconnectAttempts = 0;

        function connectWebSocket() {{
            console.log('Conectando WebSocket...');
            ws = new WebSocket(`ws://${{window.location.host}}/ws/team`);

            ws.onopen = () => {{
                console.log('WebSocket conectado!');
                reconnectAttempts = 0;
                document.querySelector('.send-btn').disabled = false;
            }};

            ws.onmessage = (event) => {{
                console.log('Mensaje recibido:', event.data);
                const data = JSON.parse(event.data);
                addMessage(data.agent, data.emoji, data.response, 'agent');

                if (data.files && data.files.length > 0) {{
                    updateFileList(data.files);
                }}
            }};

            ws.onerror = (error) => {{
                console.error('WebSocket error:', error);
            }};

            ws.onclose = () => {{
                console.log('WebSocket cerrado, reconectando...');
                document.querySelector('.send-btn').disabled = true;
                if (reconnectAttempts < 5) {{
                    reconnectAttempts++;
                    setTimeout(connectWebSocket, 2000);
                }}
            }};
        }}

        function selectAgent(role) {{
            document.querySelectorAll('.agent-card').forEach(card => {{
                card.classList.remove('selected');
            }});
            document.querySelector(`[data-role="${{role}}"]`).classList.add('selected');
            selectedAgent = role;
        }}

        function addMessage(name, emoji, content, type) {{
            const chatArea = document.getElementById('chatArea');
            const div = document.createElement('div');
            div.className = `message ${{type}}`;

            // Formatear código
            content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
            content = content.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
            content = content.replace(/\n/g, '<br>');

            if (type === 'agent') {{
                div.innerHTML = `
                    <div class="agent-info">${{emoji}} ${{name}}</div>
                    <div class="bubble">${{content}}</div>
                `;
            }} else {{
                div.innerHTML = `<div class="bubble">${{content}}</div>`;
            }}

            chatArea.appendChild(div);
            chatArea.scrollTop = chatArea.scrollHeight;
        }}

        function sendMessage() {{
            const input = document.getElementById('messageInput');
            const message = input.value.trim();

            if (!message) return;

            if (!ws || ws.readyState !== WebSocket.OPEN) {{
                addMessage('Sistema', '⚠️', 'WebSocket no conectado. Reconectando...', 'agent');
                connectWebSocket();
                return;
            }}

            addMessage('Tú', '👤', message, 'user');
            input.value = '';

            const projectName = document.getElementById('projectName').value || 'mi_proyecto';

            // Mostrar indicador de carga
            addMessage(selectedAgent.toUpperCase(), '⏳', 'Procesando...', 'agent');

            ws.send(JSON.stringify({{
                agent: selectedAgent,
                message: message,
                project: projectName
            }}));
        }}

        function updateFileList(files) {{
            const list = document.getElementById('fileList');
            list.innerHTML = files.map(f => `<li class="file-item">📄 ${{f}}</li>`).join('');
        }}

        function startProject() {{
            const input = document.getElementById('messageInput');
            input.value = "Quiero crear una landing page moderna con React y Tailwind CSS. Debe tener: Hero section, Features, Testimonials, Pricing, Footer.";
            sendMessage();
        }}

        function askArchitect() {{
            selectAgent('architect');
            const input = document.getElementById('messageInput');
            input.value = "Diseña la estructura del proyecto con las carpetas y archivos necesarios.";
            sendMessage();
        }}

        function askFrontend() {{
            selectAgent('frontend');
            const input = document.getElementById('messageInput');
            input.value = "Crea los componentes React con Tailwind CSS para la landing page.";
            sendMessage();
        }}

        function askBackend() {{
            selectAgent('backend');
            const input = document.getElementById('messageInput');
            input.value = "Crea una API básica con FastAPI para el formulario de contacto.";
            sendMessage();
        }}

        function runAllAgents() {{
            const tasks = [
                {{ agent: 'pmo', message: 'Planifica el proyecto de landing page' }},
                {{ agent: 'architect', message: 'Diseña la arquitectura' }},
                {{ agent: 'frontend', message: 'Crea los componentes React' }},
            ];

            // Ejecutar secuencialmente
            let i = 0;
            function nextTask() {{
                if (i < tasks.length) {{
                    selectAgent(tasks[i].agent);
                    document.getElementById('messageInput').value = tasks[i].message;
                    sendMessage();
                    i++;
                    setTimeout(nextTask, 5000);
                }}
            }}
            nextTask();
        }}

        // Event listeners
        document.getElementById('messageInput').addEventListener('keypress', (e) => {{
            if (e.key === 'Enter') sendMessage();
        }});

        // Inicializar
        connectWebSocket();
        selectAgent('pmo');
    </script>
</body>
</html>
"""


@app.websocket("/ws/team")
async def websocket_team(websocket: WebSocket):
    """WebSocket para comunicación con el equipo"""
    await websocket.accept()
    logger.info("WebSocket conectado")

    try:
        while True:
            data = await websocket.receive_json()

            agent_role = data.get("agent", "pmo")
            message = data.get("message", "")
            project_name = data.get("project", "mi_proyecto")

            logger.info(f"Mensaje para {agent_role}: {message[:100]}...")

            # Crear directorio del proyecto
            project_path = Path(DEFAULT_PROJECT_DIR) / project_name
            project_path.mkdir(parents=True, exist_ok=True)

            # Obtener respuesta del agente
            try:
                role = AgentRole(agent_role)
            except ValueError:
                role = AgentRole.PMO

            result = coordinator.delegate_task(role, message)

            # Extraer y guardar archivos
            files = coordinator.extract_code_files(result["response"])
            if files:
                saved = coordinator.save_files(files, str(project_path))
                result["files"] = [f["filename"] for f in files]
            else:
                result["files"] = []

            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info("WebSocket desconectado")
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")


@app.post("/api/agent/{role}")
async def agent_endpoint(role: str, request: dict):
    """Endpoint REST para hablar con un agente"""
    try:
        agent_role = AgentRole(role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Agente '{role}' no válido")

    message = request.get("message", "")
    result = coordinator.delegate_task(agent_role, message)

    return result


@app.get("/api/agents")
async def list_agents():
    """Lista todos los agentes disponibles"""
    return {
        role.value: {
            "name": config["name"],
            "title": config["title"],
            "emoji": config["emoji"]
        }
        for role, config in AGENT_CONFIGS.items()
    }


def main():
    """Función principal"""
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🤖 THAU Multi-Agent Development System                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ⚡ El Equipo THAU - 7 Agentes Especializados:                               ║
║                                                                              ║
║     🌟 Nova      │ Project Orchestrator   │ Planifica y coordina            ║
║     🏛️  Atlas     │ System Architect       │ Diseña arquitectura             ║
║     🎨 Pixel     │ Frontend Wizard        │ React, Vue, Tailwind            ║
║     🔧 Forge     │ Backend Architect      │ APIs y lógica                   ║
║     💫 Aura      │ Experience Designer    │ UX/UI Design                    ║
║     🛡️  Sentinel  │ Quality Guardian       │ Testing y validación            ║
║     🚀 Rocket    │ Launch Engineer        │ CI/CD y deployment              ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🌐 URL: http://localhost:{PORT}                                              ║
║  📁 Proyectos: {DEFAULT_PROJECT_DIR:<46}║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
