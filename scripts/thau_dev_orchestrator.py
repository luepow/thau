#!/usr/bin/env python3
"""
THAU Development Orchestrator
Sistema completo de desarrollo con Nova como orquestador principal

Nova coordina automáticamente a todos los agentes para:
1. Planificar el proyecto
2. Diseñar la arquitectura
3. Crear todos los archivos
4. Ejecutar y probar la aplicación
5. Permitir revisiones y correcciones
"""

import json
import re
import subprocess
import os
import asyncio
import sys
import signal
import hashlib
import secrets
import zipfile
import io
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import uvicorn
from loguru import logger
import requests

# Importar servicio TTS de THAU
sys.path.insert(0, str(Path(__file__).parent.parent))
from capabilities.voice.tts_service import ThauVoice, SPANISH_VOICES

# ============================================================================
# CONFIGURATION
# ============================================================================

OLLAMA_MODEL = "thau:developer"
OLLAMA_URL = "http://localhost:11434"
DEFAULT_PROJECT_DIR = os.path.expanduser("~/thau_projects")
DEFAULT_PORT = 7868
DEFAULT_PREVIEW_PORT = 3000  # Puerto base para preview de aplicaciones
GENERATION_TIMEOUT = 300  # 5 minutos para generación completa
USERS_DB_FILE = os.path.expanduser("~/thau_projects/.users.json")
PROJECTS_DB_FILE = os.path.expanduser("~/thau_projects/.projects.json")


# ============================================================================
# PORT UTILITIES
# ============================================================================

import socket

def is_port_in_use(port: int) -> bool:
    """Verifica si un puerto está en uso"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except OSError:
            return True


def find_available_port(start_port: int, max_attempts: int = 100) -> int:
    """Encuentra un puerto disponible comenzando desde start_port"""
    for i in range(max_attempts):
        port = start_port + i
        if not is_port_in_use(port):
            return port
    raise RuntimeError(f"No se encontró puerto disponible después de {max_attempts} intentos desde {start_port}")


def get_server_port() -> int:
    """Obtiene un puerto disponible para el servidor"""
    if not is_port_in_use(DEFAULT_PORT):
        return DEFAULT_PORT

    # Buscar puerto alternativo
    alt_port = find_available_port(DEFAULT_PORT + 1)
    logger.warning(f"Puerto {DEFAULT_PORT} en uso. Usando puerto alternativo: {alt_port}")
    return alt_port


# Puerto del servidor (se determina al inicio)
PORT = None  # Se asignará en main()


# ============================================================================
# USER AUTHENTICATION
# ============================================================================

@dataclass
class User:
    """Usuario del sistema"""
    username: str
    password_hash: str
    created_at: str
    projects: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "username": self.username,
            "password_hash": self.password_hash,
            "created_at": self.created_at,
            "projects": self.projects
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "User":
        return cls(
            username=data["username"],
            password_hash=data["password_hash"],
            created_at=data.get("created_at", datetime.now().isoformat()),
            projects=data.get("projects", [])
        )


class UserManager:
    """Gestión de usuarios y autenticación"""

    def __init__(self):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, str] = {}  # token -> username
        self._load_users()

    def _load_users(self):
        """Carga usuarios desde archivo"""
        if os.path.exists(USERS_DB_FILE):
            try:
                with open(USERS_DB_FILE, 'r') as f:
                    data = json.load(f)
                    for username, user_data in data.items():
                        self.users[username] = User.from_dict(user_data)
                logger.info(f"Cargados {len(self.users)} usuarios")
            except Exception as e:
                logger.error(f"Error cargando usuarios: {e}")

    def _save_users(self):
        """Guarda usuarios a archivo"""
        os.makedirs(os.path.dirname(USERS_DB_FILE), exist_ok=True)
        try:
            with open(USERS_DB_FILE, 'w') as f:
                data = {username: user.to_dict() for username, user in self.users.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando usuarios: {e}")

    def _hash_password(self, password: str) -> str:
        """Hash de contraseña"""
        return hashlib.sha256(password.encode()).hexdigest()

    def register(self, username: str, password: str) -> tuple:
        """Registra un nuevo usuario"""
        if not username or len(username) < 3:
            return False, "Usuario debe tener al menos 3 caracteres"
        if not password or len(password) < 4:
            return False, "Contraseña debe tener al menos 4 caracteres"
        if username in self.users:
            return False, "El usuario ya existe"

        user = User(
            username=username,
            password_hash=self._hash_password(password),
            created_at=datetime.now().isoformat(),
            projects=[]
        )
        self.users[username] = user
        self._save_users()
        logger.info(f"Usuario registrado: {username}")
        return True, "Usuario registrado exitosamente"

    def login(self, username: str, password: str) -> tuple:
        """Inicia sesión y devuelve token"""
        if username not in self.users:
            return None, "Usuario no encontrado"

        user = self.users[username]
        if user.password_hash != self._hash_password(password):
            return None, "Contraseña incorrecta"

        # Generar token de sesión
        token = secrets.token_hex(32)
        self.sessions[token] = username
        logger.info(f"Usuario autenticado: {username}")
        return token, "Sesión iniciada"

    def validate_token(self, token: str) -> Optional[str]:
        """Valida token y devuelve username"""
        return self.sessions.get(token)

    def logout(self, token: str) -> bool:
        """Cierra sesión"""
        if token in self.sessions:
            del self.sessions[token]
            return True
        return False

    def get_user(self, username: str) -> Optional[User]:
        """Obtiene usuario por nombre"""
        return self.users.get(username)

    def add_project_to_user(self, username: str, project_name: str):
        """Añade proyecto a usuario"""
        if username in self.users:
            if project_name not in self.users[username].projects:
                self.users[username].projects.append(project_name)
                self._save_users()

    def get_user_projects(self, username: str) -> List[str]:
        """Obtiene proyectos del usuario"""
        if username in self.users:
            return self.users[username].projects
        return []


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

class AgentRole(Enum):
    NOVA = "nova"           # Orquestador principal
    ATLAS = "atlas"         # Arquitecto
    PIXEL = "pixel"         # Frontend
    FORGE = "forge"         # Backend
    AURA = "aura"           # UX Designer
    SENTINEL = "sentinel"   # QA Tester
    ROCKET = "rocket"       # DevOps


AGENT_CONFIGS = {
    AgentRole.NOVA: {
        "name": "Nova",
        "title": "Project Orchestrator",
        "emoji": "🌟",
        "color": "#4A90D9",
        "system_prompt": """Eres Nova, el Project Orchestrator del equipo THAU. Coordinas el desarrollo completo de aplicaciones seleccionando los agentes apropiados.

## TU EQUIPO (6 agentes disponibles):
- 🏛️ Atlas: Arquitecto - Diseña estructura y carpetas del proyecto
- 🎨 Pixel: Frontend - Crea componentes React/HTML/CSS con Tailwind
- 🔧 Forge: Backend - Desarrolla APIs con FastAPI/Node.js
- 💫 Aura: UX Designer - Mejora estilos, animaciones y experiencia
- 🛡️ Sentinel: QA Tester - Crea tests y verifica calidad
- 🚀 Rocket: DevOps - Configura ejecución y despliegue

## CÓMO TRABAJAR:
Cuando el usuario pida crear algo, sigue este proceso:

### Paso 1: PLANIFICACIÓN
Analiza el requerimiento y responde con un plan estructurado:
```
## 📋 Plan de Desarrollo

**Proyecto:** [nombre]
**Descripción:** [qué hace]

### Agentes a usar:
1. 🏛️ Atlas → Definir estructura
2. 🎨 Pixel → Crear componentes [especificar cuáles]
3. 💫 Aura → Añadir estilos [especificar]
4. 🚀 Rocket → Configurar ejecución

### Archivos a generar:
- index.html
- styles.css
- script.js
```

### Paso 2: GENERACIÓN DE CÓDIGO
Genera TODOS los archivos COMPLETOS usando este formato EXACTO:

**index.html**
```html
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Mi App</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <!-- Contenido COMPLETO aquí -->
</body>
</html>
```

**styles.css**
```css
/* Estilos COMPLETOS con animaciones */
```

**script.js**
```javascript
// JavaScript COMPLETO y funcional
```

### Paso 3: INSTRUCCIONES DE EJECUCIÓN
```bash
# Cómo ejecutar el proyecto
```

## REGLAS IMPORTANTES:
1. SIEMPRE genera código COMPLETO, nunca fragmentos
2. Incluye TODOS los archivos necesarios
3. El código debe ser FUNCIONAL y ejecutable
4. Usa HTML/CSS/JS moderno con animaciones
5. Responde en español"""
    },

    AgentRole.ATLAS: {
        "name": "Atlas",
        "title": "System Architect",
        "emoji": "🏛️",
        "color": "#9B59B6",
        "system_prompt": """Eres Atlas, el Arquitecto de Software del equipo THAU.

Tu trabajo es diseñar la estructura COMPLETA del proyecto:
1. Definir estructura de carpetas
2. Listar TODOS los archivos necesarios
3. Elegir tecnologías apropiadas
4. Establecer patrones de diseño

Genera la estructura así:
```
proyecto/
├── src/
│   ├── components/
│   │   ├── Header.jsx
│   │   ├── Footer.jsx
│   │   └── ...
│   ├── pages/
│   ├── styles/
│   └── utils/
├── public/
├── package.json
└── README.md
```

Luego genera el código de CADA archivo listado."""
    },

    AgentRole.PIXEL: {
        "name": "Pixel",
        "title": "Frontend Wizard",
        "emoji": "🎨",
        "color": "#3498DB",
        "system_prompt": """Eres Pixel, el Frontend Developer del equipo THAU.

Creas componentes React/Next.js con Tailwind CSS COMPLETOS y FUNCIONALES.

Para cada componente:
1. Importaciones necesarias
2. Props tipadas (si aplica)
3. Estado si es necesario
4. JSX completo con Tailwind
5. Exportación

**Ejemplo de componente COMPLETO:**

**src/components/Hero.jsx**
```jsx
import React from 'react';

export default function Hero({ title, subtitle, ctaText, onCtaClick }) {
  return (
    <section className="min-h-screen bg-gradient-to-br from-blue-600 via-purple-600 to-pink-500 flex items-center justify-center px-4">
      <div className="text-center text-white max-w-4xl">
        <h1 className="text-5xl md:text-7xl font-bold mb-6 animate-fade-in">
          {title || 'Bienvenido'}
        </h1>
        <p className="text-xl md:text-2xl mb-8 opacity-90">
          {subtitle || 'Tu solución digital perfecta'}
        </p>
        <button
          onClick={onCtaClick}
          className="bg-white text-purple-600 px-8 py-4 rounded-full font-bold text-lg hover:bg-opacity-90 hover:scale-105 transition-all duration-300 shadow-lg"
        >
          {ctaText || 'Comenzar Ahora'}
        </button>
      </div>
    </section>
  );
}
```

Siempre genera código COMPLETO, nunca uses "..." o comentarios como "resto del código"."""
    },

    AgentRole.FORGE: {
        "name": "Forge",
        "title": "Backend Architect",
        "emoji": "🔧",
        "color": "#27AE60",
        "system_prompt": """Eres Forge, el Backend Developer del equipo THAU.

Creas APIs y lógica de servidor con Python (FastAPI) o Node.js.

Para cada endpoint/servicio:
1. Imports completos
2. Modelos de datos
3. Lógica de negocio
4. Manejo de errores
5. Documentación

**Ejemplo de API COMPLETA:**

**api/main.py**
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import uvicorn

app = FastAPI(title="Mi API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ContactForm(BaseModel):
    name: str
    email: EmailStr
    message: str

class ContactResponse(BaseModel):
    success: bool
    message: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API funcionando"}

@app.post("/contact", response_model=ContactResponse)
def submit_contact(form: ContactForm):
    # Aquí iría la lógica para guardar/enviar
    return ContactResponse(
        success=True,
        message=f"Gracias {form.name}, hemos recibido tu mensaje"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```"""
    },

    AgentRole.AURA: {
        "name": "Aura",
        "title": "Experience Designer",
        "emoji": "💫",
        "color": "#E74C3C",
        "system_prompt": """Eres Aura, la UX/UI Designer del equipo THAU.

Diseñas experiencias visuales completas con:
1. Paleta de colores coherente
2. Tipografía adecuada
3. Espaciado consistente (8px grid)
4. Animaciones sutiles
5. Accesibilidad

Genera archivos CSS/Tailwind config COMPLETOS:

**tailwind.config.js**
```javascript
module.exports = {
  content: ['./src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
        secondary: {
          500: '#8b5cf6',
          600: '#7c3aed',
        },
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.5s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
```"""
    },

    AgentRole.SENTINEL: {
        "name": "Sentinel",
        "title": "Quality Guardian",
        "emoji": "🛡️",
        "color": "#F39C12",
        "system_prompt": """Eres Sentinel, el QA Tester del equipo THAU.

Creas tests completos para verificar que todo funciona:
1. Tests unitarios
2. Tests de componentes
3. Tests de integración

**Ejemplo de tests COMPLETOS:**

**src/__tests__/Hero.test.jsx**
```jsx
import { render, screen, fireEvent } from '@testing-library/react';
import Hero from '../components/Hero';

describe('Hero Component', () => {
  test('renders with default props', () => {
    render(<Hero />);
    expect(screen.getByText('Bienvenido')).toBeInTheDocument();
  });

  test('renders custom title', () => {
    render(<Hero title="Mi Título" />);
    expect(screen.getByText('Mi Título')).toBeInTheDocument();
  });

  test('calls onCtaClick when button clicked', () => {
    const mockClick = jest.fn();
    render(<Hero onCtaClick={mockClick} />);
    fireEvent.click(screen.getByRole('button'));
    expect(mockClick).toHaveBeenCalledTimes(1);
  });
});
```"""
    },

    AgentRole.ROCKET: {
        "name": "Rocket",
        "title": "Launch Engineer",
        "emoji": "🚀",
        "color": "#1ABC9C",
        "system_prompt": """Eres Rocket, el DevOps del equipo THAU.

Configuras todo para que el proyecto sea ejecutable:
1. package.json con scripts
2. Dockerfile si es necesario
3. Configuración de build
4. README con instrucciones

**Ejemplo de package.json COMPLETO:**

**package.json**
```json
{
  "name": "mi-proyecto",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "test": "jest"
  },
  "dependencies": {
    "next": "14.0.0",
    "react": "18.2.0",
    "react-dom": "18.2.0"
  },
  "devDependencies": {
    "tailwindcss": "^3.3.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0",
    "@testing-library/react": "^14.0.0",
    "@testing-library/jest-dom": "^6.0.0",
    "jest": "^29.0.0",
    "jest-environment-jsdom": "^29.0.0"
  }
}
```

**Instrucciones de ejecución:**
```bash
# Instalar dependencias
npm install

# Ejecutar en desarrollo
npm run dev

# La aplicación estará en http://localhost:3000
```"""
    },
}


# ============================================================================
# PROJECT MANAGER
# ============================================================================

@dataclass
class Project:
    """Representa un proyecto en desarrollo"""
    name: str
    path: Path
    type: str = "nextjs"  # nextjs, react, python, html
    files: Dict[str, str] = field(default_factory=dict)
    status: str = "planning"
    process: Optional[subprocess.Popen] = None
    port: int = 3000

    def to_dict(self):
        return {
            "name": self.name,
            "path": str(self.path),
            "type": self.type,
            "files": list(self.files.keys()),
            "status": self.status,
            "port": self.port
        }


class ProjectManager:
    """Gestiona proyectos y su ejecución"""

    def __init__(self):
        self.projects: Dict[str, Project] = {}
        self.base_dir = Path(DEFAULT_PROJECT_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # Encontrar primer puerto disponible para proyectos
        self.next_port = find_available_port(DEFAULT_PREVIEW_PORT)
        logger.info(f"Puerto base para proyectos: {self.next_port}")
        # Cargar proyectos existentes desde disco
        self._load_projects_from_disk()

    def _load_projects_from_disk(self):
        """Carga proyectos existentes desde el directorio base"""
        if not self.base_dir.exists():
            return

        # Cargar metadata si existe
        if os.path.exists(PROJECTS_DB_FILE):
            try:
                with open(PROJECTS_DB_FILE, 'r') as f:
                    metadata = json.load(f)
            except:
                metadata = {}
        else:
            metadata = {}

        # Escanear directorios de proyectos
        for project_dir in self.base_dir.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith('.'):
                safe_name = project_dir.name
                # Detectar tipo de proyecto
                project_type = self._detect_project_type(project_dir)

                # Obtener metadata o usar defaults
                proj_meta = metadata.get(safe_name, {})
                port = self._get_next_available_port()

                project = Project(
                    name=proj_meta.get('name', safe_name.replace('_', ' ').title()),
                    path=project_dir,
                    type=project_type,
                    port=port,
                    status=proj_meta.get('status', 'ready')
                )

                # Cargar archivos existentes
                for file_path in project_dir.rglob('*'):
                    if file_path.is_file() and 'node_modules' not in str(file_path) and '.git' not in str(file_path):
                        rel_path = str(file_path.relative_to(project_dir))
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                project.files[rel_path] = f.read()
                        except:
                            pass

                self.projects[safe_name] = project
                logger.info(f"Proyecto cargado: {safe_name} ({project_type})")

        logger.info(f"Total proyectos cargados: {len(self.projects)}")

    def _detect_project_type(self, project_dir: Path) -> str:
        """Detecta el tipo de proyecto basándose en los archivos"""
        if (project_dir / "package.json").exists():
            # Leer package.json para detectar si es Next.js o React
            try:
                with open(project_dir / "package.json", 'r') as f:
                    pkg = json.load(f)
                    deps = pkg.get('dependencies', {})
                    if 'next' in deps:
                        return 'nextjs'
                    elif 'react' in deps:
                        return 'react'
            except:
                pass
            return 'nextjs'
        elif (project_dir / "main.py").exists() or (project_dir / "app.py").exists():
            return 'python'
        elif (project_dir / "index.html").exists():
            return 'html'
        return 'html'

    def _save_projects_metadata(self):
        """Guarda metadata de proyectos"""
        os.makedirs(os.path.dirname(PROJECTS_DB_FILE), exist_ok=True)
        try:
            metadata = {}
            for name, project in self.projects.items():
                metadata[name] = {
                    'name': project.name,
                    'type': project.type,
                    'status': project.status,
                    'created_at': getattr(project, 'created_at', datetime.now().isoformat())
                }
            with open(PROJECTS_DB_FILE, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando metadata de proyectos: {e}")

    def _get_next_available_port(self) -> int:
        """Obtiene el siguiente puerto disponible para proyectos"""
        port = find_available_port(self.next_port)
        self.next_port = port + 1
        return port

    def create_project(self, name: str, project_type: str = "nextjs", username: str = None) -> Project:
        """Crea un nuevo proyecto"""
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
        project_path = self.base_dir / safe_name
        project_path.mkdir(parents=True, exist_ok=True)

        # Obtener puerto disponible
        port = self._get_next_available_port()

        project = Project(
            name=name,
            path=project_path,
            type=project_type,
            port=port
        )
        self.projects[safe_name] = project

        # Guardar metadata
        self._save_projects_metadata()

        logger.info(f"Proyecto creado: {safe_name} en {project_path} (puerto: {port})")
        return project

    def delete_project(self, name: str) -> bool:
        """Elimina un proyecto"""
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
        if safe_name not in self.projects:
            return False

        project = self.projects[safe_name]

        # Detener si está corriendo
        if project.process:
            self.stop_project(project)

        # Eliminar directorio
        import shutil
        try:
            shutil.rmtree(project.path)
        except Exception as e:
            logger.error(f"Error eliminando directorio: {e}")

        del self.projects[safe_name]
        self._save_projects_metadata()

        logger.info(f"Proyecto eliminado: {safe_name}")
        return True

    def get_all_projects(self) -> List[Dict]:
        """Retorna lista de todos los proyectos"""
        return [
            {
                "id": name,
                "name": proj.name,
                "type": proj.type,
                "status": proj.status,
                "file_count": len(proj.files),
                "path": str(proj.path),
                "port": proj.port
            }
            for name, proj in self.projects.items()
        ]

    def create_project_zip(self, name: str) -> Optional[io.BytesIO]:
        """Crea un archivo ZIP del proyecto"""
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
        project = self.projects.get(safe_name)
        if not project:
            return None

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in project.path.rglob('*'):
                if file_path.is_file() and 'node_modules' not in str(file_path) and '.git' not in str(file_path):
                    arc_name = str(file_path.relative_to(project.path))
                    zip_file.write(file_path, arc_name)

        zip_buffer.seek(0)
        return zip_buffer

    def get_project(self, name: str) -> Optional[Project]:
        """Obtiene un proyecto por nombre"""
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
        return self.projects.get(safe_name)

    def save_file(self, project: Project, filepath: str, content: str) -> str:
        """Guarda un archivo en el proyecto"""
        full_path = project.path / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        project.files[filepath] = content
        logger.info(f"Archivo guardado: {filepath}")
        return str(full_path)

    def extract_and_save_files(self, project: Project, text: str) -> List[str]:
        """Extrae archivos del texto y los guarda"""
        saved = []

        # Múltiples patrones para detectar archivos
        patterns = [
            # Patrón 1: **ruta/archivo.ext** seguido de ```lang
            r'\*\*([^*]+\.[a-zA-Z0-9]+)\*\*\s*\n```(?:\w+)?\n(.*?)```',
            # Patrón 2: `ruta/archivo.ext` seguido de ```lang
            r'`([^`]+\.[a-zA-Z0-9]+)`\s*\n```(?:\w+)?\n(.*?)```',
            # Patrón 3: <!-- ruta/archivo.ext --> seguido de código
            r'<!--\s*([^>]+\.[a-zA-Z0-9]+)\s*-->\s*\n```(?:\w+)?\n(.*?)```',
            # Patrón 4: // archivo: ruta/archivo.ext seguido de código
            r'(?://|#)\s*(?:archivo|file):\s*([^\n]+\.[a-zA-Z0-9]+)\s*\n```(?:\w+)?\n(.*?)```',
            # Patrón 5: ruta/archivo.ext: con código después
            r'^([a-zA-Z0-9_/.-]+\.[a-zA-Z0-9]+):\s*\n```(?:\w+)?\n(.*?)```',
        ]

        found_files = {}
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.DOTALL | re.MULTILINE):
                filepath = match.group(1).strip()
                content = match.group(2).strip()

                # Limpiar el path
                filepath = filepath.lstrip('./')
                filepath = filepath.strip('`*')

                if content and len(content) > 10 and filepath not in found_files:
                    found_files[filepath] = content

        # Guardar todos los archivos encontrados
        for filepath, content in found_files.items():
            self.save_file(project, filepath, content)
            saved.append(filepath)

        return saved

    def run_project(self, project: Project) -> Dict:
        """Ejecuta el proyecto"""
        if project.process and project.process.poll() is None:
            return {"status": "running", "port": project.port}

        try:
            if project.type in ["nextjs", "react"]:
                # Primero instalar dependencias si existe package.json
                package_json = project.path / "package.json"
                if package_json.exists():
                    logger.info("Instalando dependencias...")
                    install_process = subprocess.run(
                        ["npm", "install"],
                        cwd=project.path,
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    if install_process.returncode != 0:
                        logger.error(f"Error npm install: {install_process.stderr}")

                # Ejecutar dev server
                project.process = subprocess.Popen(
                    ["npm", "run", "dev", "--", "-p", str(project.port)],
                    cwd=project.path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                project.status = "running"
                return {"status": "running", "port": project.port, "url": f"http://localhost:{project.port}"}

            elif project.type == "python":
                main_file = project.path / "main.py"
                if main_file.exists():
                    project.process = subprocess.Popen(
                        [sys.executable, "main.py"],
                        cwd=project.path,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    project.status = "running"
                    return {"status": "running", "port": project.port}

            elif project.type == "html":
                # Para HTML simple, usar un servidor básico
                project.process = subprocess.Popen(
                    [sys.executable, "-m", "http.server", str(project.port)],
                    cwd=project.path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                project.status = "running"
                return {"status": "running", "port": project.port, "url": f"http://localhost:{project.port}"}

            return {"status": "error", "message": "Tipo de proyecto no soportado"}

        except Exception as e:
            logger.error(f"Error ejecutando proyecto: {e}")
            return {"status": "error", "message": str(e)}

    def stop_project(self, project: Project) -> Dict:
        """Detiene el proyecto"""
        if project.process:
            project.process.terminate()
            try:
                project.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                project.process.kill()
            project.process = None
            project.status = "stopped"
            return {"status": "stopped"}
        return {"status": "not_running"}

    def get_file_content(self, project: Project, filepath: str) -> Optional[str]:
        """Lee el contenido de un archivo"""
        full_path = project.path / filepath
        if full_path.exists():
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    def list_files(self, project: Project) -> List[str]:
        """Lista todos los archivos del proyecto"""
        files = []
        for path in project.path.rglob('*'):
            if path.is_file() and 'node_modules' not in str(path) and '.git' not in str(path):
                files.append(str(path.relative_to(project.path)))
        return sorted(files)


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

        if context:
            messages.append({
                "role": "system",
                "content": f"Contexto del proyecto:\n{context}"
            })

        messages.extend(self.conversation_history[-6:])
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
                        "num_predict": 4096
                    }
                },
                timeout=GENERATION_TIMEOUT
            )

            if response.status_code == 200:
                result = response.json()
                assistant_message = result.get("message", {}).get("content", "")

                self.conversation_history.append({"role": "user", "content": message})
                self.conversation_history.append({"role": "assistant", "content": assistant_message})

                return assistant_message
            else:
                return f"Error: {response.status_code}"

        except Exception as e:
            return f"Error de comunicación: {str(e)}"

    def clear_history(self):
        self.conversation_history = []


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class NovaOrchestrator:
    """Nova - Orquestador principal que coordina todo el desarrollo"""

    def __init__(self, project_manager: ProjectManager):
        self.project_manager = project_manager
        self.agents: Dict[AgentRole, Agent] = {}
        self.current_project: Optional[Project] = None

        # Inicializar agentes
        for role, config in AGENT_CONFIGS.items():
            self.agents[role] = Agent(role=role, config=config)

    def get_agent(self, role: AgentRole) -> Agent:
        return self.agents.get(role)

    def _get_fallback_html(self, project_name: str, requirements: str) -> str:
        """Template HTML de fallback cuando el modelo no genera código"""
        title = project_name.replace('_', ' ').title()
        return f'''<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(20px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        @keyframes float {{ 0%, 100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-10px); }} }}
        .animate-fade-in {{ animation: fadeIn 0.8s ease-out forwards; }}
        .animate-float {{ animation: float 3s ease-in-out infinite; }}
        .gradient-bg {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
    </style>
</head>
<body class="min-h-screen gradient-bg">
    <!-- Hero Section -->
    <header class="container mx-auto px-6 py-16">
        <nav class="flex justify-between items-center mb-16">
            <h1 class="text-2xl font-bold text-white">{title}</h1>
            <div class="space-x-6">
                <a href="#features" class="text-white/80 hover:text-white transition">Características</a>
                <a href="#about" class="text-white/80 hover:text-white transition">Nosotros</a>
                <a href="#contact" class="bg-white text-purple-600 px-6 py-2 rounded-full font-semibold hover:bg-purple-100 transition">Contacto</a>
            </div>
        </nav>

        <div class="flex flex-col md:flex-row items-center gap-12">
            <div class="flex-1 animate-fade-in">
                <h2 class="text-5xl md:text-6xl font-bold text-white mb-6 leading-tight">
                    Bienvenido a<br><span class="text-yellow-300">{title}</span>
                </h2>
                <p class="text-xl text-white/80 mb-8">
                    {requirements[:200] if requirements else 'Tu solución innovadora para el futuro digital.'}
                </p>
                <div class="flex gap-4">
                    <button class="bg-yellow-400 text-purple-900 px-8 py-3 rounded-full font-bold hover:bg-yellow-300 transition transform hover:scale-105">
                        Comenzar Ahora
                    </button>
                    <button class="border-2 border-white text-white px-8 py-3 rounded-full font-bold hover:bg-white/10 transition">
                        Saber Más
                    </button>
                </div>
            </div>
            <div class="flex-1 animate-float">
                <div class="w-80 h-80 bg-white/20 rounded-3xl backdrop-blur-sm flex items-center justify-center">
                    <span class="text-8xl">🚀</span>
                </div>
            </div>
        </div>
    </header>

    <!-- Features Section -->
    <section id="features" class="bg-white py-20">
        <div class="container mx-auto px-6">
            <h3 class="text-4xl font-bold text-center text-gray-800 mb-16">Características</h3>
            <div class="grid md:grid-cols-3 gap-8">
                <div class="bg-gradient-to-br from-purple-50 to-blue-50 p-8 rounded-2xl text-center hover:shadow-xl transition">
                    <div class="text-5xl mb-4">⚡</div>
                    <h4 class="text-xl font-bold text-gray-800 mb-2">Rápido</h4>
                    <p class="text-gray-600">Rendimiento optimizado para la mejor experiencia.</p>
                </div>
                <div class="bg-gradient-to-br from-purple-50 to-blue-50 p-8 rounded-2xl text-center hover:shadow-xl transition">
                    <div class="text-5xl mb-4">🔒</div>
                    <h4 class="text-xl font-bold text-gray-800 mb-2">Seguro</h4>
                    <p class="text-gray-600">Protección de datos de última generación.</p>
                </div>
                <div class="bg-gradient-to-br from-purple-50 to-blue-50 p-8 rounded-2xl text-center hover:shadow-xl transition">
                    <div class="text-5xl mb-4">💡</div>
                    <h4 class="text-xl font-bold text-gray-800 mb-2">Innovador</h4>
                    <p class="text-gray-600">Tecnología de vanguardia a tu alcance.</p>
                </div>
            </div>
        </div>
    </section>

    <!-- Contact Section -->
    <section id="contact" class="gradient-bg py-20">
        <div class="container mx-auto px-6 text-center">
            <h3 class="text-4xl font-bold text-white mb-8">¿Listo para comenzar?</h3>
            <p class="text-xl text-white/80 mb-8">Contáctanos hoy y descubre todo lo que podemos hacer por ti.</p>
            <form class="max-w-md mx-auto flex gap-4">
                <input type="email" placeholder="tu@email.com" class="flex-1 px-6 py-3 rounded-full focus:outline-none focus:ring-2 focus:ring-yellow-400">
                <button type="submit" class="bg-yellow-400 text-purple-900 px-8 py-3 rounded-full font-bold hover:bg-yellow-300 transition">
                    Enviar
                </button>
            </form>
        </div>
    </section>

    <!-- Footer -->
    <footer class="bg-gray-900 text-white py-8">
        <div class="container mx-auto px-6 text-center">
            <p>&copy; 2024 {title}. Creado con 💜 por THAU Team</p>
        </div>
    </footer>
</body>
</html>'''

    def _get_fallback_css(self) -> str:
        """CSS adicional de fallback"""
        return '''/* Estilos adicionales */
:root {
    --primary: #667eea;
    --secondary: #764ba2;
    --accent: #fbbf24;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    line-height: 1.6;
}

/* Smooth scrolling */
html {
    scroll-behavior: smooth;
}

/* Button hover effects */
button {
    cursor: pointer;
    transition: all 0.3s ease;
}

button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}

/* Card hover effects */
.card:hover {
    transform: translateY(-5px);
}

/* Input focus */
input:focus {
    outline: none;
    box-shadow: 0 0 0 3px rgba(251, 191, 36, 0.5);
}
'''

    async def develop_project(self, requirements: str, project_name: str, websocket: WebSocket) -> Project:
        """Desarrolla un proyecto completo coordinando todos los agentes"""

        # Crear proyecto como HTML simple (más fácil de ejecutar)
        project = self.project_manager.create_project(project_name, project_type="html")
        self.current_project = project
        total_saved = []

        # Fase 1: Nova analiza y planifica
        await self._send_update(websocket, "Nova", "🌟", "Analizando requerimientos y planificando el proyecto...")

        nova = self.get_agent(AgentRole.NOVA)
        plan = nova.get_response(f"""
Proyecto: {project_name}
Requerimientos: {requirements}

IMPORTANTE: Genera código HTML completo. Usa este formato:

**index.html**
```html
<!DOCTYPE html>
<html>
... código completo ...
</html>
```

Incluye:
1. HTML con secciones hero, características, contacto
2. Tailwind CSS via CDN
3. Animaciones CSS
4. Diseño responsive
""")
        await self._send_update(websocket, "Nova", "🌟", plan, project.to_dict())

        saved = self.project_manager.extract_and_save_files(project, plan)
        total_saved.extend(saved)

        # Fase 2: Pixel crea el frontend
        await self._send_update(websocket, "Pixel", "🎨", "Creando el diseño visual...")

        pixel = self.get_agent(AgentRole.PIXEL)
        frontend = pixel.get_response(f"""
Crea el archivo index.html COMPLETO para: {requirements}

OBLIGATORIO usar este formato exacto:

**index.html**
```html
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>{project_name}</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body>
    <!-- Hero con gradiente -->
    <!-- Sección de características -->
    <!-- Formulario de contacto -->
    <!-- Footer -->
</body>
</html>
```

Incluye animaciones y transiciones CSS.
""")
        await self._send_update(websocket, "Pixel", "🎨", frontend, project.to_dict())

        saved = self.project_manager.extract_and_save_files(project, frontend)
        total_saved.extend(saved)

        # Fase 3: Aura mejora los estilos
        await self._send_update(websocket, "Aura", "💫", "Añadiendo estilos y animaciones...")

        aura = self.get_agent(AgentRole.AURA)
        styles = aura.get_response(f"""
Crea un archivo CSS adicional con:

**styles.css**
```css
/* Animaciones */
@keyframes fadeIn {{ ... }}

/* Estilos hover */
button:hover {{ ... }}

/* Responsive */
@media (max-width: 768px) {{ ... }}
```
""")
        await self._send_update(websocket, "Aura", "💫", styles, project.to_dict())

        saved = self.project_manager.extract_and_save_files(project, styles)
        total_saved.extend(saved)

        # FALLBACK: Si no se crearon archivos, usar templates
        if not total_saved or 'index.html' not in total_saved:
            await self._send_update(websocket, "Sistema", "⚠️", "El modelo no generó código completo. Aplicando template de respaldo...")

            # Crear index.html de fallback
            self.project_manager.save_file(project, "index.html", self._get_fallback_html(project_name, requirements))
            total_saved.append("index.html")

            # Crear styles.css de fallback
            self.project_manager.save_file(project, "styles.css", self._get_fallback_css())
            total_saved.append("styles.css")

            await self._send_update(websocket, "Sistema", "✅", "Template de respaldo aplicado exitosamente.", project.to_dict())

        # Resumen final
        all_files = self.project_manager.list_files(project)
        summary = f"""
## ✅ Proyecto Completado: {project_name}

### Archivos Creados ({len(all_files)}):
{chr(10).join(['- ' + f for f in all_files[:20]])}

### Ubicación:
`{project.path}`

### Para ver el proyecto:
1. Haz clic en **"▶️ Ejecutar Proyecto"**
2. Luego **"🌐 Abrir en Navegador"**

### O manualmente:
```bash
cd {project.path}
python -m http.server {project.port}
# Abre http://localhost:{project.port}
```
"""
        project.status = "ready"
        await self._send_update(websocket, "Nova", "🌟", summary, project.to_dict())

        return project

    async def _send_update(self, websocket: WebSocket, agent: str, emoji: str, message: str, project_data: Dict = None):
        """Envía actualización al cliente"""
        try:
            data = {
                "agent": agent,
                "emoji": emoji,
                "response": message,
                "project": project_data,
                "files": list(self.current_project.files.keys()) if self.current_project else []
            }
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Error enviando actualización: {e}")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="THAU Development Orchestrator")
project_manager = ProjectManager()
user_manager = UserManager()
orchestrator = NovaOrchestrator(project_manager)


@app.get("/assets/logo.svg")
async def get_logo():
    """Sirve el logo SVG de THAU"""
    logo_path = Path(__file__).parent.parent / "assets" / "thau_logo.svg"
    if logo_path.exists():
        with open(logo_path, "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="image/svg+xml")
    return Response(content="", status_code=404)


# ============================================================================
# TTS VOICE ENDPOINTS
# ============================================================================

# Instancia global de TTS para THAU
thau_tts = ThauVoice(voice="eddy_mx", rate=185)


@app.post("/api/tts/synthesize")
async def synthesize_speech(data: Dict = None):
    """Sintetiza texto a voz y retorna el audio como base64"""
    if not data:
        raise HTTPException(status_code=400, detail="Datos requeridos")

    text = data.get("text", "")
    voice = data.get("voice", "eddy_mx")

    if not text:
        raise HTTPException(status_code=400, detail="Texto requerido")

    try:
        # Crear instancia con la voz seleccionada
        tts = ThauVoice(voice=voice, rate=185)
        audio_base64 = tts.synthesize_to_base64(text)

        return {
            "success": True,
            "audio": audio_base64,
            "format": "aiff",  # pyttsx3 on macOS produces AIFF
            "voice": voice
        }
    except Exception as e:
        logger.error(f"Error TTS: {e}")
        raise HTTPException(status_code=500, detail=f"Error de síntesis: {str(e)}")


@app.get("/api/tts/voices")
async def list_tts_voices():
    """Lista las voces disponibles para TTS"""
    return {
        "voices": list(SPANISH_VOICES.keys()),
        "default": "eddy_mx",
        "details": {
            name: {"id": voice_id, "type": "male" if any(m in name for m in ["eddy", "rocko", "reed", "grandpa"]) else "female"}
            for name, voice_id in SPANISH_VOICES.items()
        }
    }


@app.get("/api/tts/audio/{text}")
async def get_tts_audio(text: str, voice: str = "eddy_mx"):
    """Genera y retorna el archivo de audio directamente"""
    try:
        tts = ThauVoice(voice=voice, rate=185)
        audio_path = tts.synthesize(text)

        return FileResponse(
            audio_path,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=thau_voice.wav"}
        )
    except Exception as e:
        logger.error(f"Error TTS: {e}")
        raise HTTPException(status_code=500, detail=f"Error de síntesis: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def home():
    """Página principal"""
    template_path = Path(__file__).parent / "templates" / "orchestrator.html"
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Template not found. Run the setup first.</h1>"


def is_development_request(message: str) -> tuple:
    """Detecta si el mensaje es una solicitud de desarrollo y extrae nombre del proyecto"""
    message_lower = message.lower()

    # Palabras clave que indican solicitud de desarrollo
    dev_keywords = [
        'crea', 'crear', 'desarrolla', 'desarrollar', 'genera', 'generar',
        'haz', 'hacer', 'construye', 'construir', 'implementa', 'implementar',
        'diseña', 'diseñar', 'landing page', 'aplicación', 'app', 'página',
        'web', 'sitio', 'componente', 'sistema', 'dashboard', 'formulario',
        'create', 'build', 'make', 'develop', 'implement', 'design'
    ]

    # Verificar si contiene palabras clave de desarrollo
    is_dev = any(kw in message_lower for kw in dev_keywords)

    # Extraer posible nombre del proyecto
    project_name = "mi_proyecto"

    # Buscar patrones como "landing page para X", "app de X", etc.
    import re
    patterns = [
        r'(?:para|de|about|for)\s+(?:una?\s+)?(\w+(?:\s+\w+)?(?:\s+de\s+\w+)?)',
        r'(?:landing|page|app|web|sitio)\s+(?:de\s+)?(\w+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, message_lower)
        if match:
            name = match.group(1).strip()[:30]  # Limitar longitud
            if len(name) > 3:
                project_name = name.replace(' ', '_')
                break

    return is_dev, project_name


@app.websocket("/ws/develop")
async def websocket_develop(websocket: WebSocket):
    """WebSocket para desarrollo de proyectos"""
    await websocket.accept()
    logger.info("Cliente conectado")

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action", "chat")

            if action == "develop":
                # Desarrollo completo de proyecto
                requirements = data.get("requirements", "")
                project_name = data.get("project", "mi_proyecto")

                await orchestrator.develop_project(requirements, project_name, websocket)

            elif action == "chat":
                # Chat con un agente específico
                agent_name = data.get("agent", "nova")
                message = data.get("message", "")
                project_from_client = data.get("project", "")

                try:
                    role = AgentRole(agent_name)
                except ValueError:
                    role = AgentRole.NOVA

                # Si es Nova y el mensaje parece una solicitud de desarrollo, usar flujo completo
                if role == AgentRole.NOVA:
                    is_dev, detected_project = is_development_request(message)
                    if is_dev:
                        # Usar el nombre del proyecto del cliente si existe, sino el detectado
                        project_name = project_from_client if project_from_client else detected_project
                        logger.info(f"Detectada solicitud de desarrollo: {project_name}")

                        # Notificar al usuario
                        await websocket.send_json({
                            "agent": "Nova",
                            "emoji": "🌟",
                            "response": f"🚀 **Detecté una solicitud de desarrollo!**\n\nVoy a coordinar al equipo completo para crear tu proyecto: **{project_name}**\n\nIniciando orquestación de agentes...",
                            "files": []
                        })

                        # Ejecutar flujo de desarrollo completo
                        await orchestrator.develop_project(message, project_name, websocket)
                        continue

                agent = orchestrator.get_agent(role)
                response = agent.get_response(message)

                # Extraer archivos si hay proyecto activo
                if orchestrator.current_project:
                    saved = project_manager.extract_and_save_files(orchestrator.current_project, response)
                    files = list(orchestrator.current_project.files.keys())
                    if saved:
                        await websocket.send_json({
                            "agent": "Sistema",
                            "emoji": "📁",
                            "response": f"Archivos guardados: {', '.join(saved)}",
                            "files": files
                        })
                else:
                    files = []

                await websocket.send_json({
                    "agent": agent.name,
                    "emoji": agent.emoji,
                    "response": response,
                    "files": files
                })

            elif action == "run":
                # Ejecutar proyecto
                project_name = data.get("project", "")
                project = project_manager.get_project(project_name)

                if project:
                    result = project_manager.run_project(project)
                    await websocket.send_json({
                        "agent": "Rocket",
                        "emoji": "🚀",
                        "response": f"Proyecto iniciado: {result}",
                        "project": project.to_dict()
                    })
                else:
                    await websocket.send_json({
                        "agent": "Sistema",
                        "emoji": "⚠️",
                        "response": "Proyecto no encontrado"
                    })

            elif action == "stop":
                # Detener proyecto
                project_name = data.get("project", "")
                project = project_manager.get_project(project_name)

                if project:
                    result = project_manager.stop_project(project)
                    await websocket.send_json({
                        "agent": "Rocket",
                        "emoji": "🛑",
                        "response": f"Proyecto detenido: {result}",
                        "project": project.to_dict()
                    })

            elif action == "files":
                # Listar archivos
                project_name = data.get("project", "")
                project = project_manager.get_project(project_name)

                if project:
                    files = project_manager.list_files(project)
                    await websocket.send_json({
                        "agent": "Sistema",
                        "emoji": "📁",
                        "response": f"Archivos del proyecto:\n" + "\n".join(files),
                        "files": files
                    })

            elif action == "read":
                # Leer archivo
                project_name = data.get("project", "")
                filepath = data.get("file", "")
                project = project_manager.get_project(project_name)

                if project:
                    content = project_manager.get_file_content(project, filepath)
                    if content:
                        await websocket.send_json({
                            "agent": "Sistema",
                            "emoji": "📄",
                            "response": f"**{filepath}**\n```\n{content}\n```",
                            "file_content": content
                        })

    except WebSocketDisconnect:
        logger.info("Cliente desconectado")
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")


@app.get("/api/projects")
async def list_projects():
    """Lista todos los proyectos"""
    return {
        name: proj.to_dict()
        for name, proj in project_manager.projects.items()
    }


@app.get("/api/projects/{name}/files")
async def get_project_files(name: str):
    """Lista archivos de un proyecto"""
    project = project_manager.get_project(name)
    if not project:
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")
    return {"files": project_manager.list_files(project)}


@app.get("/api/projects/{name}/files/{filepath:path}")
async def get_file(name: str, filepath: str):
    """Lee un archivo del proyecto"""
    project = project_manager.get_project(name)
    if not project:
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")

    content = project_manager.get_file_content(project, filepath)
    if content is None:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

    return {"file": filepath, "content": content}


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/api/auth/register")
async def register_user(data: Dict = None):
    """Registra un nuevo usuario"""
    if not data:
        raise HTTPException(status_code=400, detail="Datos requeridos")

    username = data.get("username", "")
    password = data.get("password", "")

    success, message = user_manager.register(username, password)
    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {"success": True, "message": message}


@app.post("/api/auth/login")
async def login_user(data: Dict = None):
    """Inicia sesión"""
    if not data:
        raise HTTPException(status_code=400, detail="Datos requeridos")

    username = data.get("username", "")
    password = data.get("password", "")

    token, message = user_manager.login(username, password)
    if not token:
        raise HTTPException(status_code=401, detail=message)

    return {
        "success": True,
        "token": token,
        "username": username,
        "message": message
    }


@app.post("/api/auth/logout")
async def logout_user(data: Dict = None):
    """Cierra sesión"""
    if not data:
        raise HTTPException(status_code=400, detail="Datos requeridos")

    token = data.get("token", "")
    user_manager.logout(token)
    return {"success": True, "message": "Sesión cerrada"}


@app.get("/api/auth/validate/{token}")
async def validate_token(token: str):
    """Valida un token de sesión"""
    username = user_manager.validate_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Token inválido")

    user = user_manager.get_user(username)
    return {
        "valid": True,
        "username": username,
        "projects": user.projects if user else []
    }


# ============================================================================
# PROJECT MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/api/projects/list")
async def list_all_projects():
    """Lista todos los proyectos con información detallada"""
    return {
        "projects": project_manager.get_all_projects(),
        "total": len(project_manager.projects)
    }


@app.post("/api/projects/create")
async def create_new_project(data: Dict = None):
    """Crea un nuevo proyecto vacío"""
    if not data:
        raise HTTPException(status_code=400, detail="Datos requeridos")

    name = data.get("name", "")
    project_type = data.get("type", "html")
    token = data.get("token", "")

    if not name:
        raise HTTPException(status_code=400, detail="Nombre requerido")

    # Verificar si el proyecto ya existe
    if project_manager.get_project(name):
        raise HTTPException(status_code=400, detail="El proyecto ya existe")

    # Obtener usuario si hay token
    username = user_manager.validate_token(token) if token else None

    project = project_manager.create_project(name, project_type)

    # Asociar proyecto al usuario
    if username:
        user_manager.add_project_to_user(username, name)

    return {
        "success": True,
        "project": project.to_dict(),
        "message": f"Proyecto '{name}' creado"
    }


@app.delete("/api/projects/{name}")
async def delete_existing_project(name: str):
    """Elimina un proyecto"""
    if not project_manager.delete_project(name):
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")

    return {"success": True, "message": f"Proyecto '{name}' eliminado"}


@app.get("/api/projects/{name}/download")
async def download_project_zip(name: str):
    """Descarga el proyecto como archivo ZIP"""
    zip_buffer = project_manager.create_project_zip(name)
    if not zip_buffer:
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")

    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.zip"'
        }
    )


@app.post("/api/projects/{name}/select")
async def select_project(name: str):
    """Selecciona un proyecto como activo"""
    project = project_manager.get_project(name)
    if not project:
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")

    # Actualizar el proyecto actual en el orquestador
    orchestrator.current_project = project

    return {
        "success": True,
        "project": project.to_dict(),
        "files": list(project.files.keys())
    }


@app.get("/api/user/{username}/projects")
async def get_user_projects(username: str):
    """Obtiene los proyectos de un usuario"""
    projects = user_manager.get_user_projects(username)
    detailed_projects = []

    for proj_name in projects:
        project = project_manager.get_project(proj_name)
        if project:
            detailed_projects.append({
                "id": proj_name,
                "name": project.name,
                "type": project.type,
                "status": project.status,
                "file_count": len(project.files)
            })

    return {
        "username": username,
        "projects": detailed_projects,
        "total": len(detailed_projects)
    }


def main():
    """Función principal"""
    global PORT

    # Determinar puerto disponible para el servidor
    PORT = get_server_port()

    # Mostrar información de puertos en uso
    if PORT != DEFAULT_PORT:
        print(f"\n⚠️  Puerto {DEFAULT_PORT} en uso. Usando puerto alternativo: {PORT}\n")

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              🌟 THAU Development Orchestrator - Nova Edition                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Nova orquesta automáticamente a todo el equipo para desarrollar             ║
║  tu proyecto completo desde cero hasta ejecución.                            ║
║                                                                              ║
║  ⚡ Equipo THAU:                                                             ║
║     🌟 Nova      │ Orquestador      │ Coordina todo                         ║
║     🏛️  Atlas     │ Arquitecto       │ Diseña estructura                     ║
║     🎨 Pixel     │ Frontend         │ React/Tailwind                        ║
║     🔧 Forge     │ Backend          │ APIs/Lógica                           ║
║     💫 Aura      │ UX Designer      │ Estilos/UX                            ║
║     🛡️  Sentinel  │ QA               │ Tests                                 ║
║     🚀 Rocket    │ DevOps           │ Ejecución                             ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  🌐 URL: http://localhost:{PORT:<5}                                           ║
║  📁 Proyectos: {DEFAULT_PROJECT_DIR:<46}║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
