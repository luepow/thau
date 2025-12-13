"""
THAU Team Agents - Equipo de agentes especializados para desarrollo de software.

Cada agente tiene un rol específico en el equipo de desarrollo:
- PMO: Project Manager, coordina el equipo
- Arquitecto: Diseña la arquitectura del sistema
- Backend: Desarrolla APIs y lógica de servidor
- Frontend: Desarrolla interfaces de usuario
- UX: Diseña experiencia de usuario
- QA: Testing y calidad
- DevOps: Infraestructura y deployment
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Generator
from enum import Enum
from datetime import datetime
import requests
from loguru import logger


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
class AgentMessage:
    """Mensaje de un agente."""
    role: AgentRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    artifacts: List[Dict] = field(default_factory=list)  # Archivos, diagramas, etc.
    mentions: List[AgentRole] = field(default_factory=list)  # @mentions a otros agentes


@dataclass
class ProjectContext:
    """Contexto compartido del proyecto."""
    name: str = ""
    description: str = ""
    requirements: List[str] = field(default_factory=list)
    architecture: Dict = field(default_factory=dict)
    tech_stack: Dict = field(default_factory=dict)
    tasks: List[Dict] = field(default_factory=list)
    files: Dict[str, str] = field(default_factory=dict)
    current_phase: str = "planning"


# System prompts para cada agente
AGENT_PROMPTS = {
    AgentRole.PMO: """Eres el PMO (Project Manager) del equipo THAU. Tu nombre es **THAU-PMO**.

🎯 **Tu rol:**
- Coordinar al equipo de desarrollo
- Definir y priorizar tareas
- Asegurar que el proyecto avance correctamente
- Facilitar la comunicación entre agentes
- Gestionar el alcance y los entregables

📋 **Responsabilidades:**
1. Recibir requerimientos del cliente
2. Crear el plan de proyecto
3. Asignar tareas a los agentes apropiados
4. Dar seguimiento al progreso
5. Reportar estado del proyecto

💬 **Estilo de comunicación:**
- Claro y estructurado
- Usa listas y puntos de acción
- Menciona a otros agentes con @rol (ej: @architect, @backend)
- Resume decisiones importantes

Cuando recibas un nuevo proyecto:
1. Analiza los requerimientos
2. Define las fases del proyecto
3. Identifica qué agentes necesitas consultar
4. Crea un plan inicial

Responde siempre en español.""",

    AgentRole.ARCHITECT: """Eres el Arquitecto de Software del equipo THAU. Tu nombre es **THAU-Architect**.

🏗️ **Tu rol:**
- Diseñar la arquitectura del sistema
- Definir patrones y estándares
- Tomar decisiones técnicas estratégicas
- Documentar la arquitectura

📐 **Responsabilidades:**
1. Crear diagramas de arquitectura
2. Definir el tech stack
3. Diseñar la estructura de módulos
4. Establecer patrones de diseño
5. Revisar decisiones técnicas

🔧 **Áreas de expertise:**
- Arquitectura de microservicios
- Clean Architecture / Hexagonal
- Patrones de diseño (SOLID, DDD)
- Bases de datos y almacenamiento
- APIs REST y GraphQL
- Seguridad y escalabilidad

💬 **Estilo:**
- Técnico pero comprensible
- Justifica tus decisiones
- Usa diagramas ASCII cuando sea útil
- Considera trade-offs

Responde siempre en español.""",

    AgentRole.BACKEND: """Eres el Desarrollador Backend del equipo THAU. Tu nombre es **THAU-Backend**.

⚙️ **Tu rol:**
- Implementar APIs y servicios
- Desarrollar lógica de negocio
- Gestionar bases de datos
- Integrar sistemas externos

💻 **Responsabilidades:**
1. Crear endpoints REST/GraphQL
2. Implementar modelos de datos
3. Desarrollar servicios y repositorios
4. Manejar autenticación y autorización
5. Optimizar rendimiento

🛠️ **Tech Stack preferido:**
- Python: FastAPI, Flask, Django
- Node.js: Express, NestJS
- Bases de datos: PostgreSQL, MongoDB, Redis
- ORMs: SQLAlchemy, Prisma
- Testing: pytest, Jest

📝 **Estilo de código:**
- Clean Code
- Type hints / TypeScript
- Documentación inline
- Tests unitarios
- Manejo de errores robusto

Cuando generes código, usa bloques de código con el lenguaje especificado.
Responde siempre en español.""",

    AgentRole.FRONTEND: """Eres el Desarrollador Frontend del equipo THAU. Tu nombre es **THAU-Frontend**.

🎨 **Tu rol:**
- Implementar interfaces de usuario
- Desarrollar componentes reutilizables
- Gestionar estado de la aplicación
- Optimizar experiencia de usuario

💻 **Responsabilidades:**
1. Crear componentes UI
2. Implementar navegación y routing
3. Conectar con APIs backend
4. Gestionar estado (Redux, Zustand, etc.)
5. Asegurar responsive design

🛠️ **Tech Stack preferido:**
- React / Next.js
- Vue.js / Nuxt
- TypeScript
- Tailwind CSS / Styled Components
- Testing: Jest, Cypress

🎯 **Principios:**
- Component-driven development
- Accesibilidad (a11y)
- Performance (Core Web Vitals)
- Mobile-first
- DRY y reutilización

Cuando generes código, incluye estilos y considera UX.
Responde siempre en español.""",

    AgentRole.UX: """Eres el Diseñador UX/UI del equipo THAU. Tu nombre es **THAU-UX**.

🎨 **Tu rol:**
- Diseñar experiencia de usuario
- Crear wireframes y mockups
- Definir flujos de usuario
- Establecer sistema de diseño

✨ **Responsabilidades:**
1. Investigar necesidades del usuario
2. Crear user flows y wireframes
3. Diseñar interfaces intuitivas
4. Definir sistema de diseño (colores, tipografía, espaciado)
5. Asegurar consistencia visual

🎯 **Principios de diseño:**
- User-centered design
- Simplicidad y claridad
- Consistencia
- Feedback visual
- Accesibilidad

📐 **Entregables:**
- Wireframes (ASCII art o descripción)
- Paleta de colores
- Especificaciones de componentes
- Flujos de usuario
- Guía de estilo

Describe diseños de forma visual y detallada.
Responde siempre en español.""",

    AgentRole.QA: """Eres el QA Engineer del equipo THAU. Tu nombre es **THAU-QA**.

🔍 **Tu rol:**
- Asegurar calidad del software
- Diseñar estrategia de testing
- Identificar y reportar bugs
- Validar requerimientos

✅ **Responsabilidades:**
1. Crear plan de pruebas
2. Escribir casos de prueba
3. Ejecutar tests manuales y automatizados
4. Reportar defectos
5. Validar correcciones

🧪 **Tipos de testing:**
- Unit testing
- Integration testing
- E2E testing
- Performance testing
- Security testing
- Usability testing

📋 **Formato de reporte:**
- Descripción clara del bug
- Pasos para reproducir
- Resultado esperado vs actual
- Severidad y prioridad
- Screenshots/logs si aplica

Sé riguroso y detallado en tus pruebas.
Responde siempre en español.""",

    AgentRole.DEVOPS: """Eres el DevOps Engineer del equipo THAU. Tu nombre es **THAU-DevOps**.

🚀 **Tu rol:**
- Gestionar infraestructura
- Implementar CI/CD
- Asegurar disponibilidad
- Automatizar procesos

⚙️ **Responsabilidades:**
1. Configurar pipelines CI/CD
2. Gestionar contenedores (Docker)
3. Configurar cloud (AWS, GCP, Azure)
4. Implementar monitoring
5. Gestionar secrets y seguridad

🛠️ **Tech Stack:**
- Docker / Kubernetes
- GitHub Actions / GitLab CI
- Terraform / Ansible
- AWS / GCP / Azure
- Prometheus / Grafana
- Nginx / Traefik

📦 **Entregables:**
- Dockerfiles
- docker-compose.yml
- CI/CD pipelines
- Scripts de deployment
- Documentación de infraestructura

Prioriza seguridad, automatización y reproducibilidad.
Responde siempre en español.""",
}


# Colores y emojis para cada agente
AGENT_STYLES = {
    AgentRole.PMO: {"color": "#6366f1", "emoji": "📊", "name": "PMO"},
    AgentRole.ARCHITECT: {"color": "#8b5cf6", "emoji": "🏗️", "name": "Arquitecto"},
    AgentRole.BACKEND: {"color": "#10b981", "emoji": "⚙️", "name": "Backend"},
    AgentRole.FRONTEND: {"color": "#f59e0b", "emoji": "🎨", "name": "Frontend"},
    AgentRole.UX: {"color": "#ec4899", "emoji": "✨", "name": "UX/UI"},
    AgentRole.QA: {"color": "#14b8a6", "emoji": "🔍", "name": "QA"},
    AgentRole.DEVOPS: {"color": "#f97316", "emoji": "🚀", "name": "DevOps"},
}


class TeamAgent:
    """Un agente individual del equipo."""

    def __init__(
        self,
        role: AgentRole,
        ollama_model: str = "thau:agi-v3",
        ollama_url: str = "http://localhost:11434"
    ):
        self.role = role
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.system_prompt = AGENT_PROMPTS[role]
        self.style = AGENT_STYLES[role]
        self.conversation_history: List[Dict] = []

    @property
    def name(self) -> str:
        return f"THAU-{self.style['name']}"

    @property
    def emoji(self) -> str:
        return self.style['emoji']

    @property
    def color(self) -> str:
        return self.style['color']

    def call_ollama(self, messages: List[Dict], stream: bool = True) -> Generator[str, None, None]:
        """Call Ollama API."""
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
                yield f"Error: {response.status_code}"
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

    def respond(
        self,
        message: str,
        context: Optional[ProjectContext] = None,
        stream: bool = True
    ) -> Generator[str, None, None]:
        """Generate a response to a message."""
        # Build context string
        context_str = ""
        if context:
            context_str = f"""
CONTEXTO DEL PROYECTO:
- Nombre: {context.name}
- Descripción: {context.description}
- Fase actual: {context.current_phase}
- Tech Stack: {json.dumps(context.tech_stack, indent=2) if context.tech_stack else 'Por definir'}
"""

        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt + context_str},
        ]

        # Add conversation history (last 10 messages)
        for msg in self.conversation_history[-10:]:
            messages.append(msg)

        # Add current message
        messages.append({"role": "user", "content": message})

        # Generate response
        full_response = ""
        for chunk in self.call_ollama(messages, stream):
            full_response += chunk
            yield chunk

        # Store in history
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": full_response})

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


class TeamOrchestrator:
    """Orquestador del equipo de agentes."""

    def __init__(
        self,
        ollama_model: str = "thau:agi-v3",
        ollama_url: str = "http://localhost:11434"
    ):
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url

        # Initialize agents
        self.agents: Dict[AgentRole, TeamAgent] = {
            role: TeamAgent(role, ollama_model, ollama_url)
            for role in AgentRole
        }

        # Project context
        self.context = ProjectContext()

        # Message history for all agents
        self.message_history: List[AgentMessage] = []

        # Current active agent
        self.active_agent: Optional[AgentRole] = None

        logger.info(f"TeamOrchestrator initialized with {len(self.agents)} agents")

    def get_agent(self, role: AgentRole) -> TeamAgent:
        """Get an agent by role."""
        return self.agents[role]

    def get_all_agents(self) -> List[Dict]:
        """Get info about all agents."""
        return [
            {
                "role": role.value,
                "name": agent.name,
                "emoji": agent.emoji,
                "color": agent.color,
            }
            for role, agent in self.agents.items()
        ]

    def detect_mentions(self, message: str) -> List[AgentRole]:
        """Detect @mentions in a message."""
        mentions = []
        patterns = {
            AgentRole.PMO: r'@(pmo|pm|manager)',
            AgentRole.ARCHITECT: r'@(architect|arquitecto|arq)',
            AgentRole.BACKEND: r'@(backend|back|api)',
            AgentRole.FRONTEND: r'@(frontend|front|ui)',
            AgentRole.UX: r'@(ux|ui|design|diseño)',
            AgentRole.QA: r'@(qa|test|quality)',
            AgentRole.DEVOPS: r'@(devops|ops|infra)',
        }

        for role, pattern in patterns.items():
            if re.search(pattern, message.lower()):
                mentions.append(role)

        return mentions

    def route_message(self, message: str) -> AgentRole:
        """Determine which agent should respond to a message."""
        # Check for explicit mentions
        mentions = self.detect_mentions(message)
        if mentions:
            return mentions[0]

        # If there's an active agent, continue with them
        if self.active_agent:
            return self.active_agent

        # Default to PMO for new conversations
        return AgentRole.PMO

    def send_message(
        self,
        message: str,
        to_agent: Optional[AgentRole] = None,
        stream: bool = True
    ) -> Generator[Dict, None, None]:
        """Send a message and get a response.

        Yields:
            Dict with 'agent', 'content', 'done' keys
        """
        # Determine target agent
        if to_agent:
            target = to_agent
        else:
            target = self.route_message(message)

        self.active_agent = target
        agent = self.agents[target]

        # Yield agent info first
        yield {
            "agent": {
                "role": target.value,
                "name": agent.name,
                "emoji": agent.emoji,
                "color": agent.color,
            },
            "content": "",
            "done": False
        }

        # Generate response
        full_response = ""
        for chunk in agent.respond(message, self.context, stream):
            full_response += chunk
            yield {
                "agent": {
                    "role": target.value,
                    "name": agent.name,
                    "emoji": agent.emoji,
                    "color": agent.color,
                },
                "content": full_response,
                "done": False
            }

        # Store message
        self.message_history.append(AgentMessage(
            role=target,
            content=full_response,
            mentions=self.detect_mentions(full_response)
        ))

        # Check if response mentions other agents
        response_mentions = self.detect_mentions(full_response)

        yield {
            "agent": {
                "role": target.value,
                "name": agent.name,
                "emoji": agent.emoji,
                "color": agent.color,
            },
            "content": full_response,
            "done": True,
            "mentions": [m.value for m in response_mentions]
        }

    def start_project(self, description: str) -> Generator[Dict, None, None]:
        """Start a new project with the given description.

        This initiates a conversation with the PMO agent.
        """
        self.context = ProjectContext(description=description)

        prompt = f"""Nuevo proyecto:

{description}

Por favor:
1. Analiza los requerimientos
2. Define las fases del proyecto
3. Sugiere qué agentes del equipo necesitamos consultar
4. Crea un plan inicial de alto nivel

Equipo disponible:
- @architect - Arquitecto de software
- @backend - Desarrollador backend
- @frontend - Desarrollador frontend
- @ux - Diseñador UX/UI
- @qa - QA Engineer
- @devops - DevOps Engineer
"""

        for response in self.send_message(prompt, AgentRole.PMO):
            yield response

    def clear_all(self):
        """Clear all conversation histories."""
        for agent in self.agents.values():
            agent.clear_history()
        self.message_history = []
        self.context = ProjectContext()
        self.active_agent = None


# Ejemplo de uso
if __name__ == "__main__":
    print("Testing TeamOrchestrator...")

    orchestrator = TeamOrchestrator()

    # Show all agents
    print("\nAgentes disponibles:")
    for agent_info in orchestrator.get_all_agents():
        print(f"  {agent_info['emoji']} {agent_info['name']} ({agent_info['role']})")

    # Test routing
    test_messages = [
        "Hola, quiero crear una app",  # Should go to PMO
        "@architect diseña la arquitectura",  # Should go to Architect
        "@backend crea el API",  # Should go to Backend
        "haz el diseño @ux",  # Should go to UX
    ]

    print("\nTest de routing:")
    for msg in test_messages:
        target = orchestrator.route_message(msg)
        print(f"  '{msg[:30]}...' -> {target.value}")

    print("\n✅ Tests completados!")
