#!/usr/bin/env python3
"""
THAU Agent System - Sistema de Agentes Especializados
Inspirado en Claude Code Agent System

Permite a THAU crear y gestionar agentes especializados para tareas complejas
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from datetime import datetime
import uuid


class AgentRole(Enum):
    """Roles de agentes especializados"""
    GENERAL = "general"                    # Tareas generales
    CODE_WRITER = "code_writer"           # Escribir código
    CODE_REVIEWER = "code_reviewer"       # Revisar código
    DEBUGGER = "debugger"                 # Depurar errores
    RESEARCHER = "researcher"             # Investigación
    PLANNER = "planner"                   # Planificación
    ARCHITECT = "architect"               # Arquitectura
    TESTER = "tester"                     # Testing
    DOCUMENTER = "documenter"             # Documentación
    API_SPECIALIST = "api_specialist"     # APIs y REST
    DATA_ANALYST = "data_analyst"         # Análisis de datos
    SECURITY = "security"                 # Seguridad
    VISUAL_CREATOR = "visual_creator"     # Creación visual (imágenes)


@dataclass
class AgentCapability:
    """Capacidad específica de un agente"""
    name: str
    description: str
    tools: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    """Configuración de un agente"""
    role: AgentRole
    name: str
    description: str
    capabilities: List[AgentCapability] = field(default_factory=list)
    max_steps: int = 100
    temperature: float = 0.7
    specialized_prompts: Dict[str, str] = field(default_factory=dict)


@dataclass
class Task:
    """Tarea asignada a un agente"""
    id: str
    description: str
    agent_role: AgentRole
    status: str = "pending"  # pending, in_progress, completed, failed
    steps: List[Dict[str, Any]] = field(default_factory=list)
    result: Optional[Any] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAction:
    """Acción ejecutada por un agente"""
    action_type: str  # "tool_use", "think", "plan", "code", "test"
    content: str
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)


class ThauAgent:
    """
    Agente Especializado de THAU

    Similar a los agentes de Claude Code, cada agente tiene:
    - Un rol específico
    - Capacidades únicas
    - Herramientas especializadas
    - Sistema de planificación
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.id = str(uuid.uuid4())[:8]
        self.current_task: Optional[Task] = None
        self.history: List[AgentAction] = []
        self.tools_available: Dict[str, Callable] = {}

        print(f"🤖 Agent creado: {config.name} ({config.role.value})")
        print(f"   ID: {self.id}")
        print(f"   Capacidades: {len(config.capabilities)}")

    def register_tool(self, name: str, tool_func: Callable):
        """Registra una herramienta disponible para este agente"""
        self.tools_available[name] = tool_func
        print(f"   🔧 Tool registrado: {name}")

    def execute_action(self, action: AgentAction) -> Any:
        """
        Ejecuta una acción

        Args:
            action: Acción a ejecutar

        Returns:
            Resultado de la acción
        """
        if action.action_type == "tool_use":
            if action.tool_name in self.tools_available:
                tool_func = self.tools_available[action.tool_name]
                result = tool_func(**action.parameters)
                action.result = result
                self.history.append(action)
                return result
            else:
                raise ValueError(f"Tool '{action.tool_name}' no disponible")

        elif action.action_type == "think":
            # Registro de pensamiento interno
            self.history.append(action)
            return {"thought": action.content}

        elif action.action_type == "plan":
            # Crear plan de pasos
            self.history.append(action)
            return {"plan": action.content}

        else:
            self.history.append(action)
            return {"action": action.action_type, "content": action.content}

    def start_task(self, task: Task):
        """Inicia una tarea"""
        self.current_task = task
        task.status = "in_progress"
        print(f"\n📋 Tarea iniciada: {task.description}")
        print(f"   Agent: {self.config.name}")

    def complete_task(self, result: Any):
        """Completa la tarea actual"""
        if self.current_task:
            self.current_task.status = "completed"
            self.current_task.result = result
            self.current_task.completed_at = datetime.now()
            print(f"\n✅ Tarea completada: {self.current_task.description}")
            self.current_task = None

    def fail_task(self, error: str):
        """Marca tarea como fallida"""
        if self.current_task:
            self.current_task.status = "failed"
            self.current_task.result = {"error": error}
            self.current_task.completed_at = datetime.now()
            print(f"\n❌ Tarea fallida: {self.current_task.description}")
            print(f"   Error: {error}")
            self.current_task = None


class AgentOrchestrator:
    """
    Orquestador de Agentes

    Gestiona múltiples agentes y coordina tareas complejas
    Similar al sistema de Claude Code
    """

    def __init__(self):
        self.agents: Dict[str, ThauAgent] = {}
        self.tasks: Dict[str, Task] = {}
        self.agent_configs = self._create_default_agents()

        print("🎭 Agent Orchestrator inicializado")

    def _create_default_agents(self) -> Dict[AgentRole, AgentConfig]:
        """Crea configuraciones de agentes por defecto"""
        configs = {}

        # General Purpose Agent
        configs[AgentRole.GENERAL] = AgentConfig(
            role=AgentRole.GENERAL,
            name="THAU General",
            description="Agente general para tareas variadas",
            capabilities=[
                AgentCapability(
                    name="conversacion",
                    description="Mantener conversaciones naturales",
                    skills=["dialogo", "comprension", "razonamiento"]
                )
            ]
        )

        # Code Writer Agent
        configs[AgentRole.CODE_WRITER] = AgentConfig(
            role=AgentRole.CODE_WRITER,
            name="THAU Code Writer",
            description="Especializado en escribir código de calidad",
            capabilities=[
                AgentCapability(
                    name="escribir_codigo",
                    description="Escribir código limpio y eficiente",
                    skills=["python", "javascript", "typescript", "java"],
                    tools=["code_editor", "formatter", "linter"]
                ),
                AgentCapability(
                    name="refactoring",
                    description="Refactorizar código existente",
                    skills=["clean_code", "design_patterns", "optimization"]
                )
            ],
            specialized_prompts={
                "system": "Eres un experto programador. Escribe código limpio, bien documentado y siguiendo mejores prácticas."
            }
        )

        # Code Reviewer Agent
        configs[AgentRole.CODE_REVIEWER] = AgentConfig(
            role=AgentRole.CODE_REVIEWER,
            name="THAU Code Reviewer",
            description="Experto en revisar y mejorar código",
            capabilities=[
                AgentCapability(
                    name="code_review",
                    description="Revisar código para calidad y bugs",
                    skills=["code_analysis", "security", "performance"],
                    tools=["linter", "security_scanner", "profiler"]
                )
            ]
        )

        # Planner Agent (¡Como yo!)
        configs[AgentRole.PLANNER] = AgentConfig(
            role=AgentRole.PLANNER,
            name="THAU Planner",
            description="Planifica tareas complejas en pasos manejables",
            capabilities=[
                AgentCapability(
                    name="task_breakdown",
                    description="Descomponer tareas complejas",
                    skills=["planning", "task_analysis", "dependency_mapping"]
                ),
                AgentCapability(
                    name="coordination",
                    description="Coordinar múltiples agentes",
                    skills=["orchestration", "scheduling", "prioritization"]
                )
            ],
            specialized_prompts={
                "system": "Eres un planificador experto. Descompones tareas complejas en pasos claros y manejables."
            }
        )

        # Researcher Agent
        configs[AgentRole.RESEARCHER] = AgentConfig(
            role=AgentRole.RESEARCHER,
            name="THAU Researcher",
            description="Investiga temas y recopila información",
            capabilities=[
                AgentCapability(
                    name="research",
                    description="Investigar temas a fondo",
                    skills=["web_search", "analysis", "synthesis"],
                    tools=["web_search", "document_reader", "summarizer"]
                )
            ]
        )

        # Visual Creator Agent
        configs[AgentRole.VISUAL_CREATOR] = AgentConfig(
            role=AgentRole.VISUAL_CREATOR,
            name="THAU Visual Creator",
            description="Crea imágenes desde la imaginación de THAU",
            capabilities=[
                AgentCapability(
                    name="image_generation",
                    description="Generar imágenes con VAE propio",
                    skills=["vae", "latent_space", "image_synthesis"],
                    tools=["thau_vae", "camera_capture", "image_editor"]
                )
            ]
        )

        # API Specialist Agent
        configs[AgentRole.API_SPECIALIST] = AgentConfig(
            role=AgentRole.API_SPECIALIST,
            name="THAU API Specialist",
            description="Especializado en APIs, REST, webhooks",
            capabilities=[
                AgentCapability(
                    name="api_integration",
                    description="Integrar APIs externas",
                    skills=["rest", "graphql", "websockets", "oauth"],
                    tools=["http_client", "api_tester", "webhook_handler"]
                ),
                AgentCapability(
                    name="api_creation",
                    description="Crear APIs propias",
                    skills=["fastapi", "flask", "authentication", "documentation"]
                )
            ]
        )

        return configs

    def create_agent(self, role: AgentRole) -> ThauAgent:
        """
        Crea un nuevo agente del rol especificado

        Args:
            role: Rol del agente

        Returns:
            Agente creado
        """
        if role not in self.agent_configs:
            raise ValueError(f"Rol de agente desconocido: {role}")

        config = self.agent_configs[role]
        agent = ThauAgent(config)
        self.agents[agent.id] = agent

        return agent

    def assign_task(self, task_description: str, role: AgentRole) -> Task:
        """
        Asigna una tarea a un agente

        Args:
            task_description: Descripción de la tarea
            role: Rol del agente que debe ejecutarla

        Returns:
            Tarea creada
        """
        # Crea agente si no existe uno del rol adecuado
        suitable_agents = [a for a in self.agents.values() if a.config.role == role]

        if not suitable_agents:
            agent = self.create_agent(role)
        else:
            agent = suitable_agents[0]

        # Crea tarea
        task = Task(
            id=str(uuid.uuid4())[:8],
            description=task_description,
            agent_role=role
        )

        self.tasks[task.id] = task
        agent.start_task(task)

        return task

    def delegate_complex_task(self, description: str) -> List[Task]:
        """
        Delega una tarea compleja a múltiples agentes

        Usa el Planner Agent para descomponer la tarea
        y luego asigna subtareas a agentes especializados

        Args:
            description: Descripción de la tarea compleja

        Returns:
            Lista de subtareas creadas
        """
        print(f"\n{'='*70}")
        print(f"🎯 Tarea Compleja Recibida")
        print(f"{'='*70}")
        print(f"Descripción: {description}")
        print(f"\n{'='*70}")
        print(f"📋 Fase 1: Planificación")
        print(f"{'='*70}")

        # Fase 1: Planner descompone tarea
        planner = self.create_agent(AgentRole.PLANNER)
        planning_task = Task(
            id=str(uuid.uuid4())[:8],
            description=f"Planificar: {description}",
            agent_role=AgentRole.PLANNER
        )

        planner.start_task(planning_task)

        # Simula planificación (en producción, el modelo THAU-2B haría esto)
        plan_action = AgentAction(
            action_type="plan",
            content=f"Plan para: {description}"
        )

        planner.execute_action(plan_action)

        # Fase 2: Crear subtareas basadas en el plan
        print(f"\n{'='*70}")
        print(f"⚙️  Fase 2: Creación de Subtareas")
        print(f"{'='*70}")

        subtasks = []

        # Ejemplo: descomponer en pasos
        # En producción, THAU-2B generaría estos pasos

        planner.complete_task({"subtasks": subtasks})

        return subtasks

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estado de una tarea"""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]
        return {
            "id": task.id,
            "description": task.description,
            "status": task.status,
            "agent_role": task.agent_role.value,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "result": task.result
        }

    def list_agents(self) -> List[Dict[str, Any]]:
        """Lista todos los agentes activos"""
        return [
            {
                "id": agent.id,
                "name": agent.config.name,
                "role": agent.config.role.value,
                "capabilities": [c.name for c in agent.config.capabilities],
                "active_task": agent.current_task.description if agent.current_task else None
            }
            for agent in self.agents.values()
        ]

    def save_state(self, filepath: str = "data/agents/orchestrator_state.json"):
        """Guarda estado del orquestador"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        state = {
            "agents": self.list_agents(),
            "tasks": {
                task_id: self.get_task_status(task_id)
                for task_id in self.tasks
            }
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"💾 Estado guardado: {filepath}")


# Global orchestrator instance
_global_orchestrator = None


def get_agent_orchestrator() -> AgentOrchestrator:
    """Obtiene instancia global del orquestador"""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = AgentOrchestrator()
    return _global_orchestrator


if __name__ == "__main__":
    print("="*70)
    print("🧪 Testing THAU Agent System")
    print("="*70)

    # Create orchestrator
    orchestrator = get_agent_orchestrator()

    print("\n" + "="*70)
    print("🤖 Agentes Disponibles")
    print("="*70)

    for role in AgentRole:
        if role in orchestrator.agent_configs:
            config = orchestrator.agent_configs[role]
            print(f"\n{config.name}")
            print(f"  Role: {role.value}")
            print(f"  Description: {config.description}")
            print(f"  Capabilities:")
            for cap in config.capabilities:
                print(f"    - {cap.name}: {cap.description}")

    print("\n" + "="*70)
    print("🧪 Test: Crear agentes y asignar tareas")
    print("="*70)

    # Test 1: Code Writer Agent
    task1 = orchestrator.assign_task(
        "Escribir una función para calcular fibonacci",
        AgentRole.CODE_WRITER
    )

    # Test 2: Visual Creator Agent
    task2 = orchestrator.assign_task(
        "Generar imagen de un robot aprendiendo",
        AgentRole.VISUAL_CREATOR
    )

    # Test 3: API Specialist Agent
    task3 = orchestrator.assign_task(
        "Integrar API de calendario de Google",
        AgentRole.API_SPECIALIST
    )

    print("\n" + "="*70)
    print("📋 Agentes Activos")
    print("="*70)

    for agent_info in orchestrator.list_agents():
        print(f"\n{agent_info['name']} (ID: {agent_info['id']})")
        print(f"  Role: {agent_info['role']}")
        print(f"  Active Task: {agent_info['active_task']}")

    print("\n" + "="*70)
    print("🎯 Test: Tarea Compleja")
    print("="*70)

    complex_task = "Crear un dashboard web con autenticación, conexión a API de calendario, y generación de gráficos"
    subtasks = orchestrator.delegate_complex_task(complex_task)

    print("\n" + "="*70)
    print("💾 Guardar Estado")
    print("="*70)

    orchestrator.save_state()

    print("\n" + "="*70)
    print("✅ Tests Completados")
    print("="*70)
