"""
THAU Multi-Agent System - Comunicación y Colaboración Inter-Agente

Sistema que permite a múltiples agentes especializados colaborar:
- Comunicación asíncrona entre agentes via mensajes
- Delegación inteligente de tareas
- Memoria compartida
- Consenso y resolución de conflictos
- Supervisión y coordinación

Arquitectura:
- MessageBus: Canal de comunicación entre agentes
- SharedMemory: Memoria compartida para contexto
- AgentCoordinator: Orquestador de colaboración
- SpecializedAgent: Agente con rol específico
"""

import json
import time
import uuid
import queue
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class MessageType(Enum):
    """Tipos de mensajes entre agentes"""
    REQUEST = "request"           # Solicitud de tarea
    RESPONSE = "response"         # Respuesta a solicitud
    BROADCAST = "broadcast"       # Mensaje a todos
    DELEGATE = "delegate"         # Delegar tarea
    STATUS = "status"             # Actualización de estado
    QUERY = "query"               # Consulta de información
    RESULT = "result"             # Resultado de tarea
    ERROR = "error"               # Error reportado
    CONSENSUS = "consensus"       # Solicitud de consenso
    VOTE = "vote"                 # Voto en consenso


class AgentRole(Enum):
    """Roles de agentes especializados"""
    COORDINATOR = "coordinator"     # Coordina a otros agentes
    CODER = "coder"                 # Escribe código
    REVIEWER = "reviewer"           # Revisa código/texto
    RESEARCHER = "researcher"       # Investiga información
    PLANNER = "planner"             # Planifica tareas
    TESTER = "tester"               # Prueba y valida
    DOCUMENTER = "documenter"       # Documenta
    ANALYST = "analyst"             # Analiza datos
    DEBUGGER = "debugger"           # Depura problemas
    ARCHITECT = "architect"         # Diseña arquitectura
    SECURITY = "security"           # Analiza seguridad
    OPTIMIZER = "optimizer"         # Optimiza rendimiento
    GENERALIST = "generalist"       # Tareas generales


class TaskPriority(Enum):
    """Prioridad de tareas"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class Message:
    """Mensaje entre agentes"""
    id: str
    type: MessageType
    sender: str
    recipient: str  # "*" para broadcast
    content: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    reply_to: Optional[str] = None  # ID del mensaje al que responde
    requires_response: bool = False
    ttl: int = 300  # Tiempo de vida en segundos

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "reply_to": self.reply_to,
            "requires_response": self.requires_response
        }


@dataclass
class AgentState:
    """Estado de un agente"""
    id: str
    role: AgentRole
    status: str = "idle"  # idle, busy, waiting, error
    current_task: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    load: float = 0.0  # 0.0 - 1.0
    last_active: datetime = field(default_factory=datetime.now)
    messages_processed: int = 0
    tasks_completed: int = 0


@dataclass
class SharedContext:
    """Contexto compartido entre agentes"""
    key: str
    value: Any
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)


class MessageBus:
    """
    Bus de mensajes para comunicación entre agentes

    Características:
    - Cola de mensajes por prioridad
    - Subscripciones por tipo de mensaje
    - Broadcast a todos los agentes
    - Historial de mensajes
    """

    def __init__(self, max_history: int = 1000):
        self.queues: Dict[str, queue.PriorityQueue] = {}  # agent_id -> queue
        self.subscriptions: Dict[MessageType, Set[str]] = defaultdict(set)
        self.history: List[Message] = []
        self.max_history = max_history
        self.lock = threading.Lock()

    def register_agent(self, agent_id: str) -> None:
        """Registra un agente en el bus"""
        with self.lock:
            if agent_id not in self.queues:
                self.queues[agent_id] = queue.PriorityQueue()

    def unregister_agent(self, agent_id: str) -> None:
        """Desregistra un agente"""
        with self.lock:
            if agent_id in self.queues:
                del self.queues[agent_id]
            # Remover de subscripciones
            for subscribers in self.subscriptions.values():
                subscribers.discard(agent_id)

    def subscribe(self, agent_id: str, message_type: MessageType) -> None:
        """Suscribe agente a tipo de mensaje"""
        self.subscriptions[message_type].add(agent_id)

    def unsubscribe(self, agent_id: str, message_type: MessageType) -> None:
        """Desuscribe agente de tipo de mensaje"""
        self.subscriptions[message_type].discard(agent_id)

    def send(self, message: Message) -> bool:
        """
        Envía un mensaje

        Returns:
            True si el mensaje fue enviado exitosamente
        """
        with self.lock:
            # Guardar en historial
            self.history.append(message)
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]

            # Determinar destinatarios
            if message.recipient == "*":
                # Broadcast
                recipients = list(self.queues.keys())
                recipients = [r for r in recipients if r != message.sender]
            else:
                recipients = [message.recipient]

            # Enviar a cada destinatario
            for recipient_id in recipients:
                if recipient_id in self.queues:
                    # Priority queue usa (priority, timestamp, message)
                    self.queues[recipient_id].put((
                        message.priority.value,
                        message.timestamp.timestamp(),
                        message
                    ))

            return True

    def receive(self, agent_id: str, timeout: float = 0.1) -> Optional[Message]:
        """
        Recibe mensaje para un agente

        Args:
            agent_id: ID del agente
            timeout: Tiempo máximo de espera

        Returns:
            Mensaje o None si no hay mensajes
        """
        if agent_id not in self.queues:
            return None

        try:
            _, _, message = self.queues[agent_id].get(timeout=timeout)
            return message
        except queue.Empty:
            return None

    def get_pending_count(self, agent_id: str) -> int:
        """Obtiene número de mensajes pendientes"""
        if agent_id not in self.queues:
            return 0
        return self.queues[agent_id].qsize()

    def get_history(self, limit: int = 100, sender: str = None, recipient: str = None) -> List[Message]:
        """Obtiene historial de mensajes"""
        messages = self.history[-limit:]

        if sender:
            messages = [m for m in messages if m.sender == sender]
        if recipient:
            messages = [m for m in messages if m.recipient == recipient or m.recipient == "*"]

        return messages


class SharedMemory:
    """
    Memoria compartida entre agentes

    Almacena contexto, resultados y conocimiento compartido
    """

    def __init__(self):
        self.store: Dict[str, SharedContext] = {}
        self.tags_index: Dict[str, Set[str]] = defaultdict(set)
        self.lock = threading.Lock()

    def set(
        self,
        key: str,
        value: Any,
        created_by: str,
        tags: List[str] = None
    ) -> None:
        """Almacena valor en memoria compartida"""
        with self.lock:
            context = SharedContext(
                key=key,
                value=value,
                created_by=created_by,
                tags=tags or []
            )

            # Actualizar si existe
            if key in self.store:
                context.created_at = self.store[key].created_at
                context.access_count = self.store[key].access_count

            self.store[key] = context

            # Actualizar índice de tags
            for tag in context.tags:
                self.tags_index[tag].add(key)

    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor de memoria compartida"""
        with self.lock:
            if key in self.store:
                self.store[key].access_count += 1
                return self.store[key].value
            return None

    def get_by_tag(self, tag: str) -> Dict[str, Any]:
        """Obtiene todos los valores con un tag"""
        with self.lock:
            keys = self.tags_index.get(tag, set())
            return {k: self.store[k].value for k in keys if k in self.store}

    def get_context(self, key: str) -> Optional[SharedContext]:
        """Obtiene contexto completo"""
        return self.store.get(key)

    def delete(self, key: str) -> bool:
        """Elimina entrada"""
        with self.lock:
            if key in self.store:
                # Remover de índice de tags
                for tag in self.store[key].tags:
                    self.tags_index[tag].discard(key)
                del self.store[key]
                return True
            return False

    def list_keys(self, pattern: str = None) -> List[str]:
        """Lista claves, opcionalmente filtradas por patrón"""
        keys = list(self.store.keys())
        if pattern:
            import fnmatch
            keys = fnmatch.filter(keys, pattern)
        return keys

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la memoria"""
        return {
            "total_entries": len(self.store),
            "total_tags": len(self.tags_index),
            "most_accessed": sorted(
                [(k, v.access_count) for k, v in self.store.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


class SpecializedAgent:
    """
    Agente Especializado con capacidad de comunicación

    Cada agente tiene:
    - Un rol específico
    - Capacidades definidas
    - Conexión al MessageBus
    - Acceso a SharedMemory
    """

    def __init__(
        self,
        role: AgentRole,
        message_bus: MessageBus,
        shared_memory: SharedMemory,
        name: str = None
    ):
        self.id = f"{role.value}_{uuid.uuid4().hex[:6]}"
        self.role = role
        self.name = name or f"Agent-{role.value.title()}"
        self.message_bus = message_bus
        self.shared_memory = shared_memory

        # Estado
        self.state = AgentState(
            id=self.id,
            role=role,
            capabilities=self._get_default_capabilities()
        )

        # Handlers de mensajes
        self.message_handlers: Dict[MessageType, Callable] = {}

        # Registrar en bus
        self.message_bus.register_agent(self.id)

        # Suscribir a broadcasts
        self.message_bus.subscribe(self.id, MessageType.BROADCAST)

    def _get_default_capabilities(self) -> List[str]:
        """Obtiene capacidades por defecto según rol"""
        capabilities_map = {
            AgentRole.COORDINATOR: ["planning", "delegation", "monitoring", "consensus"],
            AgentRole.CODER: ["code_writing", "refactoring", "debugging", "testing"],
            AgentRole.REVIEWER: ["code_review", "quality_check", "suggestions"],
            AgentRole.RESEARCHER: ["web_search", "analysis", "summarization"],
            AgentRole.PLANNER: ["task_breakdown", "scheduling", "dependency_analysis"],
            AgentRole.TESTER: ["unit_testing", "integration_testing", "validation"],
            AgentRole.DOCUMENTER: ["documentation", "comments", "readme"],
            AgentRole.ANALYST: ["data_analysis", "visualization", "statistics"],
            AgentRole.DEBUGGER: ["error_analysis", "stack_trace", "fix_suggestion"],
            AgentRole.ARCHITECT: ["design", "patterns", "architecture"],
            AgentRole.SECURITY: ["vulnerability_scan", "security_review", "hardening"],
            AgentRole.OPTIMIZER: ["performance", "profiling", "optimization"],
            AgentRole.GENERALIST: ["general_tasks", "assistance"],
        }
        return capabilities_map.get(self.role, [])

    def register_handler(self, message_type: MessageType, handler: Callable) -> None:
        """Registra handler para tipo de mensaje"""
        self.message_handlers[message_type] = handler
        self.message_bus.subscribe(self.id, message_type)

    def send_message(
        self,
        recipient: str,
        message_type: MessageType,
        content: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        requires_response: bool = False
    ) -> str:
        """
        Envía mensaje a otro agente

        Returns:
            ID del mensaje enviado
        """
        message = Message(
            id=uuid.uuid4().hex[:12],
            type=message_type,
            sender=self.id,
            recipient=recipient,
            content=content,
            priority=priority,
            requires_response=requires_response
        )

        self.message_bus.send(message)
        return message.id

    def broadcast(
        self,
        message_type: MessageType,
        content: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """Envía mensaje a todos los agentes"""
        return self.send_message("*", message_type, content, priority)

    def reply(self, original_message: Message, content: Dict[str, Any]) -> str:
        """Responde a un mensaje"""
        message = Message(
            id=uuid.uuid4().hex[:12],
            type=MessageType.RESPONSE,
            sender=self.id,
            recipient=original_message.sender,
            content=content,
            reply_to=original_message.id
        )

        self.message_bus.send(message)
        return message.id

    def process_messages(self, max_messages: int = 10) -> int:
        """
        Procesa mensajes pendientes

        Returns:
            Número de mensajes procesados
        """
        processed = 0

        for _ in range(max_messages):
            message = self.message_bus.receive(self.id)
            if not message:
                break

            # Ejecutar handler si existe
            if message.type in self.message_handlers:
                try:
                    self.message_handlers[message.type](message)
                except Exception as e:
                    # Enviar mensaje de error
                    self.send_message(
                        message.sender,
                        MessageType.ERROR,
                        {"error": str(e), "original_message": message.id}
                    )

            self.state.messages_processed += 1
            processed += 1

        self.state.last_active = datetime.now()
        return processed

    def delegate_task(
        self,
        target_role: AgentRole,
        task_description: str,
        context: Dict[str, Any] = None
    ) -> str:
        """
        Delega tarea a agente de rol específico

        Returns:
            ID del mensaje de delegación
        """
        content = {
            "task": task_description,
            "context": context or {},
            "delegated_by": self.id,
            "target_role": target_role.value
        }

        # Broadcast para que agentes del rol adecuado respondan
        return self.broadcast(
            MessageType.DELEGATE,
            content,
            priority=TaskPriority.HIGH
        )

    def share_result(self, key: str, result: Any, tags: List[str] = None) -> None:
        """Comparte resultado en memoria compartida"""
        self.shared_memory.set(key, result, self.id, tags)

    def get_shared(self, key: str) -> Optional[Any]:
        """Obtiene valor de memoria compartida"""
        return self.shared_memory.get(key)

    def update_status(self, status: str, current_task: str = None) -> None:
        """Actualiza estado del agente"""
        self.state.status = status
        self.state.current_task = current_task
        self.state.last_active = datetime.now()

        # Notificar cambio de estado
        self.broadcast(
            MessageType.STATUS,
            {
                "agent_id": self.id,
                "status": status,
                "current_task": current_task
            },
            priority=TaskPriority.LOW
        )


class AgentCoordinator(SpecializedAgent):
    """
    Coordinador de Agentes

    Agente especial que:
    - Gestiona otros agentes
    - Asigna tareas según capacidades
    - Resuelve conflictos
    - Facilita consenso
    - Monitorea progreso
    """

    def __init__(self, message_bus: MessageBus, shared_memory: SharedMemory):
        super().__init__(AgentRole.COORDINATOR, message_bus, shared_memory, "Coordinator")

        self.managed_agents: Dict[str, AgentState] = {}
        self.pending_tasks: List[Dict] = []
        self.consensus_votes: Dict[str, Dict[str, Any]] = {}

        # Registrar handlers
        self.register_handler(MessageType.STATUS, self._handle_status)
        self.register_handler(MessageType.RESULT, self._handle_result)
        self.register_handler(MessageType.VOTE, self._handle_vote)
        self.register_handler(MessageType.ERROR, self._handle_error)

    def register_managed_agent(self, agent: SpecializedAgent) -> None:
        """Registra agente bajo coordinación"""
        self.managed_agents[agent.id] = agent.state

    def _handle_status(self, message: Message) -> None:
        """Maneja actualizaciones de estado"""
        agent_id = message.content.get("agent_id")
        if agent_id in self.managed_agents:
            self.managed_agents[agent_id].status = message.content.get("status", "unknown")
            self.managed_agents[agent_id].current_task = message.content.get("current_task")

    def _handle_result(self, message: Message) -> None:
        """Maneja resultados de tareas"""
        result = message.content
        task_id = result.get("task_id")

        # Almacenar resultado
        self.share_result(
            f"result_{task_id}",
            result,
            tags=["result", message.sender]
        )

        # Actualizar estado del agente
        if message.sender in self.managed_agents:
            self.managed_agents[message.sender].tasks_completed += 1
            self.managed_agents[message.sender].status = "idle"

    def _handle_vote(self, message: Message) -> None:
        """Maneja votos de consenso"""
        consensus_id = message.content.get("consensus_id")
        vote = message.content.get("vote")

        if consensus_id not in self.consensus_votes:
            self.consensus_votes[consensus_id] = {"votes": {}, "topic": ""}

        self.consensus_votes[consensus_id]["votes"][message.sender] = vote

    def _handle_error(self, message: Message) -> None:
        """Maneja errores reportados"""
        error = message.content.get("error")
        original = message.content.get("original_message")

        # Log del error
        self.share_result(
            f"error_{message.sender}_{datetime.now().timestamp()}",
            {"error": error, "original_message": original, "agent": message.sender},
            tags=["error", message.sender]
        )

    def assign_task(
        self,
        task_description: str,
        required_capabilities: List[str] = None,
        preferred_role: AgentRole = None,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> Optional[str]:
        """
        Asigna tarea al agente más adecuado

        Returns:
            ID del agente asignado o None si no hay disponibles
        """
        # Encontrar agente adecuado
        best_agent = None
        best_score = -1

        for agent_id, state in self.managed_agents.items():
            if state.status != "idle":
                continue

            score = 0

            # Puntuación por rol
            if preferred_role and state.role == preferred_role:
                score += 10

            # Puntuación por capacidades
            if required_capabilities:
                matching = set(state.capabilities) & set(required_capabilities)
                score += len(matching) * 2

            # Penalización por carga
            score -= state.load * 5

            if score > best_score:
                best_score = score
                best_agent = agent_id

        if best_agent:
            # Enviar tarea
            task_id = uuid.uuid4().hex[:8]
            self.send_message(
                best_agent,
                MessageType.REQUEST,
                {
                    "task_id": task_id,
                    "description": task_description,
                    "capabilities_required": required_capabilities or []
                },
                priority=priority,
                requires_response=True
            )

            # Actualizar estado
            self.managed_agents[best_agent].status = "busy"
            self.managed_agents[best_agent].current_task = task_id

            return best_agent

        return None

    def request_consensus(
        self,
        topic: str,
        options: List[str],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Solicita consenso de todos los agentes

        Returns:
            Resultado del consenso
        """
        consensus_id = uuid.uuid4().hex[:8]

        self.consensus_votes[consensus_id] = {
            "topic": topic,
            "options": options,
            "votes": {},
            "started_at": datetime.now()
        }

        # Broadcast solicitud de voto
        self.broadcast(
            MessageType.CONSENSUS,
            {
                "consensus_id": consensus_id,
                "topic": topic,
                "options": options
            },
            priority=TaskPriority.HIGH
        )

        # Esperar votos (simplificado - en producción sería asíncrono)
        start = time.time()
        while time.time() - start < timeout:
            if len(self.consensus_votes[consensus_id]["votes"]) >= len(self.managed_agents):
                break
            time.sleep(0.1)

        # Contar votos
        votes = self.consensus_votes[consensus_id]["votes"]
        vote_counts = defaultdict(int)
        for vote in votes.values():
            vote_counts[vote] += 1

        winner = max(vote_counts.items(), key=lambda x: x[1]) if vote_counts else (None, 0)

        return {
            "consensus_id": consensus_id,
            "topic": topic,
            "winner": winner[0],
            "votes": dict(vote_counts),
            "total_votes": len(votes),
            "total_agents": len(self.managed_agents)
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene estado del sistema multi-agente"""
        return {
            "coordinator_id": self.id,
            "total_agents": len(self.managed_agents),
            "agents": {
                agent_id: {
                    "role": state.role.value,
                    "status": state.status,
                    "current_task": state.current_task,
                    "tasks_completed": state.tasks_completed,
                    "load": state.load
                }
                for agent_id, state in self.managed_agents.items()
            },
            "pending_messages": sum(
                self.message_bus.get_pending_count(aid)
                for aid in self.managed_agents.keys()
            ),
            "shared_memory_stats": self.shared_memory.get_stats()
        }


class MultiAgentSystem:
    """
    Sistema Multi-Agente Completo

    Facade para crear y gestionar un sistema de agentes colaborativos
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.message_bus = MessageBus()
        self.shared_memory = SharedMemory()
        self.coordinator: Optional[AgentCoordinator] = None
        self.agents: Dict[str, SpecializedAgent] = {}

        if self.verbose:
            print("\n" + "=" * 60)
            print("  THAU Multi-Agent System")
            print("=" * 60)

    def initialize(self, agent_roles: List[AgentRole] = None) -> None:
        """
        Inicializa el sistema con agentes

        Args:
            agent_roles: Roles de agentes a crear. Si es None, crea set por defecto.
        """
        # Crear coordinador
        self.coordinator = AgentCoordinator(self.message_bus, self.shared_memory)

        if self.verbose:
            print(f"\n[COORDINATOR] {self.coordinator.name} creado")

        # Roles por defecto
        if agent_roles is None:
            agent_roles = [
                AgentRole.CODER,
                AgentRole.REVIEWER,
                AgentRole.RESEARCHER,
                AgentRole.PLANNER,
                AgentRole.TESTER,
            ]

        # Crear agentes
        for role in agent_roles:
            agent = self.create_agent(role)
            if self.verbose:
                print(f"[AGENT] {agent.name} ({role.value}) creado")

        if self.verbose:
            print(f"\n[SYSTEM] {len(self.agents)} agentes activos")

    def create_agent(self, role: AgentRole, name: str = None) -> SpecializedAgent:
        """Crea y registra un nuevo agente"""
        agent = SpecializedAgent(role, self.message_bus, self.shared_memory, name)
        self.agents[agent.id] = agent

        if self.coordinator:
            self.coordinator.register_managed_agent(agent)

        return agent

    def assign_task(
        self,
        task: str,
        role: AgentRole = None,
        capabilities: List[str] = None
    ) -> Optional[str]:
        """
        Asigna tarea al sistema

        Returns:
            ID del agente asignado
        """
        if not self.coordinator:
            raise RuntimeError("Sistema no inicializado")

        return self.coordinator.assign_task(task, capabilities, role)

    def collaborate(
        self,
        task: str,
        roles_needed: List[AgentRole]
    ) -> Dict[str, Any]:
        """
        Ejecuta tarea colaborativa con múltiples agentes

        Args:
            task: Descripción de la tarea
            roles_needed: Roles que deben colaborar

        Returns:
            Resultados de la colaboración
        """
        if self.verbose:
            print(f"\n[COLLABORATE] Tarea: {task}")
            print(f"[COLLABORATE] Roles: {[r.value for r in roles_needed]}")

        results = {
            "task": task,
            "roles": [r.value for r in roles_needed],
            "contributions": {},
            "status": "in_progress"
        }

        # Almacenar tarea en memoria compartida
        task_id = uuid.uuid4().hex[:8]
        self.shared_memory.set(
            f"task_{task_id}",
            {"description": task, "status": "in_progress"},
            self.coordinator.id,
            tags=["task", "active"]
        )

        # Asignar a cada rol
        for role in roles_needed:
            # Buscar agente del rol
            agent = next(
                (a for a in self.agents.values() if a.role == role),
                None
            )

            if agent:
                # Asignar subtarea
                agent.update_status("busy", task_id)
                results["contributions"][agent.id] = {
                    "role": role.value,
                    "status": "assigned"
                }

                if self.verbose:
                    print(f"[ASSIGN] {agent.name} -> {role.value}")

        results["status"] = "assigned"
        return results

    def process_all(self, iterations: int = 1) -> Dict[str, int]:
        """
        Procesa mensajes de todos los agentes

        Returns:
            Número de mensajes procesados por agente
        """
        processed = {}

        for _ in range(iterations):
            for agent_id, agent in self.agents.items():
                count = agent.process_messages()
                processed[agent_id] = processed.get(agent_id, 0) + count

            if self.coordinator:
                count = self.coordinator.process_messages()
                processed[self.coordinator.id] = processed.get(self.coordinator.id, 0) + count

        return processed

    def get_status(self) -> Dict[str, Any]:
        """Obtiene estado del sistema"""
        if self.coordinator:
            return self.coordinator.get_system_status()
        return {"error": "Sistema no inicializado"}

    def share_knowledge(self, key: str, value: Any, tags: List[str] = None) -> None:
        """Comparte conocimiento en memoria compartida"""
        self.shared_memory.set(key, value, "system", tags)

    def get_knowledge(self, key: str) -> Optional[Any]:
        """Obtiene conocimiento compartido"""
        return self.shared_memory.get(key)

    def request_consensus(self, topic: str, options: List[str]) -> Dict[str, Any]:
        """Solicita consenso del sistema"""
        if not self.coordinator:
            raise RuntimeError("Sistema no inicializado")
        return self.coordinator.request_consensus(topic, options)


# Singleton global
_multi_agent_system: Optional[MultiAgentSystem] = None


def get_multi_agent_system() -> MultiAgentSystem:
    """Obtiene instancia singleton del sistema multi-agente"""
    global _multi_agent_system
    if _multi_agent_system is None:
        _multi_agent_system = MultiAgentSystem()
    return _multi_agent_system


if __name__ == "__main__":
    print("=" * 70)
    print("  THAU Multi-Agent System - Demo")
    print("=" * 70)

    # Crear sistema
    system = MultiAgentSystem(verbose=True)

    # Inicializar con agentes
    system.initialize([
        AgentRole.CODER,
        AgentRole.REVIEWER,
        AgentRole.RESEARCHER,
        AgentRole.PLANNER,
        AgentRole.TESTER,
        AgentRole.DOCUMENTER
    ])

    # Test 1: Asignar tarea simple
    print("\n" + "-" * 60)
    print("[TEST 1] Asignación de tarea simple")
    print("-" * 60)

    assigned = system.assign_task(
        "Escribir función para calcular fibonacci",
        role=AgentRole.CODER
    )
    print(f"Tarea asignada a: {assigned}")

    # Test 2: Colaboración
    print("\n" + "-" * 60)
    print("[TEST 2] Tarea colaborativa")
    print("-" * 60)

    collab_result = system.collaborate(
        "Crear API REST con autenticación",
        roles_needed=[AgentRole.PLANNER, AgentRole.CODER, AgentRole.REVIEWER, AgentRole.TESTER]
    )
    print(f"Colaboración: {collab_result['status']}")
    for agent_id, contrib in collab_result["contributions"].items():
        print(f"  - {agent_id}: {contrib['role']}")

    # Test 3: Memoria compartida
    print("\n" + "-" * 60)
    print("[TEST 3] Memoria compartida")
    print("-" * 60)

    system.share_knowledge("api_spec", {"version": "1.0", "endpoints": ["/users", "/auth"]}, ["api", "spec"])
    retrieved = system.get_knowledge("api_spec")
    print(f"Conocimiento compartido: {retrieved}")

    # Test 4: Estado del sistema
    print("\n" + "-" * 60)
    print("[TEST 4] Estado del sistema")
    print("-" * 60)

    status = system.get_status()
    print(f"Total agentes: {status['total_agents']}")
    print(f"Mensajes pendientes: {status['pending_messages']}")
    print("Agentes:")
    for agent_id, agent_status in status["agents"].items():
        print(f"  - {agent_id}: {agent_status['role']} ({agent_status['status']})")

    # Test 5: Procesar mensajes
    print("\n" + "-" * 60)
    print("[TEST 5] Procesamiento de mensajes")
    print("-" * 60)

    processed = system.process_all(iterations=3)
    print(f"Mensajes procesados: {sum(processed.values())}")

    print("\n" + "=" * 70)
    print("  Demo completada!")
    print("=" * 70)
