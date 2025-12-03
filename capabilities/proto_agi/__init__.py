"""
THAU Proto-AGI - Sistema de Agente Avanzado

Módulos:
- thau_proto_agi: Ciclo ReAct básico
- thau_agi: Sistema AGI integrado con aprendizaje
- thau_agi_v2: Sistema AGI completo unificado
- experiential_learning: Sistema de aprendizaje experiencial
- multi_agent: Sistema multi-agente colaborativo
- knowledge_base: Base de conocimiento con RAG

Componentes principales:
- ThauAGIv2: Sistema Proto-AGI completo (RECOMENDADO)
- ThauAGI: Agente con aprendizaje y metacognición
- ThauProtoAGI: Agente básico con ciclo ReAct
- MultiAgentSystem: Sistema de colaboración multi-agente
- KnowledgeStore: Base de conocimiento con RAG
- FeedbackSystem: Sistema de retroalimentación
"""

# Componentes básicos
from .thau_proto_agi import ThauProtoAGI, ThauTools, ToolResult, ThoughtStep, AgentState

# Sistema AGI integrado v1
from .thau_agi import ThauAGI, AGIConfig

# Sistema AGI v2 - Completo
from .thau_agi_v2 import ThauAGIv2, ThauConfig, ThauMode

# Sistema de aprendizaje experiencial
from .experiential_learning import (
    ExperienceStore,
    MetacognitiveEngine,
    AdaptiveStrategy,
    Experience,
    Pattern,
    OutcomeType,
    StrategyType,
    get_experience_store,
    get_metacognitive_engine,
    get_adaptive_strategy,
)

# Sistema multi-agente
from .multi_agent import (
    MultiAgentSystem,
    SpecializedAgent,
    AgentCoordinator,
    AgentRole,
    MessageBus,
    SharedMemory,
    Message,
    MessageType,
    TaskPriority,
)

# Knowledge Base con RAG
from .knowledge_base import (
    KnowledgeStore,
    ContextBuilder,
    KnowledgeLearner,
    FeedbackSystem,
    KnowledgeType,
    RetrievalStrategy,
    get_knowledge_store,
    get_context_builder,
    get_knowledge_learner,
    get_feedback_system,
)

__all__ = [
    # AGI v2 (Principal - Recomendado)
    "ThauAGIv2",
    "ThauConfig",
    "ThauMode",
    # AGI v1
    "ThauAGI",
    "AGIConfig",
    # Proto-AGI básico
    "ThauProtoAGI",
    "ThauTools",
    "ToolResult",
    "ThoughtStep",
    "AgentState",
    # Aprendizaje experiencial
    "ExperienceStore",
    "MetacognitiveEngine",
    "AdaptiveStrategy",
    "Experience",
    "Pattern",
    "OutcomeType",
    "StrategyType",
    "get_experience_store",
    "get_metacognitive_engine",
    "get_adaptive_strategy",
    # Multi-agente
    "MultiAgentSystem",
    "SpecializedAgent",
    "AgentCoordinator",
    "AgentRole",
    "MessageBus",
    "SharedMemory",
    "Message",
    "MessageType",
    "TaskPriority",
    # Knowledge Base
    "KnowledgeStore",
    "ContextBuilder",
    "KnowledgeLearner",
    "FeedbackSystem",
    "KnowledgeType",
    "RetrievalStrategy",
    "get_knowledge_store",
    "get_context_builder",
    "get_knowledge_learner",
    "get_feedback_system",
]

__version__ = "2.0.0"
