"""
THAU Knowledge Base - Sistema de Conocimiento con RAG

Integra memoria a largo plazo con ChromaDB para:
- Almacenamiento semántico de conocimiento
- Recuperación inteligente (RAG)
- Aprendizaje incremental
- Contexto persistente entre sesiones

Componentes:
- KnowledgeStore: Almacén de conocimiento con categorías
- SemanticRetriever: Recuperación semántica mejorada
- ContextBuilder: Construcción de contexto para prompts
- KnowledgeLearner: Aprendizaje de nuevas interacciones
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from memory.vector_store import VectorStore
    from memory.long_term import LongTermMemory
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

try:
    from memory.episodic import EpisodicMemory
    EPISODIC_AVAILABLE = True
except ImportError:
    EPISODIC_AVAILABLE = False


class KnowledgeType(Enum):
    """Tipos de conocimiento"""
    FACT = "fact"                    # Hecho verificado
    CONCEPT = "concept"              # Concepto o definición
    PROCEDURE = "procedure"          # Procedimiento o how-to
    EXPERIENCE = "experience"        # Experiencia pasada
    USER_PREFERENCE = "preference"   # Preferencia del usuario
    CONVERSATION = "conversation"    # Fragmento de conversación
    CODE = "code"                    # Código o snippet
    ERROR = "error"                  # Error y solución
    INSIGHT = "insight"              # Insight o aprendizaje


class RetrievalStrategy(Enum):
    """Estrategias de recuperación"""
    SEMANTIC = "semantic"            # Búsqueda semántica pura
    HYBRID = "hybrid"                # Semántica + keywords
    TEMPORAL = "temporal"            # Priorizar recientes
    RELEVANCE = "relevance"          # Por relevancia calculada
    DIVERSITY = "diversity"          # Resultados diversos


@dataclass
class Knowledge:
    """Unidad de conocimiento"""
    id: str
    content: str
    knowledge_type: KnowledgeType
    source: str  # Fuente del conocimiento
    confidence: float = 1.0  # Confianza en el conocimiento
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    related_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "knowledge_type": self.knowledge_type.value,
            "source": self.source,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "tags": self.tags,
            "related_ids": self.related_ids,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Knowledge':
        return cls(
            id=data["id"],
            content=data["content"],
            knowledge_type=KnowledgeType(data["knowledge_type"]),
            source=data["source"],
            confidence=data.get("confidence", 1.0),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data.get("last_accessed", data["created_at"])),
            access_count=data.get("access_count", 0),
            tags=data.get("tags", []),
            related_ids=data.get("related_ids", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class RetrievalResult:
    """Resultado de recuperación"""
    knowledge: Knowledge
    score: float  # Similaridad/relevancia
    context: str  # Contexto adicional
    retrieval_method: str


class KnowledgeStore:
    """
    Almacén de conocimiento con búsqueda semántica

    Usa ChromaDB para almacenar y recuperar conocimiento
    de forma semánticamente relevante.
    """

    def __init__(
        self,
        db_path: str = "./data/memory/knowledge_db",
        collection_name: str = "thau_knowledge"
    ):
        self.db_path = db_path
        self.collection_name = collection_name

        # Inicializar vector store si disponible
        if MEMORY_AVAILABLE:
            self.vector_store = VectorStore(
                db_path=db_path,
                collection_name=collection_name
            )
        else:
            self.vector_store = None
            print("[WARN] VectorStore no disponible, usando almacenamiento simple")

        # Caché local
        self.cache: Dict[str, Knowledge] = {}
        self.cache_max_size = 1000

        # Índices locales
        self.type_index: Dict[KnowledgeType, List[str]] = {t: [] for t in KnowledgeType}
        self.tag_index: Dict[str, List[str]] = {}

    def store(
        self,
        content: str,
        knowledge_type: KnowledgeType,
        source: str = "system",
        confidence: float = 1.0,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Almacena nuevo conocimiento

        Returns:
            ID del conocimiento almacenado
        """
        # Generar ID
        knowledge_id = hashlib.md5(
            f"{content}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        knowledge = Knowledge(
            id=knowledge_id,
            content=content,
            knowledge_type=knowledge_type,
            source=source,
            confidence=confidence,
            tags=tags or [],
            metadata=metadata or {}
        )

        # Almacenar en vector store
        if self.vector_store:
            self.vector_store.store(
                text=content,
                metadata={
                    "knowledge_id": knowledge_id,
                    "type": knowledge_type.value,
                    "source": source,
                    "confidence": confidence,
                    "tags": json.dumps(tags or []),
                    "created_at": knowledge.created_at.isoformat()
                },
                doc_id=knowledge_id
            )

        # Almacenar en caché
        self._add_to_cache(knowledge)

        # Actualizar índices
        self.type_index[knowledge_type].append(knowledge_id)
        for tag in knowledge.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            self.tag_index[tag].append(knowledge_id)

        return knowledge_id

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        knowledge_type: KnowledgeType = None,
        min_confidence: float = 0.0,
        tags: List[str] = None,
        strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC
    ) -> List[RetrievalResult]:
        """
        Recupera conocimiento relevante

        Args:
            query: Consulta de búsqueda
            n_results: Número de resultados
            knowledge_type: Filtrar por tipo
            min_confidence: Confianza mínima
            tags: Filtrar por tags
            strategy: Estrategia de recuperación

        Returns:
            Lista de RetrievalResult
        """
        results = []

        if not self.vector_store:
            # Fallback: búsqueda simple en caché
            return self._simple_search(query, n_results, knowledge_type)

        # Construir filtros
        where_filter = {}
        if knowledge_type:
            where_filter["type"] = knowledge_type.value
        if min_confidence > 0:
            where_filter["confidence"] = {"$gte": min_confidence}

        # Búsqueda semántica
        search_results = self.vector_store.search(
            query=query,
            n_results=n_results * 2,  # Buscar más para filtrar
            where=where_filter if where_filter else None
        )

        for result in search_results:
            metadata = result.get("metadata", {})
            knowledge_id = metadata.get("knowledge_id", result.get("id", ""))

            # Recuperar conocimiento completo
            knowledge = self._get_knowledge(knowledge_id, result)

            if knowledge:
                # Filtrar por tags si se especificaron
                if tags and not any(t in knowledge.tags for t in tags):
                    continue

                # Actualizar acceso
                knowledge.access_count += 1
                knowledge.last_accessed = datetime.now()

                results.append(RetrievalResult(
                    knowledge=knowledge,
                    score=result.get("score", 0.0),
                    context=f"Tipo: {knowledge.knowledge_type.value}",
                    retrieval_method=strategy.value
                ))

        # Aplicar estrategia
        results = self._apply_strategy(results, strategy, n_results)

        return results[:n_results]

    def _get_knowledge(self, knowledge_id: str, search_result: Dict) -> Optional[Knowledge]:
        """Obtiene conocimiento de caché o construye desde resultado"""
        # Intentar caché
        if knowledge_id in self.cache:
            return self.cache[knowledge_id]

        # Construir desde resultado de búsqueda
        metadata = search_result.get("metadata", {})
        content = search_result.get("text", search_result.get("document", ""))

        if not content:
            return None

        try:
            knowledge = Knowledge(
                id=knowledge_id,
                content=content,
                knowledge_type=KnowledgeType(metadata.get("type", "fact")),
                source=metadata.get("source", "unknown"),
                confidence=float(metadata.get("confidence", 1.0)),
                created_at=datetime.fromisoformat(metadata.get("created_at", datetime.now().isoformat())),
                tags=json.loads(metadata.get("tags", "[]")),
                metadata=metadata
            )

            self._add_to_cache(knowledge)
            return knowledge
        except Exception:
            return None

    def _add_to_cache(self, knowledge: Knowledge) -> None:
        """Agrega conocimiento a caché con límite de tamaño"""
        if len(self.cache) >= self.cache_max_size:
            # Remover el menos accedido
            least_accessed = min(self.cache.values(), key=lambda k: k.access_count)
            del self.cache[least_accessed.id]

        self.cache[knowledge.id] = knowledge

    def _simple_search(
        self,
        query: str,
        n_results: int,
        knowledge_type: KnowledgeType = None
    ) -> List[RetrievalResult]:
        """Búsqueda simple en caché (fallback)"""
        results = []
        query_lower = query.lower()

        for knowledge in self.cache.values():
            if knowledge_type and knowledge.knowledge_type != knowledge_type:
                continue

            # Calcular score simple por coincidencia de palabras
            words = query_lower.split()
            content_lower = knowledge.content.lower()
            matches = sum(1 for w in words if w in content_lower)
            score = matches / len(words) if words else 0

            if score > 0:
                results.append(RetrievalResult(
                    knowledge=knowledge,
                    score=score,
                    context="Simple search",
                    retrieval_method="simple"
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:n_results]

    def _apply_strategy(
        self,
        results: List[RetrievalResult],
        strategy: RetrievalStrategy,
        n_results: int
    ) -> List[RetrievalResult]:
        """Aplica estrategia de recuperación"""
        if strategy == RetrievalStrategy.SEMANTIC:
            # Ya ordenado por score semántico
            return results

        elif strategy == RetrievalStrategy.TEMPORAL:
            # Priorizar recientes
            return sorted(results, key=lambda r: r.knowledge.created_at, reverse=True)

        elif strategy == RetrievalStrategy.RELEVANCE:
            # Combinar score con confianza y accesos
            for r in results:
                r.score = (
                    r.score * 0.5 +
                    r.knowledge.confidence * 0.3 +
                    min(r.knowledge.access_count / 100, 0.2)
                )
            return sorted(results, key=lambda r: r.score, reverse=True)

        elif strategy == RetrievalStrategy.DIVERSITY:
            # Seleccionar resultados diversos por tipo
            diverse = []
            seen_types = set()
            for r in results:
                if r.knowledge.knowledge_type not in seen_types:
                    diverse.append(r)
                    seen_types.add(r.knowledge.knowledge_type)
                if len(diverse) >= n_results:
                    break
            # Rellenar con más resultados si hace falta
            for r in results:
                if r not in diverse and len(diverse) < n_results:
                    diverse.append(r)
            return diverse

        return results

    def get_by_type(self, knowledge_type: KnowledgeType, limit: int = 100) -> List[Knowledge]:
        """Obtiene conocimiento por tipo"""
        ids = self.type_index.get(knowledge_type, [])[-limit:]
        return [self.cache[id] for id in ids if id in self.cache]

    def get_by_tag(self, tag: str, limit: int = 100) -> List[Knowledge]:
        """Obtiene conocimiento por tag"""
        ids = self.tag_index.get(tag, [])[-limit:]
        return [self.cache[id] for id in ids if id in self.cache]

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del conocimiento"""
        return {
            "total_in_cache": len(self.cache),
            "total_in_store": self.vector_store.count() if self.vector_store else 0,
            "by_type": {t.value: len(ids) for t, ids in self.type_index.items()},
            "total_tags": len(self.tag_index),
            "most_accessed": sorted(
                [(k.id, k.access_count) for k in self.cache.values()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


class ContextBuilder:
    """
    Constructor de contexto para prompts

    Combina conocimiento recuperado para crear
    contexto relevante para el modelo.
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        max_context_length: int = 2000
    ):
        self.knowledge_store = knowledge_store
        self.max_context_length = max_context_length

    def build_context(
        self,
        query: str,
        include_types: List[KnowledgeType] = None,
        strategy: RetrievalStrategy = RetrievalStrategy.RELEVANCE,
        n_items: int = 5
    ) -> str:
        """
        Construye contexto para un query

        Returns:
            String de contexto para incluir en prompt
        """
        # Recuperar conocimiento relevante
        results = self.knowledge_store.retrieve(
            query=query,
            n_results=n_items,
            strategy=strategy
        )

        if not results:
            return ""

        # Construir contexto
        context_parts = ["Contexto relevante:"]

        current_length = len(context_parts[0])

        for result in results:
            knowledge = result.knowledge

            # Formatear entrada
            entry = f"\n[{knowledge.knowledge_type.value.upper()}] {knowledge.content}"

            # Verificar límite de longitud
            if current_length + len(entry) > self.max_context_length:
                break

            context_parts.append(entry)
            current_length += len(entry)

        return "\n".join(context_parts)

    def build_conversation_context(
        self,
        conversation_history: List[Dict[str, str]],
        current_query: str,
        max_history: int = 5
    ) -> str:
        """
        Construye contexto incluyendo historial de conversación

        Returns:
            Contexto con historial y conocimiento relevante
        """
        parts = []

        # Historial de conversación
        if conversation_history:
            parts.append("Historial reciente:")
            for msg in conversation_history[-max_history:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")[:200]
                parts.append(f"  {role}: {content}")

        # Conocimiento relevante
        knowledge_context = self.build_context(current_query, n_items=3)
        if knowledge_context:
            parts.append("\n" + knowledge_context)

        return "\n".join(parts)


class KnowledgeLearner:
    """
    Sistema de aprendizaje de conocimiento

    Extrae y almacena conocimiento de:
    - Conversaciones
    - Resultados de herramientas
    - Feedback del usuario
    - Experiencias exitosas
    """

    def __init__(self, knowledge_store: KnowledgeStore):
        self.knowledge_store = knowledge_store

        # Patrones para extracción
        self.fact_patterns = [
            r"es un[ao]?\s+(.+)",
            r"se define como\s+(.+)",
            r"significa\s+(.+)",
            r"consiste en\s+(.+)",
        ]

    def learn_from_conversation(
        self,
        user_message: str,
        assistant_response: str,
        was_helpful: bool = True
    ) -> List[str]:
        """
        Aprende de una conversación

        Returns:
            IDs de conocimiento aprendido
        """
        learned_ids = []

        if not was_helpful:
            return learned_ids

        # Almacenar respuesta útil como conocimiento
        if len(assistant_response) > 50:  # Respuestas sustanciales
            knowledge_id = self.knowledge_store.store(
                content=f"Q: {user_message[:100]}... A: {assistant_response[:500]}",
                knowledge_type=KnowledgeType.CONVERSATION,
                source="conversation",
                confidence=0.7,
                tags=["learned", "conversation"]
            )
            learned_ids.append(knowledge_id)

        return learned_ids

    def learn_from_experience(
        self,
        task: str,
        result: str,
        was_successful: bool,
        strategy_used: str
    ) -> str:
        """
        Aprende de una experiencia

        Returns:
            ID del conocimiento aprendido
        """
        # Determinar tipo y confianza
        if was_successful:
            knowledge_type = KnowledgeType.EXPERIENCE
            confidence = 0.8
            content = f"Exitoso: {task} -> Estrategia: {strategy_used}. Resultado: {result[:300]}"
        else:
            knowledge_type = KnowledgeType.ERROR
            confidence = 0.6
            content = f"Fallido: {task} -> Estrategia: {strategy_used}. Error: {result[:300]}"

        return self.knowledge_store.store(
            content=content,
            knowledge_type=knowledge_type,
            source="experience",
            confidence=confidence,
            tags=["learned", "experience", "success" if was_successful else "failure"]
        )

    def learn_from_feedback(
        self,
        original_response: str,
        feedback: str,
        feedback_type: str = "correction"  # correction, preference, clarification
    ) -> str:
        """
        Aprende de feedback del usuario

        Returns:
            ID del conocimiento aprendido
        """
        if feedback_type == "preference":
            knowledge_type = KnowledgeType.USER_PREFERENCE
            confidence = 0.9
        elif feedback_type == "correction":
            knowledge_type = KnowledgeType.FACT
            confidence = 0.85
        else:
            knowledge_type = KnowledgeType.INSIGHT
            confidence = 0.7

        content = f"Feedback ({feedback_type}): {feedback}. Original: {original_response[:200]}"

        return self.knowledge_store.store(
            content=content,
            knowledge_type=knowledge_type,
            source="user_feedback",
            confidence=confidence,
            tags=["learned", "feedback", feedback_type]
        )

    def extract_facts(self, text: str) -> List[str]:
        """
        Extrae hechos de un texto

        Returns:
            Lista de hechos extraídos
        """
        import re
        facts = []

        for pattern in self.fact_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) > 10:  # Hechos sustanciales
                    facts.append(match.strip())

        return facts

    def learn_facts(self, text: str, source: str = "extraction") -> List[str]:
        """
        Extrae y almacena hechos de un texto

        Returns:
            IDs de conocimiento almacenado
        """
        facts = self.extract_facts(text)
        learned_ids = []

        for fact in facts[:5]:  # Máximo 5 hechos por texto
            knowledge_id = self.knowledge_store.store(
                content=fact,
                knowledge_type=KnowledgeType.FACT,
                source=source,
                confidence=0.6,
                tags=["extracted", "fact"]
            )
            learned_ids.append(knowledge_id)

        return learned_ids


class FeedbackSystem:
    """
    Sistema de feedback del usuario

    Recopila y procesa feedback para mejorar el sistema
    """

    def __init__(
        self,
        knowledge_learner: KnowledgeLearner,
        db_path: str = "./data/feedback"
    ):
        self.learner = knowledge_learner
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.feedback_file = self.db_path / "feedback_log.jsonl"
        self.stats = {
            "total_feedback": 0,
            "positive": 0,
            "negative": 0,
            "corrections": 0
        }

        self._load_stats()

    def _load_stats(self):
        """Carga estadísticas de feedback"""
        stats_file = self.db_path / "feedback_stats.json"
        if stats_file.exists():
            try:
                self.stats = json.loads(stats_file.read_text())
            except:
                pass

    def _save_stats(self):
        """Guarda estadísticas"""
        stats_file = self.db_path / "feedback_stats.json"
        stats_file.write_text(json.dumps(self.stats, indent=2))

    def record_feedback(
        self,
        interaction_id: str,
        feedback_type: str,  # "positive", "negative", "correction", "suggestion"
        feedback_text: str = "",
        original_response: str = "",
        context: Dict[str, Any] = None
    ) -> str:
        """
        Registra feedback del usuario

        Returns:
            ID del feedback
        """
        feedback_id = hashlib.md5(
            f"{interaction_id}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        feedback_entry = {
            "id": feedback_id,
            "interaction_id": interaction_id,
            "type": feedback_type,
            "text": feedback_text,
            "original_response": original_response[:500],
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }

        # Guardar en archivo
        with open(self.feedback_file, "a") as f:
            f.write(json.dumps(feedback_entry) + "\n")

        # Actualizar estadísticas
        self.stats["total_feedback"] += 1
        if feedback_type == "positive":
            self.stats["positive"] += 1
        elif feedback_type == "negative":
            self.stats["negative"] += 1
        elif feedback_type == "correction":
            self.stats["corrections"] += 1

        self._save_stats()

        # Aprender del feedback
        if feedback_text and feedback_type in ["correction", "suggestion"]:
            self.learner.learn_from_feedback(
                original_response,
                feedback_text,
                feedback_type
            )

        return feedback_id

    def thumbs_up(self, interaction_id: str, response: str = "") -> str:
        """Registra feedback positivo"""
        return self.record_feedback(interaction_id, "positive", original_response=response)

    def thumbs_down(self, interaction_id: str, response: str = "", reason: str = "") -> str:
        """Registra feedback negativo"""
        return self.record_feedback(
            interaction_id, "negative",
            feedback_text=reason,
            original_response=response
        )

    def correct(self, interaction_id: str, correction: str, original: str = "") -> str:
        """Registra corrección"""
        return self.record_feedback(
            interaction_id, "correction",
            feedback_text=correction,
            original_response=original
        )

    def suggest(self, interaction_id: str, suggestion: str) -> str:
        """Registra sugerencia"""
        return self.record_feedback(interaction_id, "suggestion", feedback_text=suggestion)

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de feedback"""
        satisfaction_rate = 0
        if self.stats["total_feedback"] > 0:
            satisfaction_rate = self.stats["positive"] / self.stats["total_feedback"]

        return {
            **self.stats,
            "satisfaction_rate": satisfaction_rate
        }

    def get_recent_feedback(self, limit: int = 20) -> List[Dict]:
        """Obtiene feedback reciente"""
        if not self.feedback_file.exists():
            return []

        entries = []
        with open(self.feedback_file) as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except:
                    pass

        return entries[-limit:]


# Singleton instances
_knowledge_store: Optional[KnowledgeStore] = None
_context_builder: Optional[ContextBuilder] = None
_knowledge_learner: Optional[KnowledgeLearner] = None
_feedback_system: Optional[FeedbackSystem] = None


def get_knowledge_store() -> KnowledgeStore:
    """Obtiene instancia singleton de KnowledgeStore"""
    global _knowledge_store
    if _knowledge_store is None:
        _knowledge_store = KnowledgeStore()
    return _knowledge_store


def get_context_builder() -> ContextBuilder:
    """Obtiene instancia singleton de ContextBuilder"""
    global _context_builder
    if _context_builder is None:
        _context_builder = ContextBuilder(get_knowledge_store())
    return _context_builder


def get_knowledge_learner() -> KnowledgeLearner:
    """Obtiene instancia singleton de KnowledgeLearner"""
    global _knowledge_learner
    if _knowledge_learner is None:
        _knowledge_learner = KnowledgeLearner(get_knowledge_store())
    return _knowledge_learner


def get_feedback_system() -> FeedbackSystem:
    """Obtiene instancia singleton de FeedbackSystem"""
    global _feedback_system
    if _feedback_system is None:
        _feedback_system = FeedbackSystem(get_knowledge_learner())
    return _feedback_system


if __name__ == "__main__":
    print("=" * 60)
    print("  THAU Knowledge Base - Demo")
    print("=" * 60)

    # Inicializar componentes
    store = get_knowledge_store()
    builder = get_context_builder()
    learner = get_knowledge_learner()
    feedback = get_feedback_system()

    # Test 1: Almacenar conocimiento
    print("\n[1] Almacenando conocimiento...")

    store.store(
        "Python es un lenguaje de programación interpretado y de alto nivel.",
        KnowledgeType.FACT,
        source="manual",
        tags=["python", "programming"]
    )

    store.store(
        "Para crear una lista en Python usa corchetes: lista = [1, 2, 3]",
        KnowledgeType.PROCEDURE,
        source="manual",
        tags=["python", "lists", "howto"]
    )

    store.store(
        "El usuario prefiere respuestas concisas y con ejemplos de código.",
        KnowledgeType.USER_PREFERENCE,
        source="feedback",
        confidence=0.9,
        tags=["preference", "style"]
    )

    print(f"   Conocimiento almacenado: {store.get_stats()['total_in_cache']}")

    # Test 2: Recuperar conocimiento
    print("\n[2] Recuperando conocimiento...")

    results = store.retrieve("cómo crear una lista en Python", n_results=3)
    for r in results:
        print(f"   [{r.knowledge.knowledge_type.value}] Score: {r.score:.2f}")
        print(f"   {r.knowledge.content[:80]}...")

    # Test 3: Construir contexto
    print("\n[3] Construyendo contexto para prompt...")

    context = builder.build_context("explica Python")
    print(f"   Contexto ({len(context)} chars):")
    print(f"   {context[:200]}...")

    # Test 4: Aprender de conversación
    print("\n[4] Aprendiendo de conversación...")

    learned = learner.learn_from_conversation(
        "¿Qué es una función lambda?",
        "Una función lambda es una función anónima en Python que se define con la palabra clave lambda.",
        was_helpful=True
    )
    print(f"   Aprendido: {len(learned)} items")

    # Test 5: Sistema de feedback
    print("\n[5] Sistema de feedback...")

    feedback.thumbs_up("interaction_001", "Respuesta sobre Python")
    feedback.correct("interaction_002", "El resultado correcto es 42", "El resultado es 41")

    stats = feedback.get_stats()
    print(f"   Total feedback: {stats['total_feedback']}")
    print(f"   Satisfacción: {stats['satisfaction_rate']:.0%}")

    # Estadísticas finales
    print("\n[6] Estadísticas del sistema:")
    print(f"   Conocimiento total: {store.get_stats()['total_in_cache']}")
    print(f"   Por tipo: {store.get_stats()['by_type']}")

    print("\n" + "=" * 60)
    print("  Demo completada!")
    print("=" * 60)
