"""
THAU Experiential Learning System

Sistema de aprendizaje basado en experiencias que permite a THAU:
1. Recordar interacciones pasadas (éxitos y fracasos)
2. Aprender patrones de resolución efectivos
3. Adaptar estrategias basadas en experiencias previas
4. Metacognición profunda para auto-mejora

Este módulo es central para el comportamiento proto-AGI de THAU.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
from collections import defaultdict
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class OutcomeType(Enum):
    """Tipos de resultados de una interacción"""
    SUCCESS = "success"           # Meta lograda completamente
    PARTIAL_SUCCESS = "partial"   # Meta parcialmente lograda
    FAILURE = "failure"           # Meta no lograda
    ERROR = "error"               # Error durante ejecución
    TIMEOUT = "timeout"           # Tiempo excedido
    USER_ABORT = "user_abort"     # Usuario canceló


class StrategyType(Enum):
    """Tipos de estrategias de resolución"""
    DIRECT = "direct"                 # Respuesta directa sin herramientas
    TOOL_SINGLE = "tool_single"       # Una herramienta
    TOOL_CHAIN = "tool_chain"         # Cadena de herramientas
    DECOMPOSITION = "decomposition"   # Descomposición en subproblemas
    SEARCH_FIRST = "search_first"     # Buscar información primero
    ITERATIVE = "iterative"           # Refinamiento iterativo


@dataclass
class Experience:
    """Representa una experiencia de interacción"""
    id: str
    timestamp: datetime

    # Contexto de la interacción
    goal: str
    goal_type: str  # "calculation", "code", "file", "question", etc.
    context: Dict[str, Any]

    # Estrategia usada
    strategy: StrategyType
    tools_used: List[str]
    steps_taken: int

    # Resultado
    outcome: OutcomeType
    confidence: float  # 0.0 - 1.0
    execution_time: float  # segundos

    # Aprendizaje
    what_worked: List[str] = field(default_factory=list)
    what_failed: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)

    # Metadatos
    user_feedback: Optional[str] = None
    iterations: int = 1

    def to_dict(self) -> Dict:
        """Convierte a diccionario serializable"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['strategy'] = self.strategy.value
        data['outcome'] = self.outcome.value
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'Experience':
        """Crea desde diccionario"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['strategy'] = StrategyType(data['strategy'])
        data['outcome'] = OutcomeType(data['outcome'])
        return cls(**data)


@dataclass
class Pattern:
    """Patrón de resolución aprendido"""
    id: str
    pattern_type: str  # "goal", "error", "optimization"

    # Condiciones para aplicar
    trigger_keywords: List[str]
    goal_type: str

    # Estrategia recomendada
    recommended_strategy: StrategyType
    recommended_tools: List[str]

    # Estadísticas
    times_used: int = 0
    success_rate: float = 0.0
    avg_confidence: float = 0.0

    # Aprendizaje
    best_practices: List[str] = field(default_factory=list)
    common_mistakes: List[str] = field(default_factory=list)

    last_updated: datetime = field(default_factory=datetime.now)


class ExperienceStore:
    """
    Almacén persistente de experiencias

    Funcionalidades:
    - Almacenar experiencias con contexto completo
    - Buscar experiencias similares
    - Calcular estadísticas de éxito
    - Extraer patrones de resolución
    """

    def __init__(self, db_path: str = "./data/memory/experiences.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Inicializa esquema de base de datos"""
        with sqlite3.connect(self.db_path) as conn:
            # Tabla de experiencias
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    goal_type TEXT NOT NULL,
                    context TEXT,
                    strategy TEXT NOT NULL,
                    tools_used TEXT,
                    steps_taken INTEGER,
                    outcome TEXT NOT NULL,
                    confidence REAL,
                    execution_time REAL,
                    what_worked TEXT,
                    what_failed TEXT,
                    lessons_learned TEXT,
                    user_feedback TEXT,
                    iterations INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL
                )
            """)

            # Tabla de patrones aprendidos
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    trigger_keywords TEXT,
                    goal_type TEXT,
                    recommended_strategy TEXT,
                    recommended_tools TEXT,
                    times_used INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    avg_confidence REAL DEFAULT 0.0,
                    best_practices TEXT,
                    common_mistakes TEXT,
                    last_updated TEXT
                )
            """)

            # Índices para búsqueda rápida
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_goal_type ON experiences(goal_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_outcome ON experiences(outcome)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_timestamp ON experiences(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pattern_goal ON patterns(goal_type)")

            conn.commit()

    def store_experience(self, experience: Experience) -> str:
        """Almacena una nueva experiencia"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiences
                (id, timestamp, goal, goal_type, context, strategy, tools_used,
                 steps_taken, outcome, confidence, execution_time, what_worked,
                 what_failed, lessons_learned, user_feedback, iterations, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experience.id,
                experience.timestamp.isoformat(),
                experience.goal,
                experience.goal_type,
                json.dumps(experience.context),
                experience.strategy.value,
                json.dumps(experience.tools_used),
                experience.steps_taken,
                experience.outcome.value,
                experience.confidence,
                experience.execution_time,
                json.dumps(experience.what_worked),
                json.dumps(experience.what_failed),
                json.dumps(experience.lessons_learned),
                experience.user_feedback,
                experience.iterations,
                datetime.now().isoformat()
            ))
            conn.commit()

        # Actualizar patrones basados en esta experiencia
        self._update_patterns_from_experience(experience)

        return experience.id

    def find_similar_experiences(
        self,
        goal: str,
        goal_type: Optional[str] = None,
        limit: int = 5
    ) -> List[Experience]:
        """Encuentra experiencias similares a una meta"""
        query = "SELECT * FROM experiences WHERE 1=1"
        params = []

        if goal_type:
            query += " AND goal_type = ?"
            params.append(goal_type)

        # Búsqueda por palabras clave en la meta
        keywords = goal.lower().split()[:5]  # Primeras 5 palabras
        if keywords:
            keyword_conditions = " OR ".join(["goal LIKE ?" for _ in keywords])
            query += f" AND ({keyword_conditions})"
            params.extend([f"%{kw}%" for kw in keywords])

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        return self._execute_query(query, params)

    def get_successful_strategies(
        self,
        goal_type: str,
        min_confidence: float = 0.7
    ) -> List[Tuple[StrategyType, float]]:
        """Obtiene estrategias exitosas para un tipo de meta"""
        query = """
            SELECT strategy, AVG(confidence) as avg_conf, COUNT(*) as count
            FROM experiences
            WHERE goal_type = ?
              AND outcome IN ('success', 'partial')
              AND confidence >= ?
            GROUP BY strategy
            ORDER BY avg_conf DESC
        """

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, (goal_type, min_confidence))
            results = cursor.fetchall()

        return [
            (StrategyType(row[0]), row[1])
            for row in results
        ]

    def get_statistics(self, goal_type: Optional[str] = None) -> Dict[str, Any]:
        """Obtiene estadísticas de experiencias"""
        base_query = "SELECT {} FROM experiences"
        where = f" WHERE goal_type = '{goal_type}'" if goal_type else ""

        with sqlite3.connect(self.db_path) as conn:
            # Total de experiencias
            total = conn.execute(f"SELECT COUNT(*) FROM experiences{where}").fetchone()[0]

            # Por resultado
            outcomes = conn.execute(f"""
                SELECT outcome, COUNT(*)
                FROM experiences{where}
                GROUP BY outcome
            """).fetchall()

            # Promedio de confianza
            avg_conf = conn.execute(f"""
                SELECT AVG(confidence)
                FROM experiences{where}
            """).fetchone()[0] or 0.0

            # Herramientas más usadas
            tools_query = f"""
                SELECT tools_used FROM experiences{where}
            """
            all_tools = conn.execute(tools_query).fetchall()

        # Contar herramientas
        tool_counts = defaultdict(int)
        for row in all_tools:
            if row[0]:
                for tool in json.loads(row[0]):
                    tool_counts[tool] += 1

        return {
            "total_experiences": total,
            "outcomes": {row[0]: row[1] for row in outcomes},
            "average_confidence": round(avg_conf, 3),
            "success_rate": self._calculate_success_rate(outcomes, total),
            "most_used_tools": dict(sorted(
                tool_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            "goal_type_filter": goal_type
        }

    def _calculate_success_rate(self, outcomes: List[Tuple], total: int) -> float:
        """Calcula tasa de éxito"""
        if total == 0:
            return 0.0

        success_count = sum(
            count for outcome, count in outcomes
            if outcome in ('success', 'partial')
        )
        return round(success_count / total, 3)

    def _update_patterns_from_experience(self, exp: Experience) -> None:
        """Actualiza patrones basados en una nueva experiencia"""
        pattern_id = f"pattern_{exp.goal_type}"

        with sqlite3.connect(self.db_path) as conn:
            # Verificar si existe el patrón
            existing = conn.execute(
                "SELECT * FROM patterns WHERE id = ?",
                (pattern_id,)
            ).fetchone()

            if existing:
                # Actualizar patrón existente
                times_used = existing[6] + 1

                # Calcular nueva tasa de éxito
                if exp.outcome in (OutcomeType.SUCCESS, OutcomeType.PARTIAL_SUCCESS):
                    new_success = (existing[7] * existing[6] + 1) / times_used
                else:
                    new_success = (existing[7] * existing[6]) / times_used

                new_confidence = (existing[8] * existing[6] + exp.confidence) / times_used

                # Actualizar best practices y mistakes
                best_practices = json.loads(existing[9] or "[]")
                common_mistakes = json.loads(existing[10] or "[]")

                best_practices.extend(exp.what_worked)
                common_mistakes.extend(exp.what_failed)

                # Limitar a 20 items más recientes
                best_practices = list(set(best_practices))[-20:]
                common_mistakes = list(set(common_mistakes))[-20:]

                conn.execute("""
                    UPDATE patterns SET
                        times_used = ?,
                        success_rate = ?,
                        avg_confidence = ?,
                        best_practices = ?,
                        common_mistakes = ?,
                        last_updated = ?
                    WHERE id = ?
                """, (
                    times_used,
                    new_success,
                    new_confidence,
                    json.dumps(best_practices),
                    json.dumps(common_mistakes),
                    datetime.now().isoformat(),
                    pattern_id
                ))
            else:
                # Crear nuevo patrón
                keywords = exp.goal.lower().split()[:10]
                conn.execute("""
                    INSERT INTO patterns
                    (id, pattern_type, trigger_keywords, goal_type,
                     recommended_strategy, recommended_tools, times_used,
                     success_rate, avg_confidence, best_practices,
                     common_mistakes, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern_id,
                    "goal",
                    json.dumps(keywords),
                    exp.goal_type,
                    exp.strategy.value,
                    json.dumps(exp.tools_used),
                    1,
                    1.0 if exp.outcome == OutcomeType.SUCCESS else 0.0,
                    exp.confidence,
                    json.dumps(exp.what_worked),
                    json.dumps(exp.what_failed),
                    datetime.now().isoformat()
                ))

            conn.commit()

    def get_pattern(self, goal_type: str) -> Optional[Pattern]:
        """Obtiene patrón para un tipo de meta"""
        pattern_id = f"pattern_{goal_type}"

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM patterns WHERE id = ?",
                (pattern_id,)
            ).fetchone()

        if not row:
            return None

        return Pattern(
            id=row[0],
            pattern_type=row[1],
            trigger_keywords=json.loads(row[2] or "[]"),
            goal_type=row[3],
            recommended_strategy=StrategyType(row[4]),
            recommended_tools=json.loads(row[5] or "[]"),
            times_used=row[6],
            success_rate=row[7],
            avg_confidence=row[8],
            best_practices=json.loads(row[9] or "[]"),
            common_mistakes=json.loads(row[10] or "[]"),
            last_updated=datetime.fromisoformat(row[11]) if row[11] else datetime.now()
        )

    def _execute_query(self, query: str, params: List) -> List[Experience]:
        """Ejecuta query y retorna experiencias"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        experiences = []
        for row in rows:
            try:
                exp = Experience(
                    id=row['id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    goal=row['goal'],
                    goal_type=row['goal_type'],
                    context=json.loads(row['context'] or "{}"),
                    strategy=StrategyType(row['strategy']),
                    tools_used=json.loads(row['tools_used'] or "[]"),
                    steps_taken=row['steps_taken'],
                    outcome=OutcomeType(row['outcome']),
                    confidence=row['confidence'],
                    execution_time=row['execution_time'],
                    what_worked=json.loads(row['what_worked'] or "[]"),
                    what_failed=json.loads(row['what_failed'] or "[]"),
                    lessons_learned=json.loads(row['lessons_learned'] or "[]"),
                    user_feedback=row['user_feedback'],
                    iterations=row['iterations']
                )
                experiences.append(exp)
            except Exception as e:
                print(f"Error parsing experience: {e}")

        return experiences

    def get_lessons_for_goal(self, goal: str, goal_type: str) -> Dict[str, Any]:
        """
        Obtiene lecciones aprendidas relevantes para una meta

        Retorna:
        - Estrategia recomendada
        - Herramientas sugeridas
        - Errores a evitar
        - Tips de experiencias exitosas
        """
        # Buscar experiencias similares
        similar = self.find_similar_experiences(goal, goal_type, limit=10)

        # Obtener patrón si existe
        pattern = self.get_pattern(goal_type)

        # Agregar lecciones
        all_worked = []
        all_failed = []
        all_lessons = []
        successful_strategies = []

        for exp in similar:
            if exp.outcome in (OutcomeType.SUCCESS, OutcomeType.PARTIAL_SUCCESS):
                all_worked.extend(exp.what_worked)
                successful_strategies.append((exp.strategy, exp.confidence))
            else:
                all_failed.extend(exp.what_failed)
            all_lessons.extend(exp.lessons_learned)

        # Determinar mejor estrategia
        if successful_strategies:
            best_strategy = max(successful_strategies, key=lambda x: x[1])
        elif pattern:
            best_strategy = (pattern.recommended_strategy, pattern.avg_confidence)
        else:
            best_strategy = (StrategyType.DIRECT, 0.5)

        return {
            "recommended_strategy": best_strategy[0].value,
            "confidence": best_strategy[1],
            "suggested_tools": pattern.recommended_tools if pattern else [],
            "tips": list(set(all_worked))[:5],
            "avoid": list(set(all_failed))[:5],
            "lessons": list(set(all_lessons))[:5],
            "similar_experiences": len(similar),
            "pattern_exists": pattern is not None,
            "pattern_success_rate": pattern.success_rate if pattern else 0.0
        }


class MetacognitiveEngine:
    """
    Motor de Metacognición para THAU

    Implementa:
    - Auto-evaluación de respuestas
    - Detección de incertidumbre
    - Identificación de gaps de conocimiento
    - Sugerencias de mejora
    """

    def __init__(self, experience_store: ExperienceStore):
        self.experience_store = experience_store

        # Umbrales de confianza
        self.thresholds = {
            "high_confidence": 0.85,
            "medium_confidence": 0.6,
            "low_confidence": 0.4,
            "uncertain": 0.2
        }

        # Indicadores de incertidumbre
        self.uncertainty_markers = [
            "no estoy seguro",
            "podría ser",
            "quizás",
            "tal vez",
            "no sé",
            "probablemente",
            "creo que",
            "supongo",
            "parece que",
            "maybe",
            "perhaps",
            "I think",
            "not sure"
        ]

        # Indicadores de conocimiento sólido
        self.confidence_markers = [
            "definitivamente",
            "con certeza",
            "claramente",
            "evidentemente",
            "sin duda",
            "certainly",
            "definitely",
            "clearly"
        ]

    def evaluate_response(
        self,
        goal: str,
        response: str,
        tool_results: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Evalúa la calidad de una respuesta

        Returns:
            Dict con:
            - confidence: nivel de confianza 0-1
            - quality_score: puntuación de calidad
            - issues: problemas detectados
            - suggestions: sugerencias de mejora
        """
        evaluation = {
            "confidence": 0.5,
            "quality_score": 0.5,
            "issues": [],
            "suggestions": [],
            "uncertainty_detected": False,
            "knowledge_gaps": [],
            "strengths": []
        }

        # 1. Análisis de incertidumbre en el texto
        uncertainty_count = sum(
            1 for marker in self.uncertainty_markers
            if marker.lower() in response.lower()
        )

        confidence_count = sum(
            1 for marker in self.confidence_markers
            if marker.lower() in response.lower()
        )

        if uncertainty_count > 0:
            evaluation["uncertainty_detected"] = True
            evaluation["confidence"] -= 0.1 * uncertainty_count
            evaluation["issues"].append(
                f"Detectada incertidumbre ({uncertainty_count} indicadores)"
            )

        if confidence_count > 0:
            evaluation["confidence"] += 0.05 * confidence_count
            evaluation["strengths"].append("Respuesta con convicción")

        # 2. Análisis de completitud
        response_length = len(response)
        goal_length = len(goal)

        if response_length < goal_length * 0.5:
            evaluation["issues"].append("Respuesta muy corta para la complejidad de la pregunta")
            evaluation["quality_score"] -= 0.15
            evaluation["suggestions"].append("Expandir la respuesta con más detalles")
        elif response_length > goal_length * 20:
            evaluation["issues"].append("Respuesta excesivamente larga")
            evaluation["suggestions"].append("Condensar la respuesta")

        # 3. Análisis de resultados de herramientas
        if tool_results:
            successful_tools = sum(1 for r in tool_results if r.get("success", False))
            total_tools = len(tool_results)

            tool_success_rate = successful_tools / total_tools if total_tools > 0 else 0

            evaluation["confidence"] = (evaluation["confidence"] + tool_success_rate) / 2

            if tool_success_rate < 0.5:
                evaluation["issues"].append(
                    f"Solo {successful_tools}/{total_tools} herramientas exitosas"
                )
                evaluation["suggestions"].append("Revisar uso de herramientas")
            else:
                evaluation["strengths"].append("Buen uso de herramientas")

        # 4. Verificar si hay gaps de conocimiento
        knowledge_gap_indicators = [
            "no tengo información",
            "no puedo acceder",
            "fuera de mi conocimiento",
            "no tengo datos",
            "I don't have",
            "cannot access",
            "outside my knowledge"
        ]

        for indicator in knowledge_gap_indicators:
            if indicator.lower() in response.lower():
                evaluation["knowledge_gaps"].append(indicator)
                evaluation["confidence"] -= 0.1

        # 5. Normalizar scores
        evaluation["confidence"] = max(0.0, min(1.0, evaluation["confidence"]))
        evaluation["quality_score"] = max(0.0, min(1.0, evaluation["quality_score"]))

        # 6. Clasificación final
        if evaluation["confidence"] >= self.thresholds["high_confidence"]:
            evaluation["confidence_level"] = "high"
        elif evaluation["confidence"] >= self.thresholds["medium_confidence"]:
            evaluation["confidence_level"] = "medium"
        elif evaluation["confidence"] >= self.thresholds["low_confidence"]:
            evaluation["confidence_level"] = "low"
        else:
            evaluation["confidence_level"] = "uncertain"

        return evaluation

    def suggest_improvements(
        self,
        goal: str,
        current_approach: StrategyType,
        outcome: OutcomeType
    ) -> List[str]:
        """Sugiere mejoras basadas en el resultado"""
        suggestions = []

        # Buscar patrones de éxito
        lessons = self.experience_store.get_lessons_for_goal(
            goal,
            self._classify_goal(goal)
        )

        if outcome in (OutcomeType.FAILURE, OutcomeType.ERROR):
            # Obtener estrategias que funcionaron antes
            if lessons["recommended_strategy"] != current_approach.value:
                suggestions.append(
                    f"Considera usar estrategia '{lessons['recommended_strategy']}' "
                    f"(tasa de éxito: {lessons['pattern_success_rate']:.0%})"
                )

            # Agregar tips de experiencias exitosas
            for tip in lessons["tips"][:3]:
                suggestions.append(f"Tip: {tip}")

            # Advertir sobre errores comunes
            for avoid in lessons["avoid"][:2]:
                suggestions.append(f"Evitar: {avoid}")

        elif outcome == OutcomeType.PARTIAL_SUCCESS:
            suggestions.append("Considerar descomponer el problema en partes más pequeñas")
            suggestions.append("Verificar que todas las sub-metas fueron completadas")

        return suggestions

    def _classify_goal(self, goal: str) -> str:
        """Clasifica el tipo de meta"""
        goal_lower = goal.lower()

        if any(w in goal_lower for w in ["calcula", "suma", "resta", "cuanto", "math"]):
            return "calculation"
        elif any(w in goal_lower for w in ["código", "programa", "función", "code", "script"]):
            return "code"
        elif any(w in goal_lower for w in ["archivo", "lee", "escribe", "file", "read", "write"]):
            return "file"
        elif any(w in goal_lower for w in ["busca", "encuentra", "search", "find"]):
            return "search"
        elif any(w in goal_lower for w in ["explica", "qué es", "cómo", "explain", "what", "how"]):
            return "question"
        else:
            return "general"

    def reflect_on_session(
        self,
        experiences: List[Experience]
    ) -> Dict[str, Any]:
        """
        Reflexión profunda sobre una sesión completa

        Analiza patrones, éxitos, fracasos y genera insights
        """
        if not experiences:
            return {"message": "No hay experiencias para analizar"}

        reflection = {
            "session_summary": {},
            "patterns_observed": [],
            "key_learnings": [],
            "areas_for_improvement": [],
            "strengths_identified": [],
            "recommendations": []
        }

        # Resumen de sesión
        total = len(experiences)
        successes = sum(1 for e in experiences if e.outcome == OutcomeType.SUCCESS)
        partials = sum(1 for e in experiences if e.outcome == OutcomeType.PARTIAL_SUCCESS)
        failures = sum(1 for e in experiences if e.outcome in (OutcomeType.FAILURE, OutcomeType.ERROR))

        reflection["session_summary"] = {
            "total_interactions": total,
            "successes": successes,
            "partial_successes": partials,
            "failures": failures,
            "success_rate": (successes + partials * 0.5) / total if total > 0 else 0,
            "average_confidence": sum(e.confidence for e in experiences) / total if total > 0 else 0,
            "total_time": sum(e.execution_time for e in experiences)
        }

        # Patrones observados
        strategy_counts = defaultdict(int)
        goal_type_counts = defaultdict(int)
        tool_counts = defaultdict(int)

        for exp in experiences:
            strategy_counts[exp.strategy.value] += 1
            goal_type_counts[exp.goal_type] += 1
            for tool in exp.tools_used:
                tool_counts[tool] += 1

        reflection["patterns_observed"] = [
            f"Estrategia más usada: {max(strategy_counts, key=strategy_counts.get)}",
            f"Tipo de tarea más común: {max(goal_type_counts, key=goal_type_counts.get)}",
            f"Herramienta favorita: {max(tool_counts, key=tool_counts.get) if tool_counts else 'ninguna'}"
        ]

        # Aprendizajes clave
        all_lessons = []
        all_worked = []
        all_failed = []

        for exp in experiences:
            all_lessons.extend(exp.lessons_learned)
            all_worked.extend(exp.what_worked)
            all_failed.extend(exp.what_failed)

        reflection["key_learnings"] = list(set(all_lessons))[:10]
        reflection["strengths_identified"] = list(set(all_worked))[:5]
        reflection["areas_for_improvement"] = list(set(all_failed))[:5]

        # Recomendaciones
        if reflection["session_summary"]["success_rate"] < 0.5:
            reflection["recommendations"].append(
                "Tasa de éxito baja - considerar simplificar metas o mejorar estrategias"
            )

        if failures > successes:
            reflection["recommendations"].append(
                "Más fracasos que éxitos - revisar patrones de error comunes"
            )

        avg_conf = reflection["session_summary"]["average_confidence"]
        if avg_conf < 0.6:
            reflection["recommendations"].append(
                f"Confianza promedio baja ({avg_conf:.0%}) - buscar más contexto antes de responder"
            )

        return reflection


class AdaptiveStrategy:
    """
    Sistema de Estrategia Adaptativa

    Ajusta el comportamiento de THAU basándose en:
    - Experiencias pasadas
    - Patrones de éxito
    - Contexto actual
    - Feedback del usuario
    """

    def __init__(
        self,
        experience_store: ExperienceStore,
        metacognitive: MetacognitiveEngine
    ):
        self.experience_store = experience_store
        self.metacognitive = metacognitive

        # Configuración adaptativa
        self.config = {
            "exploration_rate": 0.2,  # Probar nuevas estrategias
            "confidence_threshold": 0.6,  # Umbral mínimo
            "max_retries": 3,
            "learn_from_failures": True
        }

        # Estado de sesión
        self.session_experiences: List[Experience] = []
        self.current_strategy: Optional[StrategyType] = None

    def select_strategy(
        self,
        goal: str,
        available_tools: List[str],
        context: Dict[str, Any] = None
    ) -> Tuple[StrategyType, Dict[str, Any]]:
        """
        Selecciona la mejor estrategia para una meta

        Returns:
            Tuple de (estrategia, metadata)
        """
        goal_type = self.metacognitive._classify_goal(goal)

        # Obtener lecciones de experiencias pasadas
        lessons = self.experience_store.get_lessons_for_goal(goal, goal_type)

        # Decidir si explorar o explotar
        import random
        if random.random() < self.config["exploration_rate"]:
            # Exploración: probar estrategia diferente
            all_strategies = list(StrategyType)
            recommended = StrategyType(lessons["recommended_strategy"])
            other_strategies = [s for s in all_strategies if s != recommended]

            if other_strategies:
                strategy = random.choice(other_strategies)
                reason = "exploration"
            else:
                strategy = recommended
                reason = "only_option"
        else:
            # Explotación: usar estrategia recomendada
            strategy = StrategyType(lessons["recommended_strategy"])
            reason = "recommended"

        # Determinar herramientas sugeridas
        suggested_tools = lessons.get("suggested_tools", [])
        if not suggested_tools:
            # Inferir herramientas basadas en el tipo de meta
            suggested_tools = self._infer_tools(goal_type, available_tools)

        metadata = {
            "reason": reason,
            "confidence": lessons["confidence"],
            "suggested_tools": suggested_tools,
            "tips": lessons["tips"],
            "avoid": lessons["avoid"],
            "similar_experiences": lessons["similar_experiences"],
            "pattern_success_rate": lessons["pattern_success_rate"]
        }

        self.current_strategy = strategy

        return strategy, metadata

    def _infer_tools(self, goal_type: str, available: List[str]) -> List[str]:
        """Infiere herramientas basadas en el tipo de meta"""
        tool_mapping = {
            "calculation": ["calculate"],
            "code": ["execute_python", "read_file", "write_file"],
            "file": ["read_file", "write_file", "list_directory"],
            "search": ["list_directory", "read_file"],
            "question": [],
            "general": []
        }

        suggested = tool_mapping.get(goal_type, [])
        return [t for t in suggested if t in available]

    def record_outcome(
        self,
        goal: str,
        strategy: StrategyType,
        tools_used: List[str],
        outcome: OutcomeType,
        confidence: float,
        execution_time: float,
        what_worked: List[str] = None,
        what_failed: List[str] = None,
        lessons: List[str] = None,
        context: Dict = None
    ) -> Experience:
        """Registra el resultado de una interacción"""
        experience = Experience(
            id=hashlib.md5(f"{goal}{datetime.now().isoformat()}".encode()).hexdigest()[:12],
            timestamp=datetime.now(),
            goal=goal,
            goal_type=self.metacognitive._classify_goal(goal),
            context=context or {},
            strategy=strategy,
            tools_used=tools_used,
            steps_taken=len(tools_used) + 1,
            outcome=outcome,
            confidence=confidence,
            execution_time=execution_time,
            what_worked=what_worked or [],
            what_failed=what_failed or [],
            lessons_learned=lessons or []
        )

        # Almacenar experiencia
        self.experience_store.store_experience(experience)

        # Agregar a sesión actual
        self.session_experiences.append(experience)

        return experience

    def adapt_from_feedback(self, feedback: str, experience_id: str) -> None:
        """Adapta basándose en feedback del usuario"""
        # Buscar la experiencia y actualizar
        with sqlite3.connect(self.experience_store.db_path) as conn:
            conn.execute(
                "UPDATE experiences SET user_feedback = ? WHERE id = ?",
                (feedback, experience_id)
            )
            conn.commit()

        # Ajustar configuración si hay feedback negativo
        negative_markers = ["mal", "incorrecto", "error", "wrong", "bad", "incorrect"]
        if any(marker in feedback.lower() for marker in negative_markers):
            # Reducir confianza en la estrategia actual
            self.config["exploration_rate"] = min(0.5, self.config["exploration_rate"] + 0.1)

    def end_session(self) -> Dict[str, Any]:
        """Finaliza sesión y genera reflexión"""
        if not self.session_experiences:
            return {"message": "Sesión vacía"}

        # Generar reflexión sobre la sesión
        reflection = self.metacognitive.reflect_on_session(self.session_experiences)

        # Limpiar estado de sesión
        session_count = len(self.session_experiences)
        self.session_experiences = []
        self.current_strategy = None

        reflection["session_ended"] = True
        reflection["experiences_recorded"] = session_count

        return reflection


def create_experience_id(goal: str) -> str:
    """Genera ID único para una experiencia"""
    timestamp = datetime.now().isoformat()
    return hashlib.md5(f"{goal}{timestamp}".encode()).hexdigest()[:12]


# Singleton para acceso global
_experience_store: Optional[ExperienceStore] = None
_metacognitive: Optional[MetacognitiveEngine] = None
_adaptive_strategy: Optional[AdaptiveStrategy] = None


def get_experience_store() -> ExperienceStore:
    """Obtiene instancia singleton de ExperienceStore"""
    global _experience_store
    if _experience_store is None:
        _experience_store = ExperienceStore()
    return _experience_store


def get_metacognitive_engine() -> MetacognitiveEngine:
    """Obtiene instancia singleton de MetacognitiveEngine"""
    global _metacognitive
    if _metacognitive is None:
        _metacognitive = MetacognitiveEngine(get_experience_store())
    return _metacognitive


def get_adaptive_strategy() -> AdaptiveStrategy:
    """Obtiene instancia singleton de AdaptiveStrategy"""
    global _adaptive_strategy
    if _adaptive_strategy is None:
        _adaptive_strategy = AdaptiveStrategy(
            get_experience_store(),
            get_metacognitive_engine()
        )
    return _adaptive_strategy


if __name__ == "__main__":
    print("=" * 60)
    print("  THAU Experiential Learning System - Demo")
    print("=" * 60)

    # Inicializar componentes
    store = get_experience_store()
    metacog = get_metacognitive_engine()
    adaptive = get_adaptive_strategy()

    # Demo: Registrar experiencias de prueba
    print("\n1. Registrando experiencias de prueba...")

    # Experiencia exitosa
    exp1 = adaptive.record_outcome(
        goal="Calcula 15 * 23",
        strategy=StrategyType.TOOL_SINGLE,
        tools_used=["calculate"],
        outcome=OutcomeType.SUCCESS,
        confidence=0.95,
        execution_time=0.5,
        what_worked=["Uso directo de herramienta calculate"],
        lessons=["Los cálculos simples funcionan bien con una herramienta"]
    )
    print(f"   Experiencia 1: {exp1.id} - {exp1.outcome.value}")

    # Experiencia parcial
    exp2 = adaptive.record_outcome(
        goal="Lee el archivo config.py y explica su contenido",
        strategy=StrategyType.TOOL_CHAIN,
        tools_used=["read_file"],
        outcome=OutcomeType.PARTIAL_SUCCESS,
        confidence=0.7,
        execution_time=1.2,
        what_worked=["Lectura de archivo exitosa"],
        what_failed=["Explicación podría ser más detallada"],
        lessons=["Combinar lectura con análisis detallado"]
    )
    print(f"   Experiencia 2: {exp2.id} - {exp2.outcome.value}")

    # Experiencia fallida
    exp3 = adaptive.record_outcome(
        goal="Ejecuta un script que accede a internet",
        strategy=StrategyType.TOOL_SINGLE,
        tools_used=["execute_python"],
        outcome=OutcomeType.ERROR,
        confidence=0.3,
        execution_time=30.0,
        what_failed=["Timeout en ejecución", "Sin acceso a red"],
        lessons=["Verificar requisitos antes de ejecutar scripts complejos"]
    )
    print(f"   Experiencia 3: {exp3.id} - {exp3.outcome.value}")

    # Demo: Obtener estadísticas
    print("\n2. Estadísticas de experiencias:")
    stats = store.get_statistics()
    print(f"   Total: {stats['total_experiences']}")
    print(f"   Tasa de éxito: {stats['success_rate']:.0%}")
    print(f"   Confianza promedio: {stats['average_confidence']:.2f}")

    # Demo: Seleccionar estrategia
    print("\n3. Selección de estrategia adaptativa:")
    strategy, metadata = adaptive.select_strategy(
        goal="Calcula el factorial de 10",
        available_tools=["calculate", "execute_python", "read_file"]
    )
    print(f"   Estrategia seleccionada: {strategy.value}")
    print(f"   Razón: {metadata['reason']}")
    print(f"   Confianza: {metadata['confidence']:.2f}")
    print(f"   Herramientas sugeridas: {metadata['suggested_tools']}")

    # Demo: Evaluación metacognitiva
    print("\n4. Evaluación metacognitiva:")
    evaluation = metacog.evaluate_response(
        goal="Explica qué es Python",
        response="Python es un lenguaje de programación interpretado, de alto nivel y propósito general. Creo que es muy popular.",
        tool_results=[]
    )
    print(f"   Confianza: {evaluation['confidence']:.2f}")
    print(f"   Nivel: {evaluation['confidence_level']}")
    print(f"   Incertidumbre detectada: {evaluation['uncertainty_detected']}")
    print(f"   Issues: {evaluation['issues']}")

    # Demo: Reflexión de sesión
    print("\n5. Reflexión de sesión:")
    reflection = adaptive.end_session()
    print(f"   Interacciones: {reflection['session_summary']['total_interactions']}")
    print(f"   Tasa de éxito: {reflection['session_summary']['success_rate']:.0%}")
    print(f"   Aprendizajes: {len(reflection['key_learnings'])}")
    print(f"   Recomendaciones: {reflection['recommendations']}")

    print("\n" + "=" * 60)
    print("  Demo completada!")
    print("=" * 60)
