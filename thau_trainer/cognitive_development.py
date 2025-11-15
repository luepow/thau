"""
Sistema de Desarrollo Cognitivo para THAU
Inspirado en el desarrollo humano de 0 a 15 años
"""

from dataclasses import dataclass
from typing import List, Dict, Literal
from enum import Enum
import json


class CognitiveAge(Enum):
    """Edades cognitivas de THAU"""
    NEWBORN = 0  # Recién nacido
    INFANT = 1  # Infante (1-2 años)
    TODDLER = 2  # Niño pequeño (3-5 años)
    CHILD = 6  # Niño (6-10 años)
    PRETEEN = 11  # Pre-adolescente (11-12 años)
    TEEN = 13  # Adolescente (13-15 años)
    ADULT = 15  # Adulto (15+)


@dataclass
class CognitiveStage:
    """Etapa del desarrollo cognitivo"""

    age: int
    name: str
    description: str

    # Capacidades habilitadas en esta edad
    capabilities: List[str]

    # Tipos de conocimiento que puede aprender
    learning_domains: List[str]

    # Complejidad del razonamiento (1-10)
    reasoning_complexity: int

    # Contexto máximo que puede manejar
    context_length: int

    # Parámetros de entrenamiento
    learning_rate: float
    epochs: int

    # Criterios para avanzar a la siguiente edad
    advancement_criteria: Dict


# Definición de todas las etapas
COGNITIVE_STAGES = {
    0: CognitiveStage(
        age=0,
        name="Recién Nacido",
        description="Etapa inicial. Reconocimiento de patrones básicos y respuestas simples.",
        capabilities=[
            "Reconocer palabras clave",
            "Respuestas de una palabra",
            "Patrones muy simples"
        ],
        learning_domains=[
            "Vocabulario básico",
            "Reconocimiento de entidades",
            "Respuestas sí/no"
        ],
        reasoning_complexity=1,
        context_length=128,
        learning_rate=5e-4,
        epochs=5,
        advancement_criteria={
            "min_examples": 100,
            "min_accuracy": 0.7,
            "simple_patterns_mastered": True
        }
    ),

    1: CognitiveStage(
        age=1,
        name="Infante (1-2 años)",
        description="Desarrollo del lenguaje básico. Frases cortas y conceptos simples.",
        capabilities=[
            "Frases de 2-3 palabras",
            "Comprensión de instrucciones simples",
            "Asociación básica causa-efecto"
        ],
        learning_domains=[
            "Lenguaje básico",
            "Conceptos simples (colores, números)",
            "Acciones básicas (qué es, cómo se llama)"
        ],
        reasoning_complexity=2,
        context_length=256,
        learning_rate=3e-4,
        epochs=4,
        advancement_criteria={
            "min_examples": 200,
            "min_accuracy": 0.75,
            "sentence_formation": True
        }
    ),

    3: CognitiveStage(
        age=3,
        name="Niño Pequeño (3-5 años)",
        description="Razonamiento simple. Puede explicar conceptos básicos con ejemplos.",
        capabilities=[
            "Explicaciones simples",
            "Comparaciones básicas",
            "Razonamiento causa-efecto simple",
            "Preguntas 'por qué'"
        ],
        learning_domains=[
            "Conceptos cotidianos",
            "Relaciones simples",
            "Categorización",
            "Secuencias temporales (antes/después)"
        ],
        reasoning_complexity=3,
        context_length=512,
        learning_rate=2e-4,
        epochs=3,
        advancement_criteria={
            "min_examples": 500,
            "min_accuracy": 0.80,
            "can_explain_simple_concepts": True
        }
    ),

    6: CognitiveStage(
        age=6,
        name="Niño (6-10 años)",
        description="Lógica concreta. Matemáticas básicas, lectura comprensiva.",
        capabilities=[
            "Operaciones matemáticas básicas",
            "Razonamiento lógico simple",
            "Comprensión de textos simples",
            "Resolución de problemas paso a paso"
        ],
        learning_domains=[
            "Matemáticas elementales",
            "Ciencias básicas",
            "Lectura y escritura",
            "Reglas y patrones",
            "Clasificación y ordenamiento"
        ],
        reasoning_complexity=5,
        context_length=1024,
        learning_rate=2e-4,
        epochs=3,
        advancement_criteria={
            "min_examples": 1000,
            "min_accuracy": 0.85,
            "logical_reasoning": True,
            "math_basic": True
        }
    ),

    11: CognitiveStage(
        age=11,
        name="Pre-adolescente (11-12 años)",
        description="Pensamiento abstracto emergente. Puede manejar conceptos más complejos.",
        capabilities=[
            "Razonamiento abstracto básico",
            "Pensamiento hipotético",
            "Múltiples perspectivas",
            "Análisis de información"
        ],
        learning_domains=[
            "Álgebra básica",
            "Conceptos científicos intermedios",
            "Literatura y análisis",
            "Tecnología y programación básica",
            "Relaciones complejas"
        ],
        reasoning_complexity=7,
        context_length=2048,
        learning_rate=2e-4,
        epochs=3,
        advancement_criteria={
            "min_examples": 2000,
            "min_accuracy": 0.88,
            "abstract_thinking": True,
            "multi_step_reasoning": True
        }
    ),

    13: CognitiveStage(
        age=13,
        name="Adolescente (13-15 años)",
        description="Pensamiento crítico. Razonamiento complejo y análisis profundo.",
        capabilities=[
            "Pensamiento crítico avanzado",
            "Razonamiento multi-paso complejo",
            "Análisis y síntesis",
            "Metacognición (pensar sobre pensar)",
            "Argumentación lógica"
        ],
        learning_domains=[
            "Matemáticas avanzadas",
            "Ciencias complejas",
            "Programación intermedia",
            "Filosofía y ética",
            "Análisis crítico"
        ],
        reasoning_complexity=8,
        context_length=3072,
        learning_rate=2e-4,
        epochs=3,
        advancement_criteria={
            "min_examples": 3000,
            "min_accuracy": 0.90,
            "critical_thinking": True,
            "complex_problem_solving": True
        }
    ),

    15: CognitiveStage(
        age=15,
        name="Adulto (15+ años)",
        description="Madurez cognitiva completa. Especialización y expertise profundo.",
        capabilities=[
            "Razonamiento experto",
            "Pensamiento sistémico",
            "Creatividad e innovación",
            "Transferencia de conocimiento",
            "Tool calling y razonamiento agentico",
            "Chain-of-thought avanzado"
        ],
        learning_domains=[
            "Especialización técnica",
            "Arquitectura de software",
            "Matemáticas y lógica avanzada",
            "Investigación y síntesis",
            "Diseño de sistemas complejos",
            "Cualquier dominio especializado"
        ],
        reasoning_complexity=10,
        context_length=4096,
        learning_rate=2e-4,
        epochs=3,
        advancement_criteria={
            "min_examples": 5000,
            "min_accuracy": 0.92,
            "expert_level": True
        }
    )
}


class CognitiveDevelopmentManager:
    """Gestor del desarrollo cognitivo de THAU"""

    def __init__(self):
        self.current_age = 0
        self.stage = COGNITIVE_STAGES[0]
        self.progress = {
            "total_examples_seen": 0,
            "examples_per_age": {},
            "accuracy_per_age": {},
            "milestones_achieved": []
        }
        self._load_progress()

    def _load_progress(self):
        """Carga el progreso guardado"""
        progress_file = Path("./data/logs/cognitive_progress.json")
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                data = json.load(f)
                self.current_age = data.get("current_age", 0)
                self.progress = data.get("progress", self.progress)
                self.stage = COGNITIVE_STAGES[self.current_age]

    def _save_progress(self):
        """Guarda el progreso"""
        progress_file = Path("./data/logs/cognitive_progress.json")
        progress_file.parent.mkdir(parents=True, exist_ok=True)

        with open(progress_file, 'w') as f:
            json.dump({
                "current_age": self.current_age,
                "progress": self.progress,
                "stage_name": self.stage.name
            }, f, indent=2)

    def get_current_stage(self) -> CognitiveStage:
        """Obtiene la etapa actual"""
        return self.stage

    def record_training(self, num_examples: int, accuracy: float):
        """Registra un entrenamiento"""

        self.progress["total_examples_seen"] += num_examples

        age_key = str(self.current_age)
        if age_key not in self.progress["examples_per_age"]:
            self.progress["examples_per_age"][age_key] = 0
            self.progress["accuracy_per_age"][age_key] = []

        self.progress["examples_per_age"][age_key] += num_examples
        self.progress["accuracy_per_age"][age_key].append(accuracy)

        self._save_progress()

    def check_advancement(self) -> bool:
        """Verifica si puede avanzar a la siguiente edad"""

        criteria = self.stage.advancement_criteria
        age_key = str(self.current_age)

        # Verificar ejemplos mínimos
        examples = self.progress["examples_per_age"].get(age_key, 0)
        if examples < criteria.get("min_examples", 0):
            return False

        # Verificar accuracy mínima
        accuracies = self.progress["accuracy_per_age"].get(age_key, [])
        if not accuracies:
            return False

        avg_accuracy = sum(accuracies[-5:]) / min(5, len(accuracies))  # Últimos 5
        if avg_accuracy < criteria.get("min_accuracy", 0):
            return False

        return True

    def advance_age(self):
        """Avanza a la siguiente edad"""

        # Encontrar siguiente edad disponible
        next_ages = [age for age in COGNITIVE_STAGES.keys() if age > self.current_age]

        if not next_ages:
            print("🎓 THAU ya alcanzó la madurez cognitiva completa!")
            return False

        next_age = min(next_ages)

        # Milestone
        milestone = {
            "from_age": self.current_age,
            "to_age": next_age,
            "timestamp": datetime.now().isoformat(),
            "examples_trained": self.progress["examples_per_age"].get(str(self.current_age), 0)
        }
        self.progress["milestones_achieved"].append(milestone)

        # Actualizar
        self.current_age = next_age
        self.stage = COGNITIVE_STAGES[next_age]

        self._save_progress()

        print(f"🎉 ¡THAU avanzó a edad {next_age}: {self.stage.name}!")
        print(f"📚 Nuevas capacidades: {', '.join(self.stage.capabilities)}")

        return True

    def get_learning_params(self) -> Dict:
        """Obtiene parámetros de aprendizaje según la edad"""
        return {
            "learning_rate": self.stage.learning_rate,
            "epochs": self.stage.epochs,
            "context_length": self.stage.context_length,
            "complexity_level": self.stage.reasoning_complexity
        }

    def get_status(self) -> Dict:
        """Estado del desarrollo cognitivo"""
        age_key = str(self.current_age)
        examples_at_age = self.progress["examples_per_age"].get(age_key, 0)
        accuracies = self.progress["accuracy_per_age"].get(age_key, [])
        avg_accuracy = sum(accuracies[-5:]) / min(5, len(accuracies)) if accuracies else 0

        criteria = self.stage.advancement_criteria
        min_examples = criteria.get("min_examples", 0)
        min_accuracy = criteria.get("min_accuracy", 0)

        return {
            "current_age": self.current_age,
            "stage_name": self.stage.name,
            "description": self.stage.description,
            "capabilities": self.stage.capabilities,
            "progress": {
                "examples_at_age": examples_at_age,
                "examples_needed": min_examples,
                "progress_pct": min(100, (examples_at_age / min_examples * 100)) if min_examples > 0 else 0,
                "current_accuracy": avg_accuracy,
                "required_accuracy": min_accuracy,
                "can_advance": self.check_advancement()
            },
            "total_examples": self.progress["total_examples_seen"],
            "milestones": len(self.progress["milestones_achieved"])
        }


from pathlib import Path
from datetime import datetime
