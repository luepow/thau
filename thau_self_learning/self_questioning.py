"""
Self-Questioning System for THAU
Generates questions to drive autonomous learning and knowledge expansion
"""

import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class QuestionType(Enum):
    """Types of self-generated questions"""
    CONCEPTUAL = "conceptual"          # What is X? Define Y
    PROCEDURAL = "procedural"          # How to do X? Steps for Y
    COMPARATIVE = "comparative"        # Difference between X and Y
    CAUSAL = "causal"                  # Why does X happen? What causes Y?
    EXPLORATORY = "exploratory"        # What if X? Explore Y
    REFLECTIVE = "reflective"          # How did I solve X? What did I learn from Y?


@dataclass
class SelfQuestion:
    """Represents a self-generated question for learning"""
    question: str
    question_type: QuestionType
    domain: str
    difficulty: int  # 1-10
    context: str = ""
    expected_answer_length: int = 100  # tokens


class SelfQuestioningEngine:
    """
    Generates questions for autonomous learning

    The model asks itself questions to:
    1. Identify knowledge gaps
    2. Explore new concepts
    3. Practice reasoning
    4. Reinforce learning
    """

    def __init__(self):
        self.question_templates = self._init_templates()
        self.domains = [
            "programming", "architecture", "algorithms", "databases",
            "web_development", "machine_learning", "systems_design",
            "security", "devops", "mathematics", "physics", "philosophy"
        ]

    def _init_templates(self) -> Dict[QuestionType, List[str]]:
        """Initialize question templates for each type"""
        return {
            QuestionType.CONCEPTUAL: [
                "¿Qué es {concept}?",
                "Define {concept} en términos simples",
                "Explica el concepto de {concept}",
                "¿Cuáles son las características principales de {concept}?",
                "¿Qué significa {concept} en el contexto de {domain}?",
            ],
            QuestionType.PROCEDURAL: [
                "¿Cómo implemento {task}?",
                "¿Cuáles son los pasos para {task}?",
                "¿Cómo puedo resolver {problem}?",
                "¿Qué proceso debo seguir para {task}?",
                "Explica paso a paso cómo hacer {task}",
            ],
            QuestionType.COMPARATIVE: [
                "¿Cuál es la diferencia entre {concept_a} y {concept_b}?",
                "Compara {concept_a} con {concept_b}",
                "¿Cuándo usar {concept_a} en lugar de {concept_b}?",
                "¿Qué ventajas tiene {concept_a} sobre {concept_b}?",
                "¿En qué se parecen y diferencian {concept_a} y {concept_b}?",
            ],
            QuestionType.CAUSAL: [
                "¿Por qué ocurre {phenomenon}?",
                "¿Qué causa {effect}?",
                "¿Cuál es la razón detrás de {concept}?",
                "¿Por qué es importante {concept}?",
                "¿Qué factores influyen en {outcome}?",
            ],
            QuestionType.EXPLORATORY: [
                "¿Qué pasaría si {scenario}?",
                "¿Cómo afectaría {change} a {system}?",
                "¿Qué alternativas existen para {approach}?",
                "Explora las posibilidades de {concept}",
                "¿Qué innovaciones hay en {field}?",
            ],
            QuestionType.REFLECTIVE: [
                "¿Cómo resolvería {problem} de manera diferente ahora?",
                "¿Qué aprendí de {experience}?",
                "¿Cómo mejoró mi comprensión de {concept}?",
                "¿Qué errores comunes ocurren con {topic}?",
                "¿Cómo puedo aplicar {knowledge} en otros contextos?",
            ],
        }

    def generate_question(
        self,
        question_type: QuestionType = None,
        domain: str = None,
        difficulty: int = 5
    ) -> SelfQuestion:
        """
        Generate a single self-directed question

        Args:
            question_type: Type of question to generate (random if None)
            domain: Domain for the question (random if None)
            difficulty: Difficulty level 1-10

        Returns:
            SelfQuestion object
        """
        if question_type is None:
            question_type = random.choice(list(QuestionType))

        if domain is None:
            domain = random.choice(self.domains)

        template = random.choice(self.question_templates[question_type])

        # Fill template with domain-specific content
        question_text = self._fill_template(template, domain, question_type)

        return SelfQuestion(
            question=question_text,
            question_type=question_type,
            domain=domain,
            difficulty=difficulty,
            expected_answer_length=self._estimate_answer_length(question_type)
        )

    def _fill_template(
        self,
        template: str,
        domain: str,
        question_type: QuestionType
    ) -> str:
        """Fill template placeholders with domain-specific content"""

        # Domain-specific concept banks
        concepts = {
            "programming": ["variables", "funciones", "clases", "herencia", "polimorfismo"],
            "architecture": ["Clean Architecture", "microservicios", "event sourcing", "CQRS"],
            "algorithms": ["búsqueda binaria", "ordenamiento", "grafos", "árboles"],
            "databases": ["índices", "transacciones", "normalización", "sharding"],
            "machine_learning": ["gradient descent", "backpropagation", "attention", "transformers"],
        }

        domain_concepts = concepts.get(domain, ["concepto A", "concepto B"])

        replacements = {
            "{concept}": random.choice(domain_concepts),
            "{concept_a}": random.choice(domain_concepts),
            "{concept_b}": random.choice([c for c in domain_concepts]),
            "{task}": f"implementar {random.choice(domain_concepts)}",
            "{problem}": f"optimizar {random.choice(domain_concepts)}",
            "{phenomenon}": f"el comportamiento de {random.choice(domain_concepts)}",
            "{effect}": f"mejora de rendimiento",
            "{outcome}": f"escalabilidad del sistema",
            "{scenario}": f"cambio de {random.choice(domain_concepts)}",
            "{change}": "modificación arquitectural",
            "{system}": "sistema distribuido",
            "{approach}": random.choice(domain_concepts),
            "{field}": domain,
            "{experience}": "implementación anterior",
            "{topic}": random.choice(domain_concepts),
            "{knowledge}": random.choice(domain_concepts),
            "{domain}": domain,
        }

        result = template
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, value)

        return result

    def _estimate_answer_length(self, question_type: QuestionType) -> int:
        """Estimate expected answer length in tokens"""
        lengths = {
            QuestionType.CONCEPTUAL: 150,
            QuestionType.PROCEDURAL: 300,
            QuestionType.COMPARATIVE: 200,
            QuestionType.CAUSAL: 180,
            QuestionType.EXPLORATORY: 250,
            QuestionType.REFLECTIVE: 200,
        }
        return lengths.get(question_type, 150)

    def generate_learning_session(
        self,
        num_questions: int = 10,
        focus_domain: str = None,
        difficulty_range: Tuple[int, int] = (3, 7)
    ) -> List[SelfQuestion]:
        """
        Generate a session of questions for learning

        Args:
            num_questions: Number of questions to generate
            focus_domain: Domain to focus on (None for mixed)
            difficulty_range: Min and max difficulty (1-10)

        Returns:
            List of SelfQuestion objects
        """
        questions = []

        for _ in range(num_questions):
            difficulty = random.randint(*difficulty_range)
            domain = focus_domain or random.choice(self.domains)
            question_type = random.choice(list(QuestionType))

            question = self.generate_question(question_type, domain, difficulty)
            questions.append(question)

        return questions

    def generate_curriculum(
        self,
        domain: str,
        phases: int = 3
    ) -> List[List[SelfQuestion]]:
        """
        Generate a progressive learning curriculum

        Args:
            domain: Domain to create curriculum for
            phases: Number of learning phases (increasing difficulty)

        Returns:
            List of question sessions, one per phase
        """
        curriculum = []

        for phase in range(phases):
            # Difficulty increases with each phase
            min_difficulty = 1 + (phase * 3)
            max_difficulty = 4 + (phase * 3)

            # More questions in advanced phases
            num_questions = 5 + (phase * 5)

            session = self.generate_learning_session(
                num_questions=num_questions,
                focus_domain=domain,
                difficulty_range=(min_difficulty, max_difficulty)
            )

            curriculum.append(session)

        return curriculum


def main():
    """Demo of self-questioning system"""
    engine = SelfQuestioningEngine()

    print("="*70)
    print("THAU Self-Questioning System - Demo")
    print("="*70)

    # Generate single questions of each type
    print("\n1. Ejemplos de cada tipo de pregunta:\n")
    for q_type in QuestionType:
        question = engine.generate_question(question_type=q_type, domain="programming")
        print(f"[{q_type.value.upper()}]")
        print(f"   {question.question}")
        print(f"   Domain: {question.domain}, Difficulty: {question.difficulty}\n")

    # Generate learning session
    print("\n2. Sesión de aprendizaje (10 preguntas):\n")
    session = engine.generate_learning_session(num_questions=10, focus_domain="architecture")
    for i, q in enumerate(session, 1):
        print(f"{i}. [{q.question_type.value}] {q.question}")

    # Generate curriculum
    print("\n\n3. Currículum progresivo (3 fases):\n")
    curriculum = engine.generate_curriculum(domain="machine_learning", phases=3)
    for phase_num, phase_questions in enumerate(curriculum, 1):
        print(f"\nFASE {phase_num} ({len(phase_questions)} preguntas):")
        for q in phase_questions[:3]:  # Show first 3
            print(f"  - {q.question} (diff: {q.difficulty})")
        if len(phase_questions) > 3:
            print(f"  ... y {len(phase_questions) - 3} preguntas más")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
