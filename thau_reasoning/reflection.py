"""
Self-Reflection and Critique
THAU puede reflexionar sobre sus propias respuestas y mejorarlas
"""

from typing import List, Dict, Optional
from datetime import datetime
import json


class SelfReflection:
    """
    Sistema de auto-reflexión para THAU

    Permite al modelo criticar y mejorar sus propias respuestas
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.reflection_history = []

    def reflect_on_response(self, question: str, response: str, criteria: Optional[List[str]] = None) -> Dict:
        """
        Reflexiona sobre una respuesta generada

        Args:
            question: Pregunta original
            response: Respuesta generada
            criteria: Criterios de evaluación

        Returns:
            Análisis de la respuesta con sugerencias de mejora
        """
        if criteria is None:
            criteria = [
                "accuracy",      # Precisión
                "completeness",  # Completitud
                "clarity",       # Claridad
                "relevance",     # Relevancia
                "coherence"      # Coherencia
            ]

        # Evaluar respuesta según cada criterio
        evaluation = {}
        for criterion in criteria:
            evaluation[criterion] = self._evaluate_criterion(response, criterion)

        # Identificar problemas
        issues = self._identify_issues(evaluation)

        # Generar sugerencias de mejora
        suggestions = self._generate_suggestions(response, issues)

        reflection = {
            "question": question,
            "original_response": response,
            "evaluation": evaluation,
            "overall_score": sum(evaluation.values()) / len(evaluation),
            "issues": issues,
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat()
        }

        self.reflection_history.append(reflection)

        return reflection

    def _evaluate_criterion(self, response: str, criterion: str) -> float:
        """Evalúa un criterio específico (0.0 - 1.0)"""
        # Placeholder - se conectará con el modelo LLM
        scores = {
            "accuracy": 0.85,
            "completeness": 0.75,
            "clarity": 0.90,
            "relevance": 0.80,
            "coherence": 0.88
        }

        return scores.get(criterion, 0.5)

    def _identify_issues(self, evaluation: Dict[str, float]) -> List[Dict]:
        """Identifica problemas en la respuesta"""
        issues = []

        for criterion, score in evaluation.items():
            if score < 0.7:
                issues.append({
                    "criterion": criterion,
                    "severity": "high" if score < 0.5 else "medium",
                    "score": score,
                    "description": f"La respuesta tiene problemas de {criterion}"
                })

        return issues

    def _generate_suggestions(self, response: str, issues: List[Dict]) -> List[str]:
        """Genera sugerencias de mejora"""
        suggestions = []

        for issue in issues:
            criterion = issue["criterion"]

            if criterion == "accuracy":
                suggestions.append("Verificar hechos y datos incluidos en la respuesta")
            elif criterion == "completeness":
                suggestions.append("Ampliar la respuesta con más detalles o ejemplos")
            elif criterion == "clarity":
                suggestions.append("Simplificar el lenguaje y estructura de la respuesta")
            elif criterion == "relevance":
                suggestions.append("Enfocarse más en el tema central de la pregunta")
            elif criterion == "coherence":
                suggestions.append("Mejorar la conexión lógica entre ideas")

        return suggestions

    def improve_response(self, reflection: Dict) -> str:
        """
        Genera una versión mejorada de la respuesta

        Args:
            reflection: Resultado de reflect_on_response

        Returns:
            Respuesta mejorada
        """
        # Placeholder - se conectará con el modelo LLM
        original = reflection["original_response"]
        suggestions = reflection["suggestions"]

        improved = f"{original}\n\n[Versión mejorada considerando: {', '.join(suggestions)}]"

        return improved

    def self_critique_loop(self, question: str, initial_response: str, max_iterations: int = 3) -> Dict:
        """
        Ciclo de auto-crítica y mejora iterativa

        Args:
            question: Pregunta original
            initial_response: Respuesta inicial
            max_iterations: Máximo de iteraciones de mejora

        Returns:
            Historial de mejoras y respuesta final
        """
        iterations = []
        current_response = initial_response

        for i in range(max_iterations):
            # Reflexionar sobre la respuesta actual
            reflection = self.reflect_on_response(question, current_response)

            # Si la respuesta es suficientemente buena, parar
            if reflection["overall_score"] >= 0.85:
                iterations.append({
                    "iteration": i + 1,
                    "response": current_response,
                    "score": reflection["overall_score"],
                    "stopped_reason": "quality_threshold_reached"
                })
                break

            # Generar respuesta mejorada
            improved_response = self.improve_response(reflection)

            iterations.append({
                "iteration": i + 1,
                "response": current_response,
                "score": reflection["overall_score"],
                "issues": reflection["issues"],
                "suggestions": reflection["suggestions"]
            })

            current_response = improved_response

        return {
            "question": question,
            "initial_response": initial_response,
            "iterations": iterations,
            "final_response": current_response,
            "improvement_score": iterations[-1]["score"] - iterations[0]["score"] if iterations else 0
        }


if __name__ == "__main__":
    # Testing
    print("🔍 Testing Self-Reflection\n")

    reflector = SelfReflection()

    question = "¿Qué es Python?"
    response = "Python es un lenguaje de programación."

    print(f"Pregunta: {question}")
    print(f"Respuesta: {response}\n")

    reflection = reflector.reflect_on_response(question, response)

    print("Evaluación por criterio:")
    for criterion, score in reflection["evaluation"].items():
        print(f"  {criterion}: {score:.2f}")

    print(f"\nPuntuación general: {reflection['overall_score']:.2f}")

    if reflection["issues"]:
        print("\nProblemas identificados:")
        for issue in reflection["issues"]:
            print(f"  - {issue['description']} (severidad: {issue['severity']})")

    if reflection["suggestions"]:
        print("\nSugerencias de mejora:")
        for suggestion in reflection["suggestions"]:
            print(f"  - {suggestion}")

    print("\n" + "="*70)
    print("\nCiclo de auto-mejora:")
    result = reflector.self_critique_loop(question, response, max_iterations=2)

    for iteration in result["iterations"]:
        print(f"\nIteración {iteration['iteration']}")
        print(f"Respuesta: {iteration['response']}")
        print(f"Score: {iteration['score']:.2f}")

    print(f"\nRespuesta final: {result['final_response']}")
    print(f"Mejora total: {result['improvement_score']:.2f}")
