"""
Tree of Thoughts (ToT) Reasoning
Explora múltiples caminos de razonamiento en paralelo
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class ThoughtNode:
    """Nodo en el árbol de pensamientos"""
    thought: str
    depth: int
    score: float
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class TreeOfThoughts:
    """
    Implementa razonamiento Tree-of-Thoughts

    Genera múltiples líneas de razonamiento en paralelo,
    evalúa cada una, y selecciona el mejor camino.
    """

    def __init__(self, llm_client=None, branching_factor: int = 3, max_depth: int = 3):
        """
        Args:
            llm_client: Cliente del modelo LLM
            branching_factor: Número de ramas por nodo
            max_depth: Profundidad máxima del árbol
        """
        self.llm_client = llm_client
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.thought_trees = []

    def explore_thoughts(self, question: str) -> Dict:
        """
        Explora múltiples caminos de pensamiento

        Args:
            question: Pregunta a analizar

        Returns:
            Dict con árbol de pensamientos y mejor camino
        """
        # Crear nodo raíz
        root = ThoughtNode(
            thought=f"Analizando: {question}",
            depth=0,
            score=1.0
        )

        # Generar árbol de pensamientos
        self._generate_tree(root, question)

        # Encontrar mejor camino
        best_path = self._find_best_path(root)

        result = {
            "question": question,
            "root_thought": root.thought,
            "total_nodes": self._count_nodes(root),
            "best_path": [node.thought for node in best_path],
            "best_score": best_path[-1].score if best_path else 0.0,
            "final_answer": best_path[-1].thought if best_path else "No se encontró respuesta",
            "timestamp": datetime.now().isoformat(),
            "method": "tree_of_thoughts"
        }

        self.thought_trees.append(result)

        return result

    def _generate_tree(self, node: ThoughtNode, question: str):
        """Genera árbol de pensamientos recursivamente"""
        if node.depth >= self.max_depth:
            return

        # Generar pensamientos hijo
        for i in range(self.branching_factor):
            child_thought = self._generate_thought(node, question, i)
            child_score = self._evaluate_thought(child_thought, question)

            child = ThoughtNode(
                thought=child_thought,
                depth=node.depth + 1,
                score=child_score,
                parent=node
            )

            node.children.append(child)

            # Continuar explorando solo los mejores
            if child_score > 0.6:
                self._generate_tree(child, question)

    def _generate_thought(self, parent: ThoughtNode, question: str, branch_idx: int) -> str:
        """Genera un pensamiento hijo"""
        # Placeholder - se conectará con el modelo LLM
        thought_templates = [
            f"Considerando el aspecto {['técnico', 'práctico', 'teórico'][branch_idx]}...",
            f"Desde la perspectiva {['científica', 'filosófica', 'pragmática'][branch_idx]}...",
            f"Analizando el {['contexto', 'fundamento', 'impacto'][branch_idx]}..."
        ]

        return thought_templates[branch_idx % 3]

    def _evaluate_thought(self, thought: str, question: str) -> float:
        """Evalúa la calidad de un pensamiento"""
        # Placeholder - se conectará con el modelo LLM
        # Por ahora, score aleatorio simulado
        import random
        return random.uniform(0.5, 1.0)

    def _find_best_path(self, root: ThoughtNode) -> List[ThoughtNode]:
        """Encuentra el mejor camino en el árbol"""
        if not root.children:
            return [root]

        best_child_path = max(
            [self._find_best_path(child) for child in root.children],
            key=lambda path: sum(node.score for node in path) / len(path)
        )

        return [root] + best_child_path

    def _count_nodes(self, node: ThoughtNode) -> int:
        """Cuenta nodos en el árbol"""
        return 1 + sum(self._count_nodes(child) for child in node.children)

    def visualize_tree(self, result: Dict) -> str:
        """Visualiza el árbol de pensamientos"""
        output = f"Árbol de Pensamientos para: {result['question']}\n"
        output += f"Total de nodos explorados: {result['total_nodes']}\n\n"
        output += "Mejor camino encontrado:\n"

        for i, thought in enumerate(result['best_path'], 1):
            output += f"  {i}. {thought}\n"

        output += f"\nPuntuación: {result['best_score']:.2f}"

        return output


if __name__ == "__main__":
    # Testing
    print("🌲 Testing Tree of Thoughts Reasoning\n")

    tot = TreeOfThoughts(branching_factor=2, max_depth=2)

    question = "¿Cómo puedo mejorar mi productividad?"

    print(f"Pregunta: {question}\n")

    result = tot.explore_thoughts(question)

    print(tot.visualize_tree(result))
    print(f"\nRespuesta final: {result['final_answer']}")
