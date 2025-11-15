"""
Task Planning and Decomposition
THAU puede planificar y descomponer tareas complejas
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class Task:
    """Representa una tarea en el plan"""
    id: str
    description: str
    dependencies: List[str]
    status: str = "pending"  # pending, in_progress, completed
    priority: int = 0
    estimated_effort: str = "unknown"


class TaskPlanner:
    """
    Planificador de tareas para THAU

    Descompone tareas complejas en sub-tareas manejables
    y crea planes de ejecución
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.plans = []

    def create_plan(self, goal: str, context: Optional[Dict] = None) -> Dict:
        """
        Crea un plan para lograr un objetivo

        Args:
            goal: Objetivo a lograr
            context: Contexto adicional (recursos, restricciones, etc.)

        Returns:
            Plan estructurado con tareas y dependencias
        """
        # Analizar el objetivo
        analysis = self._analyze_goal(goal, context)

        # Generar tareas
        tasks = self._generate_tasks(goal, analysis)

        # Determinar dependencias
        tasks = self._determine_dependencies(tasks)

        # Priorizar tareas
        tasks = self._prioritize_tasks(tasks)

        plan = {
            "goal": goal,
            "created_at": datetime.now().isoformat(),
            "total_tasks": len(tasks),
            "tasks": [self._task_to_dict(t) for t in tasks],
            "estimated_duration": self._estimate_duration(tasks),
            "critical_path": self._find_critical_path(tasks),
            "status": "created"
        }

        self.plans.append(plan)

        return plan

    def _analyze_goal(self, goal: str, context: Optional[Dict]) -> Dict:
        """Analiza el objetivo para entender qué se necesita"""
        return {
            "complexity": "medium",
            "domain": "general",
            "requires_resources": [],
            "constraints": []
        }

    def _generate_tasks(self, goal: str, analysis: Dict) -> List[Task]:
        """Genera tareas necesarias para el objetivo"""
        # Placeholder - se conectará con el modelo LLM
        tasks = [
            Task("1", "Entender el problema", [], "pending", 1, "1 hora"),
            Task("2", "Investigar soluciones existentes", ["1"], "pending", 2, "2 horas"),
            Task("3", "Diseñar solución", ["2"], "pending", 1, "3 horas"),
            Task("4", "Implementar solución", ["3"], "pending", 0, "5 horas"),
            Task("5", "Probar solución", ["4"], "pending", 1, "2 horas"),
            Task("6", "Documentar resultado", ["5"], "pending", 2, "1 hora"),
        ]

        return tasks

    def _determine_dependencies(self, tasks: List[Task]) -> List[Task]:
        """Determina dependencias entre tareas"""
        # Ya están definidas en este ejemplo
        return tasks

    def _prioritize_tasks(self, tasks: List[Task]) -> List[Task]:
        """Prioriza tareas según importancia y dependencias"""
        # Ordenar por prioridad (0 = más alta)
        return sorted(tasks, key=lambda t: (t.priority, t.id))

    def _estimate_duration(self, tasks: List[Task]) -> str:
        """Estima duración total del plan"""
        return "1-2 días"

    def _find_critical_path(self, tasks: List[Task]) -> List[str]:
        """Encuentra la ruta crítica del plan"""
        # Simplificado - retorna todas las tareas con dependencias
        return [t.id for t in tasks if t.dependencies]

    def _task_to_dict(self, task: Task) -> Dict:
        """Convierte Task a diccionario"""
        return {
            "id": task.id,
            "description": task.description,
            "dependencies": task.dependencies,
            "status": task.status,
            "priority": task.priority,
            "estimated_effort": task.estimated_effort
        }

    def execute_plan(self, plan: Dict, executor_fn=None) -> Dict:
        """
        Ejecuta un plan (simulado por ahora)

        Args:
            plan: Plan a ejecutar
            executor_fn: Función que ejecuta cada tarea

        Returns:
            Resultado de la ejecución
        """
        results = {
            "plan_id": id(plan),
            "started_at": datetime.now().isoformat(),
            "completed_tasks": 0,
            "failed_tasks": 0,
            "task_results": []
        }

        for task in plan["tasks"]:
            if self._can_execute(task, results):
                result = self._execute_task(task, executor_fn)
                results["task_results"].append(result)

                if result["success"]:
                    results["completed_tasks"] += 1
                else:
                    results["failed_tasks"] += 1

        results["completed_at"] = datetime.now().isoformat()
        results["success"] = results["failed_tasks"] == 0

        return results

    def _can_execute(self, task: Dict, results: Dict) -> bool:
        """Verifica si una tarea puede ejecutarse"""
        # Verificar que las dependencias estén completadas
        completed_ids = {r["task_id"] for r in results["task_results"] if r["success"]}

        for dep_id in task["dependencies"]:
            if dep_id not in completed_ids:
                return False

        return True

    def _execute_task(self, task: Dict, executor_fn=None) -> Dict:
        """Ejecuta una tarea"""
        if executor_fn:
            return executor_fn(task)

        # Ejecución simulada
        return {
            "task_id": task["id"],
            "success": True,
            "output": f"Tarea '{task['description']}' completada",
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Testing
    print("📋 Testing Task Planner\n")

    planner = TaskPlanner()

    goal = "Crear una aplicación web para gestión de tareas"

    print(f"Objetivo: {goal}\n")

    plan = planner.create_plan(goal)

    print(f"Plan creado con {plan['total_tasks']} tareas:")
    print(f"Duración estimada: {plan['estimated_duration']}\n")

    print("Tareas:")
    for task in plan['tasks']:
        deps = f" (depende de: {', '.join(task['dependencies'])})" if task['dependencies'] else ""
        print(f"  {task['id']}. {task['description']}{deps}")
        print(f"      Prioridad: {task['priority']}, Esfuerzo: {task['estimated_effort']}")

    print(f"\nRuta crítica: {' → '.join(plan['critical_path'])}")
