#!/usr/bin/env python3
"""
THAU Planner - Sistema de Planificación Inteligente
Mis Mejores Prácticas como Claude Code

Este es el sistema que me permite planificar tareas complejas.
Ahora THAU tendrá esta misma capacidad.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
from pathlib import Path


class TaskComplexity(Enum):
    """Complejidad de una tarea"""
    TRIVIAL = "trivial"          # 1 paso simple
    SIMPLE = "simple"            # 2-3 pasos
    MODERATE = "moderate"        # 4-7 pasos
    COMPLEX = "complex"          # 8-15 pasos
    VERY_COMPLEX = "very_complex"  # 15+ pasos


class TaskPriority(Enum):
    """Prioridad de una tarea"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class PlanStep:
    """
    Un paso en el plan

    Basado en cómo yo (Claude Code) descompongo tareas
    """
    step_number: int
    description: str
    action_type: str  # "read", "write", "edit", "bash", "think", "tool_use"
    estimated_effort: str  # "low", "medium", "high"
    dependencies: List[int] = field(default_factory=list)  # Depende de qué pasos
    tools_needed: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Any] = None


@dataclass
class Plan:
    """
    Plan completo para una tarea

    Estructura similar a cómo yo planifico internamente
    """
    task_description: str
    complexity: TaskComplexity
    priority: TaskPriority
    steps: List[PlanStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    estimated_total_time: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)


class ThauPlanner:
    """
    Sistema de Planificación de THAU

    Inspirado en mis capacidades (Claude Code):
    1. Analizar tareas complejas
    2. Descomponerlas en pasos manejables
    3. Identificar dependencias
    4. Priorizar acciones
    5. Gestionar riesgos
    """

    def __init__(self):
        print("🧠 THAU Planner inicializado")
        print("   Capacidades:")
        print("   - Task decomposition")
        print("   - Dependency analysis")
        print("   - Risk assessment")
        print("   - Priority management")

    def analyze_task(self, description: str) -> Dict[str, Any]:
        """
        Analiza una tarea para entender su complejidad

        Args:
            description: Descripción de la tarea

        Returns:
            Análisis de la tarea
        """
        # Keywords que indican complejidad
        complex_keywords = [
            "sistema", "arquitectura", "integrar", "múltiple",
            "completo", "end-to-end", "pipeline", "automatizar"
        ]

        moderate_keywords = [
            "implementar", "crear", "desarrollar", "añadir",
            "modificar", "refactorizar", "optimizar"
        ]

        simple_keywords = [
            "leer", "mostrar", "listar", "ver", "revisar",
            "explicar", "describir"
        ]

        desc_lower = description.lower()

        # Determina complejidad
        if any(kw in desc_lower for kw in complex_keywords):
            complexity = TaskComplexity.VERY_COMPLEX if len(desc_lower) > 100 else TaskComplexity.COMPLEX
        elif any(kw in desc_lower for kw in moderate_keywords):
            complexity = TaskComplexity.MODERATE
        elif any(kw in desc_lower for kw in simple_keywords):
            complexity = TaskComplexity.SIMPLE
        else:
            complexity = TaskComplexity.SIMPLE

        # Estima pasos necesarios
        estimated_steps = {
            TaskComplexity.TRIVIAL: 1,
            TaskComplexity.SIMPLE: 3,
            TaskComplexity.MODERATE: 6,
            TaskComplexity.COMPLEX: 12,
            TaskComplexity.VERY_COMPLEX: 20
        }[complexity]

        return {
            "complexity": complexity,
            "estimated_steps": estimated_steps,
            "keywords_found": [kw for kw in (complex_keywords + moderate_keywords + simple_keywords) if kw in desc_lower]
        }

    def create_plan(
        self,
        task_description: str,
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> Plan:
        """
        Crea un plan detallado para una tarea

        Esta es mi metodología (Claude Code):
        1. Analizar la tarea
        2. Identificar pasos clave
        3. Determinar dependencias
        4. Estimar esfuerzo
        5. Identificar riesgos

        Args:
            task_description: Descripción de la tarea
            priority: Prioridad de la tarea

        Returns:
            Plan completo
        """
        print(f"\n{'='*70}")
        print(f"📋 Creando Plan")
        print(f"{'='*70}")
        print(f"Tarea: {task_description}")

        # Fase 1: Análisis
        analysis = self.analyze_task(task_description)
        complexity = analysis["complexity"]

        print(f"\n📊 Análisis:")
        print(f"   Complejidad: {complexity.value}")
        print(f"   Pasos estimados: {analysis['estimated_steps']}")

        # Fase 2: Descomposición
        plan = Plan(
            task_description=task_description,
            complexity=complexity,
            priority=priority
        )

        # Genera pasos basado en el tipo de tarea
        steps = self._decompose_task(task_description, complexity)
        plan.steps = steps

        # Fase 3: Identificar riesgos
        plan.risks = self._identify_risks(task_description, complexity)

        # Fase 4: Documentar suposiciones
        plan.assumptions = self._identify_assumptions(task_description)

        print(f"\n✅ Plan creado con {len(plan.steps)} pasos")

        return plan

    def _decompose_task(
        self,
        task: str,
        complexity: TaskComplexity
    ) -> List[PlanStep]:
        """
        Descompone tarea en pasos

        Esto simula cómo yo analizo y descompongo tareas.
        En producción, THAU-2B generaría estos pasos.
        """
        steps = []
        task_lower = task.lower()

        # Paso 1: Siempre empezar entendiendo el contexto
        steps.append(PlanStep(
            step_number=1,
            description="Analizar requisitos y contexto",
            action_type="think",
            estimated_effort="low",
            tools_needed=["read", "grep"]
        ))

        # Paso 2: Investigación si es necesario
        if "crear" in task_lower or "implementar" in task_lower or "añadir" in task_lower:
            steps.append(PlanStep(
                step_number=2,
                description="Investigar código/arquitectura existente",
                action_type="read",
                estimated_effort="medium",
                dependencies=[1],
                tools_needed=["read", "grep", "glob"]
            ))

        # Paso 3: Diseño/Planificación
        if complexity in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]:
            steps.append(PlanStep(
                step_number=len(steps) + 1,
                description="Diseñar arquitectura de la solución",
                action_type="think",
                estimated_effort="medium",
                dependencies=[len(steps)],
                tools_needed=[]
            ))

        # Paso 4: Implementación core
        if "crear" in task_lower or "escribir" in task_lower or "implementar" in task_lower:
            steps.append(PlanStep(
                step_number=len(steps) + 1,
                description="Implementar funcionalidad core",
                action_type="write",
                estimated_effort="high",
                dependencies=[len(steps)],
                tools_needed=["write", "edit"]
            ))

        # Paso 5: Integración
        if "integrar" in task_lower or "conectar" in task_lower:
            steps.append(PlanStep(
                step_number=len(steps) + 1,
                description="Integrar con sistemas existentes",
                action_type="edit",
                estimated_effort="medium",
                dependencies=[len(steps)],
                tools_needed=["edit", "read"]
            ))

        # Paso 6: Testing
        if complexity in [TaskComplexity.MODERATE, TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]:
            steps.append(PlanStep(
                step_number=len(steps) + 1,
                description="Probar funcionalidad",
                action_type="bash",
                estimated_effort="medium",
                dependencies=[len(steps)],
                tools_needed=["bash"]
            ))

        # Paso 7: Documentación
        if "crear" in task_lower or "implementar" in task_lower:
            steps.append(PlanStep(
                step_number=len(steps) + 1,
                description="Documentar cambios",
                action_type="write",
                estimated_effort="low",
                dependencies=[len(steps)],
                tools_needed=["write", "edit"]
            ))

        # Si no hay pasos específicos, crear plan genérico
        if len(steps) == 1:
            steps.append(PlanStep(
                step_number=2,
                description="Ejecutar tarea",
                action_type="tool_use",
                estimated_effort="medium",
                dependencies=[1]
            ))

            steps.append(PlanStep(
                step_number=3,
                description="Verificar resultado",
                action_type="think",
                estimated_effort="low",
                dependencies=[2]
            ))

        return steps

    def _identify_risks(
        self,
        task: str,
        complexity: TaskComplexity
    ) -> List[str]:
        """Identifica riesgos potenciales"""
        risks = []
        task_lower = task.lower()

        if "integrar" in task_lower or "api" in task_lower:
            risks.append("Dependencias externas pueden fallar")
            risks.append("Autenticación/autorización puede ser compleja")

        if "database" in task_lower or "datos" in task_lower:
            risks.append("Migración de datos puede requerir tiempo")
            risks.append("Backup necesario antes de cambios")

        if complexity in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]:
            risks.append("Tarea compleja puede requerir múltiples iteraciones")
            risks.append("Testing exhaustivo necesario")

        if "seguridad" in task_lower or "security" in task_lower:
            risks.append("Implicaciones de seguridad críticas")
            risks.append("Revisión de seguridad necesaria")

        return risks

    def _identify_assumptions(self, task: str) -> List[str]:
        """Identifica suposiciones"""
        assumptions = []
        task_lower = task.lower()

        if "crear" in task_lower:
            assumptions.append("Tengo permisos de escritura en directorios necesarios")

        if "integrar" in task_lower or "api" in task_lower:
            assumptions.append("APIs externas están disponibles y documentadas")
            assumptions.append("Credenciales de API están disponibles")

        if "modificar" in task_lower or "actualizar" in task_lower:
            assumptions.append("Código existente está disponible")
            assumptions.append("Tests existentes pueden validar cambios")

        return assumptions

    def execute_plan(self, plan: Plan, executor_func: Optional[callable] = None) -> Dict[str, Any]:
        """
        Ejecuta un plan paso a paso

        Args:
            plan: Plan a ejecutar
            executor_func: Función que ejecuta cada paso (None = simular)

        Returns:
            Resultado de la ejecución
        """
        print(f"\n{'='*70}")
        print(f"▶️  Ejecutando Plan")
        print(f"{'='*70}")
        print(f"Tarea: {plan.task_description}")
        print(f"Pasos: {len(plan.steps)}")

        results = []

        for step in plan.steps:
            print(f"\n{'─'*70}")
            print(f"Paso {step.step_number}: {step.description}")
            print(f"   Tipo: {step.action_type}")
            print(f"   Esfuerzo: {step.estimated_effort}")

            # Verifica dependencias
            for dep in step.dependencies:
                dep_step = plan.steps[dep - 1]
                if dep_step.status != "completed":
                    print(f"   ⚠️  Esperando paso {dep}: {dep_step.description}")
                    step.status = "blocked"
                    continue

            step.status = "in_progress"

            if executor_func:
                # Ejecutor real
                try:
                    result = executor_func(step)
                    step.result = result
                    step.status = "completed"
                    results.append(result)
                    print(f"   ✅ Completado")
                except Exception as e:
                    step.status = "failed"
                    step.result = {"error": str(e)}
                    print(f"   ❌ Fallido: {e}")
                    break
            else:
                # Simulación
                step.status = "completed"
                step.result = {"simulated": True}
                results.append(step.result)
                print(f"   ✅ Completado (simulado)")

        print(f"\n{'='*70}")
        print(f"{'✅' if all(s.status == 'completed' for s in plan.steps) else '⚠️'} Ejecución Finalizada")
        print(f"{'='*70}")

        return {
            "plan": plan.task_description,
            "total_steps": len(plan.steps),
            "completed_steps": sum(1 for s in plan.steps if s.status == "completed"),
            "failed_steps": sum(1 for s in plan.steps if s.status == "failed"),
            "results": results
        }

    def save_plan(self, plan: Plan, filepath: str = "data/plans/"):
        """Guarda plan a disco"""
        Path(filepath).mkdir(parents=True, exist_ok=True)

        filename = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        full_path = Path(filepath) / filename

        plan_dict = {
            "task": plan.task_description,
            "complexity": plan.complexity.value,
            "priority": plan.priority.value,
            "created_at": plan.created_at.isoformat(),
            "steps": [
                {
                    "step": s.step_number,
                    "description": s.description,
                    "action_type": s.action_type,
                    "effort": s.estimated_effort,
                    "dependencies": s.dependencies,
                    "tools": s.tools_needed,
                    "status": s.status
                }
                for s in plan.steps
            ],
            "risks": plan.risks,
            "assumptions": plan.assumptions
        }

        with open(full_path, 'w') as f:
            json.dump(plan_dict, f, indent=2)

        print(f"💾 Plan guardado: {full_path}")

    def print_plan(self, plan: Plan):
        """Imprime plan de forma legible"""
        print(f"\n{'='*70}")
        print(f"📋 PLAN: {plan.task_description}")
        print(f"{'='*70}")
        print(f"Complejidad: {plan.complexity.value}")
        print(f"Prioridad: {plan.priority.value}")
        print(f"Pasos: {len(plan.steps)}")

        print(f"\n{'─'*70}")
        print(f"PASOS DEL PLAN")
        print(f"{'─'*70}")

        for step in plan.steps:
            deps_str = f" (depends on: {', '.join(map(str, step.dependencies))})" if step.dependencies else ""
            print(f"\n{step.step_number}. {step.description}{deps_str}")
            print(f"   Action: {step.action_type}")
            print(f"   Effort: {step.estimated_effort}")
            print(f"   Tools: {', '.join(step.tools_needed) if step.tools_needed else 'none'}")
            print(f"   Status: {step.status}")

        if plan.risks:
            print(f"\n{'─'*70}")
            print(f"⚠️  RIESGOS")
            print(f"{'─'*70}")
            for risk in plan.risks:
                print(f"  - {risk}")

        if plan.assumptions:
            print(f"\n{'─'*70}")
            print(f"📝 SUPOSICIONES")
            print(f"{'─'*70}")
            for assumption in plan.assumptions:
                print(f"  - {assumption}")

        print(f"\n{'='*70}")


if __name__ == "__main__":
    print("="*70)
    print("🧪 Testing THAU Planner")
    print("="*70)

    planner = ThauPlanner()

    # Test 1: Tarea simple
    print("\n" + "="*70)
    print("Test 1: Tarea Simple")
    print("="*70)

    plan1 = planner.create_plan(
        "Leer el archivo de configuración",
        priority=TaskPriority.LOW
    )

    planner.print_plan(plan1)

    # Test 2: Tarea moderada
    print("\n" + "="*70)
    print("Test 2: Tarea Moderada")
    print("="*70)

    plan2 = planner.create_plan(
        "Implementar función de autenticación con JWT",
        priority=TaskPriority.HIGH
    )

    planner.print_plan(plan2)

    # Test 3: Tarea compleja
    print("\n" + "="*70)
    print("Test 3: Tarea Compleja")
    print("="*70)

    plan3 = planner.create_plan(
        "Crear sistema completo de agentes con integración API REST, calendario y generación de imágenes",
        priority=TaskPriority.CRITICAL
    )

    planner.print_plan(plan3)

    # Test 4: Ejecución de plan
    print("\n" + "="*70)
    print("Test 4: Ejecución de Plan")
    print("="*70)

    result = planner.execute_plan(plan1)

    print(f"\n📊 Resultado:")
    print(json.dumps(result, indent=2))

    # Test 5: Guardar plan
    print("\n" + "="*70)
    print("Test 5: Guardar Plan")
    print("="*70)

    planner.save_plan(plan3)

    print("\n" + "="*70)
    print("✅ Tests Completados")
    print("="*70)
