#!/usr/bin/env python3
"""
Test y Demo de THAU AGI

Script interactivo para probar las capacidades del sistema proto-AGI.

Uso:
    python scripts/test_thau_agi.py           # Demo completa
    python scripts/test_thau_agi.py --chat    # Modo chat interactivo
    python scripts/test_thau_agi.py --quick   # Tests rápidos sin modelo
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_experiential_learning():
    """Test del sistema de aprendizaje experiencial (sin modelo)"""
    print("\n" + "=" * 60)
    print("  TEST: Sistema de Aprendizaje Experiencial")
    print("=" * 60)

    from capabilities.proto_agi.experiential_learning import (
        ExperienceStore,
        MetacognitiveEngine,
        AdaptiveStrategy,
        OutcomeType,
        StrategyType,
        get_experience_store,
        get_metacognitive_engine,
        get_adaptive_strategy
    )

    # Inicializar componentes
    store = get_experience_store()
    metacog = get_metacognitive_engine()
    adaptive = get_adaptive_strategy()

    print("\n[1] Registrando experiencias de prueba...")

    # Simular varias experiencias
    test_cases = [
        {
            "goal": "Calcula 100 + 50",
            "strategy": StrategyType.TOOL_SINGLE,
            "tools": ["calculate"],
            "outcome": OutcomeType.SUCCESS,
            "confidence": 0.95,
            "time": 0.3,
            "worked": ["Uso directo de calculate"],
            "lessons": ["Cálculos simples = una herramienta"]
        },
        {
            "goal": "Lee config.py y explica",
            "strategy": StrategyType.TOOL_CHAIN,
            "tools": ["read_file"],
            "outcome": OutcomeType.PARTIAL_SUCCESS,
            "confidence": 0.7,
            "time": 1.5,
            "worked": ["Lectura exitosa"],
            "failed": ["Explicación incompleta"],
            "lessons": ["Combinar lectura con análisis"]
        },
        {
            "goal": "Ejecuta script de red",
            "strategy": StrategyType.TOOL_SINGLE,
            "tools": ["execute_python"],
            "outcome": OutcomeType.ERROR,
            "confidence": 0.2,
            "time": 30.0,
            "failed": ["Timeout", "Sin acceso a red"],
            "lessons": ["Verificar requisitos antes de ejecutar"]
        },
        {
            "goal": "¿Qué es Python?",
            "strategy": StrategyType.DIRECT,
            "tools": [],
            "outcome": OutcomeType.SUCCESS,
            "confidence": 0.85,
            "time": 0.8,
            "worked": ["Respuesta directa sin herramientas"],
            "lessons": ["Preguntas de conocimiento no necesitan herramientas"]
        },
        {
            "goal": "Lista archivos en src/",
            "strategy": StrategyType.TOOL_SINGLE,
            "tools": ["list_directory"],
            "outcome": OutcomeType.SUCCESS,
            "confidence": 0.9,
            "time": 0.2,
            "worked": ["list_directory efectivo"],
            "lessons": ["Exploración de directorios es rápida"]
        }
    ]

    for i, tc in enumerate(test_cases, 1):
        exp = adaptive.record_outcome(
            goal=tc["goal"],
            strategy=tc["strategy"],
            tools_used=tc["tools"],
            outcome=tc["outcome"],
            confidence=tc["confidence"],
            execution_time=tc["time"],
            what_worked=tc.get("worked", []),
            what_failed=tc.get("failed", []),
            lessons=tc.get("lessons", [])
        )
        status = "✓" if tc["outcome"] in (OutcomeType.SUCCESS, OutcomeType.PARTIAL_SUCCESS) else "✗"
        print(f"   {status} Experiencia {i}: {tc['goal'][:40]}... -> {tc['outcome'].value}")

    # Estadísticas
    print("\n[2] Estadísticas de experiencias:")
    stats = store.get_statistics()
    print(f"   Total: {stats['total_experiences']}")
    print(f"   Tasa de éxito: {stats['success_rate']:.0%}")
    print(f"   Confianza promedio: {stats['average_confidence']:.2f}")
    print(f"   Herramientas más usadas: {list(stats['most_used_tools'].items())[:3]}")

    # Selección de estrategia
    print("\n[3] Selección de estrategia adaptativa:")
    test_goals = [
        "Calcula el factorial de 5",
        "Lee el archivo main.py",
        "¿Cómo funciona async en Python?"
    ]

    for goal in test_goals:
        strategy, meta = adaptive.select_strategy(
            goal,
            ["calculate", "read_file", "execute_python", "list_directory"]
        )
        print(f"   Meta: {goal}")
        print(f"        → Estrategia: {strategy.value} (razón: {meta['reason']})")
        print(f"        → Confianza: {meta['confidence']:.0%}")
        print()

    # Evaluación metacognitiva
    print("[4] Evaluación metacognitiva:")
    test_responses = [
        ("Explica qué es una API", "Una API es una interfaz de programación de aplicaciones."),
        ("¿Qué hace este código?", "Creo que este código podría ser un servidor, pero no estoy seguro."),
        ("Calcula 10+5", "El resultado es definitivamente 15."),
    ]

    for goal, response in test_responses:
        eval_result = metacog.evaluate_response(goal, response)
        print(f"   Meta: {goal}")
        print(f"   Respuesta: {response[:50]}...")
        print(f"   → Confianza: {eval_result['confidence']:.2f} ({eval_result['confidence_level']})")
        print(f"   → Incertidumbre: {'Sí' if eval_result['uncertainty_detected'] else 'No'}")
        print()

    # Reflexión de sesión
    print("[5] Reflexión de sesión:")
    reflection = adaptive.end_session()
    summary = reflection.get("session_summary", {})
    print(f"   Interacciones: {summary.get('total_interactions', 0)}")
    print(f"   Éxitos: {summary.get('successes', 0)}")
    print(f"   Fracasos: {summary.get('failures', 0)}")
    print(f"   Tasa de éxito: {summary.get('success_rate', 0):.0%}")

    if reflection.get("recommendations"):
        print(f"   Recomendaciones:")
        for rec in reflection["recommendations"]:
            print(f"      - {rec}")

    print("\n" + "=" * 60)
    print("  TEST COMPLETADO: Sistema de aprendizaje funcional ✓")
    print("=" * 60)


def test_tools():
    """Test de herramientas (sin modelo)"""
    print("\n" + "=" * 60)
    print("  TEST: Herramientas del Sistema")
    print("=" * 60)

    from capabilities.proto_agi.thau_proto_agi import ThauTools

    # Test calculate
    print("\n[1] Herramienta: calculate")
    result = ThauTools.calculate("25 * 4 + 100")
    print(f"   25 * 4 + 100 = {result.output}")
    print(f"   Éxito: {result.success}")

    # Test list_directory
    print("\n[2] Herramienta: list_directory")
    result = ThauTools.list_directory(".")
    files = result.output.split("\n")[:5]
    print(f"   Primeros 5 items:")
    for f in files:
        print(f"      {f}")
    print(f"   Éxito: {result.success}")

    # Test read_file
    print("\n[3] Herramienta: read_file")
    result = ThauTools.read_file("README.md")
    if result.success:
        print(f"   Primeras 100 chars: {result.output[:100]}...")
    else:
        print(f"   Error: {result.error}")

    # Test execute_python
    print("\n[4] Herramienta: execute_python")
    result = ThauTools.execute_python("print('Hello from THAU!')")
    print(f"   Output: {result.output}")
    print(f"   Éxito: {result.success}")

    print("\n" + "=" * 60)
    print("  TEST COMPLETADO: Herramientas funcionales ✓")
    print("=" * 60)


def demo_full():
    """Demo completa con modelo"""
    print("\n" + "=" * 60)
    print("  DEMO: THAU AGI Completo (requiere modelo)")
    print("=" * 60)

    from capabilities.proto_agi import ThauAGI, AGIConfig

    config = AGIConfig(
        verbose=True,
        enable_learning=True,
        enable_metacognition=True
    )

    agent = ThauAGI(config)

    # Tests
    tests = [
        "Calcula cuanto es 15 multiplicado por 8",
        "Lista los archivos Python en el directorio actual",
        "¿Qué es una función recursiva?",
    ]

    for test in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {test}")
        print('='*60)
        result = agent.run(test)
        print(f"\nResultado: {result['goal_achieved']}")
        print(f"Confianza: {result['confidence']:.0%}")

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE SESIÓN")
    print("=" * 60)
    summary = agent.get_session_summary()
    print(f"Interacciones: {summary.get('session_summary', {}).get('total_interactions', 0)}")


def chat_mode():
    """Modo chat interactivo"""
    print("\n" + "=" * 60)
    print("  THAU AGI - Modo Chat Interactivo")
    print("=" * 60)
    print("  Escribe 'salir' o 'exit' para terminar")
    print("  Escribe 'stats' para ver estadísticas")
    print("=" * 60)

    from capabilities.proto_agi import ThauAGI, AGIConfig

    config = AGIConfig(
        verbose=False,  # Menos verbose en modo chat
        enable_learning=True,
        enable_metacognition=True
    )

    agent = ThauAGI(config)

    while True:
        try:
            user_input = input("\n[Tú]: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ('salir', 'exit', 'quit', 'q'):
                print("\n[THAU]: ¡Hasta luego! Generando resumen de sesión...")
                summary = agent.get_session_summary()
                if 'session_summary' in summary:
                    s = summary['session_summary']
                    print(f"\nResumen: {s.get('total_interactions', 0)} interacciones, "
                          f"{s.get('success_rate', 0):.0%} éxito")
                break

            if user_input.lower() == 'stats':
                stats = agent.get_statistics()
                print(f"\n[Stats] Total experiencias: {stats['total_experiences']}")
                print(f"[Stats] Tasa de éxito: {stats['success_rate']:.0%}")
                continue

            # Procesar mensaje
            result = agent.run(user_input)

            print(f"\n[THAU]: {result['response']}")
            print(f"        (confianza: {result['confidence']:.0%}, "
                  f"herramientas: {result['tools_used'] or 'ninguna'})")

        except KeyboardInterrupt:
            print("\n\n[THAU]: Sesión interrumpida. ¡Hasta pronto!")
            break
        except Exception as e:
            print(f"\n[Error]: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test y Demo de THAU AGI")
    parser.add_argument("--chat", action="store_true", help="Modo chat interactivo")
    parser.add_argument("--quick", action="store_true", help="Tests rápidos sin modelo")
    parser.add_argument("--tools", action="store_true", help="Test solo herramientas")
    parser.add_argument("--learning", action="store_true", help="Test solo aprendizaje")
    args = parser.parse_args()

    if args.chat:
        chat_mode()
    elif args.quick or args.tools or args.learning:
        if args.tools or args.quick:
            test_tools()
        if args.learning or args.quick:
            test_experiential_learning()
    else:
        # Demo completa
        test_tools()
        test_experiential_learning()
        print("\n¿Deseas ejecutar la demo completa con el modelo? (requiere GPU/tiempo)")
        response = input("Escribe 'si' para continuar: ").strip().lower()
        if response in ('si', 'sí', 'yes', 'y'):
            demo_full()
        else:
            print("Demo completa omitida.")


if __name__ == "__main__":
    main()
