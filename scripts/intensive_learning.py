#!/usr/bin/env python3
"""
Script para sesiones de aprendizaje intensivo de THAU
Permite que THAU aprenda rápidamente haciendo muchas preguntas
"""

import argparse
import sys
from pathlib import Path

# Agregar path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from thau_trainer.self_questioning import SelfQuestioningSystem


def main():
    parser = argparse.ArgumentParser(
        description="Sesión de aprendizaje intensivo para THAU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Aprender 100 preguntas generales
  python scripts/intensive_learning.py --questions 100

  # Explorar categoría específica
  python scripts/intensive_learning.py --category ia_ml --questions 50

  # Aprendizaje profundo con modelos potentes
  python scripts/intensive_learning.py --questions 200 --model best --age 6

  # Listar categorías disponibles
  python scripts/intensive_learning.py --list-categories

Categorías disponibles:
  - programacion_basica
  - estructuras_datos
  - algoritmos
  - bases_datos
  - web
  - cloud_devops
  - seguridad
  - ia_ml
  - arquitectura
  - sistemas
  - redes
  - testing
        """
    )

    parser.add_argument(
        "--questions", "-n",
        type=int,
        default=50,
        help="Número de preguntas a generar (default: 50)"
    )

    parser.add_argument(
        "--category", "-c",
        type=str,
        default=None,
        help="Categoría específica a explorar"
    )

    parser.add_argument(
        "--age",
        type=int,
        default=3,
        help="Edad cognitiva de THAU (0-15, default: 3)"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="auto",
        choices=["auto", "random", "best", "ollama", "deepseek", "llama", "mistral", "phi3", "gpt_oss", "coder", "gemini"],
        help="Modelo a usar para respuestas (default: auto)"
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Segundos entre preguntas (default: 2.0)"
    )

    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="Listar categorías disponibles y salir"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Mostrar estadísticas del sistema y salir"
    )

    args = parser.parse_args()

    # Inicializar sistema
    system = SelfQuestioningSystem()

    # Listar categorías
    if args.list_categories:
        print("\n📂 Categorías disponibles:")
        stats = system.get_knowledge_stats()
        for cat, size in stats['category_sizes'].items():
            print(f"   - {cat}: {size} conceptos")
        return

    # Mostrar estadísticas
    if args.stats:
        print("\n📊 Estadísticas del sistema de THAU:")
        knowledge = system.get_knowledge_stats()
        activity = system.get_stats()

        print(f"\n📚 Base de conocimiento:")
        print(f"   Total conceptos: {knowledge['total_concepts']}")
        print(f"   Categorías: {knowledge['categories']}")
        print(f"   Niveles de edad: {knowledge['template_levels']}")

        print(f"\n⚡ Límites:")
        print(f"   Preguntas/hora: {knowledge['limits']['questions_per_hour']}")
        print(f"   Preguntas/día: {knowledge['limits']['questions_per_day']}")
        print(f"   Espera mínima: {knowledge['limits']['min_seconds_between']}s")

        print(f"\n📈 Actividad:")
        print(f"   Total preguntas realizadas: {activity['total_questions']}")
        print(f"   Preguntas hoy: {activity['questions_today']}")
        print(f"   Esta hora: {activity['questions_this_hour']}")
        print(f"   Puede preguntar ahora: {'Sí' if activity['can_ask_now'] else 'No'}")
        return

    # Ejecutar sesión intensiva
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     🧠 THAU - SESIÓN DE APRENDIZAJE INTENSIVO                ║
╠══════════════════════════════════════════════════════════════╣
║  Preguntas objetivo: {args.questions:4d}                                  ║
║  Edad cognitiva: {args.age:2d} años                                     ║
║  Modelo: {args.model:12s}                                       ║
║  Categoría: {(args.category or 'Todas'):15s}                             ║
║  Delay: {args.delay:.1f}s                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    if args.category:
        results = system.explore_category(
            category=args.category,
            cognitive_age=args.age,
            depth=args.questions
        )
    else:
        results = system.intensive_learning_session(
            cognitive_age=args.age,
            num_questions=args.questions,
            delay_between=args.delay
        )

    # Guardar resultados
    print(f"\n✅ Sesión completada!")
    print(f"   Tasa de éxito: {results['successful']}/{results['total_attempted']} ({100*results['successful']/max(1,results['total_attempted']):.1f}%)")


if __name__ == "__main__":
    main()
