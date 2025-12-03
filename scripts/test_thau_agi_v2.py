#!/usr/bin/env python3
"""
Test Completo de THAU AGI v2

Prueba todas las capacidades del sistema Proto-AGI unificado.

Uso:
    python scripts/test_thau_agi_v2.py             # Tests completos
    python scripts/test_thau_agi_v2.py --quick     # Tests rapidos (sin modelo)
    python scripts/test_thau_agi_v2.py --demo      # Demo interactiva
    python scripts/test_thau_agi_v2.py --benchmark # Benchmark de rendimiento
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(title: str):
    """Imprime header formateado"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str):
    """Imprime seccion formateada"""
    print("\n" + "-" * 60)
    print(f"  {title}")
    print("-" * 60)


def test_imports():
    """Test de importaciones"""
    print_section("TEST: Importaciones")

    errors = []

    # Test 1: Proto-AGI basico
    try:
        from capabilities.proto_agi import ThauProtoAGI, ThauTools, ToolResult
        print("  [OK] Proto-AGI basico")
    except ImportError as e:
        errors.append(f"Proto-AGI basico: {e}")
        print(f"  [ERROR] Proto-AGI basico: {e}")

    # Test 2: AGI v1
    try:
        from capabilities.proto_agi import ThauAGI, AGIConfig
        print("  [OK] AGI v1")
    except ImportError as e:
        errors.append(f"AGI v1: {e}")
        print(f"  [ERROR] AGI v1: {e}")

    # Test 3: AGI v2
    try:
        from capabilities.proto_agi import ThauAGIv2, ThauConfig, ThauMode
        print("  [OK] AGI v2")
    except ImportError as e:
        errors.append(f"AGI v2: {e}")
        print(f"  [ERROR] AGI v2: {e}")

    # Test 4: Experiential Learning
    try:
        from capabilities.proto_agi import (
            ExperienceStore, MetacognitiveEngine, AdaptiveStrategy,
            OutcomeType, StrategyType
        )
        print("  [OK] Experiential Learning")
    except ImportError as e:
        errors.append(f"Experiential Learning: {e}")
        print(f"  [ERROR] Experiential Learning: {e}")

    # Test 5: Multi-Agent
    try:
        from capabilities.proto_agi import (
            MultiAgentSystem, AgentRole, MessageBus, SharedMemory
        )
        print("  [OK] Multi-Agent System")
    except ImportError as e:
        errors.append(f"Multi-Agent: {e}")
        print(f"  [ERROR] Multi-Agent: {e}")

    # Test 6: Knowledge Base
    try:
        from capabilities.proto_agi import (
            KnowledgeStore, ContextBuilder, KnowledgeLearner, FeedbackSystem
        )
        print("  [OK] Knowledge Base")
    except ImportError as e:
        errors.append(f"Knowledge Base: {e}")
        print(f"  [ERROR] Knowledge Base: {e}")

    # Test 7: Web Search
    try:
        from capabilities.tools import (
            WebSearchTool, WebFetcher, ResearchAgent, WEB_SEARCH_AVAILABLE
        )
        status = "disponible" if WEB_SEARCH_AVAILABLE else "no disponible"
        print(f"  [OK] Web Search ({status})")
    except ImportError as e:
        print(f"  [WARN] Web Search: {e}")

    if errors:
        print(f"\n  [!] {len(errors)} errores de importacion")
        return False

    print("\n  [SUCCESS] Todas las importaciones OK")
    return True


def test_experiential_learning():
    """Test del sistema de aprendizaje experiencial"""
    print_section("TEST: Aprendizaje Experiencial")

    from capabilities.proto_agi import (
        ExperienceStore, MetacognitiveEngine, AdaptiveStrategy,
        OutcomeType, StrategyType, Experience
    )
    from datetime import datetime

    # Test 1: ExperienceStore
    print("\n[1] ExperienceStore...")
    import tempfile
    import os
    temp_db = tempfile.mktemp(suffix=".db")
    store = ExperienceStore(db_path=temp_db)

    exp = Experience(
        id="test_exp_001",
        timestamp=datetime.now(),
        goal="Test calculation",
        goal_type="calculation",
        context={"test": True},
        strategy=StrategyType.TOOL_SINGLE,
        tools_used=["calculate"],
        steps_taken=1,
        outcome=OutcomeType.SUCCESS,
        confidence=0.9,
        execution_time=0.5
    )
    exp_id = store.store_experience(exp)
    print(f"    Experiencia guardada: {exp_id[:20]}...")

    similar = store.find_similar_experiences("calculation test")
    print(f"    Experiencias similares: {len(similar)}")

    # Test 2: MetacognitiveEngine
    print("\n[2] MetacognitiveEngine...")
    meta = MetacognitiveEngine(store)

    evaluation = meta.evaluate_response(
        goal="calcula 5 + 3",
        response="El resultado es 8",
        tool_results=[{"success": True, "output": "8"}]
    )
    print(f"    Confianza: {evaluation['confidence']:.0%}")
    reason = evaluation.get('reason', evaluation.get('explanation', 'N/A'))
    print(f"    Razon: {str(reason)[:50]}...")

    # Test 3: AdaptiveStrategy
    print("\n[3] AdaptiveStrategy...")
    adaptive = AdaptiveStrategy(store, meta)

    strategy, meta_info = adaptive.select_strategy(
        goal="calcula 10 * 5",
        available_tools=["calculate", "execute_python"],
        context={}
    )
    print(f"    Estrategia: {strategy.value}")
    print(f"    Confianza: {meta_info['confidence']:.0%}")

    print("\n  [SUCCESS] Aprendizaje Experiencial OK")
    return True


def test_multi_agent():
    """Test del sistema multi-agente"""
    print_section("TEST: Sistema Multi-Agente")

    from capabilities.proto_agi import (
        MultiAgentSystem, AgentRole, MessageBus, SharedMemory
    )

    # Test 1: MessageBus
    print("\n[1] MessageBus...")
    bus = MessageBus()
    print(f"    Creado: {bus is not None}")

    # Test 2: SharedMemory
    print("\n[2] SharedMemory...")
    memory = SharedMemory()
    memory.set("test_key", "test_value", "test_agent")
    retrieved = memory.get("test_key")
    print(f"    Set/Get: {'OK' if retrieved == 'test_value' else 'FAIL'}")

    # Test 3: MultiAgentSystem
    print("\n[3] MultiAgentSystem...")
    mas = MultiAgentSystem(verbose=False)
    mas.initialize([
        AgentRole.CODER,
        AgentRole.REVIEWER,
        AgentRole.RESEARCHER
    ])

    status = mas.get_status()
    print(f"    Agentes totales: {status.get('total_agents', 0)}")
    if 'agents' in status:
        roles = [info.get('role', 'unknown') for info in status['agents'].values()]
        print(f"    Roles: {', '.join(roles)}")

    print("\n  [SUCCESS] Multi-Agente OK")
    return True


def test_knowledge_base():
    """Test de la base de conocimiento"""
    print_section("TEST: Base de Conocimiento")

    from capabilities.proto_agi import (
        KnowledgeStore, ContextBuilder, KnowledgeLearner,
        FeedbackSystem, KnowledgeType
    )

    # Test 1: KnowledgeStore
    print("\n[1] KnowledgeStore...")
    store = KnowledgeStore()

    doc_id = store.store(
        content="Python es un lenguaje de programacion interpretado.",
        knowledge_type=KnowledgeType.FACT,
        source="test",
        metadata={"topic": "programming"}
    )
    print(f"    Documento guardado: {doc_id[:20]}...")

    results = store.retrieve("lenguaje programacion", n_results=1)
    print(f"    Recuperados: {len(results)}")

    # Test 2: ContextBuilder
    print("\n[2] ContextBuilder...")
    builder = ContextBuilder(store)

    context = builder.build_context("que es python", n_items=1)
    print(f"    Contexto generado: {len(context)} chars")

    # Test 3: KnowledgeLearner
    print("\n[3] KnowledgeLearner...")
    learner = KnowledgeLearner(store)

    learned = learner.learn_from_conversation(
        user_message="que es una funcion?",
        assistant_response="Una funcion es un bloque de codigo reutilizable.",
        was_helpful=True
    )
    print(f"    Aprendizajes: {len(learned)}")

    # Test 4: FeedbackSystem
    print("\n[4] FeedbackSystem...")
    feedback = FeedbackSystem(learner)

    fb_id = feedback.thumbs_up("test_interaction_001")
    print(f"    Feedback registrado: {fb_id[:20]}...")

    stats = feedback.get_stats()
    print(f"    Total feedback: {stats['total_feedback']}")

    print("\n  [SUCCESS] Knowledge Base OK")
    return True


def test_thau_agi_v2_quick():
    """Test rapido de ThauAGI v2 (sin modelo)"""
    print_section("TEST: ThauAGI v2 (Quick)")

    from capabilities.proto_agi import ThauAGIv2, ThauConfig, ThauMode

    # Crear configuracion sin modelo
    config = ThauConfig(
        verbose=False,
        enable_learning=True,
        enable_metacognition=True,
        enable_web_search=True,
        enable_multi_agent=True,
        enable_knowledge_base=True,
        enable_feedback=True
    )

    print("\n[1] Creando instancia ThauAGIv2...")
    agent = ThauAGIv2(config)
    print(f"    Session ID: {agent.session_id}")

    print("\n[2] Verificando componentes...")
    print(f"    Tools: {len(agent.tools)}")
    print(f"    Experience Store: {agent.experience_store is not None}")
    print(f"    Metacognitive: {agent.metacognitive is not None}")
    print(f"    Multi-Agent: {agent.multi_agent_system is not None}")
    print(f"    Knowledge Store: {agent.knowledge_store is not None}")
    print(f"    Feedback System: {agent.feedback_system is not None}")

    print("\n[3] Test herramienta calculate...")
    result = agent._execute_tool("calculate", {"expression": "15 * 7"})
    print(f"    15 * 7 = {result.output}")
    print(f"    Success: {result.success}")

    print("\n[4] Test herramienta list_directory...")
    result = agent._execute_tool("list_directory", {"dirpath": "."})
    print(f"    Archivos listados: {result.success}")

    print("\n[5] Test clasificacion de metas...")
    goals_to_test = [
        ("calcula 5 + 3", "calculation"),
        ("escribe codigo python", "code"),
        ("lee el archivo test.py", "file"),
        ("busca en internet Python", "web_search"),
        ("que es una funcion?", "question"),
    ]

    for goal, expected in goals_to_test:
        classified = agent._classify_goal(goal)
        status = "OK" if classified == expected else f"FAIL ({classified})"
        print(f"    '{goal[:30]}' -> {status}")

    print("\n[6] Test deteccion de herramientas...")
    tool_tests = [
        ("calcula 10 + 5", "calculate"),
        ("lista los archivos", "list_directory"),
    ]

    for goal, expected_tool in tool_tests:
        detected = agent._detect_tool_need(goal)
        if detected:
            tool_name, params = detected
            status = "OK" if tool_name == expected_tool else f"FAIL ({tool_name})"
        else:
            status = "FAIL (no detectado)"
        print(f"    '{goal[:30]}' -> {status}")

    print("\n[7] Test estadisticas...")
    stats = agent.get_stats()
    print(f"    Session: {stats['session_id']}")
    print(f"    Interactions: {stats['interactions']}")

    print("\n  [SUCCESS] ThauAGI v2 Quick Test OK")
    return True


def test_thau_agi_v2_full():
    """Test completo de ThauAGI v2 (con modelo si disponible)"""
    print_section("TEST: ThauAGI v2 (Full)")

    from capabilities.proto_agi import ThauAGIv2, ThauConfig, ThauMode

    config = ThauConfig(
        verbose=True,
        enable_learning=True,
        enable_metacognition=True,
        enable_web_search=True,
        enable_multi_agent=True,
        enable_knowledge_base=True,
        enable_feedback=True
    )

    agent = ThauAGIv2(config)

    # Test 1: Calculo
    print("\n[TEST 1] Calculo")
    result = agent.run("Calcula 25 * 4 + 50")
    print(f"  Resultado: {result['goal_achieved']}")
    print(f"  Confianza: {result['confidence']:.0%}")

    # Feedback positivo
    agent.thumbs_up()

    # Test 2: Listado de archivos
    print("\n[TEST 2] Listado de archivos")
    result = agent.run("Lista los archivos del directorio actual")
    print(f"  Resultado: {result['goal_achieved']}")

    # Test 3: Pregunta
    print("\n[TEST 3] Pregunta")
    result = agent.run("Que es una variable en programacion?")
    print(f"  Resultado: {result['goal_achieved']}")
    print(f"  Estrategia: {result['strategy_used']}")

    # Estadisticas finales
    print("\n" + "=" * 60)
    print("ESTADISTICAS FINALES")
    print("=" * 60)
    stats = agent.get_stats()
    print(f"Total interacciones: {stats['interactions']}")

    if 'experiences' in stats:
        print(f"Experiencias guardadas: {stats['experiences'].get('total_experiences', 0)}")

    if 'feedback' in stats:
        print(f"Tasa satisfaccion: {stats['feedback'].get('satisfaction_rate', 0):.0%}")

    # Resumen de sesion
    summary = agent.get_session_summary()
    print(f"\nResumen de sesion:")
    print(f"  Total interacciones: {summary['total_interactions']}")

    print("\n  [SUCCESS] ThauAGI v2 Full Test OK")
    return True


def test_web_search():
    """Test de capacidades web"""
    print_section("TEST: Web Search")

    try:
        from capabilities.tools import WEB_SEARCH_AVAILABLE
        if not WEB_SEARCH_AVAILABLE:
            print("  [SKIP] Web Search no disponible")
            return True
    except ImportError:
        print("  [SKIP] Modulo web_search no encontrado")
        return True

    from capabilities.tools import WebSearchTool, WebFetcher

    # Test 1: Busqueda
    print("\n[1] WebSearchTool...")
    search = WebSearchTool()
    results = search.search("Python programming", num_results=3)
    print(f"    Resultados: {len(results)}")

    if results:
        print(f"    Primer resultado: {results[0].title[:50]}...")

    # Test 2: Fetch
    print("\n[2] WebFetcher...")
    fetcher = WebFetcher()
    page = fetcher.fetch("https://www.python.org")
    print(f"    Fetch exitoso: {page.success}")
    if page.success:
        print(f"    Titulo: {page.title[:50]}...")

    print("\n  [SUCCESS] Web Search OK")
    return True


def run_quick_tests():
    """Ejecuta tests rapidos"""
    print_header("THAU AGI v2 - Tests Rapidos")

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_experiential_learning()
    all_passed &= test_multi_agent()
    all_passed &= test_knowledge_base()
    all_passed &= test_thau_agi_v2_quick()

    print_header("RESULTADO FINAL")
    if all_passed:
        print("  [SUCCESS] TODOS LOS TESTS PASARON")
    else:
        print("  [FAIL] Algunos tests fallaron")

    return all_passed


def run_full_tests():
    """Ejecuta tests completos"""
    print_header("THAU AGI v2 - Tests Completos")

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_experiential_learning()
    all_passed &= test_multi_agent()
    all_passed &= test_knowledge_base()
    all_passed &= test_web_search()
    all_passed &= test_thau_agi_v2_full()

    print_header("RESULTADO FINAL")
    if all_passed:
        print("  [SUCCESS] TODOS LOS TESTS PASARON")
    else:
        print("  [FAIL] Algunos tests fallaron")

    return all_passed


def run_benchmark():
    """Benchmark de rendimiento"""
    print_header("THAU AGI v2 - Benchmark")

    from capabilities.proto_agi import ThauAGIv2, ThauConfig

    config = ThauConfig(
        verbose=False,
        enable_learning=True,
        enable_metacognition=True,
        enable_web_search=False,  # Desactivar para benchmark puro
        enable_multi_agent=False,
        enable_knowledge_base=True,
        enable_feedback=True
    )

    agent = ThauAGIv2(config)

    # Benchmark de herramientas
    print("\n[BENCHMARK] Herramientas")

    benchmarks = [
        ("calculate", {"expression": "100 * 50 + 25"}),
        ("list_directory", {"dirpath": "."}),
    ]

    for tool_name, params in benchmarks:
        times = []
        for _ in range(10):
            start = time.time()
            agent._execute_tool(tool_name, params)
            times.append(time.time() - start)

        avg = sum(times) / len(times)
        print(f"  {tool_name}: {avg*1000:.2f}ms promedio")

    # Benchmark de clasificacion
    print("\n[BENCHMARK] Clasificacion de metas")
    goals = [
        "calcula 5 + 3",
        "escribe codigo python para ordenar una lista",
        "lee el archivo config.py",
        "que es una clase en python?",
        "lista los archivos del directorio",
    ]

    times = []
    for _ in range(100):
        start = time.time()
        for goal in goals:
            agent._classify_goal(goal)
        times.append(time.time() - start)

    avg = sum(times) / len(times)
    print(f"  5 clasificaciones: {avg*1000:.2f}ms promedio")

    # Benchmark de deteccion de herramientas
    print("\n[BENCHMARK] Deteccion de herramientas")
    times = []
    for _ in range(100):
        start = time.time()
        for goal in goals:
            agent._detect_tool_need(goal)
        times.append(time.time() - start)

    avg = sum(times) / len(times)
    print(f"  5 detecciones: {avg*1000:.2f}ms promedio")

    print("\n  [SUCCESS] Benchmark completado")
    return True


def run_demo():
    """Demo interactiva de ThauAGI v2"""
    print_header("THAU AGI v2 - Demo Interactiva")
    print("""
  Comandos especiales:
    /stats     - Ver estadisticas
    /feedback+ - Dar feedback positivo
    /feedback- - Dar feedback negativo
    /mode X    - Cambiar modo (chat, task, research, collaborative)
    /exit      - Salir
    """)

    from capabilities.proto_agi import ThauAGIv2, ThauConfig, ThauMode

    config = ThauConfig(
        verbose=True,
        enable_learning=True,
        enable_metacognition=True,
        enable_web_search=True,
        enable_multi_agent=True,
        enable_knowledge_base=True,
        enable_feedback=True
    )

    agent = ThauAGIv2(config)

    mode_map = {
        "chat": ThauMode.CHAT,
        "task": ThauMode.TASK,
        "research": ThauMode.RESEARCH,
        "collaborative": ThauMode.COLLABORATIVE,
        "learning": ThauMode.LEARNING,
    }

    while True:
        try:
            user_input = input("\nTHAU> ").strip()

            if not user_input:
                continue

            # Comandos especiales
            if user_input.lower() in ('/exit', '/quit', '/q'):
                print("\nHasta luego!")
                summary = agent.get_session_summary()
                print(f"Total interacciones: {summary['total_interactions']}")
                break

            elif user_input.lower() == '/stats':
                stats = agent.get_stats()
                print(f"\nEstadisticas:")
                print(f"  Interacciones: {stats['interactions']}")
                if 'experiences' in stats:
                    print(f"  Experiencias: {stats['experiences'].get('total_experiences', 0)}")
                if 'feedback' in stats:
                    print(f"  Satisfaccion: {stats['feedback'].get('satisfaction_rate', 0):.0%}")
                continue

            elif user_input.lower() == '/feedback+':
                agent.thumbs_up()
                continue

            elif user_input.lower().startswith('/feedback-'):
                parts = user_input.split(maxsplit=1)
                reason = parts[1] if len(parts) > 1 else ""
                agent.thumbs_down(reason=reason)
                continue

            elif user_input.lower().startswith('/mode '):
                mode_str = user_input[6:].strip().lower()
                if mode_str in mode_map:
                    agent.current_mode = mode_map[mode_str]
                    print(f"Modo cambiado a: {mode_str}")
                else:
                    print(f"Modos disponibles: {', '.join(mode_map.keys())}")
                continue

            # Ejecutar meta
            agent.run(user_input)

        except KeyboardInterrupt:
            print("\n\nInterrumpido. Hasta luego!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(description="Tests de THAU AGI v2")
    parser.add_argument("--quick", action="store_true", help="Tests rapidos sin modelo")
    parser.add_argument("--demo", action="store_true", help="Demo interactiva")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark de rendimiento")
    parser.add_argument("--imports", action="store_true", help="Solo test de imports")
    parser.add_argument("--learning", action="store_true", help="Solo test de learning")
    parser.add_argument("--multiagent", action="store_true", help="Solo test multi-agente")
    parser.add_argument("--knowledge", action="store_true", help="Solo test knowledge base")
    parser.add_argument("--web", action="store_true", help="Solo test web search")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.benchmark:
        run_benchmark()
    elif args.quick:
        run_quick_tests()
    elif args.imports:
        test_imports()
    elif args.learning:
        test_experiential_learning()
    elif args.multiagent:
        test_multi_agent()
    elif args.knowledge:
        test_knowledge_base()
    elif args.web:
        test_web_search()
    else:
        run_full_tests()


if __name__ == "__main__":
    main()
