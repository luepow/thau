#!/usr/bin/env python3
"""
Test de Web Search para THAU

Prueba las capacidades de búsqueda web e investigación.

Uso:
    python scripts/test_web_search.py           # Tests completos
    python scripts/test_web_search.py --quick   # Tests rápidos
    python scripts/test_web_search.py --demo    # Demo interactiva
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_dependencies():
    """Verifica dependencias disponibles"""
    print("\n" + "=" * 60)
    print("  Verificando Dependencias")
    print("=" * 60)

    deps = {}

    try:
        import httpx
        deps["httpx"] = True
        print("  ✓ httpx disponible")
    except ImportError:
        deps["httpx"] = False
        print("  ✗ httpx no instalado (usando urllib)")

    try:
        from bs4 import BeautifulSoup
        deps["bs4"] = True
        print("  ✓ beautifulsoup4 disponible")
    except ImportError:
        deps["bs4"] = False
        print("  ✗ beautifulsoup4 no instalado (usando regex)")

    try:
        import html2text
        deps["html2text"] = True
        print("  ✓ html2text disponible")
    except ImportError:
        deps["html2text"] = False
        print("  ✗ html2text no instalado")

    print("=" * 60)
    return deps


def test_web_search():
    """Test de WebSearchTool"""
    print("\n" + "-" * 60)
    print("  TEST: WebSearchTool")
    print("-" * 60)

    from capabilities.tools.web_search import WebSearchTool

    search = WebSearchTool(cache_enabled=True)

    # Test 1: Búsqueda simple
    print("\n[1] Búsqueda: 'Python programming'")
    results = search.search("Python programming", num_results=3)

    if results:
        print(f"   Encontrados: {len(results)} resultados")
        for i, r in enumerate(results, 1):
            print(f"\n   {i}. {r.title[:50]}...")
            print(f"      URL: {r.url[:60]}...")
            print(f"      Snippet: {r.snippet[:80]}...")
    else:
        print("   ✗ No se encontraron resultados")

    # Test 2: Búsqueda en español
    print("\n[2] Búsqueda: 'Inteligencia artificial aplicaciones'")
    results_es = search.search("Inteligencia artificial aplicaciones", num_results=3)

    if results_es:
        print(f"   Encontrados: {len(results_es)} resultados")
        for r in results_es[:2]:
            print(f"   - {r.title[:60]}...")
    else:
        print("   ✗ No se encontraron resultados")

    # Test 3: Verificar cache
    print("\n[3] Verificando cache...")
    results_cached = search.search("Python programming", num_results=3)
    print(f"   Cache funcionando: {len(results_cached) > 0}")

    print("\n   ✓ WebSearchTool test completado")


def test_web_fetcher():
    """Test de WebFetcher"""
    print("\n" + "-" * 60)
    print("  TEST: WebFetcher")
    print("-" * 60)

    from capabilities.tools.web_search import WebFetcher

    fetcher = WebFetcher()

    # Test 1: Fetch página simple
    print("\n[1] Fetch: python.org")
    page = fetcher.fetch("https://www.python.org")

    if page.success:
        print(f"   ✓ Éxito en {page.fetch_time:.2f}s")
        print(f"   Título: {page.title}")
        print(f"   Contenido: {len(page.content)} caracteres")
        print(f"   Links: {len(page.links)}")
        print(f"   Preview: {page.content[:200]}...")
    else:
        print(f"   ✗ Error: {page.error}")

    # Test 2: Fetch con meta description
    print("\n[2] Fetch: wikipedia (con meta)")
    page2 = fetcher.fetch("https://es.wikipedia.org/wiki/Python")

    if page2.success:
        print(f"   ✓ Éxito")
        print(f"   Título: {page2.title}")
        print(f"   Meta: {page2.meta_description[:100] if page2.meta_description else 'N/A'}...")
    else:
        print(f"   ✗ Error: {page2.error}")

    print("\n   ✓ WebFetcher test completado")


def test_research_agent():
    """Test de ResearchAgent"""
    print("\n" + "-" * 60)
    print("  TEST: ResearchAgent")
    print("-" * 60)

    from capabilities.tools.web_search import ResearchAgent

    research = ResearchAgent(verbose=True, max_sources=2)

    # Test 1: Investigación rápida
    print("\n[1] Investigación rápida: 'Machine Learning'")
    result = research.research("Machine Learning basics", depth="quick")

    print(f"\n   Resumen:")
    print(f"   {result.summary[:300]}...")
    print(f"\n   Confianza: {result.confidence:.0%}")
    print(f"   Fuentes: {len(result.sources)}")
    print(f"   Tiempo: {result.research_time:.2f}s")

    if result.key_facts:
        print(f"\n   Hechos clave:")
        for fact in result.key_facts[:3]:
            print(f"   - {fact[:80]}...")

    if result.related_topics:
        print(f"\n   Temas relacionados: {', '.join(result.related_topics[:5])}")

    print("\n   ✓ ResearchAgent test completado")


def test_integration():
    """Test de integración con ThauAGI"""
    print("\n" + "-" * 60)
    print("  TEST: Integración con ThauAGI")
    print("-" * 60)

    from capabilities.proto_agi.thau_agi import ThauAGI, AGIConfig, WEB_SEARCH_AVAILABLE

    print(f"\n[1] Web Search disponible en ThauAGI: {WEB_SEARCH_AVAILABLE}")

    if not WEB_SEARCH_AVAILABLE:
        print("   ✗ Web Search no disponible, saltando test")
        return

    # Crear agente sin cargar modelo (solo probar herramientas)
    config = AGIConfig(
        verbose=True,
        enable_web_search=True,
        enable_learning=False  # Desactivar para test rápido
    )

    agent = ThauAGI(config)

    # Verificar herramientas disponibles
    print(f"\n[2] Herramientas disponibles:")
    for tool_name in agent.tools.keys():
        print(f"   - {tool_name}")

    web_tools = ["web_search", "fetch_url", "research"]
    web_tools_present = all(t in agent.tools for t in web_tools)
    print(f"\n[3] Herramientas web integradas: {web_tools_present}")

    # Test directo de herramientas (sin modelo)
    print("\n[4] Test directo de _web_search:")
    result = agent._web_search("Python tutorial")
    print(f"   Éxito: {result.success}")
    if result.success:
        print(f"   Output preview: {result.output[:200]}...")
    else:
        print(f"   Error: {result.error}")

    print("\n   ✓ Integración test completado")


def demo_interactive():
    """Demo interactiva de Web Search"""
    print("\n" + "=" * 60)
    print("  THAU Web Search - Demo Interactiva")
    print("=" * 60)
    print("  Comandos:")
    print("    search <query>  - Buscar en internet")
    print("    fetch <url>     - Obtener contenido de URL")
    print("    research <topic>- Investigar tema")
    print("    exit            - Salir")
    print("=" * 60)

    from capabilities.tools.web_search import WebSearchTool, WebFetcher, ResearchAgent

    search = WebSearchTool()
    fetcher = WebFetcher()
    research = ResearchAgent(verbose=False)

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ('exit', 'quit', 'q'):
                print("¡Hasta luego!")
                break

            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if command == "search":
                if not args:
                    print("Uso: search <query>")
                    continue

                print(f"Buscando: {args}...")
                results = search.search(args, num_results=5)

                if results:
                    for i, r in enumerate(results, 1):
                        print(f"\n{i}. {r.title}")
                        print(f"   {r.url}")
                        print(f"   {r.snippet[:150]}...")
                else:
                    print("No se encontraron resultados")

            elif command == "fetch":
                if not args:
                    print("Uso: fetch <url>")
                    continue

                print(f"Obteniendo: {args}...")
                page = fetcher.fetch(args)

                if page.success:
                    print(f"\nTítulo: {page.title}")
                    print(f"Tiempo: {page.fetch_time:.2f}s")
                    print(f"\nContenido ({len(page.content)} chars):")
                    print(page.content[:500] + "...")
                else:
                    print(f"Error: {page.error}")

            elif command == "research":
                if not args:
                    print("Uso: research <topic>")
                    continue

                print(f"Investigando: {args}...")
                result = research.research(args, depth="normal")

                print(f"\n{result.summary}")
                print(f"\nConfianza: {result.confidence:.0%}")
                print(f"Fuentes: {len(result.sources)}")

                if result.key_facts:
                    print("\nHechos clave:")
                    for fact in result.key_facts[:5]:
                        print(f"  • {fact}")

            else:
                print(f"Comando desconocido: {command}")
                print("Usa: search, fetch, research, o exit")

        except KeyboardInterrupt:
            print("\n\nInterrumpido. ¡Hasta luego!")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_quick_tests():
    """Tests rápidos sin conexión a internet"""
    print("\n" + "=" * 60)
    print("  Tests Rápidos (sin red)")
    print("=" * 60)

    # Test 1: Importación
    print("\n[1] Importando módulos...")
    try:
        from capabilities.tools.web_search import (
            WebSearchTool, WebFetcher, ResearchAgent,
            SearchResult, WebPage, ResearchResult,
            SearchCache
        )
        print("   ✓ Todos los módulos importados")
    except ImportError as e:
        print(f"   ✗ Error de importación: {e}")
        return

    # Test 2: Crear instancias
    print("\n[2] Creando instancias...")
    search = WebSearchTool(cache_enabled=False)
    fetcher = WebFetcher()
    research = ResearchAgent(verbose=False)
    print("   ✓ Instancias creadas")

    # Test 3: SearchCache
    print("\n[3] Probando SearchCache...")
    cache = SearchCache()
    cache.set("test_query", [{"title": "Test", "url": "http://test.com", "snippet": "Test snippet"}])
    cached = cache.get("test_query")
    print(f"   Cache set/get: {'✓' if cached else '✗'}")

    # Test 4: Dataclasses
    print("\n[4] Probando dataclasses...")
    from datetime import datetime
    sr = SearchResult(title="Test", url="http://test.com", snippet="Snippet", source="test")
    wp = WebPage(url="http://test.com", title="Test", content="Content")
    print(f"   SearchResult.to_dict(): {'✓' if sr.to_dict() else '✗'}")
    print(f"   WebPage.to_dict(): {'✓' if wp.to_dict() else '✗'}")

    print("\n" + "=" * 60)
    print("  Tests Rápidos Completados ✓")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test de Web Search para THAU")
    parser.add_argument("--quick", action="store_true", help="Tests rápidos sin red")
    parser.add_argument("--demo", action="store_true", help="Demo interactiva")
    parser.add_argument("--search", action="store_true", help="Solo test de búsqueda")
    parser.add_argument("--fetch", action="store_true", help="Solo test de fetch")
    parser.add_argument("--research", action="store_true", help="Solo test de research")
    parser.add_argument("--integration", action="store_true", help="Solo test de integración")
    args = parser.parse_args()

    # Verificar dependencias
    deps = check_dependencies()

    if args.demo:
        demo_interactive()
    elif args.quick:
        run_quick_tests()
    elif args.search:
        test_web_search()
    elif args.fetch:
        test_web_fetcher()
    elif args.research:
        test_research_agent()
    elif args.integration:
        test_integration()
    else:
        # Tests completos
        test_web_search()
        test_web_fetcher()
        test_research_agent()
        test_integration()

        print("\n" + "=" * 60)
        print("  TODOS LOS TESTS COMPLETADOS ✓")
        print("=" * 60)


if __name__ == "__main__":
    main()
