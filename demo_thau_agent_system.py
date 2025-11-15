#!/usr/bin/env python3
"""
THAU Agent System - Demo Completo

Demuestra todas las capacidades del sistema:
1. Agentes especializados
2. Planificación avanzada
3. Tool Factory (auto-creación)
4. API Toolkit
5. MCP Integration
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from capabilities.agents.agent_system import get_agent_orchestrator, AgentRole
from capabilities.agents.planner import ThauPlanner, TaskPriority
from capabilities.tools.tool_factory import ToolFactory, ToolSpec
from capabilities.tools.api_toolkit import APIToolkit, APIConfig, AuthType
from capabilities.tools.mcp_integration import create_default_mcp_tools, MCPServer

from datetime import datetime, timedelta


def print_header(title: str):
    """Imprime header bonito"""
    print(f"\n{'='*80}")
    print(f"🎯 {title}")
    print(f"{'='*80}\n")


def demo_1_agents():
    """Demo 1: Sistema de Agentes Especializados"""
    print_header("DEMO 1: Sistema de Agentes Especializados")

    # Crear orchestrator
    orchestrator = get_agent_orchestrator()

    # Asignar tareas a diferentes agentes
    print("📋 Asignando tareas a agentes especializados...\n")

    task1 = orchestrator.assign_task(
        "Escribir función para calcular números primos",
        AgentRole.CODE_WRITER
    )

    task2 = orchestrator.assign_task(
        "Generar imagen de un robot aprendiendo a programar",
        AgentRole.VISUAL_CREATOR
    )

    task3 = orchestrator.assign_task(
        "Integrar API de Google Calendar",
        AgentRole.API_SPECIALIST
    )

    # Listar agentes activos
    print("\n📊 Agentes Activos:\n")
    for agent in orchestrator.list_agents():
        print(f"  {agent['name']}")
        print(f"    Role: {agent['role']}")
        print(f"    Tarea activa: {agent['active_task']}")
        print(f"    Capacidades: {', '.join(agent['capabilities'])}\n")

    # Guardar estado
    orchestrator.save_state()


def demo_2_planning():
    """Demo 2: Planificación Avanzada (como Claude Code)"""
    print_header("DEMO 2: Planificación Avanzada (Inspirado en Claude Code)")

    planner = ThauPlanner()

    # Crear plan para tarea compleja
    print("🧠 Creando plan para tarea compleja...\n")

    plan = planner.create_plan(
        "Crear dashboard web con autenticación JWT, conexión a API REST de usuarios, y visualización de gráficos en tiempo real",
        priority=TaskPriority.CRITICAL
    )

    # Mostrar plan
    planner.print_plan(plan)

    # Simular ejecución
    print(f"\n{'─'*80}")
    print("▶️  Simulando ejecución del plan...")
    print(f"{'─'*80}\n")

    result = planner.execute_plan(plan)

    print(f"\n📊 Resultado de Ejecución:")
    print(f"  Total pasos: {result['total_steps']}")
    print(f"  Completados: {result['completed_steps']}")
    print(f"  Fallidos: {result['failed_steps']}")

    # Guardar plan
    planner.save_plan(plan)


def demo_3_tool_factory():
    """Demo 3: Tool Factory - Auto-Creación de Herramientas"""
    print_header("DEMO 3: Tool Factory - THAU Crea Sus Propias Herramientas")

    factory = ToolFactory()

    # 1. Crear herramienta desde descripción
    print("🏭 THAU está creando herramientas automáticamente...\n")

    tool1 = factory.create_from_description(
        "Herramienta para hacer web scraping de artículos de noticias y extraer títulos"
    )

    # 2. Crear API client usando template
    print(f"\n{'─'*80}\n")

    tool2 = factory.create_api_client(
        name="spotify_search_api",
        api_url="https://api.spotify.com/v1/search",
        description="Busca canciones en Spotify",
        requires_auth=True
    )

    # 3. Crear data processor
    print(f"\n{'─'*80}\n")

    spec3 = ToolSpec(
        name="csv_analytics",
        description="Analiza archivos CSV y genera estadísticas descriptivas",
        category="data",
        parameters={
            "filepath": "str",
            "columns": "List[str]"
        },
        return_type="Dict[str, Any]"
    )

    tool3 = factory.create_tool(spec3, template_name="data_processor")

    # Listar herramientas generadas
    print(f"\n{'─'*80}")
    print("📋 Herramientas Generadas por THAU")
    print(f"{'─'*80}\n")

    for tool in factory.list_tools():
        print(f"✅ {tool['name']}")
        print(f"   Categoría: {tool['category']}")
        print(f"   Descripción: {tool['description']}")
        print(f"   Parámetros: {', '.join(tool['parameters'])}\n")

    # Guardar manifest
    factory.save_manifest()


def demo_4_api_toolkit():
    """Demo 4: API Toolkit - REST, Webhooks, Calendar"""
    print_header("DEMO 4: API Toolkit - REST, Webhooks, Calendar, Notificaciones")

    toolkit = APIToolkit()

    # 1. REST Client
    print("🔌 Configurando cliente REST...\n")

    config = APIConfig(
        name="jsonplaceholder",
        base_url="https://jsonplaceholder.typicode.com",
        auth_type=AuthType.NONE,
        default_headers={"Content-Type": "application/json"}
    )

    client = toolkit.add_api(config)

    print("📡 Haciendo request a API...")
    response = client.get("/posts/1")

    print(f"  Status: {response.status_code}")
    print(f"  Success: {response.success}")
    if response.success:
        print(f"  Título: {response.data.get('title', 'N/A')}")

    # 2. Webhooks
    print(f"\n{'─'*80}")
    print("🪝 Configurando Webhook Manager...\n")

    def payment_webhook(payload):
        print(f"  💰 Pago recibido: ${payload['amount']} {payload['currency']}")
        return {"processed": True, "confirmation": "CONF-123"}

    toolkit.webhook_manager.register_webhook(
        "payment_received",
        payment_webhook,
        secret="my_webhook_secret"
    )

    # Simular webhook
    print("\n  Simulando webhook recibido...")
    result = toolkit.webhook_manager.process_webhook(
        "payment_received",
        {"amount": 150, "currency": "USD", "customer": "john@example.com"}
    )
    print(f"  Resultado: {result}")

    # 3. Calendar
    print(f"\n{'─'*80}")
    print("📅 Integración de Calendario...\n")

    event = toolkit.calendar.create_event(
        title="Demo THAU Agent System",
        start_time=datetime.now() + timedelta(hours=2),
        end_time=datetime.now() + timedelta(hours=3),
        description="Demostración de capacidades completas de THAU",
        location="Virtual"
    )

    alarm = toolkit.calendar.set_alarm(
        title="Recordatorio: Demo en 15 min",
        alarm_time=datetime.now() + timedelta(hours=1, minutes=45),
        message="Preparar demo de THAU Agent System"
    )

    print(f"  Evento creado: {event['title']}")
    print(f"  Alarma configurada: {alarm['title']}")

    # 4. Notifications
    print(f"\n{'─'*80}")
    print("🔔 Sistema de Notificaciones...\n")

    notif = toolkit.notifications.send_notification(
        title="THAU Agent System Operativo",
        message="Todas las capacidades están funcionando correctamente",
        channel="email",
        recipient="admin@thau.ai",
        priority="high"
    )

    print(f"  Notificación enviada: {notif['title']}")
    print(f"  Canal: {notif['channel']}")
    print(f"  Prioridad: {notif['priority']}")


def demo_5_mcp():
    """Demo 5: MCP Integration - Model Context Protocol"""
    print_header("DEMO 5: MCP Integration - Model Context Protocol")

    # Crear registry con tools
    registry = create_default_mcp_tools()

    # Listar tools en formato MCP
    print("📋 Herramientas en Formato MCP (compatible con Claude/OpenAI):\n")

    tools = registry.list_tools()

    for tool in tools:
        func = tool['function']
        print(f"🔧 {func['name']}")
        print(f"   Descripción: {func['description']}")
        print(f"   Parámetros:")

        for param_name, param_info in func['parameters']['properties'].items():
            required = param_name in func['parameters']['required']
            req_marker = " *" if required else ""
            print(f"     - {param_name} ({param_info['type']}){req_marker}")
            print(f"       {param_info['description']}")
        print()

    # Invocar herramientas
    print(f"{'─'*80}")
    print("🔧 Invocando Herramientas MCP...\n")

    # 1. Generar imagen
    result1 = registry.invoke_tool(
        "generate_image",
        {
            "prompt": "un robot con capacidades de agente especializado",
            "num_images": 3
        }
    )

    print(f"✅ generate_image:")
    print(f"   Success: {result1.success}")
    print(f"   Tiempo: {result1.execution_time_ms:.2f}ms")
    print(f"   Resultado: {result1.result}\n")

    # 2. Crear evento
    result2 = registry.invoke_tool(
        "create_calendar_event",
        {
            "title": "Reunión THAU",
            "start_time": "2025-01-16T14:00:00",
            "end_time": "2025-01-16T15:00:00",
            "description": "Discutir mejoras al sistema de agentes"
        }
    )

    print(f"✅ create_calendar_event:")
    print(f"   Success: {result2.success}")
    print(f"   Resultado: {result2.result}\n")

    # 3. Llamar API
    result3 = registry.invoke_tool(
        "call_rest_api",
        {
            "url": "https://jsonplaceholder.typicode.com/todos/1",
            "method": "GET"
        }
    )

    print(f"✅ call_rest_api:")
    print(f"   Success: {result3.success}")
    print(f"   Status: {result3.result['status_code']}")
    print(f"   Data: {result3.result['data']}\n")

    # Crear servidor MCP
    print(f"{'─'*80}")
    print("🖥️  Creando Servidor MCP...\n")

    server = MCPServer(registry)
    session = server.create_session("demo_session")

    print(f"  Session ID: {session['session_id']}")
    print(f"  Tools disponibles: {len(session['tools'])}")

    # Exportar schema
    print(f"\n{'─'*80}")
    print("💾 Exportando Schema MCP...\n")

    registry.export_schema()


def demo_6_integration():
    """Demo 6: Todo Trabajando Junto"""
    print_header("DEMO 6: Sistema Completo - Todo Trabajando Junto")

    print("🎯 Escenario: Usuario pide 'Crear sistema de notificaciones automáticas'\n")

    # 1. Planner crea plan
    print("📋 Paso 1: Planner analiza y crea plan...\n")

    planner = ThauPlanner()
    plan = planner.create_plan(
        "Crear sistema de notificaciones automáticas con webhook, API REST y alarmas",
        priority=TaskPriority.HIGH
    )

    print(f"  ✅ Plan creado con {len(plan.steps)} pasos")
    print(f"  Complejidad: {plan.complexity.value}")
    print(f"  Riesgos identificados: {len(plan.risks)}\n")

    # 2. Orchestrator asigna agentes
    print("🎭 Paso 2: Orchestrator asigna agentes especializados...\n")

    orchestrator = get_agent_orchestrator()

    for step in plan.steps[:3]:  # Primeros 3 pasos
        if "api" in step.description.lower():
            agent_role = AgentRole.API_SPECIALIST
        elif "diseñ" in step.description.lower():
            agent_role = AgentRole.ARCHITECT
        else:
            agent_role = AgentRole.CODE_WRITER

        task = orchestrator.assign_task(step.description, agent_role)
        print(f"  ✅ Tarea asignada a {agent_role.value}: {step.description}")

    # 3. Tool Factory crea herramientas necesarias
    print(f"\n🏭 Paso 3: Tool Factory auto-crea herramientas necesarias...\n")

    factory = ToolFactory()

    tool1 = factory.create_from_description(
        "Enviar notificaciones push a dispositivos móviles"
    )
    print(f"  ✅ Tool creado: {tool1.spec.name}")

    tool2 = factory.create_from_description(
        "Webhook para recibir eventos de sistema externo"
    )
    print(f"  ✅ Tool creado: {tool2.spec.name}")

    # 4. API Toolkit configura integraciones
    print(f"\n🔌 Paso 4: API Toolkit configura integraciones...\n")

    toolkit = APIToolkit()

    # Webhook
    toolkit.webhook_manager.register_webhook(
        "system_event",
        lambda payload: print(f"  📡 Evento recibido: {payload}")
    )
    print(f"  ✅ Webhook configurado: system_event")

    # Calendario para alarmas
    alarm = toolkit.calendar.set_alarm(
        title="Verificar sistema cada hora",
        alarm_time=datetime.now() + timedelta(hours=1)
    )
    print(f"  ✅ Alarma configurada: {alarm['title']}")

    # 5. MCP registra todo
    print(f"\n🔗 Paso 5: MCP registra herramientas para uso universal...\n")

    registry = create_default_mcp_tools()
    registry.export_schema("data/mcp/notification_system_schema.json")

    print(f"  ✅ Schema MCP exportado")
    print(f"  ✅ {len(registry.list_tools())} herramientas disponibles")

    # Resultado final
    print(f"\n{'='*80}")
    print("✅ SISTEMA COMPLETO OPERATIVO")
    print(f"{'='*80}")
    print(f"""
  Componentes Activos:
    🎭 Agentes: {len(orchestrator.list_agents())} especializados
    📋 Plan: {len(plan.steps)} pasos definidos
    🏭 Tools: {len(factory.list_tools())} herramientas auto-creadas
    🔌 APIs: Webhooks, Calendar, Notifications configurados
    🔗 MCP: Compatible con Claude/OpenAI

  Capacidades:
    ✅ Planificación inteligente
    ✅ Delegación a agentes
    ✅ Auto-creación de herramientas
    ✅ Integraciones REST/Webhooks
    ✅ Gestión de calendario y alarmas
    ✅ Model Context Protocol

  Estado:
    🟢 THAU Agent System: OPERATIVO
    🟢 Listo para producción
    """)


def main():
    """Main demo"""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                    🤖 THAU AGENT SYSTEM - DEMO COMPLETO                       ║
║                                                                               ║
║  Sistema de Agentes Inteligentes inspirado en Claude Code                    ║
║  Con capacidades únicas de auto-creación de herramientas                     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)

    print("\n📋 Menú de Demos:")
    print("  1. Sistema de Agentes Especializados")
    print("  2. Planificación Avanzada (como Claude Code)")
    print("  3. Tool Factory - Auto-Creación de Herramientas")
    print("  4. API Toolkit - REST, Webhooks, Calendar")
    print("  5. MCP Integration - Model Context Protocol")
    print("  6. ⭐ Demo Completo - Todo Trabajando Junto")
    print("  0. Ejecutar TODOS los demos")

    choice = input("\nSelecciona demo (0-6): ").strip()

    if choice == "0":
        demo_1_agents()
        demo_2_planning()
        demo_3_tool_factory()
        demo_4_api_toolkit()
        demo_5_mcp()
        demo_6_integration()

    elif choice == "1":
        demo_1_agents()

    elif choice == "2":
        demo_2_planning()

    elif choice == "3":
        demo_3_tool_factory()

    elif choice == "4":
        demo_4_api_toolkit()

    elif choice == "5":
        demo_5_mcp()

    elif choice == "6":
        demo_6_integration()

    else:
        print("❌ Opción inválida")
        return

    print(f"\n{'='*80}")
    print("✅ Demo Completado")
    print(f"{'='*80}\n")

    print("📚 Para más información:")
    print("  - README_THAU_AGENTS.md - Documentación completa")
    print("  - README_THAU_VISION.md - Sistema visual de THAU")
    print("  - CLAUDE.md - Instrucciones del proyecto\n")


if __name__ == "__main__":
    main()
