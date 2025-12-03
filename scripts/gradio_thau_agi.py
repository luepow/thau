#!/usr/bin/env python3
"""
THAU AGI v2 - Interfaz Gradio

Interfaz web interactiva para probar el sistema Proto-AGI.

Uso:
    python scripts/gradio_thau_agi.py

    Luego abre http://localhost:7860 en tu navegador
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gradio as gr
from typing import List, Tuple
import json

# Import THAU AGI v2
from capabilities.proto_agi import ThauAGIv2, ThauConfig, ThauMode

# Global agent instance
agent = None


def initialize_agent(
    enable_learning: bool,
    enable_metacognition: bool,
    enable_web_search: bool,
    enable_multi_agent: bool,
    enable_knowledge: bool,
    enable_feedback: bool
) -> str:
    """Inicializa el agente con la configuracion seleccionada"""
    global agent

    config = ThauConfig(
        verbose=False,
        enable_learning=enable_learning,
        enable_metacognition=enable_metacognition,
        enable_web_search=enable_web_search,
        enable_multi_agent=enable_multi_agent,
        enable_knowledge_base=enable_knowledge,
        enable_feedback=enable_feedback
    )

    agent = ThauAGIv2(config)

    # Construir mensaje de estado
    status = "THAU AGI v2 inicializado!\n\n"
    status += "Componentes activos:\n"
    status += f"  - Aprendizaje: {'✓' if enable_learning else '✗'}\n"
    status += f"  - Metacognicion: {'✓' if enable_metacognition else '✗'}\n"
    status += f"  - Web Search: {'✓' if enable_web_search else '✗'}\n"
    status += f"  - Multi-Agente: {'✓' if enable_multi_agent else '✗'}\n"
    status += f"  - Knowledge Base: {'✓' if enable_knowledge else '✗'}\n"
    status += f"  - Feedback: {'✓' if enable_feedback else '✗'}\n"
    status += f"\nHerramientas: {len(agent.tools)}"
    status += f"\nSession ID: {agent.session_id}"

    return status


def chat(message: str, history: List[Tuple[str, str]], mode: str) -> Tuple[List[Tuple[str, str]], str]:
    """Procesa mensaje del usuario"""
    global agent

    if agent is None:
        # Inicializar con valores por defecto
        initialize_agent(True, True, True, True, True, True)

    # Mapear modo
    mode_map = {
        "Chat": ThauMode.CHAT,
        "Tarea": ThauMode.TASK,
        "Investigacion": ThauMode.RESEARCH,
        "Colaborativo": ThauMode.COLLABORATIVE,
        "Aprendizaje": ThauMode.LEARNING
    }
    thau_mode = mode_map.get(mode, ThauMode.CHAT)

    # Ejecutar
    result = agent.run(message, thau_mode)

    # Construir respuesta con metricas
    response = result["response"]

    # Info adicional
    info = f"Confianza: {result['confidence']:.0%} | "
    info += f"Estrategia: {result['strategy_used']} | "
    info += f"Tiempo: {result['execution_time']:.2f}s"

    if result["tools_used"]:
        info += f"\nHerramientas: {', '.join(result['tools_used'])}"

    # Actualizar historial
    history.append((message, response))

    return history, info


def give_feedback(feedback_type: str) -> str:
    """Registra feedback del usuario"""
    global agent

    if agent is None:
        return "Agente no inicializado"

    if feedback_type == "👍 Positivo":
        agent.thumbs_up()
        return "Feedback positivo registrado!"
    elif feedback_type == "👎 Negativo":
        agent.thumbs_down()
        return "Feedback negativo registrado!"

    return "Feedback no reconocido"


def get_stats() -> str:
    """Obtiene estadisticas del agente"""
    global agent

    if agent is None:
        return "Agente no inicializado"

    stats = agent.get_stats()

    output = "=== Estadisticas THAU AGI v2 ===\n\n"
    output += f"Session ID: {stats['session_id']}\n"
    output += f"Interacciones: {stats['interactions']}\n"
    output += f"Modo actual: {stats['mode']}\n\n"

    if 'experiences' in stats:
        exp = stats['experiences']
        output += "--- Experiencias ---\n"
        output += f"Total: {exp.get('total_experiences', 0)}\n"
        output += f"Tasa exito: {exp.get('success_rate', 0):.0%}\n\n"

    if 'feedback' in stats:
        fb = stats['feedback']
        output += "--- Feedback ---\n"
        output += f"Total: {fb.get('total_feedback', 0)}\n"
        output += f"Positivos: {fb.get('positive', 0)}\n"
        output += f"Negativos: {fb.get('negative', 0)}\n"
        output += f"Satisfaccion: {fb.get('satisfaction_rate', 0):.0%}\n\n"

    if 'knowledge' in stats:
        kb = stats['knowledge']
        output += "--- Knowledge Base ---\n"
        output += f"Documentos: {kb.get('total_documents', 0)}\n"

    return output


def clear_chat() -> Tuple[List, str]:
    """Limpia el chat"""
    return [], ""


# Crear interfaz Gradio
with gr.Blocks(title="THAU AGI v2", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 THAU AGI v2 - Proto-AGI System

    Sistema de Inteligencia Artificial General Prototipo con:
    - Ciclo ReAct (Reason-Act-Observe-Reflect)
    - Aprendizaje Experiencial
    - Metacognicion
    - Busqueda Web
    - Sistema Multi-Agente
    - Knowledge Base con RAG
    - Feedback Loop
    """)

    with gr.Tab("💬 Chat"):
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversacion",
                    height=400,
                    show_copy_button=True
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Tu mensaje",
                        placeholder="Escribe tu mensaje aqui...",
                        scale=4,
                        lines=2
                    )
                    send_btn = gr.Button("Enviar", variant="primary", scale=1)

                info_box = gr.Textbox(
                    label="Info de respuesta",
                    interactive=False,
                    lines=2
                )

                with gr.Row():
                    mode = gr.Dropdown(
                        choices=["Chat", "Tarea", "Investigacion", "Colaborativo", "Aprendizaje"],
                        value="Chat",
                        label="Modo",
                        scale=2
                    )
                    clear_btn = gr.Button("Limpiar Chat", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### Feedback")
                feedback_btn_pos = gr.Button("👍 Positivo", variant="secondary")
                feedback_btn_neg = gr.Button("👎 Negativo", variant="secondary")
                feedback_output = gr.Textbox(label="Estado", interactive=False, lines=1)

                gr.Markdown("### Acciones rapidas")

                example_btns = []
                examples = [
                    "Calcula 25 * 4 + 100",
                    "Lista los archivos del directorio",
                    "Que es una funcion recursiva?",
                    "Busca en internet que es Python",
                ]

                for ex in examples:
                    btn = gr.Button(ex[:30] + "...", size="sm")
                    btn.click(lambda x=ex: x, outputs=msg)

        # Eventos
        send_btn.click(
            chat,
            inputs=[msg, chatbot, mode],
            outputs=[chatbot, info_box]
        ).then(
            lambda: "",
            outputs=msg
        )

        msg.submit(
            chat,
            inputs=[msg, chatbot, mode],
            outputs=[chatbot, info_box]
        ).then(
            lambda: "",
            outputs=msg
        )

        clear_btn.click(clear_chat, outputs=[chatbot, info_box])
        feedback_btn_pos.click(lambda: give_feedback("👍 Positivo"), outputs=feedback_output)
        feedback_btn_neg.click(lambda: give_feedback("👎 Negativo"), outputs=feedback_output)

    with gr.Tab("⚙️ Configuracion"):
        gr.Markdown("### Configurar Componentes")

        with gr.Row():
            with gr.Column():
                chk_learning = gr.Checkbox(label="Aprendizaje Experiencial", value=True)
                chk_meta = gr.Checkbox(label="Metacognicion", value=True)
                chk_web = gr.Checkbox(label="Web Search", value=True)

            with gr.Column():
                chk_multi = gr.Checkbox(label="Multi-Agente", value=True)
                chk_knowledge = gr.Checkbox(label="Knowledge Base", value=True)
                chk_feedback = gr.Checkbox(label="Feedback System", value=True)

        init_btn = gr.Button("Reinicializar Agente", variant="primary")
        init_output = gr.Textbox(label="Estado", interactive=False, lines=10)

        init_btn.click(
            initialize_agent,
            inputs=[chk_learning, chk_meta, chk_web, chk_multi, chk_knowledge, chk_feedback],
            outputs=init_output
        )

    with gr.Tab("📊 Estadisticas"):
        stats_output = gr.Textbox(
            label="Estadisticas del Sistema",
            interactive=False,
            lines=20
        )
        refresh_btn = gr.Button("Actualizar Estadisticas", variant="secondary")
        refresh_btn.click(get_stats, outputs=stats_output)

    with gr.Tab("ℹ️ Ayuda"):
        gr.Markdown("""
        ## Guia de Uso

        ### Modos de Operacion

        - **Chat**: Conversacion casual y preguntas generales
        - **Tarea**: Ejecucion de tareas especificas (calculos, archivos, codigo)
        - **Investigacion**: Busqueda profunda de informacion
        - **Colaborativo**: Usa multiples agentes especializados
        - **Aprendizaje**: Modo de aprendizaje intensivo

        ### Herramientas Disponibles

        - `calculate`: Evalua expresiones matematicas
        - `read_file`: Lee contenido de archivos
        - `write_file`: Escribe archivos
        - `list_directory`: Lista archivos de un directorio
        - `execute_python`: Ejecuta codigo Python
        - `web_search`: Busca en internet (DuckDuckGo)
        - `fetch_url`: Obtiene contenido de una URL
        - `research`: Investigacion profunda de un tema

        ### Ejemplos de Prompts

        ```
        Calcula 15 * 23 + 100
        Lista los archivos del directorio actual
        Lee el archivo requirements.txt
        Busca en internet que es machine learning
        Investiga sobre inteligencia artificial
        Que es una funcion lambda en Python?
        Escribe un programa que calcule el factorial
        ```

        ### Feedback

        Usa los botones de feedback para mejorar el sistema:
        - 👍 cuando la respuesta es util
        - 👎 cuando la respuesta no es correcta

        El sistema aprende de tu feedback para mejorar futuras respuestas.
        """)

    # Inicializar agente al cargar
    demo.load(
        lambda: initialize_agent(True, True, True, True, True, True),
        outputs=[]
    )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  THAU AGI v2 - Interfaz Gradio")
    print("=" * 60)
    print("\n  Abriendo en http://localhost:7860")
    print("  Presiona Ctrl+C para detener\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
