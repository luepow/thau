#!/usr/bin/env python3
"""
THAU GUI - Interfaz Grafica para probar todas las capacidades de THAU

Ejecutar con:
    python thau_gui.py

Abre en: http://localhost:7860
"""

import gradio as gr
import requests
import json
import os
import re
from datetime import datetime
from pathlib import Path


# ============================================================================
# CONFIGURACION
# ============================================================================

OLLAMA_URL = "http://localhost:11434"
MODELS = ["thau:spanish", "thau:clean", "thau", "llama3.1:8b", "mistral", "gemma2"]
DEFAULT_MODEL = "thau:spanish"

# Directorios
DATA_DIR = Path("./data")
IMAGES_DIR = DATA_DIR / "generated_images"
LOGS_DIR = DATA_DIR / "logs"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# FUNCIONES DE CHAT
# ============================================================================

def get_available_models():
    """Lista modelos disponibles en Ollama"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            return models if models else MODELS
    except:
        pass
    return MODELS


def chat_with_model(message, history, model, temperature, system_prompt):
    """Chatea con el modelo seleccionado"""
    if not message.strip():
        return "", history

    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Agregar historial (nuevo formato Gradio 6.0)
    for msg in history:
        if isinstance(msg, dict):
            messages.append(msg)
        else:
            # Compatibilidad con formato antiguo
            user_msg, assistant_msg = msg
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": message})

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature}
            },
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            assistant_message = result.get("message", {}).get("content", "")
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": assistant_message})
            return "", history
        else:
            error_msg = f"Error: {response.status_code}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return "", history

    except Exception as e:
        error_msg = f"Error de conexion: {str(e)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return "", history


# ============================================================================
# FUNCIONES DE GENERACION DE IMAGENES
# ============================================================================

def generate_image(prompt, width, height):
    """Genera imagen usando Pollinations.ai"""
    if not prompt.strip():
        return None, "Por favor ingresa un prompt"

    try:
        import urllib.parse
        encoded_prompt = urllib.parse.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true"

        response = requests.get(url, timeout=60)

        if response.status_code == 200:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gui_{timestamp}.png"
            filepath = IMAGES_DIR / filename

            with open(filepath, "wb") as f:
                f.write(response.content)

            return str(filepath), f"Imagen guardada en: {filepath}"
        else:
            return None, f"Error: {response.status_code}"

    except Exception as e:
        return None, f"Error: {str(e)}"


def generate_image_with_sd(prompt, negative_prompt, steps, cfg_scale):
    """Genera imagen usando Stable Diffusion local (si esta disponible)"""
    try:
        from capabilities.tools.image_generation import ImageGenerator
        gen = ImageGenerator(backend="stable_diffusion")
        result = gen.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            cfg_scale=cfg_scale
        )

        if result.get("success"):
            return result["images"][0], f"Generada con Stable Diffusion"
        else:
            return None, result.get("error", "Error desconocido")
    except Exception as e:
        return None, f"SD no disponible: {str(e)}"


# ============================================================================
# FUNCIONES DE VOZ
# ============================================================================

def transcribe_audio(audio_path):
    """Transcribe audio a texto"""
    if audio_path is None:
        return "No se recibio audio"

    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()

        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio, language="es-ES")
        return text

    except Exception as e:
        return f"Error de transcripcion: {str(e)}"


def voice_chat(audio_path, history, model, temperature):
    """Chat por voz: transcribe y responde"""
    if audio_path is None:
        return history, "No se recibio audio"

    # Transcribir
    text = transcribe_audio(audio_path)

    if text.startswith("Error"):
        return history, text

    # Chatear
    _, new_history = chat_with_model(text, history, model, temperature, "")

    return new_history, f"Transcripcion: {text}"


# ============================================================================
# FUNCIONES DE ENTRENAMIENTO
# ============================================================================

def get_training_stats():
    """Obtiene estadisticas de entrenamiento"""
    stats_file = LOGS_DIR / "integrated_stats.json"

    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        return f"""
## Estadisticas de Entrenamiento

- **Edad Cognitiva**: {stats.get('current_age', 'N/A')}
- **Ejemplos Entrenados**: {stats.get('total_trained', 'N/A')}
- **Sesiones Intensivas**: {stats.get('intensive_sessions', 'N/A')}
- **Ultima Actualizacion**: {stats.get('last_update', 'N/A')}

### Por Categoria:
```json
{json.dumps(stats.get('by_category', {}), indent=2)}
```
"""
    else:
        return "No hay estadisticas disponibles"


def get_training_queue():
    """Lista archivos en cola de entrenamiento"""
    queue_dir = DATA_DIR / "training_queue"

    if queue_dir.exists():
        files = list(queue_dir.glob("*.json"))
        return f"Archivos en cola: {len(files)}"

    return "Cola vacia"


def start_training_session(category, questions, age):
    """Inicia una sesion de entrenamiento"""
    try:
        import subprocess
        cmd = f"python scripts/intensive_learning.py --category {category} --questions {questions} --age {age} --delay 1.0"
        subprocess.Popen(cmd, shell=True)
        return f"Sesion iniciada: {category} ({questions} preguntas)"
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# FUNCIONES DE HERRAMIENTAS
# ============================================================================

def execute_python_code(code):
    """Ejecuta codigo Python"""
    try:
        import io
        import contextlib

        output = io.StringIO()
        namespace = {"__builtins__": __builtins__}

        with contextlib.redirect_stdout(output):
            exec(code, namespace)

        stdout = output.getvalue()
        result = namespace.get('result', stdout if stdout else 'Codigo ejecutado')

        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def web_search(query):
    """Busqueda web simple"""
    try:
        url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; ThauBot/1.0)'}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            snippets = re.findall(r'class="result__snippet">(.*?)</a>', response.text)[:5]
            results = []
            for i, snippet in enumerate(snippets):
                clean = re.sub(r'<[^>]+>', '', snippet)
                results.append(f"{i+1}. {clean[:200]}")

            return "\n\n".join(results) if results else "Sin resultados"

        return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# INTERFAZ GRADIO
# ============================================================================

def create_interface():
    """Crea la interfaz Gradio"""

    available_models = get_available_models()

    with gr.Blocks(title="THAU GUI") as demo:

        gr.Markdown("""
        # THAU - Interfaz de Pruebas

        Prueba todas las capacidades de THAU: chat, generacion de imagenes, voz, herramientas.
        """)

        with gr.Tabs():

            # ================================================================
            # TAB: CHAT
            # ================================================================
            with gr.TabItem("Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=500, label="Conversacion")
                        msg = gr.Textbox(
                            label="Mensaje",
                            placeholder="Escribe tu mensaje aqui...",
                            lines=2
                        )
                        with gr.Row():
                            send_btn = gr.Button("Enviar", variant="primary")
                            clear_btn = gr.Button("Limpiar")

                    with gr.Column(scale=1):
                        model_dropdown = gr.Dropdown(
                            choices=available_models,
                            value=DEFAULT_MODEL if DEFAULT_MODEL in available_models else available_models[0] if available_models else "thau",
                            label="Modelo"
                        )
                        temperature = gr.Slider(0, 2, value=0.7, label="Temperatura")
                        system_prompt = gr.Textbox(
                            label="System Prompt",
                            placeholder="Instrucciones para el modelo...",
                            lines=4
                        )

                send_btn.click(
                    chat_with_model,
                    inputs=[msg, chatbot, model_dropdown, temperature, system_prompt],
                    outputs=[msg, chatbot]
                )
                msg.submit(
                    chat_with_model,
                    inputs=[msg, chatbot, model_dropdown, temperature, system_prompt],
                    outputs=[msg, chatbot]
                )
                clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

            # ================================================================
            # TAB: GENERACION DE IMAGENES
            # ================================================================
            with gr.TabItem("Imagenes"):
                gr.Markdown("## Generacion de Imagenes con IA")

                with gr.Row():
                    with gr.Column():
                        img_prompt = gr.Textbox(
                            label="Prompt (descripcion de la imagen)",
                            placeholder="a cute robot learning to code, digital art",
                            lines=3
                        )
                        with gr.Row():
                            img_width = gr.Slider(256, 1024, value=512, step=64, label="Ancho")
                            img_height = gr.Slider(256, 1024, value=512, step=64, label="Alto")

                        generate_btn = gr.Button("Generar Imagen", variant="primary")
                        img_status = gr.Textbox(label="Estado", interactive=False)

                    with gr.Column():
                        output_image = gr.Image(label="Imagen Generada", type="filepath")

                generate_btn.click(
                    generate_image,
                    inputs=[img_prompt, img_width, img_height],
                    outputs=[output_image, img_status]
                )

                gr.Markdown("""
                ### Consejos para prompts:
                - Usa ingles para mejores resultados
                - Agrega estilos: "digital art", "oil painting", "photorealistic"
                - Agrega calidad: "4k", "highly detailed", "professional"
                """)

            # ================================================================
            # TAB: VOZ
            # ================================================================
            with gr.TabItem("Voz"):
                gr.Markdown("## Chat por Voz")

                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            label="Graba tu mensaje",
                            sources=["microphone"],
                            type="filepath"
                        )
                        voice_model = gr.Dropdown(
                            choices=available_models,
                            value=DEFAULT_MODEL if DEFAULT_MODEL in available_models else available_models[0] if available_models else "thau",
                            label="Modelo"
                        )
                        voice_temp = gr.Slider(0, 2, value=0.7, label="Temperatura")
                        voice_btn = gr.Button("Enviar Audio", variant="primary")
                        transcription = gr.Textbox(label="Transcripcion", interactive=False)

                    with gr.Column():
                        voice_chatbot = gr.Chatbot(height=400, label="Conversacion por Voz")
                        voice_clear = gr.Button("Limpiar Conversacion")

                voice_btn.click(
                    voice_chat,
                    inputs=[audio_input, voice_chatbot, voice_model, voice_temp],
                    outputs=[voice_chatbot, transcription]
                )
                voice_clear.click(lambda: [], outputs=[voice_chatbot])

            # ================================================================
            # TAB: HERRAMIENTAS
            # ================================================================
            with gr.TabItem("Herramientas"):
                gr.Markdown("## Herramientas del Agente")

                with gr.Tabs():
                    with gr.TabItem("Ejecutar Python"):
                        code_input = gr.Code(
                            label="Codigo Python",
                            language="python",
                            value="# Escribe tu codigo aqui\nresult = 2 + 2\nprint(f'Resultado: {result}')"
                        )
                        run_btn = gr.Button("Ejecutar", variant="primary")
                        code_output = gr.Textbox(label="Salida", lines=10)

                        run_btn.click(execute_python_code, inputs=[code_input], outputs=[code_output])

                    with gr.TabItem("Busqueda Web"):
                        search_query = gr.Textbox(
                            label="Busqueda",
                            placeholder="Python asyncio tutorial"
                        )
                        search_btn = gr.Button("Buscar", variant="primary")
                        search_results = gr.Textbox(label="Resultados", lines=15)

                        search_btn.click(web_search, inputs=[search_query], outputs=[search_results])

            # ================================================================
            # TAB: ENTRENAMIENTO
            # ================================================================
            with gr.TabItem("Entrenamiento"):
                gr.Markdown("## Estado del Entrenamiento de THAU")

                with gr.Row():
                    with gr.Column():
                        refresh_stats = gr.Button("Actualizar Estadisticas")
                        stats_display = gr.Markdown(get_training_stats())

                        refresh_stats.click(get_training_stats, outputs=[stats_display])

                    with gr.Column():
                        gr.Markdown("### Iniciar Sesion de Entrenamiento")
                        train_category = gr.Dropdown(
                            choices=[
                                "programacion_basica",
                                "programacion_avanzada",
                                "python_advanced",
                                "javascript",
                                "mcp_tools",
                                "patrones_diseno"
                            ],
                            value="programacion_basica",
                            label="Categoria"
                        )
                        train_questions = gr.Slider(10, 100, value=50, step=10, label="Preguntas")
                        train_age = gr.Slider(1, 15, value=12, step=1, label="Edad Cognitiva")
                        train_btn = gr.Button("Iniciar Entrenamiento", variant="primary")
                        train_status = gr.Textbox(label="Estado", interactive=False)

                        train_btn.click(
                            start_training_session,
                            inputs=[train_category, train_questions, train_age],
                            outputs=[train_status]
                        )

                        queue_status = gr.Textbox(
                            label="Cola de Entrenamiento",
                            value=get_training_queue(),
                            interactive=False
                        )

            # ================================================================
            # TAB: ACERCA DE
            # ================================================================
            with gr.TabItem("Acerca de"):
                gr.Markdown("""
                ## THAU - Self-Learning AI

                **THAU** es un framework para desarrollar LLMs con capacidades de aprendizaje incremental.

                ### Componentes:

                1. **Modelo LLM** (~367M parametros)
                   - Arquitectura Transformer decoder-only
                   - Sistema de edades cognitivas (0-15)
                   - Entrenamiento continuo

                2. **Framework de Agente**
                   - Tool calling (MCP)
                   - Generacion de imagenes
                   - Reconocimiento de voz
                   - Busqueda web

                3. **Sistema de Memoria**
                   - Memoria a corto plazo
                   - Memoria a largo plazo (ChromaDB)
                   - Memoria episodica (SQLite)

                ### Tecnologias:
                - Python 3.10+
                - PyTorch 2.0+
                - Transformers (HuggingFace)
                - Ollama (inferencia)
                - Gradio (GUI)

                ---

                **Version**: 1.0.0
                **Repositorio**: github.com/thau-llm
                """)

        gr.Markdown("---\n*THAU GUI v1.0 - Desarrollado para probar capacidades del modelo*")

    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  THAU GUI - Interfaz Grafica")
    print("=" * 60)
    print("\nIniciando servidor...")

    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
