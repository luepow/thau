#!/usr/bin/env python3
"""
THAU Developer Studio - Interfaz gráfica para desarrollo autónomo

Esta aplicación permite:
1. Seleccionar una carpeta de trabajo
2. Describir la aplicación a crear
3. THAU planifica y desarrolla automáticamente
4. Preview y ejecución de la aplicación generada
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
import subprocess
import threading
import time
import json
import signal
from typing import Optional, Generator
from datetime import datetime

import gradio as gr

# Import developer agent
from capabilities.agent.developer_agent import DeveloperAgent, PROJECT_TEMPLATES


# Global state
class AppState:
    """Global application state."""
    def __init__(self):
        self.current_project_path: Optional[str] = None
        self.agent: Optional[DeveloperAgent] = None
        self.running_server: Optional[subprocess.Popen] = None
        self.server_port: int = 5000
        self.ollama_models: list = []

    def get_ollama_models(self) -> list:
        """Get available Ollama models."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if parts:
                            models.append(parts[0])
                # Prioritize THAU models
                thau_models = [m for m in models if 'thau' in m.lower()]
                other_models = [m for m in models if 'thau' not in m.lower()]
                return thau_models + other_models
        except Exception as e:
            print(f"Error getting models: {e}")
        return ["thau:agi-v3", "thau:reasoning", "llama3.2:latest"]


state = AppState()


def check_ollama_status() -> str:
    """Check if Ollama is running."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.ok:
            return "✅ Ollama activo"
    except:
        pass
    return "❌ Ollama no disponible"


def refresh_models():
    """Refresh the list of available models."""
    state.ollama_models = state.get_ollama_models()
    return gr.Dropdown(choices=state.ollama_models, value=state.ollama_models[0] if state.ollama_models else None)


def list_directory_contents(path: str) -> str:
    """List contents of a directory."""
    if not path or not os.path.exists(path):
        return "📁 Selecciona una carpeta válida"

    items = []
    try:
        for item in sorted(os.listdir(path)):
            full_path = os.path.join(path, item)
            if item.startswith('.'):
                continue
            if os.path.isdir(full_path):
                items.append(f"📁 {item}/")
            else:
                size = os.path.getsize(full_path)
                items.append(f"📄 {item} ({size:,} bytes)")
    except Exception as e:
        return f"❌ Error: {e}"

    if not items:
        return "📁 (carpeta vacía)"

    return "\n".join(items)


def read_file_content(path: str, filename: str) -> str:
    """Read content of a file."""
    if not path or not filename:
        return ""

    file_path = os.path.join(path, filename)
    if not os.path.exists(file_path):
        return f"Archivo no encontrado: {filename}"

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error leyendo archivo: {e}"


def get_tree_structure(path: str, prefix: str = "", max_depth: int = 3, current_depth: int = 0) -> str:
    """Get tree structure of a directory."""
    if current_depth >= max_depth:
        return prefix + "...\n"

    if not os.path.exists(path):
        return ""

    result = []
    try:
        items = sorted(os.listdir(path))
        items = [i for i in items if not i.startswith('.')]

        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            full_path = os.path.join(path, item)

            if os.path.isdir(full_path):
                result.append(f"{prefix}{current_prefix}📁 {item}/")
                next_prefix = prefix + ("    " if is_last else "│   ")
                result.append(get_tree_structure(full_path, next_prefix, max_depth, current_depth + 1))
            else:
                result.append(f"{prefix}{current_prefix}📄 {item}")
    except Exception as e:
        result.append(f"{prefix}Error: {e}")

    return "\n".join(result)


def start_development(
    project_path: str,
    task_description: str,
    model_name: str,
    project_template: str,
    max_iterations: int
) -> Generator:
    """Start autonomous development process."""

    if not project_path:
        yield "❌ Por favor selecciona una carpeta de trabajo"
        return

    if not task_description:
        yield "❌ Por favor describe la aplicación que quieres crear"
        return

    # Create project directory if needed
    os.makedirs(project_path, exist_ok=True)

    # Initialize agent
    state.current_project_path = project_path
    state.agent = DeveloperAgent(
        working_dir=project_path,
        ollama_model=model_name,
        max_iterations=max_iterations
    )

    yield f"""
╔══════════════════════════════════════════════════════════════╗
║  🚀 THAU DEVELOPER STUDIO                                     ║
╠══════════════════════════════════════════════════════════════╣
║  Proyecto: {os.path.basename(project_path):<47} ║
║  Modelo: {model_name:<49} ║
╚══════════════════════════════════════════════════════════════╝

📋 TAREA: {task_description}

"""

    # Add template info if selected
    if project_template and project_template != "Ninguno":
        template = PROJECT_TEMPLATES.get(project_template, {})
        yield f"📦 Template: {project_template}\n"
        yield f"   {template.get('description', '')}\n\n"
        task_description = f"{task_description}\n\nUsa el template {project_template} como base."

    yield "=" * 60 + "\n\n"

    # Run the agent
    try:
        full_output = ""
        for output in state.agent.run(task_description):
            full_output += output
            yield full_output

    except Exception as e:
        yield full_output + f"\n\n❌ Error: {str(e)}"


def stop_development():
    """Stop the development process."""
    if state.agent and state.agent.state.running_process:
        state.agent.tool_stop_server()
    return "⏹️ Proceso detenido"


def start_preview_server(project_path: str, command: str, port: int) -> str:
    """Start a preview server for the generated application."""
    if not project_path or not os.path.exists(project_path):
        return "❌ Carpeta del proyecto no válida"

    stop_preview_server()

    try:
        state.running_server = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_path,
            preexec_fn=os.setsid
        )

        state.server_port = port
        time.sleep(2)

        if state.running_server.poll() is None:
            return f"""✅ Servidor iniciado

🌐 URL: http://localhost:{port}

Comando: {command}
"""
        else:
            stdout, stderr = state.running_server.communicate()
            return f"❌ Error iniciando servidor:\n{stderr.decode()}"

    except Exception as e:
        return f"❌ Error: {str(e)}"


def stop_preview_server() -> str:
    """Stop the preview server."""
    if state.running_server:
        try:
            os.killpg(os.getpgid(state.running_server.pid), signal.SIGTERM)
        except:
            pass
        state.running_server = None
    return "⏹️ Servidor detenido"


def export_project(project_path: str):
    """Create a zip file of the project."""
    if not project_path or not os.path.exists(project_path):
        return None, "❌ Carpeta del proyecto no válida"

    import shutil

    export_dir = project_root / "export" / "projects"
    export_dir.mkdir(parents=True, exist_ok=True)

    project_name = os.path.basename(project_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"{project_name}_{timestamp}"
    zip_path = export_dir / zip_name

    shutil.make_archive(str(zip_path), 'zip', project_path)

    return str(zip_path) + ".zip", f"✅ Proyecto exportado: {zip_name}.zip"


def create_ui():
    """Create the Gradio interface."""

    state.ollama_models = state.get_ollama_models()

    with gr.Blocks(title="THAU Developer Studio", fill_width=True) as app:

        gr.Markdown("""
        # 🚀 THAU Developer Studio

        **Desarrollo autónomo de aplicaciones con IA**

        Describe lo que quieres crear y THAU lo construirá automáticamente.
        """)

        with gr.Row():
            ollama_status = gr.Textbox(
                value=check_ollama_status(),
                label="Estado Ollama",
                interactive=False,
                scale=1
            )
            refresh_btn = gr.Button("🔄 Refrescar", scale=0)

        with gr.Tabs():

            # TAB 1: CONFIGURACIÓN
            with gr.Tab("⚙️ Configuración"):
                with gr.Row():
                    with gr.Column(scale=2):
                        project_path = gr.Textbox(
                            label="📁 Carpeta del Proyecto",
                            placeholder="/ruta/a/tu/proyecto",
                            info="Ruta donde se creará el proyecto"
                        )

                        with gr.Row():
                            browse_home = gr.Button("🏠 Home")
                            browse_desktop = gr.Button("🖥️ Desktop")
                            browse_documents = gr.Button("📄 Documents")

                        task_description = gr.Textbox(
                            label="📝 Descripción del Proyecto",
                            placeholder="Describe la aplicación...\n\nEjemplo: Una aplicación web con Flask que muestre un dashboard con gráficos.",
                            lines=6,
                            info="Sé específico sobre las funcionalidades"
                        )

                    with gr.Column(scale=1):
                        model_dropdown = gr.Dropdown(
                            choices=state.ollama_models,
                            value=state.ollama_models[0] if state.ollama_models else None,
                            label="🤖 Modelo THAU",
                            info="Modelo de Ollama a usar"
                        )

                        template_dropdown = gr.Dropdown(
                            choices=["Ninguno"] + list(PROJECT_TEMPLATES.keys()),
                            value="Ninguno",
                            label="📦 Template Base",
                            info="Template inicial para el proyecto"
                        )

                        max_iterations = gr.Slider(
                            minimum=5,
                            maximum=50,
                            value=20,
                            step=5,
                            label="🔄 Iteraciones Máximas"
                        )

                        with gr.Row():
                            start_btn = gr.Button("🚀 Iniciar Desarrollo", variant="primary")
                            stop_btn = gr.Button("⏹️ Detener", variant="stop")

                browse_home.click(fn=lambda: str(Path.home()), outputs=project_path)
                browse_desktop.click(fn=lambda: str(Path.home() / "Desktop"), outputs=project_path)
                browse_documents.click(fn=lambda: str(Path.home() / "Documents"), outputs=project_path)

            # TAB 2: DESARROLLO
            with gr.Tab("🔨 Desarrollo"):
                with gr.Row():
                    with gr.Column(scale=2):
                        output_box = gr.Textbox(
                            label="📺 Salida del Agente",
                            lines=30,
                            max_lines=50,
                            interactive=False
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 📁 Estructura del Proyecto")
                        file_tree = gr.Textbox(
                            label="",
                            lines=20,
                            max_lines=30,
                            interactive=False
                        )
                        refresh_tree_btn = gr.Button("🔄 Actualizar Árbol")

            # TAB 3: ARCHIVOS
            with gr.Tab("📄 Archivos"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📁 Archivos del Proyecto")
                        file_list = gr.Textbox(
                            label="",
                            lines=20,
                            interactive=False
                        )
                        file_selector = gr.Textbox(
                            label="Nombre del archivo",
                            placeholder="Nombre del archivo a ver"
                        )
                        view_file_btn = gr.Button("👁️ Ver Archivo")

                    with gr.Column(scale=2):
                        file_content = gr.Code(
                            label="Contenido del Archivo",
                            language="python",
                            lines=25
                        )

            # TAB 4: PREVIEW
            with gr.Tab("👁️ Preview"):
                with gr.Row():
                    server_command = gr.Textbox(
                        label="Comando del servidor",
                        value="python app.py",
                        placeholder="python app.py, npm start, etc."
                    )
                    server_port = gr.Number(
                        label="Puerto",
                        value=5000,
                        minimum=3000,
                        maximum=9999
                    )

                with gr.Row():
                    start_server_btn = gr.Button("▶️ Iniciar Servidor", variant="primary")
                    stop_server_btn = gr.Button("⏹️ Detener Servidor", variant="stop")
                    open_browser_btn = gr.Button("🌐 Abrir en Navegador")

                server_status = gr.Textbox(
                    label="Estado del Servidor",
                    interactive=False
                )

            # TAB 5: EXPORTAR
            with gr.Tab("📦 Exportar"):
                gr.Markdown("""
                ### Exportar Proyecto
                Descarga tu proyecto como archivo ZIP.
                """)

                export_btn = gr.Button("📦 Exportar como ZIP", variant="primary")
                export_status = gr.Textbox(label="Estado", interactive=False)
                export_file = gr.File(label="Archivo Exportado")

        # EVENT HANDLERS
        refresh_btn.click(
            fn=lambda: (check_ollama_status(), refresh_models()),
            outputs=[ollama_status, model_dropdown]
        )

        start_btn.click(
            fn=start_development,
            inputs=[project_path, task_description, model_dropdown, template_dropdown, max_iterations],
            outputs=output_box
        )

        stop_btn.click(fn=stop_development, outputs=output_box)

        def update_tree(path):
            if path and os.path.exists(path):
                return get_tree_structure(path)
            return "Selecciona una carpeta válida"

        refresh_tree_btn.click(fn=update_tree, inputs=project_path, outputs=file_tree)

        project_path.change(
            fn=lambda p: (list_directory_contents(p), update_tree(p)),
            inputs=project_path,
            outputs=[file_list, file_tree]
        )

        view_file_btn.click(
            fn=read_file_content,
            inputs=[project_path, file_selector],
            outputs=file_content
        )

        start_server_btn.click(
            fn=start_preview_server,
            inputs=[project_path, server_command, server_port],
            outputs=server_status
        )

        stop_server_btn.click(fn=stop_preview_server, outputs=server_status)

        open_browser_btn.click(
            fn=lambda port: subprocess.run(["open", f"http://localhost:{int(port)}"]),
            inputs=server_port
        )

        def do_export(path):
            zip_path, status = export_project(path)
            return status, zip_path

        export_btn.click(fn=do_export, inputs=project_path, outputs=[export_status, export_file])

        gr.Markdown("""
        ---
        ### 💡 Ideas de Proyectos

        1. **Web App Flask**: "Crea una app web con Flask que tenga formulario de contacto"
        2. **API REST**: "Desarrolla una API REST con FastAPI para gestionar tareas"
        3. **CLI Tool**: "Crea una herramienta CLI en Python para organizar archivos"
        4. **Dashboard**: "Construye un dashboard web con gráficos de datos CSV"
        """)

    return app


def main():
    """Main entry point."""
    print("=" * 60)
    print("  THAU Developer Studio")
    print("  Desarrollo autónomo de aplicaciones con IA")
    print("=" * 60)

    status = check_ollama_status()
    print(f"\n{status}")

    if "❌" in status:
        print("\n⚠️ Ollama no está corriendo. Inicia Ollama con:")
        print("   ollama serve")
        print("\nLuego asegúrate de tener un modelo THAU:")
        print("   ollama run thau:agi-v3")

    app = create_ui()

    print("\n🚀 Iniciando servidor...")
    print("   URL: http://localhost:7861")
    print("\n   Presiona Ctrl+C para detener\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )


if __name__ == "__main__":
    main()
