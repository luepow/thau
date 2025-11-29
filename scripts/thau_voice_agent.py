#!/usr/bin/env python3
"""
THAU Voice Agent - Agente con capacidad de escuchar por micrófono

Uso:
    python scripts/thau_voice_agent.py --voice      # Modo voz
    python scripts/thau_voice_agent.py --text       # Modo texto
    python scripts/thau_voice_agent.py --hybrid     # Ambos modos

Requisitos:
    pip install SpeechRecognition pyaudio
    brew install portaudio  # En macOS
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import re
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import THAU tools
from capabilities.tools.mcp_integration import create_default_mcp_tools, MCPRegistry, MCPParameter

# Intentar importar módulo de voz
try:
    from capabilities.audio.speech_recognition import ThauSpeechRecognition
    HAS_VOICE = True
except ImportError:
    HAS_VOICE = False


class ThauVoiceAgent:
    """
    Agente THAU con capacidades de voz
    Puede escuchar por micrófono y responder
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",  # Modelo con mejor tool calling
        ollama_url: str = "http://localhost:11434",
        voice_backend: str = "google",  # google o whisper
        language: str = "es"
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.registry = create_default_mcp_tools()
        self.conversation_history = []

        # Inicializar reconocimiento de voz
        self.speech = None
        if HAS_VOICE:
            try:
                self.speech = ThauSpeechRecognition(
                    backend=voice_backend,
                    language=language
                )
                print(f"🎤 Reconocimiento de voz inicializado ({voice_backend})")
            except Exception as e:
                print(f"⚠️ Error inicializando voz: {e}")
        else:
            print("⚠️ Módulo de voz no disponible")
            print("   Instala: pip install SpeechRecognition pyaudio")

        # Agregar herramientas
        self._add_tools()

        print(f"\n🤖 THAU Voice Agent inicializado")
        print(f"   Modelo: {model}")
        print(f"   Tools: {len(self.registry.tools)}")

    def _add_tools(self):
        """Agrega herramientas al agente"""

        # Tool: Get Current Time
        def get_current_time() -> Dict:
            now = datetime.now()
            return {
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "day_of_week": now.strftime("%A"),
                "iso": now.isoformat()
            }

        self.registry.register_function(
            name="get_current_time",
            description="Obtiene la fecha y hora actual",
            function=get_current_time,
            parameters=[],
            returns={"type": "object", "description": "Fecha y hora"}
        )

        # Tool: Web Search
        def web_search(query: str, num_results: int = 5) -> Dict:
            try:
                url = f"https://html.duckduckgo.com/html/?q={query}"
                headers = {'User-Agent': 'Mozilla/5.0 (compatible; ThauBot/1.0)'}
                response = requests.get(url, headers=headers, timeout=10)

                results = []
                if response.status_code == 200:
                    text = response.text
                    snippets = re.findall(r'class="result__snippet">(.*?)</a>', text)[:num_results]
                    for i, snippet in enumerate(snippets):
                        clean = re.sub(r'<[^>]+>', '', snippet)
                        results.append({"rank": i + 1, "snippet": clean[:300]})

                return {
                    "success": True,
                    "query": query,
                    "results": results if results else [{"message": "Busqueda completada"}]
                }
            except Exception as e:
                return {"error": str(e)}

        self.registry.register_function(
            name="web_search",
            description="Busca información en internet",
            function=web_search,
            parameters=[
                MCPParameter(name="query", type="string", description="Término de búsqueda", required=True),
            ],
            returns={"type": "object", "description": "Resultados"}
        )

        # Tool: Execute Python
        def execute_python(code: str) -> Dict:
            try:
                import io
                import contextlib
                output = io.StringIO()
                namespace = {"__builtins__": __builtins__}
                with contextlib.redirect_stdout(output):
                    exec(code, namespace)
                stdout = output.getvalue()
                result = namespace.get('result', stdout if stdout else 'Código ejecutado')
                return {"success": True, "output": str(result), "stdout": stdout}
            except Exception as e:
                return {"success": False, "error": str(e)}

        self.registry.register_function(
            name="execute_python",
            description="Ejecuta código Python",
            function=execute_python,
            parameters=[
                MCPParameter(name="code", type="string", description="Código Python", required=True)
            ],
            returns={"type": "object", "description": "Resultado"}
        )

        # Tool: Generate Image
        def generate_image(prompt: str) -> Dict:
            try:
                from capabilities.tools.image_generation import ImageGenerator
                gen = ImageGenerator(backend="pollinations")
                result = gen.generate(prompt=prompt, width=512, height=512)
                return result
            except Exception as e:
                return {"success": False, "error": str(e)}

        self.registry.register_function(
            name="generate_image",
            description="Genera una imagen a partir de un prompt de texto",
            function=generate_image,
            parameters=[
                MCPParameter(name="prompt", type="string", description="Descripción de la imagen (en inglés preferiblemente)", required=True)
            ],
            returns={"type": "object", "description": "Path de la imagen generada"}
        )

        # Tool: Read File
        def read_file(filepath: str) -> Dict:
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                return {"success": True, "filepath": filepath, "content": content[:5000]}
            except Exception as e:
                return {"success": False, "error": str(e)}

        self.registry.register_function(
            name="read_file",
            description="Lee el contenido de un archivo",
            function=read_file,
            parameters=[
                MCPParameter(name="filepath", type="string", description="Ruta del archivo", required=True)
            ],
            returns={"type": "object", "description": "Contenido"}
        )

        # Tool: List Directory
        def list_directory(path: str = ".") -> Dict:
            try:
                import os
                items = os.listdir(path)
                files = []
                dirs = []
                for item in items[:50]:
                    full_path = os.path.join(path, item)
                    if os.path.isdir(full_path):
                        dirs.append(item)
                    else:
                        files.append(item)
                return {"success": True, "path": path, "directories": dirs, "files": files}
            except Exception as e:
                return {"success": False, "error": str(e)}

        self.registry.register_function(
            name="list_directory",
            description="Lista archivos en un directorio",
            function=list_directory,
            parameters=[
                MCPParameter(name="path", type="string", description="Ruta", required=False, default=".")
            ],
            returns={"type": "object", "description": "Lista de archivos"}
        )

    def _call_ollama(self, prompt: str, system_prompt: str = None) -> str:
        """Llama a Ollama API"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 2048}
                },
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                return f"Error de Ollama: {response.status_code}"

        except Exception as e:
            return f"Error: {e}"

    def _parse_tool_calls(self, text: str) -> List[Dict]:
        """Extrae llamadas a herramientas"""
        tool_calls = []

        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match.strip())
                tool_calls.append(data)
            except:
                continue

        return tool_calls

    def _get_tools_prompt(self) -> str:
        """Genera prompt con herramientas"""
        tools = self.registry.list_tools()
        tools_desc = []

        for tool in tools:
            func = tool['function']
            params = []
            for pname, pinfo in func['parameters']['properties'].items():
                req = " (required)" if pname in func['parameters']['required'] else ""
                params.append(f"    - {pname}: {pinfo['description']}{req}")

            tools_desc.append(f"""
- {func['name']}: {func['description']}
  Parameters:
{chr(10).join(params) if params else '    (ninguno)'}""")

        return f"""Eres THAU, un asistente AI con capacidades de agente. Puedes usar herramientas.

HERRAMIENTAS DISPONIBLES:
{''.join(tools_desc)}

Para usar una herramienta, responde EXACTAMENTE con este formato:
<tool_call>{{"name": "nombre", "arguments": {{"param": "valor"}}}}</tool_call>

Si no necesitas herramientas, responde directamente.
Responde siempre en español."""

    def chat(self, user_message: str, max_iterations: int = 3) -> str:
        """Procesa mensaje con herramientas"""
        system_prompt = self._get_tools_prompt()
        current_prompt = user_message

        for iteration in range(max_iterations):
            response = self._call_ollama(current_prompt, system_prompt)
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": response})
                return response

            # Ejecutar herramientas
            results = []
            for call in tool_calls:
                tool_name = call.get("name")
                arguments = call.get("arguments", {})

                print(f"\n🔧 Ejecutando: {tool_name}")
                print(f"   Args: {json.dumps(arguments, ensure_ascii=False)}")

                result = self.registry.invoke_tool(tool_name, arguments)

                if result.success:
                    print(f"   ✅ Éxito")
                    results.append({"tool": tool_name, "result": result.result})
                else:
                    print(f"   ❌ Error: {result.error}")
                    results.append({"tool": tool_name, "error": result.error})

            results_text = json.dumps(results, ensure_ascii=False, indent=2)
            current_prompt = f"""Resultados de herramientas:
{results_text}

Basándote en estos resultados, proporciona una respuesta al usuario.
Pregunta original: {user_message}"""

        return response

    def listen_and_respond(self) -> Optional[str]:
        """Escucha por micrófono y responde"""
        if not self.speech:
            print("❌ Sistema de voz no disponible")
            return None

        print("\n🎤 Escuchando... (habla ahora)")
        result = self.speech.listen_once(timeout=15)

        if result["success"]:
            text = result["text"]
            print(f"\n👤 Escuché: {text}")
            print(f"\n🤖 THAU: ", end="")
            response = self.chat(text)
            print(response)
            return response
        else:
            print(f"⚠️ {result['error']}")
            return None

    def run_voice_mode(self):
        """Modo de voz continuo"""
        if not self.speech:
            print("❌ Sistema de voz no disponible")
            print("   Instala: pip install SpeechRecognition pyaudio")
            return

        print("\n" + "=" * 60)
        print("  🎤 THAU Voice Agent - Modo Voz")
        print("=" * 60)
        print("\nDi 'salir' o 'detener' para terminar")

        # Calibrar
        self.speech.calibrate(duration=2)

        while True:
            result = self.speech.listen_once(timeout=30)

            if result["success"]:
                text = result["text"]
                print(f"\n👤 {text}")

                # Verificar salida
                if any(word in text.lower() for word in ['salir', 'detener', 'exit', 'quit']):
                    print("\n👋 ¡Hasta luego!")
                    break

                # Responder
                print(f"\n🤖 THAU: ", end="")
                response = self.chat(text)
                print(response)

    def run_text_mode(self):
        """Modo de texto interactivo"""
        print("\n" + "=" * 60)
        print("  🤖 THAU Voice Agent - Modo Texto")
        print("=" * 60)
        print("\nEscribe 'salir' para terminar, 'tools' para ver herramientas")

        while True:
            try:
                user_input = input("\n👤 Tú: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['salir', 'exit', 'quit']:
                    print("👋 ¡Hasta luego!")
                    break

                if user_input.lower() == 'tools':
                    self._list_tools()
                    continue

                print(f"\n🤖 THAU: ", end="")
                response = self.chat(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break

    def run_hybrid_mode(self):
        """Modo híbrido: voz o texto"""
        print("\n" + "=" * 60)
        print("  🎤 THAU Voice Agent - Modo Híbrido")
        print("=" * 60)
        print("\nComandos:")
        print("  - Escribe tu mensaje para texto")
        print("  - Escribe 'voz' para activar micrófono")
        print("  - Escribe 'salir' para terminar")

        while True:
            try:
                user_input = input("\n👤 [texto/voz]: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['salir', 'exit']:
                    print("👋 ¡Hasta luego!")
                    break

                if user_input.lower() in ['voz', 'mic', 'escucha']:
                    self.listen_and_respond()
                    continue

                print(f"\n🤖 THAU: ", end="")
                response = self.chat(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break

    def _list_tools(self):
        """Lista herramientas disponibles"""
        print("\n📋 Herramientas disponibles:")
        print("=" * 50)

        for tool in self.registry.list_tools():
            func = tool['function']
            print(f"\n🔧 {func['name']}")
            print(f"   {func['description']}")


def main():
    parser = argparse.ArgumentParser(description="THAU Voice Agent")
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="Modelo Ollama")
    parser.add_argument("--voice", "-v", action="store_true", help="Modo voz")
    parser.add_argument("--text", "-t", action="store_true", help="Modo texto")
    parser.add_argument("--hybrid", "-H", action="store_true", help="Modo híbrido")
    parser.add_argument("--backend", type=str, default="google", choices=["google", "whisper"], help="Backend de voz")
    parser.add_argument("--language", type=str, default="es", help="Idioma")
    args = parser.parse_args()

    agent = ThauVoiceAgent(
        model=args.model,
        voice_backend=args.backend,
        language=args.language
    )

    if args.voice:
        agent.run_voice_mode()
    elif args.text:
        agent.run_text_mode()
    elif args.hybrid:
        agent.run_hybrid_mode()
    else:
        # Por defecto: modo híbrido
        agent.run_hybrid_mode()


if __name__ == "__main__":
    main()
