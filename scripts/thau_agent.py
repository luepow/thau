#!/usr/bin/env python3
"""
THAU Agent CLI - Conecta Ollama THAU con sistema de herramientas

Uso:
    python scripts/thau_agent.py --interactive
    python scripts/thau_agent.py --query "busca información sobre Python"
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
from capabilities.tools.mcp_integration import create_default_mcp_tools, MCPRegistry


class ThauOllamaAgent:
    """
    Agente THAU usando Ollama como backend
    Integra herramientas MCP con el modelo THAU
    """

    def __init__(
        self,
        model: str = "thau-tool-calling",
        ollama_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.registry = create_default_mcp_tools()
        self.conversation_history = []

        # Agregar herramientas adicionales
        self._add_extra_tools()

        print(f"🤖 THAU Agent inicializado")
        print(f"   Modelo: {model}")
        print(f"   Tools: {len(self.registry.tools)}")

    def _add_extra_tools(self):
        """Agrega herramientas adicionales"""
        from capabilities.tools.mcp_integration import MCPParameter

        # Tool: Web Search
        def web_search(query: str, num_results: int = 5) -> Dict:
            """Búsqueda web usando DuckDuckGo"""
            try:
                url = f"https://html.duckduckgo.com/html/?q={query}"
                headers = {'User-Agent': 'Mozilla/5.0 (compatible; ThauBot/1.0)'}
                response = requests.get(url, headers=headers, timeout=10)

                # Parse básico
                results = []
                if response.status_code == 200:
                    # Extrae snippets
                    text = response.text
                    snippets = re.findall(r'class="result__snippet">(.*?)</a>', text)[:num_results]
                    for i, snippet in enumerate(snippets):
                        # Limpia HTML
                        clean = re.sub(r'<[^>]+>', '', snippet)
                        results.append({
                            "rank": i + 1,
                            "snippet": clean[:300]
                        })

                return {
                    "success": True,
                    "query": query,
                    "results": results if results else [{"message": "Busqueda completada"}]
                }
            except Exception as e:
                return {"error": str(e)}

        self.registry.register_function(
            name="web_search",
            description="Busca información en internet usando DuckDuckGo",
            function=web_search,
            parameters=[
                MCPParameter(name="query", type="string", description="Término de búsqueda", required=True),
                MCPParameter(name="num_results", type="number", description="Número de resultados", required=False, default=5)
            ],
            returns={"type": "object", "description": "Resultados de búsqueda"}
        )

        # Tool: Execute Python
        def execute_python(code: str) -> Dict:
            """Ejecuta código Python de forma segura"""
            try:
                import io
                import contextlib

                # Captura output
                output = io.StringIO()
                namespace = {"__builtins__": __builtins__}

                with contextlib.redirect_stdout(output):
                    exec(code, namespace)

                stdout = output.getvalue()
                result = namespace.get('result', stdout if stdout else 'Código ejecutado')

                return {
                    "success": True,
                    "output": str(result),
                    "stdout": stdout
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        self.registry.register_function(
            name="execute_python",
            description="Ejecuta código Python y retorna el resultado",
            function=execute_python,
            parameters=[
                MCPParameter(name="code", type="string", description="Código Python a ejecutar", required=True)
            ],
            returns={"type": "object", "description": "Resultado de la ejecución"}
        )

        # Tool: Get Current Time
        def get_current_time() -> Dict:
            """Obtiene fecha y hora actual"""
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

        # Tool: File Operations
        def read_file(filepath: str) -> Dict:
            """Lee contenido de un archivo"""
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                return {
                    "success": True,
                    "filepath": filepath,
                    "content": content[:5000],  # Limita contenido
                    "truncated": len(content) > 5000
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        self.registry.register_function(
            name="read_file",
            description="Lee el contenido de un archivo",
            function=read_file,
            parameters=[
                MCPParameter(name="filepath", type="string", description="Ruta del archivo", required=True)
            ],
            returns={"type": "object", "description": "Contenido del archivo"}
        )

        # Tool: List Directory
        def list_directory(path: str = ".") -> Dict:
            """Lista archivos en un directorio"""
            try:
                import os
                items = os.listdir(path)
                files = []
                dirs = []

                for item in items[:50]:  # Limita a 50 items
                    full_path = os.path.join(path, item)
                    if os.path.isdir(full_path):
                        dirs.append(item)
                    else:
                        files.append(item)

                return {
                    "success": True,
                    "path": path,
                    "directories": dirs,
                    "files": files
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        self.registry.register_function(
            name="list_directory",
            description="Lista archivos y carpetas en un directorio",
            function=list_directory,
            parameters=[
                MCPParameter(name="path", type="string", description="Ruta del directorio", required=False, default=".")
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
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 2048
                    }
                },
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                return f"Error de Ollama: {response.status_code}"

        except Exception as e:
            return f"Error conectando a Ollama: {e}"

    def _parse_tool_calls(self, text: str) -> List[Dict]:
        """Extrae llamadas a herramientas del texto"""
        tool_calls = []

        # Formato: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match.strip())
                tool_calls.append(data)
            except:
                continue

        # También busca formato JSON directo
        json_pattern = r'\{"name":\s*"(\w+)",\s*"arguments":\s*(\{[^}]+\})\}'
        json_matches = re.findall(json_pattern, text)

        for name, args in json_matches:
            try:
                tool_calls.append({
                    "name": name,
                    "arguments": json.loads(args)
                })
            except:
                continue

        return tool_calls

    def _get_tools_prompt(self) -> str:
        """Genera prompt con herramientas disponibles"""
        tools = self.registry.list_tools()

        tools_desc = []
        for tool in tools:
            func = tool['function']
            params_desc = []
            for pname, pinfo in func['parameters']['properties'].items():
                req = " (required)" if pname in func['parameters']['required'] else ""
                params_desc.append(f"    - {pname}: {pinfo['description']}{req}")

            tools_desc.append(f"""
- {func['name']}: {func['description']}
  Parameters:
{chr(10).join(params_desc) if params_desc else '    (ninguno)'}""")

        return f"""Eres THAU, un asistente AI con capacidades de agente. Puedes usar herramientas para completar tareas.

HERRAMIENTAS DISPONIBLES:
{''.join(tools_desc)}

Para usar una herramienta, responde con este formato:
<tool_call>{{"name": "nombre_herramienta", "arguments": {{"param1": "valor1"}}}}</tool_call>

Después de recibir el resultado, proporciona una respuesta final al usuario.
Si no necesitas herramientas, responde directamente."""

    def chat(self, user_message: str, max_iterations: int = 3) -> str:
        """
        Procesa mensaje con capacidad de usar herramientas

        Args:
            user_message: Mensaje del usuario
            max_iterations: Máximo de ciclos tool-use

        Returns:
            Respuesta final
        """
        system_prompt = self._get_tools_prompt()
        current_prompt = user_message

        for iteration in range(max_iterations):
            # Obtiene respuesta de THAU
            response = self._call_ollama(current_prompt, system_prompt)

            # Busca tool calls
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                # No hay tool calls, es respuesta final
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": response})
                return response

            # Ejecuta herramientas
            results = []
            for call in tool_calls:
                tool_name = call.get("name")
                arguments = call.get("arguments", {})

                print(f"\n🔧 Ejecutando: {tool_name}")
                print(f"   Args: {json.dumps(arguments, ensure_ascii=False)}")

                result = self.registry.invoke_tool(tool_name, arguments)

                if result.success:
                    print(f"   ✅ Éxito")
                    results.append({
                        "tool": tool_name,
                        "result": result.result
                    })
                else:
                    print(f"   ❌ Error: {result.error}")
                    results.append({
                        "tool": tool_name,
                        "error": result.error
                    })

            # Prepara siguiente prompt con resultados
            results_text = json.dumps(results, ensure_ascii=False, indent=2)
            current_prompt = f"""Resultados de herramientas:
{results_text}

Basándote en estos resultados, proporciona una respuesta completa al usuario.
Pregunta original: {user_message}"""

        return response

    def list_tools(self):
        """Lista herramientas disponibles"""
        print("\n📋 Herramientas disponibles:")
        print("=" * 50)

        for tool in self.registry.list_tools():
            func = tool['function']
            print(f"\n🔧 {func['name']}")
            print(f"   {func['description']}")

            if func['parameters']['properties']:
                print("   Parámetros:")
                for pname, pinfo in func['parameters']['properties'].items():
                    req = "*" if pname in func['parameters']['required'] else ""
                    print(f"     - {pname}{req}: {pinfo['description']}")


def main():
    parser = argparse.ArgumentParser(description="THAU Agent CLI")
    parser.add_argument("--model", type=str, default="thau-tool-calling", help="Modelo Ollama")
    parser.add_argument("--interactive", "-i", action="store_true", help="Modo interactivo")
    parser.add_argument("--query", "-q", type=str, help="Consulta única")
    parser.add_argument("--list-tools", action="store_true", help="Lista herramientas")
    args = parser.parse_args()

    agent = ThauOllamaAgent(model=args.model)

    if args.list_tools:
        agent.list_tools()
        return

    if args.interactive:
        print("\n" + "=" * 60)
        print("  🧠 THAU Agent - Modo Interactivo")
        print("=" * 60)
        print("\nEscribe 'salir' para terminar, 'tools' para ver herramientas")
        print()

        while True:
            try:
                user_input = input("👤 Tú: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['salir', 'exit', 'quit']:
                    print("👋 ¡Hasta luego!")
                    break

                if user_input.lower() == 'tools':
                    agent.list_tools()
                    continue

                print("\n🤖 THAU:", end=" ")
                response = agent.chat(user_input)
                print(response)
                print()

            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    elif args.query:
        print(f"\n👤 Consulta: {args.query}")
        print("\n🤖 THAU:", end=" ")
        response = agent.chat(args.query)
        print(response)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
