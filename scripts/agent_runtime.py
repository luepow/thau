#!/usr/bin/env python3
"""
Runtime de Agente AI con Tool Calling real
Integración con LangChain, web search, code execution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re
import json
import requests
from typing import Dict, List, Any
import subprocess


class ToolExecutor:
    """Ejecutor de herramientas para el agente"""

    @staticmethod
    def web_search(query: str, num_results: int = 5) -> List[Dict]:
        """Búsqueda web usando DuckDuckGo"""
        try:
            # Usar DuckDuckGo HTML API (no requiere API key)
            url = f"https://html.duckduckgo.com/html/?q={query}"
            headers = {'User-Agent': 'Mozilla/5.0'}

            response = requests.get(url, headers=headers, timeout=10)

            # Parse simple (en producción usa BeautifulSoup)
            results = []
            snippets = response.text.split('result__snippet')[:num_results]

            for snippet in snippets:
                # Extracción básica
                results.append({
                    "title": "Result",
                    "snippet": snippet[:200] if len(snippet) > 200 else snippet,
                    "url": "https://duckduckgo.com"
                })

            return results
        except Exception as e:
            return [{"error": str(e)}]

    @staticmethod
    def execute_python(code: str) -> Dict[str, Any]:
        """Ejecuta código Python de forma segura"""
        try:
            # En producción, usa sandbox (Docker, PyPy sandbox, etc.)
            # Por ahora, ejecutamos con restricciones

            allowed_imports = ['math', 'json', 'datetime', 'collections']

            # Verificar imports
            if 'import' in code:
                for line in code.split('\n'):
                    if 'import' in line:
                        module = line.split('import')[1].strip().split()[0]
                        if module not in allowed_imports:
                            return {"error": f"Import not allowed: {module}"}

            # Ejecutar en namespace aislado
            namespace = {}
            exec(code, {"__builtins__": __builtins__}, namespace)

            # Capturar output
            return {
                "success": True,
                "result": str(namespace.get('result', 'Code executed successfully')),
                "namespace": {k: str(v) for k, v in namespace.items() if not k.startswith('_')}
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def code_analysis(code: str, language: str = "python") -> Dict[str, Any]:
        """Análisis estático de código"""
        issues = []

        # Análisis simple de seguridad
        dangerous_patterns = {
            'eval': 'Uso de eval() es peligroso',
            'exec': 'Uso de exec() es peligroso',
            '__import__': 'Import dinámico puede ser peligroso',
            'os.system': 'Ejecución de comandos del sistema',
            'subprocess': 'Ejecución de procesos externos',
        }

        for pattern, message in dangerous_patterns.items():
            if pattern in code:
                issues.append({
                    "severity": "high",
                    "type": "security",
                    "message": message,
                    "pattern": pattern
                })

        # SQL Injection
        if '${' in code or f'"{' in code or "f'" in code:
            if 'SELECT' in code.upper() or 'INSERT' in code.upper():
                issues.append({
                    "severity": "critical",
                    "type": "sql_injection",
                    "message": "Posible SQL Injection con string interpolation",
                })

        return {
            "issues": issues,
            "severity_counts": {
                "critical": len([i for i in issues if i['severity'] == 'critical']),
                "high": len([i for i in issues if i['severity'] == 'high']),
            }
        }


class AgentAI:
    """Agente AI con capacidades de razonamiento y tool calling"""

    def __init__(self, model_path: str = "./data/checkpoints/agent-expert"):
        print("🤖 Inicializando Agente AI...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        self.tools = ToolExecutor()

        print("✅ Agente listo!")

    def parse_tool_calls(self, text: str) -> List[Dict]:
        """Extrae llamadas a herramientas del texto generado"""
        tool_calls = []

        # Buscar bloques <tool_call>...</tool_call>
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            try:
                tool_data = json.loads(match.strip())
                tool_calls.append(tool_data)
            except json.JSONDecodeError:
                continue

        return tool_calls

    def execute_tools(self, tool_calls: List[Dict]) -> List[Dict]:
        """Ejecuta las herramientas solicitadas"""
        results = []

        for call in tool_calls:
            tool_name = call.get('name')
            arguments = call.get('arguments', {})

            print(f"\n🔧 Ejecutando: {tool_name}")
            print(f"   Argumentos: {arguments}")

            if tool_name == 'web_search':
                result = self.tools.web_search(**arguments)
            elif tool_name == 'execute_python':
                result = self.tools.execute_python(**arguments)
            elif tool_name == 'code_analysis':
                result = self.tools.code_analysis(**arguments)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            results.append({
                "tool": tool_name,
                "result": result
            })

        return results

    def chat(self, user_message: str, max_iterations: int = 3) -> str:
        """Chat con capacidad de razonamiento y tool calling"""

        print(f"\n💬 Usuario: {user_message}")

        conversation = []
        current_message = user_message

        for iteration in range(max_iterations):
            print(f"\n🔄 Iteración {iteration + 1}/{max_iterations}")

            # Generar respuesta
            prompt = f"<|im_start|>user\n{current_message}</s>\n<|im_start|>assistant\n"

            response = self.generator(prompt, num_return_sequences=1)[0]["generated_text"]
            assistant_response = response.split("<|im_start|>assistant\n")[-1].replace("</s>", "").strip()

            conversation.append({"role": "assistant", "content": assistant_response})

            # Buscar tool calls
            tool_calls = self.parse_tool_calls(assistant_response)

            if not tool_calls:
                # No hay más tool calls, respuesta final
                return assistant_response

            # Ejecutar herramientas
            tool_results = self.execute_tools(tool_calls)

            # Preparar siguiente prompt con resultados
            results_text = "\n\n".join([
                f"Tool: {r['tool']}\nResult: {json.dumps(r['result'], indent=2)}"
                for r in tool_results
            ])

            current_message = f"Herramientas ejecutadas:\n{results_text}\n\nAhora proporciona la respuesta final al usuario basándote en estos resultados."

        return conversation[-1]["content"]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Agente AI con tool calling")
    parser.add_argument("--model", type=str, default="./data/checkpoints/agent-expert")
    parser.add_argument("--interactive", action="store_true", help="Modo interactivo")
    parser.add_argument("--query", type=str, help="Consulta única")
    args = parser.parse_args()

    # Inicializar agente
    agent = AgentAI(model_path=args.model)

    if args.interactive:
        print("\n" + "=" * 80)
        print("🤖 Agente AI Interactivo - Tool Calling Enabled")
        print("=" * 80)
        print("\nHerramientas disponibles:")
        print("  - web_search: Búsqueda en internet")
        print("  - execute_python: Ejecución de código Python")
        print("  - code_analysis: Análisis de seguridad de código")
        print("\nEscribe 'salir' para terminar\n")

        while True:
            try:
                user_input = input("\n👤 Tú: ")
                if user_input.lower() in ['salir', 'exit', 'quit']:
                    break

                response = agent.chat(user_input)
                print(f"\n🤖 Agente: {response}")

            except KeyboardInterrupt:
                break

    elif args.query:
        response = agent.chat(args.query)
        print(f"\n🤖 Agente: {response}")

    else:
        print("Usa --interactive para modo interactivo o --query 'tu pregunta' para una consulta única")


if __name__ == "__main__":
    main()
