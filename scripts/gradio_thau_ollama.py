#!/usr/bin/env python3
"""
THAU AGI v2 - Interfaz Gradio con Ollama

Interfaz web interactiva para probar THAU usando Ollama como backend de LLM.

Características:
    - Chat con Ollama (local)
    - Text-to-Speech (TTS)
    - Image Generation (Stable Diffusion)
    - MCP Integration (Model Context Protocol)
    - Web Search
    - Multi-Agent System

Requisitos:
    1. Ollama instalado y corriendo: ollama serve
    2. Modelo descargado: ollama pull llama3.2 (o el modelo que prefieras)
    3. Para TTS: pip install gtts
    4. Para Image Gen: pip install diffusers torch

Uso:
    python scripts/gradio_thau_ollama.py

    Luego abre http://localhost:7860 en tu navegador
"""

import sys
from pathlib import Path
import tempfile
import base64
import subprocess
import io

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gradio as gr
from typing import List, Tuple, Dict, Any, Optional
import json
import requests
import time
from dataclasses import dataclass

# === TTS SUPPORT ===
TTS_AVAILABLE = False
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    pass

# === IMAGE GENERATION SUPPORT ===
IMAGE_GEN_AVAILABLE = False
try:
    from capabilities.vision.image_generator import ThauImageGenerator
    IMAGE_GEN_AVAILABLE = True
except ImportError:
    try:
        from diffusers import StableDiffusionPipeline
        import torch
        IMAGE_GEN_AVAILABLE = True
    except ImportError:
        pass

# === MCP SUPPORT ===
MCP_AVAILABLE = False
EXTERNAL_MCP_AVAILABLE = False
try:
    from capabilities.tools.mcp_integration import MCPRegistry, create_default_mcp_tools, MCPServer, ExternalMCPClient, get_external_mcp_client
    MCP_AVAILABLE = True
    EXTERNAL_MCP_AVAILABLE = True
except ImportError:
    try:
        from capabilities.tools.mcp_integration import MCPRegistry, create_default_mcp_tools, MCPServer
        MCP_AVAILABLE = True
    except ImportError:
        pass

# Import THAU components
from capabilities.proto_agi import (
    ThauTools, ToolResult,
    ExperienceStore, MetacognitiveEngine, AdaptiveStrategy,
    OutcomeType, StrategyType,
    MultiAgentSystem, AgentRole,
    KnowledgeStore, ContextBuilder, KnowledgeLearner, FeedbackSystem,
    KnowledgeType,
    get_experience_store, get_metacognitive_engine, get_adaptive_strategy,
    get_knowledge_store, get_context_builder, get_knowledge_learner, get_feedback_system,
)

# Web search (optional)
try:
    from capabilities.tools import WebSearchTool, WebFetcher, ResearchAgent, WEB_SEARCH_AVAILABLE
except ImportError:
    WEB_SEARCH_AVAILABLE = False


@dataclass
class OllamaConfig:
    """Configuracion de Ollama"""
    base_url: str = "http://localhost:11434"
    model: str = "thau:latest"  # Modelo THAU por defecto
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 120


class ThauOllama:
    """
    THAU AGI v2 con Ollama como backend

    Integra todos los componentes de THAU usando Ollama para generacion de texto.
    """

    def __init__(
        self,
        ollama_config: OllamaConfig = None,
        enable_learning: bool = True,
        enable_web_search: bool = True,
        enable_multi_agent: bool = True,
        enable_knowledge: bool = True,
        enable_feedback: bool = True,
        verbose: bool = True
    ):
        self.config = ollama_config or OllamaConfig()
        self.verbose = verbose
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.interaction_count = 0

        # Verificar conexion con Ollama
        self._check_ollama_connection()

        # === COMPONENTES ===

        # 1. Herramientas basicas
        self.tools = {
            "calculate": self._calculate,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "list_directory": self._list_directory,
            "execute_python": self._execute_python,
        }

        # 2. Aprendizaje experiencial
        if enable_learning:
            self.experience_store = get_experience_store()
            self.metacognitive = get_metacognitive_engine()
            self.adaptive = get_adaptive_strategy()
        else:
            self.experience_store = None
            self.metacognitive = None
            self.adaptive = None

        # 3. Web search
        self.web_search = None
        self.web_fetcher = None
        self.research_agent = None

        if enable_web_search and WEB_SEARCH_AVAILABLE:
            self.web_search = WebSearchTool()
            self.web_fetcher = WebFetcher()
            self.research_agent = ResearchAgent(
                search_tool=self.web_search,
                fetcher=self.web_fetcher,
                verbose=False
            )
            self.tools["web_search"] = self._web_search
            self.tools["fetch_url"] = self._fetch_url
            self.tools["research"] = self._research

        # 4. Multi-agente
        self.multi_agent_system = None
        if enable_multi_agent:
            self.multi_agent_system = MultiAgentSystem(verbose=False)
            self.multi_agent_system.initialize([
                AgentRole.CODER,
                AgentRole.REVIEWER,
                AgentRole.RESEARCHER,
                AgentRole.PLANNER
            ])

        # 5. Knowledge base
        if enable_knowledge:
            self.knowledge_store = get_knowledge_store()
            self.context_builder = get_context_builder()
            self.knowledge_learner = get_knowledge_learner()
        else:
            self.knowledge_store = None
            self.context_builder = None
            self.knowledge_learner = None

        # 6. Feedback system
        if enable_feedback:
            self.feedback_system = get_feedback_system()
        else:
            self.feedback_system = None

        # Memoria de sesion
        self.conversation_history = []
        self.last_interaction_id = None

        # 7. TTS Support
        self.tts_enabled = TTS_AVAILABLE
        if self.tts_enabled:
            self.tools["text_to_speech"] = self._text_to_speech

        # 8. Image Generation Support
        self.image_generator = None
        if IMAGE_GEN_AVAILABLE:
            try:
                self.image_generator = ThauImageGenerator()
                self.tools["generate_image"] = self._generate_image
            except Exception as e:
                print(f"[WARN] Image generation not available: {e}")

        # 9. MCP Support
        self.mcp_registry = None
        self.mcp_server = None
        self.external_mcp = None
        if MCP_AVAILABLE:
            try:
                self.mcp_registry = create_default_mcp_tools()
                self.mcp_server = MCPServer(self.mcp_registry)
            except Exception as e:
                print(f"[WARN] MCP internal not available: {e}")

        # 10. External MCP Client (Docker-based servers like mcp/fetch)
        if EXTERNAL_MCP_AVAILABLE:
            try:
                self.external_mcp = get_external_mcp_client()
                # Registrar herramienta de fetch externa
                if self.external_mcp and "fetch" in self.external_mcp.servers:
                    self.tools["mcp_fetch"] = self._mcp_fetch
            except Exception as e:
                print(f"[WARN] External MCP not available: {e}")

        if self.verbose:
            self._print_status()

    def _check_ollama_connection(self) -> bool:
        """Verifica conexion con Ollama"""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                if self.verbose:
                    print(f"[OK] Ollama conectado. Modelos: {', '.join(model_names[:5])}")
                return True
        except Exception as e:
            print(f"[ERROR] No se pudo conectar con Ollama: {e}")
            print("        Asegurate de que Ollama este corriendo: ollama serve")
        return False

    def _print_status(self):
        """Imprime estado del sistema"""
        print("\n" + "=" * 60)
        print("  THAU AGI v2 + Ollama")
        print("=" * 60)
        print(f"  Modelo: {self.config.model}")
        print(f"  Session: {self.session_id}")
        print(f"  Herramientas: {len(self.tools)}")
        print(f"  Web Search: {'✓' if self.web_search else '✗'}")
        print(f"  Multi-Agent: {'✓' if self.multi_agent_system else '✗'}")
        print(f"  Knowledge: {'✓' if self.knowledge_store else '✗'}")
        print(f"  TTS (Text-to-Speech): {'✓' if self.tts_enabled else '✗'}")
        print(f"  Image Generation: {'✓' if self.image_generator else '✗'}")
        print(f"  MCP Protocol: {'✓' if self.mcp_registry else '✗'}")
        print(f"  External MCP: {'✓' if self.external_mcp else '✗'}")
        if self.external_mcp and self.external_mcp.servers:
            print(f"    Servers: {', '.join(self.external_mcp.list_servers())}")
        print("=" * 60 + "\n")

    def _generate(self, prompt: str, system_prompt: str = None) -> str:
        """Genera respuesta usando Ollama"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Agregar historial de conversacion (ultimos 3 turnos)
        for msg in self.conversation_history[-6:]:
            messages.append(msg)

        messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens
                    }
                },
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                return response.json().get("message", {}).get("content", "")
            else:
                return f"Error de Ollama: {response.status_code}"

        except requests.exceptions.Timeout:
            return "Error: Timeout al conectar con Ollama"
        except Exception as e:
            return f"Error: {str(e)}"

    def _get_system_prompt(self) -> str:
        """Genera el prompt del sistema"""
        prompt = """Eres THAU, un asistente de IA avanzado con capacidades proto-AGI.

Caracteristicas:
- Piensas paso a paso antes de actuar
- Usas herramientas cuando es necesario
- Respondes siempre en espanol de forma clara y concisa

Herramientas disponibles:
- calculate: Evaluar expresiones matematicas. Ejemplo: TOOL:calculate:5+3*2
- read_file: Leer archivos. Ejemplo: TOOL:read_file:ruta/archivo.txt
- write_file: Escribir archivos. Ejemplo: TOOL:write_file:archivo.txt:contenido
- list_directory: Listar directorios. Ejemplo: TOOL:list_directory:.
- execute_python: Ejecutar codigo Python. Ejemplo: TOOL:execute_python:print("hola")"""

        if self.web_search:
            prompt += """
- web_search: Buscar en internet. Ejemplo: TOOL:web_search:Python programming
- fetch_url: Obtener contenido de URL. Ejemplo: TOOL:fetch_url:https://python.org
- research: Investigar tema. Ejemplo: TOOL:research:machine learning"""

        prompt += """

Para usar una herramienta, escribe:
TOOL:nombre_herramienta:parametros

Si no necesitas herramientas, responde directamente.
Siempre piensa antes de responder y explica tu razonamiento brevemente."""

        return prompt

    # === HERRAMIENTAS ===

    def _calculate(self, expression: str) -> ToolResult:
        """Calcula expresion matematica"""
        try:
            # Sanitizar expresion
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return ToolResult("calculate", False, "", "Expresion no valida")

            result = eval(expression)
            return ToolResult("calculate", True, str(result))
        except Exception as e:
            return ToolResult("calculate", False, "", str(e))

    def _read_file(self, filepath: str) -> ToolResult:
        """Lee contenido de archivo"""
        try:
            path = Path(filepath)
            if not path.exists():
                return ToolResult("read_file", False, "", f"Archivo no existe: {filepath}")

            content = path.read_text(encoding='utf-8')
            # Limitar tamano
            if len(content) > 5000:
                content = content[:5000] + "\n... (truncado)"

            return ToolResult("read_file", True, content)
        except Exception as e:
            return ToolResult("read_file", False, "", str(e))

    def _write_file(self, filepath: str, content: str = "") -> ToolResult:
        """Escribe contenido a archivo"""
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            return ToolResult("write_file", True, f"Archivo escrito: {filepath}")
        except Exception as e:
            return ToolResult("write_file", False, "", str(e))

    def _list_directory(self, dirpath: str = ".") -> ToolResult:
        """Lista contenido de directorio"""
        try:
            path = Path(dirpath)
            if not path.exists():
                return ToolResult("list_directory", False, "", f"Directorio no existe: {dirpath}")

            items = []
            for item in sorted(path.iterdir())[:50]:
                prefix = "📁" if item.is_dir() else "📄"
                items.append(f"{prefix} {item.name}")

            return ToolResult("list_directory", True, "\n".join(items))
        except Exception as e:
            return ToolResult("list_directory", False, "", str(e))

    def _execute_python(self, code: str) -> ToolResult:
        """Ejecuta codigo Python"""
        try:
            import io
            import contextlib

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(code, {"__builtins__": __builtins__})

            result = output.getvalue()
            return ToolResult("execute_python", True, result if result else "Codigo ejecutado sin output")
        except Exception as e:
            return ToolResult("execute_python", False, "", str(e))

    def _web_search(self, query: str) -> ToolResult:
        """Busqueda web"""
        if not self.web_search:
            return ToolResult("web_search", False, "", "Web search no disponible")

        try:
            results = self.web_search.search(query, num_results=5)
            if not results:
                return ToolResult("web_search", False, "", "Sin resultados")

            output = f"Resultados para '{query}':\n"
            for i, r in enumerate(results, 1):
                output += f"\n{i}. {r.title}\n   {r.url}\n   {r.snippet[:150]}..."

            return ToolResult("web_search", True, output)
        except Exception as e:
            return ToolResult("web_search", False, "", str(e))

    def _fetch_url(self, url: str) -> ToolResult:
        """Obtener contenido de URL"""
        if not self.web_fetcher:
            return ToolResult("fetch_url", False, "", "Web fetcher no disponible")

        try:
            page = self.web_fetcher.fetch(url)
            if page.success:
                return ToolResult("fetch_url", True, f"{page.title}\n\n{page.content[:3000]}")
            return ToolResult("fetch_url", False, "", page.error)
        except Exception as e:
            return ToolResult("fetch_url", False, "", str(e))

    def _research(self, query: str) -> ToolResult:
        """Investigacion profunda"""
        if not self.research_agent:
            return ToolResult("research", False, "", "Research agent no disponible")

        try:
            result = self.research_agent.research(query, depth="quick")
            output = f"Investigacion: {query}\n"
            output += f"Confianza: {result.confidence:.0%}\n\n"
            output += result.summary

            if result.key_facts:
                output += "\n\nHechos clave:"
                for fact in result.key_facts[:5]:
                    output += f"\n  - {fact}"

            return ToolResult("research", True, output)
        except Exception as e:
            return ToolResult("research", False, "", str(e))

    def _text_to_speech(self, text: str, lang: str = "es") -> ToolResult:
        """Convierte texto a voz"""
        if not TTS_AVAILABLE:
            return ToolResult("text_to_speech", False, "", "TTS no disponible. Instala: pip install gtts")

        try:
            tts = gTTS(text=text, lang=lang)
            # Guardar temporalmente
            output_dir = Path("data/tts_output")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"tts_{timestamp}.mp3"

            tts.save(str(output_path))

            # Guardar path en el output para recuperarlo después
            return ToolResult(
                "text_to_speech",
                True,
                str(output_path)  # El path va en output
            )
        except Exception as e:
            return ToolResult("text_to_speech", False, "", str(e))

    def _generate_image(self, prompt: str, steps: int = 30) -> ToolResult:
        """Genera imagen a partir de texto"""
        if not self.image_generator:
            return ToolResult("generate_image", False, "", "Image generation no disponible")

        try:
            result = self.image_generator.generate_image(
                prompt=prompt,
                num_inference_steps=steps,
                save_image=True
            )

            if result.get("success"):
                # El path de la imagen va en output
                return ToolResult(
                    "generate_image",
                    True,
                    str(result.get("path", ""))
                )
            else:
                return ToolResult("generate_image", False, "", result.get("error", "Error desconocido"))
        except Exception as e:
            return ToolResult("generate_image", False, "", str(e))

    def _mcp_fetch(self, url: str) -> ToolResult:
        """Fetch URL usando MCP externo (Docker mcp/fetch)"""
        if not self.external_mcp:
            return ToolResult("mcp_fetch", False, "", "External MCP not available")

        try:
            result = self.external_mcp.fetch_url(url)
            if result.success:
                # Formatear output
                content = result.result
                if isinstance(content, dict):
                    content = json.dumps(content, indent=2, ensure_ascii=False)
                return ToolResult("mcp_fetch", True, str(content))
            else:
                return ToolResult("mcp_fetch", False, "", result.error)
        except Exception as e:
            return ToolResult("mcp_fetch", False, "", str(e))

    def call_external_mcp(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Llama a cualquier herramienta MCP externa"""
        if not self.external_mcp:
            return {"success": False, "error": "External MCP not available"}

        result = self.external_mcp.call_tool(server_name, tool_name, arguments)
        return {
            "success": result.success,
            "result": result.result,
            "error": result.error,
            "execution_time_ms": result.execution_time_ms
        }

    def list_external_mcp_servers(self) -> List[str]:
        """Lista servidores MCP externos disponibles"""
        if not self.external_mcp:
            return []
        return self.external_mcp.list_servers()

    def add_external_mcp_server(self, name: str, command: str, args: List[str]) -> bool:
        """Añade un servidor MCP externo"""
        if not self.external_mcp:
            return False
        self.external_mcp.add_server(name, command, args)
        return True

    def invoke_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoca una herramienta MCP externa"""
        if not self.mcp_registry:
            return {"success": False, "error": "MCP no disponible"}

        result = self.mcp_registry.invoke_tool(tool_name, arguments)
        return {
            "success": result.success,
            "result": result.result,
            "error": result.error,
            "execution_time_ms": result.execution_time_ms
        }

    def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Lista herramientas MCP disponibles"""
        if not self.mcp_registry:
            return []
        return self.mcp_registry.list_tools()

    def _detect_and_execute_tool(self, response: str) -> Tuple[str, List[ToolResult]]:
        """Detecta y ejecuta herramientas en la respuesta"""
        tool_results = []
        modified_response = response

        # Buscar patron TOOL:nombre:params
        import re
        tool_pattern = r'TOOL:(\w+):([^\n]+)'
        matches = re.findall(tool_pattern, response)

        for tool_name, params in matches:
            if tool_name in self.tools:
                # Ejecutar herramienta
                if tool_name == "write_file" and ":" in params:
                    filepath, content = params.split(":", 1)
                    result = self.tools[tool_name](filepath.strip(), content.strip())
                else:
                    result = self.tools[tool_name](params.strip())

                tool_results.append(result)

                # Reemplazar en respuesta
                tool_call = f"TOOL:{tool_name}:{params}"
                if result.success:
                    replacement = f"\n[Herramienta {tool_name}]\n{result.output}\n"
                else:
                    replacement = f"\n[Error en {tool_name}: {result.error}]\n"

                modified_response = modified_response.replace(tool_call, replacement)

        return modified_response, tool_results

    def _classify_goal(self, goal: str) -> str:
        """Clasifica el tipo de meta"""
        goal_lower = goal.lower()

        if any(w in goal_lower for w in ["calcula", "suma", "resta", "cuanto", "+", "-", "*", "/"]):
            return "calculation"
        elif any(w in goal_lower for w in ["codigo", "programa", "funcion", "code", "script"]):
            return "code"
        elif any(w in goal_lower for w in ["archivo", "lee", "escribe", "file"]):
            return "file"
        elif any(w in goal_lower for w in ["lista", "directorio", "carpeta"]):
            return "directory"
        elif any(w in goal_lower for w in ["busca en internet", "investiga", "research"]):
            return "web_search"
        elif any(w in goal_lower for w in ["explica", "que es", "como", "define"]):
            return "question"

        return "general"

    def _needs_web_search(self, message: str) -> Optional[str]:
        """
        Detecta si el mensaje necesita busqueda web automatica.
        Retorna el query de busqueda o None.
        """
        msg_lower = message.lower()

        # Palabras clave explicitas para busqueda
        explicit_keywords = [
            "busca en internet", "buscar en internet",
            "investiga", "investigar",
            "busca online", "buscar online",
            "search", "research",
            "busca informacion sobre",
            "encuentra informacion sobre"
        ]

        for kw in explicit_keywords:
            if kw in msg_lower:
                # Extraer el tema de busqueda
                idx = msg_lower.find(kw) + len(kw)
                query = message[idx:].strip()
                if query:
                    return query
                return message

        # Preguntas sobre informacion actual/tiempo real
        current_info_patterns = [
            ("tasa", "tipo de cambio"),
            ("precio", "cotizacion"),
            ("dolar", "usd"),
            ("euro", "eur"),
            ("clima", "tiempo"),
            ("noticias", "news"),
            ("hoy", "actual"),
            ("ultima", "ultimo"),
            ("reciente", "2024", "2025"),
        ]

        # Si menciona algo que requiere info actual
        needs_current = any(
            any(p in msg_lower for p in patterns)
            for patterns in current_info_patterns
        )

        if needs_current:
            return message

        return None

    def chat(self, message: str) -> Dict[str, Any]:
        """
        Procesa un mensaje del usuario

        Returns:
            Dict con response, confidence, tools_used, etc.
        """
        start_time = time.time()
        self.interaction_count += 1
        self.last_interaction_id = f"{self.session_id}_{self.interaction_count}"

        tool_results = []
        tools_used = []

        # === DETECCION AUTOMATICA DE HERRAMIENTAS ===

        # 1. Detectar si necesita busqueda web
        web_query = self._needs_web_search(message)
        if web_query and self.web_search:
            print(f"[AUTO] Detectada necesidad de busqueda web: {web_query}")
            result = self._web_search(web_query)
            if result.success:
                tool_results.append(result)
                tools_used.append("web_search")

        # 2. Detectar calculos
        import re
        if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', message) or any(w in message.lower() for w in ["calcula", "cuanto es", "suma", "resta"]):
            # Extraer expresion
            expr_match = re.search(r'[\d\+\-\*\/\.\(\)\s]+', message)
            if expr_match:
                expr = expr_match.group().strip()
                if any(c in expr for c in "+-*/"):
                    result = self._calculate(expr)
                    if result.success:
                        tool_results.append(result)
                        tools_used.append("calculate")

        # 3. Detectar listado de directorio
        if any(w in message.lower() for w in ["lista archivos", "listar archivos", "lista el directorio", "listame", "muestra archivos"]):
            result = self._list_directory(".")
            if result.success:
                tool_results.append(result)
                tools_used.append("list_directory")

        # 4. Detectar lectura de archivo
        file_match = re.search(r'lee\s+(?:el\s+)?(?:archivo\s+)?([^\s]+\.[a-zA-Z]{1,5})', message.lower())
        if file_match:
            filepath = file_match.group(1)
            result = self._read_file(filepath)
            if result.success:
                tool_results.append(result)
                tools_used.append("read_file")

        # === GENERAR RESPUESTA ===

        # Obtener contexto de knowledge base
        context = ""
        if self.context_builder:
            context = self.context_builder.build_context(message, n_items=3)

        # Construir prompt
        system_prompt = self._get_system_prompt()
        if context:
            system_prompt += f"\n\nContexto relevante:\n{context}"

        # Si hay resultados de herramientas, incluirlos en el contexto
        if tool_results:
            tool_context = "\n\n=== Informacion obtenida ===\n"
            for r in tool_results:
                if r.success:
                    tool_context += f"\n[{r.tool}]:\n{r.output}\n"

            # Generar respuesta con contexto de herramientas
            prompt = f"{message}\n{tool_context}\n\nBasandote en la informacion anterior, responde de forma clara y concisa:"
            response = self._generate(prompt, system_prompt)
        else:
            # Generar respuesta normal
            response = self._generate(message, system_prompt)

        # Detectar herramientas en la respuesta del modelo (por si acaso)
        response, additional_results = self._detect_and_execute_tool(response)
        tool_results.extend(additional_results)
        tools_used.extend([r.tool for r in additional_results])

        # Calcular confianza
        confidence = 0.7
        if self.metacognitive:
            eval_result = self.metacognitive.evaluate_response(
                message, response,
                [{"success": r.success, "output": r.output} for r in tool_results]
            )
            confidence = eval_result.get("confidence", 0.7)

        # Registrar en knowledge base
        if self.knowledge_learner:
            self.knowledge_learner.learn_from_conversation(message, response, confidence > 0.6)

        # Actualizar historial
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": response})

        execution_time = time.time() - start_time

        return {
            "response": response,
            "interaction_id": self.last_interaction_id,
            "confidence": confidence,
            "tools_used": tools_used,
            "execution_time": execution_time,
            "goal_type": self._classify_goal(message)
        }

    def thumbs_up(self) -> None:
        """Feedback positivo"""
        if self.feedback_system and self.last_interaction_id:
            self.feedback_system.thumbs_up(self.last_interaction_id)

    def thumbs_down(self, reason: str = "") -> None:
        """Feedback negativo"""
        if self.feedback_system and self.last_interaction_id:
            self.feedback_system.thumbs_down(self.last_interaction_id, reason=reason)

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadisticas"""
        stats = {
            "session_id": self.session_id,
            "interactions": self.interaction_count,
            "model": self.config.model,
            "tools": list(self.tools.keys())
        }

        if self.experience_store:
            stats["experiences"] = self.experience_store.get_statistics()

        if self.feedback_system:
            stats["feedback"] = self.feedback_system.get_stats()

        if self.knowledge_store:
            stats["knowledge"] = self.knowledge_store.get_stats()

        return stats

    def get_available_models(self) -> List[str]:
        """Obtiene modelos disponibles en Ollama"""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name", "") for m in models]
        except:
            pass
        return []

    def set_model(self, model: str) -> None:
        """Cambia el modelo"""
        self.config.model = model


# === INTERFAZ GRADIO ===

agent: Optional[ThauOllama] = None


def get_available_models() -> List[str]:
    """Obtiene modelos de Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [m.get("name", "") for m in models]
    except:
        pass
    return ["llama3.2", "llama3.1", "mistral", "codellama"]


def initialize_agent(
    model: str,
    temperature: float,
    enable_web: bool,
    enable_multi: bool,
    enable_knowledge: bool,
    enable_feedback: bool
) -> str:
    """Inicializa el agente"""
    global agent

    config = OllamaConfig(
        model=model,
        temperature=temperature
    )

    try:
        agent = ThauOllama(
            ollama_config=config,
            enable_learning=True,
            enable_web_search=enable_web,
            enable_multi_agent=enable_multi,
            enable_knowledge=enable_knowledge,
            enable_feedback=enable_feedback,
            verbose=False
        )

        status = f"THAU + Ollama inicializado!\n\n"
        status += f"Modelo: {model}\n"
        status += f"Temperatura: {temperature}\n"
        status += f"Herramientas: {len(agent.tools)}\n"
        status += f"Session: {agent.session_id}\n\n"
        status += f"Web Search: {'✓' if enable_web else '✗'}\n"
        status += f"Multi-Agent: {'✓' if enable_multi else '✗'}\n"
        status += f"Knowledge: {'✓' if enable_knowledge else '✗'}\n"
        status += f"Feedback: {'✓' if enable_feedback else '✗'}"

        return status

    except Exception as e:
        return f"Error al inicializar: {str(e)}\n\nAsegurate de que Ollama este corriendo."


def chat(message: str, history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], str]:
    """Procesa mensaje - formato Gradio 6.0"""
    global agent

    if agent is None:
        models = get_available_models()
        model = models[0] if models else "thau:latest"
        initialize_agent(model, 0.7, True, True, True, True)

    result = agent.chat(message)

    info = f"Confianza: {result['confidence']:.0%} | "
    info += f"Tiempo: {result['execution_time']:.2f}s"

    if result["tools_used"]:
        info += f"\nHerramientas: {', '.join(result['tools_used'])}"

    # Gradio 6.0 requiere formato de mensajes con role/content
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": result["response"]})

    return history, info


def give_feedback(feedback_type: str) -> str:
    """Registra feedback"""
    global agent

    if agent is None:
        return "Agente no inicializado"

    if "Positivo" in feedback_type:
        agent.thumbs_up()
        return "👍 Feedback positivo registrado!"
    else:
        agent.thumbs_down()
        return "👎 Feedback negativo registrado!"


def get_stats() -> str:
    """Obtiene estadisticas"""
    global agent

    if agent is None:
        return "Agente no inicializado"

    stats = agent.get_stats()

    output = "=== Estadisticas THAU + Ollama ===\n\n"
    output += f"Session: {stats['session_id']}\n"
    output += f"Modelo: {stats['model']}\n"
    output += f"Interacciones: {stats['interactions']}\n"
    output += f"Herramientas: {', '.join(stats['tools'])}\n"

    if 'feedback' in stats:
        fb = stats['feedback']
        output += f"\n--- Feedback ---\n"
        output += f"Total: {fb.get('total_feedback', 0)}\n"
        output += f"Satisfaccion: {fb.get('satisfaction_rate', 0):.0%}\n"

    return output


def clear_chat() -> Tuple[List[Dict[str, str]], str]:
    """Limpia el chat"""
    return [], ""


def text_to_speech(text: str, lang: str = "es") -> Tuple[Optional[str], str]:
    """Convierte texto a voz y retorna path del audio"""
    global agent

    if not TTS_AVAILABLE:
        return None, "❌ TTS no disponible. Instala: pip install gtts"

    if agent is None:
        models = get_available_models()
        model = models[0] if models else "thau:latest"
        initialize_agent(model, 0.7, True, True, True, True)

    try:
        result = agent._text_to_speech(text, lang)
        if result.success:
            audio_path = result.output  # El path está directamente en output
            return audio_path, f"✅ Audio generado: {audio_path}"
        return None, f"❌ Error: {result.error}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def generate_image_ui(prompt: str, steps: int = 30) -> Tuple[Optional[str], str]:
    """Genera imagen desde prompt"""
    global agent

    if not IMAGE_GEN_AVAILABLE:
        return None, "❌ Image generation no disponible. Instala: pip install diffusers torch"

    if agent is None:
        models = get_available_models()
        model = models[0] if models else "thau:latest"
        initialize_agent(model, 0.7, True, True, True, True)

    if not agent.image_generator:
        return None, "❌ Image generator no inicializado"

    try:
        result = agent._generate_image(prompt, steps)
        if result.success:
            image_path = result.output  # El path está directamente en output
            return image_path, f"✅ Imagen generada: {image_path}"
        return None, f"❌ Error: {result.error}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def get_mcp_tools_list() -> str:
    """Lista herramientas MCP disponibles"""
    global agent

    if agent is None or not agent.mcp_registry:
        return "MCP no disponible o agente no inicializado"

    tools = agent.get_mcp_tools()
    if not tools:
        return "No hay herramientas MCP registradas"

    output = "=== Herramientas MCP Disponibles ===\n\n"
    for tool in tools:
        func = tool.get("function", {})
        output += f"🔧 {func.get('name', 'Unknown')}\n"
        output += f"   {func.get('description', 'Sin descripcion')}\n"

        params = func.get("parameters", {}).get("properties", {})
        if params:
            output += "   Parametros:\n"
            for param_name, param_info in params.items():
                required = param_name in func.get("parameters", {}).get("required", [])
                output += f"     - {param_name} ({param_info.get('type', 'any')})"
                if required:
                    output += " *"
                output += f": {param_info.get('description', '')}\n"
        output += "\n"

    return output


def invoke_mcp(tool_name: str, arguments_json: str) -> str:
    """Invoca herramienta MCP"""
    global agent

    if agent is None or not agent.mcp_registry:
        return "❌ MCP no disponible o agente no inicializado"

    try:
        arguments = json.loads(arguments_json) if arguments_json.strip() else {}
        result = agent.invoke_mcp_tool(tool_name, arguments)

        if result.get("success"):
            return f"✅ Resultado:\n{json.dumps(result.get('result', {}), indent=2, ensure_ascii=False)}\n\nTiempo: {result.get('execution_time_ms', 0):.2f}ms"
        else:
            return f"❌ Error: {result.get('error', 'Error desconocido')}"
    except json.JSONDecodeError:
        return "❌ Error: Los argumentos deben ser JSON valido"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Crear interfaz
with gr.Blocks(title="THAU + Ollama") as demo:
    gr.Markdown("""
    # 🧠 THAU AGI v2 + Ollama

    Sistema Proto-AGI ejecutando localmente con Ollama.

    **Requisitos:**
    1. Ollama instalado y corriendo (`ollama serve`)
    2. Al menos un modelo descargado (`ollama pull llama3.2`)
    """)

    with gr.Tab("💬 Chat"):
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversacion con THAU",
                    height=450
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Tu mensaje",
                        placeholder="Escribe aqui... (Ej: Calcula 25*4, Busca en internet Python, Lista archivos)",
                        scale=4,
                        lines=2
                    )
                    send_btn = gr.Button("Enviar", variant="primary", scale=1)

                info_box = gr.Textbox(label="Info", interactive=False, lines=2)

                with gr.Row():
                    clear_btn = gr.Button("Limpiar Chat")

            with gr.Column(scale=1):
                gr.Markdown("### Feedback")
                fb_pos = gr.Button("👍 Positivo", variant="secondary")
                fb_neg = gr.Button("👎 Negativo", variant="secondary")
                fb_output = gr.Textbox(label="Estado", interactive=False, lines=1)

                gr.Markdown("### Ejemplos")
                examples = [
                    "Calcula 25 * 4 + 100",
                    "Lista los archivos del directorio",
                    "Que es una funcion recursiva?",
                    "Busca en internet que es Python",
                    "Lee el archivo requirements.txt",
                ]
                for ex in examples:
                    btn = gr.Button(ex[:35] + ("..." if len(ex) > 35 else ""), size="sm")
                    btn.click(lambda x=ex: x, outputs=msg)

        send_btn.click(chat, inputs=[msg, chatbot], outputs=[chatbot, info_box]).then(lambda: "", outputs=msg)
        msg.submit(chat, inputs=[msg, chatbot], outputs=[chatbot, info_box]).then(lambda: "", outputs=msg)
        clear_btn.click(clear_chat, outputs=[chatbot, info_box])
        fb_pos.click(lambda: give_feedback("Positivo"), outputs=fb_output)
        fb_neg.click(lambda: give_feedback("Negativo"), outputs=fb_output)

    with gr.Tab("⚙️ Configuracion"):
        gr.Markdown("### Configurar THAU + Ollama")

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=get_available_models() or ["thau:latest"],
                value="thau:latest",
                label="Modelo Ollama"
            )
            refresh_models = gr.Button("🔄", scale=0)

        temperature = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Temperatura")

        with gr.Row():
            with gr.Column():
                chk_web = gr.Checkbox(label="Web Search", value=True)
                chk_multi = gr.Checkbox(label="Multi-Agent", value=True)
            with gr.Column():
                chk_knowledge = gr.Checkbox(label="Knowledge Base", value=True)
                chk_feedback = gr.Checkbox(label="Feedback", value=True)

        init_btn = gr.Button("Inicializar Agente", variant="primary")
        init_output = gr.Textbox(label="Estado", interactive=False, lines=10)

        refresh_models.click(
            lambda: gr.Dropdown(choices=get_available_models()),
            outputs=model_dropdown
        )

        init_btn.click(
            initialize_agent,
            inputs=[model_dropdown, temperature, chk_web, chk_multi, chk_knowledge, chk_feedback],
            outputs=init_output
        )

    with gr.Tab("🔊 Text-to-Speech"):
        gr.Markdown("""
        ### Convertir Texto a Voz

        Escribe un texto y THAU lo convertira en audio usando Google Text-to-Speech.

        **Nota:** Requiere `pip install gtts`
        """)

        with gr.Row():
            with gr.Column(scale=2):
                tts_text = gr.Textbox(
                    label="Texto para convertir a voz",
                    placeholder="Escribe aqui el texto que quieres escuchar...",
                    lines=4
                )
                tts_lang = gr.Dropdown(
                    choices=[("Espanol", "es"), ("Ingles", "en"), ("Frances", "fr"), ("Aleman", "de"), ("Italiano", "it"), ("Portugues", "pt")],
                    value="es",
                    label="Idioma"
                )
                tts_btn = gr.Button("🔊 Generar Audio", variant="primary")

            with gr.Column(scale=1):
                tts_audio = gr.Audio(label="Audio Generado", type="filepath")
                tts_status = gr.Textbox(label="Estado", interactive=False, lines=2)

        tts_btn.click(
            text_to_speech,
            inputs=[tts_text, tts_lang],
            outputs=[tts_audio, tts_status]
        )

        gr.Markdown("""
        **Ejemplos:**
        - "Hola, soy THAU, tu asistente de inteligencia artificial"
        - "Buenos dias, como puedo ayudarte hoy?"
        """)

    with gr.Tab("🎨 Generar Imagenes"):
        gr.Markdown("""
        ### Generacion de Imagenes con Stable Diffusion

        Describe la imagen que deseas y THAU la generara usando IA.

        **Nota:** Requiere `pip install diffusers torch` y bastante RAM/VRAM.
        """)

        with gr.Row():
            with gr.Column(scale=2):
                img_prompt = gr.Textbox(
                    label="Descripcion de la imagen",
                    placeholder="Describe la imagen que quieres generar... (ej: un robot pintando un cuadro)",
                    lines=3
                )
                img_steps = gr.Slider(10, 50, value=30, step=5, label="Pasos de inferencia (mas = mejor calidad, mas lento)")
                img_btn = gr.Button("🎨 Generar Imagen", variant="primary")

            with gr.Column(scale=1):
                img_output = gr.Image(label="Imagen Generada", type="filepath")
                img_status = gr.Textbox(label="Estado", interactive=False, lines=2)

        img_btn.click(
            generate_image_ui,
            inputs=[img_prompt, img_steps],
            outputs=[img_output, img_status]
        )

        gr.Markdown("""
        **Ejemplos de prompts:**
        - "a cute robot learning to paint, digital art"
        - "a serene mountain landscape at sunset, photorealistic"
        - "abstract representation of artificial intelligence, colorful"
        """)

    with gr.Tab("🔌 MCP Tools"):
        gr.Markdown("""
        ### Model Context Protocol (MCP)

        MCP es el estandar para que los modelos LLM invoquen herramientas externas.
        THAU puede usar herramientas MCP compatibles con Claude/OpenAI.

        **Para usar tus propios MCP servers:**
        Configura en `~/.mcp/config.json` o pasa la configuracion al inicializar.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                mcp_tools_output = gr.Textbox(
                    label="Herramientas MCP Disponibles",
                    interactive=False,
                    lines=15
                )
                mcp_refresh_btn = gr.Button("🔄 Actualizar Lista")

            with gr.Column(scale=1):
                mcp_tool_name = gr.Textbox(label="Nombre de herramienta", placeholder="generate_image")
                mcp_args = gr.Textbox(
                    label="Argumentos (JSON)",
                    placeholder='{"prompt": "un robot", "num_images": 1}',
                    lines=4
                )
                mcp_invoke_btn = gr.Button("⚡ Invocar Tool", variant="primary")
                mcp_result = gr.Textbox(label="Resultado", interactive=False, lines=8)

        mcp_refresh_btn.click(get_mcp_tools_list, outputs=mcp_tools_output)
        mcp_invoke_btn.click(
            invoke_mcp,
            inputs=[mcp_tool_name, mcp_args],
            outputs=mcp_result
        )

    with gr.Tab("📊 Estadisticas"):
        stats_output = gr.Textbox(label="Estadisticas", interactive=False, lines=15)
        refresh_stats = gr.Button("Actualizar")
        refresh_stats.click(get_stats, outputs=stats_output)

    with gr.Tab("ℹ️ Ayuda"):
        gr.Markdown("""
        ## Guia Rapida de THAU AGI v2 + Ollama

        ### Herramientas Disponibles

        | Herramienta | Descripcion | Ejemplo |
        |-------------|-------------|---------|
        | `calculate` | Calculos matematicos | "Calcula 25*4+100" |
        | `read_file` | Leer archivos | "Lee el archivo config.py" |
        | `write_file` | Escribir archivos | "Escribe 'hola' en test.txt" |
        | `list_directory` | Listar directorio | "Lista los archivos del directorio" |
        | `execute_python` | Ejecutar Python | "Ejecuta print(2**10)" |
        | `web_search` | Buscar en internet | "Busca en internet que es Python" |
        | `fetch_url` | Obtener URL | "Obten el contenido de python.org" |
        | `research` | Investigar tema | "Investiga sobre machine learning" |
        | `text_to_speech` | Convertir texto a voz | Tab 🔊 Text-to-Speech |
        | `generate_image` | Generar imagenes con IA | Tab 🎨 Generar Imagenes |

        ### Comandos de Ollama

        ```bash
        # Iniciar Ollama
        ollama serve

        # Descargar modelos
        ollama pull llama3.2
        ollama pull thau:latest
        ollama pull mistral

        # Ver modelos disponibles
        ollama list
        ```

        ### Instalacion de Dependencias Opcionales

        ```bash
        # Para Text-to-Speech
        pip install gtts

        # Para Generacion de Imagenes
        pip install diffusers torch accelerate

        # Para Web Search
        pip install httpx beautifulsoup4 html2text
        ```

        ### Tips

        - Usa frases claras como "Calcula...", "Busca en internet...", "Lista..."
        - El modelo detecta automaticamente cuando usar herramientas
        - Da feedback para mejorar las respuestas futuras
        - Para info actual (clima, tasas, noticias), pregunta directamente

        ### MCP (Model Context Protocol)

        THAU soporta MCP para integrar herramientas externas:

        1. Ve a la tab **🔌 MCP Tools**
        2. Actualiza la lista de herramientas disponibles
        3. Selecciona una herramienta e invocala con argumentos JSON

        Para configurar tus propios MCP servers, edita `~/.mcp/config.json`
        """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  THAU AGI v2 + Ollama")
    print("=" * 60)
    print("\n  Requisitos:")
    print("    1. Ollama corriendo: ollama serve")
    print("    2. Modelo descargado: ollama pull llama3.2")
    print("\n  Abriendo en http://localhost:7860")
    print("  Presiona Ctrl+C para detener\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
