"""
THAU AGI v2 - Sistema Proto-AGI Unificado

Integra TODOS los componentes desarrollados:
1. Ciclo ReAct (Reason-Act-Observe-Reflect)
2. Aprendizaje Experiencial
3. Metacognición
4. Estrategia Adaptativa
5. Web Search & Research
6. Sistema Multi-Agente
7. Knowledge Base con RAG
8. Feedback Loop

Este es el sistema proto-AGI completo de THAU.
"""

import json
import time
import uuid
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Core imports
try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Proto-AGI components
from capabilities.proto_agi.thau_proto_agi import (
    ThauTools, ToolResult, ThoughtStep, AgentState
)

from capabilities.proto_agi.experiential_learning import (
    ExperienceStore, MetacognitiveEngine, AdaptiveStrategy,
    Experience, OutcomeType, StrategyType,
    get_experience_store, get_metacognitive_engine, get_adaptive_strategy
)

from capabilities.proto_agi.multi_agent import (
    MultiAgentSystem, AgentRole as MultiAgentRole,
    SpecializedAgent, AgentCoordinator,
    MessageType, TaskPriority
)

from capabilities.proto_agi.knowledge_base import (
    KnowledgeStore, ContextBuilder, KnowledgeLearner, FeedbackSystem,
    KnowledgeType, RetrievalStrategy,
    get_knowledge_store, get_context_builder, get_knowledge_learner, get_feedback_system
)

# Web search (optional)
try:
    from capabilities.tools.web_search import (
        WebSearchTool, WebFetcher, ResearchAgent
    )
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False


class ThauMode(Enum):
    """Modos de operación de THAU"""
    CHAT = "chat"                    # Conversación simple
    TASK = "task"                    # Ejecución de tarea
    RESEARCH = "research"            # Investigación
    COLLABORATIVE = "collaborative"  # Multi-agente
    LEARNING = "learning"            # Modo de aprendizaje intensivo


@dataclass
class ThauConfig:
    """Configuración completa de THAU AGI v2"""
    # Modelo
    checkpoint_path: str = "data/checkpoints/incremental/specialized/thau_v3_20251202_211505"
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    use_lora: bool = True

    # Comportamiento
    max_iterations: int = 10
    confidence_threshold: float = 0.6
    default_mode: ThauMode = ThauMode.CHAT

    # Features
    enable_learning: bool = True
    enable_metacognition: bool = True
    enable_web_search: bool = True
    enable_multi_agent: bool = True
    enable_knowledge_base: bool = True
    enable_feedback: bool = True

    # Límites
    max_tokens: int = 500
    max_context_length: int = 2000
    timeout_seconds: float = 120.0

    # Verbose
    verbose: bool = True


class ThauAGIv2:
    """
    THAU AGI v2 - Sistema Proto-AGI Completo

    Capacidades:
    1. Razonamiento con ciclo ReAct mejorado
    2. Aprendizaje de experiencias pasadas
    3. Auto-evaluación metacognitiva
    4. Adaptación de estrategias
    5. Búsqueda e investigación web
    6. Colaboración multi-agente
    7. Base de conocimiento con RAG
    8. Feedback loop para mejora continua
    """

    def __init__(self, config: ThauConfig = None):
        self.config = config or ThauConfig()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Estado
        self.state: Optional[AgentState] = None
        self.pipe = None
        self.current_mode = self.config.default_mode
        self.interaction_count = 0

        # === COMPONENTES ===

        # 1. Herramientas básicas
        self.tools = {
            "execute_python": ThauTools.execute_python,
            "read_file": ThauTools.read_file,
            "write_file": ThauTools.write_file,
            "calculate": ThauTools.calculate,
            "list_directory": ThauTools.list_directory,
        }

        # 2. Aprendizaje experiencial
        if self.config.enable_learning:
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

        if self.config.enable_web_search and WEB_SEARCH_AVAILABLE:
            self.web_search = WebSearchTool()
            self.web_fetcher = WebFetcher()
            self.research_agent = ResearchAgent(
                search_tool=self.web_search,
                fetcher=self.web_fetcher,
                verbose=self.config.verbose
            )
            self.tools["web_search"] = self._web_search
            self.tools["fetch_url"] = self._fetch_url
            self.tools["research"] = self._research

        # 4. Multi-agente
        self.multi_agent_system = None
        if self.config.enable_multi_agent:
            self.multi_agent_system = MultiAgentSystem(verbose=False)
            self.multi_agent_system.initialize([
                MultiAgentRole.CODER,
                MultiAgentRole.REVIEWER,
                MultiAgentRole.RESEARCHER,
                MultiAgentRole.PLANNER,
                MultiAgentRole.TESTER
            ])

        # 5. Knowledge base
        if self.config.enable_knowledge_base:
            self.knowledge_store = get_knowledge_store()
            self.context_builder = get_context_builder()
            self.knowledge_learner = get_knowledge_learner()
        else:
            self.knowledge_store = None
            self.context_builder = None
            self.knowledge_learner = None

        # 6. Feedback system
        if self.config.enable_feedback:
            self.feedback_system = get_feedback_system()
        else:
            self.feedback_system = None

        # Memoria de sesión
        self.session_memory = {
            "interactions": [],
            "context": {},
            "learnings": [],
            "conversation_history": []
        }

        self._print_banner()

    def _print_banner(self):
        """Muestra banner de inicio"""
        if not self.config.verbose:
            return

        print("\n" + "=" * 70)
        print("""
  ████████╗██╗  ██╗ █████╗ ██╗   ██╗     █████╗  ██████╗ ██╗
  ╚══██╔══╝██║  ██║██╔══██╗██║   ██║    ██╔══██╗██╔════╝ ██║
     ██║   ███████║███████║██║   ██║    ███████║██║  ███╗██║
     ██║   ██╔══██║██╔══██║██║   ██║    ██╔══██║██║   ██║██║
     ██║   ██║  ██║██║  ██║╚██████╔╝    ██║  ██║╚██████╔╝██║
     ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝
                    Proto-AGI System v2.0
        """)
        print("=" * 70)
        print(f"  Session: {self.session_id}")
        print(f"  Mode: {self.current_mode.value}")
        print("-" * 70)
        print("  Components:")
        print(f"    Learning:      {'✓' if self.config.enable_learning else '✗'}")
        print(f"    Metacognition: {'✓' if self.config.enable_metacognition else '✗'}")
        print(f"    Web Search:    {'✓' if (self.config.enable_web_search and WEB_SEARCH_AVAILABLE) else '✗'}")
        print(f"    Multi-Agent:   {'✓' if self.config.enable_multi_agent else '✗'}")
        print(f"    Knowledge:     {'✓' if self.config.enable_knowledge_base else '✗'}")
        print(f"    Feedback:      {'✓' if self.config.enable_feedback else '✗'}")
        print("=" * 70 + "\n")

    def load_model(self):
        """Carga el modelo THAU"""
        if not TRANSFORMERS_AVAILABLE:
            print("[WARN] Transformers no disponible, usando modo sin modelo")
            return

        if self.config.verbose:
            print("[LOADING] Cargando modelo THAU...")

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            device_map="auto",
            torch_dtype="auto",
        )

        if self.config.use_lora:
            try:
                model = PeftModel.from_pretrained(model, self.config.checkpoint_path)
                model = model.merge_and_unload()
                if self.config.verbose:
                    print("[OK] LoRA checkpoint aplicado!")
            except Exception as e:
                if self.config.verbose:
                    print(f"[WARN] Usando modelo base: {e}")

        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        if self.config.verbose:
            print("[OK] Modelo cargado!\n")

    def _generate(self, prompt: str, max_tokens: int = None) -> str:
        """Genera respuesta del modelo"""
        if not self.pipe:
            # Modo sin modelo - respuesta básica
            return f"[Sin modelo] Procesando: {prompt[:100]}..."

        max_tokens = max_tokens or self.config.max_tokens

        # Construir contexto si knowledge base está habilitado
        context = ""
        if self.context_builder:
            context = self.context_builder.build_context(prompt, n_items=3)

        system_prompt = self._get_system_prompt()
        if context:
            system_prompt += f"\n\n{context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Agregar historial de conversación
        for msg in self.session_memory["conversation_history"][-3:]:
            messages.insert(-1, msg)

        outputs = self.pipe(
            messages,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
        )

        return outputs[0]["generated_text"][-1]["content"]

    def _get_system_prompt(self) -> str:
        """Genera prompt del sistema"""
        prompt = """Eres THAU, un asistente de IA avanzado con capacidades proto-AGI.

Características:
- Piensas paso a paso antes de actuar
- Aprendes de experiencias pasadas
- Usas herramientas cuando es necesario
- Te auto-evalúas para mejorar
- Colaboras con agentes especializados
- Consultas tu base de conocimiento

Herramientas disponibles:
- calculate: Evaluar expresiones matemáticas
- execute_python: Ejecutar código Python
- read_file: Leer archivos
- write_file: Escribir archivos
- list_directory: Listar directorios"""

        if self.web_search:
            prompt += """
- web_search: Buscar en internet
- fetch_url: Obtener contenido de URL
- research: Investigar tema en profundidad"""

        prompt += """

Para usar una herramienta, responde:
TOOL: nombre_herramienta
PARAMS: {"param": "valor"}

Responde siempre en español de forma clara y concisa."""

        # Agregar aprendizajes recientes
        if self.session_memory["learnings"]:
            learnings = "\n".join(f"- {l}" for l in self.session_memory["learnings"][-5:])
            prompt += f"\n\nAprendizajes recientes:\n{learnings}"

        return prompt

    # === HERRAMIENTAS WEB ===

    def _web_search(self, query: str, num_results: int = 5) -> ToolResult:
        """Búsqueda web"""
        if not self.web_search:
            return ToolResult("web_search", False, "", "Web search no disponible")

        try:
            start = time.time()
            results = self.web_search.search(query, num_results)

            if not results:
                return ToolResult("web_search", False, "", "Sin resultados")

            output = f"Resultados para '{query}':\n"
            for i, r in enumerate(results, 1):
                output += f"\n{i}. {r.title}\n   {r.url}\n   {r.snippet[:150]}..."

            return ToolResult("web_search", True, output, execution_time=time.time() - start)
        except Exception as e:
            return ToolResult("web_search", False, "", str(e))

    def _fetch_url(self, url: str) -> ToolResult:
        """Obtener contenido de URL"""
        if not self.web_fetcher:
            return ToolResult("fetch_url", False, "", "Web fetcher no disponible")

        try:
            page = self.web_fetcher.fetch(url)
            if page.success:
                return ToolResult("fetch_url", True, f"{page.title}\n\n{page.content[:2000]}")
            return ToolResult("fetch_url", False, "", page.error)
        except Exception as e:
            return ToolResult("fetch_url", False, "", str(e))

    def _research(self, query: str, depth: str = "normal") -> ToolResult:
        """Investigación profunda"""
        if not self.research_agent:
            return ToolResult("research", False, "", "Research agent no disponible")

        try:
            result = self.research_agent.research(query, depth)

            output = f"Investigación: {query}\n"
            output += f"Confianza: {result.confidence:.0%}\n\n"
            output += result.summary

            if result.key_facts:
                output += "\n\nHechos clave:"
                for fact in result.key_facts[:5]:
                    output += f"\n  • {fact}"

            return ToolResult("research", True, output, execution_time=result.research_time)
        except Exception as e:
            return ToolResult("research", False, "", str(e))

    # === CICLO PRINCIPAL ===

    def _classify_goal(self, goal: str) -> str:
        """Clasifica el tipo de meta"""
        goal_lower = goal.lower()

        classifications = [
            (["calcula", "suma", "resta", "cuanto", "+", "-", "*", "/"], "calculation"),
            (["código", "programa", "función", "code", "script"], "code"),
            (["archivo", "lee", "escribe", "file"], "file"),
            (["lista", "directorio", "carpeta"], "directory"),
            (["busca en internet", "investiga", "research"], "web_search"),
            (["busca", "encuentra", "search"], "search"),
            (["explica", "qué es", "cómo", "define"], "question"),
        ]

        for keywords, category in classifications:
            if any(kw in goal_lower for kw in keywords):
                return category

        return "general"

    def _detect_tool_need(self, goal: str) -> Optional[Tuple[str, Dict]]:
        """Detecta si necesitamos usar una herramienta"""
        import re
        goal_lower = goal.lower()

        # Cálculos
        if any(w in goal_lower for w in ["calcula", "suma", "resta", "multiplica", "divide", "cuanto es"]):
            numbers = re.findall(r'\d+\s*[\+\-\*\/]\s*\d+', goal)
            if numbers:
                return ("calculate", {"expression": numbers[0].replace(" ", "")})

            nums = re.findall(r'\d+', goal)
            if len(nums) >= 2:
                ops = {"multiplica": "*", "suma": "+", "resta": "-", "divide": "/"}
                for word, op in ops.items():
                    if word in goal_lower:
                        return ("calculate", {"expression": f"{nums[0]}{op}{nums[1]}"})

        # Directorio
        elif any(w in goal_lower for w in ["lista", "archivos", "directorio"]):
            return ("list_directory", {"dirpath": "."})

        # Leer archivo
        elif any(w in goal_lower for w in ["lee", "leer", "muestra", "contenido"]):
            files = re.findall(r'[\w\-\.\/]+\.[a-zA-Z]{1,5}', goal)
            if files:
                return ("read_file", {"filepath": files[0]})

        # Web search
        elif self.web_search and any(w in goal_lower for w in [
            "busca en internet", "investiga", "qué es", "quién es", "define"
        ]):
            search_term = goal
            for prefix in ["busca en internet", "investiga sobre", "qué es", "define"]:
                if prefix in goal_lower:
                    idx = goal_lower.find(prefix) + len(prefix)
                    search_term = goal[idx:].strip()
                    break

            if "investiga" in goal_lower:
                return ("research", {"query": search_term})
            return ("web_search", {"query": search_term})

        # URL
        elif self.web_fetcher:
            url_match = re.search(r'https?://[^\s<>"\']+', goal)
            if url_match:
                return ("fetch_url", {"url": url_match.group(0)})

        return None

    def _execute_tool(self, tool_name: str, params: Dict) -> ToolResult:
        """Ejecuta una herramienta"""
        if tool_name == "calculate":
            return self.tools[tool_name](params.get("expression", ""))
        elif tool_name == "read_file":
            return self.tools[tool_name](params.get("filepath", ""))
        elif tool_name == "write_file":
            return self.tools[tool_name](params.get("filepath", ""), params.get("content", ""))
        elif tool_name == "list_directory":
            return self.tools[tool_name](params.get("dirpath", "."))
        elif tool_name == "execute_python":
            return self.tools[tool_name](params.get("code", ""))
        elif tool_name == "web_search":
            return self._web_search(params.get("query", ""))
        elif tool_name == "fetch_url":
            return self._fetch_url(params.get("url", ""))
        elif tool_name == "research":
            return self._research(params.get("query", ""), params.get("depth", "normal"))
        else:
            return ToolResult(tool_name, False, "", f"Herramienta desconocida: {tool_name}")

    def run(self, goal: str, mode: ThauMode = None) -> Dict[str, Any]:
        """
        Ejecuta el ciclo AGI completo

        Args:
            goal: Meta a lograr
            mode: Modo de operación (opcional)

        Returns:
            Dict con respuesta y métricas
        """
        start_time = time.time()
        mode = mode or self.current_mode
        self.interaction_count += 1
        interaction_id = f"{self.session_id}_{self.interaction_count}"

        # Cargar modelo si no está cargado
        if not self.pipe and TRANSFORMERS_AVAILABLE:
            self.load_model()

        # Inicializar estado
        self.state = AgentState(goal=goal)
        tool_results = []
        tools_used = []

        if self.config.verbose:
            print(f"\n{'=' * 60}")
            print(f"META: {goal}")
            print(f"MODO: {mode.value}")
            print("=" * 60)

        # === 1. SELECCIÓN DE ESTRATEGIA ===
        strategy = StrategyType.DIRECT
        strategy_meta = {"confidence": 0.5}

        if self.adaptive:
            strategy, strategy_meta = self.adaptive.select_strategy(
                goal, list(self.tools.keys()), {}
            )
            if self.config.verbose:
                print(f"\n[STRATEGY] {strategy.value} (confianza: {strategy_meta['confidence']:.0%})")

        # === 2. THINK - Razonamiento ===
        if self.config.verbose:
            print("\n[THINK] Analizando...")

        # Obtener contexto de knowledge base
        context = ""
        if self.context_builder:
            context = self.context_builder.build_context(goal, n_items=3)

        thought = self._generate(f"Meta: {goal}\n{context}\n\nPiensa cómo resolver esto:", max_tokens=200)
        self.state.thoughts.append(ThoughtStep(step_type="think", content=thought))

        if self.config.verbose:
            print(f"   {thought[:150]}...")

        # === 3. ACT - Ejecución ===
        if self.config.verbose:
            print("\n[ACT] Ejecutando...")

        # Detectar necesidad de herramienta
        tool_need = self._detect_tool_need(goal)

        if tool_need:
            tool_name, params = tool_need
            if self.config.verbose:
                print(f"   [TOOL] {tool_name}: {params}")

            result = self._execute_tool(tool_name, params)
            tool_results.append(result)
            tools_used.append(tool_name)

            if self.config.verbose:
                status = "✓" if result.success else "✗"
                print(f"   [{status}] {result.output[:100] if result.output else result.error}")

        # === 4. MODO COLABORATIVO (opcional) ===
        if mode == ThauMode.COLLABORATIVE and self.multi_agent_system:
            if self.config.verbose:
                print("\n[COLLABORATE] Activando agentes...")

            # Determinar roles necesarios
            goal_type = self._classify_goal(goal)
            roles_needed = self._determine_roles_needed(goal_type)

            collab = self.multi_agent_system.collaborate(goal, roles_needed)
            self.session_memory["context"]["last_collaboration"] = collab

        # === 5. GENERATE RESPONSE ===
        if self.config.verbose:
            print("\n[GENERATE] Generando respuesta...")

        if tool_results:
            results_context = "\n".join([
                f"- {r.tool}: {r.output if r.success else r.error}"
                for r in tool_results
            ])
            prompt = f"Meta: {goal}\n\nResultados:\n{results_context}\n\nResponde de forma clara:"
        else:
            prompt = goal

        response = self._generate(prompt)

        # === 6. REFLECT - Reflexión ===
        if self.config.verbose:
            print("\n[REFLECT] Evaluando...")

        reflection = {
            "goal_achieved": True,
            "confidence": 0.7,
            "what_worked": [],
            "what_failed": [],
            "lessons_learned": []
        }

        if self.metacognitive:
            tool_results_dict = [{"success": r.success, "output": r.output} for r in tool_results]
            evaluation = self.metacognitive.evaluate_response(goal, response, tool_results_dict)
            reflection["confidence"] = evaluation["confidence"]
            reflection["goal_achieved"] = evaluation["confidence"] >= self.config.confidence_threshold

        # === 7. LEARN - Aprender ===
        execution_time = time.time() - start_time

        if self.adaptive:
            outcome = OutcomeType.SUCCESS if reflection["goal_achieved"] else OutcomeType.FAILURE

            self.adaptive.record_outcome(
                goal=goal,
                strategy=strategy,
                tools_used=tools_used,
                outcome=outcome,
                confidence=reflection["confidence"],
                execution_time=execution_time,
                what_worked=reflection["what_worked"],
                what_failed=reflection["what_failed"],
                lessons=reflection["lessons_learned"]
            )

        # Aprender en knowledge base
        if self.knowledge_learner:
            self.knowledge_learner.learn_from_conversation(goal, response, reflection["goal_achieved"])

        # === 8. ACTUALIZAR MEMORIA ===
        self.session_memory["conversation_history"].append({"role": "user", "content": goal})
        self.session_memory["conversation_history"].append({"role": "assistant", "content": response})
        self.session_memory["interactions"].append({
            "id": interaction_id,
            "goal": goal,
            "response": response[:500],
            "success": reflection["goal_achieved"]
        })

        # Actualizar estado
        self.state.final_answer = response
        self.state.completed = reflection["goal_achieved"]

        # Resultado
        result = {
            "response": response,
            "interaction_id": interaction_id,
            "goal_achieved": reflection["goal_achieved"],
            "confidence": reflection["confidence"],
            "strategy_used": strategy.value,
            "tools_used": tools_used,
            "mode": mode.value,
            "execution_time": execution_time
        }

        # Mostrar resultado
        if self.config.verbose:
            print(f"\n{'=' * 60}")
            print("RESULTADO:")
            print("=" * 60)
            print(response)
            print(f"\n{'-' * 60}")
            print(f"Meta lograda: {'Sí' if result['goal_achieved'] else 'No'}")
            print(f"Confianza: {result['confidence']:.0%}")
            print(f"Tiempo: {result['execution_time']:.2f}s")
            print("=" * 60)

        return result

    def _determine_roles_needed(self, goal_type: str) -> List[MultiAgentRole]:
        """Determina qué roles de agentes se necesitan"""
        role_mapping = {
            "code": [MultiAgentRole.CODER, MultiAgentRole.REVIEWER],
            "calculation": [MultiAgentRole.CODER],
            "file": [MultiAgentRole.CODER],
            "web_search": [MultiAgentRole.RESEARCHER],
            "search": [MultiAgentRole.RESEARCHER],
            "question": [MultiAgentRole.RESEARCHER, MultiAgentRole.PLANNER],
            "general": [MultiAgentRole.PLANNER]
        }
        return role_mapping.get(goal_type, [MultiAgentRole.GENERALIST])

    # === INTERFACES DE USUARIO ===

    def chat(self, message: str) -> str:
        """Interfaz simple de chat"""
        result = self.run(message, ThauMode.CHAT)
        return result["response"]

    def research(self, topic: str) -> str:
        """Interfaz de investigación"""
        result = self.run(f"Investiga sobre: {topic}", ThauMode.RESEARCH)
        return result["response"]

    def execute_task(self, task: str) -> Dict[str, Any]:
        """Ejecuta tarea compleja"""
        return self.run(task, ThauMode.TASK)

    def collaborate(self, task: str) -> Dict[str, Any]:
        """Ejecuta tarea colaborativa"""
        return self.run(task, ThauMode.COLLABORATIVE)

    # === FEEDBACK ===

    def thumbs_up(self, interaction_id: str = None) -> None:
        """Feedback positivo"""
        if self.feedback_system:
            iid = interaction_id or self.session_memory["interactions"][-1]["id"]
            self.feedback_system.thumbs_up(iid)
            if self.config.verbose:
                print("[FEEDBACK] 👍 Registrado")

    def thumbs_down(self, interaction_id: str = None, reason: str = "") -> None:
        """Feedback negativo"""
        if self.feedback_system:
            iid = interaction_id or self.session_memory["interactions"][-1]["id"]
            self.feedback_system.thumbs_down(iid, reason=reason)
            if self.config.verbose:
                print("[FEEDBACK] 👎 Registrado")

    def correct(self, correction: str, interaction_id: str = None) -> None:
        """Corregir respuesta"""
        if self.feedback_system:
            iid = interaction_id or self.session_memory["interactions"][-1]["id"]
            original = self.session_memory["interactions"][-1].get("response", "")
            self.feedback_system.correct(iid, correction, original)
            if self.config.verbose:
                print("[FEEDBACK] Corrección registrada")

    # === ESTADÍSTICAS ===

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema"""
        stats = {
            "session_id": self.session_id,
            "interactions": self.interaction_count,
            "mode": self.current_mode.value,
        }

        if self.experience_store:
            stats["experiences"] = self.experience_store.get_statistics()

        if self.knowledge_store:
            stats["knowledge"] = self.knowledge_store.get_stats()

        if self.feedback_system:
            stats["feedback"] = self.feedback_system.get_stats()

        if self.multi_agent_system:
            stats["multi_agent"] = self.multi_agent_system.get_status()

        return stats

    def get_session_summary(self) -> Dict[str, Any]:
        """Resumen de la sesión"""
        summary = {
            "session_id": self.session_id,
            "total_interactions": self.interaction_count,
            "conversation_length": len(self.session_memory["conversation_history"]),
            "learnings": len(self.session_memory["learnings"])
        }

        if self.adaptive:
            reflection = self.adaptive.end_session()
            summary["session_reflection"] = reflection.get("session_summary", {})

        return summary


def main():
    """Demo de THAU AGI v2"""
    config = ThauConfig(
        verbose=True,
        enable_learning=True,
        enable_metacognition=True,
        enable_web_search=WEB_SEARCH_AVAILABLE,
        enable_multi_agent=True,
        enable_knowledge_base=True,
        enable_feedback=True
    )

    agent = ThauAGIv2(config)

    # Tests
    print("\n" + "=" * 70)
    print("TEST 1: Cálculo")
    print("=" * 70)
    agent.run("Calcula 25 * 4 + 100")

    print("\n" + "=" * 70)
    print("TEST 2: Exploración")
    print("=" * 70)
    agent.run("Lista los archivos del directorio actual")

    if WEB_SEARCH_AVAILABLE:
        print("\n" + "=" * 70)
        print("TEST 3: Búsqueda web")
        print("=" * 70)
        agent.run("Busca en internet qué es Python")

    print("\n" + "=" * 70)
    print("TEST 4: Pregunta")
    print("=" * 70)
    agent.run("¿Qué es una función recursiva?")

    # Feedback demo
    agent.thumbs_up()

    # Estadísticas
    print("\n" + "=" * 70)
    print("ESTADÍSTICAS")
    print("=" * 70)
    stats = agent.get_stats()
    print(f"Interacciones: {stats['interactions']}")
    if 'experiences' in stats:
        print(f"Experiencias: {stats['experiences'].get('total_experiences', 0)}")
    if 'feedback' in stats:
        print(f"Satisfacción: {stats['feedback'].get('satisfaction_rate', 0):.0%}")

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE SESIÓN")
    print("=" * 70)
    summary = agent.get_session_summary()
    print(f"Total interacciones: {summary['total_interactions']}")


if __name__ == "__main__":
    main()
