"""
THAU AGI - Sistema Integrado de Agente con Aprendizaje

Integra todos los componentes proto-AGI:
- Ciclo ReAct (Reason-Act-Observe-Reflect)
- Aprendizaje Experiencial
- Metacognición
- Estrategia Adaptativa
- Herramientas Ejecutables
- Memoria Multi-nivel
- Web Search & Research (búsqueda e investigación autónoma)

Este es el punto de entrada principal para THAU como proto-AGI.
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Componentes proto-AGI
from capabilities.proto_agi.thau_proto_agi import (
    ThauTools,
    ToolResult,
    ThoughtStep,
    AgentState,
    ToolType
)
from capabilities.proto_agi.experiential_learning import (
    ExperienceStore,
    MetacognitiveEngine,
    AdaptiveStrategy,
    Experience,
    OutcomeType,
    StrategyType,
    get_experience_store,
    get_metacognitive_engine,
    get_adaptive_strategy
)

# Web Search (opcional - funciona sin dependencias)
try:
    from capabilities.tools.web_search import (
        WebSearchTool,
        WebFetcher,
        ResearchAgent,
        web_search,
        fetch_url,
        research_topic
    )
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False


@dataclass
class AGIConfig:
    """Configuración del sistema AGI"""
    # Modelo
    checkpoint_path: str = "data/checkpoints/incremental/specialized/thau_v3_20251202_211505"
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    use_lora: bool = True

    # Comportamiento
    max_iterations: int = 10
    confidence_threshold: float = 0.6
    enable_learning: bool = True
    enable_metacognition: bool = True
    enable_web_search: bool = True  # Habilitar búsqueda web
    verbose: bool = True

    # Límites
    max_tokens: int = 500
    timeout_seconds: float = 60.0


class ThauAGI:
    """
    THAU AGI - Sistema de Inteligencia Artificial General (Proto)

    Características:
    1. Razonamiento con ciclo ReAct mejorado
    2. Aprendizaje de experiencias pasadas
    3. Auto-evaluación metacognitiva
    4. Adaptación de estrategias
    5. Uso inteligente de herramientas
    """

    def __init__(self, config: AGIConfig = None):
        self.config = config or AGIConfig()

        # Estado
        self.state: Optional[AgentState] = None
        self.pipe = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Componentes de aprendizaje
        self.experience_store = get_experience_store()
        self.metacognitive = get_metacognitive_engine()
        self.adaptive = get_adaptive_strategy()

        # Herramientas disponibles
        self.tools = {
            "execute_python": ThauTools.execute_python,
            "read_file": ThauTools.read_file,
            "write_file": ThauTools.write_file,
            "calculate": ThauTools.calculate,
            "list_directory": ThauTools.list_directory,
        }

        # Herramientas de web search (si están disponibles)
        self.web_search_tool = None
        self.web_fetcher = None
        self.research_agent = None

        if WEB_SEARCH_AVAILABLE and self.config.enable_web_search:
            self.web_search_tool = WebSearchTool()
            self.web_fetcher = WebFetcher()
            self.research_agent = ResearchAgent(
                search_tool=self.web_search_tool,
                fetcher=self.web_fetcher,
                verbose=self.config.verbose
            )
            # Agregar herramientas de búsqueda
            self.tools["web_search"] = self._web_search
            self.tools["fetch_url"] = self._fetch_url
            self.tools["research"] = self._research

        # Memoria de sesión
        self.session_memory = {
            "interactions": [],
            "context": {},
            "learnings": []
        }

        self._print_banner()

    def _print_banner(self):
        """Muestra banner de inicio"""
        print("\n" + "=" * 60)
        print("  ████████╗██╗  ██╗ █████╗ ██╗   ██╗")
        print("  ╚══██╔══╝██║  ██║██╔══██╗██║   ██║")
        print("     ██║   ███████║███████║██║   ██║")
        print("     ██║   ██╔══██║██╔══██║██║   ██║")
        print("     ██║   ██║  ██║██║  ██║╚██████╔╝")
        print("     ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝")
        print("           Proto-AGI System v1.1")
        print("=" * 60)
        print(f"  Session: {self.session_id}")
        print(f"  Learning: {'Enabled' if self.config.enable_learning else 'Disabled'}")
        print(f"  Metacognition: {'Enabled' if self.config.enable_metacognition else 'Disabled'}")
        print(f"  Web Search: {'Enabled' if (WEB_SEARCH_AVAILABLE and self.config.enable_web_search) else 'Disabled'}")
        print("=" * 60 + "\n")

    def load_model(self):
        """Carga el modelo THAU"""
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
        max_tokens = max_tokens or self.config.max_tokens

        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {"role": "user", "content": prompt},
        ]

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
        """Genera prompt del sistema con contexto de aprendizaje"""
        base_prompt = """Eres THAU, un asistente de IA avanzado con capacidades de razonamiento y aprendizaje.

Características:
- Piensas paso a paso antes de actuar
- Aprendes de experiencias pasadas
- Usas herramientas cuando es necesario
- Te auto-evalúas para mejorar
- Puedes buscar información en internet

Herramientas disponibles:
- calculate: Evaluar expresiones matemáticas
- execute_python: Ejecutar código Python
- read_file: Leer archivos
- write_file: Escribir archivos
- list_directory: Listar directorios"""

        # Agregar herramientas de búsqueda si están disponibles
        if self.web_search_tool:
            base_prompt += """
- web_search: Buscar información en internet
- fetch_url: Obtener contenido de una URL
- research: Investigar un tema en profundidad"""

        base_prompt += """

Para usar una herramienta, responde:
TOOL: nombre_herramienta
PARAMS: {"param": "valor"}

Responde siempre en español de forma clara y concisa."""

        # Agregar contexto de aprendizaje si está habilitado
        if self.config.enable_learning and self.session_memory["learnings"]:
            learnings = "\n".join(f"- {l}" for l in self.session_memory["learnings"][-5:])
            base_prompt += f"\n\nAprendizajes recientes:\n{learnings}"

        return base_prompt

    def _classify_goal(self, goal: str) -> str:
        """Clasifica el tipo de meta"""
        goal_lower = goal.lower()

        if any(w in goal_lower for w in ["calcula", "suma", "resta", "cuanto", "math", "+", "-", "*", "/"]):
            return "calculation"
        elif any(w in goal_lower for w in ["código", "programa", "función", "code", "script", "python"]):
            return "code"
        elif any(w in goal_lower for w in ["archivo", "lee", "escribe", "file", "read", "write"]):
            return "file"
        elif any(w in goal_lower for w in ["lista", "directorio", "carpeta", "archivos"]):
            return "directory"
        elif any(w in goal_lower for w in ["busca en internet", "busca en la web", "busca online", "investiga sobre", "research"]):
            return "web_search"
        elif any(w in goal_lower for w in ["busca", "encuentra", "search", "find"]):
            return "search"
        elif any(w in goal_lower for w in ["explica", "qué es", "cómo", "explain", "what", "how", "por qué", "define"]):
            return "question"
        else:
            return "general"

    def _detect_tool_need(self, goal: str) -> Optional[Tuple[str, Dict]]:
        """Detecta si necesitamos usar una herramienta"""
        goal_lower = goal.lower()
        import re

        # Cálculos
        if any(word in goal_lower for word in ["calcula", "suma", "resta", "multiplica", "divide", "cuanto es"]):
            numbers = re.findall(r'\d+\s*[\+\-\*\/]\s*\d+', goal)
            if numbers:
                return ("calculate", {"expression": numbers[0].replace(" ", "")})

            nums = re.findall(r'\d+', goal)
            if len(nums) >= 2:
                if "multiplica" in goal_lower or "x" in goal_lower:
                    return ("calculate", {"expression": f"{nums[0]}*{nums[1]}"})
                elif "suma" in goal_lower:
                    return ("calculate", {"expression": f"{nums[0]}+{nums[1]}"})
                elif "resta" in goal_lower:
                    return ("calculate", {"expression": f"{nums[0]}-{nums[1]}"})
                elif "divide" in goal_lower:
                    return ("calculate", {"expression": f"{nums[0]}/{nums[1]}"})

        # Listado de directorio
        elif any(word in goal_lower for word in ["lista", "archivos", "directorio", "carpeta"]):
            path_match = re.search(r'(?:en|de|del?)\s+["\']?([^"\']+)["\']?', goal)
            path = path_match.group(1).strip() if path_match else "."
            return ("list_directory", {"dirpath": path})

        # Lectura de archivos
        elif any(word in goal_lower for word in ["lee", "leer", "muestra", "contenido de", "abre"]):
            files = re.findall(r'[\w\-\.\/]+\.[a-zA-Z]{1,5}', goal)
            if files:
                return ("read_file", {"filepath": files[0]})

        # Ejecución de código
        elif any(word in goal_lower for word in ["ejecuta", "corre", "run"]) and "python" in goal_lower:
            code_match = re.search(r'["\'](.+?)["\']', goal)
            if code_match:
                return ("execute_python", {"code": code_match.group(1)})

        # Búsqueda web / Investigación
        elif self.web_search_tool and any(word in goal_lower for word in [
            "busca en internet", "busca en la web", "busca online",
            "investiga", "research", "qué es", "quién es", "cuándo",
            "noticias sobre", "información sobre", "define"
        ]):
            # Extraer el tema de búsqueda
            search_terms = goal
            for prefix in ["busca en internet", "busca en la web", "investiga sobre",
                          "busca online", "qué es", "quién es", "define", "información sobre"]:
                if prefix in goal_lower:
                    idx = goal_lower.find(prefix) + len(prefix)
                    search_terms = goal[idx:].strip()
                    break

            # Decidir entre búsqueda simple o investigación profunda
            if any(word in goal_lower for word in ["investiga", "research", "profundidad", "detalle"]):
                return ("research", {"query": search_terms, "depth": "normal"})
            else:
                return ("web_search", {"query": search_terms})

        # Obtener URL específica
        elif self.web_fetcher:
            url_match = re.search(r'https?://[^\s<>"\']+', goal)
            if url_match:
                return ("fetch_url", {"url": url_match.group(0)})

        return None

    def think(self, goal: str, context: str = "") -> Tuple[str, float]:
        """
        Fase de pensamiento con metacognición

        Returns:
            Tuple de (pensamiento, confianza)
        """
        # Consultar experiencias pasadas
        lessons = {}
        if self.config.enable_learning:
            goal_type = self._classify_goal(goal)
            lessons = self.experience_store.get_lessons_for_goal(goal, goal_type)

        prompt = f"""Meta: {goal}

{f'Contexto: {context}' if context else ''}

{f'Experiencias previas similares: {lessons.get("similar_experiences", 0)}' if lessons else ''}
{f'Tasa de éxito histórica: {lessons.get("pattern_success_rate", 0):.0%}' if lessons else ''}

Piensa sobre cómo resolver esto:
1. ¿Qué necesito hacer?
2. ¿Qué herramientas podrían ayudar?
3. ¿Cuál es el mejor enfoque?

Razonamiento:"""

        thought = self._generate(prompt, max_tokens=300)

        # Evaluar confianza
        confidence = 0.7  # Base
        if lessons:
            confidence = (confidence + lessons.get("confidence", 0.5)) / 2

        return thought, confidence

    def plan(self, goal: str, thought: str) -> List[str]:
        """
        Fase de planificación

        Returns:
            Lista de pasos del plan
        """
        prompt = f"""Meta: {goal}

Razonamiento previo:
{thought}

Crea un plan con pasos concretos (máximo 5 pasos):"""

        response = self._generate(prompt, max_tokens=300)

        # Extraer pasos
        lines = response.split('\n')
        steps = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Limpiar numeración
                step = line.lstrip('0123456789.-) ').strip()
                if step:
                    steps.append(step)

        return steps[:5]  # Máximo 5 pasos

    def act(self, action: str) -> Optional[ToolResult]:
        """
        Fase de acción - ejecuta herramientas

        Returns:
            ToolResult si se usó herramienta, None si no
        """
        tool_need = self._detect_tool_need(action)

        if not tool_need:
            return None

        tool_name, params = tool_need

        if self.config.verbose:
            print(f"   [TOOL] {tool_name}")
            print(f"   [PARAMS] {params}")

        # Ejecutar herramienta
        if tool_name == "calculate":
            result = self.tools[tool_name](params.get("expression", ""))
        elif tool_name == "read_file":
            result = self.tools[tool_name](params.get("filepath", ""))
        elif tool_name == "write_file":
            result = self.tools[tool_name](
                params.get("filepath", ""),
                params.get("content", "")
            )
        elif tool_name == "list_directory":
            result = self.tools[tool_name](params.get("dirpath", "."))
        elif tool_name == "execute_python":
            result = self.tools[tool_name](params.get("code", ""))
        # Herramientas de Web Search
        elif tool_name == "web_search":
            result = self._web_search(params.get("query", ""))
        elif tool_name == "fetch_url":
            result = self._fetch_url(params.get("url", ""))
        elif tool_name == "research":
            result = self._research(
                params.get("query", ""),
                params.get("depth", "normal")
            )
        else:
            result = ToolResult(tool_name, False, "", f"Herramienta desconocida: {tool_name}")

        if self.config.verbose:
            status = "✓" if result.success else "✗"
            print(f"   [{status}] {result.output[:100] if result.output else result.error}")

        return result

    def reflect(
        self,
        goal: str,
        response: str,
        tool_results: List[ToolResult]
    ) -> Dict[str, Any]:
        """
        Fase de reflexión y metacognición

        Returns:
            Dict con evaluación y lecciones
        """
        reflection = {
            "goal_achieved": False,
            "confidence": 0.5,
            "lessons_learned": [],
            "what_worked": [],
            "what_failed": [],
            "suggestions": []
        }

        # Evaluación metacognitiva
        if self.config.enable_metacognition:
            tool_results_dict = [
                {"success": r.success, "output": r.output, "error": r.error}
                for r in tool_results
            ]

            evaluation = self.metacognitive.evaluate_response(
                goal, response, tool_results_dict
            )

            reflection["confidence"] = evaluation["confidence"]
            reflection["suggestions"] = evaluation.get("suggestions", [])

            # Determinar si se logró la meta
            if evaluation["confidence"] >= self.config.confidence_threshold:
                reflection["goal_achieved"] = True

        # Analizar resultados de herramientas
        successful_tools = [r for r in tool_results if r.success]
        failed_tools = [r for r in tool_results if not r.success]

        if successful_tools:
            reflection["what_worked"].extend([
                f"Uso exitoso de {r.tool}" for r in successful_tools
            ])

        if failed_tools:
            reflection["what_failed"].extend([
                f"Fallo en {r.tool}: {r.error}" for r in failed_tools
            ])

        # Generar lecciones
        if reflection["goal_achieved"]:
            reflection["lessons_learned"].append(
                f"Enfoque efectivo para '{self._classify_goal(goal)}'"
            )
        else:
            reflection["lessons_learned"].append(
                "Considerar estrategia alternativa"
            )

        return reflection

    def run(self, goal: str, context: str = "") -> Dict[str, Any]:
        """
        Ejecuta el ciclo completo AGI para lograr una meta

        Args:
            goal: Meta a lograr
            context: Contexto adicional opcional

        Returns:
            Dict con respuesta, reflexión y métricas
        """
        start_time = time.time()

        if not self.pipe:
            self.load_model()

        # Inicializar estado
        self.state = AgentState(goal=goal)
        tool_results = []
        tools_used = []

        if self.config.verbose:
            print("\n" + "=" * 60)
            print(f"META: {goal}")
            print("=" * 60)

        # 1. SELECCIÓN DE ESTRATEGIA (adaptativa)
        if self.config.enable_learning:
            strategy, strategy_meta = self.adaptive.select_strategy(
                goal,
                list(self.tools.keys()),
                {"context": context}
            )

            if self.config.verbose:
                print(f"\n[STRATEGY] {strategy.value}")
                print(f"   Razón: {strategy_meta['reason']}")
                print(f"   Confianza previa: {strategy_meta['confidence']:.0%}")
        else:
            strategy = StrategyType.DIRECT
            strategy_meta = {"confidence": 0.5}

        # 2. THINK - Razonamiento
        if self.config.verbose:
            print("\n[THINK] Analizando...")

        thought, think_confidence = self.think(goal, context)

        self.state.thoughts.append(ThoughtStep(
            step_type="think",
            content=thought
        ))

        if self.config.verbose:
            print(f"   {thought[:200]}...")

        # 3. PLAN - Planificación
        if self.config.verbose:
            print("\n[PLAN] Creando plan...")

        plan_steps = self.plan(goal, thought)
        self.state.subgoals = plan_steps

        if self.config.verbose:
            for i, step in enumerate(plan_steps, 1):
                print(f"   {i}. {step}")

        # 4. ACT - Ejecución
        if self.config.verbose:
            print("\n[ACT] Ejecutando...")

        # Primero intentar acción directa sobre la meta
        direct_result = self.act(goal)
        if direct_result:
            tool_results.append(direct_result)
            tools_used.append(direct_result.tool)

        # Si no hubo herramienta directa, procesar pasos del plan
        if not direct_result:
            for step in plan_steps[:3]:  # Máximo 3 pasos
                step_result = self.act(step)
                if step_result:
                    tool_results.append(step_result)
                    tools_used.append(step_result.tool)

        # 5. GENERATE RESPONSE
        if self.config.verbose:
            print("\n[GENERATE] Generando respuesta...")

        if tool_results:
            # Respuesta basada en resultados de herramientas
            results_context = "\n".join([
                f"- {r.tool}: {r.output if r.success else r.error}"
                for r in tool_results
            ])

            response_prompt = f"""Meta del usuario: {goal}

Resultados obtenidos:
{results_context}

Genera una respuesta clara y útil basada en estos resultados:"""

            response = self._generate(response_prompt)
        else:
            # Respuesta directa
            response = self._generate(goal)

        # 6. REFLECT - Reflexión
        if self.config.verbose:
            print("\n[REFLECT] Evaluando...")

        reflection = self.reflect(goal, response, tool_results)

        self.state.thoughts.append(ThoughtStep(
            step_type="reflect",
            content=str(reflection)
        ))

        # 7. LEARN - Registrar experiencia
        execution_time = time.time() - start_time

        if self.config.enable_learning:
            # Determinar outcome
            if reflection["goal_achieved"]:
                outcome = OutcomeType.SUCCESS
            elif reflection["confidence"] >= 0.5:
                outcome = OutcomeType.PARTIAL_SUCCESS
            else:
                outcome = OutcomeType.FAILURE

            experience = self.adaptive.record_outcome(
                goal=goal,
                strategy=strategy,
                tools_used=tools_used,
                outcome=outcome,
                confidence=reflection["confidence"],
                execution_time=execution_time,
                what_worked=reflection["what_worked"],
                what_failed=reflection["what_failed"],
                lessons=reflection["lessons_learned"],
                context={"original_context": context}
            )

            if self.config.verbose:
                print(f"\n[LEARN] Experiencia registrada: {experience.id}")

            # Actualizar memoria de sesión
            self.session_memory["interactions"].append({
                "goal": goal,
                "outcome": outcome.value,
                "confidence": reflection["confidence"]
            })
            self.session_memory["learnings"].extend(reflection["lessons_learned"])

        # Actualizar estado final
        self.state.final_answer = response
        self.state.completed = reflection["goal_achieved"]

        # Resultado final
        result = {
            "response": response,
            "goal_achieved": reflection["goal_achieved"],
            "confidence": reflection["confidence"],
            "strategy_used": strategy.value,
            "tools_used": tools_used,
            "execution_time": execution_time,
            "reflection": reflection,
            "thought_steps": len(self.state.thoughts)
        }

        # Mostrar resultado
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("RESULTADO:")
            print("=" * 60)
            print(response)
            print("\n" + "-" * 60)
            print(f"Meta lograda: {'Sí' if result['goal_achieved'] else 'No'}")
            print(f"Confianza: {result['confidence']:.0%}")
            print(f"Tiempo: {result['execution_time']:.2f}s")
            print("=" * 60)

        return result

    # ==================== HERRAMIENTAS DE WEB SEARCH ====================

    def _web_search(self, query: str, num_results: int = 5) -> ToolResult:
        """
        Herramienta: Búsqueda web

        Args:
            query: Consulta de búsqueda
            num_results: Número de resultados

        Returns:
            ToolResult con resultados de búsqueda
        """
        if not self.web_search_tool:
            return ToolResult(
                tool="web_search",
                success=False,
                output="",
                error="Web search no disponible"
            )

        try:
            start_time = time.time()
            results = self.web_search_tool.search(query, num_results)

            if not results:
                return ToolResult(
                    tool="web_search",
                    success=False,
                    output="",
                    error="No se encontraron resultados",
                    execution_time=time.time() - start_time
                )

            # Formatear resultados
            output_parts = [f"Resultados para '{query}':\n"]
            for i, r in enumerate(results, 1):
                output_parts.append(f"{i}. {r.title}")
                output_parts.append(f"   URL: {r.url}")
                output_parts.append(f"   {r.snippet[:200]}...")
                output_parts.append("")

            return ToolResult(
                tool="web_search",
                success=True,
                output="\n".join(output_parts),
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return ToolResult(
                tool="web_search",
                success=False,
                output="",
                error=str(e)
            )

    def _fetch_url(self, url: str) -> ToolResult:
        """
        Herramienta: Obtener contenido de URL

        Args:
            url: URL a obtener

        Returns:
            ToolResult con contenido de la página
        """
        if not self.web_fetcher:
            return ToolResult(
                tool="fetch_url",
                success=False,
                output="",
                error="Web fetcher no disponible"
            )

        try:
            page = self.web_fetcher.fetch(url)

            if not page.success:
                return ToolResult(
                    tool="fetch_url",
                    success=False,
                    output="",
                    error=page.error,
                    execution_time=page.fetch_time
                )

            output = f"Título: {page.title}\n\n{page.content[:3000]}"

            return ToolResult(
                tool="fetch_url",
                success=True,
                output=output,
                execution_time=page.fetch_time
            )

        except Exception as e:
            return ToolResult(
                tool="fetch_url",
                success=False,
                output="",
                error=str(e)
            )

    def _research(self, query: str, depth: str = "normal") -> ToolResult:
        """
        Herramienta: Investigación profunda

        Args:
            query: Tema a investigar
            depth: Profundidad ("quick", "normal", "deep")

        Returns:
            ToolResult con resultados de investigación
        """
        if not self.research_agent:
            return ToolResult(
                tool="research",
                success=False,
                output="",
                error="Research agent no disponible"
            )

        try:
            result = self.research_agent.research(query, depth)

            # Formatear resultado
            output_parts = [
                f"Investigación: {query}",
                f"Confianza: {result.confidence:.0%}",
                f"\n{result.summary}",
            ]

            if result.key_facts:
                output_parts.append("\nHechos clave:")
                for fact in result.key_facts[:5]:
                    output_parts.append(f"  • {fact}")

            if result.related_topics:
                output_parts.append(f"\nTemas relacionados: {', '.join(result.related_topics)}")

            output_parts.append(f"\nFuentes consultadas: {len(result.sources)}")

            return ToolResult(
                tool="research",
                success=True,
                output="\n".join(output_parts),
                execution_time=result.research_time
            )

        except Exception as e:
            return ToolResult(
                tool="research",
                success=False,
                output="",
                error=str(e)
            )

    # ==================== FIN HERRAMIENTAS WEB SEARCH ====================

    def chat(self, message: str) -> str:
        """
        Interfaz simple de chat

        Args:
            message: Mensaje del usuario

        Returns:
            Respuesta del agente
        """
        result = self.run(message)
        return result["response"]

    def get_session_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de la sesión actual"""
        if self.config.enable_learning:
            return self.adaptive.end_session()
        else:
            return {
                "message": "Learning disabled",
                "interactions": len(self.session_memory["interactions"])
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas globales"""
        return self.experience_store.get_statistics()


def main():
    """Demo de THAU AGI"""
    # Configuración
    config = AGIConfig(
        verbose=True,
        enable_learning=True,
        enable_metacognition=True,
        max_iterations=5
    )

    # Crear agente
    agent = ThauAGI(config)

    # Tests
    print("\n" + "=" * 70)
    print("TEST 1: Cálculo matemático")
    print("=" * 70)
    agent.run("Calcula cuanto es 25 * 4 + 100")

    print("\n" + "=" * 70)
    print("TEST 2: Explorar directorio")
    print("=" * 70)
    agent.run("Lista los archivos del directorio actual")

    print("\n" + "=" * 70)
    print("TEST 3: Pregunta de conocimiento")
    print("=" * 70)
    agent.run("¿Qué es Python y para qué se usa?")

    print("\n" + "=" * 70)
    print("TEST 4: Leer archivo")
    print("=" * 70)
    agent.run("Lee el archivo README.md")

    # Resumen de sesión
    print("\n" + "=" * 70)
    print("RESUMEN DE SESIÓN")
    print("=" * 70)
    summary = agent.get_session_summary()
    print(f"Total interacciones: {summary.get('session_summary', {}).get('total_interactions', 0)}")
    print(f"Tasa de éxito: {summary.get('session_summary', {}).get('success_rate', 0):.0%}")

    # Estadísticas globales
    print("\n" + "=" * 70)
    print("ESTADÍSTICAS GLOBALES")
    print("=" * 70)
    stats = agent.get_statistics()
    print(f"Total experiencias: {stats['total_experiences']}")
    print(f"Tasa de éxito global: {stats['success_rate']:.0%}")


if __name__ == "__main__":
    main()
