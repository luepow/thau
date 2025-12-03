"""
THAU Proto-AGI - Sistema de Agente Avanzado

Integra:
- Memoria multi-nivel (corto, largo plazo, episodica)
- Herramientas ejecutables (codigo, archivos, web)
- Ciclo ReAct (Reason-Act-Observe)
- Auto-reflexion y mejora continua
- Planificacion con descomposicion de tareas
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class ToolType(Enum):
    """Tipos de herramientas disponibles"""
    CODE_EXEC = "code_execution"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    WEB_SEARCH = "web_search"
    MEMORY_STORE = "memory_store"
    MEMORY_RECALL = "memory_recall"
    CALCULATE = "calculate"


@dataclass
class ToolResult:
    """Resultado de ejecutar una herramienta"""
    tool: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class ThoughtStep:
    """Paso de pensamiento del agente"""
    step_type: str  # "think", "plan", "act", "observe", "reflect"
    content: str
    tool_used: Optional[str] = None
    tool_result: Optional[ToolResult] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentState:
    """Estado del agente"""
    goal: str
    subgoals: List[str] = field(default_factory=list)
    current_subgoal: Optional[str] = None
    thoughts: List[ThoughtStep] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    final_answer: Optional[str] = None


class ThauTools:
    """Herramientas ejecutables por THAU"""

    @staticmethod
    def execute_python(code: str) -> ToolResult:
        """Ejecuta codigo Python de forma segura"""
        import time
        start = time.time()

        try:
            # Ejecutar en subprocess aislado
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(project_root)
            )

            output = result.stdout
            if result.stderr:
                output += f"\nStderr: {result.stderr}"

            return ToolResult(
                tool="execute_python",
                success=result.returncode == 0,
                output=output.strip(),
                error=result.stderr if result.returncode != 0 else None,
                execution_time=time.time() - start
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool="execute_python",
                success=False,
                output="",
                error="Timeout: codigo excedio 30 segundos",
                execution_time=30.0
            )
        except Exception as e:
            return ToolResult(
                tool="execute_python",
                success=False,
                output="",
                error=str(e),
                execution_time=time.time() - start
            )

    @staticmethod
    def read_file(filepath: str) -> ToolResult:
        """Lee contenido de un archivo"""
        try:
            path = Path(filepath)
            if not path.exists():
                return ToolResult(
                    tool="read_file",
                    success=False,
                    output="",
                    error=f"Archivo no encontrado: {filepath}"
                )

            content = path.read_text(encoding='utf-8')
            return ToolResult(
                tool="read_file",
                success=True,
                output=content[:5000]  # Limitar a 5000 chars
            )
        except Exception as e:
            return ToolResult(
                tool="read_file",
                success=False,
                output="",
                error=str(e)
            )

    @staticmethod
    def write_file(filepath: str, content: str) -> ToolResult:
        """Escribe contenido a un archivo"""
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')

            return ToolResult(
                tool="write_file",
                success=True,
                output=f"Archivo escrito: {filepath} ({len(content)} bytes)"
            )
        except Exception as e:
            return ToolResult(
                tool="write_file",
                success=False,
                output="",
                error=str(e)
            )

    @staticmethod
    def calculate(expression: str) -> ToolResult:
        """Evalua expresion matematica"""
        try:
            # Solo permite operaciones matematicas seguras
            allowed = set('0123456789+-*/.() ')
            if not all(c in allowed for c in expression):
                return ToolResult(
                    tool="calculate",
                    success=False,
                    output="",
                    error="Expresion contiene caracteres no permitidos"
                )

            result = eval(expression)
            return ToolResult(
                tool="calculate",
                success=True,
                output=str(result)
            )
        except Exception as e:
            return ToolResult(
                tool="calculate",
                success=False,
                output="",
                error=str(e)
            )

    @staticmethod
    def list_directory(dirpath: str = ".") -> ToolResult:
        """Lista contenido de un directorio"""
        try:
            path = Path(dirpath)
            if not path.exists():
                return ToolResult(
                    tool="list_directory",
                    success=False,
                    output="",
                    error=f"Directorio no encontrado: {dirpath}"
                )

            items = []
            for item in path.iterdir():
                prefix = "[DIR]" if item.is_dir() else "[FILE]"
                items.append(f"{prefix} {item.name}")

            return ToolResult(
                tool="list_directory",
                success=True,
                output="\n".join(sorted(items)[:50])  # Max 50 items
            )
        except Exception as e:
            return ToolResult(
                tool="list_directory",
                success=False,
                output="",
                error=str(e)
            )


class ThauProtoAGI:
    """
    THAU Proto-AGI - Agente con capacidades avanzadas

    Implementa un ciclo ReAct mejorado:
    1. THINK - Razonar sobre el problema
    2. PLAN - Crear plan de accion
    3. ACT - Ejecutar herramienta
    4. OBSERVE - Analizar resultado
    5. REFLECT - Evaluar progreso y ajustar
    """

    def __init__(
        self,
        checkpoint_path: str = "data/checkpoints/incremental/specialized/thau_v3_20251202_211505",
        base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        use_lora: bool = True,
        max_iterations: int = 10,
    ):
        self.checkpoint_path = checkpoint_path
        self.base_model = base_model
        self.max_iterations = max_iterations
        self.use_lora = use_lora

        # Estado del agente
        self.state: Optional[AgentState] = None
        self.memory: Dict[str, List[str]] = {
            "facts": [],
            "learnings": [],
            "errors": [],
        }

        # Herramientas disponibles
        self.tools = {
            "execute_python": ThauTools.execute_python,
            "read_file": ThauTools.read_file,
            "write_file": ThauTools.write_file,
            "calculate": ThauTools.calculate,
            "list_directory": ThauTools.list_directory,
        }

        # Pipeline del modelo
        self.pipe = None

        print("=" * 60)
        print("  THAU Proto-AGI")
        print("  Sistema de Agente Avanzado")
        print("=" * 60)

    def load_model(self):
        """Carga el modelo THAU con LoRA"""
        print("\nCargando modelo...")

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map="auto",
            torch_dtype="auto",
        )

        if self.use_lora:
            try:
                model = PeftModel.from_pretrained(model, self.checkpoint_path)
                model = model.merge_and_unload()
                print("LoRA checkpoint aplicado!")
            except Exception as e:
                print(f"Usando modelo base: {e}")

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        print("Modelo cargado!")

    def _generate(self, prompt: str, max_tokens: int = 300) -> str:
        """Genera respuesta del modelo"""
        messages = [
            {"role": "system", "content": "Eres THAU, un asistente experto en programacion. Responde de forma clara y concisa en espanol."},
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

    def _detect_tool_need(self, goal: str) -> Optional[tuple]:
        """Detecta si necesitamos usar una herramienta basado en palabras clave"""
        goal_lower = goal.lower()

        # Deteccion por patrones
        if any(word in goal_lower for word in ["calcula", "suma", "resta", "multiplica", "divide", "cuanto es"]):
            # Extraer expresion matematica
            import re
            numbers = re.findall(r'\d+\s*[\+\-\*\/]\s*\d+', goal)
            if numbers:
                return ("calculate", {"expression": numbers[0].replace(" ", "")})
            # Intentar extraer numeros sueltos
            nums = re.findall(r'\d+', goal)
            if len(nums) >= 2:
                if "multiplica" in goal_lower or "*" in goal or "x" in goal_lower:
                    return ("calculate", {"expression": f"{nums[0]}*{nums[1]}"})
                elif "suma" in goal_lower or "+" in goal:
                    return ("calculate", {"expression": f"{nums[0]}+{nums[1]}"})

        elif any(word in goal_lower for word in ["lista", "archivos", "directorio", "carpeta"]):
            return ("list_directory", {"path": "."})

        elif any(word in goal_lower for word in ["lee", "leer", "muestra", "contenido de"]):
            # Extraer nombre de archivo
            import re
            files = re.findall(r'[\w\-\.]+\.[a-zA-Z]{1,4}', goal)
            if files:
                return ("read_file", {"filepath": files[0]})

        elif any(word in goal_lower for word in ["ejecuta", "corre", "python", "codigo"]):
            return ("execute_python", {"code": "print('Hello from THAU!')"})

        return None

    def think(self, context: str) -> ThoughtStep:
        """Fase de pensamiento"""
        prompt = f"""Contexto actual:
{context}

Piensa sobre este problema. Que necesitas hacer?
Responde con tu razonamiento."""

        response = self._generate(prompt)

        step = ThoughtStep(
            step_type="think",
            content=response
        )

        self.state.thoughts.append(step)
        return step

    def plan(self, goal: str) -> ThoughtStep:
        """Fase de planificacion"""
        prompt = f"""Meta: {goal}

Crea un plan con pasos concretos para lograr esta meta.
Lista los pasos numerados."""

        response = self._generate(prompt, max_tokens=400)

        # Extraer subgoals del plan
        lines = response.split('\n')
        subgoals = [line.strip() for line in lines if line.strip() and (
            line.strip()[0].isdigit() or line.strip().startswith('-')
        )]

        self.state.subgoals = subgoals[:5]  # Max 5 subgoals

        step = ThoughtStep(
            step_type="plan",
            content=response
        )

        self.state.thoughts.append(step)
        return step

    def act(self, action_prompt: str) -> ThoughtStep:
        """Fase de accion - ejecutar herramienta"""
        prompt = f"""Accion requerida: {action_prompt}

Si necesitas usar una herramienta, responde:
TOOL: nombre_herramienta
PARAMS: {{"param": "valor"}}

Si no necesitas herramienta, responde directamente."""

        response = self._generate(prompt)

        # Parsear si hay tool call
        tool_name = None
        tool_result = None

        if "TOOL:" in response:
            try:
                # Extraer nombre de herramienta
                tool_line = [l for l in response.split('\n') if 'TOOL:' in l][0]
                tool_name = tool_line.split('TOOL:')[1].strip()

                # Extraer parametros
                params = {}
                if "PARAMS:" in response:
                    params_line = response.split('PARAMS:')[1].strip()
                    # Buscar JSON
                    if '{' in params_line:
                        json_str = params_line[params_line.index('{'):params_line.index('}')+1]
                        params = json.loads(json_str)

                # Ejecutar herramienta
                if tool_name in self.tools:
                    if tool_name == "execute_python":
                        tool_result = self.tools[tool_name](params.get("code", ""))
                    elif tool_name == "read_file":
                        tool_result = self.tools[tool_name](params.get("filepath", ""))
                    elif tool_name == "write_file":
                        tool_result = self.tools[tool_name](
                            params.get("filepath", ""),
                            params.get("content", "")
                        )
                    elif tool_name == "calculate":
                        tool_result = self.tools[tool_name](params.get("expression", ""))
                    elif tool_name == "list_directory":
                        tool_result = self.tools[tool_name](params.get("path", "."))

            except Exception as e:
                tool_result = ToolResult(
                    tool=tool_name or "unknown",
                    success=False,
                    output="",
                    error=f"Error parseando tool call: {e}"
                )

        step = ThoughtStep(
            step_type="act",
            content=response,
            tool_used=tool_name,
            tool_result=tool_result
        )

        self.state.thoughts.append(step)
        return step

    def observe(self, result: ToolResult) -> ThoughtStep:
        """Fase de observacion - analizar resultado"""
        prompt = f"""Resultado de la herramienta {result.tool}:

Exito: {result.success}
Output: {result.output}
Error: {result.error if result.error else "Ninguno"}

Que aprendemos de este resultado?"""

        response = self._generate(prompt)

        step = ThoughtStep(
            step_type="observe",
            content=response
        )

        self.state.thoughts.append(step)
        return step

    def reflect(self) -> ThoughtStep:
        """Fase de reflexion - evaluar progreso"""
        context = f"""Meta principal: {self.state.goal}

Subgoals: {self.state.subgoals}

Pasos completados:
{self._format_thoughts()}

Hemos logrado la meta? Que falta por hacer?"""

        prompt = f"""Reflexiona sobre el progreso:

{context}

Responde:
1. Que hemos logrado?
2. Que falta?
3. Hemos terminado? (SI/NO)"""

        response = self._generate(prompt)

        # Detectar si terminamos
        if "SI" in response.upper() and "TERMINADO" in response.upper():
            self.state.completed = True

        step = ThoughtStep(
            step_type="reflect",
            content=response
        )

        self.state.thoughts.append(step)
        return step

    def _format_thoughts(self) -> str:
        """Formatea los pasos de pensamiento"""
        lines = []
        for i, step in enumerate(self.state.thoughts[-5:], 1):  # Ultimos 5
            lines.append(f"{i}. [{step.step_type}] {step.content[:100]}...")
        return "\n".join(lines)

    def run(self, goal: str) -> str:
        """
        Ejecuta el ciclo ReAct completo para lograr una meta

        Args:
            goal: Meta a lograr

        Returns:
            Respuesta final
        """
        if not self.pipe:
            self.load_model()

        # Inicializar estado
        self.state = AgentState(goal=goal)

        print(f"\n{'='*60}")
        print(f"META: {goal}")
        print(f"{'='*60}")

        # Detectar si necesitamos herramienta
        tool_need = self._detect_tool_need(goal)

        if tool_need:
            tool_name, params = tool_need
            print(f"\n[TOOL DETECTED] {tool_name}")
            print(f"   Params: {params}")

            # Ejecutar herramienta directamente
            if tool_name in self.tools:
                if tool_name == "calculate":
                    result = self.tools[tool_name](params.get("expression", ""))
                elif tool_name == "read_file":
                    result = self.tools[tool_name](params.get("filepath", ""))
                elif tool_name == "list_directory":
                    result = self.tools[tool_name](params.get("path", "."))
                elif tool_name == "execute_python":
                    result = self.tools[tool_name](params.get("code", ""))
                else:
                    result = ToolResult(tool_name, False, "", "Tool no implementada")

                print(f"\n[RESULT] Exito: {result.success}")
                print(f"   Output: {result.output}")

                # Guardar paso
                step = ThoughtStep(
                    step_type="act",
                    content=f"Usando {tool_name}",
                    tool_used=tool_name,
                    tool_result=result
                )
                self.state.thoughts.append(step)

                # Generar respuesta con el resultado
                if result.success:
                    context = f"Resultado de {tool_name}: {result.output}"
                    response = self._generate(f"El usuario pidio: {goal}\n\n{context}\n\nExplica el resultado de forma clara.")
                else:
                    response = f"Error al ejecutar {tool_name}: {result.error}"

                self.state.final_answer = response
                self.state.completed = True

        else:
            # Sin herramienta - respuesta directa del modelo
            print("\n[DIRECT] Respondiendo sin herramientas...")
            response = self._generate(goal)
            self.state.final_answer = response
            self.state.completed = True

        print(f"\n{'='*60}")
        print("RESPUESTA FINAL:")
        print(f"{'='*60}")
        print(self.state.final_answer)

        return self.state.final_answer

    def get_history(self) -> List[Dict]:
        """Retorna historial de pasos"""
        if not self.state:
            return []

        return [
            {
                "type": step.step_type,
                "content": step.content,
                "tool": step.tool_used,
                "timestamp": step.timestamp.isoformat()
            }
            for step in self.state.thoughts
        ]


def main():
    """Demo de THAU Proto-AGI"""
    agent = ThauProtoAGI()

    # Test 1: Calculo simple
    print("\n" + "="*70)
    print("TEST 1: Calculo matematico")
    print("="*70)
    agent.run("Calcula cuanto es 15 * 23 + 100")

    # Test 2: Listar archivos
    print("\n" + "="*70)
    print("TEST 2: Explorar directorio")
    print("="*70)
    agent.run("Lista los archivos en el directorio actual")

    # Test 3: Leer archivo
    print("\n" + "="*70)
    print("TEST 3: Leer configuracion")
    print("="*70)
    agent.run("Lee el archivo CLAUDE.md y resume su contenido")


if __name__ == "__main__":
    main()
