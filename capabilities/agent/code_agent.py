"""THAU Code Agent - Autonomous coding assistant.

This module implements the agent loop that allows THAU to:
1. Receive a task from the user
2. Plan the implementation
3. Execute tools to read/write files, run commands
4. Iterate until the task is complete
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
from pathlib import Path

from capabilities.tools.system_tools import SystemTools, ToolResult


@dataclass
class AgentMessage:
    """A message in the agent conversation."""
    role: str  # "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_results: Optional[List[ToolResult]] = None


@dataclass
class AgentState:
    """Current state of the agent."""
    messages: List[AgentMessage] = field(default_factory=list)
    task: str = ""
    plan: List[str] = field(default_factory=list)
    current_step: int = 0
    completed: bool = False
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)


class CodeAgent:
    """Autonomous code agent powered by THAU.

    This agent can:
    - Plan multi-step coding tasks
    - Create and modify files
    - Run shell commands
    - Search and analyze code
    - Iterate until task completion
    """

    SYSTEM_PROMPT = """You are THAU, an autonomous code agent. You can use tools to:
- read_file: Read file contents
- write_file: Create or overwrite files
- edit_file: Modify existing files
- bash: Execute shell commands
- glob: Find files by pattern
- grep: Search file contents
- list_directory: List directory contents
- create_directory: Create directories

When given a task:
1. First, understand and plan the approach
2. Use tools to explore the codebase if needed
3. Implement step by step, using tools
4. Verify your work
5. Report completion

To use a tool, output:
<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1"}}
</tool_call>

After receiving tool results, continue working or report completion.

Be methodical. Create proper project structure. Write clean, documented code.
"""

    def __init__(
        self,
        model_adapter=None,
        working_dir: str = ".",
        max_iterations: int = 20,
    ):
        """Initialize the code agent.

        Args:
            model_adapter: THAU model adapter for generation
            working_dir: Working directory for file operations
            max_iterations: Maximum tool-use iterations
        """
        self.model_adapter = model_adapter
        self.tools = SystemTools(working_dir=working_dir)
        self.max_iterations = max_iterations
        self.state = AgentState()

        logger.info(f"CodeAgent initialized with working_dir: {working_dir}")

    def parse_tool_calls(self, response: str) -> List[Dict]:
        """Extract tool calls from model response.

        Args:
            response: Model response text

        Returns:
            List of tool call dictionaries
        """
        tool_calls = []

        # Pattern: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            try:
                tool_call = json.loads(match)
                if "name" in tool_call:
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call: {match}")

        return tool_calls

    def execute_tool_call(self, tool_call: Dict) -> ToolResult:
        """Execute a single tool call.

        Args:
            tool_call: Tool call dictionary with name and arguments

        Returns:
            ToolResult from execution
        """
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        logger.info(f"Executing tool: {name}")

        result = self.tools.execute_tool(name, **args)

        # Track file operations
        if name == "write_file" and result.success:
            self.state.files_created.append(args.get("file_path", ""))
        elif name == "edit_file" and result.success:
            self.state.files_modified.append(args.get("file_path", ""))

        return result

    def format_tool_results(self, results: List[Tuple[Dict, ToolResult]]) -> str:
        """Format tool results for the model.

        Args:
            results: List of (tool_call, result) tuples

        Returns:
            Formatted string
        """
        formatted = []

        for tool_call, result in results:
            name = tool_call.get("name", "unknown")
            if result.success:
                formatted.append(f"<tool_result name=\"{name}\">\n{result.output}\n</tool_result>")
            else:
                formatted.append(f"<tool_result name=\"{name}\" error=\"true\">\n{result.error}\n</tool_result>")

        return "\n\n".join(formatted)

    def generate_response(self, messages: List[Dict]) -> str:
        """Generate a response using THAU model.

        Args:
            messages: Conversation messages

        Returns:
            Generated response
        """
        if self.model_adapter is None:
            # For testing without model
            return "I would process this task but no model is loaded."

        # Format prompt
        prompt = self._format_prompt(messages)

        # Generate
        from core.inference.generator import TextGenerator

        generator = TextGenerator(self.model_adapter)
        response = generator.generate(
            prompt,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
        )

        return response

    def _format_prompt(self, messages: List[Dict]) -> str:
        """Format messages into a prompt.

        Args:
            messages: List of message dicts

        Returns:
            Formatted prompt string
        """
        prompt_parts = [self.SYSTEM_PROMPT, ""]

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "tool":
                prompt_parts.append(f"Tool Results:\n{content}")

        prompt_parts.append("Assistant:")

        return "\n\n".join(prompt_parts)

    def run(self, task: str) -> Dict[str, Any]:
        """Run the agent on a task.

        Args:
            task: Task description from user

        Returns:
            Result dictionary with status and outputs
        """
        self.state = AgentState(task=task)
        self.state.messages.append(AgentMessage(role="user", content=task))

        logger.info(f"Starting agent task: {task[:100]}...")

        iteration = 0
        while iteration < self.max_iterations and not self.state.completed:
            iteration += 1
            logger.info(f"Agent iteration {iteration}/{self.max_iterations}")

            # Generate response
            messages = [
                {"role": m.role, "content": m.content}
                for m in self.state.messages
            ]

            response = self.generate_response(messages)

            # Parse tool calls
            tool_calls = self.parse_tool_calls(response)

            # Add assistant message
            self.state.messages.append(AgentMessage(
                role="assistant",
                content=response,
                tool_calls=tool_calls if tool_calls else None
            ))

            if not tool_calls:
                # No tool calls - agent is done or needs input
                if self._check_completion(response):
                    self.state.completed = True
                break

            # Execute tools
            results = []
            for tool_call in tool_calls:
                result = self.execute_tool_call(tool_call)
                results.append((tool_call, result))

            # Add tool results to conversation
            tool_results_str = self.format_tool_results(results)
            self.state.messages.append(AgentMessage(
                role="tool",
                content=tool_results_str
            ))

        return {
            "completed": self.state.completed,
            "iterations": iteration,
            "files_created": self.state.files_created,
            "files_modified": self.state.files_modified,
            "final_response": self.state.messages[-1].content if self.state.messages else "",
            "conversation": [
                {"role": m.role, "content": m.content[:500]}
                for m in self.state.messages
            ]
        }

    def _check_completion(self, response: str) -> bool:
        """Check if the agent has completed the task.

        Args:
            response: Agent's response

        Returns:
            True if task is complete
        """
        completion_phrases = [
            "task complete",
            "task is complete",
            "i have completed",
            "the task has been completed",
            "finished implementing",
            "implementation is complete",
            "done!",
            "all done",
        ]

        response_lower = response.lower()
        return any(phrase in response_lower for phrase in completion_phrases)

    def plan_task(self, task: str) -> List[str]:
        """Generate a plan for a task.

        Args:
            task: Task description

        Returns:
            List of steps
        """
        planning_prompt = f"""Analyze this task and create a step-by-step plan:

Task: {task}

Create a numbered list of steps to complete this task. Be specific and actionable.
Consider:
1. What files need to be created?
2. What dependencies are needed?
3. What's the logical order of implementation?
4. How will you verify it works?

Plan:"""

        if self.model_adapter:
            response = self.generate_response([
                {"role": "user", "content": planning_prompt}
            ])
        else:
            response = "1. Analyze requirements\n2. Create structure\n3. Implement\n4. Test"

        # Parse numbered steps
        steps = []
        for line in response.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.?\s+', line):
                step = re.sub(r'^\d+\.?\s+', '', line)
                steps.append(step)

        self.state.plan = steps
        return steps


# Agent prompt templates for training
AGENT_TRAINING_EXAMPLES = [
    {
        "task": "Crea una aplicación web simple con Flask que tenga un endpoint /hello que retorne 'Hello World'",
        "plan": [
            "Crear la estructura del proyecto",
            "Crear archivo requirements.txt con Flask",
            "Crear app.py con el endpoint",
            "Probar la aplicación"
        ],
        "conversation": [
            {"role": "user", "content": "Crea una aplicación web simple con Flask que tenga un endpoint /hello que retorne 'Hello World'"},
            {"role": "assistant", "content": """Voy a crear una aplicación Flask simple. Primero, déjame crear la estructura del proyecto.

<tool_call>
{"name": "create_directory", "arguments": {"dir_path": "flask_app"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"create_directory\">\nCreated directory: flask_app\n</tool_result>"},
            {"role": "assistant", "content": """Ahora creo el archivo requirements.txt:

<tool_call>
{"name": "write_file", "arguments": {"file_path": "flask_app/requirements.txt", "content": "flask>=2.0.0\n"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"write_file\">\nSuccessfully wrote 14 characters to flask_app/requirements.txt\n</tool_result>"},
            {"role": "assistant", "content": """Ahora creo el archivo principal de la aplicación:

<tool_call>
{"name": "write_file", "arguments": {"file_path": "flask_app/app.py", "content": "from flask import Flask\n\napp = Flask(__name__)\n\n@app.route('/hello')\ndef hello():\n    return 'Hello World'\n\nif __name__ == '__main__':\n    app.run(debug=True)\n"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"write_file\">\nSuccessfully wrote 156 characters to flask_app/app.py\n</tool_result>"},
            {"role": "assistant", "content": """¡Tarea completada! He creado una aplicación Flask con:

1. `flask_app/requirements.txt` - Dependencias
2. `flask_app/app.py` - Aplicación con endpoint /hello

Para ejecutar:
```bash
cd flask_app
pip install -r requirements.txt
python app.py
```

Luego visita http://localhost:5000/hello"""}
        ]
    },
    {
        "task": "Encuentra todos los archivos Python que contienen la palabra 'import torch' y muéstrame cuáles son",
        "conversation": [
            {"role": "user", "content": "Encuentra todos los archivos Python que contienen la palabra 'import torch' y muéstrame cuáles son"},
            {"role": "assistant", "content": """Voy a buscar en los archivos Python:

<tool_call>
{"name": "grep", "arguments": {"pattern": "import torch", "path": ".", "file_pattern": "*.py"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"grep\">\ncore/models/base_transformer.py:3: import torch\ncore/training/trainer.py:1: import torch\nadapters/device_manager.py:5: import torch\ntests/test_models.py:2: import torch\n</tool_result>"},
            {"role": "assistant", "content": """Encontré 4 archivos Python que importan torch:

1. `core/models/base_transformer.py` - Línea 3
2. `core/training/trainer.py` - Línea 1
3. `adapters/device_manager.py` - Línea 5
4. `tests/test_models.py` - Línea 2

Estos son los archivos principales que usan PyTorch en el proyecto."""}
        ]
    },
    {
        "task": "Crea una clase Python para manejar configuración con validación",
        "conversation": [
            {"role": "user", "content": "Crea una clase Python para manejar configuración con validación"},
            {"role": "assistant", "content": """Voy a crear una clase de configuración robusta con validación usando dataclasses y typing:

<tool_call>
{"name": "write_file", "arguments": {"file_path": "config_manager.py", "content": "\"\"\"Configuration management with validation.\"\"\"\n\nfrom dataclasses import dataclass, field\nfrom typing import Optional, List, Dict, Any\nfrom pathlib import Path\nimport json\nimport os\n\n\n@dataclass\nclass DatabaseConfig:\n    \"\"\"Database configuration.\"\"\"\n    host: str = \"localhost\"\n    port: int = 5432\n    name: str = \"mydb\"\n    user: str = \"postgres\"\n    password: str = \"\"\n    \n    def __post_init__(self):\n        if self.port < 1 or self.port > 65535:\n            raise ValueError(f\"Invalid port: {self.port}\")\n        if not self.name:\n            raise ValueError(\"Database name is required\")\n\n\n@dataclass \nclass ServerConfig:\n    \"\"\"Server configuration.\"\"\"\n    host: str = \"0.0.0.0\"\n    port: int = 8000\n    debug: bool = False\n    workers: int = 4\n    \n    def __post_init__(self):\n        if self.workers < 1:\n            raise ValueError(\"Workers must be at least 1\")\n\n\n@dataclass\nclass AppConfig:\n    \"\"\"Main application configuration.\"\"\"\n    app_name: str = \"MyApp\"\n    version: str = \"1.0.0\"\n    environment: str = \"development\"\n    database: DatabaseConfig = field(default_factory=DatabaseConfig)\n    server: ServerConfig = field(default_factory=ServerConfig)\n    \n    def __post_init__(self):\n        valid_envs = [\"development\", \"staging\", \"production\"]\n        if self.environment not in valid_envs:\n            raise ValueError(f\"Invalid environment: {self.environment}\")\n    \n    @classmethod\n    def from_json(cls, path: str) -> 'AppConfig':\n        \"\"\"Load config from JSON file.\"\"\"\n        with open(path, 'r') as f:\n            data = json.load(f)\n        \n        db_config = DatabaseConfig(**data.get('database', {}))\n        server_config = ServerConfig(**data.get('server', {}))\n        \n        return cls(\n            app_name=data.get('app_name', 'MyApp'),\n            version=data.get('version', '1.0.0'),\n            environment=data.get('environment', 'development'),\n            database=db_config,\n            server=server_config\n        )\n    \n    @classmethod\n    def from_env(cls) -> 'AppConfig':\n        \"\"\"Load config from environment variables.\"\"\"\n        return cls(\n            app_name=os.getenv('APP_NAME', 'MyApp'),\n            version=os.getenv('APP_VERSION', '1.0.0'),\n            environment=os.getenv('ENVIRONMENT', 'development'),\n            database=DatabaseConfig(\n                host=os.getenv('DB_HOST', 'localhost'),\n                port=int(os.getenv('DB_PORT', '5432')),\n                name=os.getenv('DB_NAME', 'mydb'),\n                user=os.getenv('DB_USER', 'postgres'),\n                password=os.getenv('DB_PASSWORD', '')\n            ),\n            server=ServerConfig(\n                host=os.getenv('SERVER_HOST', '0.0.0.0'),\n                port=int(os.getenv('SERVER_PORT', '8000')),\n                debug=os.getenv('DEBUG', 'false').lower() == 'true',\n                workers=int(os.getenv('WORKERS', '4'))\n            )\n        )\n    \n    def to_dict(self) -> Dict[str, Any]:\n        \"\"\"Convert config to dictionary.\"\"\"\n        return {\n            'app_name': self.app_name,\n            'version': self.version,\n            'environment': self.environment,\n            'database': {\n                'host': self.database.host,\n                'port': self.database.port,\n                'name': self.database.name,\n                'user': self.database.user\n            },\n            'server': {\n                'host': self.server.host,\n                'port': self.server.port,\n                'debug': self.server.debug,\n                'workers': self.server.workers\n            }\n        }\n\n\nif __name__ == '__main__':\n    # Test the config\n    config = AppConfig()\n    print(f\"App: {config.app_name} v{config.version}\")\n    print(f\"Environment: {config.environment}\")\n    print(f\"Database: {config.database.host}:{config.database.port}\")\n    print(f\"Server: {config.server.host}:{config.server.port}\")\n"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"write_file\">\nSuccessfully wrote 3245 characters to config_manager.py\n</tool_result>"},
            {"role": "assistant", "content": """¡Tarea completada! He creado `config_manager.py` con:

**Características:**
1. **Dataclasses** para configuración tipada
2. **Validación automática** en `__post_init__`
3. **Carga desde JSON** con `from_json()`
4. **Carga desde variables de entorno** con `from_env()`
5. **Serialización** con `to_dict()`

**Clases:**
- `DatabaseConfig` - Config de base de datos
- `ServerConfig` - Config del servidor
- `AppConfig` - Config principal que agrupa todo

**Uso:**
```python
# Desde código
config = AppConfig(environment="production")

# Desde JSON
config = AppConfig.from_json("config.json")

# Desde variables de entorno
config = AppConfig.from_env()
```"""}
        ]
    },
    # Bug Fixing Example
    {
        "task": "Revisa el archivo calculator.py y corrige el bug en la función divide",
        "conversation": [
            {"role": "user", "content": "Revisa el archivo calculator.py y corrige el bug en la función divide"},
            {"role": "assistant", "content": """Voy a leer el archivo para analizar el código:

<tool_call>
{"name": "read_file", "arguments": {"file_path": "calculator.py"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"read_file\">\ndef add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n\ndef multiply(a, b):\n    return a * b\n\ndef divide(a, b):\n    return a / b\n</tool_result>"},
            {"role": "assistant", "content": """Encontré el bug: la función `divide` no maneja la división por cero. Voy a corregirlo:

<tool_call>
{"name": "edit_file", "arguments": {"file_path": "calculator.py", "old_content": "def divide(a, b):\n    return a / b", "new_content": "def divide(a, b):\n    if b == 0:\n        raise ValueError(\"No se puede dividir por cero\")\n    return a / b"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"edit_file\">\nSuccessfully edited calculator.py\n</tool_result>"},
            {"role": "assistant", "content": """¡Bug corregido! El problema era que la función `divide` no verificaba si el divisor era cero.

**Cambio realizado:**
- Agregué validación para verificar si `b == 0`
- Si es cero, lanza `ValueError` con mensaje descriptivo
- Si no, realiza la división normalmente

Esto previene el error `ZeroDivisionError` y da un mensaje más claro al usuario."""}
        ]
    },
    # Code Review Example
    {
        "task": "Haz una revisión del código en auth_service.py y sugiere mejoras",
        "conversation": [
            {"role": "user", "content": "Haz una revisión del código en auth_service.py y sugiere mejoras"},
            {"role": "assistant", "content": """Voy a leer el archivo para hacer la revisión:

<tool_call>
{"name": "read_file", "arguments": {"file_path": "auth_service.py"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"read_file\">\nimport hashlib\n\ndef login(username, password):\n    users = {\"admin\": \"admin123\", \"user\": \"password\"}\n    if username in users and users[username] == password:\n        return True\n    return False\n\ndef hash_password(password):\n    return hashlib.md5(password.encode()).hexdigest()\n</tool_result>"},
            {"role": "assistant", "content": """**Revisión de Código - auth_service.py**

🔴 **Problemas Críticos:**
1. **Contraseñas en texto plano** - Las contraseñas están hardcodeadas
2. **MD5 es inseguro** - MD5 está roto para propósitos criptográficos
3. **No se usa el hash** - `hash_password()` existe pero no se usa en `login()`

🟡 **Mejoras Recomendadas:**
1. Usar bcrypt o argon2 para hash de contraseñas
2. Almacenar usuarios en base de datos
3. Agregar manejo de errores
4. Implementar rate limiting
5. Agregar logging de intentos fallidos

¿Quieres que implemente estas mejoras?"""}
        ]
    },
    # Project Scaffolding Example
    {
        "task": "Crea la estructura de un proyecto FastAPI con autenticación JWT",
        "plan": [
            "Crear estructura de directorios",
            "Crear archivos de configuración",
            "Implementar modelos",
            "Crear rutas de autenticación",
            "Agregar requirements.txt"
        ],
        "conversation": [
            {"role": "user", "content": "Crea la estructura de un proyecto FastAPI con autenticación JWT"},
            {"role": "assistant", "content": """Voy a crear la estructura del proyecto FastAPI con JWT. Primero creo los directorios:

<tool_call>
{"name": "bash", "arguments": {"command": "mkdir -p fastapi_jwt/{app/{routers,models,schemas,utils},tests}"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"bash\">\n</tool_result>"},
            {"role": "assistant", "content": """Ahora creo el archivo principal:

<tool_call>
{"name": "write_file", "arguments": {"file_path": "fastapi_jwt/app/main.py", "content": "from fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware\nfrom app.routers import auth, users\n\napp = FastAPI(title=\"FastAPI JWT Auth\", version=\"1.0.0\")\n\napp.add_middleware(\n    CORSMiddleware,\n    allow_origins=[\"*\"],\n    allow_methods=[\"*\"],\n    allow_headers=[\"*\"],\n)\n\napp.include_router(auth.router, prefix=\"/auth\", tags=[\"auth\"])\napp.include_router(users.router, prefix=\"/users\", tags=[\"users\"])\n\n@app.get(\"/health\")\ndef health_check():\n    return {\"status\": \"healthy\"}\n"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"write_file\">\nSuccessfully wrote 512 characters to fastapi_jwt/app/main.py\n</tool_result>"},
            {"role": "assistant", "content": """Creo el módulo de autenticación JWT:

<tool_call>
{"name": "write_file", "arguments": {"file_path": "fastapi_jwt/app/utils/jwt_handler.py", "content": "from datetime import datetime, timedelta\nfrom typing import Optional\nimport jwt\nfrom pydantic import BaseModel\n\nSECRET_KEY = \"your-secret-key-here\"\nALGORITHM = \"HS256\"\nACCESS_TOKEN_EXPIRE_MINUTES = 30\n\nclass TokenData(BaseModel):\n    username: Optional[str] = None\n\ndef create_access_token(data: dict, expires_delta: Optional[timedelta] = None):\n    to_encode = data.copy()\n    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))\n    to_encode.update({\"exp\": expire})\n    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)\n\ndef verify_token(token: str):\n    try:\n        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])\n        return payload\n    except jwt.PyJWTError:\n        return None\n"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"write_file\">\nSuccessfully wrote 756 characters to fastapi_jwt/app/utils/jwt_handler.py\n</tool_result>"},
            {"role": "assistant", "content": """¡Proyecto creado! Estructura:

```
fastapi_jwt/
├── app/
│   ├── main.py           # Aplicación principal
│   ├── routers/          # Endpoints
│   ├── models/           # Modelos de BD
│   ├── schemas/          # Pydantic schemas
│   └── utils/
│       └── jwt_handler.py # Manejo de JWT
├── tests/                # Tests
└── requirements.txt      # Dependencias
```

**Siguientes pasos:**
1. Configurar base de datos
2. Implementar rutas de login/register
3. Agregar middleware de autenticación"""}
        ]
    },
    # Refactoring Example
    {
        "task": "Refactoriza el código de data_processor.py para mejorar su legibilidad",
        "conversation": [
            {"role": "user", "content": "Refactoriza el código de data_processor.py para mejorar su legibilidad"},
            {"role": "assistant", "content": """Voy a analizar el código actual:

<tool_call>
{"name": "read_file", "arguments": {"file_path": "data_processor.py"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"read_file\">\ndef p(d):\n    r = []\n    for i in d:\n        if i > 0:\n            r.append(i * 2)\n    return r\n\ndef f(d):\n    t = 0\n    for i in d:\n        t = t + i\n    return t / len(d) if len(d) > 0 else 0\n</tool_result>"},
            {"role": "assistant", "content": """El código tiene problemas de legibilidad: nombres cortos y sin documentación. Voy a refactorizarlo:

<tool_call>
{"name": "write_file", "arguments": {"file_path": "data_processor.py", "content": "\"\"\"Data processing utilities.\"\"\"\nfrom typing import List\n\n\ndef process_positive_numbers(data: List[float]) -> List[float]:\n    \"\"\"Double all positive numbers in the dataset.\n    \n    Args:\n        data: List of numbers to process\n        \n    Returns:\n        List with positive numbers doubled\n    \"\"\"\n    return [num * 2 for num in data if num > 0]\n\n\ndef calculate_average(data: List[float]) -> float:\n    \"\"\"Calculate the average of a list of numbers.\n    \n    Args:\n        data: List of numbers\n        \n    Returns:\n        The arithmetic mean, or 0 if list is empty\n    \"\"\"\n    if not data:\n        return 0.0\n    return sum(data) / len(data)\n"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"write_file\">\nSuccessfully wrote 612 characters to data_processor.py\n</tool_result>"},
            {"role": "assistant", "content": """¡Refactorización completada!

**Mejoras realizadas:**
1. **Nombres descriptivos**: `p` → `process_positive_numbers`, `f` → `calculate_average`
2. **Type hints**: Agregué tipado para parámetros y retornos
3. **Docstrings**: Documentación clara de cada función
4. **List comprehension**: Más pythónico y eficiente
5. **Built-in sum()**: En lugar de loop manual
6. **Early return**: Manejo más claro de lista vacía

El código ahora es más legible, mantenible y sigue las convenciones de Python."""}
        ]
    },
    # Testing Example
    {
        "task": "Crea tests unitarios para el módulo calculator.py usando pytest",
        "conversation": [
            {"role": "user", "content": "Crea tests unitarios para el módulo calculator.py usando pytest"},
            {"role": "assistant", "content": """Primero veo qué funciones tiene el módulo:

<tool_call>
{"name": "read_file", "arguments": {"file_path": "calculator.py"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"read_file\">\ndef add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n\ndef multiply(a, b):\n    return a * b\n\ndef divide(a, b):\n    if b == 0:\n        raise ValueError(\"No se puede dividir por cero\")\n    return a / b\n</tool_result>"},
            {"role": "assistant", "content": """Ahora creo los tests:

<tool_call>
{"name": "write_file", "arguments": {"file_path": "test_calculator.py", "content": "\"\"\"Tests for calculator module.\"\"\"\nimport pytest\nfrom calculator import add, subtract, multiply, divide\n\n\nclass TestAdd:\n    def test_positive_numbers(self):\n        assert add(2, 3) == 5\n    \n    def test_negative_numbers(self):\n        assert add(-1, -1) == -2\n    \n    def test_mixed_numbers(self):\n        assert add(-1, 5) == 4\n    \n    def test_zero(self):\n        assert add(0, 0) == 0\n\n\nclass TestSubtract:\n    def test_positive_numbers(self):\n        assert subtract(5, 3) == 2\n    \n    def test_negative_result(self):\n        assert subtract(3, 5) == -2\n\n\nclass TestMultiply:\n    def test_positive_numbers(self):\n        assert multiply(3, 4) == 12\n    \n    def test_by_zero(self):\n        assert multiply(5, 0) == 0\n    \n    def test_negative_numbers(self):\n        assert multiply(-2, 3) == -6\n\n\nclass TestDivide:\n    def test_exact_division(self):\n        assert divide(10, 2) == 5\n    \n    def test_float_result(self):\n        assert divide(7, 2) == 3.5\n    \n    def test_divide_by_zero(self):\n        with pytest.raises(ValueError, match=\"No se puede dividir por cero\"):\n            divide(5, 0)\n"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"write_file\">\nSuccessfully wrote 1024 characters to test_calculator.py\n</tool_result>"},
            {"role": "assistant", "content": """Verifico que los tests funcionen:

<tool_call>
{"name": "bash", "arguments": {"command": "python -m pytest test_calculator.py -v"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"bash\">\n===== test session starts =====\ncollected 10 items\ntest_calculator.py::TestAdd::test_positive_numbers PASSED\ntest_calculator.py::TestAdd::test_negative_numbers PASSED\ntest_calculator.py::TestAdd::test_mixed_numbers PASSED\ntest_calculator.py::TestAdd::test_zero PASSED\ntest_calculator.py::TestSubtract::test_positive_numbers PASSED\ntest_calculator.py::TestSubtract::test_negative_result PASSED\ntest_calculator.py::TestMultiply::test_positive_numbers PASSED\ntest_calculator.py::TestMultiply::test_by_zero PASSED\ntest_calculator.py::TestMultiply::test_negative_numbers PASSED\ntest_calculator.py::TestDivide::test_exact_division PASSED\ntest_calculator.py::TestDivide::test_float_result PASSED\ntest_calculator.py::TestDivide::test_divide_by_zero PASSED\n===== 12 passed =====\n</tool_result>"},
            {"role": "assistant", "content": """¡Tests creados y ejecutados! Todos pasaron.

**Cobertura de tests:**
- `add`: 4 tests (positivos, negativos, mixtos, cero)
- `subtract`: 2 tests (resultado positivo y negativo)
- `multiply`: 3 tests (positivos, por cero, negativos)
- `divide`: 3 tests (exacta, decimal, división por cero)

**Para ejecutar:**
```bash
pytest test_calculator.py -v
pytest test_calculator.py --cov=calculator  # Con cobertura
```"""}
        ]
    },
    # API Creation Example
    {
        "task": "Crea un endpoint REST para CRUD de productos",
        "conversation": [
            {"role": "user", "content": "Crea un endpoint REST para CRUD de productos"},
            {"role": "assistant", "content": """Voy a crear un CRUD completo de productos con FastAPI:

<tool_call>
{"name": "write_file", "arguments": {"file_path": "products_api.py", "content": "\"\"\"Products CRUD API.\"\"\"\nfrom fastapi import FastAPI, HTTPException, status\nfrom pydantic import BaseModel\nfrom typing import List, Optional\nfrom uuid import uuid4\n\napp = FastAPI(title=\"Products API\")\n\n# In-memory database\nproducts_db = {}\n\n\nclass ProductCreate(BaseModel):\n    name: str\n    description: Optional[str] = None\n    price: float\n    stock: int = 0\n\n\nclass Product(ProductCreate):\n    id: str\n\n\n@app.post(\"/products\", response_model=Product, status_code=status.HTTP_201_CREATED)\ndef create_product(product: ProductCreate):\n    \"\"\"Create a new product.\"\"\"\n    product_id = str(uuid4())\n    db_product = Product(id=product_id, **product.dict())\n    products_db[product_id] = db_product\n    return db_product\n\n\n@app.get(\"/products\", response_model=List[Product])\ndef list_products(skip: int = 0, limit: int = 100):\n    \"\"\"List all products with pagination.\"\"\"\n    return list(products_db.values())[skip:skip + limit]\n\n\n@app.get(\"/products/{product_id}\", response_model=Product)\ndef get_product(product_id: str):\n    \"\"\"Get a product by ID.\"\"\"\n    if product_id not in products_db:\n        raise HTTPException(status_code=404, detail=\"Product not found\")\n    return products_db[product_id]\n\n\n@app.put(\"/products/{product_id}\", response_model=Product)\ndef update_product(product_id: str, product: ProductCreate):\n    \"\"\"Update an existing product.\"\"\"\n    if product_id not in products_db:\n        raise HTTPException(status_code=404, detail=\"Product not found\")\n    db_product = Product(id=product_id, **product.dict())\n    products_db[product_id] = db_product\n    return db_product\n\n\n@app.delete(\"/products/{product_id}\", status_code=status.HTTP_204_NO_CONTENT)\ndef delete_product(product_id: str):\n    \"\"\"Delete a product.\"\"\"\n    if product_id not in products_db:\n        raise HTTPException(status_code=404, detail=\"Product not found\")\n    del products_db[product_id]\n    return None\n"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"write_file\">\nSuccessfully wrote 1756 characters to products_api.py\n</tool_result>"},
            {"role": "assistant", "content": """¡API CRUD creada! Incluye:

**Endpoints:**
| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | /products | Crear producto |
| GET | /products | Listar productos |
| GET | /products/{id} | Obtener por ID |
| PUT | /products/{id} | Actualizar |
| DELETE | /products/{id} | Eliminar |

**Características:**
- Validación con Pydantic
- Códigos HTTP correctos (201, 204, 404)
- Paginación en listado
- Documentación automática en /docs

**Ejecutar:**
```bash
uvicorn products_api:app --reload
```"""}
        ]
    },
    # Planning Large Project Example
    {
        "task": "Planifica y crea la estructura para un sistema de gestión de inventario",
        "plan": [
            "Analizar requerimientos",
            "Diseñar arquitectura",
            "Crear estructura de proyecto",
            "Implementar modelos de datos",
            "Crear servicios de negocio",
            "Implementar API REST"
        ],
        "conversation": [
            {"role": "user", "content": "Planifica y crea la estructura para un sistema de gestión de inventario"},
            {"role": "assistant", "content": """Voy a planificar y crear la estructura del sistema de inventario.

**Plan de Desarrollo:**

1. **Arquitectura**: Clean Architecture con capas separadas
2. **Tecnologías**: FastAPI + SQLAlchemy + PostgreSQL
3. **Módulos principales**:
   - Productos (CRUD, categorías)
   - Inventario (stock, movimientos)
   - Proveedores (gestión, órdenes)
   - Reportes (alertas, estadísticas)

Empiezo creando la estructura:

<tool_call>
{"name": "bash", "arguments": {"command": "mkdir -p inventory_system/{app/{api/v1,core,models,schemas,services,repositories},tests,migrations}"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"bash\">\n</tool_result>"},
            {"role": "assistant", "content": """Creo el archivo de configuración principal:

<tool_call>
{"name": "write_file", "arguments": {"file_path": "inventory_system/app/core/config.py", "content": "\"\"\"Application configuration.\"\"\"\nfrom pydantic_settings import BaseSettings\nfrom functools import lru_cache\n\n\nclass Settings(BaseSettings):\n    # App\n    APP_NAME: str = \"Inventory System\"\n    VERSION: str = \"1.0.0\"\n    DEBUG: bool = False\n    \n    # Database\n    DATABASE_URL: str = \"postgresql://user:pass@localhost/inventory\"\n    \n    # Security\n    SECRET_KEY: str = \"your-secret-key\"\n    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30\n    \n    # Inventory\n    LOW_STOCK_THRESHOLD: int = 10\n    \n    class Config:\n        env_file = \".env\"\n\n\n@lru_cache()\ndef get_settings() -> Settings:\n    return Settings()\n"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"write_file\">\nSuccessfully wrote 612 characters to inventory_system/app/core/config.py\n</tool_result>"},
            {"role": "assistant", "content": """Creo el modelo de Producto:

<tool_call>
{"name": "write_file", "arguments": {"file_path": "inventory_system/app/models/product.py", "content": "\"\"\"Product model.\"\"\"\nfrom sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime\nfrom sqlalchemy.orm import relationship\nfrom datetime import datetime\nfrom app.core.database import Base\n\n\nclass Category(Base):\n    __tablename__ = \"categories\"\n    \n    id = Column(Integer, primary_key=True, index=True)\n    name = Column(String(100), unique=True, nullable=False)\n    description = Column(String(500))\n    \n    products = relationship(\"Product\", back_populates=\"category\")\n\n\nclass Product(Base):\n    __tablename__ = \"products\"\n    \n    id = Column(Integer, primary_key=True, index=True)\n    sku = Column(String(50), unique=True, nullable=False, index=True)\n    name = Column(String(200), nullable=False)\n    description = Column(String(1000))\n    price = Column(Float, nullable=False)\n    cost = Column(Float)\n    \n    category_id = Column(Integer, ForeignKey(\"categories.id\"))\n    category = relationship(\"Category\", back_populates=\"products\")\n    \n    # Stock info\n    stock_quantity = Column(Integer, default=0)\n    min_stock = Column(Integer, default=10)\n    \n    created_at = Column(DateTime, default=datetime.utcnow)\n    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)\n    \n    @property\n    def is_low_stock(self) -> bool:\n        return self.stock_quantity <= self.min_stock\n"}}
</tool_call>"""},
            {"role": "tool", "content": "<tool_result name=\"write_file\">\nSuccessfully wrote 1245 characters to inventory_system/app/models/product.py\n</tool_result>"},
            {"role": "assistant", "content": """¡Estructura base creada!

```
inventory_system/
├── app/
│   ├── api/v1/          # Endpoints REST
│   ├── core/
│   │   └── config.py    # Configuración
│   ├── models/
│   │   └── product.py   # Modelos SQLAlchemy
│   ├── schemas/         # Pydantic schemas
│   ├── services/        # Lógica de negocio
│   └── repositories/    # Acceso a datos
├── tests/               # Tests
└── migrations/          # Alembic migrations
```

**Próximos pasos:**
1. Implementar repositorios (ProductRepository)
2. Crear servicios (InventoryService)
3. Definir endpoints de la API
4. Configurar migraciones con Alembic
5. Agregar autenticación

¿Continúo con la implementación de algún módulo específico?"""}
        ]
    }
]


def generate_agent_training_dataset(output_path: str = "./data/datasets/agent_training.jsonl"):
    """Generate training dataset for code agent.

    Args:
        output_path: Path to save dataset
    """
    import json

    with open(output_path, 'w', encoding='utf-8') as f:
        for example in AGENT_TRAINING_EXAMPLES:
            # Format as instruction/output for training
            task = example.get("task", "")
            conversation = example.get("conversation", [])

            # Create full conversation as output
            output_parts = []
            for msg in conversation:
                if msg["role"] == "assistant":
                    output_parts.append(msg["content"])
                elif msg["role"] == "tool":
                    output_parts.append(f"[Tool Result]\n{msg['content']}")

            training_example = {
                "instruction": task,
                "input": "",
                "output": "\n\n".join(output_parts),
                "category": "code_agent"
            }

            f.write(json.dumps(training_example, ensure_ascii=False) + '\n')

    logger.info(f"Generated agent training dataset: {output_path}")


if __name__ == "__main__":
    # Test the code agent
    print("Testing CodeAgent...")

    agent = CodeAgent(working_dir=".")

    # Test tool parsing
    test_response = """Let me read the file:

<tool_call>
{"name": "read_file", "arguments": {"file_path": "README.md"}}
</tool_call>

I'll analyze it next."""

    tool_calls = agent.parse_tool_calls(test_response)
    print(f"Parsed tool calls: {tool_calls}")

    # Test tool execution
    if tool_calls:
        result = agent.execute_tool_call(tool_calls[0])
        print(f"Tool result success: {result.success}")
        print(f"Tool output: {result.output[:200]}...")

    # Generate training dataset
    generate_agent_training_dataset()
    print("\nGenerated agent training dataset!")
