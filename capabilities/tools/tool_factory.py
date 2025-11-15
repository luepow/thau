#!/usr/bin/env python3
"""
THAU Tool Factory - Sistema de Auto-Creación de Herramientas

¡THAU puede crear sus propias herramientas!

Esto permite a THAU:
1. Detectar que necesita una herramienta que no existe
2. Diseñar la herramienta
3. Generarla automáticamente
4. Registrarla y usarla
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import inspect
from datetime import datetime
import importlib.util
import sys


@dataclass
class ToolSpec:
    """Especificación de una herramienta"""
    name: str
    description: str
    category: str  # "api", "data", "file", "web", "custom"
    parameters: Dict[str, Any]
    return_type: str
    dependencies: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


@dataclass
class GeneratedTool:
    """Herramienta generada"""
    spec: ToolSpec
    code: str
    filepath: Path
    created_at: datetime = field(default_factory=datetime.now)
    function: Optional[Callable] = None


class ToolFactory:
    """
    Fábrica de Herramientas de THAU

    Permite a THAU crear nuevas herramientas on-the-fly
    """

    def __init__(self, tools_dir: str = "capabilities/tools/generated"):
        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(parents=True, exist_ok=True)

        self.generated_tools: Dict[str, GeneratedTool] = {}
        self.tool_templates = self._load_templates()

        print("🏭 THAU Tool Factory inicializado")
        print(f"   Directorio: {self.tools_dir}")
        print(f"   Templates: {len(self.tool_templates)}")

    def _load_templates(self) -> Dict[str, str]:
        """Carga templates de herramientas comunes"""
        return {
            "api_client": '''
def {function_name}({parameters}):
    """
    {description}

    Args:
        {args_doc}

    Returns:
        {return_doc}
    """
    import requests

    # Headers
    headers = {{
        "Content-Type": "application/json",
    }}

    # Add authentication if needed
    if api_key:
        headers["Authorization"] = f"Bearer {{api_key}}"

    # Make request
    response = requests.{method}(
        url,
        json=data,
        headers=headers,
        timeout=timeout
    )

    # Handle response
    response.raise_for_status()
    return response.json()
''',

            "data_processor": '''
def {function_name}({parameters}):
    """
    {description}

    Args:
        {args_doc}

    Returns:
        {return_doc}
    """
    import pandas as pd

    # Load data
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data

    # Process
    {processing_code}

    return df
''',

            "file_handler": '''
def {function_name}({parameters}):
    """
    {description}

    Args:
        {args_doc}

    Returns:
        {return_doc}
    """
    from pathlib import Path
    import json

    filepath = Path(filepath)

    # Read
    if mode == "read":
        with open(filepath, 'r') as f:
            {read_code}

    # Write
    elif mode == "write":
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            {write_code}

    return result
''',

            "web_scraper": '''
def {function_name}({parameters}):
    """
    {description}

    Args:
        {args_doc}

    Returns:
        {return_doc}
    """
    import requests
    from bs4 import BeautifulSoup

    # Fetch page
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    # Parse
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract data
    {extraction_code}

    return data
''',

            "calendar_integration": '''
def {function_name}({parameters}):
    """
    {description}

    Args:
        {args_doc}

    Returns:
        {return_doc}
    """
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials

    # Authenticate
    creds = Credentials.from_authorized_user_file(credentials_path)
    service = build('calendar', 'v3', credentials=creds)

    # Create event
    event = {{
        'summary': summary,
        'description': description,
        'start': {{'dateTime': start_time}},
        'end': {{'dateTime': end_time}},
    }}

    # Insert
    result = service.events().insert(calendarId='primary', body=event).execute()

    return result
'''
        }

    def create_tool(
        self,
        spec: ToolSpec,
        template_name: Optional[str] = None,
        custom_code: Optional[str] = None
    ) -> GeneratedTool:
        """
        Crea una nueva herramienta

        Args:
            spec: Especificación de la herramienta
            template_name: Template a usar (None = custom)
            custom_code: Código custom si no usa template

        Returns:
            Herramienta generada
        """
        print(f"\n{'='*70}")
        print(f"🔨 Creando herramienta: {spec.name}")
        print(f"{'='*70}")
        print(f"Categoría: {spec.category}")
        print(f"Template: {template_name or 'custom'}")

        # Genera código
        if template_name and template_name in self.tool_templates:
            code = self._generate_from_template(spec, template_name)
        elif custom_code:
            code = custom_code
        else:
            code = self._generate_basic_tool(spec)

        # Guarda a archivo
        filename = f"{spec.name}.py"
        filepath = self.tools_dir / filename

        with open(filepath, 'w') as f:
            f.write(code)

        print(f"💾 Guardado: {filepath}")

        # Carga función
        function = self._load_function(filepath, spec.name)

        # Crea GeneratedTool
        tool = GeneratedTool(
            spec=spec,
            code=code,
            filepath=filepath,
            function=function
        )

        self.generated_tools[spec.name] = tool

        print(f"✅ Herramienta creada y lista para usar")

        return tool

    def _generate_from_template(self, spec: ToolSpec, template_name: str) -> str:
        """Genera código desde template"""
        template = self.tool_templates[template_name]

        # Prepara parámetros
        params_str = ", ".join([
            f"{name}: {ptype}"
            for name, ptype in spec.parameters.items()
        ])

        # Prepara documentación de args
        args_doc = "\n        ".join([
            f"{name} ({ptype}): Description"
            for name, ptype in spec.parameters.items()
        ])

        # Rellena template
        code = template.format(
            function_name=spec.name,
            parameters=params_str,
            description=spec.description,
            args_doc=args_doc,
            return_doc=f"{spec.return_type}: Result",
            method="post",  # Default
            processing_code="# Processing logic here\n    pass",
            read_code="content = f.read()\n        result = json.loads(content)",
            write_code="json.dump(data, f, indent=2)\n        result = filepath",
            extraction_code="data = []\n    # Extract logic here",
        )

        # Añade imports y header
        header = f'''#!/usr/bin/env python3
"""
{spec.name} - Generated by THAU Tool Factory

{spec.description}

Generated: {datetime.now().isoformat()}
Category: {spec.category}
"""

'''

        full_code = header + code

        return full_code

    def _generate_basic_tool(self, spec: ToolSpec) -> str:
        """Genera herramienta básica sin template"""
        params_str = ", ".join([
            f"{name}: {ptype}"
            for name, ptype in spec.parameters.items()
        ])

        args_doc = "\n        ".join([
            f"{name} ({ptype}): Description"
            for name, ptype in spec.parameters.items()
        ])

        code = f'''#!/usr/bin/env python3
"""
{spec.name} - Generated by THAU Tool Factory

{spec.description}

Generated: {datetime.now().isoformat()}
Category: {spec.category}
"""


def {spec.name}({params_str}) -> {spec.return_type}:
    """
    {spec.description}

    Args:
        {args_doc}

    Returns:
        {spec.return_type}: Result
    """
    # TODO: Implement logic
    pass


if __name__ == "__main__":
    # Test
    print(f"Testing {spec.name}...")
'''

        return code

    def _load_function(self, filepath: Path, function_name: str) -> Callable:
        """Carga función desde archivo generado"""
        spec = importlib.util.spec_from_file_location("generated_tool", filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules["generated_tool"] = module
        spec.loader.exec_module(module)

        function = getattr(module, function_name)
        return function

    def create_api_client(
        self,
        name: str,
        api_url: str,
        description: str,
        method: str = "POST",
        requires_auth: bool = True
    ) -> GeneratedTool:
        """
        Crea un cliente API

        Ejemplo de creación rápida de herramienta específica

        Args:
            name: Nombre de la herramienta
            api_url: URL del API
            description: Descripción
            method: HTTP method
            requires_auth: Requiere autenticación

        Returns:
            Herramienta generada
        """
        spec = ToolSpec(
            name=name,
            description=description,
            category="api",
            parameters={
                "data": "Dict[str, Any]",
                "api_key": "Optional[str]",
                "timeout": "int"
            },
            return_type="Dict[str, Any]",
            dependencies=["requests"]
        )

        return self.create_tool(spec, template_name="api_client")

    def create_from_description(self, description: str) -> GeneratedTool:
        """
        Crea herramienta desde descripción en lenguaje natural

        THAU usará esto para crear herramientas on-demand

        Args:
            description: Descripción de qué debe hacer la herramienta

        Returns:
            Herramienta generada
        """
        print(f"\n{'='*70}")
        print(f"🤖 THAU está analizando qué herramienta crear")
        print(f"{'='*70}")
        print(f"Descripción: {description}")

        # Analiza descripción (en producción, THAU-2B haría esto)
        desc_lower = description.lower()

        # Detecta tipo
        if "api" in desc_lower or "rest" in desc_lower or "http" in desc_lower:
            category = "api"
            template = "api_client"
        elif "calendar" in desc_lower or "evento" in desc_lower:
            category = "calendar"
            template = "calendar_integration"
        elif "scrap" in desc_lower or "web" in desc_lower or "página" in desc_lower:
            category = "web"
            template = "web_scraper"
        elif "data" in desc_lower or "csv" in desc_lower or "excel" in desc_lower:
            category = "data"
            template = "data_processor"
        else:
            category = "custom"
            template = None

        print(f"   Categoría detectada: {category}")
        print(f"   Template: {template or 'custom'}")

        # Genera nombre
        name = self._generate_name_from_description(description)

        # Crea spec
        spec = ToolSpec(
            name=name,
            description=description,
            category=category,
            parameters=self._infer_parameters(description, category),
            return_type="Any"
        )

        # Crea tool
        return self.create_tool(spec, template_name=template)

    def _generate_name_from_description(self, description: str) -> str:
        """Genera nombre de función desde descripción"""
        # Simplificado - en producción THAU-2B haría esto mejor
        words = description.lower().split()

        # Toma primeras 3-4 palabras relevantes
        relevant_words = [w for w in words if len(w) > 3][:3]

        name = "_".join(relevant_words)
        name = "".join(c if c.isalnum() or c == "_" else "" for c in name)

        return name or "generated_tool"

    def _infer_parameters(self, description: str, category: str) -> Dict[str, str]:
        """Infiere parámetros necesarios desde descripción"""
        params = {}

        desc_lower = description.lower()

        # Parámetros comunes por categoría
        if category == "api":
            params["url"] = "str"
            params["data"] = "Dict[str, Any]"
            if "token" in desc_lower or "auth" in desc_lower:
                params["api_key"] = "Optional[str]"

        elif category == "calendar":
            params["summary"] = "str"
            params["start_time"] = "str"
            params["end_time"] = "str"
            params["credentials_path"] = "str"

        elif category == "web":
            params["url"] = "str"
            params["timeout"] = "int"

        elif category == "data":
            params["data"] = "Any"

        else:
            # Generic
            params["input"] = "Any"

        return params

    def list_tools(self) -> List[Dict[str, Any]]:
        """Lista herramientas generadas"""
        return [
            {
                "name": tool.spec.name,
                "description": tool.spec.description,
                "category": tool.spec.category,
                "created_at": tool.created_at.isoformat(),
                "parameters": list(tool.spec.parameters.keys())
            }
            for tool in self.generated_tools.values()
        ]

    def save_manifest(self, filepath: str = "data/tools/manifest.json"):
        """Guarda manifest de herramientas generadas"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        manifest = {
            "generated_at": datetime.now().isoformat(),
            "total_tools": len(self.generated_tools),
            "tools": self.list_tools()
        }

        with open(filepath, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"💾 Manifest guardado: {filepath}")


if __name__ == "__main__":
    print("="*70)
    print("🧪 Testing THAU Tool Factory")
    print("="*70)

    factory = ToolFactory()

    # Test 1: Crear API client usando template
    print("\n" + "="*70)
    print("Test 1: API Client Tool")
    print("="*70)

    tool1 = factory.create_api_client(
        name="google_calendar_api",
        api_url="https://www.googleapis.com/calendar/v3/calendars/primary/events",
        description="Crea eventos en Google Calendar",
        requires_auth=True
    )

    print(f"\n📄 Código generado:")
    print("─"*70)
    print(tool1.code[:500] + "...")

    # Test 2: Crear herramienta desde descripción
    print("\n" + "="*70)
    print("Test 2: Tool desde Descripción")
    print("="*70)

    tool2 = factory.create_from_description(
        "Herramienta para hacer web scraping de noticias y extraer titulares"
    )

    # Test 3: Crear data processor
    print("\n" + "="*70)
    print("Test 3: Data Processor")
    print("="*70)

    spec3 = ToolSpec(
        name="csv_analyzer",
        description="Analiza archivos CSV y genera estadísticas",
        category="data",
        parameters={
            "filepath": "str",
            "columns": "List[str]"
        },
        return_type="Dict[str, Any]"
    )

    tool3 = factory.create_tool(spec3, template_name="data_processor")

    # Test 4: Listar todas las herramientas
    print("\n" + "="*70)
    print("📋 Herramientas Generadas")
    print("="*70)

    for tool_info in factory.list_tools():
        print(f"\n{tool_info['name']}")
        print(f"  Category: {tool_info['category']}")
        print(f"  Description: {tool_info['description']}")
        print(f"  Parameters: {', '.join(tool_info['parameters'])}")
        print(f"  Created: {tool_info['created_at']}")

    # Test 5: Guardar manifest
    print("\n" + "="*70)
    print("💾 Guardar Manifest")
    print("="*70)

    factory.save_manifest()

    print("\n" + "="*70)
    print("✅ Tests Completados")
    print("="*70)
