#!/usr/bin/env python3
"""
Genera datos de entrenamiento para Tool Calling en THAU
Crea ejemplos que ensenan al modelo a usar el formato <tool_call>
"""

import json
from datetime import datetime
from pathlib import Path
import random

# Directorio de salida
OUTPUT_DIR = Path("./data/self_questioning")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Herramientas disponibles que THAU debe aprender a usar
TOOLS = {
    "get_current_time": {
        "description": "Obtiene la fecha y hora actual",
        "parameters": [],
        "examples": [
            {"question": "que hora es?", "call": {"name": "get_current_time", "arguments": {}}},
            {"question": "cual es la fecha de hoy?", "call": {"name": "get_current_time", "arguments": {}}},
            {"question": "dime la hora actual", "call": {"name": "get_current_time", "arguments": {}}},
            {"question": "que dia es hoy?", "call": {"name": "get_current_time", "arguments": {}}},
            {"question": "necesito saber la hora", "call": {"name": "get_current_time", "arguments": {}}},
        ]
    },
    "web_search": {
        "description": "Busca informacion en internet",
        "parameters": ["query", "num_results"],
        "examples": [
            {"question": "busca informacion sobre Python asyncio", "call": {"name": "web_search", "arguments": {"query": "Python asyncio tutorial"}}},
            {"question": "encuentra articulos sobre machine learning", "call": {"name": "web_search", "arguments": {"query": "machine learning articulos"}}},
            {"question": "busca como usar Docker", "call": {"name": "web_search", "arguments": {"query": "Docker tutorial basico"}}},
            {"question": "investiga sobre React hooks", "call": {"name": "web_search", "arguments": {"query": "React hooks guia"}}},
            {"question": "busca documentacion de FastAPI", "call": {"name": "web_search", "arguments": {"query": "FastAPI documentacion oficial"}}},
        ]
    },
    "execute_python": {
        "description": "Ejecuta codigo Python",
        "parameters": ["code"],
        "examples": [
            {"question": "calcula 15 * 23", "call": {"name": "execute_python", "arguments": {"code": "result = 15 * 23\nprint(result)"}}},
            {"question": "genera una lista del 1 al 10", "call": {"name": "execute_python", "arguments": {"code": "result = list(range(1, 11))\nprint(result)"}}},
            {"question": "calcula el factorial de 5", "call": {"name": "execute_python", "arguments": {"code": "import math\nresult = math.factorial(5)\nprint(result)"}}},
            {"question": "invierte la cadena 'hola mundo'", "call": {"name": "execute_python", "arguments": {"code": "result = 'hola mundo'[::-1]\nprint(result)"}}},
            {"question": "suma los numeros del 1 al 100", "call": {"name": "execute_python", "arguments": {"code": "result = sum(range(1, 101))\nprint(result)"}}},
        ]
    },
    "read_file": {
        "description": "Lee el contenido de un archivo",
        "parameters": ["filepath"],
        "examples": [
            {"question": "lee el archivo config.py", "call": {"name": "read_file", "arguments": {"filepath": "config.py"}}},
            {"question": "muestra el contenido de README.md", "call": {"name": "read_file", "arguments": {"filepath": "README.md"}}},
            {"question": "abre el archivo requirements.txt", "call": {"name": "read_file", "arguments": {"filepath": "requirements.txt"}}},
            {"question": "lee main.py", "call": {"name": "read_file", "arguments": {"filepath": "main.py"}}},
            {"question": "muestra el archivo .env", "call": {"name": "read_file", "arguments": {"filepath": ".env"}}},
        ]
    },
    "list_directory": {
        "description": "Lista archivos en un directorio",
        "parameters": ["path"],
        "examples": [
            {"question": "lista los archivos del directorio actual", "call": {"name": "list_directory", "arguments": {"path": "."}}},
            {"question": "que hay en la carpeta src?", "call": {"name": "list_directory", "arguments": {"path": "src"}}},
            {"question": "muestra los archivos en data/", "call": {"name": "list_directory", "arguments": {"path": "data"}}},
            {"question": "lista el contenido de scripts", "call": {"name": "list_directory", "arguments": {"path": "scripts"}}},
            {"question": "que archivos hay en el proyecto?", "call": {"name": "list_directory", "arguments": {"path": "."}}},
        ]
    },
    "call_rest_api": {
        "description": "Hace peticiones a APIs REST",
        "parameters": ["url", "method", "headers", "body"],
        "examples": [
            {"question": "haz un GET a https://api.example.com/users", "call": {"name": "call_rest_api", "arguments": {"url": "https://api.example.com/users", "method": "GET"}}},
            {"question": "consulta la API de clima", "call": {"name": "call_rest_api", "arguments": {"url": "https://api.weather.com/current", "method": "GET"}}},
            {"question": "envia datos a la API", "call": {"name": "call_rest_api", "arguments": {"url": "https://api.example.com/data", "method": "POST", "body": {"data": "value"}}}},
        ]
    }
}

# Plantillas de respuesta con tool call
def create_tool_call_response(tool_name: str, arguments: dict) -> str:
    """Crea respuesta con formato tool_call"""
    call_json = json.dumps({"name": tool_name, "arguments": arguments}, ensure_ascii=False)
    return f"<tool_call>{call_json}</tool_call>"

# Plantillas conversacionales
CONVERSATION_TEMPLATES = [
    # Formato simple - solo tool call
    "{tool_call}",
    # Con explicacion previa
    "Para hacer eso, necesito usar una herramienta.\n\n{tool_call}",
    # Con contexto
    "Voy a {action} usando la herramienta disponible.\n\n{tool_call}",
    # Profesional
    "Ejecutare la herramienta apropiada para tu solicitud.\n\n{tool_call}",
]

# Acciones por herramienta
TOOL_ACTIONS = {
    "get_current_time": "consultar la hora",
    "web_search": "buscar en internet",
    "execute_python": "ejecutar codigo",
    "read_file": "leer el archivo",
    "list_directory": "listar el directorio",
    "call_rest_api": "hacer la peticion",
}

def generate_training_examples():
    """Genera ejemplos de entrenamiento para tool calling"""
    examples = []
    timestamp = datetime.now().isoformat()

    for tool_name, tool_info in TOOLS.items():
        for example in tool_info["examples"]:
            # Crear tool call
            tool_call = create_tool_call_response(
                example["call"]["name"],
                example["call"]["arguments"]
            )

            # Seleccionar plantilla
            template = random.choice(CONVERSATION_TEMPLATES)
            action = TOOL_ACTIONS.get(tool_name, "ejecutar la accion")

            answer = template.format(tool_call=tool_call, action=action)

            # Crear entrada de entrenamiento
            entry = {
                "timestamp": timestamp,
                "question": example["question"],
                "answer": answer,
                "confidence": 0.95,
                "cognitive_age": 12,
                "category": "tool_calling",
                "self_generated": True,
                "intensive_session": True,
                "tool_training": True
            }
            examples.append(entry)

    return examples

def generate_multi_turn_examples():
    """Genera ejemplos de conversacion multi-turno con tool results"""
    examples = []
    timestamp = datetime.now().isoformat()

    multi_turn = [
        # Hora con respuesta
        {
            "question": "que hora es ahora mismo?",
            "tool_call": {"name": "get_current_time", "arguments": {}},
            "tool_result": {"date": "2025-11-27", "time": "14:30:00", "day_of_week": "Thursday"},
            "final_answer": "Son las 14:30 del jueves 27 de noviembre de 2025."
        },
        # Busqueda con respuesta
        {
            "question": "busca informacion sobre Python decorators",
            "tool_call": {"name": "web_search", "arguments": {"query": "Python decorators tutorial"}},
            "tool_result": {"success": True, "results": [{"snippet": "Los decoradores en Python son funciones que modifican el comportamiento de otras funciones..."}]},
            "final_answer": "Encontre informacion sobre decorators en Python. Los decoradores son funciones que modifican el comportamiento de otras funciones, permitiendo agregar funcionalidad de manera elegante."
        },
        # Calculo con respuesta
        {
            "question": "cuanto es 2 elevado a la 10?",
            "tool_call": {"name": "execute_python", "arguments": {"code": "result = 2 ** 10\nprint(result)"}},
            "tool_result": {"success": True, "output": "1024"},
            "final_answer": "2 elevado a la 10 es 1024."
        },
        # Lista directorio
        {
            "question": "muestra los archivos Python en el directorio actual",
            "tool_call": {"name": "list_directory", "arguments": {"path": "."}},
            "tool_result": {"success": True, "files": ["main.py", "config.py", "utils.py"], "directories": ["src", "tests"]},
            "final_answer": "En el directorio actual hay 3 archivos Python: main.py, config.py y utils.py. Tambien hay 2 carpetas: src y tests."
        },
    ]

    for item in multi_turn:
        # Formato completo con tool call y respuesta
        tool_call_str = create_tool_call_response(
            item["tool_call"]["name"],
            item["tool_call"]["arguments"]
        )

        # Primera parte: genera tool call
        entry1 = {
            "timestamp": timestamp,
            "question": item["question"],
            "answer": f"Voy a usar una herramienta para responder.\n\n{tool_call_str}",
            "confidence": 0.95,
            "cognitive_age": 12,
            "category": "tool_calling",
            "self_generated": True,
            "intensive_session": True,
            "tool_training": True
        }
        examples.append(entry1)

        # Segunda parte: respuesta basada en resultado
        result_prompt = f"Resultado de la herramienta: {json.dumps(item['tool_result'], ensure_ascii=False)}\n\nPregunta original: {item['question']}"
        entry2 = {
            "timestamp": timestamp,
            "question": result_prompt,
            "answer": item["final_answer"],
            "confidence": 0.95,
            "cognitive_age": 12,
            "category": "tool_calling_response",
            "self_generated": True,
            "intensive_session": True,
            "tool_training": True
        }
        examples.append(entry2)

    return examples

def generate_no_tool_examples():
    """Genera ejemplos donde NO se necesita herramienta"""
    examples = []
    timestamp = datetime.now().isoformat()

    no_tool_needed = [
        {"question": "que es Python?", "answer": "Python es un lenguaje de programacion de alto nivel, interpretado y de proposito general. Es conocido por su sintaxis clara y legible."},
        {"question": "como funciona un bucle for?", "answer": "Un bucle for en programacion permite iterar sobre una secuencia de elementos, ejecutando un bloque de codigo para cada elemento."},
        {"question": "que es una funcion?", "answer": "Una funcion es un bloque de codigo reutilizable que realiza una tarea especifica. Se define una vez y puede llamarse multiples veces."},
        {"question": "explica que es una variable", "answer": "Una variable es un contenedor para almacenar datos en memoria. Tiene un nombre y puede contener diferentes tipos de valores."},
        {"question": "que es un array?", "answer": "Un array o arreglo es una estructura de datos que almacena una coleccion de elementos del mismo tipo en posiciones contiguas de memoria."},
    ]

    for item in no_tool_needed:
        entry = {
            "timestamp": timestamp,
            "question": item["question"],
            "answer": item["answer"],
            "confidence": 0.9,
            "cognitive_age": 12,
            "category": "general_knowledge",
            "self_generated": True,
            "intensive_session": True,
            "tool_training": False
        }
        examples.append(entry)

    return examples

def save_training_data(examples, filename="tool_calling_training.jsonl"):
    """Guarda datos de entrenamiento en archivo JSONL"""
    output_path = OUTPUT_DIR / filename

    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"Guardados {len(examples)} ejemplos en {output_path}")
    return output_path

def main():
    print("=" * 60)
    print("GENERADOR DE DATOS DE ENTRENAMIENTO PARA TOOL CALLING")
    print("=" * 60)

    all_examples = []

    # Generar ejemplos basicos
    print("\n1. Generando ejemplos basicos de tool calling...")
    basic_examples = generate_training_examples()
    all_examples.extend(basic_examples)
    print(f"   Generados: {len(basic_examples)} ejemplos")

    # Generar ejemplos multi-turno
    print("\n2. Generando ejemplos multi-turno...")
    multi_examples = generate_multi_turn_examples()
    all_examples.extend(multi_examples)
    print(f"   Generados: {len(multi_examples)} ejemplos")

    # Generar ejemplos sin herramientas
    print("\n3. Generando ejemplos sin herramientas (balance)...")
    no_tool = generate_no_tool_examples()
    all_examples.extend(no_tool)
    print(f"   Generados: {len(no_tool)} ejemplos")

    # Duplicar y variar para mas datos
    print("\n4. Expandiendo dataset con variaciones...")
    expanded = []
    variations = [
        ("por favor ", ""),
        ("puedes ", ""),
        ("necesito que ", ""),
        ("podrias ", ""),
    ]

    for example in basic_examples:
        for prefix, suffix in variations:
            new_example = example.copy()
            new_example["question"] = prefix + example["question"] + suffix
            new_example["timestamp"] = datetime.now().isoformat()
            expanded.append(new_example)

    all_examples.extend(expanded[:50])  # Limitar expansiones
    print(f"   Agregados: {min(50, len(expanded))} variaciones")

    # Guardar
    print("\n5. Guardando datos...")
    random.shuffle(all_examples)
    output_path = save_training_data(all_examples)

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Total ejemplos generados: {len(all_examples)}")
    print(f"Archivo de salida: {output_path}")
    print("\nDistribucion por categoria:")

    categories = {}
    for ex in all_examples:
        cat = ex.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count}")

    print("\nPara entrenar THAU con estos datos:")
    print("  python scripts/train_phase1.py --epochs 3 --batch-size 4")
    print("\nPara probar el agente:")
    print("  python scripts/thau_agent.py --interactive")

if __name__ == "__main__":
    main()
