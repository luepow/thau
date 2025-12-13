"""Generate Python syntax Q&A pairs from extracted book content."""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger
from datetime import datetime

# Python syntax topics and templates for Q&A generation
PYTHON_TOPICS = {
    "variables": {
        "keywords": ["variable", "assignment", "=", "type", "int", "float", "str", "bool"],
        "questions": [
            "¿Cómo se declara una variable en Python?",
            "¿Cuáles son los tipos de datos básicos en Python?",
            "¿Cómo se asigna un valor a una variable?",
            "¿Python requiere declarar el tipo de variable?",
            "¿Qué es el tipado dinámico en Python?",
        ]
    },
    "strings": {
        "keywords": ["string", "cadena", "texto", "comillas", "f-string", "format", "concatenar"],
        "questions": [
            "¿Cómo se crea un string en Python?",
            "¿Qué son los f-strings en Python?",
            "¿Cómo se concatenan strings en Python?",
            "¿Cómo se accede a caracteres individuales de un string?",
            "¿Cuáles son los métodos más comunes de strings?",
        ]
    },
    "lists": {
        "keywords": ["list", "lista", "array", "append", "extend", "index", "slice"],
        "questions": [
            "¿Cómo se crea una lista en Python?",
            "¿Cómo se agregan elementos a una lista?",
            "¿Qué es el slicing en listas de Python?",
            "¿Cuál es la diferencia entre append y extend?",
            "¿Cómo se accede a elementos de una lista?",
        ]
    },
    "dictionaries": {
        "keywords": ["dict", "diccionario", "key", "value", "clave", "valor", "{}"],
        "questions": [
            "¿Cómo se crea un diccionario en Python?",
            "¿Cómo se accede a valores en un diccionario?",
            "¿Cómo se agregan pares clave-valor a un diccionario?",
            "¿Qué métodos tienen los diccionarios?",
            "¿Cómo se itera sobre un diccionario?",
        ]
    },
    "conditionals": {
        "keywords": ["if", "else", "elif", "condición", "condicional", "comparación"],
        "questions": [
            "¿Cuál es la sintaxis del if en Python?",
            "¿Cómo se usa elif en Python?",
            "¿Qué operadores de comparación existen en Python?",
            "¿Cómo se anidan condiciones en Python?",
            "¿Qué son los operadores lógicos and, or, not?",
        ]
    },
    "loops": {
        "keywords": ["for", "while", "loop", "bucle", "ciclo", "range", "iterate"],
        "questions": [
            "¿Cuál es la sintaxis del for en Python?",
            "¿Cuál es la sintaxis del while en Python?",
            "¿Cómo se usa range() en un for loop?",
            "¿Qué hacen break y continue en Python?",
            "¿Cómo se itera sobre una lista con enumerate?",
        ]
    },
    "functions": {
        "keywords": ["def", "function", "función", "return", "parámetro", "argumento", "lambda"],
        "questions": [
            "¿Cómo se define una función en Python?",
            "¿Cómo se devuelve un valor de una función?",
            "¿Qué son los parámetros por defecto?",
            "¿Qué son *args y **kwargs?",
            "¿Qué es una función lambda en Python?",
        ]
    },
    "classes": {
        "keywords": ["class", "clase", "object", "objeto", "__init__", "self", "método", "atributo"],
        "questions": [
            "¿Cómo se define una clase en Python?",
            "¿Qué es el método __init__?",
            "¿Para qué sirve self en Python?",
            "¿Cómo se crea un objeto de una clase?",
            "¿Qué es la herencia en Python?",
        ]
    },
    "exceptions": {
        "keywords": ["try", "except", "exception", "error", "raise", "finally"],
        "questions": [
            "¿Cómo se maneja una excepción en Python?",
            "¿Cuál es la sintaxis de try/except?",
            "¿Qué hace el bloque finally?",
            "¿Cómo se lanza una excepción con raise?",
            "¿Cuáles son las excepciones más comunes en Python?",
        ]
    },
    "imports": {
        "keywords": ["import", "from", "module", "módulo", "package", "paquete", "as"],
        "questions": [
            "¿Cómo se importa un módulo en Python?",
            "¿Cuál es la diferencia entre import y from...import?",
            "¿Cómo se usa el alias con as?",
            "¿Qué son los paquetes en Python?",
            "¿Cómo se instalan paquetes con pip?",
        ]
    },
    "files": {
        "keywords": ["open", "read", "write", "file", "archivo", "with", "close"],
        "questions": [
            "¿Cómo se abre un archivo en Python?",
            "¿Cómo se lee el contenido de un archivo?",
            "¿Cómo se escribe en un archivo?",
            "¿Para qué sirve el context manager with?",
            "¿Cuáles son los modos de apertura de archivos?",
        ]
    },
    "comprehensions": {
        "keywords": ["comprehension", "comprensión", "list comprehension", "dict comprehension"],
        "questions": [
            "¿Qué es una list comprehension en Python?",
            "¿Cómo se escribe una list comprehension?",
            "¿Qué es una dict comprehension?",
            "¿Cuándo usar comprehensions vs loops normales?",
            "¿Se pueden anidar comprehensions?",
        ]
    },
    "decorators": {
        "keywords": ["decorator", "decorador", "@", "wrapper", "functools"],
        "questions": [
            "¿Qué es un decorador en Python?",
            "¿Cómo se crea un decorador?",
            "¿Cómo se aplica un decorador a una función?",
            "¿Qué decoradores built-in existen?",
            "¿Para qué sirve @property?",
        ]
    },
    "generators": {
        "keywords": ["yield", "generator", "generador", "iter", "next"],
        "questions": [
            "¿Qué es un generador en Python?",
            "¿Qué hace yield en Python?",
            "¿Cuál es la diferencia entre return y yield?",
            "¿Cómo se crea una generator expression?",
            "¿Cuándo usar generadores vs listas?",
        ]
    },
}


class PythonQAGenerator:
    """Generate Q&A pairs from Python book content."""

    def __init__(self, output_dir: str = "./data/datasets"):
        """Initialize generator.

        Args:
            output_dir: Directory to save generated datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.qa_pairs: List[Dict] = []

    def load_text(self, text_path: str) -> str:
        """Load extracted text from file.

        Args:
            text_path: Path to text file

        Returns:
            Text content
        """
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_chapters(self, json_path: str) -> List[Dict]:
        """Load chapters from JSON file.

        Args:
            json_path: Path to chapters JSON

        Returns:
            List of chapter dictionaries
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def find_relevant_content(self, text: str, topic: str) -> List[str]:
        """Find paragraphs relevant to a topic.

        Args:
            text: Full text content
            topic: Topic name from PYTHON_TOPICS

        Returns:
            List of relevant paragraphs
        """
        if topic not in PYTHON_TOPICS:
            return []

        keywords = PYTHON_TOPICS[topic]["keywords"]
        paragraphs = text.split('\n\n')
        relevant = []

        for para in paragraphs:
            para_lower = para.lower()
            # Check if paragraph contains any keywords
            matches = sum(1 for kw in keywords if kw.lower() in para_lower)
            if matches >= 2 and len(para) > 100:
                relevant.append(para.strip())

        return relevant[:5]  # Limit to top 5

    def extract_code_examples(self, text: str) -> List[Tuple[str, str]]:
        """Extract code examples with their context.

        Args:
            text: Text content

        Returns:
            List of (context, code) tuples
        """
        examples = []

        # Pattern: text followed by code block
        pattern = r'([^\n]{20,200})\n\n((?:>>>.*?\n|    .*?\n)+)'
        matches = re.findall(pattern, text, re.MULTILINE)

        for context, code in matches:
            if 'python' in context.lower() or 'ejemplo' in context.lower():
                examples.append((context.strip(), code.strip()))

        return examples

    def generate_qa_from_content(self, content: str, topic: str) -> List[Dict]:
        """Generate Q&A pairs from content about a topic.

        Args:
            content: Relevant text content
            topic: Topic name

        Returns:
            List of Q&A dictionaries
        """
        qa_pairs = []
        questions = PYTHON_TOPICS.get(topic, {}).get("questions", [])

        # Simple heuristic: use content as base for answers
        for question in questions:
            # Find the most relevant paragraph for this question
            best_match = ""
            best_score = 0

            for para in content.split('\n\n'):
                if len(para) < 50:
                    continue

                # Score based on keyword overlap
                q_words = set(question.lower().split())
                p_words = set(para.lower().split())
                score = len(q_words & p_words)

                if score > best_score:
                    best_score = score
                    best_match = para

            if best_match and best_score >= 2:
                qa_pairs.append({
                    "instruction": question,
                    "input": "",
                    "output": self._format_answer(best_match, topic),
                    "topic": topic,
                })

        return qa_pairs

    def _format_answer(self, content: str, topic: str) -> str:
        """Format content as a proper answer.

        Args:
            content: Raw content
            topic: Topic for context

        Returns:
            Formatted answer
        """
        # Clean up the content
        answer = content.strip()

        # Add structure if it's too raw
        if len(answer) > 500:
            answer = answer[:500] + "..."

        # Ensure it starts properly
        if not answer[0].isupper():
            answer = answer.capitalize()

        return answer

    def generate_syntax_qa(self) -> List[Dict]:
        """Generate Q&A pairs about Python syntax (from templates).

        Returns:
            List of syntax Q&A pairs
        """
        syntax_qa = [
            # Variables
            {
                "instruction": "¿Cómo se declara una variable en Python?",
                "input": "",
                "output": "En Python, las variables se declaran simplemente asignando un valor con el operador =. No es necesario declarar el tipo:\n\n```python\nnombre = \"Juan\"  # String\nedad = 25        # Integer\nprecio = 19.99   # Float\nactivo = True    # Boolean\n```\n\nPython usa tipado dinámico, lo que significa que el tipo se infiere automáticamente del valor asignado."
            },
            {
                "instruction": "¿Cuáles son los tipos de datos básicos en Python?",
                "input": "",
                "output": "Los tipos de datos básicos en Python son:\n\n1. **int**: Números enteros (42, -17, 0)\n2. **float**: Números decimales (3.14, -0.001)\n3. **str**: Cadenas de texto (\"hola\", 'mundo')\n4. **bool**: Booleanos (True, False)\n5. **None**: Valor nulo (None)\n\nTipos compuestos:\n- **list**: Listas ordenadas [1, 2, 3]\n- **tuple**: Tuplas inmutables (1, 2, 3)\n- **dict**: Diccionarios {'clave': 'valor'}\n- **set**: Conjuntos {1, 2, 3}"
            },
            # Strings
            {
                "instruction": "¿Cómo se crean y manipulan strings en Python?",
                "input": "",
                "output": "Los strings en Python se crean con comillas simples o dobles:\n\n```python\n# Creación\ntexto1 = \"Hola mundo\"\ntexto2 = 'Python es genial'\nmultilinea = \"\"\"Texto\nen varias\nlíneas\"\"\"\n\n# Concatenación\nnombre = \"Juan\"\nsaludo = \"Hola, \" + nombre\n\n# f-strings (Python 3.6+)\nedad = 25\nmensaje = f\"Tengo {edad} años\"\n\n# Métodos comunes\ntexto.upper()      # MAYÚSCULAS\ntexto.lower()      # minúsculas\ntexto.strip()      # Quitar espacios\ntexto.split(\",\")   # Dividir por coma\ntexto.replace(\"a\", \"e\")  # Reemplazar\n```"
            },
            # Lists
            {
                "instruction": "¿Cómo funcionan las listas en Python?",
                "input": "",
                "output": "Las listas son colecciones ordenadas y mutables:\n\n```python\n# Crear lista\nnumeros = [1, 2, 3, 4, 5]\nmixta = [1, \"dos\", 3.0, True]\n\n# Acceder elementos\nprimer = numeros[0]   # 1\nultimo = numeros[-1]  # 5\n\n# Slicing\nnumeros[1:3]   # [2, 3]\nnumeros[:2]    # [1, 2]\nnumeros[2:]    # [3, 4, 5]\n\n# Modificar\nnumeros.append(6)      # Agregar al final\nnumeros.insert(0, 0)   # Insertar en posición\nnumeros.extend([7, 8]) # Agregar múltiples\nnumeros.remove(3)      # Eliminar valor\nnumeros.pop()          # Eliminar último\n\n# Otros métodos\nlen(numeros)           # Longitud\nnumeros.sort()         # Ordenar\nnumeros.reverse()      # Invertir\n3 in numeros           # Verificar existencia\n```"
            },
            # Dictionaries
            {
                "instruction": "¿Cómo funcionan los diccionarios en Python?",
                "input": "",
                "output": "Los diccionarios almacenan pares clave-valor:\n\n```python\n# Crear diccionario\npersona = {\n    \"nombre\": \"Ana\",\n    \"edad\": 30,\n    \"ciudad\": \"Madrid\"\n}\n\n# Acceder valores\nnombre = persona[\"nombre\"]       # \"Ana\"\nedad = persona.get(\"edad\")       # 30\ndefault = persona.get(\"pais\", \"España\")  # Con default\n\n# Modificar\npersona[\"edad\"] = 31             # Actualizar\npersona[\"email\"] = \"ana@mail.com\"  # Agregar\ndel persona[\"ciudad\"]            # Eliminar\n\n# Iterar\nfor clave in persona:\n    print(clave)\n\nfor clave, valor in persona.items():\n    print(f\"{clave}: {valor}\")\n\n# Métodos\npersona.keys()    # Todas las claves\npersona.values()  # Todos los valores\npersona.items()   # Pares (clave, valor)\n```"
            },
            # Conditionals
            {
                "instruction": "¿Cuál es la sintaxis de if/else en Python?",
                "input": "",
                "output": "La estructura condicional en Python usa if, elif, else:\n\n```python\n# if básico\nedad = 18\nif edad >= 18:\n    print(\"Mayor de edad\")\n\n# if/else\nif edad >= 18:\n    print(\"Mayor de edad\")\nelse:\n    print(\"Menor de edad\")\n\n# if/elif/else\nnota = 85\nif nota >= 90:\n    print(\"Excelente\")\nelif nota >= 80:\n    print(\"Muy bien\")\nelif nota >= 70:\n    print(\"Bien\")\nelse:\n    print(\"Necesita mejorar\")\n\n# Operadores de comparación\n# ==  igual\n# !=  diferente\n# <   menor que\n# >   mayor que\n# <=  menor o igual\n# >=  mayor o igual\n\n# Operadores lógicos\nif edad >= 18 and tiene_licencia:\n    print(\"Puede conducir\")\n\nif es_estudiante or es_jubilado:\n    print(\"Tiene descuento\")\n\nif not esta_bloqueado:\n    print(\"Acceso permitido\")\n```"
            },
            # Loops
            {
                "instruction": "¿Cómo funcionan los loops for y while en Python?",
                "input": "",
                "output": "Python tiene dos tipos de bucles: for y while.\n\n```python\n# For loop - iterar sobre secuencia\nfrutas = [\"manzana\", \"pera\", \"naranja\"]\nfor fruta in frutas:\n    print(fruta)\n\n# For con range\nfor i in range(5):        # 0, 1, 2, 3, 4\n    print(i)\n\nfor i in range(1, 6):     # 1, 2, 3, 4, 5\n    print(i)\n\nfor i in range(0, 10, 2): # 0, 2, 4, 6, 8\n    print(i)\n\n# For con enumerate\nfor i, fruta in enumerate(frutas):\n    print(f\"{i}: {fruta}\")\n\n# While loop\ncontador = 0\nwhile contador < 5:\n    print(contador)\n    contador += 1\n\n# Control de flujo\nfor i in range(10):\n    if i == 3:\n        continue  # Salta esta iteración\n    if i == 7:\n        break     # Sale del bucle\n    print(i)\n\n# Else en loops (se ejecuta si no hay break)\nfor i in range(5):\n    print(i)\nelse:\n    print(\"Bucle completado\")\n```"
            },
            # Functions
            {
                "instruction": "¿Cómo se definen funciones en Python?",
                "input": "",
                "output": "Las funciones se definen con la palabra clave def:\n\n```python\n# Función básica\ndef saludar():\n    print(\"¡Hola!\")\n\n# Con parámetros\ndef saludar(nombre):\n    print(f\"¡Hola, {nombre}!\")\n\n# Con return\ndef sumar(a, b):\n    return a + b\n\nresultado = sumar(3, 5)  # 8\n\n# Parámetros por defecto\ndef saludar(nombre, saludo=\"Hola\"):\n    return f\"{saludo}, {nombre}\"\n\nsaludar(\"Ana\")           # \"Hola, Ana\"\nsaludar(\"Ana\", \"Buenos días\")  # \"Buenos días, Ana\"\n\n# *args - argumentos variables\ndef sumar_todos(*numeros):\n    return sum(numeros)\n\nsumar_todos(1, 2, 3, 4)  # 10\n\n# **kwargs - argumentos con nombre\ndef crear_perfil(**datos):\n    return datos\n\ncrear_perfil(nombre=\"Ana\", edad=30)\n\n# Lambda (funciones anónimas)\ndoble = lambda x: x * 2\ndoble(5)  # 10\n\n# Type hints (Python 3.5+)\ndef sumar(a: int, b: int) -> int:\n    return a + b\n```"
            },
            # Classes
            {
                "instruction": "¿Cómo se definen clases en Python?",
                "input": "",
                "output": "Las clases se definen con la palabra clave class:\n\n```python\nclass Persona:\n    # Constructor\n    def __init__(self, nombre, edad):\n        self.nombre = nombre  # Atributo de instancia\n        self.edad = edad\n    \n    # Método de instancia\n    def saludar(self):\n        return f\"Hola, soy {self.nombre}\"\n    \n    # Método con parámetros\n    def cumplir_años(self, años=1):\n        self.edad += años\n\n# Crear objeto\njuan = Persona(\"Juan\", 25)\nprint(juan.nombre)      # \"Juan\"\nprint(juan.saludar())   # \"Hola, soy Juan\"\n\n# Herencia\nclass Estudiante(Persona):\n    def __init__(self, nombre, edad, carrera):\n        super().__init__(nombre, edad)  # Llamar constructor padre\n        self.carrera = carrera\n    \n    def estudiar(self):\n        return f\"{self.nombre} estudia {self.carrera}\"\n\n# Crear estudiante\nana = Estudiante(\"Ana\", 20, \"Ingeniería\")\nprint(ana.saludar())    # Hereda método\nprint(ana.estudiar())   # Método propio\n\n# Atributos y métodos de clase\nclass Contador:\n    total = 0  # Atributo de clase\n    \n    @classmethod\n    def incrementar(cls):\n        cls.total += 1\n```"
            },
            # Exception handling
            {
                "instruction": "¿Cómo se manejan excepciones en Python?",
                "input": "",
                "output": "El manejo de excepciones usa try/except:\n\n```python\n# Básico\ntry:\n    resultado = 10 / 0\nexcept ZeroDivisionError:\n    print(\"No se puede dividir por cero\")\n\n# Múltiples excepciones\ntry:\n    numero = int(input(\"Número: \"))\n    resultado = 10 / numero\nexcept ValueError:\n    print(\"Entrada no válida\")\nexcept ZeroDivisionError:\n    print(\"No dividir por cero\")\n\n# Capturar cualquier excepción\ntry:\n    # código riesgoso\n    pass\nexcept Exception as e:\n    print(f\"Error: {e}\")\n\n# finally - siempre se ejecuta\ntry:\n    archivo = open(\"datos.txt\")\n    # procesar\nexcept FileNotFoundError:\n    print(\"Archivo no encontrado\")\nfinally:\n    print(\"Limpieza\")\n\n# else - si no hay excepción\ntry:\n    resultado = 10 / 2\nexcept ZeroDivisionError:\n    print(\"Error\")\nelse:\n    print(f\"Resultado: {resultado}\")\n\n# Lanzar excepciones\ndef validar_edad(edad):\n    if edad < 0:\n        raise ValueError(\"La edad no puede ser negativa\")\n    return edad\n\n# Excepciones personalizadas\nclass MiError(Exception):\n    pass\n```"
            },
            # List comprehensions
            {
                "instruction": "¿Qué son las list comprehensions en Python?",
                "input": "",
                "output": "Las list comprehensions son una forma concisa de crear listas:\n\n```python\n# Sintaxis básica\n# [expresión for item in iterable]\n\n# Equivalente a un for loop\ncuadrados = []\nfor x in range(10):\n    cuadrados.append(x ** 2)\n\n# Con list comprehension\ncuadrados = [x ** 2 for x in range(10)]\n# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]\n\n# Con condición (filtro)\npares = [x for x in range(10) if x % 2 == 0]\n# [0, 2, 4, 6, 8]\n\n# Con if/else\nresultado = [\"par\" if x % 2 == 0 else \"impar\" for x in range(5)]\n# ['par', 'impar', 'par', 'impar', 'par']\n\n# Anidadas\nmatriz = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\naplanada = [num for fila in matriz for num in fila]\n# [1, 2, 3, 4, 5, 6, 7, 8, 9]\n\n# Dict comprehension\ncuadrados_dict = {x: x**2 for x in range(5)}\n# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}\n\n# Set comprehension\nunicas = {x % 3 for x in range(10)}\n# {0, 1, 2}\n\n# Generator expression (lazy evaluation)\ngen = (x ** 2 for x in range(1000000))  # No ocupa memoria\n```"
            },
            # File handling
            {
                "instruction": "¿Cómo se trabaja con archivos en Python?",
                "input": "",
                "output": "Python maneja archivos con open() y context managers:\n\n```python\n# Leer archivo completo\nwith open(\"archivo.txt\", \"r\") as f:\n    contenido = f.read()\n\n# Leer línea por línea\nwith open(\"archivo.txt\", \"r\") as f:\n    for linea in f:\n        print(linea.strip())\n\n# Leer todas las líneas como lista\nwith open(\"archivo.txt\", \"r\") as f:\n    lineas = f.readlines()\n\n# Escribir archivo (sobrescribe)\nwith open(\"archivo.txt\", \"w\") as f:\n    f.write(\"Hola mundo\\n\")\n\n# Agregar al final\nwith open(\"archivo.txt\", \"a\") as f:\n    f.write(\"Nueva línea\\n\")\n\n# Modos de apertura\n# \"r\"  - leer (default)\n# \"w\"  - escribir (sobrescribe)\n# \"a\"  - agregar al final\n# \"x\"  - crear (error si existe)\n# \"b\"  - modo binario (\"rb\", \"wb\")\n# \"+\"  - lectura y escritura (\"r+\")\n\n# Trabajar con JSON\nimport json\n\n# Escribir JSON\ndatos = {\"nombre\": \"Ana\", \"edad\": 30}\nwith open(\"datos.json\", \"w\") as f:\n    json.dump(datos, f, indent=2)\n\n# Leer JSON\nwith open(\"datos.json\", \"r\") as f:\n    datos = json.load(f)\n\n# Verificar si archivo existe\nfrom pathlib import Path\nif Path(\"archivo.txt\").exists():\n    print(\"Existe\")\n```"
            },
            # Imports
            {
                "instruction": "¿Cómo funcionan los imports en Python?",
                "input": "",
                "output": "Los imports permiten usar código de otros módulos:\n\n```python\n# Importar módulo completo\nimport math\nprint(math.sqrt(16))  # 4.0\nprint(math.pi)        # 3.14159...\n\n# Importar con alias\nimport numpy as np\nimport pandas as pd\n\n# Importar funciones específicas\nfrom math import sqrt, pi\nprint(sqrt(16))  # 4.0\n\n# Importar todo (no recomendado)\nfrom math import *\n\n# Importar de un paquete\nfrom collections import Counter, defaultdict\nfrom typing import List, Dict, Optional\n\n# Imports relativos (dentro de un paquete)\nfrom . import modulo_hermano\nfrom .. import modulo_padre\nfrom .submodulo import funcion\n\n# Módulos de la librería estándar comunes\nimport os           # Sistema operativo\nimport sys          # Sistema Python\nimport json         # JSON\nimport datetime     # Fechas\nimport re           # Expresiones regulares\nimport random       # Números aleatorios\nimport pathlib      # Rutas de archivos\nimport logging      # Logs\nimport unittest     # Testing\n\n# Instalar paquetes externos\n# pip install requests pandas numpy\n\nimport requests     # HTTP requests\nimport pandas as pd # Análisis de datos\n```"
            },
            # Decorators
            {
                "instruction": "¿Qué son los decoradores en Python?",
                "input": "",
                "output": "Los decoradores modifican el comportamiento de funciones o clases:\n\n```python\n# Decorador básico\ndef mi_decorador(func):\n    def wrapper(*args, **kwargs):\n        print(\"Antes de la función\")\n        resultado = func(*args, **kwargs)\n        print(\"Después de la función\")\n        return resultado\n    return wrapper\n\n@mi_decorador\ndef saludar(nombre):\n    print(f\"Hola, {nombre}\")\n\nsaludar(\"Ana\")\n# Antes de la función\n# Hola, Ana\n# Después de la función\n\n# Decorador con parámetros\ndef repetir(veces):\n    def decorador(func):\n        def wrapper(*args, **kwargs):\n            for _ in range(veces):\n                func(*args, **kwargs)\n        return wrapper\n    return decorador\n\n@repetir(3)\ndef decir_hola():\n    print(\"Hola\")\n\n# Decoradores built-in\nclass MiClase:\n    @staticmethod\n    def metodo_estatico():\n        pass\n    \n    @classmethod\n    def metodo_clase(cls):\n        pass\n    \n    @property\n    def valor(self):\n        return self._valor\n    \n    @valor.setter\n    def valor(self, v):\n        self._valor = v\n\n# Preservar metadata\nfrom functools import wraps\n\ndef mi_decorador(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        return func(*args, **kwargs)\n    return wrapper\n```"
            },
        ]

        return syntax_qa

    def process_book(self, text_path: str, chapters_path: Optional[str] = None) -> List[Dict]:
        """Process a book and generate Q&A pairs.

        Args:
            text_path: Path to extracted text file
            chapters_path: Optional path to chapters JSON

        Returns:
            List of generated Q&A pairs
        """
        logger.info(f"Processing: {text_path}")

        text = self.load_text(text_path)
        qa_pairs = []

        # Generate from syntax templates
        syntax_qa = self.generate_syntax_qa()
        qa_pairs.extend(syntax_qa)
        logger.info(f"Generated {len(syntax_qa)} syntax Q&A pairs")

        # Generate from book content
        for topic in PYTHON_TOPICS:
            relevant = self.find_relevant_content(text, topic)
            if relevant:
                content = "\n\n".join(relevant)
                topic_qa = self.generate_qa_from_content(content, topic)
                qa_pairs.extend(topic_qa)
                logger.info(f"Generated {len(topic_qa)} Q&A pairs for {topic}")

        # Extract code examples and create Q&A
        code_examples = self.extract_code_examples(text)
        for context, code in code_examples[:20]:  # Limit
            qa_pairs.append({
                "instruction": f"Muestra un ejemplo de código Python para: {context[:100]}",
                "input": "",
                "output": f"{context}\n\n```python\n{code}\n```",
                "topic": "code_examples"
            })

        self.qa_pairs = qa_pairs
        return qa_pairs

    def save_dataset(self, name: str = "python_syntax") -> Path:
        """Save generated Q&A pairs as JSONL.

        Args:
            name: Dataset name

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        output_path = self.output_dir / f"{name}_{timestamp}.jsonl"

        with open(output_path, 'w', encoding='utf-8') as f:
            for qa in self.qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')

        logger.info(f"Saved {len(self.qa_pairs)} Q&A pairs to: {output_path}")
        return output_path

    def generate_default_dataset(self) -> Path:
        """Generate default Python syntax dataset without book.

        Returns:
            Path to saved dataset
        """
        self.qa_pairs = self.generate_syntax_qa()
        return self.save_dataset("python_syntax_base")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate Python Q&A dataset")
    parser.add_argument("--text", help="Path to extracted text file")
    parser.add_argument("--chapters", help="Path to chapters JSON file")
    parser.add_argument("--output-dir", default="./data/datasets", help="Output directory")
    parser.add_argument("--name", default="python_syntax", help="Dataset name")
    parser.add_argument("--default", action="store_true", help="Generate default dataset without book")
    args = parser.parse_args()

    generator = PythonQAGenerator(output_dir=args.output_dir)

    if args.default:
        output_path = generator.generate_default_dataset()
    elif args.text:
        generator.process_book(args.text, args.chapters)
        output_path = generator.save_dataset(args.name)
    else:
        print("Use --text to process a book or --default for base dataset")
        return

    print(f"\n=== Dataset Generated ===")
    print(f"Q&A pairs: {len(generator.qa_pairs)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
