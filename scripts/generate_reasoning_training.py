#!/usr/bin/env python3
"""
Genera datos de entrenamiento de razonamiento avanzado para THAU
Incluye Chain of Thought, analisis paso a paso, y resolucion de problemas
"""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("./data/self_questioning")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Problemas de razonamiento con solucion paso a paso
REASONING_PROBLEMS = [
    # Matematica y logica
    {
        "question": "Si tengo 3 manzanas y me dan el doble, luego regalo 2, cuantas me quedan?",
        "reasoning": """Vamos a resolver esto paso a paso:

1. Empiezo con 3 manzanas
2. Me dan el doble: 3 x 2 = 6 manzanas
3. Ahora tengo: 3 + 6 = 9 manzanas
4. Regalo 2: 9 - 2 = 7 manzanas

Respuesta: Me quedan 7 manzanas."""
    },
    {
        "question": "Un tren sale a las 9:00 y tarda 2 horas y 45 minutos. A que hora llega?",
        "reasoning": """Resolvamos paso a paso:

1. Hora de salida: 9:00
2. Duracion del viaje: 2 horas y 45 minutos
3. Sumamos las horas: 9:00 + 2:00 = 11:00
4. Sumamos los minutos: 11:00 + 0:45 = 11:45

Respuesta: El tren llega a las 11:45."""
    },
    {
        "question": "Si un libro cuesta 25 euros y tiene 30% de descuento, cuanto pago?",
        "reasoning": """Calculemos el descuento:

1. Precio original: 25 euros
2. Descuento: 30%
3. Valor del descuento: 25 x 0.30 = 7.5 euros
4. Precio final: 25 - 7.5 = 17.5 euros

Respuesta: Pagas 17.50 euros."""
    },
    {
        "question": "Tengo 5 camisas y 3 pantalones. Cuantas combinaciones diferentes puedo hacer?",
        "reasoning": """Apliquemos el principio de multiplicacion:

1. Opciones de camisas: 5
2. Opciones de pantalones: 3
3. Cada camisa puede combinarse con cada pantalon
4. Total de combinaciones: 5 x 3 = 15

Respuesta: Puedes hacer 15 combinaciones diferentes."""
    },
    {
        "question": "Si Ana es mas alta que Bea, y Bea es mas alta que Clara, quien es la mas baja?",
        "reasoning": """Analicemos las relaciones:

1. Ana > Bea (Ana es mas alta que Bea)
2. Bea > Clara (Bea es mas alta que Clara)
3. Por transitividad: Ana > Bea > Clara
4. El orden de altura es: Ana (mas alta), Bea (media), Clara (mas baja)

Respuesta: Clara es la mas baja."""
    },

    # Programacion y algoritmos
    {
        "question": "Como ordenarias una lista de numeros [5, 2, 8, 1, 9] de menor a mayor?",
        "reasoning": """Usemos el algoritmo de ordenamiento burbuja explicado paso a paso:

Lista inicial: [5, 2, 8, 1, 9]

Primera pasada:
- Comparo 5 y 2: intercambio -> [2, 5, 8, 1, 9]
- Comparo 5 y 8: no intercambio -> [2, 5, 8, 1, 9]
- Comparo 8 y 1: intercambio -> [2, 5, 1, 8, 9]
- Comparo 8 y 9: no intercambio -> [2, 5, 1, 8, 9]

Segunda pasada:
- [2, 5, 1, 8, 9] -> [2, 1, 5, 8, 9]

Tercera pasada:
- [2, 1, 5, 8, 9] -> [1, 2, 5, 8, 9]

Resultado final: [1, 2, 5, 8, 9]"""
    },
    {
        "question": "Que hace este codigo: for i in range(5): print(i*2)?",
        "reasoning": """Analicemos el codigo linea por linea:

1. `for i in range(5)`: Crea un bucle que itera con i = 0, 1, 2, 3, 4
2. `print(i*2)`: Imprime el valor de i multiplicado por 2

Ejecucion paso a paso:
- i = 0: imprime 0 * 2 = 0
- i = 1: imprime 1 * 2 = 2
- i = 2: imprime 2 * 2 = 4
- i = 3: imprime 3 * 2 = 6
- i = 4: imprime 4 * 2 = 8

Salida: 0, 2, 4, 6, 8 (cada uno en una linea)"""
    },
    {
        "question": "Como encontrarias el numero mas grande en una lista?",
        "reasoning": """Algoritmo para encontrar el maximo:

1. Supongo que el primer elemento es el maximo
2. Recorro la lista comparando cada elemento
3. Si encuentro uno mayor, actualizo el maximo
4. Al terminar, tengo el maximo

Ejemplo con [3, 7, 2, 9, 4]:
- max_actual = 3
- Comparo 7 > 3? Si, max_actual = 7
- Comparo 2 > 7? No
- Comparo 9 > 7? Si, max_actual = 9
- Comparo 4 > 9? No

Resultado: El maximo es 9"""
    },

    # Analisis y decision
    {
        "question": "Deberia usar una base de datos SQL o NoSQL para mi aplicacion de redes sociales?",
        "reasoning": """Analicemos ambas opciones:

**Requisitos de redes sociales:**
- Muchos usuarios
- Posts, comentarios, likes
- Relaciones entre usuarios (amigos/seguidores)
- Consultas frecuentes y variadas
- Escalabilidad

**SQL (Relacional):**
+ Bueno para relaciones complejas (amigos, grupos)
+ Transacciones ACID
+ Consultas estructuradas
- Escalabilidad horizontal mas compleja

**NoSQL (Documento/Grafo):**
+ Escalabilidad horizontal facil
+ Esquema flexible
+ Alto rendimiento para lecturas
- Consultas complejas mas dificiles

**Recomendacion:** Una arquitectura hibrida:
- PostgreSQL para usuarios y relaciones criticas
- MongoDB/Redis para posts y cache

Esto combina lo mejor de ambos mundos."""
    },
    {
        "question": "Como debuggearias un programa que se cuelga aleatoriamente?",
        "reasoning": """Metodologia de debugging sistematica:

**Paso 1: Recopilar informacion**
- Cuando ocurre el problema?
- Hay patrones? (hora, carga, acciones especificas)
- Revisar logs de errores

**Paso 2: Reproducir el problema**
- Intentar aislar las condiciones
- Crear un caso de prueba minimo

**Paso 3: Investigar causas comunes**
- Memory leaks (uso creciente de memoria)
- Race conditions (problemas de concurrencia)
- Deadlocks (bloqueos mutuos)
- Recursos agotados (archivos, conexiones)

**Paso 4: Herramientas de diagnostico**
- Profilers de memoria
- Logs detallados
- Core dumps
- Monitoreo de recursos

**Paso 5: Aislar y corregir**
- Dividir el codigo en secciones
- Probar cada parte
- Aplicar fix y validar"""
    },

    # Creatividad estructurada
    {
        "question": "Diseña una API REST para una tienda online",
        "reasoning": """Diseñemos la API paso a paso:

**1. Identificar recursos principales:**
- Productos
- Usuarios
- Carritos
- Pedidos

**2. Definir endpoints:**

Productos:
- GET /products - Listar productos
- GET /products/{id} - Detalle de producto
- POST /products - Crear (admin)
- PUT /products/{id} - Actualizar (admin)
- DELETE /products/{id} - Eliminar (admin)

Usuarios:
- POST /users/register - Registro
- POST /users/login - Login
- GET /users/me - Perfil actual

Carrito:
- GET /cart - Ver carrito
- POST /cart/items - Añadir item
- DELETE /cart/items/{id} - Quitar item

Pedidos:
- POST /orders - Crear pedido desde carrito
- GET /orders - Historial
- GET /orders/{id} - Detalle

**3. Consideraciones:**
- Autenticacion con JWT
- Paginacion para listas
- Codigos HTTP apropiados
- Validacion de datos"""
    },
]

# Problemas que requieren usar Gemini para respuestas mas elaboradas
COMPLEX_REASONING_PROMPTS = [
    {
        "question": "Explica como funciona la recursion con un ejemplo",
        "prompt": "Explica la recursion en programacion con un ejemplo simple de factorial. Muestra el proceso paso a paso de como se ejecutan las llamadas recursivas. Responde en español, maximo 200 palabras."
    },
    {
        "question": "Cual es la diferencia entre concurrencia y paralelismo?",
        "prompt": "Explica la diferencia entre concurrencia y paralelismo en programacion. Usa analogias simples y ejemplos practicos. Responde en español, maximo 200 palabras."
    },
    {
        "question": "Como funciona el garbage collector?",
        "prompt": "Explica como funciona el garbage collector en lenguajes como Python o Java. Describe los algoritmos basicos de recoleccion de basura. Responde en español, maximo 200 palabras."
    },
    {
        "question": "Que es el patron de diseño Singleton y cuando usarlo?",
        "prompt": "Explica el patron de diseño Singleton: que problema resuelve, como se implementa, y cuando es apropiado usarlo. Incluye ventajas y desventajas. Responde en español, maximo 200 palabras."
    },
    {
        "question": "Como optimizarias una consulta SQL lenta?",
        "prompt": "Describe un proceso paso a paso para diagnosticar y optimizar una consulta SQL lenta. Incluye uso de EXPLAIN, indices, y buenas practicas. Responde en español, maximo 200 palabras."
    },
    {
        "question": "Cual es la complejidad temporal de buscar en una lista vs un diccionario?",
        "prompt": "Explica la diferencia de complejidad temporal entre buscar un elemento en una lista (array) vs en un diccionario (hash table). Incluye Big O notation y por que ocurre esta diferencia. Responde en español, maximo 200 palabras."
    },
    {
        "question": "Como diseñarias un sistema de cache?",
        "prompt": "Explica como diseñar un sistema de cache efectivo. Incluye estrategias de invalidacion (LRU, TTL), consideraciones de memoria, y cuando usar cache. Responde en español, maximo 200 palabras."
    },
    {
        "question": "Que es una API RESTful y cuales son sus principios?",
        "prompt": "Explica los principios REST para APIs: stateless, recursos, verbos HTTP, codigos de respuesta. Da ejemplos practicos. Responde en español, maximo 200 palabras."
    },
]

def query_gemini(prompt: str) -> str:
    """Consulta a Gemini CLI"""
    try:
        result = subprocess.run(
            ["gemini", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception as e:
        print(f"Error llamando a Gemini: {e}")
        return ""

def generate_static_reasoning():
    """Genera ejemplos de razonamiento predefinidos"""
    examples = []
    timestamp = datetime.now().isoformat()

    print("\n🧠 Generando ejemplos de razonamiento paso a paso...")
    print("=" * 60)

    for i, item in enumerate(REASONING_PROBLEMS):
        entry = {
            "timestamp": timestamp,
            "question": item["question"],
            "answer": item["reasoning"],
            "confidence": 0.95,
            "cognitive_age": 14,  # Mayor edad cognitiva para razonamiento
            "category": "chain_of_thought",
            "self_generated": True,
            "reasoning_training": True
        }
        examples.append(entry)
        print(f"   [{i+1}/{len(REASONING_PROBLEMS)}] {item['question'][:50]}...")

    return examples

def generate_gemini_reasoning():
    """Genera ejemplos de razonamiento usando Gemini"""
    examples = []
    timestamp = datetime.now().isoformat()

    print("\n🤖 Generando razonamiento complejo con Gemini...")
    print("=" * 60)

    for i, item in enumerate(COMPLEX_REASONING_PROMPTS):
        print(f"   [{i+1}/{len(COMPLEX_REASONING_PROMPTS)}] {item['question'][:50]}...")

        answer = query_gemini(item["prompt"])

        if answer:
            entry = {
                "timestamp": timestamp,
                "question": item["question"],
                "answer": answer,
                "confidence": 0.9,
                "cognitive_age": 14,
                "category": "technical_reasoning",
                "self_generated": True,
                "source": "gemini",
                "reasoning_training": True
            }
            examples.append(entry)
            print(f"       ✅ Respuesta obtenida")
        else:
            print(f"       ❌ Sin respuesta")

        time.sleep(1)  # Rate limiting

    return examples

def generate_step_by_step_templates():
    """Genera plantillas que ensenan el formato de razonamiento"""
    examples = []
    timestamp = datetime.now().isoformat()

    print("\n📝 Generando plantillas de razonamiento...")

    templates = [
        {
            "question": "como debo pensar para resolver problemas complejos?",
            "answer": """Para resolver problemas complejos, sigo este proceso:

1. **Entender el problema**
   - Que se me esta pidiendo exactamente?
   - Cuales son los datos de entrada?
   - Cual es el resultado esperado?

2. **Descomponer en partes**
   - Dividir el problema en subproblemas mas pequeños
   - Identificar las dependencias entre partes

3. **Resolver paso a paso**
   - Atacar cada subproblema
   - Verificar cada paso antes de continuar

4. **Verificar la solucion**
   - Funciona para el caso dado?
   - Funciona para casos limite?
   - Hay una solucion mejor?

Este enfoque sistematico me ayuda a mantener claridad y evitar errores."""
        },
        {
            "question": "cuando debo usar chain of thought?",
            "answer": """Uso razonamiento paso a paso (chain of thought) cuando:

1. **Problemas matematicos** - Calculos que requieren multiples pasos
2. **Logica y deduccion** - Cuando hay que conectar multiples premisas
3. **Debugging** - Para rastrear el origen de un error
4. **Diseño** - Al tomar decisiones arquitectonicas
5. **Explicaciones** - Cuando necesito enseñar algo

El razonamiento explicito me ayuda a:
- No saltar pasos importantes
- Detectar errores en mi logica
- Comunicar mejor mi proceso
- Obtener resultados mas precisos

En cambio, para preguntas simples o factuales, puedo responder directamente."""
        },
    ]

    for item in templates:
        entry = {
            "timestamp": timestamp,
            "question": item["question"],
            "answer": item["answer"],
            "confidence": 0.95,
            "cognitive_age": 14,
            "category": "meta_reasoning",
            "self_generated": True,
            "reasoning_training": True
        }
        examples.append(entry)
        print(f"   ✅ {item['question'][:50]}...")

    return examples

def save_training_data(examples, filename="reasoning_training.jsonl"):
    """Guarda datos de entrenamiento"""
    output_path = OUTPUT_DIR / filename

    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"\n💾 Guardados {len(examples)} ejemplos en {output_path}")
    return output_path

def main():
    print("=" * 60)
    print("  GENERADOR DE DATOS - RAZONAMIENTO AVANZADO")
    print("=" * 60)

    all_examples = []

    # 1. Ejemplos estaticos de razonamiento
    static = generate_static_reasoning()
    all_examples.extend(static)

    # 2. Razonamiento con Gemini
    gemini = generate_gemini_reasoning()
    all_examples.extend(gemini)

    # 3. Plantillas meta-cognitivas
    templates = generate_step_by_step_templates()
    all_examples.extend(templates)

    # Guardar
    save_training_data(all_examples)

    # Resumen
    print("\n" + "=" * 60)
    print("  RESUMEN")
    print("=" * 60)
    print(f"  Estaticos:    {len(static)}")
    print(f"  Gemini:       {len(gemini)}")
    print(f"  Plantillas:   {len(templates)}")
    print(f"  TOTAL:        {len(all_examples)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
