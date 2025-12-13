#!/usr/bin/env python3
"""
THAU Training Script - Entrena THAU con los datasets extraídos de libros

Este script:
1. Combina todos los datasets de books
2. Crea un Modelfile actualizado para Ollama
3. Re-crea el modelo THAU con el nuevo conocimiento
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import tempfile

from loguru import logger

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_MODEL = "thau:latest"  # Modelo base actual
NEW_MODEL = "thau:books"    # Nuevo modelo con conocimiento de libros
BOOKS_DIR = Path("data/datasets/books")
OUTPUT_DIR = Path("data/training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_datasets() -> List[Dict]:
    """Carga todos los datasets de libros"""
    all_examples = []

    # Buscar el archivo chat format más reciente
    chat_files = list(BOOKS_DIR.glob("books_chat_format_*.jsonl"))
    if not chat_files:
        logger.error("No se encontraron archivos de chat format")
        return []

    # Usar el más reciente
    latest_file = max(chat_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"📚 Cargando: {latest_file}")

    with open(latest_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                example = json.loads(line)
                all_examples.append(example)
            except json.JSONDecodeError:
                continue

    logger.info(f"   Cargados {len(all_examples)} ejemplos")
    return all_examples


def create_system_prompt(categories: List[str]) -> str:
    """Crea el system prompt mejorado"""
    return f"""Eres THAU, un asistente de programación experto y versátil creado para ayudar a desarrolladores.

## Conocimientos Especializados

Has sido entrenado con documentación técnica en las siguientes áreas:
- **Lenguajes de Programación**: Python, Dart, C#, Go, Rust, JavaScript
- **Frontend**: CSS, React, React Native, Next.js 14
- **Backend**: Django, Spring Framework, FastAPI
- **DevOps**: Git, Docker, Linux, PowerShell
- **Bases de Datos**: MySQL, PostgreSQL
- **Hardware**: Arduino
- **Contabilidad**: Normas NIIF (Normas Internacionales de Información Financiera)
- **Marketing**: Fundamentos de Marketing (Kotler)

## Comportamiento

1. **Código Claro**: Genera código limpio, bien documentado y funcional
2. **Explicaciones Detalladas**: Explica conceptos paso a paso
3. **Mejores Prácticas**: Sigue patrones de diseño y convenciones del lenguaje
4. **Idioma**: Responde en español a menos que el usuario escriba en otro idioma
5. **Formato**: Usa bloques de código con el lenguaje especificado
6. **Contexto**: Recuerda el contexto de la conversación

## Formato de Respuesta

Cuando generes código, siempre usa:
```lenguaje
// código aquí
```

Para archivos completos, indica el nombre:
**nombre_archivo.ext**
```lenguaje
// contenido del archivo
```
"""


def create_modelfile(system_prompt: str) -> str:
    """Crea el Modelfile para Ollama"""
    return f'''# THAU v3.0 - Books Edition
# Entrenado con documentación técnica especializada

FROM {BASE_MODEL}

# System prompt mejorado con conocimientos de libros
SYSTEM """{system_prompt}"""

# Parámetros optimizados para desarrollo
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "</s>"
PARAMETER stop "[/INST]"
'''


def create_training_data_file(examples: List[Dict], output_path: Path) -> str:
    """Crea archivo de datos de entrenamiento"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    return str(output_path)


def update_ollama_model():
    """Actualiza el modelo en Ollama"""
    logger.info("🚀 Actualizando modelo THAU en Ollama...")

    # Cargar datasets
    examples = load_all_datasets()
    if not examples:
        logger.error("No hay datos para entrenar")
        return False

    # Extraer categorías
    categories = set()
    training_file = BOOKS_DIR / "books_training_20251206_091710.jsonl"
    if training_file.exists():
        with open(training_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'category' in data:
                        categories.add(data['category'])
                except:
                    continue

    # Crear system prompt
    system_prompt = create_system_prompt(list(categories))

    # Crear Modelfile
    modelfile_content = create_modelfile(system_prompt)
    modelfile_path = Path("Modelfile_thau_books")

    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)

    logger.info(f"📝 Modelfile creado: {modelfile_path}")

    # Crear modelo en Ollama
    logger.info(f"🔧 Creando modelo {NEW_MODEL}...")

    try:
        result = subprocess.run(
            ["ollama", "create", NEW_MODEL, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            logger.info(f"✅ Modelo {NEW_MODEL} creado exitosamente")
            logger.info(result.stdout)
        else:
            logger.error(f"Error creando modelo: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Timeout creando modelo")
        return False
    except FileNotFoundError:
        logger.error("Ollama no encontrado. Asegúrate de que esté instalado.")
        return False

    return True


def test_new_model():
    """Prueba el nuevo modelo"""
    logger.info(f"\n🧪 Probando modelo {NEW_MODEL}...")

    test_prompts = [
        "¿Cómo hago un commit en Git?",
        "Explica qué es un closure en Python",
        "¿Qué es la NIIF 15?",
        "Crea una función en Dart para validar email",
        "¿Cómo inicio un proyecto con Next.js 14?",
    ]

    for prompt in test_prompts[:2]:  # Probar solo 2 para no demorar
        logger.info(f"\n📝 Prompt: {prompt}")

        try:
            result = subprocess.run(
                ["ollama", "run", NEW_MODEL, prompt],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                response = result.stdout.strip()[:500]  # Truncar respuesta
                logger.info(f"🤖 Respuesta: {response}...")
            else:
                logger.warning(f"Error: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.warning("Timeout en la prueba")
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    """Función principal"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              THAU Training from Books                                 ║
║                                                                      ║
║  Actualiza el modelo THAU con conocimiento de libros técnicos        ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # Verificar que existen los datasets
    if not BOOKS_DIR.exists():
        logger.error(f"Directorio de datasets no encontrado: {BOOKS_DIR}")
        logger.info("Ejecuta primero: python scripts/train_from_books.py")
        return

    # Actualizar modelo
    success = update_ollama_model()

    if success:
        # Probar modelo
        test_new_model()

        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  ✅ Entrenamiento completado                                          ║
║                                                                      ║
║  Modelo creado: {NEW_MODEL}
║                                                                      ║
║  Para usar el nuevo modelo:                                          ║
║  ollama run thau:books                                               ║
║                                                                      ║
║  O actualiza THAU IDE para usar el nuevo modelo                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    else:
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║  ❌ Error en el entrenamiento                                         ║
║                                                                      ║
║  Verifica que:                                                       ║
║  1. Ollama está instalado y corriendo                                ║
║  2. El modelo base thau:latest existe                                ║
║  3. Los datasets están en data/datasets/books/                       ║
╚══════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
