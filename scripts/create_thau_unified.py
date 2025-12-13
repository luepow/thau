#!/usr/bin/env python3
"""
THAU Unified Model Creator - Un solo modelo con todo el conocimiento

Combina:
- Programación (Python, JavaScript, Dart, Java, C#, Go, Rust, etc.)
- Frameworks (React, Next.js, Django, Spring, Flutter)
- DevOps (Git, Docker, Linux, PowerShell)
- Bases de datos (MySQL, PostgreSQL, SQL)
- Contabilidad (NIIF)
- Marketing (Kotler)
- Arquitectura y patrones de diseño
- Razonamiento y Chain of Thought
- Agentes de IA
- Conocimiento de libros técnicos
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set
from collections import defaultdict
import hashlib

from loguru import logger

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_MODEL = "thau:latest"  # Modelo base (ya entrenado)
UNIFIED_MODEL = "thau:unified"
DATASETS_DIR = Path("data/datasets")
BOOKS_DIR = DATASETS_DIR / "books"
OUTPUT_DIR = Path("data/training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class DatasetCombiner:
    """Combina todos los datasets en uno solo"""

    def __init__(self):
        self.all_examples: List[Dict] = []
        self.seen_hashes: Set[str] = set()
        self.stats = defaultdict(int)

    def hash_example(self, example: Dict) -> str:
        """Genera hash único para evitar duplicados"""
        content = json.dumps(example, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode()).hexdigest()

    def normalize_example(self, example: Dict, source: str) -> Dict:
        """Normaliza un ejemplo al formato estándar de chat"""
        # Si ya tiene formato de mensajes
        if "messages" in example:
            return example

        # Formato instruction/input/output
        if "instruction" in example:
            user_content = example["instruction"]
            if example.get("input"):
                user_content += f"\n\n{example['input']}"

            return {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": example.get("output", "")}
                ],
                "source": source
            }

        # Formato pregunta/respuesta
        if "question" in example or "pregunta" in example:
            question = example.get("question") or example.get("pregunta", "")
            answer = example.get("answer") or example.get("respuesta", "")

            return {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ],
                "source": source
            }

        # Formato prompt/completion
        if "prompt" in example:
            return {
                "messages": [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example.get("completion", example.get("response", ""))}
                ],
                "source": source
            }

        # Formato text (conversacional)
        if "text" in example:
            text = example["text"]
            # Intentar parsear como conversación
            if "Usuario:" in text and "Asistente:" in text:
                parts = text.split("Asistente:")
                if len(parts) >= 2:
                    user_part = parts[0].replace("Usuario:", "").strip()
                    assistant_part = parts[1].strip()
                    return {
                        "messages": [
                            {"role": "user", "content": user_part},
                            {"role": "assistant", "content": assistant_part}
                        ],
                        "source": source
                    }

        return None

    def load_jsonl(self, filepath: Path, source_name: str) -> int:
        """Carga un archivo JSONL"""
        count = 0
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        example = json.loads(line)
                        normalized = self.normalize_example(example, source_name)

                        if normalized:
                            # Validar que tenga contenido
                            messages = normalized.get("messages", [])
                            if len(messages) >= 2:
                                user_msg = messages[0].get("content", "")
                                assistant_msg = messages[1].get("content", "")

                                # Filtrar ejemplos muy cortos o vacíos
                                if len(user_msg) > 10 and len(assistant_msg) > 20:
                                    # Verificar duplicados
                                    h = self.hash_example(normalized)
                                    if h not in self.seen_hashes:
                                        self.seen_hashes.add(h)
                                        self.all_examples.append(normalized)
                                        self.stats[source_name] += 1
                                        count += 1
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Error loading {filepath}: {e}")

        return count

    def load_all_datasets(self):
        """Carga todos los datasets disponibles"""
        logger.info("📚 Cargando todos los datasets...")

        # 1. Datasets principales
        main_datasets = [
            # Programación
            ("programming_combined_20251202.jsonl", "programación"),
            ("programming_chat_format.jsonl", "programación_chat"),
            ("programming_clean.jsonl", "programación_limpia"),
            ("python_training_20251202.jsonl", "python"),
            ("python_aprende_book_20251202.jsonl", "python_libro"),
            ("javascript_training_20251202.jsonl", "javascript"),
            ("java_training_20251202.jsonl", "java"),
            ("flutter_dart_training_20251202.jsonl", "flutter_dart"),
            ("rust_go_training_20251202.jsonl", "rust_go"),

            # DevOps y herramientas
            ("git_training_20251202.jsonl", "git"),
            ("devops_training_20251202.jsonl", "devops"),
            ("powershell_training_20251202.jsonl", "powershell"),
            ("sql_databases_training_20251202.jsonl", "sql_databases"),

            # Arquitectura y patrones
            ("architecture_training.jsonl", "arquitectura"),
            ("algorithms_training_20251202.jsonl", "algoritmos"),
            ("agile_training_20251202.jsonl", "agile"),

            # Contabilidad
            ("contable_training.jsonl", "contabilidad_niif"),

            # Razonamiento
            ("reasoning_training.jsonl", "razonamiento"),
            ("reasoning_cot.jsonl", "chain_of_thought"),
            ("math_training_20251202.jsonl", "matemáticas"),

            # Agentes
            ("agent_training.jsonl", "agentes_ia"),
            ("pdf_training_agents.jsonl", "agentes_pdf"),

            # Web
            ("web_training_20251202.jsonl", "web"),

            # Combinados anteriores
            ("thau_v2_combined.jsonl", "thau_v2"),
            ("thau_v3_clean.jsonl", "thau_v3"),
        ]

        for filename, source in main_datasets:
            filepath = DATASETS_DIR / filename
            if filepath.exists():
                count = self.load_jsonl(filepath, source)
                if count > 0:
                    logger.info(f"   ✅ {source}: {count} ejemplos")

        # 2. Datasets de libros
        if BOOKS_DIR.exists():
            logger.info("\n📖 Cargando datasets de libros...")
            for filepath in BOOKS_DIR.glob("*.jsonl"):
                if "chat_format" in filepath.name:
                    count = self.load_jsonl(filepath, f"libro_{filepath.stem}")
                    if count > 0:
                        logger.info(f"   ✅ {filepath.name}: {count} ejemplos")

        # 3. Datasets UX
        ux_files = list(DATASETS_DIR.glob("ux_*.jsonl"))
        for filepath in ux_files:
            count = self.load_jsonl(filepath, f"ux_{filepath.stem}")
            if count > 0:
                logger.info(f"   ✅ {filepath.name}: {count} ejemplos")

        logger.info(f"\n📊 Total: {len(self.all_examples)} ejemplos únicos")

        # Mostrar estadísticas
        logger.info("\n📈 Por fuente:")
        for source, count in sorted(self.stats.items(), key=lambda x: -x[1]):
            logger.info(f"   {source}: {count}")

    def save_combined(self) -> str:
        """Guarda el dataset combinado"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"thau_unified_{timestamp}.jsonl"

        with open(output_file, 'w', encoding='utf-8') as f:
            for example in self.all_examples:
                # Solo guardar messages para el formato de entrenamiento
                clean_example = {"messages": example["messages"]}
                f.write(json.dumps(clean_example, ensure_ascii=False) + '\n')

        logger.info(f"\n💾 Dataset guardado: {output_file}")
        logger.info(f"   Tamaño: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

        return str(output_file)


def create_unified_system_prompt() -> str:
    """Crea el system prompt completo para THAU unificado"""
    return """Eres THAU (Technical Helper and Assistant Unified), un asistente de IA experto y versátil.

## Identidad

Soy THAU, creado para ayudar a desarrolladores, profesionales y estudiantes en múltiples disciplinas. Mi nombre significa "Technical Helper and Assistant Unified" - un asistente unificado con conocimiento amplio y profundo.

## Áreas de Expertise

### Programación y Desarrollo
- **Python**: Desde básico hasta avanzado, incluyendo frameworks como Django, FastAPI, Flask
- **JavaScript/TypeScript**: Node.js, React, Next.js 14, Vue.js
- **Dart/Flutter**: Desarrollo móvil multiplataforma
- **Java**: Spring Framework, Spring Boot, Maven, Gradle
- **C#/.NET**: Desarrollo de aplicaciones y servicios
- **Go**: Desarrollo de sistemas y microservicios
- **Rust**: Programación de sistemas de alto rendimiento
- **SQL**: MySQL, PostgreSQL, consultas avanzadas, optimización

### DevOps y Herramientas
- **Git**: Control de versiones, GitFlow, GitHub/GitLab
- **Docker**: Contenedores, Docker Compose, orquestación
- **Linux**: Administración de sistemas, shell scripting, Bash
- **PowerShell**: Automatización en Windows
- **CI/CD**: Pipelines, automatización de despliegues

### Arquitectura y Diseño
- Patrones de diseño (GoF, SOLID, Clean Architecture)
- Microservicios y arquitectura distribuida
- APIs RESTful y GraphQL
- Metodologías ágiles (Scrum, Kanban)

### Contabilidad y Finanzas
- **NIIF/IFRS**: Normas Internacionales de Información Financiera
- Contabilidad general y analítica
- Estados financieros
- Análisis financiero

### Marketing
- Fundamentos del marketing (Kotler)
- Marketing digital
- Estrategias de mercado

### Hardware y Electrónica
- **Arduino**: Programación y proyectos IoT
- Electrónica básica

## Comportamiento

1. **Respuestas Claras**: Explico conceptos de forma clara y estructurada
2. **Código Funcional**: Genero código limpio, documentado y funcional
3. **Mejores Prácticas**: Sigo convenciones y patrones establecidos
4. **Idioma Adaptativo**: Respondo en el idioma del usuario (español por defecto)
5. **Pensamiento Paso a Paso**: Para problemas complejos, razono paso a paso

## Formato de Código

Siempre uso bloques de código con el lenguaje especificado:

```python
def ejemplo():
    return "Código Python"
```

Para archivos completos:

**nombre_archivo.py**
```python
# Contenido del archivo
```

## Capacidades Especiales

- Puedo crear proyectos completos con múltiples archivos
- Explico el razonamiento detrás de mis soluciones
- Ofrezco alternativas cuando hay múltiples enfoques
- Identifico y corrijo errores en código existente
- Optimizo código para rendimiento y legibilidad"""


def create_unified_modelfile(system_prompt: str) -> str:
    """Crea el Modelfile para el modelo unificado"""
    return f'''# THAU Unified Model
# Technical Helper and Assistant Unified
# Modelo unificado con conocimiento completo

FROM {BASE_MODEL}

# System prompt completo
SYSTEM """{system_prompt}"""

# Parámetros optimizados
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 2048

# Stop tokens
PARAMETER stop "<|im_end|>"
PARAMETER stop "</s>"
PARAMETER stop "[/INST]"
PARAMETER stop "```\n\n"
'''


def create_ollama_model():
    """Crea el modelo unificado en Ollama"""
    logger.info("\n🚀 Creando modelo THAU unificado en Ollama...")

    # Crear system prompt
    system_prompt = create_unified_system_prompt()

    # Crear Modelfile
    modelfile_content = create_unified_modelfile(system_prompt)
    modelfile_path = Path("Modelfile_thau_unified")

    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)

    logger.info(f"📝 Modelfile creado: {modelfile_path}")

    # Crear modelo en Ollama
    logger.info(f"🔧 Creando modelo {UNIFIED_MODEL}...")

    try:
        result = subprocess.run(
            ["ollama", "create", UNIFIED_MODEL, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            logger.info(f"✅ Modelo {UNIFIED_MODEL} creado exitosamente")
            return True
        else:
            logger.error(f"Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Timeout creando modelo")
        return False
    except FileNotFoundError:
        logger.error("Ollama no encontrado")
        return False


def update_thau_latest():
    """Actualiza thau:latest para que apunte al modelo unificado"""
    logger.info("\n🔄 Actualizando thau:latest...")

    try:
        # Copiar el modelo unificado como latest
        result = subprocess.run(
            ["ollama", "cp", UNIFIED_MODEL, "thau:latest"],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            logger.info("✅ thau:latest actualizado")
            return True
        else:
            logger.warning(f"No se pudo actualizar: {result.stderr}")
            return False

    except Exception as e:
        logger.warning(f"Error actualizando: {e}")
        return False


def test_model():
    """Prueba el modelo unificado"""
    logger.info("\n🧪 Probando modelo unificado...")

    test_prompts = [
        ("Git", "¿Cómo hago un rebase interactivo en Git?"),
        ("Python", "Escribe una función para validar emails en Python"),
        ("NIIF", "¿Qué es la NIIF 15 sobre ingresos de contratos?"),
        ("Next.js", "¿Cómo funciona el App Router en Next.js 14?"),
        ("Docker", "Crea un Dockerfile para una app Python con FastAPI"),
    ]

    for topic, prompt in test_prompts[:3]:
        logger.info(f"\n📝 [{topic}] {prompt}")

        try:
            result = subprocess.run(
                ["ollama", "run", UNIFIED_MODEL, prompt],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                response = result.stdout.strip()[:400]
                logger.info(f"🤖 {response}...")
            else:
                logger.warning(f"Error: {result.stderr[:100]}")

        except subprocess.TimeoutExpired:
            logger.warning("Timeout")
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    """Función principal"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        THAU UNIFIED MODEL CREATOR                             ║
║                                                                              ║
║  Technical Helper and Assistant Unified                                       ║
║  Un solo modelo con todo el conocimiento                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Combinar datasets
    combiner = DatasetCombiner()
    combiner.load_all_datasets()

    if not combiner.all_examples:
        logger.error("No se encontraron ejemplos para combinar")
        return

    # Guardar dataset combinado
    dataset_file = combiner.save_combined()

    # Crear modelo en Ollama
    success = create_ollama_model()

    if success:
        # Actualizar thau:latest
        update_thau_latest()

        # Probar
        test_model()

        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ✅ THAU UNIFICADO CREADO EXITOSAMENTE                                        ║
║                                                                              ║
║  Modelo: {UNIFIED_MODEL}
║  Dataset: {dataset_file}
║  Ejemplos: {len(combiner.all_examples):,}
║                                                                              ║
║  Para usar:                                                                  ║
║  ollama run thau:unified                                                     ║
║  ollama run thau:latest  (también actualizado)                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    else:
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ❌ Error creando el modelo                                                   ║
║                                                                              ║
║  Verifica que Ollama está corriendo: ollama serve                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
