#!/usr/bin/env python3
"""
THAU Unified Training Script
Combina TODOS los datasets en un solo modelo unificado: thau-1.1b
"""

import json
import os
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Configuración
DATASETS_DIR = Path(__file__).parent.parent / "data" / "datasets"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "datasets"
MODELFILE_DIR = Path(__file__).parent.parent

# Todos los datasets disponibles para combinar
DATASETS_TO_COMBINE = [
    # Programación
    "programming_chat_format.jsonl",
    "programming_combined_20251202.jsonl",
    "python_training_20251202.jsonl",
    "python_aprende_book_20251202.jsonl",
    "javascript_training_20251202.jsonl",
    "java_training_20251202.jsonl",
    "rust_go_training_20251202.jsonl",
    "web_training_20251202.jsonl",
    "sql_databases_training_20251202.jsonl",

    # Razonamiento y AGI
    "reasoning_training.jsonl",
    "reasoning_cot.jsonl",

    # Desarrollo
    "thau_developer_20251207_073513.jsonl",
    "thau_advanced_20251207_080734.jsonl",

    # SVG y Assets
    "svg_assets_training.jsonl",
    "project_analysis_training.jsonl",

    # UX/UI
    "ux_chat_format.jsonl",
    "ux_css_frameworks.jsonl",

    # Contabilidad
    "contable_training.jsonl",

    # Agile y DevOps
    "agile_training_20251202.jsonl",
    "devops_training_20251202.jsonl",
    "git_training_20251202.jsonl",

    # Algoritmos y Matemáticas
    "algorithms_training_20251202.jsonl",
    "math_training_20251202.jsonl",

    # Tool calling
    "tool_calling.jsonl",

    # Otros
    "thau_v3_clean.jsonl",
    "agent_training.jsonl",
]


def load_jsonl(filepath: Path) -> List[Dict]:
    """Carga un archivo JSONL"""
    entries = []
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return entries


def combine_datasets() -> tuple:
    """Combina todos los datasets disponibles"""
    all_entries = []
    stats = {}

    print("\n📚 Combinando datasets...\n")

    for dataset in DATASETS_TO_COMBINE:
        filepath = DATASETS_DIR / dataset
        if filepath.exists():
            entries = load_jsonl(filepath)
            count = len(entries)
            if count > 0:
                all_entries.extend(entries)
                stats[dataset] = count
                print(f"  ✅ {dataset}: {count} ejemplos")
        else:
            print(f"  ⚠️  {dataset}: no encontrado")

    # Eliminar duplicados basados en instruction
    seen = set()
    unique_entries = []
    for entry in all_entries:
        instruction = entry.get('instruction', '') or entry.get('prompt', '')
        if instruction and instruction not in seen:
            seen.add(instruction)
            unique_entries.append(entry)

    print(f"\n📊 Estadísticas:")
    print(f"   Total cargados: {len(all_entries)}")
    print(f"   Únicos: {len(unique_entries)}")
    print(f"   Duplicados eliminados: {len(all_entries) - len(unique_entries)}")

    return unique_entries, stats


def save_combined_dataset(entries: List[Dict]) -> Path:
    """Guarda el dataset combinado"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"thau_unified_{timestamp}.jsonl"

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n💾 Dataset guardado: {output_file}")
    print(f"   Tamaño: {output_file.stat().st_size / 1024:.1f} KB")

    return output_file


def create_modelfile(dataset_path: Path) -> Path:
    """Crea el Modelfile para thau-1.1b"""

    modelfile_content = '''# THAU 1.1B - Modelo Unificado con Todas las Capacidades
# Parámetros: 1.1 Billion
# Creado: ''' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '''

FROM tinyllama

SYSTEM """Eres THAU (Thinking Human-like Artificial Understanding), un asistente de IA avanzado con 1.1B parámetros.

## CAPACIDADES PRINCIPALES

### 1. Desarrollo de Software
- Código completo y funcional en múltiples lenguajes
- Python, JavaScript, TypeScript, Java, Rust, Go, SQL
- React, Next.js, FastAPI, Flask, Spring Boot
- Arquitectura limpia, patrones de diseño, SOLID

### 2. Generación de Assets
- Creación de SVG (logos, iconos, animaciones)
- Diseño de interfaces UI/UX
- Sistemas de iconos completos

### 3. Razonamiento Avanzado (Chain of Thought)
- Análisis paso a paso de problemas complejos
- Debugging sistemático
- Optimización de código

### 4. Desarrollo Ágil
- Scrum, Kanban, XP
- DevOps, CI/CD, Git workflows
- Testing (unit, integration, e2e)

### 5. Bases de Datos
- SQL avanzado, PostgreSQL, MySQL
- NoSQL, MongoDB, Redis
- Diseño de esquemas, optimización

### 6. Contabilidad y Finanzas
- Partida doble, estados financieros
- Análisis de costos, presupuestos

### 7. Matemáticas y Algoritmos
- Estructuras de datos
- Algoritmos de ordenamiento y búsqueda
- Complejidad computacional

## FORMATO DE RESPUESTAS

Para código, usa:
**ruta/archivo.ext**
```lenguaje
// código completo y funcional
```

Para SVG:
**nombre.svg**
```svg
<svg>...</svg>
```

Para razonamiento:
### Paso 1: [Análisis]
### Paso 2: [Solución]
### Conclusión: [Resultado]

## REGLAS
1. Genera código COMPLETO, nunca fragmentos
2. Explica tu razonamiento cuando sea útil
3. Usa markdown para formateo claro
4. Responde en el idioma del usuario (español por defecto)
5. Sé conciso pero completo
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 8192
PARAMETER num_predict 4096
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "</s>"
'''

    modelfile_path = MODELFILE_DIR / "Modelfile_thau_1.1b"
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)

    print(f"\n📄 Modelfile creado: {modelfile_path}")
    return modelfile_path


def create_model():
    """Crea el modelo en Ollama"""
    print("\n🔨 Creando modelo thau-1.1b en Ollama...")

    modelfile_path = MODELFILE_DIR / "Modelfile_thau_1.1b"

    try:
        result = subprocess.run(
            ["ollama", "create", "thau-1.1b", "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print("✅ Modelo thau-1.1b creado exitosamente")

            # Crear alias como latest
            subprocess.run(
                ["ollama", "cp", "thau-1.1b", "thau:latest"],
                capture_output=True,
                timeout=60
            )
            print("✅ thau:latest actualizado")

            return True
        else:
            print(f"❌ Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Timeout al crear el modelo")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def show_summary(stats: Dict, total: int):
    """Muestra resumen del entrenamiento"""
    print("\n" + "="*60)
    print("📋 RESUMEN DE THAU-1.1B UNIFIED")
    print("="*60)

    categories = {
        "Programación": ["programming", "python", "javascript", "java", "rust", "web", "sql"],
        "Razonamiento": ["reasoning", "cot"],
        "Desarrollo": ["developer", "advanced", "agile", "devops", "git"],
        "Assets/SVG": ["svg", "project_analysis"],
        "UX/UI": ["ux"],
        "Contabilidad": ["contable"],
        "Algoritmos": ["algorithms", "math"],
        "Otros": ["tool_calling", "agent", "v3"]
    }

    for category, keywords in categories.items():
        count = sum(
            v for k, v in stats.items()
            if any(kw in k.lower() for kw in keywords)
        )
        if count > 0:
            print(f"  {category}: {count} ejemplos")

    print(f"\n  TOTAL: {total} ejemplos únicos")
    print("="*60)


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║             THAU-1.1B UNIFIED TRAINING                       ║
║          Combinando todas las capacidades                    ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 1. Combinar datasets
    entries, stats = combine_datasets()

    if not entries:
        print("❌ No se encontraron datos para entrenar")
        return

    # 2. Guardar dataset combinado
    dataset_path = save_combined_dataset(entries)

    # 3. Crear Modelfile
    create_modelfile(dataset_path)

    # 4. Crear modelo en Ollama
    success = create_model()

    # 5. Mostrar resumen
    show_summary(stats, len(entries))

    if success:
        print("""
╔══════════════════════════════════════════════════════════════╗
║                    ✅ COMPLETADO                             ║
║                                                              ║
║  Modelo disponible como:                                     ║
║    - thau-1.1b (principal)                                   ║
║    - thau:latest (alias)                                     ║
║                                                              ║
║  Uso: ollama run thau-1.1b                                   ║
╚══════════════════════════════════════════════════════════════╝
        """)


if __name__ == "__main__":
    main()
