#!/usr/bin/env python3
"""
THAU Complete Update Script
Merges LoRA adapters, converts to GGUF, and updates Ollama
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def step1_merge_adapters():
    """Merge LoRA adapters with base model"""
    print("\n" + "=" * 60)
    print("PASO 1: Mergeando adapters LoRA con modelo base")
    print("=" * 60)

    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    LORA_PATH = Path("data/checkpoints/specialized_training/final")
    OUTPUT_PATH = Path("thau_merged_model")

    if not LORA_PATH.exists():
        print(f"Error: No se encontro checkpoint en {LORA_PATH}")
        return False

    print(f"\nModelo base: {BASE_MODEL}")
    print(f"LoRA adapters: {LORA_PATH}")
    print(f"Salida: {OUTPUT_PATH}")

    # Load base model
    print("\nCargando modelo base...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Load LoRA weights
    print("Cargando pesos LoRA...")
    model = PeftModel.from_pretrained(base_model, str(LORA_PATH))

    # Merge LoRA with base model
    print("Mergeando adapters...")
    merged_model = model.merge_and_unload()

    # Save merged model
    print(f"Guardando modelo merged en {OUTPUT_PATH}...")
    OUTPUT_PATH.mkdir(exist_ok=True)
    merged_model.save_pretrained(str(OUTPUT_PATH))
    tokenizer.save_pretrained(str(OUTPUT_PATH))

    # Save metadata
    metadata = {
        "base_model": BASE_MODEL,
        "lora_checkpoint": str(LORA_PATH),
        "merged_date": datetime.now().isoformat(),
        "training_type": "specialized",
    }
    import json
    with open(OUTPUT_PATH / "thau_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModelo merged guardado en: {OUTPUT_PATH}")
    return True


def step2_convert_to_gguf():
    """Convert to GGUF format using llama.cpp"""
    print("\n" + "=" * 60)
    print("PASO 2: Convirtiendo a formato GGUF")
    print("=" * 60)

    MODEL_PATH = Path("thau_merged_model")
    LLAMA_CPP = Path("llama.cpp")
    OUTPUT_DIR = Path("export/models")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"Error: No se encontro modelo en {MODEL_PATH}")
        return False

    # Check if llama.cpp convert script exists
    convert_script = LLAMA_CPP / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        print(f"\nllama.cpp no encontrado. Clonando...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/ggerganov/llama.cpp.git",
            str(LLAMA_CPP)
        ], check=True)

    # Install requirements if needed
    req_file = LLAMA_CPP / "requirements.txt"
    if req_file.exists():
        print("Instalando dependencias de llama.cpp...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)
        ])

    # Convert to F16 GGUF
    output_file = OUTPUT_DIR / "thau-f16.gguf"
    print(f"\nConvirtiendo a GGUF (F16)...")
    print(f"  Entrada: {MODEL_PATH}")
    print(f"  Salida: {output_file}")

    result = subprocess.run([
        sys.executable, str(convert_script),
        str(MODEL_PATH),
        "--outfile", str(output_file),
        "--outtype", "f16"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error en conversion: {result.stderr}")
        return False

    print(f"\nGGUF creado: {output_file}")
    print(f"Tamano: {output_file.stat().st_size / 1e9:.2f} GB")

    # Also create Q4_K_M quantized version for smaller size
    output_q4 = OUTPUT_DIR / "thau-q4_k_m.gguf"
    quantize_bin = LLAMA_CPP / "build" / "bin" / "llama-quantize"

    if quantize_bin.exists():
        print(f"\nCreando version cuantizada Q4_K_M...")
        subprocess.run([
            str(quantize_bin),
            str(output_file),
            str(output_q4),
            "Q4_K_M"
        ])
        if output_q4.exists():
            print(f"GGUF Q4: {output_q4}")
            print(f"Tamano: {output_q4.stat().st_size / 1e9:.2f} GB")

    return True


def step3_create_modelfile():
    """Create Modelfile for Ollama"""
    print("\n" + "=" * 60)
    print("PASO 3: Creando Modelfile para Ollama")
    print("=" * 60)

    gguf_path = Path("export/models/thau-f16.gguf").absolute()

    if not gguf_path.exists():
        print(f"Error: No se encontro GGUF en {gguf_path}")
        return False

    modelfile_content = f'''# THAU v2.0 - Specialized AI Model
# Entrenamiento especializado con razonamiento, tool calling, y español avanzado

FROM {gguf_path}

# Template de chat (formato TinyLlama)
TEMPLATE """{{{{ if .System }}}}<|system|>
{{{{ .System }}</s>
{{{{ end }}}}{{{{ if .Prompt }}}}<|user|>
{{{{ .Prompt }}</s>
{{{{ end }}}}<|assistant|>
{{{{ .Response }}</s>
"""

# Parametros optimizados
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048

# System prompt por defecto
SYSTEM """Eres THAU, un asistente AI inteligente y servicial creado para ayudar con diversas tareas.

Capacidades:
1. Razonamiento paso a paso para problemas complejos
2. Uso de herramientas cuando es necesario
3. Generacion de imagenes mediante prompts
4. Programacion y asistencia tecnica
5. Conversacion natural en español

Para usar herramientas, responde con:
<tool_call>{{"name": "herramienta", "arguments": {{"param": "valor"}}}}</tool_call>

Herramientas disponibles:
- get_current_time: Fecha y hora actual
- web_search: Buscar en internet
- execute_python: Ejecutar codigo Python
- generate_image: Generar imagen desde texto
- read_file: Leer archivos
- list_directory: Listar directorio

Si no necesitas herramientas, responde directamente de forma clara y util."""

LICENSE """Apache 2.0 - THAU Project by Thomas & Aurora"""
'''

    modelfile_path = Path("Modelfile_thau_v2")
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)

    print(f"Modelfile creado: {modelfile_path}")
    return True


def step4_update_ollama():
    """Create/update Ollama model"""
    print("\n" + "=" * 60)
    print("PASO 4: Actualizando modelo en Ollama")
    print("=" * 60)

    modelfile = Path("Modelfile_thau_v2")
    if not modelfile.exists():
        print("Error: Modelfile no encontrado")
        return False

    # Check Ollama is running
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error: Ollama no esta corriendo")
        print("Ejecuta: ollama serve")
        return False

    # Create local model
    model_name = "thau"
    print(f"\nCreando modelo local: {model_name}")

    result = subprocess.run(
        ['ollama', 'create', model_name, '-f', str(modelfile)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False

    print(f"Modelo '{model_name}' creado exitosamente!")

    # Also create luepow/thau version for registry
    full_name = "luepow/thau"
    print(f"\nCreando modelo para registry: {full_name}")

    result = subprocess.run(
        ['ollama', 'create', full_name, '-f', str(modelfile)],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(f"Modelo '{full_name}' creado exitosamente!")

    return True


def step5_verify():
    """Verify the updated model"""
    print("\n" + "=" * 60)
    print("PASO 5: Verificando modelo")
    print("=" * 60)

    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    print("\nModelos disponibles en Ollama:")
    print(result.stdout)

    print("\nProbando modelo...")
    test_prompt = "Hola, quien eres?"

    result = subprocess.run(
        ['ollama', 'run', 'thau', test_prompt],
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode == 0:
        print(f"\nPregunta: {test_prompt}")
        print(f"Respuesta: {result.stdout[:500]}...")
        return True
    else:
        print(f"Error: {result.stderr}")
        return False


def main():
    print("=" * 60)
    print("  THAU COMPLETE UPDATE")
    print("  Merge + GGUF + Ollama")
    print("=" * 60)
    print(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    steps = [
        ("Merge LoRA adapters", step1_merge_adapters),
        ("Convert to GGUF", step2_convert_to_gguf),
        ("Create Modelfile", step3_create_modelfile),
        ("Update Ollama", step4_update_ollama),
        ("Verify", step5_verify),
    ]

    results = []
    for name, func in steps:
        try:
            success = func()
            results.append((name, success))
            if not success:
                print(f"\nError en: {name}")
                break
        except Exception as e:
            print(f"\nExcepcion en {name}: {e}")
            results.append((name, False))
            break

    # Summary
    print("\n" + "=" * 60)
    print("  RESUMEN")
    print("=" * 60)
    for name, success in results:
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {name}")

    print("\n" + "=" * 60)
    if all(r[1] for r in results):
        print("  THAU actualizado exitosamente!")
        print("\n  Para usar:")
        print("    ollama run thau")
        print("\n  Para publicar en ollama.com:")
        print("    ollama login")
        print("    ollama push luepow/thau")
    else:
        print("  Actualizacion incompleta - revisar errores")
    print("=" * 60)


if __name__ == "__main__":
    main()
