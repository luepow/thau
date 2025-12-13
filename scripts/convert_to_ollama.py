#!/usr/bin/env python3
"""
THAU 7B - Convertir a Ollama
Mergea LoRA, convierte a GGUF y crea modelo en Ollama
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Agregar path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))


def merge_lora_adapters():
    """Mergea los adaptadores LoRA con el modelo base"""

    print("\n" + "=" * 60)
    print("🔗 Paso 1: Mergeando adaptadores LoRA con modelo base")
    print("=" * 60)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("   Instala: pip install transformers peft torch")
        return None

    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    lora_path = Path(__file__).parent.parent / "data" / "checkpoints" / "thau-7b"
    output_path = Path(__file__).parent.parent / "data" / "models" / "thau-7b-merged"

    if not lora_path.exists():
        print(f"❌ No se encontró el modelo LoRA en {lora_path}")
        return None

    print(f"   Base model: {base_model_name}")
    print(f"   LoRA path: {lora_path}")
    print(f"   Output: {output_path}")

    # Cargar modelo base
    print("\n📥 Cargando modelo base...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )

    # Cargar y mergear LoRA
    print("🔗 Mergeando adaptadores LoRA...")
    model = PeftModel.from_pretrained(base_model, str(lora_path))
    model = model.merge_and_unload()

    # Guardar modelo mergeado
    print(f"💾 Guardando modelo mergeado en {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))

    print("✅ Modelo mergeado correctamente!")
    return output_path


def convert_to_gguf(merged_path: Path):
    """Convierte el modelo a formato GGUF usando llama.cpp"""

    print("\n" + "=" * 60)
    print("📦 Paso 2: Convirtiendo a GGUF")
    print("=" * 60)

    output_gguf = Path(__file__).parent.parent / "data" / "models" / "thau-7b.gguf"

    # Verificar si llama.cpp está instalado
    llama_cpp_path = Path.home() / "llama.cpp"
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        print("⚠️  llama.cpp no encontrado. Intentando clonar...")
        try:
            subprocess.run([
                "git", "clone", "https://github.com/ggerganov/llama.cpp.git",
                str(llama_cpp_path)
            ], check=True)
            # Instalar dependencias
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r",
                str(llama_cpp_path / "requirements.txt")
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error clonando llama.cpp: {e}")
            print("\n📝 Instrucciones manuales:")
            print("   1. git clone https://github.com/ggerganov/llama.cpp.git ~/llama.cpp")
            print("   2. cd ~/llama.cpp && pip install -r requirements.txt")
            print(f"   3. python convert_hf_to_gguf.py {merged_path} --outfile {output_gguf} --outtype f16")
            return None

    print(f"   Input: {merged_path}")
    print(f"   Output: {output_gguf}")

    # Convertir a GGUF
    print("\n🔄 Ejecutando conversión...")
    try:
        subprocess.run([
            sys.executable, str(convert_script),
            str(merged_path),
            "--outfile", str(output_gguf),
            "--outtype", "f16"
        ], check=True)
        print(f"✅ Modelo convertido a GGUF: {output_gguf}")
        return output_gguf
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en conversión: {e}")
        return None


def create_ollama_model(gguf_path: Path):
    """Crea el modelo en Ollama"""

    print("\n" + "=" * 60)
    print("🦙 Paso 3: Creando modelo en Ollama")
    print("=" * 60)

    # Crear Modelfile
    modelfile_path = Path(__file__).parent.parent / "Modelfile_thau_7b"

    modelfile_content = f'''# THAU 7B - Cognitive Learning LLM
FROM {gguf_path}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

TEMPLATE """<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
"""

SYSTEM """Eres THAU, un asistente de inteligencia artificial con capacidades de aprendizaje cognitivo.

Tus capacidades principales:
1. **Programacion**: Python, JavaScript, TypeScript, Java, Rust, Go, SQL
2. **Razonamiento**: Analisis paso a paso, resolucion de problemas
3. **SVG/Diseno**: Generacion de logos, iconos, diagramas
4. **Contabilidad**: Partida doble, estados financieros
5. **DevOps**: CI/CD, Docker, Kubernetes, Git

Cuando el usuario pida codigo, provee soluciones completas y funcionales.
Cuando requiera analisis, usa razonamiento paso a paso.
Responde en espanol a menos que se indique lo contrario."""
'''

    print(f"📝 Creando Modelfile en {modelfile_path}")
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)

    # Verificar que Ollama está instalado
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        print(f"   Ollama version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ Ollama no está instalado")
        print("   Instala desde: https://ollama.ai")
        return False

    # Crear modelo en Ollama
    print("\n🚀 Creando modelo thau-7b en Ollama...")
    try:
        subprocess.run([
            "ollama", "create", "thau-7b", "-f", str(modelfile_path)
        ], check=True)
        print("\n✅ ¡Modelo thau-7b creado exitosamente!")
        print("\n🎉 Para probar:")
        print("   ollama run thau-7b")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creando modelo: {e}")
        return False


def quick_convert_with_ollama():
    """Método alternativo: usar Ollama directamente con safetensors"""

    print("\n" + "=" * 60)
    print("🚀 Conversión rápida usando Ollama (sin llama.cpp)")
    print("=" * 60)

    lora_path = Path(__file__).parent.parent / "data" / "checkpoints" / "thau-7b"
    modelfile_path = Path(__file__).parent.parent / "Modelfile_thau_7b_lora"

    # Ollama puede cargar desde safetensors directamente con FROM
    modelfile_content = f'''# THAU 7B - Usando adaptadores LoRA
FROM Qwen/Qwen2.5-7B-Instruct
ADAPTER {lora_path}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

TEMPLATE """<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
"""

SYSTEM """Eres THAU, un asistente de inteligencia artificial con capacidades de aprendizaje cognitivo.

Tus capacidades:
1. Programacion: Python, JavaScript, TypeScript, Java, Rust, Go, SQL
2. Razonamiento: Analisis paso a paso
3. SVG/Diseno: Logos, iconos, diagramas
4. Contabilidad: Partida doble
5. DevOps: CI/CD, Docker, Kubernetes, Git

Responde en espanol. Provee codigo completo y funcional."""
'''

    print(f"📝 Creando Modelfile...")
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)

    print("🚀 Creando modelo en Ollama (esto puede tomar unos minutos)...")
    try:
        subprocess.run([
            "ollama", "create", "thau-7b", "-f", str(modelfile_path)
        ], check=True)
        print("\n✅ ¡Modelo thau-7b creado!")
        print("\n🎉 Para probar: ollama run thau-7b")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print("\nIntentando método completo (merge + GGUF)...")
        return False


def main():
    """Proceso principal de conversión"""

    print("=" * 60)
    print("🧠 THAU 7B - Conversión a Ollama")
    print("=" * 60)

    # Primero intentar método rápido
    print("\n¿Qué método prefieres?")
    print("1. Rápido (Ollama con LoRA directo) - Recomendado")
    print("2. Completo (Merge + GGUF + Ollama) - Más lento pero optimizado")

    choice = input("\nSelecciona (1/2) [1]: ").strip() or "1"

    if choice == "1":
        if quick_convert_with_ollama():
            return
        print("\nMétodo rápido falló, intentando completo...")

    # Método completo
    # Paso 1: Mergear LoRA
    merged_path = merge_lora_adapters()
    if not merged_path:
        return

    # Paso 2: Convertir a GGUF
    gguf_path = convert_to_gguf(merged_path)
    if not gguf_path:
        print("\n⚠️  La conversión a GGUF falló.")
        print("   Puedes intentar manualmente o usar el método rápido.")
        return

    # Paso 3: Crear en Ollama
    create_ollama_model(gguf_path)


if __name__ == "__main__":
    main()
