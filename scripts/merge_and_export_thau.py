#!/usr/bin/env python3
"""
Merge THAU 7B LoRA adapters with base model and export for Ollama

Este script:
1. Carga el modelo base (Qwen2.5-7B-Instruct)
2. Aplica los adaptadores LoRA del checkpoint
3. Mergea los pesos
4. Guarda el modelo completo en formato HuggingFace
5. Instrucciones para convertir a GGUF
"""

import os
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuración
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CHECKPOINT_PATH = "data/checkpoints/thau-7b/checkpoint-400"
OUTPUT_PATH = "data/models/thau-7b-merged"


def merge_lora_adapters():
    """Mergea los adaptadores LoRA con el modelo base"""

    print("=" * 60)
    print("🔄 THAU 7B - Merge LoRA Adapters")
    print("=" * 60)

    # Detectar dispositivo
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"\n💻 Usando Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"\n💻 Usando CUDA GPU")
    else:
        device = "cpu"
        print(f"\n💻 Usando CPU")

    # Verificar que existe el checkpoint
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        print(f"❌ Error: No se encontró el checkpoint en {CHECKPOINT_PATH}")
        sys.exit(1)

    print(f"\n📂 Checkpoint: {CHECKPOINT_PATH}")
    print(f"📦 Modelo base: {BASE_MODEL}")
    print(f"💾 Output: {OUTPUT_PATH}")

    # Crear directorio de salida
    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)

    # Cargar modelo base
    print(f"\n📥 Cargando modelo base: {BASE_MODEL}")
    print("   (Esto puede tomar unos minutos...)")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Cargar tokenizer
    print("📝 Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    # Cargar adaptadores LoRA
    print(f"\n🔗 Cargando adaptadores LoRA desde checkpoint-400...")
    model = PeftModel.from_pretrained(
        base_model,
        str(checkpoint_path),
        torch_dtype=torch.float16,
    )

    # Mergear adaptadores
    print("\n🔄 Mergeando adaptadores LoRA con modelo base...")
    merged_model = model.merge_and_unload()

    # Guardar modelo mergeado
    print(f"\n💾 Guardando modelo mergeado en {OUTPUT_PATH}...")
    merged_model.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size="4GB",
    )

    # Guardar tokenizer
    tokenizer.save_pretrained(str(output_path))

    print("\n" + "=" * 60)
    print("✅ ¡Modelo mergeado exitosamente!")
    print("=" * 60)

    print(f"""
📂 Modelo guardado en: {OUTPUT_PATH}

📦 Próximos pasos para usar con Ollama:

1. Convertir a GGUF usando llama.cpp:

   # Clonar llama.cpp si no lo tienes
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp

   # Instalar dependencias Python
   pip install -r requirements.txt

   # Convertir a GGUF (Q4_K_M es un buen balance calidad/tamaño)
   python convert_hf_to_gguf.py {os.path.abspath(OUTPUT_PATH)} \\
       --outfile thau-7b-q4_k_m.gguf \\
       --outtype q4_k_m

2. Crear Modelfile para Ollama:

   cat > Modelfile_thau_7b << 'EOF'
FROM ./thau-7b-q4_k_m.gguf

TEMPLATE \"\"\"<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
\"\"\"

PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

SYSTEM \"\"\"Eres THAU (Thinking Human-like Artificial Understanding), un asistente de IA especializado en razonamiento cognitivo, programación y resolución de problemas. Respondes de manera clara, estructurada y con razonamiento paso a paso cuando es necesario.\"\"\"
EOF

3. Crear el modelo en Ollama:
   ollama create thau-7b -f Modelfile_thau_7b

4. Usar el modelo:
   ollama run thau-7b
""")

    return str(output_path)


if __name__ == "__main__":
    merge_lora_adapters()
