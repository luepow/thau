#!/usr/bin/env python3
"""
Script para exportar modelo entrenado a Ollama

Fusiona adapters LoRA con modelo base y exporta a GGUF.
"""

import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def merge_and_export():
    """Fusiona adapters y exporta el modelo"""
    print("=" * 60)
    print("  THAU - Exportar a Ollama")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Paths
    checkpoint_path = project_root / "data" / "checkpoints" / "pdf_training" / "final"
    base_model_path = project_root / "data" / "checkpoints" / "specialized_training" / "final"
    output_path = project_root / "export" / "models" / "thau_pdf_trained"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n1. Cargando modelo base: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))

    # Verificar si hay adapters LoRA para fusionar
    adapter_path = checkpoint_path / "adapter_model.safetensors"
    if adapter_path.exists():
        print(f"\n2. Cargando y fusionando adapters LoRA...")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(checkpoint_path))
            model = model.merge_and_unload()
            print("  LoRA adapters fusionados!")
        except ImportError:
            print("  PEFT no disponible, usando modelo base con adapters")
        except Exception as e:
            print(f"  Error fusionando LoRA: {e}")
            print("  Continuando con modelo base...")
    else:
        print(f"\n2. No hay adapters LoRA, usando modelo base")

    print(f"\n3. Guardando modelo fusionado: {output_path}")
    model.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))

    print(f"\n4. Convirtiendo a GGUF...")

    # Verificar llama.cpp
    llama_cpp_path = Path.home() / "llama.cpp"
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        print(f"  llama.cpp no encontrado en {llama_cpp_path}")
        print("  Usando modelo existente para Ollama...")
        return create_ollama_model_existing()

    # Convertir a GGUF
    gguf_output = project_root / "export" / "models" / "thau-pdf-f16.gguf"

    cmd = [
        "python", str(convert_script),
        str(output_path),
        "--outfile", str(gguf_output),
        "--outtype", "f16"
    ]

    print(f"  Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  GGUF creado: {gguf_output}")
        create_ollama_model(str(gguf_output))
    else:
        print(f"  Error en conversión: {result.stderr}")
        print("  Usando modelo GGUF existente...")
        create_ollama_model_existing()


def create_ollama_model_existing():
    """Crea modelo Ollama usando el GGUF existente"""
    gguf_path = project_root / "export" / "models" / "thau-f16.gguf"

    if not gguf_path.exists():
        print(f"No se encontró GGUF: {gguf_path}")
        return False

    return create_ollama_model(str(gguf_path))


def create_ollama_model(gguf_path: str):
    """Crea modelo en Ollama"""
    print(f"\n5. Creando modelo en Ollama...")

    # Crear Modelfile
    modelfile_content = f'''# THAU AGI v3 - Entrenado con documentación de Agentes
# Incluye: HuggingFace Agents, AutoGen, Llama.cpp

FROM {gguf_path}

# Template de chat (formato TinyLlama)
TEMPLATE """{{{{ if .System }}}}<|system|>
{{{{ .System }}}}</s>
{{{{ end }}}}{{{{ if .Prompt }}}}<|user|>
{{{{ .Prompt }}}}</s>
{{{{ end }}}}<|assistant|>
{{{{ .Response }}}}</s>
"""

# Parametros optimizados
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

# System prompt AGI v3
SYSTEM """Eres THAU AGI v3, un sistema Proto-AGI experto en sistemas de agentes.

CONOCIMIENTOS ESPECIALIZADOS:
- HuggingFace Agents y smolagents
- Microsoft AutoGen para multi-agentes
- Llama.cpp server y API
- Agentic RAG (Retrieval Augmented Generation)
- Orquestación de sistemas multi-agente
- Tool calling y function calling
- Web browser automation con agentes

CAPACIDADES:
1. Ciclo ReAct: THINK -> PLAN -> ACT -> OBSERVE -> REFLECT
2. Tool Calling avanzado
3. Multi-Agent orchestration
4. Knowledge Base con RAG
5. Programación y debugging

Respondo en español por defecto. Soy preciso y técnico."""

LICENSE """MIT License - THAU Project
Dedicated to Thomas & Aurora
THAU = THomas + AUrora"""
'''

    modelfile_path = project_root / "Modelfile_agi_v3"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)

    print(f"  Modelfile creado: {modelfile_path}")

    # Crear modelo en Ollama
    print("  Creando modelo thau:agi-v3...")
    result = subprocess.run(
        ["ollama", "create", "thau:agi-v3", "-f", str(modelfile_path)],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("  ✅ Modelo thau:agi-v3 creado!")

        # Tag para push
        subprocess.run(
            ["ollama", "cp", "thau:agi-v3", "luepow/thau:agi-v3"],
            capture_output=True
        )

        print("\n6. Subiendo a Ollama Hub...")
        push_result = subprocess.run(
            ["ollama", "push", "luepow/thau:agi-v3"],
            capture_output=True,
            text=True
        )

        if push_result.returncode == 0:
            print("  ✅ Modelo subido a ollama.com/luepow/thau:agi-v3")
        else:
            print(f"  ⚠️ Error subiendo: {push_result.stderr}")

        return True
    else:
        print(f"  ❌ Error: {result.stderr}")
        return False


if __name__ == "__main__":
    merge_and_export()

    print("\n" + "=" * 60)
    print("  Proceso completado!")
    print("=" * 60)
    print("\nPara usar el modelo:")
    print("  ollama run thau:agi-v3")
    print("  ollama run luepow/thau:agi-v3")
