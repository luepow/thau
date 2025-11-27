#!/usr/bin/env python3
"""
Simple THAU to Ollama Export
"""

import sys
import os
import torch
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from thau_trainer.own_model_manager import ThauOwnModelManager


def export_for_ollama(checkpoint_path: str, age: int):
    """Export THAU to format compatible with Ollama"""

    print("="*80)
    print(f"📦 THAU Age {age} → Ollama Export")
    print("="*80)
    print(f"📁 Checkpoint: {checkpoint_path}\n")

    # Initialize model
    manager = ThauOwnModelManager()
    manager.initialize_model(cognitive_age=age)

    # Load checkpoint
    print(f"Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        manager.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model_state_dict")
    else:
        print(f"❌ No model_state_dict found in checkpoint")
        return False

    # Save in HuggingFace format
    output_dir = f"thau_age_{age}_hf"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📝 Saving to: {output_dir}/")

    # Save model weights
    torch.save(manager.model.state_dict(), f"{output_dir}/pytorch_model.bin")
    print(f"✅ Saved: pytorch_model.bin")

    # Save tokenizer
    manager.tokenizer.tokenizer.save_pretrained(output_dir)
    print(f"✅ Saved: tokenizer files")

    # Create config.json
    config = {
        "model_type": "thau",
        "vocab_size": manager.tokenizer.vocab_size,
        "hidden_size": manager.model_config.d_model,
        "num_hidden_layers": manager.model_config.n_layers,
        "num_attention_heads": manager.model_config.n_heads,
        "intermediate_size": manager.model_config.d_ff,
        "max_position_embeddings": manager.model_config.max_seq_length,
        "architectures": ["TinyLLM"],
    }
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"✅ Saved: config.json")

    # Create Modelfile for Ollama
    modelfile = f"""FROM ./{output_dir}

# THAU Age {age} - {sum(p.numel() for p in manager.model.parameters()):,} parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

SYSTEM \"\"\"Eres THAU (Transformative Holistic Autonomous Unit), un asistente de IA con {sum(p.numel() for p in manager.model.parameters()):,} parámetros en Age {age}. Estás especializado en programación y conceptos técnicos.\"\"\"

TEMPLATE \"\"\"{{{{ .System }}}}

{{{{ .Prompt }}}}\"\"\"
"""

    modelfile_path = f"Modelfile_age_{age}"
    with open(modelfile_path, "w") as f:
        f.write(modelfile)
    print(f"✅ Saved: {modelfile_path}")

    # Instructions
    print(f"\n{'='*80}")
    print(f"📋 NEXT STEPS")
    print(f"{'='*80}\n")
    print(f"1. Create Ollama model:")
    print(f"   ollama create thau-age-{age} -f {modelfile_path}")
    print()
    print(f"2. Test the model:")
    print(f"   ollama run thau-age-{age} '¿Qué es Python?'")
    print()
    print(f"3. List your models:")
    print(f"   ollama list")
    print()

    return True


if __name__ == "__main__":
    checkpoint = "data/model_checkpoints/age_3_final.pt"
    age = 3

    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]
    if len(sys.argv) > 2:
        age = int(sys.argv[2])

    success = export_for_ollama(checkpoint, age)

    if success:
        print("✅ Export completed!\n")
    else:
        print("❌ Export failed!\n")
        sys.exit(1)
