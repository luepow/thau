#!/usr/bin/env python3
"""
Export THAU Model to GGUF Format for Ollama

This script converts a trained THAU PyTorch model to GGUF format
that can be imported into Ollama for testing and deployment.

Usage:
    python export_to_gguf.py --checkpoint data/checkpoints/age_0_final.pt --output thau_age_0.gguf
"""

import torch
import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thau_trainer.own_model_manager import ThauOwnModelManager
from core.tokenizer.tokenizer import Tokenizer
from adapters.device_manager import get_device_manager


def export_to_gguf(checkpoint_path: str, output_path: str, age: int = 0):
    """
    Export THAU model to GGUF format for Ollama.

    Args:
        checkpoint_path: Path to the THAU checkpoint file (.pt)
        output_path: Path to save the GGUF file
        age: Model age (0-15) to determine architecture
    """

    print(f"══════════════════════════════════════════════════════════════")
    print(f"📦 THAU Model → GGUF Export")
    print(f"══════════════════════════════════════════════════════════════")
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"💾 Output: {output_path}")
    print(f"🧠 Age: {age}")
    print()

    # Check if llama.cpp tools are available
    try:
        import subprocess
        result = subprocess.run(['which', 'convert.py'], capture_output=True, text=True)
        convert_tool = result.stdout.strip()

        if not convert_tool:
            print(f"❌ ERROR: convert.py from llama.cpp not found in PATH")
            print()
            print(f"📚 Installation Instructions:")
            print(f"   1. Clone llama.cpp:")
            print(f"      git clone https://github.com/ggerganov/llama.cpp")
            print(f"   2. Build llama.cpp:")
            print(f"      cd llama.cpp && make")
            print(f"   3. Add to PATH:")
            print(f"      export PATH=$PATH:$(pwd)/llama.cpp")
            print()
            print(f"⚠️  Alternative: Use HuggingFace transformers to export")
            print(f"   We'll try an alternative method...")
            convert_tool = None
    except Exception as e:
        print(f"⚠️  Warning: Could not check for convert.py: {e}")
        convert_tool = None

    # Load model using ThauOwnModelManager
    print(f"🔧 Loading THAU model...")

    manager = ThauOwnModelManager()
    manager.initialize_model(cognitive_age=age)

    # Load checkpoint
    checkpoint_name = Path(checkpoint_path).stem
    try:
        manager.load_checkpoint(checkpoint_name)
        print(f"✅ Checkpoint loaded: {checkpoint_path}")
    except Exception as e:
        print(f"❌ ERROR loading checkpoint: {e}")
        return False

    model = manager.model
    tokenizer = manager.tokenizer

    print(f"✅ Model loaded successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {manager.device}")
    print()

    # Method 1: Convert to HuggingFace format first (most compatible)
    print(f"🔄 Converting to HuggingFace format...")

    hf_output_dir = output_path.replace('.gguf', '_hf')
    os.makedirs(hf_output_dir, exist_ok=True)

    # Save in HuggingFace format
    try:
        model.save_pretrained(hf_output_dir)
        tokenizer.tokenizer.save_pretrained(hf_output_dir)
    except AttributeError:
        # Fallback: save as PyTorch checkpoint
        torch.save(model.state_dict(), os.path.join(hf_output_dir, "pytorch_model.bin"))
        tokenizer.tokenizer.save_pretrained(hf_output_dir)
        # Create config.json
        import json
        config = {
            "model_type": "thau",
            "vocab_size": tokenizer.vocab_size,
            "hidden_size": manager.model_config.d_model,
            "num_hidden_layers": manager.model_config.n_layers,
            "num_attention_heads": manager.model_config.n_heads,
            "intermediate_size": manager.model_config.d_ff,
            "max_position_embeddings": manager.model_config.max_seq_length,
        }
        with open(os.path.join(hf_output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    print(f"✅ HuggingFace model saved to: {hf_output_dir}")
    print()

    # Create Modelfile for Ollama
    modelfile_path = output_path.replace('.gguf', '.Modelfile')

    modelfile_content = f"""FROM {hf_output_dir}

# THAU Model Age {age}
# Temperature and other parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

# System message for THAU
SYSTEM You are THAU (Transformative Holistic Autonomous Unit), an AI assistant with cognitive growth capabilities. You are currently at age {age}, with {sum(p.numel() for p in model.parameters()):,} parameters.

# Template
TEMPLATE \"\"\"{{ .System }}

{{ .Prompt }}
\"\"\"
"""

    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)

    print(f"✅ Modelfile created: {modelfile_path}")
    print()

    # Instructions for Ollama
    print(f"══════════════════════════════════════════════════════════════")
    print(f"📋 NEXT STEPS - Import to Ollama")
    print(f"══════════════════════════════════════════════════════════════")
    print()
    print(f"1️⃣  Create Ollama model from HuggingFace format:")
    print(f"   ollama create thau-age-{age} -f {modelfile_path}")
    print()
    print(f"2️⃣  Test the model:")
    print(f"   ollama run thau-age-{age} \"¿Qué es Python?\"")
    print()
    print(f"3️⃣  List your models:")
    print(f"   ollama list")
    print()
    print(f"4️⃣  Use in THAU CLI:")
    print(f"   thau code")
    print(f"   /model switch ollama:thau-age-{age}")
    print()
    print(f"══════════════════════════════════════════════════════════════")
    print(f"📚 ALTERNATIVE: Convert to GGUF manually")
    print(f"══════════════════════════════════════════════════════════════")
    print()
    print(f"If you want a pure GGUF file (for advanced use):")
    print()
    print(f"1️⃣  Clone llama.cpp:")
    print(f"   git clone https://github.com/ggerganov/llama.cpp")
    print(f"   cd llama.cpp && make")
    print()
    print(f"2️⃣  Convert HuggingFace to GGUF:")
    print(f"   python3 convert.py {hf_output_dir} --outfile {output_path} --outtype f16")
    print()
    print(f"3️⃣  Quantize (optional, for smaller size):")
    print(f"   ./quantize {output_path} {output_path.replace('.gguf', '_q4.gguf')} Q4_K_M")
    print()
    print(f"══════════════════════════════════════════════════════════════")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Export THAU model to GGUF format for Ollama',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export age 0 model
  python export_to_gguf.py --checkpoint data/checkpoints/age_0_final.pt --output thau_age_0.gguf --age 0

  # Export age 5 model
  python export_to_gguf.py --checkpoint data/checkpoints/age_5_final.pt --output thau_age_5.gguf --age 5

  # Export with custom name
  python export_to_gguf.py --checkpoint my_checkpoint.pt --output my_model.gguf --age 0
        """
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to THAU checkpoint file (.pt)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for GGUF file'
    )

    parser.add_argument(
        '--age',
        type=int,
        default=0,
        help='Model age (0-15), default: 0'
    )

    args = parser.parse_args()

    # Validate age
    if not 0 <= args.age <= 15:
        print(f"❌ ERROR: Age must be between 0 and 15, got {args.age}")
        return 1

    # Export
    success = export_to_gguf(args.checkpoint, args.output, args.age)

    if success:
        print()
        print(f"✅ Export completed successfully!")
        return 0
    else:
        print()
        print(f"❌ Export failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
