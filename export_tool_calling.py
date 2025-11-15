#!/usr/bin/env python3
"""
Export Tool Calling model with merged LoRA adapters
"""

import sys
from pathlib import Path
import torch
from loguru import logger

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from adapters.model_adapter import ModelAdapter
from config.base_config import get_config

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


def export_merged_model(
    checkpoint_path: str = "./data/checkpoints/incremental/tool_calling_final",
    output_path: str = "./export/merged/thau-tool-calling",
):
    """
    Export model with merged LoRA adapters

    Args:
        checkpoint_path: Path to LoRA checkpoint
        output_path: Where to save merged model
    """
    logger.info("🚀 Exportando THAU con Tool Calling")
    logger.info(f"   Checkpoint: {checkpoint_path}")
    logger.info(f"   Output: {output_path}")

    config = get_config()

    # Load base model
    logger.info("\n📥 Cargando modelo base...")
    model_adapter = ModelAdapter(
        model_name=config.MODEL_NAME,
        use_quantization=False,  # No quantization for export
    )
    model_adapter.load_model()
    model_adapter.load_tokenizer()

    # Load LoRA adapters from checkpoint
    logger.info(f"\n🔧 Cargando adaptadores LoRA desde: {checkpoint_path}")

    from peft import PeftModel

    # Load PEFT model with LoRA adapters
    try:
        model_with_lora = PeftModel.from_pretrained(
            model_adapter.model,
            checkpoint_path,
        )
        logger.info("✅ Adaptadores LoRA cargados")

        # Merge and unload adapters
        logger.info("\n🔄 Fusionando adaptadores LoRA con modelo base...")
        merged_model = model_with_lora.merge_and_unload()
        logger.info("✅ Adaptadores fusionados")

    except Exception as e:
        logger.error(f"❌ Error cargando LoRA: {e}")
        logger.info("ℹ️  Usando modelo base sin adaptadores")
        merged_model = model_adapter.model

    # Save merged model
    logger.info(f"\n💾 Guardando modelo fusionado en: {output_path}")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    merged_model.save_pretrained(output_dir)
    logger.info("✅ Modelo guardado")

    # Save tokenizer
    model_adapter.tokenizer.save_pretrained(output_dir)
    logger.info("✅ Tokenizer guardado")

    # Save config
    if hasattr(merged_model, "config"):
        merged_model.config.save_pretrained(output_dir)
        logger.info("✅ Config guardado")

    logger.info(f"\n{'='*80}")
    logger.info("🎉 Exportación completada!")
    logger.info(f"📁 Modelo fusionado en: {output_dir}")
    logger.info(f"{'='*80}\n")

    # Show next steps
    logger.info("📋 Próximos pasos:")
    logger.info(f"   1. Convertir a GGUF:")
    logger.info(f"      python convert_hf_to_gguf.py {output_dir} --outfile thau-tool-calling.gguf")
    logger.info(f"   2. Importar a Ollama:")
    logger.info(f"      ollama create thau-tool-calling -f Modelfile-tool-calling")

    return str(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export THAU tool calling model with merged LoRA adapters"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./data/checkpoints/incremental/tool_calling_final",
        help="Path to LoRA checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./export/merged/thau-tool-calling",
        help="Output directory for merged model"
    )

    args = parser.parse_args()

    export_merged_model(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )
