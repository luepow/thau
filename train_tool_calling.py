#!/usr/bin/env python3
"""
Train THAU on Tool Calling
Teaches THAU when and how to use the image generation tool
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import torch
from loguru import logger

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from config.base_config import get_config
from adapters.model_adapter import ModelAdapter
from core.training.incremental_trainer import IncrementalTrainer

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


class ToolCallingTrainer:
    """Trainer for THAU tool calling capabilities"""

    def __init__(self, dataset_path: str = "data/datasets/tool_calling_dataset.json"):
        self.config = get_config()
        self.dataset_path = Path(dataset_path)
        self.dataset = self._load_dataset()

        # Initialize model and trainer
        logger.info("🤖 Inicializando THAU para tool calling training...")
        self.model_adapter = ModelAdapter(
            model_name=self.config.MODEL_NAME,
            use_quantization=self.config.USE_QUANTIZATION,
        )
        self.model_adapter.load_model()
        self.model_adapter.load_tokenizer()

        # Prepare for LoRA fine-tuning
        from config.model_configs import LORA_DEFAULT
        self.model_adapter.prepare_for_lora(LORA_DEFAULT)

        self.trainer = IncrementalTrainer(
            model_adapter=self.model_adapter,
            config=self.config
        )

        logger.info(f"✅ Modelo cargado: {self.config.MODEL_NAME}")
        logger.info(f"✅ Dataset cargado: {len(self.dataset['examples'])} ejemplos")

    def _load_dataset(self) -> Dict:
        """Load the tool calling dataset"""
        logger.info(f"📂 Cargando dataset de {self.dataset_path}...")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        return dataset

    def _format_example_as_instruction(self, example: Dict) -> str:
        """
        Format a single example as instruction-following text

        Format:
        ### Instrucción:
        Eres THAU, un asistente con capacidad de generar imágenes.
        Cuando el usuario pida una imagen, usa el formato: <TOOL:generate_image>{"prompt": "description"}</TOOL>

        ### Usuario:
        {user message}

        ### Asistente:
        {assistant response}
        """
        instruction = (
            "### Instrucción:\n"
            "Eres THAU, un asistente de IA con capacidad de generar imágenes. "
            "Cuando el usuario solicite una imagen o visualización, usa el formato: "
            "<TOOL:generate_image>{\"prompt\": \"descripción en inglés\"}</TOOL>\n"
            "Para conversación normal, responde directamente sin usar herramientas.\n\n"
            f"### Usuario:\n{example['user']}\n\n"
            f"### Asistente:\n{example['assistant']}"
        )

        return instruction

    def prepare_training_data(self) -> List[str]:
        """Prepare all examples for training"""
        logger.info("📝 Preparando datos de entrenamiento...")

        training_texts = []

        for example in self.dataset['examples']:
            formatted = self._format_example_as_instruction(example)
            training_texts.append(formatted)

        # Show samples
        logger.info(f"\n{'='*80}")
        logger.info("📋 Ejemplo de formato de entrenamiento:")
        logger.info(f"{'='*80}")
        logger.info(training_texts[0])
        logger.info(f"{'='*80}\n")

        return training_texts

    def train(
        self,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        batch_size: int = 4,
    ):
        """
        Train THAU on tool calling using batch learning

        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate for fine-tuning
            batch_size: Batch size for training
        """
        logger.info("🚀 Iniciando entrenamiento de tool calling...")
        logger.info(f"   Epochs: {num_epochs}")
        logger.info(f"   Learning rate: {learning_rate}")
        logger.info(f"   Batch size: {batch_size}")

        # Prepare data in batch format
        logger.info("📝 Preparando datos de entrenamiento...")

        interactions = []
        for example in self.dataset['examples']:
            formatted_text = self._format_example_as_instruction(example)
            parts = formatted_text.split("### Asistente:")

            interactions.append({
                "prompt": parts[0].strip(),
                "response": parts[1].strip() if len(parts) > 1 else ""
            })

        # Show example
        logger.info(f"\n{'='*80}")
        logger.info("📋 Ejemplo de formato de entrenamiento:")
        logger.info(f"{'='*80}")
        logger.info(f"Prompt: {interactions[0]['prompt'][:200]}...")
        logger.info(f"Response: {interactions[0]['response'][:200]}...")
        logger.info(f"{'='*80}\n")

        logger.info(f"📊 Total de ejemplos: {len(interactions)}")

        # Update config learning rate
        original_lr = self.config.LEARNING_RATE
        self.config.LEARNING_RATE = learning_rate

        # Train using batch learning
        result = self.trainer.learn_from_batch(
            interactions=interactions,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

        # Restore original LR
        self.config.LEARNING_RATE = original_lr

        # Final checkpoint is already saved by learn_from_batch
        # Just rename it
        import shutil
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        old_checkpoint = Path("./data/checkpoints/incremental") / f"incremental_{timestamp}"

        # Find the latest checkpoint
        checkpoint_dir = Path("./data/checkpoints/incremental")
        checkpoints = sorted(checkpoint_dir.glob("incremental_*"))

        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            final_checkpoint_path = checkpoint_dir / "tool_calling_final"

            # Remove if exists
            if final_checkpoint_path.exists():
                shutil.rmtree(final_checkpoint_path)

            # Rename
            shutil.copytree(latest_checkpoint, final_checkpoint_path)

            logger.info(f"\n{'='*80}")
            logger.info(f"🎉 Entrenamiento completado!")
            logger.info(f"💾 Modelo final guardado en: {final_checkpoint_path}")
            logger.info(f"📊 Loss promedio: {result['avg_loss']:.4f}")
            logger.info(f"{'='*80}\n")

            return str(final_checkpoint_path)
        else:
            logger.warning("No se encontró checkpoint guardado")
            return None

    def test_model(self, test_prompts: List[str] = None):
        """Test the trained model on sample prompts"""
        if test_prompts is None:
            test_prompts = [
                "Genera una imagen de un perro en el espacio",
                "¿Qué es machine learning?",
                "Muéstrame una imagen de un atardecer",
                "Explica qué es una API REST",
            ]

        logger.info(f"\n{'='*80}")
        logger.info("🧪 Probando modelo entrenado")
        logger.info(f"{'='*80}\n")

        from core.inference.generator import TextGenerator

        generator = TextGenerator(
            model_adapter=self.model_adapter,
            config=self.config
        )

        for prompt in test_prompts:
            logger.info(f"👤 Usuario: {prompt}")

            # Format as instruction
            full_prompt = (
                "### Instrucción:\n"
                "Eres THAU, un asistente de IA con capacidad de generar imágenes. "
                "Cuando el usuario solicite una imagen o visualización, usa el formato: "
                "<TOOL:generate_image>{\"prompt\": \"descripción en inglés\"}</TOOL>\n"
                "Para conversación normal, responde directamente sin usar herramientas.\n\n"
                f"### Usuario:\n{prompt}\n\n"
                "### Asistente:\n"
            )

            # Generate response
            response = generator.generate(
                prompt=full_prompt,
                max_length=256,
                temperature=0.7,
                do_sample=True,
            )

            logger.info(f"🤖 THAU: {response}")
            logger.info("")


def main():
    """Main training script"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train THAU on tool calling for image generation"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training (default: 4)"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test the model, skip training"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/datasets/tool_calling_dataset.json",
        help="Path to tool calling dataset"
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = ToolCallingTrainer(dataset_path=args.dataset)

    if args.test_only:
        # Only test
        logger.info("🧪 Modo test - solo probando modelo")
        trainer.test_model()
    else:
        # Train
        checkpoint = trainer.train(
            num_epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )

        # Test after training
        logger.info("\n🧪 Probando modelo recién entrenado...")
        trainer.test_model()

    logger.info("\n✅ Script completado!")


if __name__ == "__main__":
    main()
