"""Train THAU to specialize in a programming language syntax."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.base_config import get_config
from adapters.model_adapter import ModelAdapter
from core.training.incremental_trainer import IncrementalTrainer
from config.model_configs import LoRAConfig


class LanguageSpecializationTrainer:
    """Trainer for programming language syntax specialization."""

    def __init__(
        self,
        language: str = "python",
        output_dir: str = "./data/checkpoints/specialized",
    ):
        """Initialize specialized trainer.

        Args:
            language: Programming language name
            output_dir: Directory for saving checkpoints
        """
        self.language = language
        self.output_dir = Path(output_dir) / language
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = get_config()
        self.training_history = []

        logger.info(f"Initializing {language} specialization trainer")

    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load Q&A dataset from JSONL file.

        Args:
            dataset_path: Path to JSONL dataset

        Returns:
            List of Q&A dictionaries
        """
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Normalize to prompt/response format
                    data.append({
                        "prompt": item.get("instruction", item.get("prompt", "")),
                        "response": item.get("output", item.get("response", "")),
                        "topic": item.get("topic", "general"),
                    })

        logger.info(f"Loaded {len(data)} training examples")
        return data

    def load_multiple_datasets(self, dataset_paths: List[str]) -> List[Dict]:
        """Load multiple datasets and combine them.

        Args:
            dataset_paths: List of paths to JSONL files

        Returns:
            Combined list of Q&A dictionaries
        """
        all_data = []
        for path in dataset_paths:
            if Path(path).exists():
                data = self.load_dataset(path)
                all_data.extend(data)
                logger.info(f"Added {len(data)} examples from {path}")
            else:
                logger.warning(f"Dataset not found: {path}")

        logger.info(f"Total training examples: {len(all_data)}")
        return all_data

    def prepare_model(self) -> IncrementalTrainer:
        """Prepare model with LoRA for training.

        Returns:
            IncrementalTrainer instance
        """
        logger.info("Loading model...")

        model_adapter = ModelAdapter(
            model_name=self.config.MODEL_NAME,
            use_quantization=self.config.USE_QUANTIZATION,
        )
        model_adapter.load_model()
        model_adapter.load_tokenizer()

        # Prepare for LoRA with custom config for language specialization
        lora_config = LoRAConfig(
            r=16,  # Higher rank for more capacity
            lora_alpha=32,
            lora_dropout=0.05,
        )

        model_adapter.prepare_for_lora(lora_config)

        trainer = IncrementalTrainer(model_adapter=model_adapter)
        logger.info("Model ready for training")

        return trainer

    def train(
        self,
        dataset_paths: List[str],
        num_epochs: int = 5,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        save_every: int = 100,
    ) -> Dict:
        """Train on programming language syntax.

        Args:
            dataset_paths: Paths to training datasets
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            save_every: Save checkpoint every N steps

        Returns:
            Training metrics
        """
        # Load all datasets
        training_data = self.load_multiple_datasets(dataset_paths)

        if not training_data:
            raise ValueError("No training data loaded!")

        # Prepare model
        trainer = self.prepare_model()

        # Train
        logger.info(f"Starting {self.language} specialization training...")
        logger.info(f"Examples: {len(training_data)}, Epochs: {num_epochs}, Batch: {batch_size}")

        # Format for batch training
        interactions = [
            {"prompt": item["prompt"], "response": item["response"]}
            for item in training_data
        ]

        metrics = trainer.learn_from_batch(
            interactions=interactions,
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

        # Save specialized checkpoint
        checkpoint_name = f"{self.language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trainer.save_incremental_checkpoint(f"specialized/{checkpoint_name}")

        # Save training info
        training_info = {
            "language": self.language,
            "timestamp": datetime.now().isoformat(),
            "num_examples": len(training_data),
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "metrics": metrics,
            "datasets": dataset_paths,
        }

        info_path = self.output_dir / f"{checkpoint_name}_info.json"
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)

        logger.info(f"Training complete! Checkpoint: {checkpoint_name}")
        logger.info(f"Final metrics: {metrics}")

        return metrics

    def train_incremental(
        self,
        dataset_path: str,
        steps_per_example: int = 5,
    ) -> Dict:
        """Train incrementally on each example.

        Better for small datasets or when you want more control.

        Args:
            dataset_path: Path to dataset
            steps_per_example: Training steps per example

        Returns:
            Training metrics
        """
        training_data = self.load_dataset(dataset_path)
        trainer = self.prepare_model()

        all_losses = []

        for i, item in enumerate(training_data):
            logger.info(f"Training on example {i+1}/{len(training_data)}: {item['topic']}")

            metrics = trainer.learn_from_interaction(
                prompt=item["prompt"],
                response=item["response"],
                num_steps=steps_per_example,
            )

            all_losses.append(metrics["avg_loss"])

            # Progress logging
            if (i + 1) % 10 == 0:
                avg = sum(all_losses[-10:]) / 10
                logger.info(f"Progress: {i+1}/{len(training_data)}, Recent avg loss: {avg:.4f}")

        # Save checkpoint
        checkpoint_name = f"{self.language}_incremental_{datetime.now().strftime('%Y%m%d')}"
        trainer.save_incremental_checkpoint(f"specialized/{checkpoint_name}")

        return {
            "avg_loss": sum(all_losses) / len(all_losses),
            "final_loss": all_losses[-1] if all_losses else None,
            "num_examples": len(training_data),
        }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train THAU on programming language syntax"
    )
    parser.add_argument(
        "--language",
        default="python",
        help="Programming language (python, javascript, rust, etc.)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Paths to JSONL training datasets"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Use incremental training (one example at a time)"
    )
    args = parser.parse_args()

    trainer = LanguageSpecializationTrainer(language=args.language)

    if args.incremental:
        # Incremental training
        for dataset in args.datasets:
            metrics = trainer.train_incremental(dataset)
            print(f"\n=== Incremental Training Complete ===")
            print(f"Dataset: {dataset}")
            print(f"Avg Loss: {metrics['avg_loss']:.4f}")
    else:
        # Batch training
        metrics = trainer.train(
            dataset_paths=args.datasets,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

        print(f"\n=== Training Complete ===")
        print(f"Language: {args.language}")
        print(f"Final Loss: {metrics['avg_loss']:.4f}")
        print(f"Epochs: {args.epochs}")


if __name__ == "__main__":
    main()
