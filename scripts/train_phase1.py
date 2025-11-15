"""Phase 1 training script: Basic language modeling."""

import argparse
from pathlib import Path
from loguru import logger

from config.base_config import get_config
from config.training_configs import STANDARD_TRAIN
from adapters.model_adapter import ModelAdapter
from core.training.trainer import Trainer

# Sample training dataset
from torch.utils.data import Dataset
import torch


class SimpleTextDataset(Dataset):
    """Simple text dataset for demonstration."""

    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.encodings["input_ids"][idx],
        }


def main():
    """Run Phase 1 training."""
    parser = argparse.ArgumentParser(description="Phase 1: Basic Language Modeling")
    parser.add_argument("--data-file", type=str, help="Path to training data file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    logger.info("Starting Phase 1 Training: Basic Language Modeling")

    # Load config
    config = get_config()

    # Initialize model
    logger.info("Loading model...")
    model_adapter = ModelAdapter(
        model_name=config.MODEL_NAME,
        use_quantization=config.USE_QUANTIZATION,
    )
    model_adapter.load_model()
    model_adapter.load_tokenizer()

    # Prepare for LoRA
    model_adapter.prepare_for_lora()

    # Load or create sample data
    if args.data_file and Path(args.data_file).exists():
        with open(args.data_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        logger.warning("No data file provided, using sample data")
        texts = [
            "Python is a high-level programming language.",
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are inspired by the human brain.",
            "Deep learning uses multiple layers of neural networks.",
            "Natural language processing helps computers understand text.",
        ] * 10  # Repeat for more training data

    # Create datasets
    train_size = int(0.8 * len(texts))
    train_texts = texts[:train_size]
    eval_texts = texts[train_size:]

    train_dataset = SimpleTextDataset(train_texts, model_adapter.tokenizer)
    eval_dataset = SimpleTextDataset(eval_texts, model_adapter.tokenizer)

    logger.info(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # Configure training
    train_config = STANDARD_TRAIN
    train_config.num_epochs = args.epochs
    train_config.batch_size = args.batch_size

    # Create trainer
    trainer = Trainer(
        model=model_adapter.model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=train_config,
        device_manager=model_adapter.device_manager,
    )

    # Train
    logger.info("Starting training...")
    metrics = trainer.train()

    logger.info(f"Training completed! Final metrics: {metrics}")

    # Save model
    output_dir = "./data/checkpoints/phase1"
    model_adapter.save_model(output_dir)
    logger.info(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
