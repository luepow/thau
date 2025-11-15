"""Phase 2 training script: Instruction fine-tuning."""

import argparse
from loguru import logger

from config.base_config import get_config
from adapters.model_adapter import ModelAdapter
from core.training.incremental_trainer import IncrementalTrainer


# Sample instruction dataset
INSTRUCTION_DATA = [
    {
        "prompt": "What is Python?",
        "response": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, machine learning, and automation.",
    },
    {
        "prompt": "Explain machine learning in simple terms.",
        "response": "Machine learning is a way for computers to learn from data without being explicitly programmed. Instead of following fixed rules, the computer finds patterns in examples and uses them to make predictions or decisions.",
    },
    {
        "prompt": "What is the difference between AI and ML?",
        "response": "AI (Artificial Intelligence) is the broader concept of machines being able to carry out tasks in a smart way. ML (Machine Learning) is a subset of AI that focuses on the idea that machines can learn from data and improve from experience.",
    },
    {
        "prompt": "How do neural networks work?",
        "response": "Neural networks are inspired by the human brain. They consist of layers of interconnected nodes (neurons) that process information. Each connection has a weight that adjusts as learning proceeds, allowing the network to recognize patterns and make decisions.",
    },
    {
        "prompt": "What is natural language processing?",
        "response": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and generate human language. It's used in applications like chatbots, translation services, and text analysis.",
    },
]


def main():
    """Run Phase 2 training."""
    parser = argparse.ArgumentParser(description="Phase 2: Instruction Fine-tuning")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    logger.info("Starting Phase 2 Training: Instruction Fine-tuning")

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

    # Initialize incremental trainer
    trainer = IncrementalTrainer(model_adapter=model_adapter)

    # Train on instruction data
    logger.info("Training on instruction dataset...")
    metrics = trainer.learn_from_batch(
        interactions=INSTRUCTION_DATA,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    logger.info(f"Training completed! Metrics: {metrics}")

    # Save checkpoint
    trainer.save_incremental_checkpoint("phase2_complete")
    logger.info("Phase 2 training complete!")


if __name__ == "__main__":
    main()
