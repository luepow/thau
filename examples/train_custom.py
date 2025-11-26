#!/usr/bin/env python3
"""
Custom Training Example
Shows how to train THAU with your own dataset
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from thau_trainer.own_model_manager import ThauOwnModelManager
import random


def main():
    print("="*60)
    print("🎓 THAU Custom Training Example")
    print("="*60)
    print()

    # Your custom dataset
    custom_dataset = [
        # Add your own training data here
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neurons.",
        "Training requires data, compute, and algorithms.",
        "Overfitting occurs when a model learns noise.",
        "Regularization helps prevent overfitting.",
        "Gradient descent is an optimization algorithm.",
        "Backpropagation calculates gradients efficiently.",
        "Transformers use self-attention mechanisms.",
        "BERT is a bidirectional transformer model.",
        "GPT uses unidirectional attention for generation.",
    ]

    # Initialize model for Age 0
    print("🔧 Initializing THAU Age 0...")
    manager = ThauOwnModelManager()
    manager.initialize_model(cognitive_age=0)

    print(f"✅ Model initialized")
    print(f"   Parameters: {sum(p.numel() for p in manager.model.parameters()):,}")
    print()

    # Training configuration
    num_steps = 50
    batch_size = 3
    learning_rate = 5e-4

    print(f"📊 Training Configuration:")
    print(f"   Steps: {num_steps}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Dataset size: {len(custom_dataset)}")
    print()

    # Training loop
    print("🚀 Starting training...\n")

    for step in range(num_steps):
        # Sample random batch
        batch = random.sample(custom_dataset, min(batch_size, len(custom_dataset)))

        # Train step
        result = manager.train_step(
            texts=batch,
            learning_rate=learning_rate,
            gradient_accumulation_steps=4
        )

        # Print progress
        if (step + 1) % 10 == 0 or step == 0:
            print(f"Step {step+1}/{num_steps} | "
                  f"Loss: {result['loss']:.4f} | "
                  f"Perplexity: {result['perplexity']:.2f}")

    print()
    print("✅ Training completed!")

    # Save checkpoint
    checkpoint_name = "custom_trained_age_0"
    manager.save_checkpoint(checkpoint_name=checkpoint_name)
    print(f"💾 Model saved: {checkpoint_name}.pt")

    # Test generation
    print()
    print("🧪 Testing generation...")
    test_prompts = [
        "Machine learning",
        "Neural networks",
        "Transformers"
    ]

    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        print(f"🤖 THAU: ", end="")
        try:
            response = manager.generate_text(
                prompt=prompt,
                max_new_tokens=50,
                temperature=0.7
            )
            print(response)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
