#!/usr/bin/env python3
"""
Simple Chat Example with THAU
Demonstrates basic chat interaction with a trained THAU model
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from thau_trainer.own_model_manager import ThauOwnModelManager
import torch


def main():
    print("="*60)
    print("🧠 THAU Simple Chat Example")
    print("="*60)
    print()

    # Load trained model
    print("📂 Loading THAU Age 3 model...")
    manager = ThauOwnModelManager()
    manager.initialize_model(cognitive_age=3)

    # Load checkpoint
    checkpoint_path = "data/model_checkpoints/age_3_final.pt"
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        manager.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from {checkpoint_path}")
    else:
        print(f"⚠️  No checkpoint found at {checkpoint_path}")
        print(f"   Run: python train_ages_simple.py")
        return

    print(f"🧠 Parameters: {sum(p.numel() for p in manager.model.parameters()):,}")
    print()

    # Interactive chat loop
    print("💬 Chat with THAU (type 'exit' to quit)")
    print("-"*60)

    while True:
        # Get user input
        try:
            user_input = input("\n👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nGoodbye! 👋")
            break

        # Generate response
        print("🤖 THAU: ", end="", flush=True)
        try:
            response = manager.generate_text(
                prompt=user_input,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9
            )
            print(response)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
