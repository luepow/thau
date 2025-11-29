#!/usr/bin/env python3
"""
THAU-Vision Training Script
============================

Complete script to train THAU-Vision from scratch.

Usage:
    python thau_vision/train_thau_vision.py --data path/to/data.jsonl
    python thau_vision/train_thau_vision.py --images_dir path/to/images --annotations annotations.json
    python thau_vision/train_thau_vision.py --demo  # Train with demo data
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_demo_data():
    """Create demo training data for testing."""
    from PIL import Image
    import random

    print("Creating demo training data...")

    demo_dir = Path("data/vision_demo")
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic images with captions
    colors = {
        "rojo": (255, 50, 50),
        "azul": (50, 50, 255),
        "verde": (50, 255, 50),
        "amarillo": (255, 255, 50),
        "naranja": (255, 150, 50),
        "morado": (150, 50, 255),
    }

    objects = {
        "circulo": "circle",
        "cuadrado": "square",
        "triangulo": "triangle",
    }

    examples = []

    for color_name, rgb in colors.items():
        for obj_name, obj_type in objects.items():
            # Create image
            img = Image.new("RGB", (224, 224), rgb)
            filename = f"{color_name}_{obj_name}.jpg"
            img.save(demo_dir / filename)

            # Caption example
            examples.append({
                "image_path": str(demo_dir / filename),
                "caption": f"Una imagen de color {color_name}.",
            })

            # VQA example
            examples.append({
                "image_path": str(demo_dir / filename),
                "question": "De que color es esta imagen?",
                "answer": f"La imagen es de color {color_name}.",
            })

            # Object example
            examples.append({
                "image_path": str(demo_dir / filename),
                "labels": [color_name, "imagen", "color"],
            })

    # Save as JSONL
    data_file = demo_dir / "training_data.jsonl"
    with open(data_file, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Created {len(examples)} demo examples in {demo_dir}")
    return str(data_file)


def main():
    parser = argparse.ArgumentParser(description="Train THAU-Vision model")

    # Data arguments
    parser.add_argument("--data", type=str, help="Path to training data (JSONL)")
    parser.add_argument("--images_dir", type=str, help="Directory containing images")
    parser.add_argument("--val_data", type=str, help="Path to validation data")
    parser.add_argument("--demo", action="store_true", help="Use demo data")

    # Model arguments
    parser.add_argument("--config", type=str, default="thau-vision-tiny",
                       choices=["thau-vision-tiny", "thau-vision-small", "thau-vision-pro"],
                       help="Model configuration")
    parser.add_argument("--vision_encoder", type=str, help="Override vision encoder")
    parser.add_argument("--checkpoint", type=str, help="Resume from checkpoint")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, help="Max training steps")
    parser.add_argument("--gradient_accum", type=int, default=4, help="Gradient accumulation steps")

    # LoRA arguments
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="checkpoints/thau_vision",
                       help="Output directory")
    parser.add_argument("--save_steps", type=int, default=100, help="Save every N steps")

    args = parser.parse_args()

    print("="*60)
    print("THAU-Vision Training")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Config: {args.config}")
    print("-"*60)

    # Get training data
    if args.demo:
        data_path = create_demo_data()
    elif args.data:
        data_path = args.data
    else:
        print("Error: Must specify --data or --demo")
        sys.exit(1)

    # Import after path setup
    from thau_vision.models import THAUVisionModel
    from thau_vision.training import VisionDataset, VisionTrainer

    # Create model
    print("\nLoading model...")
    model = THAUVisionModel(
        config_name=args.config,
        vision_encoder=args.vision_encoder,
    )

    # Create dataset
    print("\nLoading training data...")
    train_dataset = VisionDataset(
        data_path=data_path,
        images_dir=args.images_dir,
        tokenizer=model.tokenizer,
        processor=lambda img: model.vision_encoder.preprocess(img),
    )

    val_dataset = None
    if args.val_data:
        val_dataset = VisionDataset(
            data_path=args.val_data,
            images_dir=args.images_dir,
            tokenizer=model.tokenizer,
            processor=lambda img: model.vision_encoder.preprocess(img),
        )

    # Create trainer
    print("\nSetting up trainer...")
    trainer = VisionTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accum,
        learning_rate=args.lr,
        epochs=args.epochs if not args.max_steps else None,
        max_steps=args.max_steps or 10000,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        save_steps=args.save_steps,
    )

    # Resume from checkpoint
    if args.checkpoint:
        print(f"Resuming from: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # Train
    print("\nStarting training...")
    trainer.train()

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
