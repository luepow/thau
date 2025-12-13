"""Train THAU with TinyLlama chat format."""

import json
import sys
from pathlib import Path
from datetime import datetime
import torch
from loguru import logger

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.base_config import get_config
from adapters.model_adapter import ModelAdapter
from config.model_configs import LoRAConfig


def train_with_chat_format(
    dataset_path: str,
    output_name: str = "thau_v2",
    num_epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
):
    """Train model using apply_chat_template for correct format."""

    print("=" * 60, flush=True)
    print("  THAU v2.0 - Training with Chat Format", flush=True)
    print("=" * 60, flush=True)

    config = get_config()

    # Load model first to get tokenizer for apply_chat_template
    print("\n[1] Loading model...", flush=True)
    adapter = ModelAdapter(
        model_name=config.MODEL_NAME,
        use_quantization=config.USE_QUANTIZATION,
    )
    adapter.load_model()
    adapter.load_tokenizer()

    # Load dataset
    print(f"\n[2] Loading dataset: {dataset_path}", flush=True)
    texts = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                instruction = item.get('instruction', item.get('prompt', ''))
                output = item.get('output', item.get('response', ''))

                # Use apply_chat_template for correct format
                messages = [
                    {'role': 'system', 'content': 'Eres THAU, un asistente experto en programacion, UX/UI y desarrollo web. Responde de forma clara y concisa en espanol.'},
                    {'role': 'user', 'content': instruction},
                    {'role': 'assistant', 'content': output},
                ]
                text = adapter.tokenizer.apply_chat_template(messages, tokenize=False)
                texts.append(text)

    print(f"    Loaded {len(texts)} training examples", flush=True)

    # Apply LoRA
    print("\n[3] Applying LoRA...", flush=True)
    lora_config = LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    adapter.prepare_for_lora(lora_config)

    # Tokenize
    print("\n[4] Tokenizing...", flush=True)
    print(f"    Sample text format: {texts[0][:150]}...", flush=True)
    encodings = adapter.tokenizer(
        texts,
        truncation=True,
        max_length=config.MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )

    dataset = torch.utils.data.TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        adapter.model.parameters(),
        lr=learning_rate,
    )

    # Training
    print(f"\n[5] Training: {num_epochs} epochs, batch size {batch_size}", flush=True)
    print("-" * 60, flush=True)

    adapter.model.train()
    all_losses = []

    for epoch in range(num_epochs):
        epoch_losses = []

        for batch_idx, batch in enumerate(dataloader):
            input_ids, attention_mask = batch
            input_ids = adapter.device_manager.to_device(input_ids)
            attention_mask = adapter.device_manager.to_device(attention_mask)

            # Forward pass
            outputs = adapter.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        all_losses.append(avg_loss)
        print(f"    Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}", flush=True)

    # Save checkpoint
    print("\n[6] Saving checkpoint...", flush=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_name = f"{output_name}_{timestamp}"
    checkpoint_path = f"data/checkpoints/incremental/specialized/{checkpoint_name}"

    adapter.save_model(checkpoint_path)

    # Save training info
    info = {
        "name": output_name,
        "timestamp": timestamp,
        "examples": len(texts),
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_loss": all_losses[-1],
        "all_losses": all_losses,
    }

    with open(f"{checkpoint_path}/training_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print(f"    Saved to: {checkpoint_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"  Training Complete!", flush=True)
    print(f"  Final Loss: {all_losses[-1]:.4f}", flush=True)
    print(f"  Checkpoint: {checkpoint_name}", flush=True)
    print("=" * 60, flush=True)

    return checkpoint_path, all_losses[-1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/datasets/programming_chat_format.jsonl")
    parser.add_argument("--name", default="thau_v2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    train_with_chat_format(
        dataset_path=args.dataset,
        output_name=args.name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
