#!/usr/bin/env python3
"""
THAU Full Training Script
Fine-tune TinyLlama with all generated Q&A data
"""

import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def load_all_qa_data():
    """Load all Q&A pairs from jsonl files"""
    qa_dir = Path("data/self_questioning")
    all_qa = []

    for jsonl_file in qa_dir.glob("*.jsonl"):
        print(f"Loading {jsonl_file.name}...")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'question' in data and 'answer' in data:
                        q = data['question']
                        a = data['answer']
                        if q and a and len(a) > 20:
                            all_qa.append({
                                'question': q,
                                'answer': a,
                                'source': jsonl_file.name
                            })
                except:
                    continue

    print(f"\nLoaded {len(all_qa)} Q&A pairs total")
    return all_qa


def format_training_text(qa: dict) -> str:
    """Format Q&A pair as training text"""
    return f"""<|system|>
Eres THAU, un asistente AI inteligente y servicial.</s>
<|user|>
{qa['question']}</s>
<|assistant|>
{qa['answer']}</s>"""


def main():
    print("="*80)
    print("THAU FULL TRAINING - TinyLlama Fine-tuning")
    print("="*80)

    # Config
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    OUTPUT_DIR = Path("data/checkpoints/full_training")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 2
    GRADIENT_ACCUM = 4
    LEARNING_RATE = 2e-4
    EPOCHS = 3
    MAX_LENGTH = 512
    SAVE_STEPS = 200

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA")
    else:
        device = torch.device("cpu")
        print(f"Using CPU")

    # Load data
    print("\nLoading training data...")
    qa_data = load_all_qa_data()

    if len(qa_data) < 100:
        print("Not enough training data!")
        return

    # Shuffle
    random.shuffle(qa_data)

    # Load model and tokenizer
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.to(device)
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training stats
    total_steps = (len(qa_data) // BATCH_SIZE) * EPOCHS
    print(f"\nTraining config:")
    print(f"  Data points: {len(qa_data)}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUM}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Total steps: {total_steps}")
    print(f"  Learning rate: {LEARNING_RATE}")

    print("\n" + "-"*80)
    print("STARTING TRAINING")
    print("-"*80 + "\n")

    start_time = time.time()
    global_step = 0
    total_loss = 0
    loss_count = 0

    for epoch in range(EPOCHS):
        print(f"\n{'='*40}")
        print(f"EPOCH {epoch+1}/{EPOCHS}")
        print(f"{'='*40}\n")

        random.shuffle(qa_data)
        epoch_loss = 0
        epoch_steps = 0

        for i in range(0, len(qa_data), BATCH_SIZE):
            batch = qa_data[i:i+BATCH_SIZE]

            # Format texts
            texts = [format_training_text(qa) for qa in batch]

            # Tokenize
            try:
                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Forward
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss / GRADIENT_ACCUM

                # Backward
                loss.backward()

                # Update
                if (global_step + 1) % GRADIENT_ACCUM == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * GRADIENT_ACCUM
                total_loss += loss.item() * GRADIENT_ACCUM
                loss_count += 1
                epoch_steps += 1
                global_step += 1

                # Log
                if global_step % 25 == 0:
                    avg_loss = total_loss / loss_count if loss_count > 0 else 0
                    elapsed = time.time() - start_time
                    speed = global_step / elapsed if elapsed > 0 else 0
                    eta = (total_steps - global_step) / speed if speed > 0 else 0

                    print(f"Step {global_step:5d}/{total_steps} | "
                          f"Loss: {loss.item()*GRADIENT_ACCUM:.4f} | "
                          f"Avg: {avg_loss:.4f} | "
                          f"Speed: {speed:.2f} s/s | "
                          f"ETA: {eta/60:.1f}min")

                # Save checkpoint
                if global_step % SAVE_STEPS == 0:
                    ckpt_path = OUTPUT_DIR / f"checkpoint_step_{global_step}"
                    model.save_pretrained(str(ckpt_path))
                    tokenizer.save_pretrained(str(ckpt_path))
                    print(f"Saved checkpoint: {ckpt_path}")

            except Exception as e:
                print(f"Error at step {global_step}: {e}")
                if device.type == "mps":
                    torch.mps.empty_cache()
                continue

        # Epoch summary
        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        print(f"\nEpoch {epoch+1} complete:")
        print(f"  Steps: {epoch_steps}")
        print(f"  Average loss: {avg_epoch_loss:.4f}")

        # Save epoch checkpoint
        ckpt_path = OUTPUT_DIR / f"epoch_{epoch+1}"
        model.save_pretrained(str(ckpt_path))
        tokenizer.save_pretrained(str(ckpt_path))
        print(f"Saved epoch checkpoint: {ckpt_path}")

    # Final save
    elapsed = time.time() - start_time
    final_path = OUTPUT_DIR / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Total steps: {global_step}")
    print(f"  Final avg loss: {total_loss/loss_count if loss_count > 0 else 0:.4f}")
    print(f"  Model saved to: {final_path}")

    # Save stats
    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_steps": global_step,
        "total_time_seconds": elapsed,
        "final_loss": total_loss/loss_count if loss_count > 0 else 0,
        "epochs": EPOCHS,
        "data_points": len(qa_data),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }

    with open(OUTPUT_DIR / "training_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
