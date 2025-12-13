#!/usr/bin/env python3
"""
THAU Specialized Training Script
Fine-tune from previous checkpoint with specialized data
"""

import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_specialized_data():
    """Load specialized training data"""
    qa_dir = Path("data/self_questioning")
    all_qa = []

    # Priority files for specialized training
    priority_files = [
        "reasoning_training.jsonl",
        "spanish_advanced_training.jsonl",
        "tool_calling_training.jsonl",
        "image_generation_training.jsonl",
    ]

    for filename in priority_files:
        jsonl_file = qa_dir / filename
        if jsonl_file.exists():
            print(f"Loading {filename}...")
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'question' in data and 'answer' in data:
                            q = data['question']
                            a = data['answer']
                            if q and a and len(a) > 10:
                                all_qa.append({
                                    'question': q,
                                    'answer': a,
                                    'source': filename,
                                    'category': data.get('category', 'general')
                                })
                    except:
                        continue

    # Also load other jsonl files
    for jsonl_file in qa_dir.glob("*.jsonl"):
        if jsonl_file.name not in priority_files:
            print(f"Loading {jsonl_file.name}...")
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'question' in data and 'answer' in data:
                            q = data['question']
                            a = data['answer']
                            if q and a and len(a) > 10:
                                all_qa.append({
                                    'question': q,
                                    'answer': a,
                                    'source': jsonl_file.name,
                                    'category': data.get('category', 'general')
                                })
                    except:
                        continue

    print(f"\nLoaded {len(all_qa)} specialized Q&A pairs")
    return all_qa


def format_training_text(qa: dict) -> str:
    """Format Q&A pair as training text with category awareness"""
    category = qa.get('category', 'general')

    # Use different system prompts based on category
    if category in ['chain_of_thought', 'technical_reasoning', 'meta_reasoning']:
        system = "Eres THAU, un asistente AI que razona paso a paso para resolver problemas."
    elif category in ['tool_calling', 'tool_calling_response']:
        system = "Eres THAU, un asistente AI que puede usar herramientas cuando es necesario."
    elif category in ['image_generation', 'prompt_engineering']:
        system = "Eres THAU, un asistente AI experto en generacion de imagenes."
    else:
        system = "Eres THAU, un asistente AI inteligente, servicial y amigable."

    return f"""<|system|>
{system}</s>
<|user|>
{qa['question']}</s>
<|assistant|>
{qa['answer']}</s>"""


def main():
    print("=" * 80)
    print("THAU SPECIALIZED TRAINING - Continuing from checkpoint")
    print("=" * 80)

    # Config
    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    CHECKPOINT_PATH = Path("data/checkpoints/full_training/final")
    OUTPUT_DIR = Path("data/checkpoints/specialized_training")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 2
    GRADIENT_ACCUM = 4
    LEARNING_RATE = 1e-4  # Lower LR for fine-tuning
    EPOCHS = 5  # More epochs for smaller dataset
    MAX_LENGTH = 768  # Longer for reasoning
    SAVE_STEPS = 100

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

    # Load specialized data
    print("\nLoading specialized training data...")
    qa_data = load_specialized_data()

    if len(qa_data) < 50:
        print("Not enough specialized training data!")
        return

    # Show distribution
    print("\nData distribution by category:")
    categories = {}
    for qa in qa_data:
        cat = qa.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    random.shuffle(qa_data)

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Check if checkpoint exists
    if CHECKPOINT_PATH.exists():
        print(f"\nLoading from checkpoint: {CHECKPOINT_PATH}")
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        # Load LoRA weights
        model = PeftModel.from_pretrained(base_model, str(CHECKPOINT_PATH), is_trainable=True)

        # Ensure LoRA adapters are trainable
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True

        print("Loaded previous LoRA weights with training enabled!")
    else:
        print(f"\nNo checkpoint found at {CHECKPOINT_PATH}")
        print("Starting fresh training...")
        from peft import LoraConfig, get_peft_model, TaskType

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
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

    print("\n" + "-" * 80)
    print("STARTING SPECIALIZED TRAINING")
    print("-" * 80 + "\n")

    start_time = time.time()
    global_step = 0
    total_loss = 0
    loss_count = 0

    for epoch in range(EPOCHS):
        print(f"\n{'=' * 40}")
        print(f"EPOCH {epoch + 1}/{EPOCHS}")
        print(f"{'=' * 40}\n")

        random.shuffle(qa_data)
        epoch_loss = 0
        epoch_steps = 0

        for i in range(0, len(qa_data), BATCH_SIZE):
            batch = qa_data[i:i + BATCH_SIZE]
            texts = [format_training_text(qa) for qa in batch]

            try:
                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss / GRADIENT_ACCUM

                loss.backward()

                if (global_step + 1) % GRADIENT_ACCUM == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * GRADIENT_ACCUM
                total_loss += loss.item() * GRADIENT_ACCUM
                loss_count += 1
                epoch_steps += 1
                global_step += 1

                if global_step % 20 == 0:
                    avg_loss = total_loss / loss_count if loss_count > 0 else 0
                    elapsed = time.time() - start_time
                    speed = global_step / elapsed if elapsed > 0 else 0
                    eta = (total_steps - global_step) / speed if speed > 0 else 0

                    print(f"Step {global_step:4d}/{total_steps} | "
                          f"Loss: {loss.item() * GRADIENT_ACCUM:.4f} | "
                          f"Avg: {avg_loss:.4f} | "
                          f"Speed: {speed:.2f} s/s | "
                          f"ETA: {eta / 60:.1f}min")

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

        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        print(f"\nEpoch {epoch + 1} complete:")
        print(f"  Steps: {epoch_steps}")
        print(f"  Average loss: {avg_epoch_loss:.4f}")

        ckpt_path = OUTPUT_DIR / f"epoch_{epoch + 1}"
        model.save_pretrained(str(ckpt_path))
        tokenizer.save_pretrained(str(ckpt_path))
        print(f"Saved epoch checkpoint: {ckpt_path}")

    # Final save
    elapsed = time.time() - start_time
    final_path = OUTPUT_DIR / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print("\n" + "=" * 80)
    print("SPECIALIZED TRAINING COMPLETE!")
    print("=" * 80)
    print(f"  Total time: {elapsed / 60:.1f} minutes")
    print(f"  Total steps: {global_step}")
    print(f"  Final avg loss: {total_loss / loss_count if loss_count > 0 else 0:.4f}")
    print(f"  Model saved to: {final_path}")

    # Save stats
    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_steps": global_step,
        "total_time_seconds": elapsed,
        "final_loss": total_loss / loss_count if loss_count > 0 else 0,
        "epochs": EPOCHS,
        "data_points": len(qa_data),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "training_type": "specialized",
        "categories": categories,
    }

    with open(OUTPUT_DIR / "training_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
