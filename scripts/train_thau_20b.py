#!/usr/bin/env python3
"""
THAU 20B Training Script
Fine-tune a 20B parameter model using QLoRA for efficient training.

Supported base models:
- mistralai/Mistral-Small-24B-Instruct-2501 (24B, Apache 2.0)
- Qwen/Qwen2.5-14B-Instruct (14B, Apache 2.0)
- Qwen/Qwen2.5-32B-Instruct (32B, Apache 2.0)
- deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct (16B, MIT)

Requirements:
- GPU with 24GB+ VRAM (RTX 4090, A6000, A100)
- Or use cloud: RunPod, Lambda Labs, Vast.ai
"""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import Dataset, load_dataset
from loguru import logger

# ============================================
# CONFIGURATION
# ============================================

@dataclass
class TrainingConfig:
    """Configuration for THAU 20B training."""

    # Model selection
    base_model: str = "Qwen/Qwen2.5-14B-Instruct"  # Good balance of size/quality

    # Alternative models (uncomment to use):
    # base_model: str = "mistralai/Mistral-Small-24B-Instruct-2501"  # 24B, best quality
    # base_model: str = "Qwen/Qwen2.5-32B-Instruct"  # 32B, highest capacity
    # base_model: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"  # 16B, code focused

    # Output
    output_dir: str = "data/checkpoints/thau-20b"
    model_name: str = "thau-20b"

    # QLoRA Configuration
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = True

    # LoRA Configuration
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training Configuration
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 4096

    # Optimization
    optim: str = "paged_adamw_32bit"
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    # Data
    datasets_dir: str = "data/datasets"
    max_samples: Optional[int] = None  # None = use all


# ============================================
# DATA PREPARATION
# ============================================

def load_thau_datasets(config: TrainingConfig) -> Dataset:
    """Load and combine all THAU training datasets."""

    datasets_path = Path(config.datasets_dir)
    all_data = []

    # Priority datasets for THAU capabilities
    priority_files = [
        "programming_combined_20251202.jsonl",
        "thau_v3_clean.jsonl",
        "contable_training.jsonl",
        "agent_training.jsonl",
        "devops_training_20251202.jsonl",
        "python_training_20251202.jsonl",
        "javascript_training_20251202.jsonl",
        "java_training_20251202.jsonl",
        "flutter_dart_training_20251202.jsonl",
        "algorithms_training_20251202.jsonl",
        "agile_training_20251202.jsonl",
        "git_training_20251202.jsonl",
        "sql_databases_training_20251202.jsonl",
        "web_training_20251202.jsonl",
    ]

    # Load priority files first
    for filename in priority_files:
        filepath = datasets_path / filename
        if filepath.exists():
            logger.info(f"Loading priority dataset: {filename}")
            all_data.extend(load_jsonl(filepath))

    # Load remaining jsonl files
    for filepath in datasets_path.glob("*.jsonl"):
        if filepath.name not in priority_files:
            logger.info(f"Loading dataset: {filepath.name}")
            all_data.extend(load_jsonl(filepath))

    logger.info(f"Total examples loaded: {len(all_data)}")

    # Limit samples if specified
    if config.max_samples and len(all_data) > config.max_samples:
        import random
        random.shuffle(all_data)
        all_data = all_data[:config.max_samples]
        logger.info(f"Limited to {config.max_samples} samples")

    # Convert to HuggingFace Dataset
    return Dataset.from_list(all_data)


def load_jsonl(filepath: Path) -> List[dict]:
    """Load JSONL file and convert to training format."""
    data = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    # Handle different formats
                    if "messages" in item:
                        # Chat format
                        data.append(item)
                    elif "instruction" in item and "output" in item:
                        # Instruction format -> convert to chat
                        messages = [
                            {"role": "user", "content": item["instruction"]},
                            {"role": "assistant", "content": item["output"]}
                        ]
                        if "input" in item and item["input"]:
                            messages[0]["content"] += f"\n\nInput: {item['input']}"
                        data.append({"messages": messages})
                    elif "question" in item and "answer" in item:
                        # QA format -> convert to chat
                        messages = [
                            {"role": "user", "content": item["question"]},
                            {"role": "assistant", "content": item["answer"]}
                        ]
                        data.append({"messages": messages})
                    elif "prompt" in item and "completion" in item:
                        # Completion format -> convert to chat
                        messages = [
                            {"role": "user", "content": item["prompt"]},
                            {"role": "assistant", "content": item["completion"]}
                        ]
                        data.append({"messages": messages})
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"Error loading {filepath}: {e}")

    return data


def format_chat_template(example: dict, tokenizer) -> dict:
    """Format example using the model's chat template."""

    messages = example.get("messages", [])

    # Add system message for THAU identity
    system_msg = {
        "role": "system",
        "content": (
            "Eres THAU, un asistente de inteligencia artificial avanzado. "
            "Tienes capacidades en programacion, generacion de SVG, razonamiento paso a paso, "
            "contabilidad, DevOps, y mas. Responde de manera clara, concisa y util."
        )
    }

    # Insert system message if not present
    if not messages or messages[0].get("role") != "system":
        messages = [system_msg] + messages

    # Apply chat template
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    except Exception:
        # Fallback format
        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                text += f"<|system|>\n{content}\n"
            elif role == "user":
                text += f"<|user|>\n{content}\n"
            elif role == "assistant":
                text += f"<|assistant|>\n{content}\n"

    return {"text": text}


# ============================================
# MODEL SETUP
# ============================================

def setup_model_and_tokenizer(config: TrainingConfig):
    """Setup model with QLoRA configuration."""

    logger.info(f"Loading base model: {config.base_model}")

    # Quantization config
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.use_nested_quant,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
    )

    # Set padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


# ============================================
# TRAINING
# ============================================

def train(config: TrainingConfig):
    """Main training function."""

    logger.info("=" * 60)
    logger.info("THAU 20B Training")
    logger.info("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        logger.warning("CUDA not available! Training will be very slow on CPU.")
        logger.warning("Consider using a cloud GPU service like RunPod or Lambda Labs.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)

    # Load datasets
    logger.info("Loading datasets...")
    dataset = load_thau_datasets(config)

    # Format with chat template
    logger.info("Formatting with chat template...")
    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=dataset.column_names,
    )

    # Tokenize
    logger.info("Tokenizing...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
        )

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    # Split into train/eval
    dataset = dataset.train_test_split(test_size=0.05)

    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Eval samples: {len(dataset['test'])}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to=["tensorboard"],
        run_name=f"thau-20b-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    logger.info(f"Output directory: {config.output_dir}")

    trainer.train()

    # Save final model
    final_path = Path(config.output_dir) / "final"
    logger.info(f"Saving final model to {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Save config
    config_path = final_path / "thau_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            "base_model": config.base_model,
            "model_name": config.model_name,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "training_date": datetime.now().isoformat(),
        }, f, indent=2)

    logger.info("Training complete!")
    logger.info(f"Model saved to: {final_path}")

    return final_path


# ============================================
# EXPORT TO OLLAMA
# ============================================

def export_to_ollama(model_path: Path, config: TrainingConfig):
    """Export trained model to Ollama format."""

    logger.info("Exporting to Ollama format...")

    # Create Modelfile
    modelfile_content = f'''FROM {model_path}

TEMPLATE """{{{{- if .System }}}}
<|system|>
{{{{ .System }}}}
{{{{- end }}}}
<|user|>
{{{{ .Prompt }}}}
<|assistant|>
{{{{ .Response }}}}
"""

SYSTEM """Eres THAU, un asistente de inteligencia artificial avanzado creado por Luis Perez.
Tienes capacidades en:
- Programacion (Python, JavaScript, Java, Rust, Go, SQL)
- Generacion de SVG y assets visuales
- Razonamiento paso a paso (Chain of Thought)
- Contabilidad y asientos contables
- DevOps y CI/CD
- Explicacion de codigo

Responde siempre de manera clara, concisa y util."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
'''

    modelfile_path = model_path / "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)

    logger.info(f"Modelfile created at: {modelfile_path}")
    logger.info(f"To create Ollama model, run:")
    logger.info(f"  ollama create {config.model_name} -f {modelfile_path}")


# ============================================
# CLOUD TRAINING SETUP
# ============================================

def print_cloud_instructions():
    """Print instructions for cloud training."""

    print("""
================================================================================
CLOUD TRAINING INSTRUCTIONS FOR THAU 20B
================================================================================

Since training a 20B model requires significant GPU resources, here are options:

1. RUNPOD (Recommended)
   - Cost: ~$0.40-0.80/hour for A100 40GB
   - Setup:
     a) Create account at runpod.io
     b) Deploy a pod with PyTorch template
     c) Select A100 40GB or A6000 48GB
     d) Clone this repo and run train_thau_20b.py

2. LAMBDA LABS
   - Cost: ~$1.10/hour for A100 40GB
   - Good for longer training runs
   - More stable than spot instances

3. VAST.AI
   - Cost: ~$0.20-0.40/hour (spot)
   - Cheapest option but less reliable
   - Good for experimentation

4. GOOGLE COLAB PRO+
   - Cost: $50/month
   - A100 40GB available (limited time)
   - Good for short training runs

ESTIMATED TRAINING COSTS:
- Qwen 14B with 2000 samples: ~2-4 hours = $1-3
- Mistral 24B with 2000 samples: ~4-6 hours = $2-5
- Full dataset (~5000 samples): ~8-12 hours = $4-10

================================================================================
""")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train THAU 20B")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct",
                       help="Base model to fine-tune")
    parser.add_argument("--output", type=str, default="data/checkpoints/thau-20b",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Per-device batch size")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of training samples")
    parser.add_argument("--cloud-info", action="store_true",
                       help="Print cloud training instructions")
    parser.add_argument("--dry-run", action="store_true",
                       help="Check setup without training")

    args = parser.parse_args()

    if args.cloud_info:
        print_cloud_instructions()
        sys.exit(0)

    # Create config
    config = TrainingConfig(
        base_model=args.model,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    if args.dry_run:
        logger.info("Dry run - checking setup...")
        logger.info(f"Base model: {config.base_model}")
        logger.info(f"Output: {config.output_dir}")

        # Check GPU
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("No GPU detected!")

        # Check datasets
        dataset = load_thau_datasets(config)
        logger.info(f"Dataset samples: {len(dataset)}")

        logger.info("Dry run complete!")
    else:
        # Train
        model_path = train(config)

        # Export to Ollama
        export_to_ollama(model_path, config)
