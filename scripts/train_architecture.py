#!/usr/bin/env python3
"""
Script de entrenamiento especializado para conocimientos de arquitectura de software
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import argparse


def load_architecture_dataset(file_path: str):
    """Carga el dataset de arquitectura de software"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    # Formatear para entrenamiento conversacional
    formatted_data = []
    for item in data:
        # Formato de chat para TinyLlama
        text = f"<|user|>\n{item['instruction']}</s>\n<|assistant|>\n{item['output']}</s>"
        formatted_data.append({"text": text})

    return Dataset.from_list(formatted_data)


def setup_model_and_tokenizer(model_name: str, use_quantization: bool = True):
    """Configura el modelo y tokenizer con cuantización"""

    # Configuración de cuantización para Apple Silicon
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        bnb_config = None

    # Cargar modelo
    print(f"Cargando modelo {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if use_quantization else None,
        device_map="auto",
        trust_remote_code=True,
    )

    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Configurar padding token si no existe
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def setup_lora(model):
    """Configura LoRA para fine-tuning eficiente"""

    lora_config = LoraConfig(
        r=16,  # Rank de las matrices LoRA
        lora_alpha=32,  # Scaling factor
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Módulos a adaptar
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Preparar modelo para entrenamiento
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model


def tokenize_function(examples, tokenizer, max_length=1024):
    """Tokeniza los ejemplos"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def main():
    parser = argparse.ArgumentParser(description="Entrenar modelo con conocimientos de arquitectura")
    parser.add_argument("--dataset", type=str, default="./data/datasets/architecture_training.jsonl")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output", type=str, default="./data/checkpoints/architecture-expert")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--no-quantization", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("🏗️  Entrenamiento de Arquitecto de Software AI")
    print("=" * 80)

    # Cargar dataset
    print(f"\n📚 Cargando dataset desde {args.dataset}...")
    dataset = load_architecture_dataset(args.dataset)
    print(f"✅ Dataset cargado: {len(dataset)} ejemplos")

    # Cargar modelo y tokenizer
    use_quant = not args.no_quantization
    model, tokenizer = setup_model_and_tokenizer(args.model, use_quantization=use_quant)

    # Configurar LoRA
    print("\n🔧 Configurando LoRA...")
    model = setup_lora(model)

    # Tokenizar dataset
    print("\n🔤 Tokenizando dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Configurar entrenamiento
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        fp16=False,  # No usar fp16 en Apple Silicon
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        warmup_steps=50,
        optim="adamw_torch",
        report_to="none",
        push_to_hub=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Crear trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Entrenar
    print("\n🚀 Iniciando entrenamiento...")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Output: {args.output}")
    print()

    trainer.train()

    # Guardar modelo
    print("\n💾 Guardando modelo entrenado...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    print(f"\n✅ Entrenamiento completado!")
    print(f"📁 Modelo guardado en: {args.output}")
    print(f"🎯 Modelo especializado en arquitectura de software")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
