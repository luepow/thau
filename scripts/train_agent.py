#!/usr/bin/env python3
"""
Entrenamiento avanzado con capacidades de Agente AI
Incluye: Chain-of-Thought, Tool Calling, ReAct pattern
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, concatenate_datasets
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import argparse
import json


def load_combined_datasets():
    """Carga y combina todos los datasets de entrenamiento"""

    datasets = []

    # 1. Dataset de arquitectura básico
    arch_file = "./data/datasets/architecture_training.jsonl"
    if Path(arch_file).exists():
        with open(arch_file, 'r', encoding='utf-8') as f:
            arch_data = [json.loads(line) for line in f]
        datasets.append(arch_data)
        print(f"✅ Cargado: {len(arch_data)} ejemplos de arquitectura")

    # 2. Dataset de razonamiento Chain-of-Thought
    cot_file = "./data/datasets/reasoning_cot.jsonl"
    if Path(cot_file).exists():
        with open(cot_file, 'r', encoding='utf-8') as f:
            cot_data = [json.loads(line) for line in f]
        datasets.append(cot_data)
        print(f"✅ Cargado: {len(cot_data)} ejemplos de razonamiento")

    # 3. Dataset de Tool Calling
    tool_file = "./data/datasets/tool_calling.jsonl"
    if Path(tool_file).exists():
        with open(tool_file, 'r', encoding='utf-8') as f:
            tool_data = [json.loads(line) for line in f]
        datasets.append(tool_data)
        print(f"✅ Cargado: {len(tool_data)} ejemplos de tool calling")

    # Combinar todos los datasets
    all_data = []
    for dataset in datasets:
        all_data.extend(dataset)

    # Formatear para entrenamiento
    formatted_data = []
    for item in all_data:
        # Formato para TinyLlama/Qwen
        text = f"<|im_start|>user\n{item['instruction']}</s>\n<|im_start|>assistant\n{item['output']}</s>"
        formatted_data.append({"text": text})

    print(f"\n🎯 Total: {len(formatted_data)} ejemplos de entrenamiento")
    return Dataset.from_list(formatted_data)


def setup_model_and_tokenizer(model_name: str, use_quantization: bool = True):
    """Configura modelo con cuantización optimizada"""

    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        bnb_config = None

    print(f"\n🔧 Cargando modelo: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if use_quantization else None,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def setup_lora(model, r=32, alpha=64):
    """Configura LoRA con parámetros optimizados para agentes"""

    lora_config = LoraConfig(
        r=r,  # Rank más alto para capacidades complejas
        lora_alpha=alpha,  # Alpha más alto para mejor aprendizaje
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    print("\n📊 Parámetros entrenables:")
    model.print_trainable_parameters()

    return model


def tokenize_function(examples, tokenizer, max_length=2048):
    """Tokeniza con mayor contexto para razonamiento complejo"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def main():
    parser = argparse.ArgumentParser(description="Entrenar modelo con capacidades de agente")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output", type=str, default="./data/checkpoints/agent-expert")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--no-quantization", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("🤖 Entrenamiento de Agente AI con Tool Calling")
    print("=" * 80)

    # Cargar datasets combinados
    print("\n📚 Cargando datasets...")
    dataset = load_combined_datasets()

    # Cargar modelo
    use_quant = not args.no_quantization
    model, tokenizer = setup_model_and_tokenizer(args.model, use_quantization=use_quant)

    # Configurar LoRA
    print("\n🔧 Configurando LoRA...")
    model = setup_lora(model, r=args.lora_r, alpha=args.lora_alpha)

    # Tokenizar
    print("\n🔤 Tokenizando dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=2048),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Configurar entrenamiento
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,  # Más acumulación para estabilidad
        learning_rate=args.learning_rate,
        fp16=False,
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        warmup_steps=100,
        optim="adamw_torch",
        report_to="none",
        push_to_hub=False,
        lr_scheduler_type="cosine",  # Scheduler cosine para mejor convergencia
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
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Entrenar
    print("\n🚀 Iniciando entrenamiento avanzado...")
    print(f"   - Epochs: {args.epochs}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - LoRA rank: {args.lora_r}")
    print(f"   - LoRA alpha: {args.lora_alpha}")
    print(f"   - Contexto: 2048 tokens")
    print(f"   - Output: {args.output}")
    print()

    trainer.train()

    # Guardar
    print("\n💾 Guardando modelo entrenado...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    print(f"\n✅ Entrenamiento completado!")
    print(f"📁 Modelo guardado en: {args.output}")
    print("\n🎯 Capacidades del modelo:")
    print("   ✅ Razonamiento Chain-of-Thought")
    print("   ✅ Tool Calling (web_search, code_execution, etc.)")
    print("   ✅ Análisis de arquitectura de software")
    print("   ✅ Debugging y seguridad")
    print("   ✅ Investigación y síntesis")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
