"""
Entrenador de THAU - Maneja el proceso de fine-tuning incremental
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from typing import List, Dict
import logging

from .config import get_config

logger = logging.getLogger('ThauTrainer')


class ThauTrainer:
    """Entrenador incremental para THAU"""

    def __init__(self):
        self.config = get_config()
        self.model = None
        self.tokenizer = None

    def _load_or_create_model(self):
        """Carga el modelo existente o crea uno nuevo"""

        # Intentar cargar versión actual
        current_checkpoint = self.config.get_checkpoint_path()

        try:
            if Path(current_checkpoint).exists():
                logger.info(f"📂 Cargando modelo existente: {current_checkpoint}")
                model_path = current_checkpoint
            else:
                logger.info(f"🆕 Creando nuevo modelo desde: {self.config.get_base_model()}")
                model_path = self.config.get_base_model()

            # Configuración de cuantización
            bnb_config = None
            if self.config.use_quantization:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )

            # Cargar modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path if Path(current_checkpoint).exists() else self.config.get_base_model(),
                trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id

            logger.info("✅ Modelo cargado correctamente")

        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise

    def _setup_lora(self):
        """Configura LoRA adapters"""

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        logger.info(f"📊 Parámetros entrenables: {trainable_params:,} / {total_params:,} "
                   f"({100 * trainable_params / total_params:.2f}%)")

    def _prepare_dataset(self, examples: List[Dict]) -> Dataset:
        """Prepara dataset para entrenamiento"""

        # Formatear ejemplos
        formatted = []
        for ex in examples:
            text = f"<|im_start|>user\n{ex['instruction']}</s>\n<|im_start|>assistant\n{ex['output']}</s>"
            formatted.append({"text": text})

        # Crear dataset
        dataset = Dataset.from_list(formatted)

        # Tokenizar
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_context_length,
                padding="max_length",
            )

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        return tokenized

    def train(self, examples: List[Dict], epochs: int = 3, batch_size: int = 2, learning_rate: float = 2e-4):
        """Entrena el modelo con nuevos ejemplos"""

        logger.info(f"🎓 Iniciando entrenamiento con {len(examples)} ejemplos")

        try:
            # Cargar modelo
            self._load_or_create_model()

            # Setup LoRA
            self._setup_lora()

            # Preparar dataset
            dataset = self._prepare_dataset(examples)

            # Configurar training
            training_args = TrainingArguments(
                output_dir=f"./data/temp/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=8,
                learning_rate=learning_rate,
                fp16=False,
                logging_steps=5,
                save_steps=50,
                save_total_limit=2,
                warmup_steps=10,
                optim="adamw_torch",
                report_to="none",
                push_to_hub=False,
                lr_scheduler_type="cosine",
            )

            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )

            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )

            # Train
            logger.info("🚀 Entrenamiento en progreso...")
            result = trainer.train()

            logger.info(f"✅ Entrenamiento completado! Loss: {result.training_loss:.4f}")

            return {
                "success": True,
                "loss": result.training_loss,
                "examples_trained": len(examples),
                "epochs": epochs
            }

        except Exception as e:
            logger.error(f"❌ Error en entrenamiento: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    def save(self, path: str):
        """Guarda el modelo entrenado"""

        logger.info(f"💾 Guardando modelo en: {path}")

        try:
            Path(path).mkdir(parents=True, exist_ok=True)

            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

            logger.info("✅ Modelo guardado correctamente")

        except Exception as e:
            logger.error(f"❌ Error guardando modelo: {e}")
            raise


from pathlib import Path
from datetime import datetime
