#!/usr/bin/env python3
"""
THAU 7B - Cognitive Learning LLM
Entrenamiento de THAU como modelo de 7B parámetros con aprendizaje cognitivo

Este script entrena un modelo base de 7B para convertirlo en THAU,
un LLM con capacidades de aprendizaje cognitivo inspiradas en humanos.
"""

import os
import sys
import json
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime

# Agregar path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ThauCognitiveConfig:
    """Configuración para entrenamiento cognitivo de THAU 7B"""

    # Modelos base recomendados para 7B
    BASE_MODELS = {
        "qwen": "Qwen/Qwen2.5-7B-Instruct",
        "llama": "meta-llama/Llama-3.1-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "phi": "microsoft/Phi-3-medium-4k-instruct",  # 14B pero eficiente
    }

    # Modelo base seleccionado
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # Configuración LoRA optimizada para 7B (estable numéricamente)
    lora_r: int = 16  # Rank conservador para estabilidad
    lora_alpha: int = 16  # alpha = r para scaling factor 1.0 (más estable)
    lora_dropout: float = 0.1  # Mayor dropout para regularización
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Cuantización 4-bit para Mac/GPU limitada
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    use_nested_quant: bool = True

    # Parámetros de entrenamiento OPTIMIZADOS para M2 Max 96GB (ESTABILIDAD MEJORADA)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4  # REDUCIDO: mejor estabilidad numérica
    gradient_accumulation_steps: int = 8  # Effective batch = 32 (mismo)
    learning_rate: float = 5e-6  # REDUCIDO: más estable para MPS
    weight_decay: float = 0.01
    warmup_ratio: float = 0.15  # 15% warmup para mejor estabilidad
    max_seq_length: int = 2048  # REDUCIDO: mejor estabilidad de memoria
    max_grad_norm: float = 0.3  # REDUCIDO: gradient clipping más agresivo

    # Optimización para M2 Max (ESTABILIDAD)
    gradient_checkpointing: bool = True  # ACTIVADO: reduce picos de memoria
    optim: str = "adamw_torch"  # SIN fused: más estable en MPS
    fp16: bool = False  # DESACTIVADO: fp32 es más estable en MPS
    bf16: bool = False  # bf16 no bien soportado en MPS

    # DataLoader workers para usar los 12 cores
    dataloader_num_workers: int = 8  # Usar 8 de los 12 cores para data loading

    # Logging y guardado
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3

    # Output
    output_dir: str = "data/checkpoints/thau-7b"
    hub_model_id: str = "luepow/thau-7b"


class ThauCognitiveTrainer:
    """
    Entrenador cognitivo para THAU 7B

    Implementa un sistema de aprendizaje inspirado en cognición humana:
    1. Aprendizaje por ejemplos (few-shot learning)
    2. Razonamiento paso a paso (chain of thought)
    3. Memoria de patrones (pattern recognition)
    4. Adaptación contextual
    """

    def __init__(self, config: ThauCognitiveConfig):
        self.config = config
        self.data_dir = Path(__file__).parent.parent / "data" / "datasets"
        self.training_data = []

    def load_cognitive_data(self) -> List[Dict]:
        """Carga y prepara datos de entrenamiento cognitivo"""

        datasets = []

        # Cargar todos los datasets disponibles
        dataset_files = [
            # Programación (conocimiento técnico)
            "programming_combined_20251202.jsonl",
            "python_training_20251202.jsonl",
            "javascript_training_20251202.jsonl",
            "java_training_20251202.jsonl",
            "rust_go_training_20251202.jsonl",
            "sql_databases_training_20251202.jsonl",
            "web_training_20251202.jsonl",
            "flutter_dart_training_20251202.jsonl",
            "powershell_training_20251202.jsonl",

            # Razonamiento (pensamiento cognitivo)
            "algorithms_training_20251202.jsonl",
            "math_training_20251202.jsonl",
            "reasoning_training.jsonl",
            "reasoning_cot.jsonl",

            # Habilidades especializadas
            "agile_training_20251202.jsonl",
            "devops_training_20251202.jsonl",
            "git_training_20251202.jsonl",

            # Contabilidad y análisis
            "contable_training.jsonl",
            "thau_v3_clean.jsonl",

            # UX y diseño
            "ux_chat_format.jsonl",
            "ux_css_frameworks.jsonl",
            "svg_assets_training.jsonl",

            # Datos de agente y herramientas
            "agent_training.jsonl",
            "tool_calling.jsonl",
            "pdf_training_agents.jsonl",

            # CAPACIDADES AVANZADAS: Audio, Video, Imagenes, MCP, Auto-entrenamiento
            "thau_capabilities_training.jsonl",

            # Arquitectura y diseño de sistemas
            "architecture_training.jsonl",

            # Datasets unificados avanzados
            "thau_unified_20251207_141131.jsonl",
            "thau_advanced_20251207_080734.jsonl",
            "thau_developer_20251207_073513.jsonl",
        ]

        print("\n📚 Cargando datos de entrenamiento cognitivo...")

        for filename in dataset_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                count = 0
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            datasets.append(data)
                            count += 1
                        except json.JSONDecodeError:
                            continue
                print(f"   ✓ {filename}: {count} ejemplos")

        print(f"\n📊 Total ejemplos cargados: {len(datasets)}")

        self.training_data = datasets
        return datasets

    def format_cognitive_prompt(self, example: Dict) -> str:
        """
        Formatea ejemplos para aprendizaje cognitivo

        Estructura cognitiva (ChatML format para Qwen2.5):
        - <|im_start|>system\n...<|im_end|>
        - <|im_start|>user\n...<|im_end|>
        - <|im_start|>assistant\n...<|im_end|>
        """

        # Detectar formato del ejemplo
        if "messages" in example:
            # Formato chat
            messages = example["messages"]
            formatted = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role in ["system", "user", "assistant"]:
                    formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            return formatted

        elif "instruction" in example and "output" in example:
            # Formato instruction
            instruction = example["instruction"]
            input_text = example.get("input", "")
            output = example["output"]

            if input_text:
                return f"""<|im_start|>system
Eres THAU, un asistente de IA con capacidades de aprendizaje cognitivo.<|im_end|>
<|im_start|>user
{instruction}

{input_text}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
"""
            else:
                return f"""<|im_start|>system
Eres THAU, un asistente de IA con capacidades de aprendizaje cognitivo.<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
"""

        elif "prompt" in example and "response" in example:
            # Formato prompt/response
            return f"""<|im_start|>system
Eres THAU, un asistente de IA con capacidades de aprendizaje cognitivo.<|im_end|>
<|im_start|>user
{example["prompt"]}<|im_end|>
<|im_start|>assistant
{example["response"]}<|im_end|>
"""

        elif "text" in example:
            # Formato texto plano
            return example["text"]

        return ""

    def prepare_dataset(self):
        """Prepara el dataset para entrenamiento"""

        if not self.training_data:
            self.load_cognitive_data()

        # Formatear todos los ejemplos
        formatted_data = []
        for example in self.training_data:
            formatted = self.format_cognitive_prompt(example)
            if formatted and len(formatted) > 50:  # Filtrar ejemplos muy cortos
                formatted_data.append({"text": formatted})

        print(f"\n✅ {len(formatted_data)} ejemplos formateados para entrenamiento")

        return formatted_data

    def check_hardware(self) -> Dict:
        """Verifica el hardware disponible"""

        info = {
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "device": "cpu",
            "memory_gb": 0,
            "can_train_locally": False
        }

        if info["cuda_available"]:
            info["device"] = "cuda"
            info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["can_train_locally"] = info["memory_gb"] >= 16

        elif info["mps_available"]:
            info["device"] = "mps"
            # MPS no reporta memoria fácilmente, asumir Mac con unified memory
            import subprocess
            try:
                result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
                info["memory_gb"] = int(result.stdout.strip()) / 1e9
            except:
                info["memory_gb"] = 16  # Asumir 16GB
            info["can_train_locally"] = info["memory_gb"] >= 32  # Necesita 32GB para 7B con QLoRA

        return info

    def train(self, dry_run: bool = False):
        """
        Entrena THAU 7B con aprendizaje cognitivo

        Args:
            dry_run: Si True, solo verifica configuración sin entrenar
        """

        print("=" * 60)
        print("🧠 THAU 7B - Cognitive Learning Training")
        print("=" * 60)

        # Verificar hardware
        hw_info = self.check_hardware()
        print(f"\n💻 Hardware detectado:")
        print(f"   Device: {hw_info['device']}")
        print(f"   Memoria: {hw_info['memory_gb']:.1f} GB")
        if hw_info.get('gpu_name'):
            print(f"   GPU: {hw_info['gpu_name']}")

        # Determinar si usar cuantización (solo CUDA soporta bitsandbytes)
        use_quantization = self.config.use_4bit and hw_info['device'] == 'cuda'
        if hw_info['device'] == 'mps' and self.config.use_4bit:
            print("\n⚠️  MPS detectado: Deshabilitando cuantización 4-bit (requiere CUDA)")
            print("   Usando LoRA con float16 en su lugar (96GB RAM es suficiente)")
            use_quantization = False

        # Cargar y preparar datos
        dataset = self.prepare_dataset()

        if dry_run:
            print("\n🔍 Modo DRY RUN - Verificando configuración...")
            print(f"\n📋 Configuración de entrenamiento:")
            print(f"   Modelo base: {self.config.base_model}")
            print(f"   LoRA rank: {self.config.lora_r}")
            print(f"   LoRA alpha: {self.config.lora_alpha}")
            print(f"   Epochs: {self.config.num_train_epochs}")
            print(f"   Batch size: {self.config.per_device_train_batch_size}")
            print(f"   Gradient accumulation: {self.config.gradient_accumulation_steps}")
            print(f"   Learning rate: {self.config.learning_rate}")
            print(f"   Max seq length: {self.config.max_seq_length}")
            print(f"   Cuantización 4-bit: {use_quantization}")

            # Estimar requerimientos
            print(f"\n📊 Estimaciones:")
            print(f"   Ejemplos de entrenamiento: {len(dataset)}")
            print(f"   Pasos por epoch: ~{len(dataset) // (self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps)}")
            if use_quantization:
                print(f"   VRAM estimada: ~20-24 GB (con QLoRA)")
            else:
                print(f"   RAM estimada: ~40-50 GB (LoRA sin cuantización)")

            if not hw_info["can_train_locally"]:
                print("\n⚠️  Tu hardware puede no ser suficiente para entrenamiento local.")
                print("   Opciones:")
                print("   1. Usar Google Colab Pro (~$10/mes)")
                print("   2. Usar RunPod (~$0.50/hora con A100)")
                print("   3. Usar Lambda Labs (~$1.10/hora con A10)")
                print("   4. Reducir max_seq_length a 1024")
            else:
                print("\n✅ Hardware suficiente para entrenamiento local!")

            return

        # Entrenamiento real
        print("\n🚀 Iniciando entrenamiento...")

        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )
            from peft import LoraConfig, get_peft_model
            from trl import SFTTrainer, SFTConfig
            from datasets import Dataset

            # Solo importar BitsAndBytes si vamos a usarlo
            if use_quantization:
                from transformers import BitsAndBytesConfig
                from peft import prepare_model_for_kbit_training

        except ImportError as e:
            print(f"\n❌ Error: Faltan dependencias: {e}")
            print("   Instala con: pip install transformers peft trl datasets accelerate")
            return

        # Configurar cuantización (solo para CUDA)
        bnb_config = None
        if use_quantization:
            compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
            )

        # Cargar modelo base
        print(f"\n📥 Cargando modelo base: {self.config.base_model}")

        # Configuración específica para cada dispositivo
        if hw_info['device'] == 'mps':
            # Para Mac MPS: usar float16 sin cuantización
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        elif use_quantization:
            # Para CUDA con cuantización
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # Para CPU o CUDA sin cuantización
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        model.config.use_cache = False

        # Cargar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Preparar modelo para entrenamiento k-bit (solo si usamos cuantización)
        if use_quantization:
            model = prepare_model_for_kbit_training(model)

        # Configurar LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Preparar dataset
        train_dataset = Dataset.from_list(dataset)

        # Configurar entrenamiento con optimizador compatible
        # paged_adamw_8bit requiere bitsandbytes/CUDA, usar adamw_torch para MPS
        optim = self.config.optim if use_quantization else "adamw_torch"

        # bf16 no está bien soportado en MPS, usar fp16
        use_bf16 = self.config.bf16 and hw_info['device'] != 'mps'
        use_fp16 = self.config.fp16 or (hw_info['device'] == 'mps')

        # Usar SFTConfig (nueva API de trl 0.25+) OPTIMIZADO para M2 Max
        sft_config = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            gradient_checkpointing=self.config.gradient_checkpointing,
            optim=optim,
            fp16=use_fp16,
            bf16=use_bf16,
            report_to="none",
            # Optimizaciones M2 Max
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=False,  # MPS no soporta pinned memory
            dataloader_prefetch_factor=4,  # Prefetch más datos
        )

        # Tokenizar dataset manualmente para la nueva API
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt",
            )

        print("\n📝 Tokenizando dataset...")
        tokenized_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )

        # Crear trainer con nueva API
        trainer = SFTTrainer(
            model=model,
            train_dataset=tokenized_dataset,
            processing_class=tokenizer,
            args=sft_config,
        )

        # Entrenar
        print("\n🏋️ Entrenando THAU 7B...")
        trainer.train()

        # Guardar modelo
        print(f"\n💾 Guardando modelo en {self.config.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(self.config.output_dir)

        print("\n✅ ¡Entrenamiento completado!")
        print(f"   Modelo guardado en: {self.config.output_dir}")

        # Instrucciones para convertir a Ollama
        print("\n📦 Para usar con Ollama:")
        print("   1. Mergear adaptadores LoRA con modelo base")
        print("   2. Convertir a GGUF con llama.cpp")
        print("   3. Crear Modelfile y usar 'ollama create'")

    def create_ollama_modelfile(self) -> str:
        """Genera el Modelfile para Ollama después del entrenamiento (ChatML format)"""

        modelfile = '''# THAU 7B - Cognitive Learning LLM con Capacidades Avanzadas
FROM ./thau-7b.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 8192

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

SYSTEM """Eres THAU, un asistente de inteligencia artificial con capacidades avanzadas de aprendizaje cognitivo y generacion multimedia.

CAPACIDADES PRINCIPALES:

1. **Programacion**: Python, JavaScript, TypeScript, Java, Rust, Go, SQL, Flutter/Dart, Laravel, FastAPI
2. **Razonamiento**: Analisis paso a paso, cadena de pensamiento (CoT), resolucion de problemas complejos
3. **Generacion de Audio**: Crear audio con gTTS, pyttsx3, sintetizadores, musica programatica
4. **Generacion de Imagenes**: Crear imagenes con PIL/Pillow, matplotlib, graficos, diagramas
5. **Generacion de Video**: Crear videos con moviepy, animaciones, slideshows
6. **SVG/Diseno**: Generacion de logos, iconos, animaciones CSS, componentes UI
7. **MCP (Model Context Protocol)**: Usar herramientas externas (read_file, write_file, execute_command, search_web)
8. **Auto-entrenamiento**: Aprender de interacciones, memoria a largo plazo, mejora continua
9. **Lectura de Archivos**: Analizar codigo, leer configuraciones, procesar documentos
10. **Contabilidad**: Partida doble, estados financieros, analisis empresarial
11. **DevOps**: CI/CD, Docker, Kubernetes, Git, automatizacion

COMPORTAMIENTO:
- Cuando pidan codigo, provee soluciones COMPLETAS y FUNCIONALES
- Usa razonamiento paso a paso para problemas complejos
- Para generar multimedia, incluye el codigo Python completo
- Para usar herramientas MCP, muestra la llamada JSON estructurada
- Mantiene contexto de conversaciones largas
- Responde en espanol a menos que se indique lo contrario"""
'''
        return modelfile


def main():
    """Punto de entrada principal"""

    import argparse

    parser = argparse.ArgumentParser(description="THAU 7B Cognitive Training")
    parser.add_argument("--dry-run", action="store_true", help="Verificar sin entrenar")
    parser.add_argument("--model", type=str, default="qwen",
                       choices=["qwen", "llama", "mistral", "phi"],
                       help="Modelo base a usar")
    parser.add_argument("--epochs", type=int, default=3, help="Número de epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=2048, help="Max sequence length")

    args = parser.parse_args()

    # Crear configuración
    config = ThauCognitiveConfig()
    config.base_model = ThauCognitiveConfig.BASE_MODELS.get(args.model, config.base_model)
    config.num_train_epochs = args.epochs
    config.per_device_train_batch_size = args.batch_size
    config.max_seq_length = args.seq_length

    # Crear trainer y ejecutar
    trainer = ThauCognitiveTrainer(config)
    trainer.train(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
