"""
Gestor del Modelo LLM Propio de THAU
Maneja el modelo transformer que THAU entrena desde cero
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

from core.models.base_transformer import TinyLLM
from config.model_configs import TransformerConfig, LoRAConfig
from core.tokenizer.tokenizer import Tokenizer
from adapters.device_manager import get_device_manager


class ThauOwnModelManager:
    """
    Gestiona el modelo LLM propio de THAU que crece con su edad cognitiva

    El modelo comienza pequeño (edad 0) y va creciendo en capacidad
    conforme THAU avanza de edad.
    """

    # Configuraciones por edad cognitiva
    AGE_MODEL_CONFIGS = {
        0: {  # Recién nacido - Modelo muy pequeño
            "d_model": 256,
            "n_heads": 4,
            "n_layers": 2,
            "d_ff": 1024,
            "max_seq_length": 512,  # Incrementado de 256 a 512 tokens
            "description": "Modelo básico para conceptos simples"
        },
        1: {  # 1 año - Modelo pequeño
            "d_model": 384,
            "n_heads": 6,
            "n_layers": 3,
            "d_ff": 1536,
            "max_seq_length": 512,
            "description": "Modelo ampliado para oraciones completas"
        },
        3: {  # 3 años - Modelo mediano
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 6,
            "d_ff": 2048,
            "max_seq_length": 1024,
            "description": "Modelo estándar para razonamiento básico"
        },
        6: {  # 6 años - Modelo grande
            "d_model": 768,
            "n_heads": 12,
            "n_layers": 12,
            "d_ff": 3072,
            "max_seq_length": 2048,
            "description": "Modelo avanzado para razonamiento complejo"
        },
        11: {  # 11 años - Modelo pre-adulto
            "d_model": 896,
            "n_heads": 14,
            "n_layers": 18,
            "d_ff": 3584,
            "max_seq_length": 3072,
            "description": "Modelo pre-adulto con capacidad ampliada"
        },
        12: {  # 12+ años - Modelo completo
            "d_model": 1024,
            "n_heads": 16,
            "n_layers": 24,
            "d_ff": 4096,
            "max_seq_length": 4096,
            "description": "Modelo completo con máxima capacidad"
        },
        15: {  # 15+ años - THAU-2B (Modelo adulto)
            "d_model": 2560,
            "n_heads": 32,
            "n_layers": 24,
            "d_ff": 10240,
            "max_seq_length": 4096,
            "description": "THAU-2B: Modelo adulto de 2B parámetros"
        }
    }

    def __init__(
        self,
        checkpoints_dir: Path = Path("./data/model_checkpoints"),
        use_lora: bool = True
    ):
        self.checkpoints_dir = checkpoints_dir
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.use_lora = use_lora
        self.device_manager = get_device_manager()
        self.device = self.device_manager.device

        # Tokenizer
        self.tokenizer = Tokenizer()

        # Modelo actual
        self.model: Optional[TinyLLM] = None
        self.model_config: Optional[TransformerConfig] = None
        self.current_age = 0

        # Estadísticas
        self.training_stats = {
            "total_steps": 0,
            "total_tokens_seen": 0,
            "age_advancements": 0,
            "checkpoints_saved": 0,
            "training_history": []
        }

        print("🤖 ThauOwnModelManager inicializado")
        print(f"   Dispositivo: {self.device}")
        print(f"   LoRA habilitado: {self.use_lora}")

    def get_config_for_age(self, age: int) -> TransformerConfig:
        """Obtiene configuración del modelo según edad cognitiva"""
        # Encontrar configuración más cercana
        suitable_ages = [a for a in self.AGE_MODEL_CONFIGS.keys() if a <= age]

        if not suitable_ages:
            config_age = 0
        else:
            config_age = max(suitable_ages)

        age_config = self.AGE_MODEL_CONFIGS[config_age]

        return TransformerConfig(
            vocab_size=self.tokenizer.vocab_size,
            d_model=age_config["d_model"],
            n_heads=age_config["n_heads"],
            n_layers=age_config["n_layers"],
            d_ff=age_config["d_ff"],
            max_seq_length=age_config["max_seq_length"],
            dropout=0.1,
            attention_dropout=0.1,
            use_rotary_embeddings=True,
            use_flash_attention=False,  # Puede activarse en GPUs compatibles
        )

    def initialize_model(self, cognitive_age: int = 0):
        """
        Inicializa el modelo según la edad cognitiva

        Args:
            cognitive_age: Edad cognitiva de THAU
        """
        print(f"\n🧠 Inicializando modelo propio de THAU (Edad: {cognitive_age})")

        self.current_age = cognitive_age
        self.model_config = self.get_config_for_age(cognitive_age)

        # Crear modelo
        self.model = TinyLLM(self.model_config)
        self.model = self.model.to(self.device)

        # Contar parámetros
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"✅ Modelo inicializado:")
        print(f"   Parámetros totales: {total_params:,}")
        print(f"   Parámetros entrenables: {trainable_params:,}")
        print(f"   Dimensión del modelo: {self.model_config.d_model}")
        print(f"   Capas: {self.model_config.n_layers}")
        print(f"   Cabezas de atención: {self.model_config.n_heads}")
        print(f"   Longitud máxima: {self.model_config.max_seq_length} tokens")

        return self.model

    def train_step(
        self,
        texts: List[str],
        learning_rate: float = 1e-4,
        gradient_accumulation_steps: int = 4
    ) -> Dict:
        """
        Ejecuta un paso de entrenamiento

        Args:
            texts: Lista de textos para entrenar
            learning_rate: Tasa de aprendizaje
            gradient_accumulation_steps: Pasos de acumulación de gradientes

        Returns:
            Diccionario con métricas del entrenamiento
        """
        if self.model is None:
            raise ValueError("Modelo no inicializado. Llama a initialize_model() primero.")

        self.model.train()

        # Optimizer
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01
        )

        total_loss = 0.0
        total_tokens = 0

        optimizer.zero_grad()

        for i, text in enumerate(texts):
            # Tokenizar
            encoded = self.tokenizer.encode(
                text,
                padding=False,
                return_tensors=None,  # Get list instead of tensor
                return_attention_mask=False
            )

            # Extraer input_ids
            tokens = encoded['input_ids']
            # Si es tensor, convertir a lista
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.squeeze().tolist()
            # Si es lista de listas (batched), tomar primera
            if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], list):
                tokens = tokens[0]

            # Truncar si es necesario
            max_len = self.model_config.max_seq_length
            if len(tokens) > max_len:
                tokens = tokens[:max_len]

            # Convertir a tensor
            input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)

            # Forward pass
            try:
                outputs = self.model(input_ids, return_dict=True)

                # Extraer logits del diccionario o tensor
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs

                # Calcular loss (next token prediction)
                # Shift para predecir siguiente token
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()

                # Cross entropy loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # Normalizar por gradient accumulation
                loss = loss / gradient_accumulation_steps

                # Backward
                loss.backward()

                total_loss += loss.item() * gradient_accumulation_steps
                total_tokens += len(tokens)

                # Actualizar cada N pasos
                if (i + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    optimizer.step()
                    optimizer.zero_grad()

            except Exception as e:
                print(f"⚠️  Error en training step: {e}")
                continue

        # Actualizar estadísticas
        avg_loss = total_loss / len(texts) if texts else 0.0

        self.training_stats["total_steps"] += 1
        self.training_stats["total_tokens_seen"] += total_tokens
        self.training_stats["training_history"].append({
            "timestamp": datetime.now().isoformat(),
            "loss": avg_loss,
            "tokens": total_tokens,
            "age": self.current_age
        })

        return {
            "loss": avg_loss,
            "tokens_processed": total_tokens,
            "perplexity": torch.exp(torch.tensor(avg_loss)).item() if avg_loss > 0 else float('inf')
        }

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Genera texto usando el modelo

        Args:
            prompt: Texto inicial
            max_new_tokens: Máximo de tokens a generar
            temperature: Temperatura de sampling
            top_p: Nucleus sampling threshold

        Returns:
            Texto generado
        """
        if self.model is None:
            return "Modelo no inicializado"

        self.model.eval()

        # Tokenizar prompt
        encoded = self.tokenizer.encode(
            prompt,
            padding=False,
            return_tensors=None,
            return_attention_mask=False
        )

        # Extraer tokens
        tokens = encoded['input_ids']
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.squeeze().tolist()
        if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], list):
            tokens = tokens[0]

        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)

        generated_tokens = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass (model returns dict with 'logits' key)
                outputs = self.model(input_ids, return_dict=True)

                # Extraer logits del diccionario
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs

                # Obtener logits del último token
                next_token_logits = logits[0, -1, :]

                # Aplicar temperatura
                next_token_logits = next_token_logits / temperature

                # Top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Eliminar tokens con probabilidad acumulada > top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')

                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Añadir a secuencia
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

                # Parar si genera token de fin
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        # Decodificar
        generated_text = self.tokenizer.decode(generated_tokens)

        return generated_text

    def save_checkpoint(self, checkpoint_name: Optional[str] = None):
        """Guarda checkpoint del modelo"""
        if self.model is None:
            print("⚠️  No hay modelo para guardar")
            return

        if checkpoint_name is None:
            checkpoint_name = f"thau_age_{self.current_age}_step_{self.training_stats['total_steps']}"

        checkpoint_path = self.checkpoints_dir / f"{checkpoint_name}.pt"

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model_config.__dict__,
            "cognitive_age": self.current_age,
            "training_stats": self.training_stats,
            "timestamp": datetime.now().isoformat()
        }

        torch.save(checkpoint, checkpoint_path)

        self.training_stats["checkpoints_saved"] += 1

        print(f"💾 Checkpoint guardado: {checkpoint_path.name}")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path):
        """Carga checkpoint del modelo"""
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint no encontrado: {checkpoint_path}")
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Cargar configuración
        config_dict = checkpoint["model_config"]
        self.model_config = TransformerConfig(**config_dict)
        self.current_age = checkpoint["cognitive_age"]

        # Crear y cargar modelo
        self.model = TinyLLM(self.model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)

        # Cargar estadísticas
        self.training_stats = checkpoint.get("training_stats", self.training_stats)

        print(f"✅ Checkpoint cargado: {checkpoint_path.name}")
        print(f"   Edad cognitiva: {self.current_age}")
        print(f"   Steps: {self.training_stats['total_steps']}")

        return True

    def advance_age(self, new_age: int):
        """
        Avanza el modelo a una nueva edad cognitiva

        Crea un modelo más grande y transfiere conocimiento del anterior
        """
        if new_age <= self.current_age:
            print(f"⚠️  Nueva edad ({new_age}) debe ser mayor que actual ({self.current_age})")
            return

        print(f"\n📈 AVANZANDO EDAD: {self.current_age} → {new_age} años")

        # Guardar checkpoint del modelo actual
        old_checkpoint = self.save_checkpoint(f"before_age_{new_age}")

        # Obtener nueva configuración
        new_config = self.get_config_for_age(new_age)

        # Si el modelo nuevo es más grande, transferir pesos
        if self.model is not None and new_config.d_model > self.model_config.d_model:
            print("🔄 Transfiriendo conocimiento a modelo más grande...")

            # Guardar pesos del modelo antiguo
            old_state = self.model.state_dict()

            # Crear nuevo modelo
            new_model = TinyLLM(new_config).to(self.device)

            # Transferir pesos compatibles
            new_state = new_model.state_dict()

            for name, param in old_state.items():
                if name in new_state:
                    new_param = new_state[name]

                    # Si dimensiones coinciden, copiar directamente
                    if param.shape == new_param.shape:
                        new_state[name] = param
                    # Si nueva dimensión es mayor, copiar parcialmente
                    elif param.ndim == new_param.ndim:
                        try:
                            # Copiar la parte que cabe
                            slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(param.shape, new_param.shape))
                            new_state[name][slices] = param[slices]
                        except:
                            pass  # Skip si no se puede transferir

            new_model.load_state_dict(new_state)
            self.model = new_model

            print("✅ Conocimiento transferido exitosamente")
        else:
            # Modelo nuevo desde cero
            self.model = TinyLLM(new_config).to(self.device)
            print("✅ Nuevo modelo inicializado")

        self.current_age = new_age
        self.model_config = new_config
        self.training_stats["age_advancements"] += 1

        # Guardar nuevo checkpoint
        self.save_checkpoint(f"age_{new_age}_initialized")

        # Mostrar info del nuevo modelo
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n🎉 Modelo avanzado a edad {new_age}:")
        print(f"   Parámetros: {total_params:,}")
        print(f"   Dimensión: {new_config.d_model}")
        print(f"   Capas: {new_config.n_layers}")

    def get_stats(self) -> Dict:
        """Obtiene estadísticas del modelo"""
        if self.model is None:
            return {"status": "no_initialized"}

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "status": "initialized",
            "cognitive_age": self.current_age,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": (total_params * 4) / (1024 * 1024),  # float32 = 4 bytes
            "config": {
                "d_model": self.model_config.d_model,
                "n_layers": self.model_config.n_layers,
                "n_heads": self.model_config.n_heads,
                "max_seq_length": self.model_config.max_seq_length
            },
            "training": self.training_stats
        }


# Testing
if __name__ == "__main__":
    print("🧪 Probando ThauOwnModelManager\n")

    manager = ThauOwnModelManager()

    # Inicializar modelo edad 0
    manager.initialize_model(cognitive_age=0)

    # Mostrar stats
    stats = manager.get_stats()
    print(f"\n📊 Estadísticas:")
    print(json.dumps(stats, indent=2))

    # Simular training
    print("\n🎓 Entrenando con ejemplos...")
    train_texts = [
        "Hola, ¿cómo estás?",
        "Python es un lenguaje de programación",
        "El cielo es azul",
    ]

    result = manager.train_step(train_texts, learning_rate=1e-4)
    print(f"Resultado: Loss={result['loss']:.4f}, Tokens={result['tokens_processed']}")

    # Generar texto
    print("\n💬 Generando texto...")
    prompt = "Hola"
    generated = manager.generate_text(prompt, max_new_tokens=20)
    print(f"Prompt: {prompt}")
    print(f"Generado: {generated}")

    # Guardar checkpoint
    manager.save_checkpoint()

    # Avanzar edad
    print("\n" + "="*70)
    manager.advance_age(new_age=1)

    print("\n✅ Test completado")
