"""
THAU-Vision: Training System
============================

Complete training pipeline for vision-language models.

Features:
- LoRA fine-tuning for efficient training
- Mixed precision training
- Gradient checkpointing for memory efficiency
- Checkpoint saving and resuming
- Training statistics and logging
- Support for MPS/CUDA/CPU
"""

import os
import json
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Union, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

from .dataset import VisionDataset, VisionDataCollator


class VisionTrainer:
    """
    Trainer for THAU-Vision models.

    Supports:
    - Full fine-tuning (projection + LLM)
    - LoRA fine-tuning (projection only)
    - Frozen backbone training (projection only)
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: VisionDataset,
        val_dataset: Optional[VisionDataset] = None,
        output_dir: Union[str, Path] = "checkpoints/vision",
        # Training params
        batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_steps: int = 1000,
        epochs: Optional[int] = None,
        # LoRA params
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        # Optimization
        fp16: bool = True,
        gradient_checkpointing: bool = False,
        # Logging
        logging_steps: int = 10,
        save_steps: int = 100,
        eval_steps: int = 100,
        # Device
        device: Optional[torch.device] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: THAU-Vision model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory for checkpoints
            batch_size: Training batch size
            gradient_accumulation_steps: Steps to accumulate gradients
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            warmup_steps: LR warmup steps
            max_steps: Maximum training steps
            epochs: Number of epochs (overrides max_steps)
            use_lora: Apply LoRA to LLM
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            fp16: Use mixed precision
            gradient_checkpointing: Use gradient checkpointing
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            device: Training device
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training config
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.epochs = epochs

        # LoRA config
        self.use_lora = use_lora and HAS_PEFT
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # Optimization
        self.fp16 = fp16
        self.gradient_checkpointing = gradient_checkpointing

        # Logging
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps

        # Device
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.training_history: List[Dict] = []

        # Setup
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()

    def _setup_model(self):
        """Setup model with LoRA if enabled."""
        print(f"Setting up model for training...")

        # Apply LoRA to LLM if requested
        if self.use_lora and hasattr(self.model, "llm"):
            print("Applying LoRA to LLM...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            self.model.llm = get_peft_model(self.model.llm, lora_config)
            self.model.llm.print_trainable_parameters()

        # Enable gradient checkpointing
        if self.gradient_checkpointing and hasattr(self.model, "llm"):
            self.model.llm.gradient_checkpointing_enable()

        # Move to device
        self.model = self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    def _setup_data(self):
        """Setup data loaders."""
        # Create collator
        tokenizer = getattr(self.model, "tokenizer", None)
        collator = VisionDataCollator(tokenizer=tokenizer)

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,  # MPS doesn't support multiprocessing
            pin_memory=False,
        )

        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=0,
            )

        # Calculate steps
        if self.epochs is not None:
            self.max_steps = len(self.train_loader) * self.epochs

        print(f"Training data: {len(self.train_dataset)} examples")
        print(f"Batch size: {self.batch_size}")
        print(f"Steps per epoch: {len(self.train_loader)}")
        print(f"Max steps: {self.max_steps}")

    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Separate projection and LLM parameters
        projection_params = []
        llm_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "projection" in name:
                    projection_params.append(param)
                else:
                    llm_params.append(param)

        # Different learning rates
        param_groups = [
            {"params": projection_params, "lr": self.learning_rate},
            {"params": llm_params, "lr": self.learning_rate * 0.1},
        ]

        self.optimizer = AdamW(
            param_groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Cosine scheduler with warmup
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_steps - self.warmup_steps,
        )

    def _warmup_lr(self, step: int):
        """Apply learning rate warmup."""
        if step < self.warmup_steps:
            lr_scale = step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * lr_scale

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single training step."""
        # Move to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward pass
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fp16):
            outputs = self.model(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                pixel_values=batch.get("pixel_values"),
                labels=batch.get("labels"),
            )
            loss = outputs.loss / self.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        return loss.item() * self.gradient_accumulation_steps

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        if self.val_dataset is None:
            return {}

        self.model.eval()
        total_loss = 0
        total_steps = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                    pixel_values=batch.get("pixel_values"),
                    labels=batch.get("labels"),
                )
                total_loss += outputs.loss.item()
                total_steps += 1

        self.model.train()

        avg_loss = total_loss / total_steps if total_steps > 0 else 0
        return {"val_loss": avg_loss}

    def save_checkpoint(self, name: str = "checkpoint"):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / f"{name}_step_{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(checkpoint_dir)
        else:
            torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        torch.save(state, checkpoint_dir / "training_state.pt")

        # Save config
        config = {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "use_lora": self.use_lora,
            "global_step": self.global_step,
            "timestamp": datetime.now().isoformat(),
        }
        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Checkpoint saved: {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: Union[str, Path]):
        """Load training checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)

        # Load model
        if hasattr(self.model, "from_pretrained"):
            self.model = self.model.__class__.from_pretrained(checkpoint_dir)
        else:
            self.model.load_state_dict(torch.load(checkpoint_dir / "model.pt"))

        # Load training state
        state = torch.load(checkpoint_dir / "training_state.pt")
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.best_loss = state["best_loss"]
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])

        print(f"Resumed from step {self.global_step}")

    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("STARTING THAU-VISION TRAINING")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        print("-"*60 + "\n")

        self.model.train()
        start_time = time.time()
        running_loss = 0
        loss_count = 0

        # Training loop
        while self.global_step < self.max_steps:
            self.epoch += 1
            print(f"\n=== Epoch {self.epoch} ===\n")

            for batch_idx, batch in enumerate(self.train_loader):
                if self.global_step >= self.max_steps:
                    break

                # Warmup learning rate
                self._warmup_lr(self.global_step)

                # Training step
                loss = self.train_step(batch)
                running_loss += loss
                loss_count += 1

                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    # Update weights
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Update scheduler
                    if self.global_step >= self.warmup_steps:
                        self.scheduler.step()

                    self.global_step += 1

                    # Logging
                    if self.global_step % self.logging_steps == 0:
                        avg_loss = running_loss / loss_count
                        elapsed = time.time() - start_time
                        speed = self.global_step / elapsed
                        eta = (self.max_steps - self.global_step) / speed if speed > 0 else 0

                        lr = self.optimizer.param_groups[0]["lr"]
                        print(f"Step {self.global_step:5d}/{self.max_steps} | "
                              f"Loss: {loss:.4f} | Avg: {avg_loss:.4f} | "
                              f"LR: {lr:.2e} | ETA: {eta/60:.1f}min")

                        self.training_history.append({
                            "step": self.global_step,
                            "loss": avg_loss,
                            "lr": lr,
                        })

                    # Evaluation
                    if self.global_step % self.eval_steps == 0 and self.val_dataset is not None:
                        eval_results = self.evaluate()
                        print(f"  Validation loss: {eval_results.get('val_loss', 0):.4f}")

                        if eval_results.get("val_loss", float("inf")) < self.best_loss:
                            self.best_loss = eval_results["val_loss"]
                            self.save_checkpoint("best")

                    # Save checkpoint
                    if self.global_step % self.save_steps == 0:
                        self.save_checkpoint()

                    # Clear MPS cache
                    if self.device.type == "mps":
                        torch.mps.empty_cache()

        # Final save
        elapsed = time.time() - start_time
        self.save_checkpoint("final")

        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Total steps: {self.global_step}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Final loss: {running_loss/loss_count if loss_count > 0 else 0:.4f}")
        print(f"Best val loss: {self.best_loss:.4f}")
        print(f"Model saved to: {self.output_dir}")

        # Save training stats
        stats = {
            "total_steps": self.global_step,
            "total_time_seconds": elapsed,
            "final_loss": running_loss / loss_count if loss_count > 0 else 0,
            "best_val_loss": self.best_loss,
            "history": self.training_history,
        }
        with open(self.output_dir / "training_stats.json", "w") as f:
            json.dump(stats, f, indent=2)


# Convenience function
def train_vision_model(
    model,
    train_data: Union[str, List[Dict], VisionDataset],
    val_data: Optional[Union[str, List[Dict], VisionDataset]] = None,
    **kwargs,
) -> VisionTrainer:
    """
    Train a THAU-Vision model.

    Args:
        model: Model to train
        train_data: Training data (path, list, or dataset)
        val_data: Validation data
        **kwargs: Additional trainer arguments

    Returns:
        Trained model via trainer
    """
    # Create datasets
    if isinstance(train_data, VisionDataset):
        train_dataset = train_data
    else:
        train_dataset = VisionDataset(
            data_path=train_data,
            tokenizer=getattr(model, "tokenizer", None),
            processor=lambda img: model.vision_encoder.preprocess(img),
        )

    if val_data is not None and not isinstance(val_data, VisionDataset):
        val_dataset = VisionDataset(
            data_path=val_data,
            tokenizer=getattr(model, "tokenizer", None),
            processor=lambda img: model.vision_encoder.preprocess(img),
        )
    else:
        val_dataset = val_data

    # Create trainer
    trainer = VisionTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        **kwargs,
    )

    # Train
    trainer.train()

    return trainer


# Test
if __name__ == "__main__":
    print("Vision Trainer module loaded.")
    print("Use train_vision_model() to train a THAU-Vision model.")
