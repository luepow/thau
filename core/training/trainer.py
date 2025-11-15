"""Base trainer for model training."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import json

from adapters.device_manager import DeviceManager
from core.training.optimizer import get_optimizer, get_scheduler
from config.training_configs import TrainingConfig


class Trainer:
    """Base trainer class for model training.

    Handles training loop, evaluation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[TrainingConfig] = None,
        device_manager: Optional[DeviceManager] = None,
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            config: Training configuration
            device_manager: Device manager instance
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config or TrainingConfig()
        self.device_manager = device_manager or DeviceManager()

        # Move model to device
        self.model = self.device_manager.to_device(self.model)

        # Create dataloaders
        self.train_dataloader = self._create_dataloader(train_dataset, shuffle=True)
        self.eval_dataloader = (
            self._create_dataloader(eval_dataset, shuffle=False)
            if eval_dataset else None
        )

        # Calculate training steps
        self.num_training_steps = (
            len(self.train_dataloader) * self.config.num_epochs
            // self.config.gradient_accumulation_steps
        )

        # Create optimizer and scheduler
        self.optimizer = get_optimizer(
            self.model.parameters(),
            optimizer_type=self.config.optimizer_type,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_type=self.config.lr_scheduler_type,
            num_training_steps=self.num_training_steps,
            num_warmup_steps=self.config.warmup_steps,
        )

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Trainer initialized for {self.num_training_steps} steps")

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """Create dataloader."""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
        )

    def train(self) -> Dict[str, Any]:
        """Run training loop.

        Returns:
            Training metrics
        """
        logger.info("Starting training...")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            train_metrics = self._train_epoch()

            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                f"Loss: {train_metrics['loss']:.4f}"
            )

            # Evaluation
            if self.eval_dataloader and (epoch + 1) % 1 == 0:
                eval_metrics = self.evaluate()
                logger.info(f"Eval Loss: {eval_metrics['loss']:.4f}")

                # Save best model
                if eval_metrics['loss'] < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics['loss']
                    self.save_checkpoint("best_model")

            # Save periodic checkpoint
            if (epoch + 1) % max(1, self.config.num_epochs // 3) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}")

        # Save final model
        self.save_checkpoint("final_model")

        logger.info("Training completed!")

        return {"final_loss": train_metrics['loss']}

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch + 1}",
        )

        for batch_idx, batch in enumerate(progress_bar):
            loss = self._training_step(batch, batch_idx)

            total_loss += loss
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss:.4f}"})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {"loss": avg_loss}

    def _training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> float:
        """Perform a single training step."""
        # Move batch to device
        batch = {k: self.device_manager.to_device(v) for k, v in batch.items()}

        # Forward pass
        outputs = self.model(**batch)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights if accumulated enough gradients
        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            # Clip gradients
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            self.global_step += 1

        return loss.item() * self.config.gradient_accumulation_steps

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model."""
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: self.device_manager.to_device(v) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {"loss": avg_loss}

    def save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / name

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "config": self.config.__dict__,
        }

        torch.save(checkpoint, f"{checkpoint_path}.pt")
        logger.info(f"Checkpoint saved: {checkpoint_path}.pt")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device_manager.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["epoch"]

        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
