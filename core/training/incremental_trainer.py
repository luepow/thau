"""Incremental trainer for continuous learning from interactions."""

import torch
from typing import List, Dict, Any, Optional
from loguru import logger
from pathlib import Path
import json
from datetime import datetime

from adapters.model_adapter import ModelAdapter
from config.model_configs import LORA_DEFAULT, QUANT_8BIT
from config.base_config import get_config


class IncrementalTrainer:
    """Trainer for incremental learning from user interactions.

    Supports:
    - Fine-tuning with LoRA for efficiency
    - Learning from conversation history
    - Incremental checkpoint saving
    - Memory-efficient training with quantization
    """

    def __init__(
        self,
        model_adapter: Optional[ModelAdapter] = None,
        config=None,
    ):
        """Initialize incremental trainer.

        Args:
            model_adapter: Optional ModelAdapter instance
            config: Configuration object
        """
        self.config = config or get_config()

        # Initialize or use provided model adapter
        if model_adapter is None:
            self.model_adapter = ModelAdapter(
                model_name=self.config.MODEL_NAME,
                use_quantization=self.config.USE_QUANTIZATION,
                quantization_config=QUANT_8BIT if self.config.USE_QUANTIZATION else None,
            )
            # Load model and tokenizer
            self.model_adapter.load_model()
            self.model_adapter.load_tokenizer()
            # Prepare for LoRA
            self.model_adapter.prepare_for_lora(LORA_DEFAULT)
        else:
            self.model_adapter = model_adapter

        self.model = self.model_adapter.model
        self.tokenizer = self.model_adapter.tokenizer

        # Training history
        self.training_history = []

        # Create checkpoint directory
        checkpoint_dir = Path("./data/checkpoints/incremental")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info("IncrementalTrainer initialized")

    def learn_from_interaction(
        self,
        prompt: str,
        response: str,
        feedback: Optional[str] = None,
        num_steps: int = 10,
        learning_rate: float = 5e-5,
    ) -> Dict[str, Any]:
        """Learn from a single interaction.

        Args:
            prompt: User prompt/input
            response: Model response or desired response
            feedback: Optional feedback on the response
            num_steps: Number of training steps
            learning_rate: Learning rate for this update

        Returns:
            Training metrics
        """
        logger.info(f"Learning from interaction: {len(prompt)} chars prompt")

        # Format training text
        if feedback:
            training_text = f"User: {prompt}\nAssistant: {response}\nFeedback: {feedback}"
        else:
            training_text = f"User: {prompt}\nAssistant: {response}"

        # Tokenize
        encodings = self.tokenizer(
            training_text,
            truncation=True,
            max_length=self.config.MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )

        # Move to device
        input_ids = self.model_adapter.device_manager.to_device(encodings["input_ids"])
        labels = input_ids.clone()

        # Set up optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
        )

        # Training loop
        self.model.train()
        losses = []

        for step in range(num_steps):
            optimizer.zero_grad()

            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)

        # Record in history
        self.training_history.append({
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt[:100],  # Store truncated prompt
            "avg_loss": avg_loss,
            "num_steps": num_steps,
        })

        logger.info(f"Learned from interaction - Avg Loss: {avg_loss:.4f}")

        return {
            "avg_loss": avg_loss,
            "losses": losses,
        }

    def learn_from_batch(
        self,
        interactions: List[Dict[str, str]],
        num_epochs: int = 3,
        batch_size: int = 4,
    ) -> Dict[str, Any]:
        """Learn from a batch of interactions.

        Args:
            interactions: List of dicts with 'prompt' and 'response' keys
            num_epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Training metrics
        """
        logger.info(f"Learning from {len(interactions)} interactions")

        # Prepare training data
        texts = [
            f"User: {item['prompt']}\nAssistant: {item['response']}"
            for item in interactions
        ]

        # Tokenize all texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.config.MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )

        # Create simple dataset
        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        # Set up optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
        )

        # Training loop
        self.model.train()
        all_losses = []

        for epoch in range(num_epochs):
            epoch_losses = []

            for batch in dataloader:
                input_ids, attention_mask = batch
                input_ids = self.model_adapter.device_manager.to_device(input_ids)
                attention_mask = self.model_adapter.device_manager.to_device(attention_mask)

                optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )

                loss = outputs.loss
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            all_losses.append(avg_epoch_loss)

            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_epoch_loss:.4f}")

        # Save incremental checkpoint
        self.save_incremental_checkpoint()

        return {
            "avg_loss": sum(all_losses) / len(all_losses),
            "epoch_losses": all_losses,
        }

    def save_incremental_checkpoint(self, name: Optional[str] = None) -> None:
        """Save incremental checkpoint.

        Args:
            name: Optional checkpoint name
        """
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"incremental_{timestamp}"

        checkpoint_dir = Path("./data/checkpoints/incremental") / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapters
        self.model_adapter.save_model(str(checkpoint_dir), save_full_model=False)

        # Save training history
        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        logger.info(f"Incremental checkpoint saved: {checkpoint_dir}")

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics.

        Returns:
            Dictionary of training stats
        """
        if not self.training_history:
            return {"total_interactions": 0}

        losses = [item["avg_loss"] for item in self.training_history]

        return {
            "total_interactions": len(self.training_history),
            "avg_loss": sum(losses) / len(losses),
            "latest_loss": losses[-1] if losses else None,
            "total_steps": sum(item["num_steps"] for item in self.training_history),
        }


if __name__ == "__main__":
    # Test incremental trainer
    print("Testing IncrementalTrainer...")

    trainer = IncrementalTrainer()

    # Test single interaction
    metrics = trainer.learn_from_interaction(
        prompt="What is machine learning?",
        response="Machine learning is a subset of AI that enables systems to learn from data.",
        num_steps=5,
    )

    print(f"Single interaction metrics: {metrics}")

    # Test batch learning
    interactions = [
        {"prompt": "What is Python?", "response": "Python is a programming language."},
        {"prompt": "What is PyTorch?", "response": "PyTorch is a deep learning framework."},
    ]

    batch_metrics = trainer.learn_from_batch(interactions, num_epochs=2, batch_size=2)
    print(f"Batch metrics: {batch_metrics}")

    # Get stats
    stats = trainer.get_training_stats()
    print(f"Training stats: {stats}")

    print("\nTest completed!")
