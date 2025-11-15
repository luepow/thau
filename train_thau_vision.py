#!/usr/bin/env python3
"""
THAU Visual Training - Entrenamiento Progresivo de Capacidad Visual

THAU aprende a generar imágenes desde cero, creciendo progresivamente
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, Optional

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from core.models.visual_vae import ThauVisualVAE, VAE_CONFIGS, vae_loss
from thau_trainer.visual_dataset import VisualDatasetManager
from adapters.device_manager import get_device_manager

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


class VisualTrainer:
    """
    Entrenador progresivo de capacidad visual para THAU

    Entrena VAE progresivamente desde Age 0 hasta Age 15
    """

    def __init__(
        self,
        age: int = 0,
        output_dir: str = "data/checkpoints/thau_vision",
        use_cifar: bool = False,  # Desactivado por ahora (error SSL)
        use_camera: bool = True,
        device: Optional[str] = None,
    ):
        """
        Args:
            age: Edad de THAU a entrenar (0-15)
            output_dir: Directorio de checkpoints
            use_cifar: Usar CIFAR-10
            use_camera: Usar capturas de cámara
            device: Dispositivo (None = auto)
        """
        self.age = age
        self.output_dir = Path(output_dir) / f"age_{age}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device_manager = get_device_manager()
        self.device = self.device_manager.device

        # Modelo
        self.vae = ThauVisualVAE(age=age).to(self.device)
        self.config = VAE_CONFIGS[age]

        # Dataset manager
        self.dataset_manager = VisualDatasetManager(
            image_size=self.config.image_size,
            use_cifar=use_cifar,
            use_camera_captures=use_camera,
            augmentation_level="medium",
        )

        # Training stats
        self.training_stats = {
            "age": age,
            "total_epochs": 0,
            "total_steps": 0,
            "best_loss": float('inf'),
            "history": [],
        }

        logger.info(f"🧠 THAU Visual Trainer - Age {age}")
        logger.info(f"   Model params: {self.vae.total_params:,}")
        logger.info(f"   Image size: {self.config.image_size}x{self.config.image_size}")
        logger.info(f"   Latent dim: {self.config.latent_dim}")
        logger.info(f"   Device: {self.device}")

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        kl_weight: float = 1.0,
    ) -> Dict[str, float]:
        """
        Entrena una época

        Args:
            train_loader: DataLoader de entrenamiento
            optimizer: Optimizador
            kl_weight: Peso de KL divergence (para annealing)

        Returns:
            Dict con losses promedio
        """
        self.vae.train()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Age {self.age} Training")

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)

            # Forward pass
            recon, mu, log_var = self.vae(images)

            # Compute losses
            loss, recon_loss, kl_loss = vae_loss(
                recon, images, mu, log_var, kl_weight=kl_weight
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}',
            })

        # Average losses
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches

        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss,
        }

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        kl_weight: float = 1.0,
    ) -> Dict[str, float]:
        """
        Valida el modelo

        Args:
            val_loader: DataLoader de validación
            kl_weight: Peso de KL divergence

        Returns:
            Dict con losses promedio
        """
        self.vae.eval()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        for images, _ in tqdm(val_loader, desc=f"Age {self.age} Validation"):
            images = images.to(self.device)

            # Forward pass
            recon, mu, log_var = self.vae(images)

            # Compute losses
            loss, recon_loss, kl_loss = vae_loss(
                recon, images, mu, log_var, kl_weight=kl_weight
            )

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1

        # Average losses
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches

        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss,
        }

    def train(
        self,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        kl_annealing_epochs: int = 10,
        val_split: float = 0.1,
        save_every: int = 10,
    ):
        """
        Entrena THAU visual progresivamente

        Args:
            num_epochs: Número de épocas
            batch_size: Tamaño de batch
            learning_rate: Learning rate
            kl_annealing_epochs: Épocas para KL annealing
            val_split: Split de validación
            save_every: Guardar cada N épocas
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🎓 Entrenamiento Visual THAU - Age {self.age}")
        logger.info(f"{'='*80}")
        logger.info(f"Épocas: {num_epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"KL annealing: {kl_annealing_epochs} épocas")

        # Prepare dataset
        train_loader, val_loader = self.dataset_manager.get_train_val_loaders(
            batch_size=batch_size,
            val_split=val_split,
            num_workers=0,  # 0 for MPS compatibility
        )

        # Optimizer
        optimizer = optim.Adam(self.vae.parameters(), lr=learning_rate)

        # Scheduler (reduce LR on plateau)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        # Training loop
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Época {epoch}/{num_epochs}")
            logger.info(f"{'='*80}")

            # KL annealing (aumenta peso gradualmente)
            if epoch <= kl_annealing_epochs:
                kl_weight = epoch / kl_annealing_epochs
            else:
                kl_weight = 1.0

            logger.info(f"KL weight: {kl_weight:.3f}")

            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, kl_weight)

            logger.info(f"\n📊 Train Metrics:")
            logger.info(f"   Loss: {train_metrics['loss']:.4f}")
            logger.info(f"   Recon: {train_metrics['recon_loss']:.4f}")
            logger.info(f"   KL: {train_metrics['kl_loss']:.4f}")

            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader, kl_weight)
                logger.info(f"\n📊 Val Metrics:")
                logger.info(f"   Loss: {val_metrics['loss']:.4f}")
                logger.info(f"   Recon: {val_metrics['recon_loss']:.4f}")
                logger.info(f"   KL: {val_metrics['kl_loss']:.4f}")

                # Scheduler step
                scheduler.step(val_metrics['loss'])

                # Update stats
                is_best = val_metrics['loss'] < self.training_stats['best_loss']
                if is_best:
                    self.training_stats['best_loss'] = val_metrics['loss']
                    logger.info(f"✨ Nuevo mejor modelo!")

            else:
                val_metrics = None
                is_best = train_metrics['loss'] < self.training_stats['best_loss']
                if is_best:
                    self.training_stats['best_loss'] = train_metrics['loss']

            # Save history
            self.training_stats['history'].append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics,
                'kl_weight': kl_weight,
                'lr': optimizer.param_groups[0]['lr'],
            })
            self.training_stats['total_epochs'] = epoch

            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                suffix = "_best" if is_best else f"_epoch_{epoch}"
                self.save_checkpoint(suffix)

            # Generate sample images
            if epoch % save_every == 0:
                self.generate_samples(num_samples=9, suffix=f"_epoch_{epoch}")

        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Entrenamiento completado!")
        logger.info(f"   Best val loss: {self.training_stats['best_loss']:.4f}")
        logger.info(f"   Total épocas: {self.training_stats['total_epochs']}")
        logger.info(f"{'='*80}")

        # Save final
        self.save_checkpoint("_final")
        self.generate_samples(num_samples=16, suffix="_final")

    def save_checkpoint(self, suffix: str = ""):
        """Guarda checkpoint"""
        checkpoint_path = self.output_dir / f"vae{suffix}.pt"

        torch.save({
            'age': self.age,
            'model_state_dict': self.vae.state_dict(),
            'training_stats': self.training_stats,
            'config': self.config,
        }, checkpoint_path)

        logger.info(f"💾 Checkpoint guardado: {checkpoint_path}")

        # Save stats JSON
        stats_path = self.output_dir / f"stats{suffix}.json"
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)

    def load_checkpoint(self, checkpoint_path: str):
        """Carga checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.vae.load_state_dict(checkpoint['model_state_dict'])
        self.training_stats = checkpoint['training_stats']

        logger.info(f"✅ Checkpoint cargado: {checkpoint_path}")
        logger.info(f"   Época: {self.training_stats['total_epochs']}")
        logger.info(f"   Best loss: {self.training_stats['best_loss']:.4f}")

    def generate_samples(self, num_samples: int = 9, suffix: str = ""):
        """Genera muestras de imágenes"""
        import torchvision.utils as vutils
        from PIL import Image

        self.vae.eval()

        # Genera imágenes
        with torch.no_grad():
            samples = self.vae.generate(num_images=num_samples, device=str(self.device))

        # Desnormaliza [-1, 1] -> [0, 1]
        samples = (samples + 1) / 2

        # Crea grid
        grid = vutils.make_grid(samples, nrow=int(num_samples ** 0.5), padding=2)

        # Convierte a PIL
        grid_np = grid.cpu().permute(1, 2, 0).numpy()
        grid_np = (grid_np * 255).astype('uint8')
        img = Image.fromarray(grid_np)

        # Guarda
        output_path = self.output_dir / f"samples{suffix}.png"
        img.save(output_path)

        logger.info(f"🎨 Muestras guardadas: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train THAU Visual System")
    parser.add_argument("--age", type=int, default=0, choices=[0, 1, 3, 6, 12, 15],
                       help="Age of THAU to train")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--use-cifar", action="store_true",
                       help="Use CIFAR-10 dataset")
    parser.add_argument("--use-camera", action="store_true", default=True,
                       help="Use camera captures")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Resume from checkpoint")

    args = parser.parse_args()

    # Create trainer
    trainer = VisualTrainer(
        age=args.age,
        use_cifar=args.use_cifar,
        use_camera=args.use_camera,
    )

    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Train
    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
