#!/usr/bin/env python3
"""
THAU Visual Inference System
Sistema de inferencia: Texto → Imaginación → Imagen

Permite a THAU generar imágenes desde su imaginación
"""

import sys
import torch
from pathlib import Path
from typing import Optional, List
import numpy as np
from PIL import Image
import json

sys.path.insert(0, str(Path(__file__).parent))

from core.models.visual_vae import ThauVisualVAE, VAE_CONFIGS
from adapters.device_manager import get_device_manager


class ThauVisualInference:
    """
    Sistema de Inferencia Visual de THAU

    Genera imágenes desde la imaginación de THAU
    """

    def __init__(
        self,
        age: int = 0,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            age: Edad de THAU (0-15)
            checkpoint_path: Path al checkpoint (None = último best)
            device: Dispositivo (None = auto)
        """
        self.age = age
        self.config = VAE_CONFIGS[age]

        # Device
        self.device_manager = get_device_manager()
        self.device = self.device_manager.device

        # Carga modelo
        self.vae = ThauVisualVAE(age=age).to(self.device)

        # Carga checkpoint si existe
        if checkpoint_path is None:
            # Intenta cargar el mejor checkpoint
            default_path = Path(f"data/checkpoints/thau_vision/age_{age}/vae_best.pt")
            if default_path.exists():
                checkpoint_path = str(default_path)

        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)
            print(f"✅ Checkpoint cargado: {checkpoint_path}")
        else:
            print(f"⚠️  Sin checkpoint - usando modelo sin entrenar")

        self.vae.eval()

        print(f"🎨 THAU Visual Inference - Age {age}")
        print(f"   Image size: {self.config.image_size}x{self.config.image_size}")
        print(f"   Latent dim: {self.config.latent_dim}")
        print(f"   Device: {self.device}")

    def load_checkpoint(self, checkpoint_path: str):
        """Carga checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.vae.load_state_dict(checkpoint['model_state_dict'])

    def generate_from_imagination(
        self,
        num_images: int = 1,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Genera imágenes desde imaginación aleatoria

        Args:
            num_images: Número de imágenes
            temperature: Control de variabilidad (mayor = más aleatorio)
            seed: Seed para reproducibilidad

        Returns:
            Lista de imágenes PIL
        """
        if seed is not None:
            torch.manual_seed(seed)

        with torch.no_grad():
            # Sample desde distribución normal con temperatura
            z = torch.randn(num_images, self.config.latent_dim).to(self.device) * temperature

            # Genera imágenes
            images_tensor = self.vae.decoder(z)

            # Desnormaliza [-1, 1] → [0, 1]
            images_tensor = (images_tensor + 1) / 2

            # Convierte a PIL
            images = []
            for i in range(num_images):
                img_np = images_tensor[i].cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype('uint8')
                img = Image.fromarray(img_np)
                images.append(img)

        return images

    def generate_from_text(
        self,
        text_prompt: str,
        num_images: int = 1,
        temperature: float = 1.0,
    ) -> List[Image.Image]:
        """
        Genera imágenes desde texto (simplificado por ahora)

        NOTA: Por ahora usa imaginación aleatoria.
        En el futuro se integrará con THAU-2B para generar
        embeddings semánticos del texto.

        Args:
            text_prompt: Texto descriptivo
            num_images: Número de imágenes
            temperature: Control de variabilidad

        Returns:
            Lista de imágenes PIL
        """
        print(f"📝 Prompt: {text_prompt}")
        print(f"⚠️  Nota: Por ahora usando imaginación aleatoria")
        print(f"   (En el futuro: texto → embeddings → latent space)")

        # Por ahora genera aleatoriamente
        # TODO: Integrar con THAU-2B para mapear texto → latent space
        return self.generate_from_imagination(
            num_images=num_images,
            temperature=temperature,
        )

    def interpolate_imagination(
        self,
        start_seed: int,
        end_seed: int,
        steps: int = 10,
    ) -> List[Image.Image]:
        """
        Interpola entre dos "ideas" en la imaginación de THAU

        Args:
            start_seed: Seed inicial
            end_seed: Seed final
            steps: Pasos de interpolación

        Returns:
            Lista de imágenes interpoladas
        """
        # Genera latentes de inicio y fin
        torch.manual_seed(start_seed)
        z_start = torch.randn(1, self.config.latent_dim).to(self.device)

        torch.manual_seed(end_seed)
        z_end = torch.randn(1, self.config.latent_dim).to(self.device)

        # Interpola
        images = []
        alphas = torch.linspace(0, 1, steps)

        with torch.no_grad():
            for alpha in alphas:
                z = (1 - alpha) * z_start + alpha * z_end
                img_tensor = self.vae.decoder(z)

                # Desnormaliza
                img_tensor = (img_tensor + 1) / 2

                # Convierte a PIL
                img_np = img_tensor[0].cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255).astype('uint8')
                img = Image.fromarray(img_np)
                images.append(img)

        return images

    def save_images(
        self,
        images: List[Image.Image],
        output_dir: str = "data/generated_images/thau_visual",
        prefix: str = "thau",
    ) -> List[Path]:
        """
        Guarda imágenes

        Args:
            images: Lista de imágenes PIL
            output_dir: Directorio de salida
            prefix: Prefijo de archivos

        Returns:
            Lista de paths guardados
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        paths = []
        for i, img in enumerate(images):
            filename = f"{prefix}_{i:03d}.png"
            filepath = output_path / filename
            img.save(filepath)
            paths.append(filepath)

        print(f"💾 {len(images)} imágenes guardadas en: {output_path}")
        return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="THAU Visual Inference")
    parser.add_argument("--age", type=int, default=0, choices=[0, 1, 3, 6, 12, 15],
                       help="Age of THAU")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Checkpoint path")
    parser.add_argument("--mode", type=str, default="generate",
                       choices=["generate", "interpolate", "text"],
                       help="Generation mode")
    parser.add_argument("--num-images", type=int, default=9,
                       help="Number of images to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--text", type=str, default="un robot aprendiendo",
                       help="Text prompt (for text mode)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    parser.add_argument("--output-dir", type=str,
                       default="data/generated_images/thau_visual",
                       help="Output directory")

    args = parser.parse_args()

    # Create inference system
    inference = ThauVisualInference(
        age=args.age,
        checkpoint_path=args.checkpoint,
    )

    print(f"\n{'='*80}")
    print(f"Modo: {args.mode}")
    print(f"{'='*80}\n")

    # Generate based on mode
    if args.mode == "generate":
        images = inference.generate_from_imagination(
            num_images=args.num_images,
            temperature=args.temperature,
            seed=args.seed,
        )

    elif args.mode == "interpolate":
        start_seed = args.seed if args.seed else 42
        end_seed = start_seed + 1000
        images = inference.interpolate_imagination(
            start_seed=start_seed,
            end_seed=end_seed,
            steps=args.num_images,
        )

    elif args.mode == "text":
        images = inference.generate_from_text(
            text_prompt=args.text,
            num_images=args.num_images,
            temperature=args.temperature,
        )

    # Save
    paths = inference.save_images(
        images,
        output_dir=args.output_dir,
        prefix=f"thau_age{args.age}_{args.mode}",
    )

    print(f"\n✅ Generación completada!")
    print(f"   Imágenes: {len(paths)}")
    print(f"   Directorio: {args.output_dir}")

    # Create grid
    if len(images) > 1:
        import math
        rows = int(math.sqrt(len(images)))
        cols = math.ceil(len(images) / rows)

        grid_width = cols * inference.config.image_size
        grid_height = rows * inference.config.image_size

        grid = Image.new('RGB', (grid_width, grid_height))

        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            x = col * inference.config.image_size
            y = row * inference.config.image_size
            grid.paste(img, (x, y))

        grid_path = Path(args.output_dir) / f"thau_age{args.age}_{args.mode}_grid.png"
        grid.save(grid_path)
        print(f"   Grid guardado: {grid_path}")
