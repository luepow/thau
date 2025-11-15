"""
Image Generation for THAU
Uses Stable Diffusion to generate images from text prompts
"""

import torch
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
from PIL import Image
import json

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from adapters.device_manager import get_device_manager


class ThauImageGenerator:
    """
    THAU's image generation capability using Stable Diffusion

    Permite a THAU generar imágenes a partir de descripciones en texto
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        output_dir: Path = Path("./data/generated_images"),
        use_half_precision: bool = True
    ):
        """
        Initialize image generator

        Args:
            model_id: HuggingFace model identifier
            output_dir: Directory to save generated images
            use_half_precision: Use FP16 to save memory (recommended)
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device_manager = get_device_manager()
        self.device = self.device_manager.device

        # Metadata
        self.generation_history = []
        self.total_images_generated = 0

        print(f"🎨 Inicializando THAU Image Generator")
        print(f"   Modelo: {model_id}")
        print(f"   Dispositivo: {self.device}")
        print(f"   Half precision: {use_half_precision}")

        # Load pipeline
        self.pipeline = None
        self.use_half_precision = use_half_precision

    def _load_pipeline(self):
        """Load Stable Diffusion pipeline (lazy loading)"""
        if self.pipeline is not None:
            return

        print("📥 Cargando modelo Stable Diffusion...")

        try:
            # Configuración según dispositivo
            if self.device.type == "mps":
                # Apple Silicon
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.use_half_precision else torch.float32,
                )
                self.pipeline.enable_attention_slicing()

            elif self.device.type == "cuda":
                # NVIDIA GPU
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.use_half_precision else torch.float32,
                    variant="fp16" if self.use_half_precision else None,
                )
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipeline.scheduler.config
                )

            else:
                # CPU
                self.pipeline = StableDiffusionPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,
                )

            self.pipeline = self.pipeline.to(self.device)

            print("✅ Modelo cargado exitosamente")

        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "blurry, bad quality, distorted",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        save_image: bool = True
    ) -> Dict:
        """
        Generate an image from a text prompt

        Args:
            prompt: Text description of the image to generate
            negative_prompt: What to avoid in the image
            num_inference_steps: Number of denoising steps (more = better quality, slower)
            guidance_scale: How closely to follow the prompt (7-15 recommended)
            width: Image width in pixels (must be divisible by 8)
            height: Image height in pixels (must be divisible by 8)
            seed: Random seed for reproducibility
            save_image: Whether to save the generated image

        Returns:
            Dictionary with image, path, metadata
        """
        # Load pipeline if not loaded
        self._load_pipeline()

        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"\n🎨 Generando imagen...")
        print(f"   Prompt: {prompt}")
        print(f"   Steps: {num_inference_steps}")

        # Generate image
        try:
            with torch.inference_mode():
                output = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator,
                )

            image = output.images[0]

            # Save image
            image_path = None
            if save_image:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt)
                safe_prompt = safe_prompt[:50]  # Limit filename length

                filename = f"{timestamp}_{safe_prompt}.png"
                image_path = self.output_dir / filename

                image.save(image_path)
                print(f"✅ Imagen guardada: {image_path}")

            # Metadata
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "seed": seed,
                "timestamp": datetime.now().isoformat(),
                "image_path": str(image_path) if image_path else None,
            }

            # Record generation
            self.generation_history.append(metadata)
            self.total_images_generated += 1

            # Save metadata
            if image_path:
                metadata_path = image_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

            return {
                "image": image,
                "path": image_path,
                "metadata": metadata,
                "success": True
            }

        except Exception as e:
            print(f"❌ Error generando imagen: {e}")
            return {
                "image": None,
                "path": None,
                "metadata": None,
                "success": False,
                "error": str(e)
            }

    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Dict]:
        """
        Generate multiple images from a list of prompts

        Args:
            prompts: List of text descriptions
            **kwargs: Arguments passed to generate_image()

        Returns:
            List of generation results
        """
        results = []

        print(f"\n🎨 Generando {len(prompts)} imágenes...")

        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}]")
            result = self.generate_image(prompt, **kwargs)
            results.append(result)

        print(f"\n✅ Generación por lotes completada")
        print(f"   Exitosas: {sum(1 for r in results if r['success'])}/{len(prompts)}")

        return results

    def get_stats(self) -> Dict:
        """Get generation statistics"""
        return {
            "total_images_generated": self.total_images_generated,
            "recent_generations": len(self.generation_history),
            "output_directory": str(self.output_dir),
            "model": self.model_id,
            "device": str(self.device),
        }

    def display_image(self, image_path: Path):
        """
        Display an image (works in Jupyter/IPython)

        Args:
            image_path: Path to image file
        """
        try:
            from IPython.display import display, Image as IPImage
            display(IPImage(filename=str(image_path)))
        except ImportError:
            # Fallback: open with default viewer
            img = Image.open(image_path)
            img.show()


# Helper function for quick generation
def generate_image_quick(prompt: str, **kwargs) -> Dict:
    """
    Quick helper to generate a single image

    Usage:
        result = generate_image_quick("a beautiful sunset over mountains")
        print(f"Image saved to: {result['path']}")
    """
    generator = ThauImageGenerator()
    return generator.generate_image(prompt, **kwargs)


# Testing
if __name__ == "__main__":
    print("="*70)
    print("🧪 Testing THAU Image Generator")
    print("="*70)

    # Initialize generator
    generator = ThauImageGenerator()

    # Test prompts
    test_prompts = [
        "a cute robot learning to paint, digital art",
        "a serene mountain landscape at sunset, photorealistic",
        "an abstract representation of artificial intelligence",
    ]

    print(f"\n🎨 Generating {len(test_prompts)} test images...")

    for prompt in test_prompts:
        result = generator.generate_image(
            prompt=prompt,
            num_inference_steps=20,  # Reduced for faster testing
            width=512,
            height=512,
        )

        if result['success']:
            print(f"✅ Generated: {result['path']}")
        else:
            print(f"❌ Failed: {result.get('error')}")

    # Show stats
    stats = generator.get_stats()
    print(f"\n📊 Statistics:")
    print(json.dumps(stats, indent=2))

    print("\n" + "="*70)
    print("✅ Test completed!")
    print(f"Images saved to: {generator.output_dir}")
