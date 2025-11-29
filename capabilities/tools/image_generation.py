#!/usr/bin/env python3
"""
Generacion de Imagenes para THAU Agent
Conecta con diferentes backends de generacion de imagenes

Backends soportados:
1. Stable Diffusion (Automatic1111 API)
2. ComfyUI API
3. Replicate API (cloud)
4. Pollinations.ai (gratis, sin API key)
"""

import requests
import base64
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import os


class ImageGenerator:
    """Generador de imagenes multi-backend"""

    def __init__(
        self,
        backend: str = "pollinations",  # Default: gratis y sin config
        sd_url: str = "http://127.0.0.1:7860",
        comfy_url: str = "http://127.0.0.1:8188",
        replicate_token: Optional[str] = None,
        output_dir: str = "./data/generated_images"
    ):
        self.backend = backend
        self.sd_url = sd_url
        self.comfy_url = comfy_url
        self.replicate_token = replicate_token or os.getenv("REPLICATE_API_TOKEN")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "ugly, blurry, low quality, distorted",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = -1,
        num_images: int = 1
    ) -> Dict[str, Any]:
        """
        Genera imagenes usando el backend configurado

        Returns:
            Dict con status, paths de imagenes, y metadata
        """
        if self.backend == "stable_diffusion":
            return self._generate_sd(prompt, negative_prompt, width, height, steps, cfg_scale, seed, num_images)
        elif self.backend == "pollinations":
            return self._generate_pollinations(prompt, width, height, num_images)
        elif self.backend == "replicate":
            return self._generate_replicate(prompt, negative_prompt, width, height, num_images)
        else:
            return {"success": False, "error": f"Backend desconocido: {self.backend}"}

    def _generate_sd(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        seed: int,
        num_images: int
    ) -> Dict[str, Any]:
        """Genera via Stable Diffusion (Automatic1111)"""
        try:
            payload = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "batch_size": num_images,
                "sampler_name": "DPM++ 2M Karras"
            }

            response = requests.post(
                f"{self.sd_url}/sdapi/v1/txt2img",
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                images = result.get("images", [])
                saved_paths = []

                for i, img_data in enumerate(images):
                    # Decodificar base64 y guardar
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"sd_{timestamp}_{i}.png"
                    filepath = self.output_dir / filename

                    img_bytes = base64.b64decode(img_data)
                    with open(filepath, "wb") as f:
                        f.write(img_bytes)

                    saved_paths.append(str(filepath))

                return {
                    "success": True,
                    "backend": "stable_diffusion",
                    "images": saved_paths,
                    "count": len(saved_paths),
                    "prompt": prompt,
                    "parameters": {
                        "width": width,
                        "height": height,
                        "steps": steps,
                        "cfg_scale": cfg_scale
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"SD API error: {response.status_code}",
                    "hint": "Asegurate de tener Automatic1111 corriendo con --api"
                }

        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "No se pudo conectar a Stable Diffusion",
                "hint": "Inicia Automatic1111 con: ./webui.sh --api"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_pollinations(
        self,
        prompt: str,
        width: int,
        height: int,
        num_images: int
    ) -> Dict[str, Any]:
        """
        Genera via Pollinations.ai (GRATIS, sin API key)
        https://pollinations.ai - Servicio gratuito de generacion de imagenes
        """
        try:
            saved_paths = []

            for i in range(num_images):
                # Pollinations usa URL con el prompt
                # Formato: https://image.pollinations.ai/prompt/{prompt}?width=X&height=Y
                import urllib.parse
                encoded_prompt = urllib.parse.quote(prompt)
                url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true"

                response = requests.get(url, timeout=60)

                if response.status_code == 200:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"pollinations_{timestamp}_{i}.png"
                    filepath = self.output_dir / filename

                    with open(filepath, "wb") as f:
                        f.write(response.content)

                    saved_paths.append(str(filepath))

            if saved_paths:
                return {
                    "success": True,
                    "backend": "pollinations",
                    "images": saved_paths,
                    "count": len(saved_paths),
                    "prompt": prompt,
                    "url_preview": f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt[:50])}",
                    "note": "Imagenes generadas gratis via Pollinations.ai"
                }
            else:
                return {"success": False, "error": "No se generaron imagenes"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_replicate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_images: int
    ) -> Dict[str, Any]:
        """Genera via Replicate API (requiere token)"""
        if not self.replicate_token:
            return {
                "success": False,
                "error": "REPLICATE_API_TOKEN no configurado",
                "hint": "export REPLICATE_API_TOKEN='r8_...'"
            }

        try:
            import replicate

            output = replicate.run(
                "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                input={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "num_outputs": num_images
                }
            )

            saved_paths = []
            for i, url in enumerate(output):
                response = requests.get(url)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"replicate_{timestamp}_{i}.png"
                filepath = self.output_dir / filename

                with open(filepath, "wb") as f:
                    f.write(response.content)
                saved_paths.append(str(filepath))

            return {
                "success": True,
                "backend": "replicate",
                "images": saved_paths,
                "count": len(saved_paths),
                "prompt": prompt
            }

        except ImportError:
            return {
                "success": False,
                "error": "pip install replicate",
                "hint": "Instala: pip install replicate"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def check_backends(self) -> Dict[str, bool]:
        """Verifica que backends estan disponibles"""
        status = {}

        # Check Stable Diffusion
        try:
            r = requests.get(f"{self.sd_url}/sdapi/v1/sd-models", timeout=2)
            status["stable_diffusion"] = r.status_code == 200
        except:
            status["stable_diffusion"] = False

        # Check Pollinations (siempre disponible)
        try:
            r = requests.head("https://pollinations.ai", timeout=5)
            status["pollinations"] = r.status_code < 400
        except:
            status["pollinations"] = False

        # Check Replicate
        status["replicate"] = bool(self.replicate_token)

        return status


def create_image_tool():
    """Crea la herramienta de generacion de imagenes para el agente"""
    generator = ImageGenerator(backend="pollinations")

    def generate_image(prompt: str, num_images: int = 1) -> Dict[str, Any]:
        """
        Genera imagenes a partir de un prompt de texto.

        Args:
            prompt: Descripcion de la imagen a generar (en ingles preferiblemente)
            num_images: Numero de imagenes a generar (1-4)

        Returns:
            Dict con paths de las imagenes generadas
        """
        # Limitar a 4 imagenes max
        num_images = min(num_images, 4)

        result = generator.generate(
            prompt=prompt,
            num_images=num_images
        )

        return result

    return generate_image


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("  THAU Image Generator - Test")
    print("=" * 60)

    gen = ImageGenerator(backend="pollinations")

    # Check backends
    print("\n1. Verificando backends disponibles...")
    status = gen.check_backends()
    for backend, available in status.items():
        emoji = "ok" if available else "no"
        print(f"   [{emoji}] {backend}")

    # Test generation
    print("\n2. Generando imagen de prueba...")
    result = gen.generate(
        prompt="a cute robot learning to code, digital art style",
        width=512,
        height=512
    )

    if result["success"]:
        print(f"   Imagen generada: {result['images'][0]}")
        print(f"   Backend: {result['backend']}")
    else:
        print(f"   Error: {result['error']}")

    print("\n" + "=" * 60)
