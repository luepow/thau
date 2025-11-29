"""
THAU-Vision: Image Question Answering
=====================================

High-level interface for visual question answering
and image understanding tasks.

Capabilities:
- Image captioning
- Visual Q&A
- Object identification
- Scene description
- OCR and text extraction
- Multi-image comparison
"""

import torch
from typing import Optional, Dict, List, Union, Any
from PIL import Image
from pathlib import Path
import base64
import io


class ImageQA:
    """
    High-level interface for THAU-Vision image understanding.

    Provides easy-to-use methods for common vision tasks.
    """

    def __init__(
        self,
        model=None,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize ImageQA.

        Args:
            model: Pre-loaded THAUVisionModel
            model_path: Path to load model from
            device: Device to use
        """
        self.device = device
        self.model = model

        # Load model if path provided
        if model_path is not None and model is None:
            from ..models import THAUVisionModel
            self.model = THAUVisionModel.from_pretrained(model_path, device=device)

        if self.model is not None:
            self.model.eval()

    def _load_image(self, image: Union[str, Path, Image.Image, bytes]) -> Image.Image:
        """Load image from various sources."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, (str, Path)):
            path = str(image)
            # Check if base64
            if path.startswith("data:image"):
                # Extract base64 data
                data = path.split(",")[1]
                img_bytes = base64.b64decode(data)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            else:
                return Image.open(path).convert("RGB")
        else:
            raise ValueError(f"Unknown image type: {type(image)}")

    def caption(
        self,
        image: Union[str, Path, Image.Image, bytes],
        style: str = "detailed",
        language: str = "es",
    ) -> str:
        """
        Generate a caption for an image.

        Args:
            image: Image to caption
            style: Caption style (brief, detailed, poetic)
            language: Output language (es, en)

        Returns:
            Generated caption
        """
        img = self._load_image(image)

        prompts = {
            "brief": {
                "es": "Describe brevemente esta imagen en una oracion:",
                "en": "Briefly describe this image in one sentence:",
            },
            "detailed": {
                "es": "Describe esta imagen en detalle, incluyendo objetos, colores y ambiente:",
                "en": "Describe this image in detail, including objects, colors, and atmosphere:",
            },
            "poetic": {
                "es": "Describe esta imagen de forma poetica y evocadora:",
                "en": "Describe this image in a poetic and evocative way:",
            },
        }

        prompt = prompts.get(style, prompts["detailed"]).get(language, prompts["detailed"]["es"])

        return self.model.caption(img, prompt)

    def answer(
        self,
        image: Union[str, Path, Image.Image, bytes],
        question: str,
    ) -> str:
        """
        Answer a question about an image.

        Args:
            image: Image to analyze
            question: Question about the image

        Returns:
            Answer to the question
        """
        img = self._load_image(image)
        return self.model.answer(img, question)

    def identify_objects(
        self,
        image: Union[str, Path, Image.Image, bytes],
        detail_level: str = "normal",
    ) -> List[str]:
        """
        Identify objects in an image.

        Args:
            image: Image to analyze
            detail_level: How detailed (basic, normal, detailed)

        Returns:
            List of identified objects
        """
        img = self._load_image(image)

        prompts = {
            "basic": "Lista los objetos principales que ves en esta imagen (solo nombres):",
            "normal": "Identifica todos los objetos visibles en esta imagen:",
            "detailed": "Haz un inventario detallado de todos los objetos, personas y elementos en esta imagen:",
        }

        response = self.model.answer(img, prompts.get(detail_level, prompts["normal"]))

        # Parse response to list
        # Simple parsing - split by commas or newlines
        objects = []
        for part in response.replace("\n", ",").split(","):
            obj = part.strip().strip("-").strip("•").strip()
            if obj and len(obj) > 1:
                objects.append(obj)

        return objects

    def describe_scene(
        self,
        image: Union[str, Path, Image.Image, bytes],
    ) -> Dict[str, Any]:
        """
        Get a comprehensive scene description.

        Args:
            image: Image to analyze

        Returns:
            Dictionary with scene analysis
        """
        img = self._load_image(image)

        # Get various aspects
        result = {
            "summary": self.caption(img, style="brief"),
            "detailed_description": self.caption(img, style="detailed"),
            "objects": self.identify_objects(img),
        }

        # Additional questions
        questions = {
            "setting": "Donde parece estar ubicada esta escena? (interior/exterior, tipo de lugar)",
            "time": "Que momento del dia parece ser? (manana, tarde, noche)",
            "mood": "Que atmosfera o estado de animo transmite esta imagen?",
        }

        for key, question in questions.items():
            result[key] = self.model.answer(img, question)

        return result

    def compare_images(
        self,
        image1: Union[str, Path, Image.Image, bytes],
        image2: Union[str, Path, Image.Image, bytes],
        aspect: str = "general",
    ) -> str:
        """
        Compare two images.

        Args:
            image1: First image
            image2: Second image
            aspect: What to compare (general, objects, colors, mood)

        Returns:
            Comparison description
        """
        img1 = self._load_image(image1)
        img2 = self._load_image(image2)

        # Get descriptions of both
        desc1 = self.caption(img1, style="detailed")
        desc2 = self.caption(img2, style="detailed")

        # Create comparison prompt
        prompts = {
            "general": f"Compara estas dos descripciones de imagenes:\nImagen 1: {desc1}\nImagen 2: {desc2}\nDescribe las similitudes y diferencias:",
            "objects": f"Basandote en estas descripciones:\nImagen 1: {desc1}\nImagen 2: {desc2}\nCompara los objetos presentes en cada imagen:",
            "colors": f"Basandote en estas descripciones:\nImagen 1: {desc1}\nImagen 2: {desc2}\nCompara los colores y tonos de cada imagen:",
            "mood": f"Basandote en estas descripciones:\nImagen 1: {desc1}\nImagen 2: {desc2}\nCompara la atmosfera o estado de animo de cada imagen:",
        }

        prompt = prompts.get(aspect, prompts["general"])

        # Use LLM to generate comparison
        return self.model.generate(prompt, images=None)  # Text-only comparison

    def extract_text(
        self,
        image: Union[str, Path, Image.Image, bytes],
    ) -> str:
        """
        Extract text from an image (OCR).

        Args:
            image: Image containing text

        Returns:
            Extracted text
        """
        img = self._load_image(image)

        prompt = "Lee y transcribe todo el texto visible en esta imagen, manteniendo el formato lo mas posible:"

        return self.model.answer(img, prompt)

    def classify(
        self,
        image: Union[str, Path, Image.Image, bytes],
        categories: List[str],
    ) -> Dict[str, float]:
        """
        Classify image into categories.

        Args:
            image: Image to classify
            categories: List of possible categories

        Returns:
            Dictionary of category -> confidence
        """
        img = self._load_image(image)

        categories_str = ", ".join(categories)
        prompt = f"Clasifica esta imagen en una de las siguientes categorias: {categories_str}. Responde SOLO con el nombre de la categoria mas apropiada."

        response = self.model.answer(img, prompt).strip().lower()

        # Create simple confidence scores
        results = {}
        for cat in categories:
            if cat.lower() in response:
                results[cat] = 1.0
            else:
                results[cat] = 0.0

        # If no exact match, assign to first mentioned
        if sum(results.values()) == 0:
            results[categories[0]] = 0.5  # Uncertain

        return results

    def learn_object(
        self,
        image: Union[str, Path, Image.Image, bytes],
        object_name: str,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Learn to recognize an object from an example image.

        Args:
            image: Example image of the object
            object_name: Name of the object
            description: Optional description

        Returns:
            Learning confirmation and object features
        """
        img = self._load_image(image)

        # Generate detailed description for learning
        features = {
            "name": object_name,
            "visual_description": self.model.answer(
                img,
                f"Describe las caracteristicas visuales de este objeto ({object_name}) para poder reconocerlo en el futuro:"
            ),
        }

        if description:
            features["user_description"] = description

        # Add more details
        features["colors"] = self.model.answer(img, "Que colores tiene este objeto?")
        features["shape"] = self.model.answer(img, "Que forma tiene este objeto?")
        features["size_estimate"] = self.model.answer(img, "Aproximadamente que tamano tiene este objeto?")

        return features

    def batch_process(
        self,
        images: List[Union[str, Path, Image.Image, bytes]],
        task: str = "caption",
        **kwargs,
    ) -> List[Any]:
        """
        Process multiple images with the same task.

        Args:
            images: List of images
            task: Task to perform (caption, identify, describe)
            **kwargs: Additional arguments for the task

        Returns:
            List of results
        """
        results = []
        task_methods = {
            "caption": self.caption,
            "identify": self.identify_objects,
            "describe": self.describe_scene,
            "extract_text": self.extract_text,
        }

        method = task_methods.get(task, self.caption)

        for img in images:
            try:
                result = method(img, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})

        return results


# Convenience function
def create_image_qa(
    model_path: Optional[str] = None,
    **kwargs,
) -> ImageQA:
    """Create an ImageQA instance."""
    return ImageQA(model_path=model_path, **kwargs)


# Test
if __name__ == "__main__":
    print("ImageQA module loaded.")
    print("Use create_image_qa() to create an instance.")
    print("\nAvailable methods:")
    print("  - caption(image)")
    print("  - answer(image, question)")
    print("  - identify_objects(image)")
    print("  - describe_scene(image)")
    print("  - compare_images(image1, image2)")
    print("  - extract_text(image)")
    print("  - classify(image, categories)")
    print("  - learn_object(image, name)")
    print("  - batch_process(images, task)")
