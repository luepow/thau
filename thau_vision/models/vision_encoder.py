"""
THAU-Vision: Vision Encoder
===========================

SigLIP-based vision encoder for extracting image features.
SigLIP is preferred over CLIP for better performance on smaller models.

Features:
- SigLIP-SO400M (high quality, 400M params)
- SigLIP-Base (balanced, 86M params)
- CLIP ViT-B/16 (fallback option)
- Support for multiple image resolutions
- Patch embedding extraction for detailed analysis
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Union
from PIL import Image
from pathlib import Path

# Try different vision model libraries
try:
    from transformers import (
        SiglipVisionModel,
        SiglipImageProcessor,
        CLIPVisionModel,
        CLIPImageProcessor,
        AutoModel,
        AutoProcessor,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class VisionEncoder(nn.Module):
    """
    Vision encoder using SigLIP or CLIP for image feature extraction.

    SigLIP (Sigmoid Loss for Image-text Pre-training) provides better
    features for smaller VLMs compared to CLIP.

    Attributes:
        encoder_type: Type of encoder ("siglip" or "clip")
        model: The vision model
        processor: Image preprocessor
        hidden_size: Output embedding dimension
        patch_size: Size of image patches
        num_patches: Number of patches per image
    """

    # Available encoder configurations
    ENCODERS = {
        "siglip-so400m": {
            "model_name": "google/siglip-so400m-patch14-384",
            "hidden_size": 1152,
            "patch_size": 14,
            "image_size": 384,
        },
        "siglip-base": {
            "model_name": "google/siglip-base-patch16-224",
            "hidden_size": 768,
            "patch_size": 16,
            "image_size": 224,
        },
        "siglip-large": {
            "model_name": "google/siglip-large-patch16-256",
            "hidden_size": 1024,
            "patch_size": 16,
            "image_size": 256,
        },
        "clip-vit-base": {
            "model_name": "openai/clip-vit-base-patch16",
            "hidden_size": 768,
            "patch_size": 16,
            "image_size": 224,
        },
        "clip-vit-large": {
            "model_name": "openai/clip-vit-large-patch14",
            "hidden_size": 1024,
            "patch_size": 14,
            "image_size": 224,
        },
    }

    def __init__(
        self,
        encoder_name: str = "siglip-base",
        freeze: bool = True,
        use_pooled: bool = False,
        output_hidden_states: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the vision encoder.

        Args:
            encoder_name: Name of encoder from ENCODERS dict
            freeze: Whether to freeze encoder weights
            use_pooled: Use pooled output vs patch embeddings
            output_hidden_states: Return intermediate hidden states
            device: Device to load model on
        """
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required. Install with: pip install transformers")

        if encoder_name not in self.ENCODERS:
            raise ValueError(f"Unknown encoder: {encoder_name}. Available: {list(self.ENCODERS.keys())}")

        self.config = self.ENCODERS[encoder_name]
        self.encoder_name = encoder_name
        self.encoder_type = "siglip" if "siglip" in encoder_name else "clip"
        self.hidden_size = self.config["hidden_size"]
        self.patch_size = self.config["patch_size"]
        self.image_size = self.config["image_size"]
        self.use_pooled = use_pooled
        self.output_hidden_states = output_hidden_states
        self.freeze = freeze

        # Calculate number of patches
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        # Load model and processor
        self._load_model()

    def _load_model(self):
        """Load the vision model and processor."""
        model_name = self.config["model_name"]
        print(f"Loading vision encoder: {model_name}")

        try:
            if self.encoder_type == "siglip":
                self.model = SiglipVisionModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                )
                self.processor = SiglipImageProcessor.from_pretrained(model_name)
            else:
                self.model = CLIPVisionModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                )
                self.processor = CLIPImageProcessor.from_pretrained(model_name)

        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print("Falling back to CLIP ViT-Base...")
            self.model = CLIPVisionModel.from_pretrained(
                "openai/clip-vit-base-patch16",
                torch_dtype=torch.float16,
            )
            self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
            self.hidden_size = 768
            self.encoder_type = "clip"

        # Move to device
        self.model = self.model.to(self.device)

        # Freeze if requested
        if self.freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        print(f"Vision encoder loaded: {self.hidden_size}d embeddings, {self.num_patches} patches")

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image], torch.Tensor],
    ) -> torch.Tensor:
        """
        Preprocess images for the encoder.

        Args:
            images: PIL Image(s) or tensor

        Returns:
            Preprocessed pixel values tensor
        """
        if isinstance(images, torch.Tensor):
            return images.to(self.device)

        if isinstance(images, Image.Image):
            images = [images]

        # Use processor
        inputs = self.processor(
            images=images,
            return_tensors="pt",
        )

        return inputs["pixel_values"].to(self.device, dtype=torch.float16)

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract vision features from images.

        Args:
            pixel_values: Preprocessed image tensor [B, C, H, W]
            output_hidden_states: Override instance setting

        Returns:
            Dictionary with:
            - patch_embeddings: [B, num_patches, hidden_size]
            - pooled_output: [B, hidden_size]
            - hidden_states: List of intermediate states (if requested)
        """
        output_hidden = output_hidden_states if output_hidden_states is not None else self.output_hidden_states

        # Move to device
        pixel_values = pixel_values.to(self.device, dtype=torch.float16)

        # Forward through encoder
        with torch.no_grad() if self.freeze else torch.enable_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                output_hidden_states=output_hidden,
            )

        result = {}

        # Get patch embeddings (without CLS token)
        if hasattr(outputs, "last_hidden_state"):
            # Remove CLS token (first position)
            patch_embeddings = outputs.last_hidden_state[:, 1:, :]
            result["patch_embeddings"] = patch_embeddings

        # Get pooled output
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            result["pooled_output"] = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            # Use CLS token as pooled output
            result["pooled_output"] = outputs.last_hidden_state[:, 0, :]

        # Hidden states
        if output_hidden and hasattr(outputs, "hidden_states"):
            result["hidden_states"] = outputs.hidden_states

        return result

    def encode_images(
        self,
        images: Union[Image.Image, List[Image.Image], str, List[str]],
    ) -> torch.Tensor:
        """
        High-level method to encode images.

        Args:
            images: PIL Image(s) or path(s) to images

        Returns:
            Image embeddings [B, num_patches, hidden_size] or [B, hidden_size]
        """
        # Load images if paths
        if isinstance(images, str):
            images = [Image.open(images).convert("RGB")]
        elif isinstance(images, list) and isinstance(images[0], str):
            images = [Image.open(p).convert("RGB") for p in images]
        elif isinstance(images, Image.Image):
            images = [images]

        # Preprocess
        pixel_values = self.preprocess(images)

        # Encode
        outputs = self.forward(pixel_values)

        if self.use_pooled:
            return outputs["pooled_output"]
        else:
            return outputs["patch_embeddings"]

    def get_image_features(
        self,
        images: Union[Image.Image, List[Image.Image]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Get normalized image features for similarity comparisons.

        Args:
            images: PIL Image(s)
            normalize: L2 normalize the features

        Returns:
            Normalized feature vectors
        """
        embeddings = self.encode_images(images)

        # Use pooled representation
        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=1)

        if normalize:
            embeddings = nn.functional.normalize(embeddings, dim=-1)

        return embeddings

    def compute_similarity(
        self,
        images1: Union[Image.Image, List[Image.Image]],
        images2: Union[Image.Image, List[Image.Image]],
    ) -> torch.Tensor:
        """
        Compute cosine similarity between image sets.

        Args:
            images1: First set of images
            images2: Second set of images

        Returns:
            Similarity matrix
        """
        feat1 = self.get_image_features(images1)
        feat2 = self.get_image_features(images2)

        return torch.mm(feat1, feat2.T)


# Convenience functions
def get_vision_encoder(
    encoder_name: str = "siglip-base",
    **kwargs,
) -> VisionEncoder:
    """Get a vision encoder instance."""
    return VisionEncoder(encoder_name, **kwargs)


def list_encoders() -> List[str]:
    """List available encoder configurations."""
    return list(VisionEncoder.ENCODERS.keys())


# Test
if __name__ == "__main__":
    print("Testing Vision Encoder...")
    print(f"Available encoders: {list_encoders()}")

    # Create encoder
    encoder = VisionEncoder(encoder_name="siglip-base", freeze=True)

    # Create test image
    test_image = Image.new("RGB", (224, 224), color="blue")

    # Encode
    embeddings = encoder.encode_images(test_image)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Expected: [1, {encoder.num_patches}, {encoder.hidden_size}]")

    # Test features
    features = encoder.get_image_features(test_image)
    print(f"Feature shape: {features.shape}")

    print("\nVision Encoder test complete!")
