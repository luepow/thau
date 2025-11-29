"""
THAU-Vision: Main Vision-Language Model
========================================

Complete VLM that combines:
- SigLIP vision encoder for image understanding
- Projection layer for dimension mapping
- TinyLlama LLM for text generation

Supports:
- Image captioning
- Visual question answering
- Image-text conversations
- Multi-image input
- Real-time camera processing
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union, Tuple
from PIL import Image
from pathlib import Path
import json

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from .vision_encoder import VisionEncoder
from .projection import ProjectionLayer


class THAUVisionModel(nn.Module):
    """
    THAU Vision-Language Model.

    Architecture:
    1. Vision Encoder (SigLIP) extracts image features
    2. Projection MLP maps to LLM embedding space
    3. Features concatenated with text embeddings
    4. TinyLlama generates response

    Special tokens:
    - <image>: Placeholder for image features
    - <|im_start|>: Image region start
    - <|im_end|>: Image region end
    """

    # Default configurations
    CONFIGS = {
        "thau-vision-tiny": {
            "vision_encoder": "siglip-base",
            "llm_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "projection_type": "mlp",
            "freeze_vision": True,
            "freeze_llm": False,
        },
        "thau-vision-small": {
            "vision_encoder": "siglip-large",
            "llm_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "projection_type": "mlp_deep",
            "freeze_vision": True,
            "freeze_llm": False,
        },
        "thau-vision-pro": {
            "vision_encoder": "siglip-so400m",
            "llm_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "projection_type": "resampler",
            "num_visual_tokens": 64,
            "freeze_vision": True,
            "freeze_llm": False,
        },
    }

    def __init__(
        self,
        config_name: str = "thau-vision-tiny",
        vision_encoder: Optional[str] = None,
        llm_model: Optional[str] = None,
        projection_type: str = "mlp",
        num_visual_tokens: Optional[int] = None,
        freeze_vision: bool = True,
        freeze_llm: bool = False,
        device: Optional[torch.device] = None,
        load_pretrained: bool = True,
    ):
        """
        Initialize THAU Vision model.

        Args:
            config_name: Predefined config name
            vision_encoder: Override vision encoder name
            llm_model: Override LLM model name
            projection_type: Type of projection layer
            num_visual_tokens: For resampler projection
            freeze_vision: Freeze vision encoder
            freeze_llm: Freeze LLM (only train projection)
            device: Device to use
            load_pretrained: Whether to load pretrained weights
        """
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required. Install: pip install transformers")

        # Get config
        if config_name in self.CONFIGS:
            config = self.CONFIGS[config_name].copy()
        else:
            config = {}

        # Override with explicit params
        self.vision_encoder_name = vision_encoder or config.get("vision_encoder", "siglip-base")
        self.llm_model_name = llm_model or config.get("llm_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.projection_type = projection_type or config.get("projection_type", "mlp")
        self.num_visual_tokens = num_visual_tokens or config.get("num_visual_tokens")
        self.freeze_vision = freeze_vision if freeze_vision is not None else config.get("freeze_vision", True)
        self.freeze_llm = freeze_llm if freeze_llm is not None else config.get("freeze_llm", False)

        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        # Special tokens
        self.image_token = "<image>"
        self.image_start_token = "<|im_start|>"
        self.image_end_token = "<|im_end|>"

        # Load components
        if load_pretrained:
            self._load_components()

    def _load_components(self):
        """Load vision encoder, LLM, and create projection."""
        print(f"Loading THAU-Vision components...")
        print(f"  Vision: {self.vision_encoder_name}")
        print(f"  LLM: {self.llm_model_name}")
        print(f"  Projection: {self.projection_type}")

        # 1. Vision Encoder
        self.vision_encoder = VisionEncoder(
            encoder_name=self.vision_encoder_name,
            freeze=self.freeze_vision,
            device=self.device,
        )

        # 2. LLM
        print(f"Loading LLM: {self.llm_model_name}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)

        # Add special tokens
        special_tokens = {
            "additional_special_tokens": [
                self.image_token,
                self.image_start_token,
                self.image_end_token,
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get LLM embedding dimension
        self.llm_dim = self.llm.config.hidden_size

        # 3. Projection Layer
        self.projection = ProjectionLayer(
            vision_dim=self.vision_encoder.hidden_size,
            llm_dim=self.llm_dim,
            projection_type=self.projection_type,
            num_visual_tokens=self.num_visual_tokens,
        )

        # Move to device
        self.llm = self.llm.to(self.device)
        self.projection = self.projection.to(self.device)

        # Freeze LLM if requested
        if self.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        # Get special token IDs
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        print(f"THAU-Vision loaded successfully!")
        self._print_trainable_params()

    def _print_trainable_params(self):
        """Print trainable parameter count."""
        total = 0
        trainable = 0

        for name, param in self.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()

        print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def encode_images(
        self,
        images: Union[Image.Image, List[Image.Image], torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode images to LLM embedding space.

        Args:
            images: PIL Image(s) or preprocessed tensor

        Returns:
            Image embeddings: [B, num_tokens, llm_dim]
        """
        # Get vision features
        if isinstance(images, torch.Tensor):
            pixel_values = images.to(self.device)
        else:
            pixel_values = self.vision_encoder.preprocess(images)

        vision_outputs = self.vision_encoder(pixel_values)
        vision_features = vision_outputs["patch_embeddings"]

        # Project to LLM space
        image_embeddings = self.projection(vision_features)

        return image_embeddings

    def prepare_inputs(
        self,
        text: str,
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for generation.

        Args:
            text: Input text (should contain <image> token for image position)
            images: Optional image(s)

        Returns:
            Dictionary with input_ids, attention_mask, inputs_embeds
        """
        # Tokenize text
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        if images is None:
            # Text-only input
            return text_inputs

        # Get image embeddings
        image_embeddings = self.encode_images(images)
        batch_size = image_embeddings.shape[0]
        num_image_tokens = image_embeddings.shape[1]

        # Get text embeddings
        text_embeddings = self.llm.get_input_embeddings()(text_inputs["input_ids"])

        # Find image token positions
        image_token_mask = text_inputs["input_ids"] == self.image_token_id

        # Replace image tokens with image embeddings
        # For simplicity, assuming one image per text
        if image_token_mask.any():
            # Create new embeddings with image inserted
            new_embeddings = []
            new_attention_mask = []

            for b in range(batch_size):
                # Find image token position in this sample
                positions = torch.where(image_token_mask[b])[0]

                if len(positions) > 0:
                    pos = positions[0].item()

                    # Split at image token
                    before = text_embeddings[b, :pos]
                    after = text_embeddings[b, pos+1:]

                    # Concatenate: before + image + after
                    combined = torch.cat([
                        before,
                        image_embeddings[b if b < len(image_embeddings) else 0],
                        after,
                    ], dim=0)
                    new_embeddings.append(combined)

                    # Update attention mask
                    before_mask = text_inputs["attention_mask"][b, :pos]
                    after_mask = text_inputs["attention_mask"][b, pos+1:]
                    image_mask = torch.ones(num_image_tokens, device=self.device)
                    combined_mask = torch.cat([before_mask, image_mask, after_mask])
                    new_attention_mask.append(combined_mask)
                else:
                    # No image token, just prepend image
                    combined = torch.cat([
                        image_embeddings[b if b < len(image_embeddings) else 0],
                        text_embeddings[b],
                    ], dim=0)
                    new_embeddings.append(combined)

                    image_mask = torch.ones(num_image_tokens, device=self.device)
                    combined_mask = torch.cat([image_mask, text_inputs["attention_mask"][b]])
                    new_attention_mask.append(combined_mask)

            # Pad to same length
            max_len = max(e.shape[0] for e in new_embeddings)
            padded_embeddings = []
            padded_masks = []

            for e, m in zip(new_embeddings, new_attention_mask):
                pad_len = max_len - e.shape[0]
                if pad_len > 0:
                    # Pad embedding
                    pad_embed = torch.zeros(pad_len, e.shape[1], device=self.device, dtype=e.dtype)
                    e = torch.cat([e, pad_embed], dim=0)
                    # Pad mask
                    pad_mask = torch.zeros(pad_len, device=self.device)
                    m = torch.cat([m, pad_mask])
                padded_embeddings.append(e)
                padded_masks.append(m)

            inputs_embeds = torch.stack(padded_embeddings)
            attention_mask = torch.stack(padded_masks)

            return {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
            }
        else:
            # No image token in text, prepend images
            inputs_embeds = torch.cat([image_embeddings, text_embeddings], dim=1)
            image_mask = torch.ones(batch_size, num_image_tokens, device=self.device)
            attention_mask = torch.cat([image_mask, text_inputs["attention_mask"]], dim=1)

            return {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
            }

    def generate(
        self,
        text: str,
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        Generate text response for image+text input.

        Args:
            text: Input text/question
            images: Optional image(s)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling p
            do_sample: Whether to sample

        Returns:
            Generated text response
        """
        # Format prompt
        if images is not None and self.image_token not in text:
            # Add image token if not present
            text = f"{self.image_token}\n{text}"

        # Prepare inputs
        inputs = self.prepare_inputs(text, images)

        # Generate
        with torch.no_grad():
            if "inputs_embeds" in inputs:
                outputs = self.llm.generate(
                    inputs_embeds=inputs["inputs_embeds"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs,
                )
            else:
                outputs = self.llm.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs,
                )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove input text from response
        if text in response:
            response = response.split(text)[-1].strip()

        return response

    def caption(
        self,
        image: Union[Image.Image, str],
        prompt: str = "Describe this image in detail:",
    ) -> str:
        """
        Generate a caption for an image.

        Args:
            image: PIL Image or path to image
            prompt: Caption prompt

        Returns:
            Image caption
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        return self.generate(prompt, images=image)

    def answer(
        self,
        image: Union[Image.Image, str],
        question: str,
    ) -> str:
        """
        Answer a question about an image.

        Args:
            image: PIL Image or path
            question: Question about the image

        Returns:
            Answer to the question
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        prompt = f"<image>\nQuestion: {question}\nAnswer:"
        return self.generate(prompt, images=image)

    def chat(
        self,
        messages: List[Dict[str, str]],
        images: Optional[List[Image.Image]] = None,
    ) -> str:
        """
        Multi-turn chat with images.

        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            images: Optional images for the conversation

        Returns:
            Assistant response
        """
        # Format conversation
        conversation = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                conversation += f"<|user|>\n{content}</s>\n"
            elif role == "assistant":
                conversation += f"<|assistant|>\n{content}</s>\n"
            elif role == "system":
                conversation += f"<|system|>\n{content}</s>\n"

        # Add assistant prompt
        conversation += "<|assistant|>\n"

        return self.generate(conversation, images=images)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            input_ids: Text token IDs
            attention_mask: Attention mask
            pixel_values: Preprocessed images
            labels: Target labels for loss

        Returns:
            Dictionary with loss and logits
        """
        # Encode images if provided
        if pixel_values is not None:
            image_embeddings = self.encode_images(pixel_values)

            # Get text embeddings
            text_embeddings = self.llm.get_input_embeddings()(input_ids)

            # Combine (prepend images to text)
            batch_size = input_ids.shape[0]
            num_image_tokens = image_embeddings.shape[1]

            inputs_embeds = torch.cat([image_embeddings, text_embeddings], dim=1)

            # Update attention mask
            image_mask = torch.ones(batch_size, num_image_tokens, device=self.device)
            attention_mask = torch.cat([image_mask, attention_mask], dim=1)

            # Update labels if provided
            if labels is not None:
                # Ignore image tokens in loss
                image_labels = torch.full(
                    (batch_size, num_image_tokens),
                    -100,
                    device=self.device,
                    dtype=labels.dtype,
                )
                labels = torch.cat([image_labels, labels], dim=1)

            # Forward through LLM
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        else:
            # Text-only forward
            outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        return outputs

    def save_pretrained(self, save_path: Union[str, Path]):
        """Save model to directory."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save projection layer
        torch.save(
            self.projection.state_dict(),
            save_path / "projection.pt"
        )

        # Save config
        config = {
            "vision_encoder": self.vision_encoder_name,
            "llm_model": self.llm_model_name,
            "projection_type": self.projection_type,
            "num_visual_tokens": self.num_visual_tokens,
            "vision_dim": self.vision_encoder.hidden_size,
            "llm_dim": self.llm_dim,
        }
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)

        print(f"Model saved to: {save_path}")

    @classmethod
    def from_pretrained(cls, load_path: Union[str, Path], **kwargs):
        """Load model from directory."""
        load_path = Path(load_path)

        # Load config
        with open(load_path / "config.json") as f:
            config = json.load(f)

        # Create model
        model = cls(
            vision_encoder=config["vision_encoder"],
            llm_model=config["llm_model"],
            projection_type=config["projection_type"],
            num_visual_tokens=config.get("num_visual_tokens"),
            **kwargs,
        )

        # Load projection weights
        model.projection.load_state_dict(
            torch.load(load_path / "projection.pt", map_location=model.device)
        )

        print(f"Model loaded from: {load_path}")
        return model


# Convenience functions
def create_thau_vision(
    config: str = "thau-vision-tiny",
    **kwargs,
) -> THAUVisionModel:
    """Create a THAU-Vision model."""
    return THAUVisionModel(config_name=config, **kwargs)


# Test
if __name__ == "__main__":
    print("Testing THAU-Vision Model...")

    # Create model
    model = THAUVisionModel(
        config_name="thau-vision-tiny",
        freeze_llm=True,  # Only train projection for testing
    )

    # Test with dummy image
    test_image = Image.new("RGB", (224, 224), color="red")

    # Test caption
    print("\nTesting caption...")
    caption = model.caption(test_image, "What is in this image?")
    print(f"Caption: {caption[:200]}...")

    # Test answer
    print("\nTesting VQA...")
    answer = model.answer(test_image, "What color is this?")
    print(f"Answer: {answer[:200]}...")

    print("\nTHAU-Vision test complete!")
