"""
THAU-Vision: Dataset Classes
============================

Dataset and data collation for vision-language training.

Supports:
- Image captioning datasets
- Visual QA datasets
- Multi-turn conversation with images
- Learning from labeled images (object recognition)
"""

import json
import random
from pathlib import Path
from typing import Optional, Dict, List, Union, Any, Callable
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from PIL import Image

try:
    from transformers import PreTrainedTokenizer
except ImportError:
    PreTrainedTokenizer = Any


@dataclass
class VisionExample:
    """A single vision-language example."""
    image_path: Optional[str] = None
    image: Optional[Image.Image] = None
    text: str = ""
    question: Optional[str] = None
    answer: Optional[str] = None
    caption: Optional[str] = None
    labels: Optional[List[str]] = None  # For object labels
    metadata: Optional[Dict] = None


class VisionDataset(Dataset):
    """
    Dataset for vision-language training.

    Supports multiple formats:
    - Captioning: (image, caption)
    - VQA: (image, question, answer)
    - Conversation: (image, messages[])
    - Object learning: (image, labels[])

    Data can be provided as:
    - JSON/JSONL files
    - Directory of images with annotations
    - In-memory list of examples
    """

    def __init__(
        self,
        data_path: Optional[Union[str, Path, List[Dict]]] = None,
        images_dir: Optional[Union[str, Path]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        processor: Optional[Callable] = None,
        max_length: int = 512,
        image_token: str = "<image>",
        format_type: str = "auto",  # auto, caption, vqa, conversation, labels
        augment: bool = True,
        cache_images: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to JSON/JSONL or list of examples
            images_dir: Directory containing images
            tokenizer: Text tokenizer
            processor: Image processor
            max_length: Maximum sequence length
            image_token: Token placeholder for images
            format_type: Type of data format
            augment: Apply data augmentation
            cache_images: Cache loaded images in memory
        """
        self.images_dir = Path(images_dir) if images_dir else None
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.image_token = image_token
        self.format_type = format_type
        self.augment = augment
        self.cache_images = cache_images

        self.examples: List[VisionExample] = []
        self.image_cache: Dict[str, Image.Image] = {}

        # Load data
        if data_path is not None:
            self._load_data(data_path)

    def _load_data(self, data_path: Union[str, Path, List[Dict]]):
        """Load data from various sources."""
        if isinstance(data_path, list):
            # In-memory list
            for item in data_path:
                self.examples.append(self._parse_example(item))

        elif isinstance(data_path, (str, Path)):
            data_path = Path(data_path)

            if data_path.is_dir():
                # Directory of images with annotations
                self._load_from_directory(data_path)

            elif data_path.suffix == ".jsonl":
                # JSONL file
                with open(data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            self.examples.append(self._parse_example(item))

            elif data_path.suffix == ".json":
                # JSON file
                with open(data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            self.examples.append(self._parse_example(item))
                    else:
                        self.examples.append(self._parse_example(data))

        print(f"Loaded {len(self.examples)} vision-language examples")

    def _load_from_directory(self, directory: Path):
        """Load from directory with image/annotation pairs."""
        # Look for annotations file
        annotations_file = directory / "annotations.json"

        if annotations_file.exists():
            with open(annotations_file) as f:
                annotations = json.load(f)

            for item in annotations:
                item["image_path"] = str(directory / item.get("image", ""))
                self.examples.append(self._parse_example(item))
        else:
            # Load images with filename as caption
            for img_path in directory.glob("*.jpg"):
                self.examples.append(VisionExample(
                    image_path=str(img_path),
                    caption=img_path.stem.replace("_", " "),
                ))
            for img_path in directory.glob("*.png"):
                self.examples.append(VisionExample(
                    image_path=str(img_path),
                    caption=img_path.stem.replace("_", " "),
                ))

    def _parse_example(self, item: Dict) -> VisionExample:
        """Parse a dictionary into VisionExample."""
        return VisionExample(
            image_path=item.get("image_path") or item.get("image"),
            text=item.get("text", ""),
            question=item.get("question"),
            answer=item.get("answer"),
            caption=item.get("caption") or item.get("description"),
            labels=item.get("labels") or item.get("objects"),
            metadata=item.get("metadata"),
        )

    def _detect_format(self, example: VisionExample) -> str:
        """Auto-detect the format of an example."""
        if self.format_type != "auto":
            return self.format_type

        if example.question and example.answer:
            return "vqa"
        elif example.caption:
            return "caption"
        elif example.labels:
            return "labels"
        elif example.text:
            return "text"
        else:
            return "caption"

    def _format_example(self, example: VisionExample) -> str:
        """Format example as training text."""
        format_type = self._detect_format(example)

        if format_type == "vqa":
            # Visual Question Answering
            text = f"""<|system|>
Eres THAU-Vision, un asistente AI que puede ver y entender imagenes.</s>
<|user|>
{self.image_token}
{example.question}</s>
<|assistant|>
{example.answer}</s>"""

        elif format_type == "caption":
            # Image captioning
            prompts = [
                "Describe esta imagen:",
                "Que ves en esta imagen?",
                "Describe lo que muestra esta imagen:",
                "Explica el contenido de esta imagen:",
            ]
            prompt = random.choice(prompts) if self.augment else prompts[0]

            text = f"""<|system|>
Eres THAU-Vision, un asistente AI que puede ver y entender imagenes.</s>
<|user|>
{self.image_token}
{prompt}</s>
<|assistant|>
{example.caption}</s>"""

        elif format_type == "labels":
            # Object recognition
            labels_str = ", ".join(example.labels)
            prompts = [
                "Que objetos hay en esta imagen?",
                "Identifica los elementos en esta imagen:",
                "Lista los objetos que ves:",
                "Que puedes identificar en esta imagen?",
            ]
            prompt = random.choice(prompts) if self.augment else prompts[0]

            text = f"""<|system|>
Eres THAU-Vision, un asistente AI que puede ver y entender imagenes.</s>
<|user|>
{self.image_token}
{prompt}</s>
<|assistant|>
En esta imagen puedo identificar: {labels_str}.</s>"""

        else:
            # Generic text
            text = f"""<|system|>
Eres THAU-Vision, un asistente AI que puede ver y entender imagenes.</s>
<|user|>
{self.image_token}
{example.text}</s>"""

        return text

    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and optionally cache an image."""
        if image_path in self.image_cache:
            return self.image_cache[image_path]

        try:
            # Try relative to images_dir first
            if self.images_dir:
                full_path = self.images_dir / image_path
                if full_path.exists():
                    image_path = str(full_path)

            image = Image.open(image_path).convert("RGB")

            if self.cache_images:
                self.image_cache[image_path] = image

            return image

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example."""
        example = self.examples[idx]

        # Load image
        image = None
        if example.image is not None:
            image = example.image
        elif example.image_path:
            image = self._load_image(example.image_path)

        # Format text
        text = self._format_example(example)

        # Process image
        pixel_values = None
        if image is not None and self.processor is not None:
            pixel_values = self.processor(image)
            if isinstance(pixel_values, dict):
                pixel_values = pixel_values.get("pixel_values")
            if pixel_values is not None and len(pixel_values.shape) == 4:
                pixel_values = pixel_values.squeeze(0)

        # Tokenize text
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)
        else:
            input_ids = None
            attention_mask = None

        result = {
            "text": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image": image,
        }

        return result

    def add_example(
        self,
        image: Union[Image.Image, str],
        caption: Optional[str] = None,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ):
        """Add a new example to the dataset."""
        if isinstance(image, str):
            example = VisionExample(
                image_path=image,
                caption=caption,
                question=question,
                answer=answer,
                labels=labels,
            )
        else:
            example = VisionExample(
                image=image,
                caption=caption,
                question=question,
                answer=answer,
                labels=labels,
            )
        self.examples.append(example)

    def save(self, path: Union[str, Path]):
        """Save dataset to JSONL file."""
        path = Path(path)

        with open(path, "w", encoding="utf-8") as f:
            for example in self.examples:
                item = {
                    "image_path": example.image_path,
                    "caption": example.caption,
                    "question": example.question,
                    "answer": example.answer,
                    "labels": example.labels,
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Saved {len(self.examples)} examples to {path}")


class VisionDataCollator:
    """
    Data collator for vision-language training.

    Handles batching of:
    - Text tokens with proper padding
    - Images with stacking
    - Labels for training
    """

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        pad_token_id: int = 0,
        ignore_index: int = -100,
    ):
        """
        Initialize collator.

        Args:
            tokenizer: Tokenizer for padding info
            pad_token_id: ID for padding token
            ignore_index: Label value to ignore in loss
        """
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

        if tokenizer is not None:
            self.pad_token_id = tokenizer.pad_token_id or 0

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of features."""
        batch = {}

        # Collect input_ids
        if features[0].get("input_ids") is not None:
            input_ids = torch.stack([f["input_ids"] for f in features])
            batch["input_ids"] = input_ids

            # Create labels (same as input_ids for causal LM)
            labels = input_ids.clone()
            # Mask padding
            labels[labels == self.pad_token_id] = self.ignore_index
            batch["labels"] = labels

        # Collect attention masks
        if features[0].get("attention_mask") is not None:
            attention_mask = torch.stack([f["attention_mask"] for f in features])
            batch["attention_mask"] = attention_mask

        # Collect images
        pixel_values = [f.get("pixel_values") for f in features]
        if pixel_values[0] is not None:
            # Filter out None values
            pixel_values = [p for p in pixel_values if p is not None]
            if pixel_values:
                batch["pixel_values"] = torch.stack(pixel_values)

        return batch


# Convenience functions
def create_caption_dataset(
    image_caption_pairs: List[tuple],
    **kwargs,
) -> VisionDataset:
    """Create dataset from (image_path, caption) pairs."""
    data = [
        {"image_path": img, "caption": cap}
        for img, cap in image_caption_pairs
    ]
    return VisionDataset(data_path=data, **kwargs)


def create_vqa_dataset(
    image_qa_pairs: List[tuple],
    **kwargs,
) -> VisionDataset:
    """Create dataset from (image_path, question, answer) tuples."""
    data = [
        {"image_path": img, "question": q, "answer": a}
        for img, q, a in image_qa_pairs
    ]
    return VisionDataset(data_path=data, **kwargs)


def create_learning_dataset(
    image_label_pairs: List[tuple],
    **kwargs,
) -> VisionDataset:
    """Create dataset from (image_path, labels[]) pairs for object learning."""
    data = [
        {"image_path": img, "labels": labels}
        for img, labels in image_label_pairs
    ]
    return VisionDataset(data_path=data, **kwargs)


# Test
if __name__ == "__main__":
    print("Testing Vision Dataset...")

    # Create test dataset
    test_data = [
        {
            "image_path": "test.jpg",
            "caption": "A red apple on a table",
        },
        {
            "image_path": "test2.jpg",
            "question": "What color is the apple?",
            "answer": "The apple is red.",
        },
        {
            "image_path": "test3.jpg",
            "labels": ["apple", "table", "knife"],
        },
    ]

    dataset = VisionDataset(data_path=test_data)
    print(f"Dataset size: {len(dataset)}")

    # Test formatting
    for i in range(len(dataset)):
        example = dataset.examples[i]
        text = dataset._format_example(example)
        print(f"\n--- Example {i+1} ---")
        print(text[:300] + "...")

    print("\nVision Dataset test complete!")
