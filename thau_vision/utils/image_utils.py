"""
THAU-Vision: Image Utilities
============================

Helper functions for image processing, augmentation, and conversion.
"""

import io
import base64
import random
from typing import Optional, Dict, Tuple, Union, List
from pathlib import Path

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np


def load_image(
    source: Union[str, Path, bytes, Image.Image],
    convert_rgb: bool = True,
) -> Image.Image:
    """
    Load image from various sources.

    Args:
        source: Image path, bytes, base64 string, or PIL Image
        convert_rgb: Convert to RGB mode

    Returns:
        PIL Image
    """
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    elif isinstance(source, (str, Path)):
        source = str(source)
        if source.startswith("data:image"):
            # Base64 data URL
            data = source.split(",")[1]
            img_bytes = base64.b64decode(data)
            img = Image.open(io.BytesIO(img_bytes))
        elif source.startswith("http"):
            # URL
            import urllib.request
            with urllib.request.urlopen(source) as response:
                img_bytes = response.read()
            img = Image.open(io.BytesIO(img_bytes))
        else:
            # File path
            img = Image.open(source)
    else:
        raise ValueError(f"Unknown source type: {type(source)}")

    if convert_rgb and img.mode != "RGB":
        img = img.convert("RGB")

    return img


def resize_image(
    image: Image.Image,
    size: Union[int, Tuple[int, int]],
    method: str = "contain",
) -> Image.Image:
    """
    Resize image to target size.

    Args:
        image: Input image
        size: Target size (single int for square, or (width, height))
        method: Resize method ("contain", "cover", "fill", "exact")

    Returns:
        Resized image
    """
    if isinstance(size, int):
        size = (size, size)

    if method == "contain":
        # Fit within size, maintaining aspect ratio
        image.thumbnail(size, Image.Resampling.LANCZOS)
        return image

    elif method == "cover":
        # Cover size, cropping if necessary
        img_ratio = image.width / image.height
        target_ratio = size[0] / size[1]

        if img_ratio > target_ratio:
            # Image is wider, crop width
            new_width = int(image.height * target_ratio)
            left = (image.width - new_width) // 2
            image = image.crop((left, 0, left + new_width, image.height))
        else:
            # Image is taller, crop height
            new_height = int(image.width / target_ratio)
            top = (image.height - new_height) // 2
            image = image.crop((0, top, image.width, top + new_height))

        return image.resize(size, Image.Resampling.LANCZOS)

    elif method == "fill":
        # Fill with padding
        image.thumbnail(size, Image.Resampling.LANCZOS)
        new_image = Image.new("RGB", size, (128, 128, 128))
        x = (size[0] - image.width) // 2
        y = (size[1] - image.height) // 2
        new_image.paste(image, (x, y))
        return new_image

    else:  # exact
        return image.resize(size, Image.Resampling.LANCZOS)


def normalize_image(
    image: Union[Image.Image, np.ndarray],
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Normalize image for model input.

    Args:
        image: PIL Image or numpy array
        mean: Normalization mean (ImageNet default)
        std: Normalization std (ImageNet default)

    Returns:
        Normalized numpy array [C, H, W]
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to float [0, 1]
    image = image.astype(np.float32) / 255.0

    # Normalize
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    image = (image - mean) / std

    # Transpose to [C, H, W]
    image = image.transpose(2, 0, 1)

    return image


def image_to_base64(
    image: Image.Image,
    format: str = "JPEG",
    quality: int = 85,
) -> str:
    """
    Convert PIL Image to base64 string.

    Args:
        image: PIL Image
        format: Output format (JPEG, PNG, WEBP)
        quality: JPEG quality (1-100)

    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()

    if format.upper() == "JPEG":
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(buffer, format=format, quality=quality)
    else:
        image.save(buffer, format=format)

    img_bytes = buffer.getvalue()
    b64_string = base64.b64encode(img_bytes).decode("utf-8")

    # Return as data URL
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{b64_string}"


def base64_to_image(b64_string: str) -> Image.Image:
    """
    Convert base64 string to PIL Image.

    Args:
        b64_string: Base64 encoded string (with or without data URL prefix)

    Returns:
        PIL Image
    """
    # Remove data URL prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]

    img_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_bytes))


def augment_image(
    image: Image.Image,
    augmentations: Optional[List[str]] = None,
    intensity: float = 0.3,
) -> Image.Image:
    """
    Apply data augmentation to image.

    Args:
        image: Input image
        augmentations: List of augmentations to apply. Options:
            - rotate: Random rotation
            - flip: Horizontal flip
            - brightness: Brightness adjustment
            - contrast: Contrast adjustment
            - saturation: Saturation adjustment
            - blur: Gaussian blur
            - noise: Add noise
        intensity: Augmentation intensity (0-1)

    Returns:
        Augmented image
    """
    if augmentations is None:
        augmentations = ["rotate", "flip", "brightness", "contrast"]

    img = image.copy()

    for aug in augmentations:
        if random.random() > 0.5:
            continue

        if aug == "rotate":
            angle = random.uniform(-15 * intensity, 15 * intensity)
            img = img.rotate(angle, fillcolor=(128, 128, 128))

        elif aug == "flip":
            img = ImageOps.mirror(img)

        elif aug == "brightness":
            factor = 1 + random.uniform(-intensity, intensity)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)

        elif aug == "contrast":
            factor = 1 + random.uniform(-intensity, intensity)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)

        elif aug == "saturation":
            factor = 1 + random.uniform(-intensity, intensity)
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(factor)

        elif aug == "blur":
            radius = random.uniform(0, 2 * intensity)
            img = img.filter(ImageFilter.GaussianBlur(radius))

        elif aug == "noise":
            arr = np.array(img)
            noise = np.random.normal(0, 25 * intensity, arr.shape).astype(np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

    return img


def create_thumbnail(
    image: Image.Image,
    size: int = 128,
) -> Image.Image:
    """
    Create a thumbnail of the image.

    Args:
        image: Input image
        size: Thumbnail size

    Returns:
        Thumbnail image
    """
    thumbnail = image.copy()
    thumbnail.thumbnail((size, size), Image.Resampling.LANCZOS)
    return thumbnail


def get_image_info(image: Image.Image) -> Dict:
    """
    Get information about an image.

    Args:
        image: PIL Image

    Returns:
        Dictionary with image info
    """
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
        "aspect_ratio": image.width / image.height,
        "pixels": image.width * image.height,
        "is_landscape": image.width > image.height,
        "is_portrait": image.height > image.width,
        "is_square": image.width == image.height,
    }


def split_image_grid(
    image: Image.Image,
    rows: int = 2,
    cols: int = 2,
) -> List[Image.Image]:
    """
    Split image into a grid of smaller images.

    Args:
        image: Input image
        rows: Number of rows
        cols: Number of columns

    Returns:
        List of image patches
    """
    w, h = image.size
    patch_w = w // cols
    patch_h = h // rows

    patches = []
    for row in range(rows):
        for col in range(cols):
            left = col * patch_w
            top = row * patch_h
            right = left + patch_w
            bottom = top + patch_h
            patch = image.crop((left, top, right, bottom))
            patches.append(patch)

    return patches


def combine_images_grid(
    images: List[Image.Image],
    cols: int = 2,
    padding: int = 10,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Combine multiple images into a grid.

    Args:
        images: List of images
        cols: Number of columns
        padding: Padding between images
        background: Background color

    Returns:
        Combined image
    """
    if not images:
        return Image.new("RGB", (100, 100), background)

    # Calculate grid size
    rows = (len(images) + cols - 1) // cols
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)

    # Create combined image
    total_w = cols * max_w + (cols + 1) * padding
    total_h = rows * max_h + (rows + 1) * padding
    combined = Image.new("RGB", (total_w, total_h), background)

    # Paste images
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        x = padding + col * (max_w + padding)
        y = padding + row * (max_h + padding)
        # Center image in cell
        x += (max_w - img.width) // 2
        y += (max_h - img.height) // 2
        combined.paste(img, (x, y))

    return combined


# Test
if __name__ == "__main__":
    print("Testing Image Utilities...")

    # Create test image
    test_img = Image.new("RGB", (256, 256), color="blue")

    # Test resize
    resized = resize_image(test_img, 128)
    print(f"Resized: {test_img.size} -> {resized.size}")

    # Test base64
    b64 = image_to_base64(test_img)
    restored = base64_to_image(b64)
    print(f"Base64 roundtrip: {restored.size}")

    # Test augment
    augmented = augment_image(test_img)
    print(f"Augmented: {augmented.size}")

    # Test info
    info = get_image_info(test_img)
    print(f"Info: {info}")

    # Test grid
    patches = split_image_grid(test_img, 2, 2)
    print(f"Split into {len(patches)} patches")

    combined = combine_images_grid(patches, 2)
    print(f"Combined: {combined.size}")

    print("\nImage Utilities test complete!")
