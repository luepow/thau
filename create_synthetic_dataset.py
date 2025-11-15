#!/usr/bin/env python3
"""
Crea dataset sintético simple para entrenar THAU visual
Genera formas básicas para que THAU aprenda
"""

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import random


def create_synthetic_dataset(
    output_dir: str = "data/visual_training/camera_captures",
    num_images: int = 1000,
    image_size: int = 128,
):
    """
    Crea dataset sintético con formas básicas

    Args:
        output_dir: Directorio de salida
        num_images: Número de imágenes a generar
        image_size: Tamaño de imágenes
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"🎨 Creando dataset sintético")
    print(f"   Output: {output_path}")
    print(f"   Imágenes: {num_images}")
    print(f"   Tamaño: {image_size}x{image_size}")

    shapes = ['circle', 'rectangle', 'triangle', 'ellipse', 'polygon']
    colors = [
        (255, 0, 0),    # Rojo
        (0, 255, 0),    # Verde
        (0, 0, 255),    # Azul
        (255, 255, 0),  # Amarillo
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Naranja
        (128, 0, 255),  # Violeta
    ]

    for i in range(num_images):
        # Crea imagen blanca
        img = Image.new('RGB', (image_size, image_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Selecciona forma y color aleatorio
        shape = random.choice(shapes)
        color = random.choice(colors)

        # Parámetros aleatorios
        x1 = random.randint(10, image_size // 2)
        y1 = random.randint(10, image_size // 2)
        x2 = random.randint(image_size // 2, image_size - 10)
        y2 = random.randint(image_size // 2, image_size - 10)

        # Dibuja forma
        if shape == 'circle':
            draw.ellipse([x1, y1, x2, y2], fill=color)
        elif shape == 'rectangle':
            draw.rectangle([x1, y1, x2, y2], fill=color)
        elif shape == 'triangle':
            points = [(x1, y2), ((x1+x2)//2, y1), (x2, y2)]
            draw.polygon(points, fill=color)
        elif shape == 'ellipse':
            draw.ellipse([x1, y1, x2, y2], fill=color)
        elif shape == 'polygon':
            num_sides = random.randint(5, 8)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = min(x2 - x1, y2 - y1) // 2
            points = []
            for j in range(num_sides):
                angle = 2 * np.pi * j / num_sides
                px = center_x + int(radius * np.cos(angle))
                py = center_y + int(radius * np.sin(angle))
                points.append((px, py))
            draw.polygon(points, fill=color)

        # Guarda
        filename = f"synthetic_{i:05d}_{shape}.png"
        img.save(output_path / filename)

        if (i + 1) % 100 == 0:
            print(f"   Generadas: {i+1}/{num_images}")

    print(f"\n✅ Dataset sintético creado: {num_images} imágenes")


if __name__ == "__main__":
    create_synthetic_dataset(
        num_images=1000,
        image_size=128,
    )
