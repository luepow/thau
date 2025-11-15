#!/usr/bin/env python3
"""
THAU Camera Capture System
Sistema de captura de cámara para que THAU aprenda de objetos reales
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict
import json
from PIL import Image
import torchvision.transforms as transforms


class CameraCapture:
    """
    Sistema de captura de cámara para THAU

    Permite a THAU:
    1. Ver objetos reales a través de la webcam
    2. Capturar imágenes para entrenamiento
    3. Crear dataset visual propio
    """

    def __init__(
        self,
        camera_id: int = 0,
        output_dir: str = "data/visual_training/camera_captures",
        image_size: int = 128,
    ):
        """
        Args:
            camera_id: ID de la cámara (0 = default webcam)
            output_dir: Directorio para guardar imágenes
            image_size: Tamaño de imágenes capturadas
        """
        self.camera_id = camera_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size

        # Metadata
        self.metadata_file = self.output_dir / "metadata.json"
        self.metadata = self._load_metadata()

        # Transform para preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        print(f"📷 THAU Camera Capture System")
        print(f"   Output: {self.output_dir}")
        print(f"   Image size: {image_size}x{image_size}")

    def _load_metadata(self) -> Dict:
        """Carga metadata de capturas previas"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"captures": [], "categories": {}}

    def _save_metadata(self):
        """Guarda metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def capture_interactive(
        self,
        num_images: int = 10,
        delay: int = 1000,
        category: Optional[str] = None,
    ) -> List[Path]:
        """
        Captura interactiva con preview

        Args:
            num_images: Número de imágenes a capturar
            delay: Delay entre capturas en ms
            category: Categoría/etiqueta para las imágenes

        Returns:
            Lista de paths de imágenes capturadas
        """
        print(f"\n📷 Iniciando captura interactiva")
        print(f"   Imágenes a capturar: {num_images}")
        print(f"   Categoría: {category or 'sin categoría'}")
        print(f"\nControles:")
        print(f"   ESPACIO: Capturar imagen")
        print(f"   Q: Salir")
        print(f"   A: Auto-captura ({num_images} imágenes)")

        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir cámara {self.camera_id}")

        captured_paths = []
        auto_mode = False
        last_capture = 0

        try:
            while len(captured_paths) < num_images:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error leyendo frame")
                    break

                # Muestra preview
                display_frame = frame.copy()
                cv2.putText(
                    display_frame,
                    f"Capturas: {len(captured_paths)}/{num_images}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                if category:
                    cv2.putText(
                        display_frame,
                        f"Categoria: {category}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2
                    )

                cv2.imshow('THAU Vision - Capturando...', display_frame)

                # Auto-captura
                current_time = cv2.getTickCount()
                if auto_mode and (current_time - last_capture) > delay * cv2.getTickFrequency() / 1000:
                    path = self._save_frame(frame, category)
                    captured_paths.append(path)
                    last_capture = current_time
                    print(f"   ✅ Captura {len(captured_paths)}/{num_images}: {path.name}")

                # Controles
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⚠️  Captura cancelada")
                    break
                elif key == ord(' '):  # Espacio - captura manual
                    path = self._save_frame(frame, category)
                    captured_paths.append(path)
                    print(f"   ✅ Captura {len(captured_paths)}/{num_images}: {path.name}")
                elif key == ord('a'):  # Auto mode
                    auto_mode = not auto_mode
                    mode_str = "activado" if auto_mode else "desactivado"
                    print(f"\n🤖 Modo automático {mode_str}")
                    last_capture = current_time

        finally:
            cap.release()
            cv2.destroyAllWindows()

        print(f"\n✅ Captura completada: {len(captured_paths)} imágenes")
        return captured_paths

    def _save_frame(self, frame: np.ndarray, category: Optional[str] = None) -> Path:
        """Guarda un frame capturado"""
        # Convierte BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        # Filename
        category_prefix = f"{category}_" if category else ""
        filename = f"capture_{category_prefix}{timestamp}.png"
        filepath = self.output_dir / filename

        # Guarda imagen
        Image.fromarray(frame_rgb).save(filepath)

        # Actualiza metadata
        capture_info = {
            "filename": filename,
            "timestamp": timestamp,
            "category": category,
            "image_size": [frame.shape[1], frame.shape[0]],
        }
        self.metadata["captures"].append(capture_info)

        # Actualiza contador de categorías
        if category:
            if category not in self.metadata["categories"]:
                self.metadata["categories"][category] = 0
            self.metadata["categories"][category] += 1

        self._save_metadata()

        return filepath

    def capture_batch(
        self,
        categories: List[str],
        images_per_category: int = 20,
        delay: int = 2000,
    ) -> Dict[str, List[Path]]:
        """
        Captura batch de imágenes por categoría

        Args:
            categories: Lista de categorías a capturar
            images_per_category: Imágenes por categoría
            delay: Delay entre capturas (ms)

        Returns:
            Dict con paths por categoría
        """
        results = {}

        print(f"\n📷 Captura por lotes")
        print(f"   Categorías: {categories}")
        print(f"   Imágenes por categoría: {images_per_category}")

        for i, category in enumerate(categories, 1):
            print(f"\n{'='*60}")
            print(f"Categoría {i}/{len(categories)}: {category}")
            print(f"{'='*60}")
            print(f"Prepara el objeto: {category}")
            print(f"Presiona cualquier tecla cuando estés listo...")

            # Espera confirmación
            cv2.waitKey(0)

            # Captura
            paths = self.capture_interactive(
                num_images=images_per_category,
                delay=delay,
                category=category,
            )
            results[category] = paths

        return results

    def load_dataset(self, categories: Optional[List[str]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Carga dataset capturado

        Args:
            categories: Categorías a cargar (None = todas)

        Returns:
            images: Tensor [N, C, H, W]
            labels: Tensor [N] con índices de categoría
        """
        # Filtra capturas
        captures = self.metadata["captures"]
        if categories:
            captures = [c for c in captures if c.get("category") in categories]

        if not captures:
            return torch.empty(0, 3, self.image_size, self.image_size), torch.empty(0, dtype=torch.long)

        # Mapea categorías a índices
        all_categories = sorted(set(c.get("category", "unknown") for c in captures))
        category_to_idx = {cat: idx for idx, cat in enumerate(all_categories)}

        # Carga imágenes
        images = []
        labels = []

        for capture in captures:
            filepath = self.output_dir / capture["filename"]
            if not filepath.exists():
                continue

            # Carga y procesa imagen
            img = Image.open(filepath).convert('RGB')
            img_tensor = self.transform(img)
            images.append(img_tensor)

            # Label
            category = capture.get("category", "unknown")
            labels.append(category_to_idx[category])

        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)

        print(f"\n📦 Dataset cargado:")
        print(f"   Total imágenes: {len(images)}")
        print(f"   Categorías: {all_categories}")
        print(f"   Shape: {images.shape}")

        return images, labels

    def get_stats(self) -> Dict:
        """Estadísticas de capturas"""
        total_captures = len(self.metadata["captures"])
        categories = self.metadata.get("categories", {})

        return {
            "total_captures": total_captures,
            "categories": categories,
            "num_categories": len(categories),
            "output_dir": str(self.output_dir),
        }

    def show_sample(self, num_images: int = 9):
        """Muestra muestra de imágenes capturadas"""
        captures = self.metadata["captures"][-num_images:]

        if not captures:
            print("❌ No hay capturas disponibles")
            return

        import matplotlib.pyplot as plt

        rows = int(np.sqrt(len(captures)))
        cols = int(np.ceil(len(captures) / rows))

        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        axes = axes.flatten() if len(captures) > 1 else [axes]

        for i, capture in enumerate(captures):
            filepath = self.output_dir / capture["filename"]
            if not filepath.exists():
                continue

            img = Image.open(filepath)
            axes[i].imshow(img)
            axes[i].set_title(f"{capture.get('category', 'N/A')}")
            axes[i].axis('off')

        # Oculta ejes vacíos
        for i in range(len(captures), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / "sample_preview.png")
        print(f"✅ Preview guardado: {self.output_dir / 'sample_preview.png'}")
        plt.close()


if __name__ == "__main__":
    """Demo del sistema de captura"""
    print("=" * 80)
    print("THAU Camera Capture System - Demo")
    print("=" * 80)

    # Crea sistema
    capture = CameraCapture(
        camera_id=0,
        output_dir="data/visual_training/camera_captures",
        image_size=128,
    )

    # Menú
    print("\n¿Qué deseas hacer?")
    print("1. Captura rápida (10 imágenes, sin categoría)")
    print("2. Captura por categorías")
    print("3. Ver estadísticas")
    print("4. Ver muestra de capturas")
    print("5. Cargar dataset")

    choice = input("\nOpción (1-5): ").strip()

    if choice == "1":
        # Captura rápida
        paths = capture.capture_interactive(num_images=10, delay=1000)
        print(f"\n✅ {len(paths)} imágenes capturadas")

    elif choice == "2":
        # Por categorías
        print("\nIngresa categorías separadas por coma:")
        print("Ejemplo: perro,gato,auto,casa")
        categories_input = input("Categorías: ").strip()
        categories = [c.strip() for c in categories_input.split(",")]

        images_per_cat = int(input("Imágenes por categoría (default 20): ") or "20")

        results = capture.capture_batch(
            categories=categories,
            images_per_category=images_per_cat,
            delay=1500,
        )

        print(f"\n✅ Captura completada:")
        for cat, paths in results.items():
            print(f"   {cat}: {len(paths)} imágenes")

    elif choice == "3":
        # Stats
        stats = capture.get_stats()
        print(f"\n📊 Estadísticas:")
        print(f"   Total capturas: {stats['total_captures']}")
        print(f"   Categorías: {stats['num_categories']}")
        for cat, count in stats['categories'].items():
            print(f"      - {cat}: {count} imágenes")

    elif choice == "4":
        # Muestra
        capture.show_sample(num_images=9)

    elif choice == "5":
        # Carga dataset
        images, labels = capture.load_dataset()
        print(f"\n✅ Dataset cargado:")
        print(f"   Images shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Unique labels: {torch.unique(labels).tolist()}")

    else:
        print("❌ Opción inválida")

    print("\n✅ Demo completado")
