#!/usr/bin/env python3
"""
THAU Visual Dataset Manager
Combina imágenes de cámara + datasets públicos para entrenamiento visual
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image


class VisualDatasetManager:
    """
    Gestor de dataset visual para THAU

    Combina:
    1. Imágenes capturadas por cámara
    2. CIFAR-10 (dataset público)
    3. Data augmentation
    """

    def __init__(
        self,
        image_size: int = 128,
        data_dir: str = "data/visual_training",
        use_cifar: bool = True,
        use_camera_captures: bool = True,
        augmentation_level: str = "medium",  # low, medium, high
    ):
        """
        Args:
            image_size: Tamaño de imágenes
            data_dir: Directorio base de datos
            use_cifar: Incluir CIFAR-10
            use_camera_captures: Incluir capturas de cámara
            augmentation_level: Nivel de data augmentation
        """
        self.image_size = image_size
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.use_cifar = use_cifar
        self.use_camera_captures = use_camera_captures
        self.augmentation_level = augmentation_level

        # Transforms
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()

        print(f"📦 THAU Visual Dataset Manager")
        print(f"   Image size: {image_size}x{image_size}")
        print(f"   CIFAR-10: {use_cifar}")
        print(f"   Camera captures: {use_camera_captures}")
        print(f"   Augmentation: {augmentation_level}")

    def _get_train_transform(self) -> transforms.Compose:
        """Transform para entrenamiento con augmentation"""
        transform_list = [transforms.Resize((self.image_size, self.image_size))]

        # Augmentation según nivel
        if self.augmentation_level == "low":
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.3),
            ])
        elif self.augmentation_level == "medium":
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        elif self.augmentation_level == "high":
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            ])

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        return transforms.Compose(transform_list)

    def _get_val_transform(self) -> transforms.Compose:
        """Transform para validación (sin augmentation)"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def get_cifar10_dataset(
        self,
        train: bool = True,
        download: bool = True,
    ) -> Optional[Dataset]:
        """
        Obtiene dataset CIFAR-10

        Args:
            train: Dataset de entrenamiento o validación
            download: Descargar si no existe

        Returns:
            Dataset CIFAR-10
        """
        if not self.use_cifar:
            return None

        cifar_dir = self.data_dir / "cifar10"
        cifar_dir.mkdir(parents=True, exist_ok=True)

        transform = self.train_transform if train else self.val_transform

        dataset = torchvision.datasets.CIFAR10(
            root=str(cifar_dir),
            train=train,
            download=download,
            transform=transform,
        )

        return dataset

    def get_camera_dataset(
        self,
        camera_dir: Optional[str] = None,
    ) -> Optional[Dataset]:
        """
        Obtiene dataset de capturas de cámara

        Args:
            camera_dir: Directorio de capturas (None = default)

        Returns:
            Dataset de capturas
        """
        if not self.use_camera_captures:
            return None

        if camera_dir is None:
            camera_dir = self.data_dir / "camera_captures"

        camera_path = Path(camera_dir)

        if not camera_path.exists():
            print(f"⚠️  No se encontró directorio de capturas: {camera_path}")
            return None

        # Cuenta imágenes
        image_files = list(camera_path.glob("*.png")) + list(camera_path.glob("*.jpg"))

        if not image_files:
            print(f"⚠️  No hay imágenes en: {camera_path}")
            return None

        print(f"📷 Capturas de cámara: {len(image_files)} imágenes")

        return CameraDataset(
            image_dir=camera_path,
            transform=self.train_transform,
        )

    def get_combined_dataset(
        self,
        train: bool = True,
        val_split: float = 0.1,
    ) -> Tuple[Dataset, Optional[Dataset]]:
        """
        Obtiene dataset combinado (CIFAR + cámara)

        Args:
            train: Dataset de entrenamiento
            val_split: Proporción de validación

        Returns:
            train_dataset, val_dataset
        """
        datasets = []

        # CIFAR-10
        if self.use_cifar:
            cifar = self.get_cifar10_dataset(train=train, download=True)
            if cifar:
                datasets.append(cifar)
                print(f"✅ CIFAR-10: {len(cifar)} imágenes")

        # Capturas de cámara
        if self.use_camera_captures and train:  # Solo en training
            camera = self.get_camera_dataset()
            if camera:
                datasets.append(camera)
                print(f"✅ Capturas: {len(camera)} imágenes")

        if not datasets:
            raise ValueError("No hay datasets disponibles")

        # Combina datasets
        if len(datasets) == 1:
            combined = datasets[0]
        else:
            combined = ConcatDataset(datasets)

        # Split train/val
        if train and val_split > 0:
            total_size = len(combined)
            val_size = int(total_size * val_split)
            train_size = total_size - val_size

            train_dataset, val_dataset = torch.utils.data.random_split(
                combined,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

            print(f"\n📊 Dataset split:")
            print(f"   Train: {len(train_dataset)} imágenes")
            print(f"   Val: {len(val_dataset)} imágenes")

            return train_dataset, val_dataset
        else:
            print(f"\n📊 Dataset total: {len(combined)} imágenes")
            return combined, None

    def get_dataloader(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 2,
    ) -> DataLoader:
        """
        Crea DataLoader

        Args:
            dataset: Dataset a cargar
            batch_size: Tamaño de batch
            shuffle: Mezclar datos
            num_workers: Workers paralelos

        Returns:
            DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_train_val_loaders(
        self,
        batch_size: int = 32,
        val_split: float = 0.1,
        num_workers: int = 2,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Obtiene loaders de train y val

        Args:
            batch_size: Tamaño de batch
            val_split: Proporción de validación
            num_workers: Workers paralelos

        Returns:
            train_loader, val_loader
        """
        train_dataset, val_dataset = self.get_combined_dataset(
            train=True,
            val_split=val_split,
        )

        train_loader = self.get_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        val_loader = None
        if val_dataset:
            val_loader = self.get_dataloader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        return train_loader, val_loader


class CameraDataset(Dataset):
    """Dataset de imágenes capturadas por cámara"""

    def __init__(self, image_dir: Path, transform: Optional[transforms.Compose] = None):
        """
        Args:
            image_dir: Directorio con imágenes
            transform: Transformaciones
        """
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Lista todas las imágenes
        self.image_files = sorted(
            list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg"))
        )

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Obtiene una imagen

        Returns:
            image: Tensor de imagen
            label: 0 (sin etiqueta específica para VAE)
        """
        img_path = self.image_files[idx]

        # Carga imagen
        image = Image.open(img_path).convert('RGB')

        # Aplica transform
        if self.transform:
            image = self.transform(image)

        # Label dummy (VAE no necesita labels)
        return image, 0


if __name__ == "__main__":
    """Test del dataset manager"""
    print("=" * 80)
    print("THAU Visual Dataset Manager - Test")
    print("=" * 80)

    # Crea manager
    manager = VisualDatasetManager(
        image_size=128,
        use_cifar=True,
        use_camera_captures=True,
        augmentation_level="medium",
    )

    print("\n" + "=" * 80)
    print("Test 1: CIFAR-10 Dataset")
    print("=" * 80)

    cifar_train = manager.get_cifar10_dataset(train=True, download=True)
    if cifar_train:
        print(f"✅ CIFAR-10 Train: {len(cifar_train)} imágenes")

        # Test sample
        img, label = cifar_train[0]
        print(f"   Sample shape: {img.shape}")
        print(f"   Label: {label}")

    print("\n" + "=" * 80)
    print("Test 2: Combined Dataset")
    print("=" * 80)

    train_dataset, val_dataset = manager.get_combined_dataset(
        train=True,
        val_split=0.1,
    )

    print(f"✅ Train dataset: {len(train_dataset)} imágenes")
    if val_dataset:
        print(f"✅ Val dataset: {len(val_dataset)} imágenes")

    print("\n" + "=" * 80)
    print("Test 3: DataLoaders")
    print("=" * 80)

    train_loader, val_loader = manager.get_train_val_loaders(
        batch_size=32,
        val_split=0.1,
        num_workers=0,  # 0 for testing
    )

    print(f"✅ Train loader: {len(train_loader)} batches")
    if val_loader:
        print(f"✅ Val loader: {len(val_loader)} batches")

    # Test batch
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"\n📦 Batch {batch_idx}:")
        print(f"   Images shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Images range: [{images.min():.2f}, {images.max():.2f}]")
        break  # Solo un batch

    print("\n" + "=" * 80)
    print("✅ Todos los tests pasaron!")
    print("=" * 80)
