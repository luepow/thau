#!/usr/bin/env python3
"""
THAU Visual VAE - Variational Autoencoder for Image Generation
Arquitectura progresiva que crece con THAU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class VAEConfig:
    """Configuración del VAE por edad de THAU"""
    age: int
    image_size: int  # Resolución de imágenes
    latent_dim: int  # Dimensión del espacio latente (imaginación)
    hidden_dims: list  # Dimensiones de capas ocultas
    in_channels: int = 3  # RGB


# Configuraciones progresivas del VAE (crecen con THAU)
VAE_CONFIGS = {
    0: VAEConfig(  # Bebé - Aprende formas básicas
        age=0,
        image_size=32,
        latent_dim=64,
        hidden_dims=[32, 64],
    ),
    1: VAEConfig(  # Infante - Más detalles
        age=1,
        image_size=32,
        latent_dim=128,
        hidden_dims=[32, 64, 128],
    ),
    3: VAEConfig(  # Niño - Formas complejas
        age=3,
        image_size=64,
        latent_dim=256,
        hidden_dims=[32, 64, 128, 256],
    ),
    6: VAEConfig(  # Escolar - Alta resolución
        age=6,
        image_size=64,
        latent_dim=512,
        hidden_dims=[32, 64, 128, 256, 512],
    ),
    12: VAEConfig(  # Adolescente - Muy detallado
        age=12,
        image_size=128,
        latent_dim=768,
        hidden_dims=[32, 64, 128, 256, 512],
    ),
    15: VAEConfig(  # Adulto - Máxima capacidad
        age=15,
        image_size=128,
        latent_dim=1024,
        hidden_dims=[32, 64, 128, 256, 512, 1024],
    ),
}


class ResidualBlock(nn.Module):
    """Bloque residual para mejor aprendizaje"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = F.relu(out)

        return out


class VAEEncoder(nn.Module):
    """
    Encoder: Imagen → Representación Latente (Imaginación)
    Aprende a comprimir imágenes en espacio latente
    """

    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config

        # Capas convolucionales progresivas
        layers = []
        in_channels = config.in_channels

        for hidden_dim in config.hidden_dims:
            layers.append(
                nn.Sequential(
                    ResidualBlock(in_channels, hidden_dim),
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.LeakyReLU(0.2),
                )
            )
            in_channels = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Calcula el tamaño después de convoluciones
        self.flatten_size = self._get_flatten_size()

        # Proyección a espacio latente (μ y σ para reparametrización)
        self.fc_mu = nn.Linear(self.flatten_size, config.latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, config.latent_dim)

    def _get_flatten_size(self) -> int:
        """Calcula tamaño después de convoluciones"""
        with torch.no_grad():
            dummy = torch.zeros(1, self.config.in_channels,
                              self.config.image_size, self.config.image_size)
            x = self.encoder(dummy)
            return x.numel()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass del encoder

        Returns:
            z: Vector latente (imaginación)
            mu: Media de la distribución
            log_var: Log varianza de la distribución
        """
        # Extrae características
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        # Parámetros de distribución gaussiana
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        # Reparametrización: z = μ + σ * ε
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, log_var


class VAEDecoder(nn.Module):
    """
    Decoder: Representación Latente → Imagen
    Genera imágenes desde la imaginación de THAU
    """

    def __init__(self, config: VAEConfig, encoder_flatten_size: int):
        super().__init__()
        self.config = config

        # Proyección desde espacio latente
        self.fc = nn.Linear(config.latent_dim, encoder_flatten_size)

        # Calcula dimensiones para reshape
        num_downsamples = len(config.hidden_dims)
        self.initial_size = config.image_size // (2 ** num_downsamples)
        self.initial_channels = config.hidden_dims[-1]

        # Capas deconvolucionales progresivas
        layers = []
        hidden_dims = list(reversed(config.hidden_dims))

        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1],
                                      3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU(0.2),
                    ResidualBlock(hidden_dims[i+1], hidden_dims[i+1]),
                )
            )

        self.decoder = nn.Sequential(*layers)

        # Capa final: genera imagen RGB
        self.final = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], config.in_channels,
                              3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # Salida en [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del decoder

        Args:
            z: Vector latente (imaginación de THAU)

        Returns:
            Imagen generada
        """
        # Proyecta latente a espacio convolucional
        x = self.fc(z)
        x = x.view(-1, self.initial_channels, self.initial_size, self.initial_size)

        # Decodifica
        x = self.decoder(x)
        x = self.final(x)

        return x


class ThauVisualVAE(nn.Module):
    """
    VAE Visual de THAU - Sistema de Imaginación

    Permite a THAU:
    1. Ver imágenes (encoder)
    2. Aprenderlas en espacio latente (imaginación)
    3. Generar nuevas imágenes (decoder)
    """

    def __init__(self, age: int = 0):
        super().__init__()

        if age not in VAE_CONFIGS:
            raise ValueError(f"Age {age} no válido. Usa: {list(VAE_CONFIGS.keys())}")

        self.config = VAE_CONFIGS[age]
        self.age = age

        # Componentes
        self.encoder = VAEEncoder(self.config)
        self.decoder = VAEDecoder(self.config, self.encoder.flatten_size)

        # Contadores
        self._count_parameters()

    def _count_parameters(self):
        """Cuenta parámetros del modelo"""
        self.total_params = sum(p.numel() for p in self.parameters())
        self.encoder_params = sum(p.numel() for p in self.encoder.parameters())
        self.decoder_params = sum(p.numel() for p in self.decoder.parameters())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass completo

        Args:
            x: Imagen de entrada [B, C, H, W]

        Returns:
            recon: Imagen reconstruida
            mu: Media de distribución latente
            log_var: Log varianza de distribución latente
        """
        z, mu, log_var = self.encoder(x)
        recon = self.decoder(z)
        return recon, mu, log_var

    def generate(self, num_images: int = 1, device: str = 'cpu') -> torch.Tensor:
        """
        Genera imágenes desde imaginación (sampling del espacio latente)

        Args:
            num_images: Número de imágenes a generar
            device: Dispositivo

        Returns:
            Tensor con imágenes generadas [num_images, C, H, W]
        """
        self.eval()
        with torch.no_grad():
            # Sample desde distribución normal
            z = torch.randn(num_images, self.config.latent_dim).to(device)
            images = self.decoder(z)
        return images

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruye una imagen (encode + decode)

        Args:
            x: Imagen de entrada

        Returns:
            Imagen reconstruida
        """
        self.eval()
        with torch.no_grad():
            recon, _, _ = self.forward(x)
        return recon

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Obtiene representación latente de una imagen

        Args:
            x: Imagen de entrada

        Returns:
            Vector latente (imaginación de THAU sobre esa imagen)
        """
        self.eval()
        with torch.no_grad():
            z, _, _ = self.encoder(x)
        return z

    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor,
                   steps: int = 10) -> torch.Tensor:
        """
        Interpola entre dos imágenes en espacio latente
        (Imaginación de THAU mezclando conceptos)

        Args:
            x1, x2: Imágenes de entrada
            steps: Número de pasos de interpolación

        Returns:
            Secuencia de imágenes interpoladas
        """
        self.eval()
        with torch.no_grad():
            # Obtiene latentes
            z1 = self.get_latent(x1)
            z2 = self.get_latent(x2)

            # Interpola
            alphas = torch.linspace(0, 1, steps).to(z1.device)
            interpolated = []

            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                img = self.decoder(z_interp)
                interpolated.append(img)

            return torch.cat(interpolated, dim=0)

    def get_info(self) -> Dict:
        """Información del modelo"""
        return {
            'age': self.age,
            'image_size': self.config.image_size,
            'latent_dim': self.config.latent_dim,
            'total_params': self.total_params,
            'encoder_params': self.encoder_params,
            'decoder_params': self.decoder_params,
            'hidden_dims': self.config.hidden_dims,
        }


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor,
            mu: torch.Tensor, log_var: torch.Tensor,
            kl_weight: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loss del VAE: Reconstrucción + KL Divergence

    Args:
        recon_x: Imagen reconstruida
        x: Imagen original
        mu: Media de distribución latente
        log_var: Log varianza de distribución latente
        kl_weight: Peso de KL divergence (para annealing)

    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')

    # KL divergence loss
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss


if __name__ == "__main__":
    """Test del VAE"""
    print("=" * 80)
    print("THAU Visual VAE - Test")
    print("=" * 80)

    # Test para cada edad
    for age in [0, 1, 3, 6, 12, 15]:
        print(f"\n{'='*80}")
        print(f"Age {age} - {VAE_CONFIGS[age].__class__.__name__}")
        print(f"{'='*80}")

        # Crear modelo
        vae = ThauVisualVAE(age=age)
        info = vae.get_info()

        print(f"📊 Configuración:")
        print(f"   Image size: {info['image_size']}x{info['image_size']}")
        print(f"   Latent dim: {info['latent_dim']}")
        print(f"   Hidden dims: {info['hidden_dims']}")
        print(f"   Total params: {info['total_params']:,}")
        print(f"   Encoder params: {info['encoder_params']:,}")
        print(f"   Decoder params: {info['decoder_params']:,}")

        # Test forward pass
        config = VAE_CONFIGS[age]
        batch_size = 4
        x = torch.randn(batch_size, 3, config.image_size, config.image_size)

        print(f"\n🧪 Test Forward Pass:")
        print(f"   Input shape: {x.shape}")

        recon, mu, log_var = vae(x)
        print(f"   Recon shape: {recon.shape}")
        print(f"   Mu shape: {mu.shape}")
        print(f"   LogVar shape: {log_var.shape}")

        # Test loss
        total_loss, recon_loss, kl_loss = vae_loss(recon, x, mu, log_var)
        print(f"\n📉 Losses:")
        print(f"   Total: {total_loss.item():.4f}")
        print(f"   Recon: {recon_loss.item():.4f}")
        print(f"   KL: {kl_loss.item():.4f}")

        # Test generation
        print(f"\n🎨 Test Generation:")
        generated = vae.generate(num_images=2)
        print(f"   Generated shape: {generated.shape}")

        # Test interpolation
        print(f"\n🔄 Test Interpolation:")
        x1 = x[0:1]
        x2 = x[1:2]
        interpolated = vae.interpolate(x1, x2, steps=5)
        print(f"   Interpolated shape: {interpolated.shape}")

        print(f"\n✅ Age {age} OK!")

    print(f"\n{'='*80}")
    print("✅ Todos los tests pasaron!")
    print("="*80)
