#!/usr/bin/env python3
"""
THAU Hybrid Memory System
Sistema de memoria híbrida RAM + SSD para entrenamiento y inferencia de modelos grandes

Este módulo implementa un sistema de memoria inteligente que:
1. Usa RAM para datos de acceso frecuente (hot data)
2. Usa SSD para datos menos frecuentes (cold data)
3. Implementa memory-mapping para archivos grandes
4. Gestiona automáticamente el swap entre RAM y disco

Inspirado en técnicas de:
- DeepSpeed ZeRO-Offload
- PyTorch Memory-Mapped Tensors
- LRU Cache con persistencia
"""

import os
import sys
import mmap
import pickle
import hashlib
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import OrderedDict
import numpy as np

# Intentar importar torch si está disponible
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class MemoryConfig:
    """Configuración del sistema de memoria híbrida"""

    # Límites de memoria RAM (en GB)
    max_ram_gb: float = 32.0  # Máximo RAM a usar
    ram_threshold_gb: float = 24.0  # Umbral para empezar offload

    # Configuración de disco
    disk_cache_dir: Path = field(default_factory=lambda: Path.home() / ".thau" / "memory_cache")
    max_disk_gb: float = 100.0  # Máximo espacio en disco

    # Configuración de offload
    offload_to_disk: bool = True
    use_memory_mapping: bool = True
    compression_enabled: bool = True

    # Configuración de caché
    hot_cache_size: int = 1000  # Número de items en hot cache
    warm_cache_size: int = 5000  # Número de items en warm cache

    # Configuración de tensores (para entrenamiento)
    offload_optimizer_states: bool = True
    offload_gradients: bool = True
    pin_memory: bool = True  # Para transferencias más rápidas

    def __post_init__(self):
        self.disk_cache_dir = Path(self.disk_cache_dir)
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)


class MemoryTier:
    """Representa un nivel de memoria (RAM o Disco)"""

    def __init__(self, name: str, max_size_bytes: int, is_persistent: bool = False):
        self.name = name
        self.max_size_bytes = max_size_bytes
        self.current_size_bytes = 0
        self.is_persistent = is_persistent
        self.items: OrderedDict = OrderedDict()
        self.access_count: Dict[str, int] = {}
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.items:
                # Mover al final (más reciente)
                self.items.move_to_end(key)
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return self.items[key]
            return None

    def put(self, key: str, value: Any, size_bytes: int) -> bool:
        with self.lock:
            # Verificar si cabe
            while self.current_size_bytes + size_bytes > self.max_size_bytes and self.items:
                # Evictar el item menos usado
                oldest_key = next(iter(self.items))
                self._evict(oldest_key)

            if self.current_size_bytes + size_bytes <= self.max_size_bytes:
                self.items[key] = value
                self.current_size_bytes += size_bytes
                self.access_count[key] = 1
                return True
            return False

    def _evict(self, key: str) -> Optional[Any]:
        if key in self.items:
            value = self.items.pop(key)
            # Estimar tamaño (simplificado)
            size = sys.getsizeof(value)
            self.current_size_bytes = max(0, self.current_size_bytes - size)
            self.access_count.pop(key, None)
            return value
        return None

    def remove(self, key: str) -> bool:
        with self.lock:
            if key in self.items:
                self._evict(key)
                return True
            return False

    @property
    def usage_percent(self) -> float:
        return (self.current_size_bytes / self.max_size_bytes) * 100 if self.max_size_bytes > 0 else 0


class DiskTier(MemoryTier):
    """Tier de memoria en disco con memory-mapping"""

    def __init__(self, cache_dir: Path, max_size_bytes: int, use_mmap: bool = True):
        super().__init__("disk", max_size_bytes, is_persistent=True)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_mmap = use_mmap
        self.mmap_files: Dict[str, mmap.mmap] = {}
        self.file_sizes: Dict[str, int] = {}

        # Cargar índice existente
        self._load_index()

    def _get_file_path(self, key: str) -> Path:
        """Genera path de archivo basado en hash del key"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.bin"

    def _load_index(self):
        """Carga el índice de archivos existentes"""
        index_file = self.cache_dir / "index.pkl"
        if index_file.exists():
            try:
                with open(index_file, 'rb') as f:
                    data = pickle.load(f)
                    self.items = OrderedDict(data.get('items', {}))
                    self.file_sizes = data.get('file_sizes', {})
                    self.current_size_bytes = sum(self.file_sizes.values())
            except Exception:
                pass

    def _save_index(self):
        """Guarda el índice de archivos"""
        index_file = self.cache_dir / "index.pkl"
        try:
            with open(index_file, 'wb') as f:
                pickle.dump({
                    'items': dict(self.items),
                    'file_sizes': self.file_sizes
                }, f)
        except Exception:
            pass

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.items:
                return None

            file_path = self._get_file_path(key)
            if not file_path.exists():
                self.items.pop(key, None)
                return None

            try:
                # Usar memory-mapping si está habilitado
                if self.use_mmap and key in self.mmap_files:
                    mm = self.mmap_files[key]
                    mm.seek(0)
                    return pickle.loads(mm.read())
                else:
                    with open(file_path, 'rb') as f:
                        return pickle.load(f)
            except Exception as e:
                print(f"Error leyendo {key}: {e}")
                return None

    def put(self, key: str, value: Any, size_bytes: int = 0) -> bool:
        with self.lock:
            file_path = self._get_file_path(key)

            try:
                # Serializar valor
                data = pickle.dumps(value)
                actual_size = len(data)

                # Verificar espacio
                while self.current_size_bytes + actual_size > self.max_size_bytes and self.items:
                    oldest_key = next(iter(self.items))
                    self._evict(oldest_key)

                # Escribir a disco
                with open(file_path, 'wb') as f:
                    f.write(data)

                # Actualizar índice
                self.items[key] = str(file_path)
                self.file_sizes[key] = actual_size
                self.current_size_bytes += actual_size

                # Crear memory-map si está habilitado
                if self.use_mmap:
                    self._create_mmap(key, file_path)

                self._save_index()
                return True

            except Exception as e:
                print(f"Error escribiendo {key}: {e}")
                return False

    def _create_mmap(self, key: str, file_path: Path):
        """Crea un memory-map para un archivo"""
        try:
            f = open(file_path, 'r+b')
            mm = mmap.mmap(f.fileno(), 0)
            self.mmap_files[key] = mm
        except Exception:
            pass

    def _evict(self, key: str) -> Optional[Any]:
        if key in self.items:
            file_path = self._get_file_path(key)

            # Cerrar mmap si existe
            if key in self.mmap_files:
                try:
                    self.mmap_files[key].close()
                except:
                    pass
                del self.mmap_files[key]

            # Eliminar archivo
            try:
                if file_path.exists():
                    file_path.unlink()
            except:
                pass

            # Actualizar contadores
            size = self.file_sizes.pop(key, 0)
            self.current_size_bytes = max(0, self.current_size_bytes - size)
            self.items.pop(key, None)

            return True
        return None

    def clear(self):
        """Limpia toda la caché de disco"""
        with self.lock:
            for key in list(self.items.keys()):
                self._evict(key)
            self._save_index()


class HybridMemoryManager:
    """
    Gestor de memoria híbrida RAM + SSD

    Implementa un sistema de tres niveles:
    1. Hot (RAM): Datos de acceso muy frecuente
    2. Warm (RAM): Datos de acceso moderado
    3. Cold (SSD): Datos de acceso infrecuente

    Los datos se mueven automáticamente entre niveles según su uso.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()

        # Calcular tamaños de cada tier
        hot_size = int(self.config.max_ram_gb * 0.3 * 1e9)  # 30% para hot
        warm_size = int(self.config.max_ram_gb * 0.5 * 1e9)  # 50% para warm
        cold_size = int(self.config.max_disk_gb * 1e9)  # 100% disco para cold

        # Crear tiers
        self.hot_tier = MemoryTier("hot", hot_size)
        self.warm_tier = MemoryTier("warm", warm_size)
        self.cold_tier = DiskTier(
            self.config.disk_cache_dir,
            cold_size,
            self.config.use_memory_mapping
        )

        # Estadísticas
        self.stats = {
            "hot_hits": 0,
            "warm_hits": 0,
            "cold_hits": 0,
            "misses": 0,
            "promotions": 0,
            "demotions": 0,
        }

        # Lock para operaciones thread-safe
        self.lock = threading.RLock()

        print(f"🧠 HybridMemoryManager inicializado:")
        print(f"   Hot tier (RAM):  {hot_size / 1e9:.1f} GB")
        print(f"   Warm tier (RAM): {warm_size / 1e9:.1f} GB")
        print(f"   Cold tier (SSD): {cold_size / 1e9:.1f} GB")

    def get(self, key: str) -> Optional[Any]:
        """
        Obtiene un valor de la memoria híbrida

        Busca en orden: hot -> warm -> cold
        Si encuentra en warm/cold, promueve a un tier superior
        """
        with self.lock:
            # Buscar en hot
            value = self.hot_tier.get(key)
            if value is not None:
                self.stats["hot_hits"] += 1
                return value

            # Buscar en warm
            value = self.warm_tier.get(key)
            if value is not None:
                self.stats["warm_hits"] += 1
                # Promover a hot si se accede frecuentemente
                access_count = self.warm_tier.access_count.get(key, 0)
                if access_count >= 3:
                    self._promote(key, value, "warm", "hot")
                return value

            # Buscar en cold (disco)
            value = self.cold_tier.get(key)
            if value is not None:
                self.stats["cold_hits"] += 1
                # Promover a warm
                self._promote(key, value, "cold", "warm")
                return value

            self.stats["misses"] += 1
            return None

    def put(self, key: str, value: Any, priority: str = "auto") -> bool:
        """
        Almacena un valor en la memoria híbrida

        Args:
            key: Clave única
            value: Valor a almacenar
            priority: "hot", "warm", "cold", o "auto" (determina automáticamente)
        """
        with self.lock:
            size = self._estimate_size(value)

            if priority == "auto":
                # Determinar tier basado en tamaño y uso de RAM
                if size < 1e6 and self.hot_tier.usage_percent < 80:  # < 1MB y hay espacio
                    priority = "hot"
                elif size < 10e6 and self.warm_tier.usage_percent < 80:  # < 10MB
                    priority = "warm"
                else:
                    priority = "cold"

            # Almacenar en el tier correspondiente
            if priority == "hot":
                if self.hot_tier.put(key, value, size):
                    return True
                # Fallback a warm
                priority = "warm"

            if priority == "warm":
                if self.warm_tier.put(key, value, size):
                    return True
                # Fallback a cold
                priority = "cold"

            if priority == "cold":
                return self.cold_tier.put(key, value, size)

            return False

    def _promote(self, key: str, value: Any, from_tier: str, to_tier: str):
        """Promueve un valor a un tier superior"""
        size = self._estimate_size(value)

        if to_tier == "hot":
            if self.hot_tier.put(key, value, size):
                self.stats["promotions"] += 1
                if from_tier == "warm":
                    self.warm_tier.remove(key)
                elif from_tier == "cold":
                    self.cold_tier.remove(key)

        elif to_tier == "warm":
            if self.warm_tier.put(key, value, size):
                self.stats["promotions"] += 1
                if from_tier == "cold":
                    self.cold_tier.remove(key)

    def _demote(self, key: str, value: Any, from_tier: str):
        """Demota un valor a un tier inferior"""
        size = self._estimate_size(value)

        if from_tier == "hot":
            if self.warm_tier.put(key, value, size):
                self.stats["demotions"] += 1
                self.hot_tier.remove(key)
            elif self.cold_tier.put(key, value, size):
                self.stats["demotions"] += 1
                self.hot_tier.remove(key)

        elif from_tier == "warm":
            if self.cold_tier.put(key, value, size):
                self.stats["demotions"] += 1
                self.warm_tier.remove(key)

    def _estimate_size(self, value: Any) -> int:
        """Estima el tamaño de un valor en bytes"""
        if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        elif isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, (bytes, bytearray)):
            return len(value)
        else:
            try:
                return len(pickle.dumps(value))
            except:
                return sys.getsizeof(value)

    def get_stats(self) -> Dict:
        """Retorna estadísticas de uso"""
        total_hits = self.stats["hot_hits"] + self.stats["warm_hits"] + self.stats["cold_hits"]
        total_requests = total_hits + self.stats["misses"]

        return {
            "tiers": {
                "hot": {
                    "usage_percent": self.hot_tier.usage_percent,
                    "items": len(self.hot_tier.items),
                    "size_mb": self.hot_tier.current_size_bytes / 1e6
                },
                "warm": {
                    "usage_percent": self.warm_tier.usage_percent,
                    "items": len(self.warm_tier.items),
                    "size_mb": self.warm_tier.current_size_bytes / 1e6
                },
                "cold": {
                    "usage_percent": self.cold_tier.usage_percent,
                    "items": len(self.cold_tier.items),
                    "size_mb": self.cold_tier.current_size_bytes / 1e6
                }
            },
            "hits": {
                "hot": self.stats["hot_hits"],
                "warm": self.stats["warm_hits"],
                "cold": self.stats["cold_hits"],
                "total": total_hits
            },
            "misses": self.stats["misses"],
            "hit_rate": (total_hits / total_requests * 100) if total_requests > 0 else 0,
            "promotions": self.stats["promotions"],
            "demotions": self.stats["demotions"]
        }

    def clear(self):
        """Limpia toda la memoria"""
        with self.lock:
            self.hot_tier.items.clear()
            self.hot_tier.current_size_bytes = 0
            self.warm_tier.items.clear()
            self.warm_tier.current_size_bytes = 0
            self.cold_tier.clear()


class TensorOffloader:
    """
    Offloader de tensores para entrenamiento de modelos grandes

    Permite mover tensores entre RAM y SSD automáticamente,
    optimizado para entrenamiento con gradientes y estados de optimizador.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.memory = HybridMemoryManager(config)
        self.pinned_tensors: Dict[str, torch.Tensor] = {}

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch es requerido para TensorOffloader")

    def offload_tensor(self, name: str, tensor: torch.Tensor) -> str:
        """
        Mueve un tensor a memoria híbrida

        Args:
            name: Nombre único del tensor
            tensor: Tensor a offload

        Returns:
            Key para recuperar el tensor
        """
        # Mover a CPU si está en GPU
        if tensor.is_cuda:
            tensor = tensor.cpu()

        # Convertir a numpy para serialización eficiente
        data = {
            "array": tensor.detach().numpy(),  # detach() para tensores con gradientes
            "dtype": str(tensor.dtype),
            "shape": tensor.shape,
            "requires_grad": tensor.requires_grad
        }

        key = f"tensor_{name}"
        self.memory.put(key, data, priority="warm")
        return key

    def load_tensor(self, name: str, device: str = "cpu") -> Optional[torch.Tensor]:
        """
        Recupera un tensor de memoria híbrida

        Args:
            name: Nombre del tensor
            device: Dispositivo destino ("cpu", "cuda", "mps")

        Returns:
            Tensor recuperado o None
        """
        key = f"tensor_{name}"
        data = self.memory.get(key)

        if data is None:
            return None

        # Reconstruir tensor
        tensor = torch.from_numpy(data["array"])

        if data["requires_grad"]:
            tensor.requires_grad_(True)

        # Mover a dispositivo
        if device != "cpu":
            tensor = tensor.to(device)

        return tensor

    def offload_optimizer_state(self, optimizer: "torch.optim.Optimizer", name: str):
        """Offload estado del optimizador a disco"""
        state_dict = optimizer.state_dict()
        key = f"optimizer_{name}"
        self.memory.put(key, state_dict, priority="cold")

    def load_optimizer_state(self, optimizer: "torch.optim.Optimizer", name: str) -> bool:
        """Carga estado del optimizador desde disco"""
        key = f"optimizer_{name}"
        state_dict = self.memory.get(key)
        if state_dict:
            optimizer.load_state_dict(state_dict)
            return True
        return False

    def pin_tensor(self, name: str, tensor: torch.Tensor):
        """Ancla un tensor en memoria RAM (no se offload)"""
        if self.config.pin_memory and tensor.is_cpu:
            tensor = tensor.pin_memory()
        self.pinned_tensors[name] = tensor

    def get_pinned_tensor(self, name: str) -> Optional[torch.Tensor]:
        """Obtiene un tensor anclado"""
        return self.pinned_tensors.get(name)


# Singleton para uso global
_hybrid_memory: Optional[HybridMemoryManager] = None


def get_hybrid_memory(config: Optional[MemoryConfig] = None) -> HybridMemoryManager:
    """Obtiene la instancia global de memoria híbrida"""
    global _hybrid_memory
    if _hybrid_memory is None:
        _hybrid_memory = HybridMemoryManager(config)
    return _hybrid_memory


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 THAU Hybrid Memory System - Test")
    print("=" * 60)

    # Configurar para usar menos memoria en test
    config = MemoryConfig(
        max_ram_gb=1.0,
        max_disk_gb=5.0,
        disk_cache_dir=Path(__file__).parent.parent / "data" / "memory_cache"
    )

    memory = HybridMemoryManager(config)

    # Test básico
    print("\n📝 Test 1: Almacenamiento básico")
    memory.put("test_key", {"data": "hello world"})
    result = memory.get("test_key")
    print(f"   Stored and retrieved: {result}")

    # Test con datos grandes
    print("\n📝 Test 2: Datos grandes (simular tensores)")
    large_data = np.random.randn(1000, 1000).astype(np.float32)  # ~4MB
    memory.put("large_tensor", large_data, priority="cold")
    retrieved = memory.get("large_tensor")
    print(f"   Large data shape: {retrieved.shape if retrieved is not None else 'None'}")

    # Test de promoción
    print("\n📝 Test 3: Promoción entre tiers")
    for i in range(5):
        memory.get("large_tensor")  # Acceder múltiples veces

    # Estadísticas
    print("\n📊 Estadísticas:")
    stats = memory.get_stats()
    print(f"   Hot tier: {stats['tiers']['hot']['items']} items, {stats['tiers']['hot']['size_mb']:.2f} MB")
    print(f"   Warm tier: {stats['tiers']['warm']['items']} items, {stats['tiers']['warm']['size_mb']:.2f} MB")
    print(f"   Cold tier: {stats['tiers']['cold']['items']} items, {stats['tiers']['cold']['size_mb']:.2f} MB")
    print(f"   Hit rate: {stats['hit_rate']:.1f}%")
    print(f"   Promotions: {stats['promotions']}")

    # Test TensorOffloader si PyTorch está disponible
    if TORCH_AVAILABLE:
        print("\n📝 Test 4: TensorOffloader")
        offloader = TensorOffloader(config)

        # Crear tensor de prueba
        tensor = torch.randn(500, 500, requires_grad=True)
        print(f"   Original tensor shape: {tensor.shape}")

        # Offload
        offloader.offload_tensor("test_tensor", tensor)
        print("   Tensor offloaded to disk")

        # Recover
        recovered = offloader.load_tensor("test_tensor")
        print(f"   Recovered tensor shape: {recovered.shape}")
        print(f"   Tensors equal: {torch.allclose(tensor, recovered)}")

    print("\n✅ Tests completados!")
