"""Device management for cross-platform compatibility (MPS, CUDA, CPU)."""

import torch
from typing import Union, Optional
from loguru import logger
import platform


class DeviceManager:
    """Manages device selection and optimization for different hardware.

    Automatically detects and selects the best available device:
    - MPS (Metal Performance Shaders) for Apple Silicon
    - CUDA for NVIDIA GPUs
    - CPU as fallback

    Also provides utilities for memory management and batch size optimization.
    """

    def __init__(self, device: Optional[str] = None):
        """Initialize the device manager.

        Args:
            device: Specific device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
        """
        self.device = self._detect_device(device)
        self.device_type = self._get_device_type()
        logger.info(f"Using device: {self.device} (type: {self.device_type})")

    def _detect_device(self, device: Optional[str] = None) -> torch.device:
        """Detect the best available device or use the specified one.

        Args:
            device: Specific device string or None for auto-detection

        Returns:
            torch.device: The selected device
        """
        if device and device != "auto":
            logger.info(f"Using manually specified device: {device}")
            return torch.device(device)

        # Check for Apple Silicon (MPS)
        if torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) device detected")
            return torch.device("mps")

        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA device detected: {gpu_name}")
            return torch.device("cuda")

        # Fallback to CPU
        logger.warning("No GPU detected, using CPU (this will be slower)")
        return torch.device("cpu")

    def _get_device_type(self) -> str:
        """Get the type of device as a string.

        Returns:
            str: Device type ('mps', 'cuda', or 'cpu')
        """
        device_str = str(self.device)
        if "mps" in device_str:
            return "mps"
        elif "cuda" in device_str:
            return "cuda"
        else:
            return "cpu"

    def to_device(self, tensor: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
        """Move a tensor or module to the selected device.

        Args:
            tensor: PyTorch tensor or module to move

        Returns:
            The tensor or module on the selected device
        """
        return tensor.to(self.device)

    def get_optimal_batch_size(self, default_batch_size: int = 4) -> int:
        """Get an optimal batch size based on the device and available memory.

        Args:
            default_batch_size: Default batch size to start with

        Returns:
            int: Recommended batch size for the current device
        """
        if self.device_type == "cpu":
            # CPU typically handles smaller batches better
            return max(1, default_batch_size // 4)

        elif self.device_type == "mps":
            # Apple Silicon - moderate batch sizes work well
            # MPS has unified memory architecture
            try:
                # Try to estimate available memory
                import psutil
                available_gb = psutil.virtual_memory().available / (1024 ** 3)

                if available_gb > 32:
                    return default_batch_size * 2
                elif available_gb > 16:
                    return default_batch_size
                else:
                    return max(1, default_batch_size // 2)
            except ImportError:
                logger.warning("psutil not installed, using default batch size")
                return default_batch_size

        elif self.device_type == "cuda":
            # NVIDIA GPU - check VRAM
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                total_memory_gb = total_memory / (1024 ** 3)

                if total_memory_gb > 24:
                    return default_batch_size * 4
                elif total_memory_gb > 16:
                    return default_batch_size * 2
                elif total_memory_gb > 8:
                    return default_batch_size
                else:
                    return max(1, default_batch_size // 2)
            except Exception as e:
                logger.warning(f"Could not determine CUDA memory: {e}")
                return default_batch_size

        return default_batch_size

    def get_device_info(self) -> dict:
        """Get detailed information about the current device.

        Returns:
            dict: Device information including type, name, and memory stats
        """
        info = {
            "device": str(self.device),
            "device_type": self.device_type,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
        }

        if self.device_type == "cuda":
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
                "allocated_memory_gb": torch.cuda.memory_allocated(0) / (1024 ** 3),
                "cached_memory_gb": torch.cuda.memory_reserved(0) / (1024 ** 3),
            })

        elif self.device_type == "mps":
            info.update({
                "mps_available": torch.backends.mps.is_available(),
                "mps_built": torch.backends.mps.is_built(),
            })

            # Try to get system memory info
            try:
                import psutil
                vm = psutil.virtual_memory()
                info.update({
                    "total_memory_gb": vm.total / (1024 ** 3),
                    "available_memory_gb": vm.available / (1024 ** 3),
                    "memory_percent": vm.percent,
                })
            except ImportError:
                pass

        elif self.device_type == "cpu":
            try:
                import psutil
                info.update({
                    "cpu_count": psutil.cpu_count(logical=False),
                    "cpu_count_logical": psutil.cpu_count(logical=True),
                    "memory_gb": psutil.virtual_memory().total / (1024 ** 3),
                })
            except ImportError:
                info["cpu_count"] = "unknown (install psutil for details)"

        return info

    def clear_cache(self):
        """Clear device memory cache if applicable."""
        if self.device_type == "cuda":
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
        elif self.device_type == "mps":
            torch.mps.empty_cache()
            logger.info("MPS cache cleared")

    def synchronize(self):
        """Synchronize device operations (useful for timing and debugging)."""
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        elif self.device_type == "mps":
            torch.mps.synchronize()

    def set_seed(self, seed: int = 42):
        """Set random seed for reproducibility across all devices.

        Args:
            seed: Random seed value
        """
        torch.manual_seed(seed)
        if self.device_type == "cuda":
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to {seed}")

    def get_memory_stats(self) -> dict:
        """Get current memory usage statistics.

        Returns:
            dict: Memory statistics for the current device
        """
        stats = {"device": str(self.device)}

        if self.device_type == "cuda":
            stats.update({
                "allocated_mb": torch.cuda.memory_allocated(0) / (1024 ** 2),
                "reserved_mb": torch.cuda.memory_reserved(0) / (1024 ** 2),
                "max_allocated_mb": torch.cuda.max_memory_allocated(0) / (1024 ** 2),
            })
        elif self.device_type == "mps":
            stats["note"] = "MPS does not provide memory statistics through PyTorch"
            try:
                import psutil
                vm = psutil.virtual_memory()
                stats["available_memory_gb"] = vm.available / (1024 ** 3)
            except ImportError:
                pass

        return stats

    def optimize_for_inference(self):
        """Apply optimizations for inference mode."""
        if self.device_type == "cuda":
            # Enable cuDNN auto-tuner
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cuDNN auto-tuner for inference")

        # Set default dtype for faster inference if supported
        if self.device_type in ["cuda", "mps"]:
            torch.set_float32_matmul_precision('high')
            logger.info("Set high precision for float32 matmul")

    def __repr__(self) -> str:
        """String representation of the device manager."""
        return f"DeviceManager(device={self.device}, type={self.device_type})"


# Global device manager instance
_device_manager: Optional[DeviceManager] = None


def get_device_manager(device: Optional[str] = None) -> DeviceManager:
    """Get or create the global device manager instance.

    Args:
        device: Specific device to use or None for auto-detection

    Returns:
        DeviceManager: The global device manager instance
    """
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager(device=device)
    return _device_manager


if __name__ == "__main__":
    # Test the device manager
    dm = DeviceManager()
    print("\nDevice Info:")
    import json
    print(json.dumps(dm.get_device_info(), indent=2))

    print("\nMemory Stats:")
    print(json.dumps(dm.get_memory_stats(), indent=2))

    print(f"\nOptimal batch size (default=4): {dm.get_optimal_batch_size(4)}")

    # Test tensor movement
    test_tensor = torch.randn(2, 3)
    test_tensor_device = dm.to_device(test_tensor)
    print(f"\nTensor moved to: {test_tensor_device.device}")
