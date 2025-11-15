"""Model adapter for loading and managing pretrained models."""

import torch
from typing import Optional, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from loguru import logger
from pathlib import Path

from adapters.device_manager import DeviceManager
from config.model_configs import LoRAConfig, QuantizationConfig


class ModelAdapter:
    """Adapter for loading and managing pretrained language models.

    Handles:
    - Loading models from HuggingFace Hub or local paths
    - Quantization (8-bit, 4-bit)
    - LoRA fine-tuning setup
    - Device management and optimization
    """

    def __init__(
        self,
        model_name: str,
        device_manager: Optional[DeviceManager] = None,
        use_quantization: bool = True,
        quantization_config: Optional[QuantizationConfig] = None,
        use_flash_attention: bool = False,
    ):
        """Initialize the model adapter.

        Args:
            model_name: Name or path of the pretrained model
            device_manager: DeviceManager instance (creates new if None)
            use_quantization: Whether to use quantization
            quantization_config: Configuration for quantization
            use_flash_attention: Whether to use flash attention (if available)
        """
        self.model_name = model_name
        self.device_manager = device_manager or DeviceManager()
        self.use_quantization = use_quantization
        self.quantization_config = quantization_config
        self.use_flash_attention = use_flash_attention

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.is_peft_model = False

        logger.info(f"Initializing ModelAdapter for: {model_name}")

    def load_model(
        self,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ) -> AutoModelForCausalLM:
        """Load the pretrained model with appropriate configurations.

        Args:
            torch_dtype: PyTorch dtype for the model (auto-detected if None)
            trust_remote_code: Whether to trust remote code in model config

        Returns:
            The loaded model
        """
        logger.info(f"Loading model: {self.model_name}")

        # Determine dtype based on device
        if torch_dtype is None:
            if self.device_manager.device_type in ["cuda", "mps"]:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

        # Prepare quantization config if needed
        quantization_config = None
        if self.use_quantization and self.device_manager.device_type == "cuda":
            if self.quantization_config is None:
                # Default to 8-bit quantization
                from config.model_configs import QUANT_8BIT
                self.quantization_config = QUANT_8BIT

            bnb_config_dict = self.quantization_config.to_bnb_config()
            quantization_config = BitsAndBytesConfig(**bnb_config_dict)
            logger.info(f"Using quantization: {bnb_config_dict}")

        # Load model
        try:
            model_kwargs = {
                "pretrained_model_name_or_path": self.model_name,
                "torch_dtype": torch_dtype,
                "trust_remote_code": trust_remote_code,
                "low_cpu_mem_usage": True,
            }

            # Add quantization config if using CUDA
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            else:
                # For MPS or CPU, load normally then move
                pass

            # Flash attention (only for compatible models and devices)
            if self.use_flash_attention and self.device_manager.device_type == "cuda":
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2")

            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

            # Move to device if not using device_map
            if quantization_config is None:
                self.model = self.device_manager.to_device(self.model)

            logger.info(f"Model loaded successfully on {self.device_manager.device}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

        return self.model

    def load_tokenizer(
        self,
        padding_side: str = "right",
        add_eos_token: bool = True,
    ) -> AutoTokenizer:
        """Load the tokenizer for the model.

        Args:
            padding_side: Side to pad sequences ("left" or "right")
            add_eos_token: Whether to add EOS token to sequences

        Returns:
            The loaded tokenizer
        """
        logger.info(f"Loading tokenizer for: {self.model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # Configure tokenizer
            self.tokenizer.padding_side = padding_side

            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")

            if add_eos_token:
                self.tokenizer.add_eos_token = True

            logger.info("Tokenizer loaded successfully")

        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise

        return self.tokenizer

    def prepare_for_lora(
        self,
        lora_config: Optional[LoRAConfig] = None,
    ) -> PeftModel:
        """Prepare the model for LoRA fine-tuning.

        Args:
            lora_config: Configuration for LoRA (uses default if None)

        Returns:
            The model with LoRA adapters
        """
        if self.model is None:
            raise ValueError("Model must be loaded before preparing for LoRA")

        logger.info("Preparing model for LoRA fine-tuning")

        # Use default LoRA config if not provided
        if lora_config is None:
            from config.model_configs import LORA_DEFAULT
            lora_config = LORA_DEFAULT

        try:
            # Prepare model for k-bit training if quantized
            if self.use_quantization:
                self.model = prepare_model_for_kbit_training(self.model)
                logger.info("Model prepared for k-bit training")

            # Create LoRA config
            peft_config = LoraConfig(**lora_config.to_peft_config())

            # Apply LoRA
            self.model = get_peft_model(self.model, peft_config)
            self.is_peft_model = True

            # Print trainable parameters
            self.model.print_trainable_parameters()

            logger.info("LoRA adapters applied successfully")

        except Exception as e:
            logger.error(f"Error preparing model for LoRA: {e}")
            raise

        return self.model

    def save_model(
        self,
        output_dir: str,
        save_full_model: bool = False,
    ) -> None:
        """Save the model (or LoRA adapters) to disk.

        Args:
            output_dir: Directory to save the model
            save_full_model: Whether to save full model or just adapters (for LoRA)
        """
        if self.model is None:
            raise ValueError("No model to save")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model to: {output_path}")

        try:
            if self.is_peft_model and not save_full_model:
                # Save only LoRA adapters
                self.model.save_pretrained(output_path)
                logger.info("LoRA adapters saved")
            else:
                # Save full model
                self.model.save_pretrained(output_path)
                logger.info("Full model saved")

            # Save tokenizer
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_path)
                logger.info("Tokenizer saved")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_lora_adapters(
        self,
        adapter_path: str,
    ) -> PeftModel:
        """Load LoRA adapters into the base model.

        Args:
            adapter_path: Path to the saved LoRA adapters

        Returns:
            The model with loaded LoRA adapters
        """
        if self.model is None:
            raise ValueError("Base model must be loaded first")

        logger.info(f"Loading LoRA adapters from: {adapter_path}")

        try:
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
            )
            self.is_peft_model = True
            logger.info("LoRA adapters loaded successfully")

        except Exception as e:
            logger.error(f"Error loading LoRA adapters: {e}")
            raise

        return self.model

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {"status": "No model loaded"}

        info = {
            "model_name": self.model_name,
            "device": str(self.device_manager.device),
            "is_peft_model": self.is_peft_model,
            "dtype": str(self.model.dtype),
        }

        # Get parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        info.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params * 100) if total_params > 0 else 0,
        })

        return info

    def __repr__(self) -> str:
        """String representation of the model adapter."""
        status = "loaded" if self.model is not None else "not loaded"
        return f"ModelAdapter(model={self.model_name}, status={status}, device={self.device_manager.device})"


if __name__ == "__main__":
    # Test the model adapter
    from config.base_config import get_config

    config = get_config()
    adapter = ModelAdapter(
        model_name=config.MODEL_NAME,
        use_quantization=config.USE_QUANTIZATION,
    )

    print("\nLoading model and tokenizer...")
    adapter.load_model()
    adapter.load_tokenizer()

    print("\nModel Info:")
    import json
    print(json.dumps(adapter.get_model_info(), indent=2))

    print("\nDevice Info:")
    print(json.dumps(adapter.device_manager.get_device_info(), indent=2))
