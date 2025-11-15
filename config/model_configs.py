"""Model-specific configurations."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    """Configuration for the base transformer model."""

    # Model architecture
    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048  # Feed-forward dimension
    max_seq_length: int = 2048

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # Activation
    activation: str = "gelu"  # Options: gelu, relu, silu

    # Advanced features
    use_flash_attention: bool = False
    use_rotary_embeddings: bool = True
    use_alibi: bool = False  # Alternative to positional embeddings

    # Initialization
    initializer_range: float = 0.02

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert 0.0 <= self.dropout <= 1.0, "dropout must be between 0 and 1"


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning."""

    # LoRA parameters
    r: int = 8  # Rank of the low-rank matrices
    lora_alpha: int = 16  # Scaling factor
    lora_dropout: float = 0.05

    # Target modules for LoRA
    target_modules: tuple = ("q_proj", "v_proj", "k_proj", "o_proj")

    # Bias handling
    bias: str = "none"  # Options: none, all, lora_only

    # Task type
    task_type: str = "CAUSAL_LM"

    def to_peft_config(self) -> dict:
        """Convert to PEFT library configuration format.

        Returns:
            Dictionary compatible with PEFT library.
        """
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": list(self.target_modules),
            "bias": self.bias,
            "task_type": self.task_type,
        }


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    # Quantization settings
    load_in_8bit: bool = True
    load_in_4bit: bool = False

    # 4-bit specific settings
    bnb_4bit_compute_dtype: str = "float16"  # Computation dtype for 4-bit
    bnb_4bit_quant_type: str = "nf4"  # Options: fp4, nf4
    bnb_4bit_use_double_quant: bool = True  # Nested quantization

    # 8-bit specific settings
    llm_int8_threshold: float = 6.0
    llm_int8_has_fp16_weight: bool = False

    def __post_init__(self):
        """Validate configuration."""
        assert not (self.load_in_8bit and self.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization"

    def to_bnb_config(self) -> dict:
        """Convert to BitsAndBytes configuration format.

        Returns:
            Dictionary compatible with BitsAndBytes library.
        """
        if self.load_in_4bit:
            return {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
                "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
                "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            }
        elif self.load_in_8bit:
            return {
                "load_in_8bit": True,
                "llm_int8_threshold": self.llm_int8_threshold,
                "llm_int8_has_fp16_weight": self.llm_int8_has_fp16_weight,
            }
        return {}


# Predefined model configurations
TINY_CONFIG = TransformerConfig(
    vocab_size=32000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    max_seq_length=2048,
)

SMALL_CONFIG = TransformerConfig(
    vocab_size=32000,
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072,
    max_seq_length=2048,
)

MEDIUM_CONFIG = TransformerConfig(
    vocab_size=32000,
    d_model=1024,
    n_heads=16,
    n_layers=24,
    d_ff=4096,
    max_seq_length=4096,
)

# THAU-2B Configuration - Custom 2B parameter model for self-learning
THAU_2B_CONFIG = TransformerConfig(
    vocab_size=32000,
    d_model=2560,          # Hidden dimension
    n_heads=32,            # Attention heads (d_model / n_heads = 80)
    n_layers=24,           # Transformer layers
    d_ff=10240,            # Feed-forward dimension (4x d_model)
    max_seq_length=4096,   # Context window
    dropout=0.1,
    attention_dropout=0.1,
    activation="gelu",
    use_flash_attention=False,  # Set True if GPU supports it
    use_rotary_embeddings=True,  # RoPE for better positional encoding
    use_alibi=False,
    initializer_range=0.02,
)

# Predefined LoRA configurations
LORA_LIGHT = LoRAConfig(r=4, lora_alpha=8, lora_dropout=0.05)
LORA_DEFAULT = LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.05)
LORA_HEAVY = LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.1)

# Predefined quantization configurations
QUANT_8BIT = QuantizationConfig(load_in_8bit=True, load_in_4bit=False)
QUANT_4BIT = QuantizationConfig(
    load_in_8bit=False,
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
