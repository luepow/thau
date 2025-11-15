"""Tests for core models."""

import pytest
import torch

from config.model_configs import TINY_CONFIG
from core.models.base_transformer import TinyLLM
from core.models.attention import MultiHeadAttention
from core.models.layers import TransformerBlock


def test_tiny_llm_initialization():
    """Test TinyLLM model initialization."""
    model = TinyLLM(TINY_CONFIG)

    assert model is not None
    assert model.get_num_params() > 0


def test_tiny_llm_forward():
    """Test forward pass."""
    model = TinyLLM(TINY_CONFIG)
    batch_size, seq_len = 2, 16

    input_ids = torch.randint(0, TINY_CONFIG.vocab_size, (batch_size, seq_len))

    outputs = model(input_ids)

    assert "logits" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, TINY_CONFIG.vocab_size)


def test_tiny_llm_generation():
    """Test text generation."""
    model = TinyLLM(TINY_CONFIG)
    prompt = torch.randint(0, TINY_CONFIG.vocab_size, (1, 10))

    generated = model.generate(prompt, max_new_tokens=20)

    assert generated.shape[0] == 1
    assert generated.shape[1] > prompt.shape[1]


def test_multi_head_attention():
    """Test multi-head attention."""
    d_model, n_heads = 512, 8
    batch_size, seq_len = 2, 10

    attention = MultiHeadAttention(d_model, n_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    output, _ = attention(x, x, x)

    assert output.shape == x.shape


def test_transformer_block():
    """Test transformer block."""
    d_model, n_heads, d_ff = 512, 8, 2048
    batch_size, seq_len = 2, 10

    block = TransformerBlock(d_model, n_heads, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)

    output = block(x)

    assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
