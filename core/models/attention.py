"""Multi-head attention mechanism implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention mechanism.

    Implements the attention mechanism from "Attention Is All You Need" paper
    with support for causal masking and optional flash attention.

    Args:
        d_model: Dimension of the model
        n_heads: Number of attention heads
        dropout: Dropout probability
        use_flash_attention: Whether to use flash attention (if available)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_flash_attention: bool = False,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.use_flash_attention = use_flash_attention

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.o_proj = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor for causal attention
            return_attention_weights: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
            - output: shape (batch_size, seq_len, d_model)
            - attention_weights: shape (batch_size, n_heads, seq_len, seq_len) or None
        """
        batch_size, seq_len, _ = query.shape

        # Linear projections and reshape to (batch_size, n_heads, seq_len, d_k)
        Q = self.q_proj(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized flash attention (PyTorch 2.0+)
            # Automatically handles masking and is more memory efficient
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=(mask is None),  # Use causal masking if no mask provided
            )
            attention_weights = None  # Flash attention doesn't return weights

        else:
            # Standard scaled dot-product attention
            # scores: (batch_size, n_heads, seq_len, seq_len)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

            # Apply mask if provided
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            # Softmax
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

            # Apply attention to values
            attn_output = torch.matmul(attention_weights, V)

        # Reshape back: (batch_size, seq_len, n_heads, d_k) -> (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final linear projection
        output = self.o_proj(attn_output)

        if return_attention_weights and not self.use_flash_attention:
            return output, attention_weights
        else:
            return output, None


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) from "RoFormer: Enhanced Transformer with Rotary Position Embedding".

    RoPE encodes positional information by rotating the query and key representations
    in a way that naturally incorporates relative positions.

    Args:
        dim: Dimension of the embeddings (typically d_k)
        max_seq_len: Maximum sequence length
        base: Base for the exponential decay (default: 10000)
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute positional encodings
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the dimensions of x."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch_size, n_heads, seq_len, d_k)
            k: Key tensor of shape (batch_size, n_heads, seq_len, d_k)

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        seq_len = q.shape[2]

        # Get cached cos and sin for the sequence length
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        # Apply rotation
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        return q_embed, k_embed


class MultiHeadAttentionWithRoPE(MultiHeadAttention):
    """Multi-head attention with Rotary Position Embeddings.

    Extends the standard multi-head attention to include RoPE for better
    positional encoding, especially for longer sequences.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_flash_attention: bool = False,
        max_seq_len: int = 2048,
    ):
        super().__init__(d_model, n_heads, dropout, use_flash_attention)

        # Add rotary embeddings
        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with RoPE applied to queries and keys."""
        batch_size, seq_len, _ = query.shape

        # Linear projections and reshape
        Q = self.q_proj(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Apply rotary embeddings
        Q, K = self.rope(Q, K)

        # Compute attention (same as parent class)
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=(mask is None),
            )
            attention_weights = None

        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

            attn_output = torch.matmul(attention_weights, V)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.o_proj(attn_output)

        if return_attention_weights and not self.use_flash_attention:
            return output, attention_weights
        else:
            return output, None


if __name__ == "__main__":
    # Test the attention mechanisms
    batch_size, seq_len, d_model, n_heads = 2, 10, 512, 8

    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)

    # Test standard multi-head attention
    print("Testing MultiHeadAttention:")
    mha = MultiHeadAttention(d_model, n_heads)
    output, attn_weights = mha(x, x, x, return_attention_weights=True)
    print(f"Output shape: {output.shape}")
    if attn_weights is not None:
        print(f"Attention weights shape: {attn_weights.shape}")

    # Test multi-head attention with RoPE
    print("\nTesting MultiHeadAttentionWithRoPE:")
    mha_rope = MultiHeadAttentionWithRoPE(d_model, n_heads)
    output_rope, _ = mha_rope(x, x, x)
    print(f"Output shape: {output_rope.shape}")

    print("\nAll tests passed!")
