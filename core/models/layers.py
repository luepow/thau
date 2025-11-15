"""Neural network layers for transformer models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network.

    Implements the FFN layer from the transformer architecture:
    FFN(x) = max(0, xW1 + b1)W2 + b2

    Args:
        d_model: Dimension of the model
        d_ff: Dimension of the feed-forward layer (typically 4 * d_model)
        dropout: Dropout probability
        activation: Activation function ('gelu', 'relu', 'silu')
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # Select activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward layers.

    Implements:
    1. Multi-head self-attention with residual connection and layer norm
    2. Feed-forward network with residual connection and layer norm

    Args:
        d_model: Dimension of the model
        n_heads: Number of attention heads
        d_ff: Dimension of feed-forward layer
        dropout: Dropout probability
        attention_dropout: Dropout for attention weights
        activation: Activation function for FFN
        layer_norm_eps: Epsilon for layer normalization
        use_flash_attention: Whether to use flash attention
        use_rope: Whether to use rotary position embeddings
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        use_flash_attention: bool = False,
        use_rope: bool = True,
    ):
        super().__init__()

        # Import here to avoid circular dependency
        from core.models.attention import MultiHeadAttention, MultiHeadAttentionWithRoPE

        # Multi-head attention
        if use_rope:
            self.attention = MultiHeadAttentionWithRoPE(
                d_model=d_model,
                n_heads=n_heads,
                dropout=attention_dropout,
                use_flash_attention=use_flash_attention,
            )
        else:
            self.attention = MultiHeadAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=attention_dropout,
                use_flash_attention=use_flash_attention,
            )

        # Feed-forward network
        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Pre-LayerNorm architecture (more stable for deep networks)

        # Self-attention with residual connection
        normalized_x = self.norm1(x)
        attn_output, _ = self.attention(normalized_x, normalized_x, normalized_x, mask)
        x = x + self.dropout(attn_output)

        # Feed-forward with residual connection
        normalized_x = self.norm2(x)
        ff_output = self.feed_forward(normalized_x)
        x = x + self.dropout(ff_output)

        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from "Attention Is All You Need".

    Adds position information to token embeddings using sine and cosine functions.

    Args:
        d_model: Dimension of the model
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    """Token embedding layer with scaling.

    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of embeddings
        padding_idx: Index for padding token (default: 0)
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with scaling.

        Args:
            x: Input token indices of shape (batch_size, seq_len)

        Returns:
            Embeddings of shape (batch_size, seq_len, d_model)
        """
        # Scale embeddings by sqrt(d_model) as in the original paper
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))


class OutputHead(nn.Module):
    """Language model output head for next-token prediction.

    Projects hidden states to vocabulary logits.

    Args:
        d_model: Dimension of the model
        vocab_size: Size of the vocabulary
        bias: Whether to use bias in the linear layer
    """

    def __init__(self, d_model: int, vocab_size: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits.

        Args:
            x: Hidden states of shape (batch_size, seq_len, d_model)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        return self.linear(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient alternative to LayerNorm, used in models like LLaMA.

    Args:
        dim: Dimension of the input
        eps: Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x_norm = x / rms
        return self.weight * x_norm


class SwiGLU(nn.Module):
    """SwiGLU activation function (used in PaLM and LLaMA).

    Combines Swish/SiLU activation with Gated Linear Units:
    SwiGLU(x) = Swish(xW) ⊗ xV

    Args:
        d_model: Input dimension
        d_ff: Hidden dimension (typically 2.67 * d_model for SwiGLU)
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w = nn.Linear(d_model, d_ff, bias=False)
        self.v = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.w2(F.silu(self.w(x)) * self.v(x))


if __name__ == "__main__":
    # Test the layers
    batch_size, seq_len, d_model = 2, 10, 512
    n_heads, d_ff = 8, 2048
    vocab_size = 32000

    print("Testing TransformerBlock:")
    block = TransformerBlock(d_model, n_heads, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    output = block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    print("\nTesting TokenEmbedding:")
    token_emb = TokenEmbedding(vocab_size, d_model)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    embeddings = token_emb(tokens)
    print(f"Token shape: {tokens.shape}")
    print(f"Embedding shape: {embeddings.shape}")

    print("\nTesting PositionalEncoding:")
    pos_enc = PositionalEncoding(d_model)
    pos_embeddings = pos_enc(embeddings)
    print(f"Positional embedding shape: {pos_embeddings.shape}")

    print("\nTesting OutputHead:")
    output_head = OutputHead(d_model, vocab_size)
    logits = output_head(output)
    print(f"Logits shape: {logits.shape}")

    print("\nAll tests passed!")
