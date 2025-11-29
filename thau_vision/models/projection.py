"""
THAU-Vision: Projection Layer
==============================

MLP projection layer to map vision embeddings to LLM embedding space.
Based on LLaVA and TinyLLaVA architectures.

The projection converts:
- Vision encoder output: [B, num_patches, vision_dim]
- To LLM input space: [B, num_visual_tokens, llm_dim]
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ProjectionLayer(nn.Module):
    """
    Multi-layer perceptron to project vision features to LLM space.

    Supports multiple projection types:
    - linear: Single linear layer
    - mlp: Two-layer MLP with GELU
    - mlp_deep: Three-layer MLP with GELU
    - resampler: Q-Former style attention resampler (reduces tokens)
    """

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        projection_type: str = "mlp",
        num_visual_tokens: Optional[int] = None,  # For resampler
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize projection layer.

        Args:
            vision_dim: Input dimension from vision encoder
            llm_dim: Output dimension for LLM
            projection_type: Type of projection ("linear", "mlp", "mlp_deep", "resampler")
            num_visual_tokens: Target number of visual tokens (for resampler)
            hidden_dim: Hidden dimension for MLP (default: avg of vision_dim and llm_dim)
            dropout: Dropout rate
        """
        super().__init__()

        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        self.projection_type = projection_type
        self.num_visual_tokens = num_visual_tokens

        # Default hidden dim
        if hidden_dim is None:
            hidden_dim = (vision_dim + llm_dim) // 2

        self.hidden_dim = hidden_dim

        # Build projection based on type
        if projection_type == "linear":
            self.projection = nn.Linear(vision_dim, llm_dim)

        elif projection_type == "mlp":
            self.projection = nn.Sequential(
                nn.Linear(vision_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, llm_dim),
            )

        elif projection_type == "mlp_deep":
            self.projection = nn.Sequential(
                nn.Linear(vision_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, llm_dim),
            )

        elif projection_type == "resampler":
            if num_visual_tokens is None:
                raise ValueError("num_visual_tokens required for resampler")
            self.projection = Resampler(
                vision_dim=vision_dim,
                output_dim=llm_dim,
                num_queries=num_visual_tokens,
                num_heads=8,
                hidden_dim=hidden_dim,
            )

        else:
            raise ValueError(f"Unknown projection_type: {projection_type}")

        # Layer normalization
        self.ln_pre = nn.LayerNorm(vision_dim)
        self.ln_post = nn.LayerNorm(llm_dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project vision features to LLM space.

        Args:
            vision_features: [B, num_patches, vision_dim]
            attention_mask: Optional mask for resampler

        Returns:
            Projected features: [B, num_tokens, llm_dim]
        """
        # Pre-normalize
        x = self.ln_pre(vision_features)

        # Project
        if self.projection_type == "resampler":
            x = self.projection(x, attention_mask)
        else:
            x = self.projection(x)

        # Post-normalize
        x = self.ln_post(x)

        return x

    def get_output_tokens(self, num_patches: int) -> int:
        """Get number of output visual tokens."""
        if self.num_visual_tokens is not None:
            return self.num_visual_tokens
        return num_patches


class Resampler(nn.Module):
    """
    Perceiver-style resampler to reduce number of visual tokens.

    Uses learnable queries to attend to vision features and
    produce a fixed number of output tokens.
    """

    def __init__(
        self,
        vision_dim: int,
        output_dim: int,
        num_queries: int = 64,
        num_heads: int = 8,
        hidden_dim: int = 1024,
        num_layers: int = 2,
    ):
        """
        Initialize resampler.

        Args:
            vision_dim: Input dimension
            output_dim: Output dimension
            num_queries: Number of learnable queries (output tokens)
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension
            num_layers: Number of cross-attention layers
        """
        super().__init__()

        self.num_queries = num_queries

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))

        # Input projection
        self.input_proj = nn.Linear(vision_dim, hidden_dim)

        # Cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Initialize queries
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(
        self,
        vision_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Resample vision features to fixed number of tokens.

        Args:
            vision_features: [B, num_patches, vision_dim]
            attention_mask: Optional attention mask

        Returns:
            Resampled features: [B, num_queries, output_dim]
        """
        batch_size = vision_features.shape[0]

        # Project input
        kv = self.input_proj(vision_features)

        # Expand queries for batch
        queries = self.queries.expand(batch_size, -1, -1)

        # Cross-attention layers
        for layer in self.layers:
            queries = layer(queries, kv, attention_mask)

        # Output projection
        output = self.output_proj(queries)

        return output


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for resampler."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention from queries to key-value features.

        Args:
            queries: [B, num_queries, dim]
            kv: [B, num_patches, dim]
            attention_mask: Optional mask

        Returns:
            Updated queries: [B, num_queries, dim]
        """
        # Cross-attention
        residual = queries
        queries = self.norm1(queries)
        queries, _ = self.attention(
            query=queries,
            key=kv,
            value=kv,
            key_padding_mask=attention_mask,
        )
        queries = residual + queries

        # FFN
        residual = queries
        queries = self.norm2(queries)
        queries = self.ffn(queries)
        queries = residual + queries

        return queries


# Factory function
def create_projection(
    vision_dim: int,
    llm_dim: int,
    projection_type: str = "mlp",
    **kwargs,
) -> ProjectionLayer:
    """Create a projection layer."""
    return ProjectionLayer(
        vision_dim=vision_dim,
        llm_dim=llm_dim,
        projection_type=projection_type,
        **kwargs,
    )


# Test
if __name__ == "__main__":
    print("Testing Projection Layers...")

    # Test MLP projection
    proj_mlp = ProjectionLayer(
        vision_dim=768,
        llm_dim=2048,
        projection_type="mlp",
    )

    # Test input
    x = torch.randn(2, 196, 768)  # [batch, patches, vision_dim]
    out = proj_mlp(x)
    print(f"MLP: {x.shape} -> {out.shape}")

    # Test resampler
    proj_resample = ProjectionLayer(
        vision_dim=768,
        llm_dim=2048,
        projection_type="resampler",
        num_visual_tokens=64,
    )

    out_resample = proj_resample(x)
    print(f"Resampler: {x.shape} -> {out_resample.shape}")

    print("\nProjection layer tests complete!")
