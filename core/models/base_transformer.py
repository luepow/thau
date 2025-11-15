"""Base transformer model implementation."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from loguru import logger

from core.models.layers import (
    TokenEmbedding,
    PositionalEncoding,
    TransformerBlock,
    OutputHead,
)
from config.model_configs import TransformerConfig


class TinyLLM(nn.Module):
    """Small transformer-based language model.

    A configurable transformer architecture suitable for learning and experimentation.
    Can be trained from scratch or used as a base for fine-tuning.

    Args:
        config: TransformerConfig object with model hyperparameters
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        # Token embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            padding_idx=0,
        )

        # Positional encoding (if not using RoPE)
        if not config.use_rotary_embeddings:
            self.pos_encoding = PositionalEncoding(
                d_model=config.d_model,
                max_seq_len=config.max_seq_length,
                dropout=config.dropout,
            )
        else:
            self.pos_encoding = None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                activation=config.activation,
                layer_norm_eps=config.layer_norm_eps,
                use_flash_attention=config.use_flash_attention,
                use_rope=config.use_rotary_embeddings,
            )
            for _ in range(config.n_layers)
        ])

        # Final layer normalization
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Output head
        self.output_head = OutputHead(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            bias=False,
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Initialized TinyLLM with {n_params:,} parameters")

    def _init_weights(self, module):
        """Initialize weights according to the standard transformer initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            labels: Optional labels for language modeling loss
            return_dict: Whether to return a dictionary

        Returns:
            Dictionary containing:
            - logits: Predicted logits of shape (batch_size, seq_len, vocab_size)
            - loss: Optional language modeling loss if labels provided
            - hidden_states: Final hidden states
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Add positional encoding (if not using RoPE)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # Create causal mask (lower triangular)
        causal_mask = self._create_causal_mask(seq_len, x.device)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention mask: (batch_size, seq_len) -> (batch_size, 1, seq_len, seq_len)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask & attention_mask.bool()

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask=causal_mask)

        # Final layer normalization
        hidden_states = self.final_norm(x)

        # Output projection to vocabulary
        logits = self.output_head(hidden_states)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for cross-entropy
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        if return_dict:
            return {
                "logits": logits,
                "loss": loss,
                "hidden_states": hidden_states,
            }
        else:
            return logits

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a causal (lower triangular) attention mask.

        Args:
            seq_len: Sequence length
            device: Device to create the mask on

        Returns:
            Causal mask of shape (1, 1, seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        return mask.bool()

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Generate text autoregressively.

        Args:
            input_ids: Input token indices of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Keep tokens with cumulative probability >= top_p (nucleus sampling)
            do_sample: Whether to sample (True) or use greedy decoding (False)

        Returns:
            Generated token indices of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions for the current sequence
                outputs = self(input_ids, return_dict=True)
                logits = outputs["logits"]

                # Get logits for the last token
                next_token_logits = logits[:, -1, :]

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[..., 0] = False

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample or take argmax
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Stop if we've reached max length
                if input_ids.shape[1] >= self.config.max_seq_length:
                    break

        return input_ids

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get the number of parameters in the model.

        Args:
            non_embedding: If True, subtract embedding parameters

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            n_params -= self.token_embedding.embedding.weight.numel()

        return n_params

    def estimate_flops(self, batch_size: int, seq_len: int) -> int:
        """Estimate FLOPs for a forward pass (rough approximation).

        Args:
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Estimated FLOPs
        """
        # This is a rough approximation
        # Main operations: attention (QK^T, softmax, attention*V) and FFN

        n = seq_len
        d = self.config.d_model
        h = self.config.n_heads
        l = self.config.n_layers
        v = self.config.vocab_size

        # Attention: 2 * batch * heads * seq^2 * d_k per layer
        attn_flops = 2 * batch_size * h * n * n * (d // h) * l

        # FFN: 2 * batch * seq * d_model * d_ff * 2 (two linear layers) per layer
        ffn_flops = 2 * batch_size * n * d * self.config.d_ff * 2 * l

        # Embedding and output projection
        emb_flops = batch_size * n * d * v

        total_flops = attn_flops + ffn_flops + emb_flops

        return total_flops

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"TinyLLM(\n"
            f"  vocab_size={self.config.vocab_size},\n"
            f"  d_model={self.config.d_model},\n"
            f"  n_heads={self.config.n_heads},\n"
            f"  n_layers={self.config.n_layers},\n"
            f"  d_ff={self.config.d_ff},\n"
            f"  parameters={self.get_num_params():,}\n"
            f")"
        )


if __name__ == "__main__":
    # Test the model
    from config.model_configs import TINY_CONFIG

    print("Creating model with TINY_CONFIG:")
    model = TinyLLM(TINY_CONFIG)
    print(model)

    # Test forward pass
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, TINY_CONFIG.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, TINY_CONFIG.vocab_size, (batch_size, seq_len))

    print(f"\nInput shape: {input_ids.shape}")

    outputs = model(input_ids, labels=labels)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")

    # Test generation
    print("\nTesting generation:")
    prompt = torch.randint(0, TINY_CONFIG.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    print(f"Generated shape: {generated.shape}")

    # Estimate FLOPs
    flops = model.estimate_flops(batch_size, seq_len)
    print(f"\nEstimated FLOPs: {flops:,}")

    print("\nAll tests passed!")
