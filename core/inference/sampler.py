"""Sampling strategies for text generation."""

import torch
import torch.nn.functional as F
from typing import Optional


def top_k_sampling(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Top-k sampling.

    Args:
        logits: Logits tensor
        k: Number of top tokens to keep
        temperature: Sampling temperature

    Returns:
        Sampled token indices
    """
    # Apply temperature
    logits = logits / temperature

    # Get top k
    top_k_logits, top_k_indices = torch.topk(logits, k)

    # Sample from top k
    probs = F.softmax(top_k_logits, dim=-1)
    sampled_index = torch.multinomial(probs, num_samples=1)

    # Map back to original indices
    return top_k_indices.gather(-1, sampled_index)


def nucleus_sampling(
    logits: torch.Tensor,
    p: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Nucleus (top-p) sampling.

    Args:
        logits: Logits tensor
        p: Cumulative probability threshold
        temperature: Sampling temperature

    Returns:
        Sampled token indices
    """
    # Apply temperature
    logits = logits / temperature

    # Sort logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 0] = False  # Keep at least one token

    # Set removed indices to -inf
    sorted_logits[sorted_indices_to_remove] = float('-inf')

    # Sample
    probs = F.softmax(sorted_logits, dim=-1)
    sampled_sorted_index = torch.multinomial(probs, num_samples=1)

    # Map back to original indices
    return sorted_indices.gather(-1, sampled_sorted_index)


def greedy_sampling(logits: torch.Tensor) -> torch.Tensor:
    """Greedy sampling (argmax).

    Args:
        logits: Logits tensor

    Returns:
        Token with highest probability
    """
    return torch.argmax(logits, dim=-1, keepdim=True)


def beam_search(
    model,
    input_ids: torch.Tensor,
    num_beams: int = 4,
    max_length: int = 100,
    eos_token_id: int = 2,
) -> torch.Tensor:
    """Beam search decoding.

    Args:
        model: Language model
        input_ids: Input token IDs
        num_beams: Number of beams
        max_length: Maximum sequence length
        eos_token_id: End-of-sequence token ID

    Returns:
        Best sequence found
    """
    batch_size = input_ids.shape[0]
    device = input_ids.device

    # Initialize beams
    beam_scores = torch.zeros(batch_size, num_beams, device=device)
    beam_scores[:, 1:] = float('-inf')

    beam_sequences = input_ids.unsqueeze(1).expand(-1, num_beams, -1).reshape(batch_size * num_beams, -1)

    for _ in range(max_length - input_ids.shape[1]):
        with torch.no_grad():
            outputs = model(beam_sequences)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

        next_token_logits = logits[:, -1, :]
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)

        # Reshape for beam operations
        next_token_scores = next_token_scores.view(batch_size, num_beams, -1)

        # Add beam scores
        next_token_scores = next_token_scores + beam_scores.unsqueeze(-1)

        # Flatten to find top beams
        next_token_scores = next_token_scores.view(batch_size, -1)
        top_scores, top_indices = torch.topk(next_token_scores, num_beams, dim=1)

        # Get beam and token indices
        beam_indices = top_indices // next_token_scores.shape[1]
        token_indices = top_indices % next_token_scores.shape[1]

        # Update sequences
        beam_sequences = beam_sequences.view(batch_size, num_beams, -1)
        beam_sequences = beam_sequences.gather(
            1,
            beam_indices.unsqueeze(-1).expand(-1, -1, beam_sequences.shape[-1])
        )
        beam_sequences = beam_sequences.view(batch_size * num_beams, -1)

        next_tokens = token_indices.view(batch_size * num_beams, 1)
        beam_sequences = torch.cat([beam_sequences, next_tokens], dim=-1)

        beam_scores = top_scores

        # Check for EOS
        if (next_tokens == eos_token_id).all():
            break

    # Return best sequence
    best_beam = beam_sequences.view(batch_size, num_beams, -1)[:, 0, :]
    return best_beam
