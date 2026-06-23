"""Metropolis-Hastings rejection sampling for lossless speculative decoding."""

import torch


@torch.no_grad()
def rejection_sample(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[list[int], int]:
    """Apply rejection sampling to decide which draft tokens to accept.

    Implements the standard Metropolis-Hastings acceptance rule:
        Accept t_j if u < min(1, q_j / p_j),  u ~ U(0, 1)

    where q_j = target_prob, p_j = draft_prob.

    Args:
        draft_logits: (draft_k, vocab_size) or (1, draft_k, vocab_size).
        target_logits: (draft_k, vocab_size) or (1, draft_k, vocab_size).
        draft_tokens: (draft_k,) — drafted token IDs.
        temperature: Sampling temperature.

    Returns:
        accepted_tokens: List of accepted token IDs.
        num_accepted: Number of accepted tokens (int).
    """
    # Squeeze batch dimension if present
    if draft_logits.dim() == 3:
        draft_logits = draft_logits[0]
    if target_logits.dim() == 3:
        target_logits = target_logits[0]

    if temperature > 0:
        draft_probs = torch.softmax(draft_logits / temperature, dim=-1)
        target_probs = torch.softmax(target_logits / temperature, dim=-1)
    else:
        draft_probs = torch.softmax(draft_logits, dim=-1)
        target_probs = torch.softmax(target_logits, dim=-1)

    accepted = []
    for i, token_id in enumerate(draft_tokens):
        token_id = token_id.item() if torch.is_tensor(token_id) else token_id

        if i >= draft_probs.shape[0] or i >= target_probs.shape[0]:
            break

        p_draft = draft_probs[i, token_id].item()
        p_target = target_probs[i, token_id].item()

        if p_target >= p_draft:
            # Always accept — target agrees or rates it higher
            accepted.append(token_id)
        else:
            # Accept probabilistically
            u = torch.rand(1).item()
            if u < p_target / p_draft:
                accepted.append(token_id)
            else:
                # Rejected — resample from residual distribution
                residual = torch.clamp(target_probs[i] - draft_probs[i], min=0)
                residual_sum = residual.sum()
                if residual_sum > 0:
                    residual = residual / residual_sum
                    resampled = torch.multinomial(residual, 1).item()
                else:
                    # Fallback: sample from target distribution
                    resampled = torch.multinomial(target_probs[i], 1).item()
                accepted.append(resampled)
                break

    return accepted, len(accepted)
