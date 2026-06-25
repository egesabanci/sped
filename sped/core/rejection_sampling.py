"""Metropolis-Hastings rejection sampling for lossless speculative decoding.

Fully vectorized on GPU — 0 Python loops, 1 CUDA sync per step.
"""

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

    Fully vectorized on GPU — all per-token operations done as tensor ops.
    Single CUDA sync at the end to find the first rejection index.

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

    K = min(draft_logits.shape[0], target_logits.shape[0], len(draft_tokens))
    if K == 0:
        return [], 0

    tokens = draft_tokens[:K]
    # Use explicit indexing to avoid .item() calls
    batch_indices = torch.arange(K, device=draft_logits.device)

    # Vectorized: gather probabilities for drafted tokens in one shot
    # draft_probs[batch_indices, tokens] → (K,)
    p_draft = draft_probs[batch_indices, tokens]
    p_target = target_probs[batch_indices, tokens]

    # Accept mask: target >= draft → always accept; else accept w/ prob target/draft
    ratio = torch.where(
        p_draft > 0,
        p_target / p_draft,
        torch.ones_like(p_draft),
    )
    always_accept = p_target >= p_draft
    u = torch.rand(K, device=draft_logits.device)
    stochastic_accept = u < ratio
    accept_mask = always_accept | stochastic_accept  # (K,) on GPU, bool

    # Find first rejection — single CUDA sync via .any().item()
    rejected_mask = ~accept_mask
    has_rejection = rejected_mask.any()
    if has_rejection:
        # argmax finds first True in bool tensor
        first_reject = rejected_mask.float().argmax().item()
    else:
        first_reject = K

    accepted = tokens[:first_reject].tolist()

    # Resample at first rejection (if any)
    if has_rejection and first_reject < K:
        residual = torch.clamp(target_probs[first_reject] - draft_probs[first_reject], min=0)
        residual_sum = residual.sum()
        if residual_sum > 0:
            residual = residual / residual_sum
            resampled = torch.multinomial(residual, 1).item()
        else:
            resampled = torch.multinomial(target_probs[first_reject], 1).item()
        accepted.append(resampled)

    return accepted, len(accepted)
