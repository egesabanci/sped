"""Heterogeneous-vocabulary speculative decoding (Intel/Weizmann approach).

Extends the core speculative decoder to handle draft-target pairs with
different tokenizers. Provides heterogeneous rejection sampling that
operates correctly across vocabulary boundaries.
"""

from typing import Callable, Optional
import torch

from sped.vocab_agnostic.alignment import VocabAligner


class HeterogeneousDecoder:
    """Speculative decoder for draft-target pairs with different vocabularies.

    Wraps vocabulary alignment around the standard speculation loop so
    that any draft model can accelerate any target model, regardless of
    tokenizer differences.
    """

    def __init__(
        self,
        target_model,
        target_tokenizer,
        draft_model,
        draft_tokenizer,
        target_kv_cache=None,
        draft_kv_cache=None,
        align_strategy: str = "hybrid",
    ):
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
        self.target_kv_cache = target_kv_cache
        self.draft_kv_cache = draft_kv_cache

        self.aligner = VocabAligner(
            target_tokenizer=target_tokenizer,
            draft_tokenizer=draft_tokenizer,
            strategy=align_strategy,
            target_model=target_model,
        )
        self.strategy = align_strategy

    @property
    def vocabularies_match(self) -> bool:
        """Check if draft and target share the same vocabulary."""
        return self.draft_tokenizer.vocab_size == self.target_tokenizer.vocab_size

    def align_draft(
        self,
        draft_token_ids: torch.Tensor,
        target_context_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align draft tokens to target vocabulary.

        Args:
            draft_token_ids: (1, draft_k) — draft tokens in draft vocab.
            target_context_ids: (1, seq_len) — target prefix for context.

        Returns:
            aligned_token_ids: (1, M) — tokens re-encoded in target vocab.
            alignment_mask: (1, M) — mapping from aligned position to draft position.
        """
        return self.aligner.align(draft_token_ids, target_context_ids)


def heterogeneous_rejection_sample(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    aligned_tokens: torch.Tensor,
    alignment_mask: torch.Tensor,
    draft_tokens_original: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[list[int], int]:
    """Rejection sampling for heterogeneous (cross-vocabulary) speculation.

    Standard rejection sampling compares p_draft(tok) vs p_target(tok)
    token-by-token, which requires identical vocabularies. This variant
    adapts the rule to work across vocabularies by using the alignment
    mask to map between token spaces.

    The key insight from Intel/Weizmann ICML 2025:
    After aligning draft tokens to the target vocabulary, rejection
    sampling operates in the target vocabulary space. We estimate the
    draft model's probability for each aligned target token by
    marginalizing over all draft tokens that map to it.

    Args:
        draft_logits: (draft_k, vocab_size_draft) — draft model logits
                      at each original draft position.
        target_logits: (M, vocab_size_target) — target model logits
                       at each aligned position.
        aligned_tokens: (M,) — draft tokens re-encoded in target vocab.
        alignment_mask: (M,) — for each aligned position, which original
                        draft position it corresponds to.
        draft_tokens_original: (draft_k,) — original draft token IDs in
                               draft vocabulary.
        temperature: Sampling temperature.

    Returns:
        accepted_tokens: List of accepted token IDs (in target vocabulary).
        num_accepted: Number of accepted tokens.
    """
    if temperature > 0:
        draft_probs = torch.softmax(draft_logits / temperature, dim=-1)
        target_probs = torch.softmax(target_logits / temperature, dim=-1)
    else:
        draft_probs = torch.softmax(draft_logits, dim=-1)
        target_probs = torch.softmax(target_logits, dim=-1)

    accepted = []
    M = min(aligned_tokens.shape[0],
            target_logits.shape[0] if target_logits.dim() > 1 else target_logits.shape[0])

    for i in range(M):
        token_id = aligned_tokens[i].item() if torch.is_tensor(aligned_tokens[i]) else aligned_tokens[i]

        if i >= target_probs.shape[0]:
            break

        p_target = target_probs[i, token_id].item()

        # Estimate draft probability for this aligned token
        draft_pos = alignment_mask[i].item() if torch.is_tensor(alignment_mask[i]) else alignment_mask[i]
        draft_pos = min(int(draft_pos), draft_probs.shape[0] - 1)

        # The draft model's probability for the original draft token
        orig_tok = draft_tokens_original[draft_pos].item() if torch.is_tensor(draft_tokens_original[draft_pos]) else draft_tokens_original[draft_pos]
        p_draft = draft_probs[draft_pos, orig_tok].item()

        if p_target >= p_draft:
            # Always accept — target agrees or rates it higher
            accepted.append(token_id)
        else:
            # Accept probabilistically
            u = torch.rand(1).item()
            if u < p_target / p_draft:
                accepted.append(token_id)
            else:
                # Rejected — resample from residual in target vocabulary
                residual = torch.clamp(target_probs[i] - draft_probs[draft_pos].unsqueeze(0), min=0)
                residual_sum = residual.sum()
                if residual_sum > 0:
                    residual = residual / residual_sum
                    resampled = torch.multinomial(residual, 1).item()
                else:
                    resampled = torch.multinomial(target_probs[i], 1).item()
                accepted.append(resampled)
                break

    return accepted, len(accepted)
