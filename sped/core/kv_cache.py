"""KV cache management for speculative decoding.

Handles cache extension, truncation, and sharing across speculation steps
for both draft and target models.
"""

from typing import Optional
import torch
from dataclasses import dataclass


@dataclass
class CacheState:
    """Tracks KV cache state across speculation iterations."""

    past_key_values: Optional[tuple[tuple[torch.Tensor, ...], ...]] = None
    seq_len: int = 0


class KVCacheManager:
    """Manages KV cache for speculative decoding steps.

    Supports:
    - Extension after accepted tokens
    - Truncation on rejection (rollback to pre-draft state)
    - Prefix sharing across tree branches
    """

    def __init__(self, model, max_length: int = 8192, device: str = "cpu"):
        self.model = model
        self.max_length = max_length
        self.device = device
        self.cache: Optional[tuple[tuple[torch.Tensor, ...], ...]] = None
        self.base_seq_len: int = 0  # length before draft speculation

    @staticmethod
    def _unpack_output(outputs):
        """Extract logits and past_key_values from a model forward pass.

        Handles three output formats:
        1. HF CausalLMOutputWithPast: has .logits and .past_key_values
        2. Unsloth fast-inference: returns (logits, pkv, ...) as tuple
        """
        if hasattr(outputs, "logits"):
            return outputs.logits, outputs.past_key_values
        # Unsloth fast path returns (logits, pkv, ...) tuple
        logits = outputs[0]
        pkv = outputs[1]
        return logits, pkv

    def reset(self):
        """Clear the cache entirely."""
        self.cache = None
        self.base_seq_len = 0

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run initial forward pass to populate KV cache from a prefix.

        Args:
            input_ids: Shape (1, seq_len) — the prompt prefix.

        Returns:
            logits for the last position.
        """
        outputs = self.model(
            input_ids,
            use_cache=True,
            past_key_values=None,
        )
        logits, pkv = self._unpack_output(outputs)
        self.cache = pkv
        self.base_seq_len = input_ids.shape[-1]
        return logits

    @torch.no_grad()
    def extend(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Extend cache with new tokens (greedy single-step forward).

        Args:
            token_ids: Shape (1, n_tokens) — new tokens to process.

        Returns:
            logits: (1, n_tokens, vocab_size) — logits for new positions.
        """
        if self.cache is None:
            return self.prefill(token_ids)

        # Compute position_ids for new tokens (required by unsloth's patched forward)
        n_new = token_ids.shape[-1]
        position_ids = torch.arange(
            self.base_seq_len,
            self.base_seq_len + n_new,
            dtype=torch.long,
            device=token_ids.device,
        ).unsqueeze(0)

        outputs = self.model(
            token_ids,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=self.cache,
        )
        logits, pkv = self._unpack_output(outputs)
        self.cache = pkv
        self.base_seq_len += n_new
        return logits

    @torch.no_grad()
    def verify_draft(
        self,
        input_ids: torch.Tensor,
        draft_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Verify draft tokens incrementally using KV cache.

        Uses the existing KV cache to process only the K draft tokens,
        avoiding a full forward pass over the entire context.

        Falls back to full forward if:
        - No cache is available (cold start)
        - Unsloth's fast path rejects multi-token extension

        Args:
            input_ids: Shape (1, seq_len) — known prefix (used for fallback only).
            draft_ids: Shape (1, draft_k) — candidate draft tokens.

        Returns:
            logits at each draft position (1, draft_k, vocab_size).
        """
        draft_k = draft_ids.shape[-1]

        if self.cache is not None:
            # Attempt incremental: extend cache with draft tokens
            try:
                logits = self.extend(draft_ids)
                # extend() advances base_seq_len by draft_k; we need to rollback
                # to the pre-draft state since verification hasn't been committed yet
                self.base_seq_len -= draft_k
                # Return logits at each draft position
                return logits[:, -draft_k:, :]
            except (RuntimeError, AssertionError):
                # Unsloth fast path may reject multi-token extension (q_len=1 assert)
                # Fall through to incremental single-token approach
                # Rollback the partial extend
                self.rollback()

        # Fallback 1: incremental single-token extends (for Unsloth compatibility)
        if self.cache is not None:
            draft_logits_list = []
            saved_base = self.base_seq_len
            try:
                for i in range(draft_k):
                    tok = draft_ids[:, i:i+1]
                    logits = self.extend(tok)  # q_len=1, safe for Unsloth
                    draft_logits_list.append(logits[:, -1:, :])
                # Rollback to pre-draft state
                self.base_seq_len = saved_base
                self.rollback()
                return torch.cat(draft_logits_list, dim=1)
            except (RuntimeError, AssertionError):
                self.base_seq_len = saved_base
                self.rollback()

        # Fallback 2: full forward pass (original behavior)
        combined = torch.cat([input_ids, draft_ids], dim=-1)
        outputs = self.model(
            combined,
            use_cache=True,
            past_key_values=None,
        )
        logits, pkv = self._unpack_output(outputs)
        # Return logits at draft positions only
        logits = logits[:, input_ids.shape[-1] - 1 : -1, :]
        self.cache = pkv
        self.base_seq_len = combined.shape[-1]
        return logits

    def commit(self, num_tokens: int):
        """Commit accepted tokens to the base prefix length.

        Called after rejection sampling to mark accepted tokens as
        the new baseline for the next speculation round.
        """
        self.base_seq_len += num_tokens

    def rollback(self):
        """Rollback cache to the last committed prefix.

        Called when draft tokens are rejected — discards all KV entries
        beyond the committed prefix.
        """
        if self.cache is None:
            return

        # Truncate KV cache to base_seq_len
        truncated = []
        for layer_kv in self.cache:
            layer_trunc = []
            for kv in layer_kv:
                layer_trunc.append(kv[:, :, : self.base_seq_len, :])
            truncated.append(tuple(layer_trunc))
        self.cache = tuple(truncated)

    def clone_prefix(self, prefix_length: int) -> Optional[tuple]:
        """Clone the KV cache up to a given prefix length.

        Used for tree attention where multiple branches share a prefix.
        """
        if self.cache is None:
            return None

        cloned = []
        for layer_kv in self.cache:
            layer_clone = []
            for kv in layer_kv:
                layer_clone.append(kv[:, :, :prefix_length, :].clone())
            cloned.append(tuple(layer_clone))
        return tuple(cloned)

    @property
    def is_full(self) -> bool:
        """Check if cache is approaching the maximum context length."""
        return self.base_seq_len >= self.max_length - 128

    @property
    def usage_ratio(self) -> float:
        """Fraction of max context length used."""
        return self.base_seq_len / self.max_length
