"""Token-level alignment between heterogeneous vocabularies."""

from typing import Optional
import torch
from transformers import PreTrainedTokenizer


class VocabAligner:
    """Aligns draft-token sequences into the target model's vocabulary space.

    Implements the three alignment strategies from Intel/Weizmann:
        1. String-level mapping (subword → subword)
        2. Probabilistic mapping via target tokenizer
        3. Hybrid (dynamic selection)
    """

    def __init__(
        self,
        target_tokenizer: PreTrainedTokenizer,
        draft_tokenizer: Optional[PreTrainedTokenizer] = None,
        strategy: str = "hybrid",
    ):
        self.target_tokenizer = target_tokenizer
        self.draft_tokenizer = draft_tokenizer
        self.strategy = strategy

    def align(
        self,
        draft_token_ids: torch.Tensor,
        target_prefix_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map draft tokens to target vocabulary.

        Returns:
            aligned_token_ids: Draft tokens re-encoded in target vocab.
            alignment_mask: Boolean mask of valid alignments.
        """
        if self.strategy == "string":
            return self._string_alignment(draft_token_ids)
        elif self.strategy == "probabilistic":
            return self._probabilistic_alignment(draft_token_ids, target_prefix_ids)
        else:
            return self._hybrid_alignment(draft_token_ids, target_prefix_ids)

    def _string_alignment(self, draft_token_ids: torch.Tensor):
        """Algorithm 1: Map via string equivalence."""
        draft_text = self.draft_tokenizer.decode(
            draft_token_ids[0], skip_special_tokens=True
        )
        target_ids = self.target_tokenizer.encode(
            draft_text, add_special_tokens=False
        )
        return torch.tensor([target_ids], device=draft_token_ids.device), None

    def _probabilistic_alignment(self, draft_token_ids, target_prefix_ids):
        """Algorithm 2: Probabilistic mapping via target tokenizer scores."""
        raise NotImplementedError("Coming in next iteration.")

    def _hybrid_alignment(self, draft_token_ids, target_prefix_ids):
        """Algorithm 3: Dynamically selects best strategy."""
        raise NotImplementedError("Coming in next iteration.")
