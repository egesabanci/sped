"""Heterogeneous-vocabulary speculative decoder (Intel/Weizmann approach)."""

from sped.vocab_agnostic.alignment import VocabAligner


class HeterogeneousDecoder:
    """Speculative decoder for draft-target pairs with different vocabularies.

    Wraps vocabulary alignment around the standard speculation loop.
    """

    def __init__(
        self,
        target_model,
        target_tokenizer,
        draft_model,
        draft_tokenizer,
        align_strategy: str = "hybrid",
    ):
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
        self.aligner = VocabAligner(
            target_tokenizer=target_tokenizer,
            draft_tokenizer=draft_tokenizer,
            strategy=align_strategy,
        )

    @property
    def vocabularies_match(self) -> bool:
        """Check if draft and target share the same tokenizer."""
        return self.draft_tokenizer.vocab_size == self.target_tokenizer.vocab_size
