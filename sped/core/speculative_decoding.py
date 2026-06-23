"""High-level speculative decoder orchestrating draft → verify → accept."""

from typing import Optional, Callable
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class SpeculativeDecoder:
    """Orchestrates draft generation, verification, and acceptance.

    Supports vocabulary-agnostic draft-target pairs via an optional
    alignment layer.
    """

    def __init__(
        self,
        target_model: PreTrainedModel,
        target_tokenizer: PreTrainedTokenizer,
        draft_model: Optional[PreTrainedModel] = None,
        draft_tokenizer: Optional[PreTrainedTokenizer] = None,
        vocab_aligner: Optional[Callable] = None,
        max_draft_tokens: int = 5,
        device: str = "auto",
    ):
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
        self.vocab_aligner = vocab_aligner
        self.max_draft_tokens = max_draft_tokens
        self.device = device

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        draft_k: int = 5,
        verbose: bool = False,
    ) -> str:
        """Generate text using speculative decoding.

        Args:
            prompt: Input text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            draft_k: Number of draft tokens per speculation step.
            verbose: Print per-step stats.

        Returns:
            Generated text.
        """
        if self.draft_model is not None and self.draft_tokenizer is not None:
            return self._speculate(
                prompt, max_new_tokens, temperature, draft_k, verbose
            )
        else:
            return self._standard_generate(prompt, max_new_tokens, temperature)

    def _speculate(self, prompt, max_new_tokens, temperature, draft_k, verbose):
        """Draft-then-verify loop."""
        raise NotImplementedError("Full implementation coming in next iteration.")

    def _standard_generate(self, prompt, max_new_tokens, temperature):
        """Fallback to standard autoregressive generation."""
        inputs = self.target_tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.target_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )
        return self.target_tokenizer.decode(outputs[0], skip_special_tokens=True)
