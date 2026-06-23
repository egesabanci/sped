"""Parallel verification of draft tokens."""

import torch


class Verifier:
    """Verifies draft tokens against the target model in a single forward pass."""

    def __init__(self, target_model, target_tokenizer, device="cuda"):
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.device = device

    @torch.no_grad()
    def verify_draft(
        self,
        input_ids: torch.Tensor,
        draft_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run a single forward pass over the prefix + draft tokens.

        Args:
            input_ids: Shape (1, seq_len) — the known prefix.
            draft_token_ids: Shape (1, draft_k) — candidate draft tokens.

        Returns:
            logits: Shape (1, draft_k, vocab_size) — target model logits
                    at each draft position.
        """
        combined = torch.cat([input_ids, draft_token_ids], dim=-1)
        outputs = self.target_model(combined)
        # Logits at the draft positions only
        logits = outputs.logits[:, input_ids.shape[-1] - 1 : -1, :]
        return logits
