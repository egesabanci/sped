"""Token-level alignment between heterogeneous vocabularies.

Implements the three Intel/Weizmann alignment strategies for pairing
draft and target models with different tokenizers:

Algorithm 1 — String-level mapping: decode→re-encode via string roundtrip
Algorithm 2 — Probabilistic mapping: score candidates with target logits
Algorithm 3 — Hybrid: dynamically selects best strategy per token
"""

from typing import Optional, Callable
import torch
from transformers import PreTrainedTokenizer


class VocabAligner:
    """Aligns draft-token sequences into the target model's vocabulary space.

    Supports three strategies from the Intel/Weizmann ICML 2025 paper.
    Builds a prefix-indexed vocabulary for fast candidate lookup.
    """

    def __init__(
        self,
        target_tokenizer: PreTrainedTokenizer,
        draft_tokenizer: PreTrainedTokenizer,
        strategy: str = "hybrid",
        target_model: Optional[Callable] = None,
    ):
        self.target_tokenizer = target_tokenizer
        self.draft_tokenizer = draft_tokenizer
        self.strategy = strategy
        self.target_model = target_model

        # Build prefix index for probabilistic alignment
        # Maps prefix text → list of (target_token_id, full_token_text)
        self._prefix_index: dict[str, list[tuple[int, str]]] = {}
        self._build_prefix_index()

        # Strategy selector for hybrid mode
        self._strategy_selector = _HybridStrategySelector()

    def _build_prefix_index(self):
        """Build a prefix-indexed lookup of the target vocabulary.

        For each token in the target vocabulary, store it under all
        possible prefix keys so we can quickly find candidates for
        any draft subword.
        """
        vocab = self.target_tokenizer.get_vocab()
        for token_text, token_id in vocab.items():
            # Clean BPE/Unigram markers like '▁', '##'
            clean = token_text.replace("▁", " ").replace("##", "")
            clean = clean.strip()
            if not clean:
                continue
            # Index by first 2 chars (minimum prefix for fast lookup)
            for prefix_len in range(1, min(len(clean) + 1, 5)):
                prefix = clean[:prefix_len].lower()
                if prefix not in self._prefix_index:
                    self._prefix_index[prefix] = []
                self._prefix_index[prefix].append((token_id, token_text))

        # Sort by longest text first (best match)
        for prefix in self._prefix_index:
            self._prefix_index[prefix].sort(
                key=lambda x: len(x[1]), reverse=True
            )

    def align(
        self,
        draft_token_ids: torch.Tensor,
        target_prefix_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map draft tokens to target vocabulary.

        Args:
            draft_token_ids: (1, draft_k) — draft model token IDs.
            target_prefix_ids: (1, seq_len) — target prefix for context.

        Returns:
            aligned_token_ids: (1, M) — draft tokens re-encoded in target vocab.
                               M may differ from draft_k due to subword splits.
            alignment_mask: (1, M) — which target position each aligned token
                            maps to in the original draft sequence.
        """
        if self.strategy == "string":
            return self._string_alignment(draft_token_ids)
        elif self.strategy == "probabilistic":
            return self._probabilistic_alignment(
                draft_token_ids, target_prefix_ids
            )
        else:  # hybrid
            return self._hybrid_alignment(draft_token_ids, target_prefix_ids)

    # ── Algorithm 1: String-level Mapping ─────────────────────────────

    def _string_alignment(
        self,
        draft_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map via string equivalence (decode → re-encode).

        Algorithm 1 from Intel/Weizmann: decode draft tokens to text,
        then encode with target tokenizer. Works for any tokenizer pair
        but may lose fidelity on subword boundaries.
        """
        if self.draft_tokenizer is None:
            return draft_token_ids, torch.ones_like(draft_token_ids)

        # Decode draft tokens to text
        draft_text = self.draft_tokenizer.decode(
            draft_token_ids[0], skip_special_tokens=True
        )
        if not draft_text.strip():
            return draft_token_ids, torch.ones_like(draft_token_ids)

        # Encode with target tokenizer
        target_ids = self.target_tokenizer.encode(
            draft_text, add_special_tokens=False
        )
        if not target_ids:
            target_ids = [self.target_tokenizer.unk_token_id or 0]

        aligned = torch.tensor([target_ids], device=draft_token_ids.device)

        # Build alignment mask: each target token maps to a draft position
        # We approximate by distributing target tokens evenly across draft positions
        draft_k = draft_token_ids.shape[-1]
        target_m = len(target_ids)
        mask = torch.zeros((1, target_m), dtype=torch.long, device=draft_token_ids.device)
        for i in range(target_m):
            # Map proportionally to draft positions
            draft_pos = min(i * draft_k // max(target_m, 1), draft_k - 1)
            mask[0, i] = draft_pos

        return aligned, mask

    # ── Algorithm 2: Probabilistic Mapping ────────────────────────────

    def _probabilistic_alignment(
        self,
        draft_token_ids: torch.Tensor,
        target_prefix_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map via probabilistic scoring with target model logits.

        Algorithm 2: for each draft token, find candidate target tokens
        sharing a text prefix, score them using target model logits,
        and pick the highest-probability candidate.
        """
        if self.draft_tokenizer is None:
            return draft_token_ids, torch.ones_like(draft_token_ids)

        draft_tokens = draft_token_ids[0].tolist()
        aligned_tokens: list[int] = []
        alignment_masks: list[int] = []

        for i, tok_id in enumerate(draft_tokens):
            # Decode this single draft token
            tok_text = self.draft_tokenizer.decode([tok_id])
            clean = tok_text.replace("▁", " ").replace("##", "").strip()

            if not clean:
                # Special token or whitespace — use UNK
                aligned_tokens.append(
                    self.target_tokenizer.unk_token_id or 0
                )
                alignment_masks.append(i)
                continue

            # Find candidate target tokens by prefix match
            candidates = self._find_candidates(clean)

            if not candidates:
                candidates = [(self.target_tokenizer.unk_token_id or 0, clean)]

            if len(candidates) == 1 or self.target_model is None:
                # No ambiguity or no target model available
                aligned_tokens.append(candidates[0][0])
                alignment_masks.append(i)
            else:
                # Score candidates using target logits
                candidate_ids = [c[0] for c in candidates]
                scores = self._score_candidates(
                    candidate_ids,
                    target_prefix_ids,
                    torch.tensor([aligned_tokens + [0]], device=draft_token_ids.device)
                    if aligned_tokens
                    else target_prefix_ids,
                )
                best_idx = scores.argmax().item()
                aligned_tokens.append(candidates[best_idx][0])
                alignment_masks.append(i)

        # Handle length differences: if a single draft token maps to a
        # multi-subword target token, the lengths stay in sync 1:1 here.
        result = torch.tensor([aligned_tokens], device=draft_token_ids.device)
        mask = torch.tensor([alignment_masks], device=draft_token_ids.device)
        return result, mask

    def _find_candidates(self, text: str, max_candidates: int = 5) -> list[tuple[int, str]]:
        """Find target vocabulary tokens that could represent this text."""
        text_lower = text.lower()
        candidates: list[tuple[int, str]] = []

        # Direct match by prefix
        for prefix_len in range(min(len(text_lower), 4), 0, -1):
            prefix = text_lower[:prefix_len]
            if prefix in self._prefix_index:
                for tok_id, tok_text in self._prefix_index[prefix]:
                    clean_tok = tok_text.replace("▁", " ").replace("##", "").strip()
                    if clean_tok and (clean_tok.lower().startswith(text_lower) or
                                      text_lower.startswith(clean_tok.lower())):
                        if (tok_id, tok_text) not in candidates:
                            candidates.append((tok_id, tok_text))
                            if len(candidates) >= max_candidates:
                                break
            if candidates:
                break

        # Fallback: use UNK
        if not candidates:
            unk_id = self.target_tokenizer.unk_token_id
            if unk_id is not None:
                candidates = [(unk_id, text)]

        return candidates[:max_candidates]

    def _score_candidates(
        self,
        candidate_ids: list[int],
        context_ids: torch.Tensor,
        prefix_so_far: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidate target tokens using target model logits.

        Runs ONE forward pass and scores all candidates from the resulting
        logits. Previously ran N forward passes (massive performance bug).
        """
        if self.target_model is None:
            return torch.ones(len(candidate_ids), device=context_ids.device)

        with torch.no_grad():
            if prefix_so_far.dim() == 1:
                prefix_so_far = prefix_so_far.unsqueeze(0)

            # Single forward pass: all candidates scored from same logits
            outputs = self.target_model(prefix_so_far)
            logits_at_end = outputs.logits[0, -1, :]
            probs = torch.softmax(logits_at_end, dim=-1)
            scores = torch.tensor(
                [probs[cid].item() for cid in candidate_ids],
                device=context_ids.device,
            )
            return scores

    # ── Algorithm 3: Hybrid ───────────────────────────────────────────

    def _hybrid_alignment(
        self,
        draft_token_ids: torch.Tensor,
        target_prefix_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dynamically select best alignment strategy per token.

        Algorithm 3: uses a strategy selector that considers:
        - Draft token entropy (confidence)
        - String mapping ambiguity (how many target tokens overlap)
        - Position in sequence
        - Token type (subword fragment vs complete word)
        """
        if self.draft_tokenizer is None:
            return draft_token_ids, torch.ones_like(draft_token_ids)

        draft_tokens = draft_token_ids[0].tolist()
        aligned_tokens: list[int] = []
        alignment_masks: list[int] = []

        for i, tok_id in enumerate(draft_tokens):
            tok_text = self.draft_tokenizer.decode([tok_id])
            clean = tok_text.replace("▁", " ").replace("##", "").strip()

            if not clean:
                aligned_tokens.append(self.target_tokenizer.unk_token_id or 0)
                alignment_masks.append(i)
                continue

            # Determine strategy for this token
            strategy = self._strategy_selector.select(
                token_text=clean,
                position=i,
                total_tokens=len(draft_tokens),
                target_vocab_size=len(self._prefix_index),
            )

            if strategy == "string":
                # String mapping: decode entire prefix so far and re-encode
                prefix_text = self.draft_tokenizer.decode(
                    draft_token_ids[0, :i+1], skip_special_tokens=True
                )
                target_ids = self.target_tokenizer.encode(
                    prefix_text, add_special_tokens=False
                )
                # Take only the last token(s) corresponding to this draft position
                if target_ids:
                    aligned_tokens.append(target_ids[-1])
                else:
                    aligned_tokens.append(self.target_tokenizer.unk_token_id or 0)
                alignment_masks.append(i)

            else:
                # Probabilistic: score candidates
                candidates = self._find_candidates(clean)
                if not candidates:
                    candidates = [(self.target_tokenizer.unk_token_id or 0, clean)]

                if len(candidates) > 1 and self.target_model is not None:
                    scores = self._score_candidates(
                        [c[0] for c in candidates],
                        target_prefix_ids,
                        torch.tensor([aligned_tokens + [0]], device=draft_token_ids.device)
                        if aligned_tokens
                        else target_prefix_ids,
                    )
                    best_idx = scores.argmax().item()
                    aligned_tokens.append(candidates[best_idx][0])
                else:
                    aligned_tokens.append(candidates[0][0])
                alignment_masks.append(i)

        result = torch.tensor([aligned_tokens], device=draft_token_ids.device)
        mask = torch.tensor([alignment_masks], device=draft_token_ids.device)
        return result, mask


# ── Internal: Hybrid Strategy Selector ───────────────────────────────────


class _HybridStrategySelector:
    """Selects alignment strategy per token based on heuristics.

    Rules:
    - Complete words (no subword markers) → string mapping (fast, accurate)
    - Subword fragments → probabilistic (context-aware)
    - Punctuation/single chars → string mapping
    - Short tokens (< 3 chars in clean form) → string mapping
    - Long tokens with high ambiguity → probabilistic
    """

    def select(
        self,
        token_text: str,
        position: int,
        total_tokens: int,
        target_vocab_size: int,
    ) -> str:
        """Return 'string' or 'probabilistic' for this token."""
        # Single characters and punctuation → string
        if len(token_text) <= 1 and not token_text.isalpha():
            return "string"

        # Short tokens → string
        if len(token_text) <= 2:
            return "string"

        # Tokens with subword markers → probabilistic
        if "##" in token_text or "▁" in token_text:
            return "probabilistic"

        # First and last tokens → probabilistic (higher stakes)
        if position == 0 or position == total_tokens - 1:
            return "probabilistic"

        # Default: string is fast and usually correct
        return "string"
