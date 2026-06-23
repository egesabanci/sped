"""Tests for vocabulary-agnostic alignment (Phase 3)."""

import torch
import pytest


# ── VocabAligner Tests ───────────────────────────────────


class TestVocabAlignerInit:
    def test_init_with_tokenizers(self):
        from transformers import AutoTokenizer
        from sped.vocab_agnostic.alignment import VocabAligner

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-GPTNeoXForCausalLM")
        aligner = VocabAligner(
            target_tokenizer=tokenizer,
            draft_tokenizer=tokenizer,
            strategy="hybrid",
        )
        assert aligner.strategy == "hybrid"
        assert aligner.target_tokenizer is not None
        assert aligner.draft_tokenizer is not None
        assert aligner._prefix_index is not None

    def test_build_prefix_index(self):
        from transformers import AutoTokenizer
        from sped.vocab_agnostic.alignment import VocabAligner

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-GPTNeoXForCausalLM")
        aligner = VocabAligner(
            target_tokenizer=tokenizer,
            draft_tokenizer=tokenizer,
        )
        # Prefix index should have entries
        assert len(aligner._prefix_index) > 0


class TestVocabAlignerStringAlignment:
    def test_same_vocab_identity(self):
        from transformers import AutoTokenizer
        from sped.vocab_agnostic.alignment import VocabAligner

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-GPTNeoXForCausalLM")
        aligner = VocabAligner(
            target_tokenizer=tokenizer,
            draft_tokenizer=tokenizer,
            strategy="string",
        )

        # Encode a simple phrase with draft tokenizer,
        # then align (should decode→re-encode to same tokens)
        text = "Hello world"
        draft_ids = tokenizer.encode(text, return_tensors="pt")

        aligned_ids, mask = aligner.align(draft_ids, draft_ids)

        assert aligned_ids is not None
        assert aligned_ids.shape[-1] >= 1
        # The decoded text should match
        aligned_text = tokenizer.decode(aligned_ids[0], skip_special_tokens=True)
        assert aligned_text.strip().lower() == text.strip().lower()

    def test_empty_draft(self):
        from transformers import AutoTokenizer
        from sped.vocab_agnostic.alignment import VocabAligner

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-GPTNeoXForCausalLM")
        aligner = VocabAligner(
            target_tokenizer=tokenizer,
            draft_tokenizer=tokenizer,
            strategy="string",
        )

        # Empty tensor
        draft_ids = torch.tensor([[]], dtype=torch.long)
        aligned_ids, mask = aligner.align(draft_ids, draft_ids)
        assert aligned_ids is not None


class TestVocabAlignerProbabilistic:
    def test_find_candidates(self):
        from transformers import AutoTokenizer
        from sped.vocab_agnostic.alignment import VocabAligner

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-GPTNeoXForCausalLM")
        aligner = VocabAligner(
            target_tokenizer=tokenizer,
            draft_tokenizer=tokenizer,
            strategy="probabilistic",
        )

        # Find candidates for a common word
        candidates = aligner._find_candidates("hello")
        assert candidates is not None
        assert len(candidates) > 0
        assert all(isinstance(c, tuple) for c in candidates)

    def test_find_candidates_empty_text(self):
        from transformers import AutoTokenizer
        from sped.vocab_agnostic.alignment import VocabAligner

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-GPTNeoXForCausalLM")
        aligner = VocabAligner(
            target_tokenizer=tokenizer,
            draft_tokenizer=tokenizer,
        )

        # Unknown text should return at least UNK candidate
        candidates = aligner._find_candidates("xyznonexistent12345")
        assert len(candidates) >= 1


class TestVocabAlignerHybrid:
    def test_hybrid_alignment(self):
        from transformers import AutoTokenizer
        from sped.vocab_agnostic.alignment import VocabAligner

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-GPTNeoXForCausalLM")
        aligner = VocabAligner(
            target_tokenizer=tokenizer,
            draft_tokenizer=tokenizer,
            strategy="hybrid",
        )

        text = "test input"
        draft_ids = tokenizer.encode(text, return_tensors="pt")
        aligned_ids, mask = aligner.align(draft_ids, draft_ids)

        assert aligned_ids is not None
        assert mask is not None


# ── Strategy Selector Tests ──────────────────────────────


class TestHybridStrategySelector:
    def test_single_char_returns_string(self):
        from sped.vocab_agnostic.alignment import _HybridStrategySelector

        sel = _HybridStrategySelector()
        result = sel.select(",", 0, 10, 1000)
        assert result == "string"

    def test_short_token_returns_string(self):
        from sped.vocab_agnostic.alignment import _HybridStrategySelector

        sel = _HybridStrategySelector()
        result = sel.select("ab", 2, 10, 1000)
        assert result == "string"

    def test_subword_marker_returns_probabilistic(self):
        from sped.vocab_agnostic.alignment import _HybridStrategySelector

        sel = _HybridStrategySelector()
        result = sel.select("##ing", 2, 10, 1000)
        assert result == "probabilistic"

    def test_first_token_returns_probabilistic(self):
        from sped.vocab_agnostic.alignment import _HybridStrategySelector

        sel = _HybridStrategySelector()
        result = sel.select("hello", 0, 10, 1000)
        assert result == "probabilistic"

    def test_middle_token_defaults_string(self):
        from sped.vocab_agnostic.alignment import _HybridStrategySelector

        sel = _HybridStrategySelector()
        result = sel.select("hello", 3, 10, 1000)
        assert result == "string"


# ── HeterogeneousDecoder Tests ───────────────────────────


class TestHeterogeneousDecoder:
    def test_init(self):
        from transformers import AutoTokenizer
        from sped.vocab_agnostic.heterogeneous import HeterogeneousDecoder

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-GPTNeoXForCausalLM")
        decoder = HeterogeneousDecoder(
            target_model=None,
            target_tokenizer=tokenizer,
            draft_model=None,
            draft_tokenizer=tokenizer,
        )
        assert decoder.aligner is not None
        assert decoder.vocabularies_match is True

    def test_vocabularies_match_same_model(self):
        from transformers import AutoTokenizer
        from sped.vocab_agnostic.heterogeneous import HeterogeneousDecoder

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-GPTNeoXForCausalLM")
        decoder = HeterogeneousDecoder(
            target_model=None,
            target_tokenizer=tokenizer,
            draft_model=None,
            draft_tokenizer=tokenizer,
        )
        assert decoder.vocabularies_match

    def test_align_draft_returns_tensors(self):
        from transformers import AutoTokenizer
        from sped.vocab_agnostic.heterogeneous import HeterogeneousDecoder

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-GPTNeoXForCausalLM")
        decoder = HeterogeneousDecoder(
            target_model=None,
            target_tokenizer=tokenizer,
            draft_model=None,
            draft_tokenizer=tokenizer,
        )

        draft_ids = torch.tensor([[1, 2, 3]])
        context_ids = torch.tensor([[0, 1]])
        aligned, mask = decoder.align_draft(draft_ids, context_ids)
        assert aligned is not None
        assert mask is not None


# ── Heterogeneous Rejection Sampling Tests ───────────────


class TestHeterogeneousRejectionSample:
    def test_accept_when_target_confident(self):
        from sped.vocab_agnostic.heterogeneous import heterogeneous_rejection_sample

        # Target agrees with draft → always accept
        draft_logits = torch.tensor([[1.0, 0.0, 0.0]])
        target_logits = torch.tensor([[0.0, 2.0, 0.0]])
        aligned_tokens = torch.tensor([1])
        alignment_mask = torch.tensor([0])
        draft_original = torch.tensor([0])

        accepted, n = heterogeneous_rejection_sample(
            draft_logits, target_logits, aligned_tokens,
            alignment_mask, draft_original, temperature=0.0,
        )
        assert n >= 1

    def test_reject_on_mismatch(self):
        from sped.vocab_agnostic.heterogeneous import heterogeneous_rejection_sample

        # Target strongly disagrees with draft
        draft_logits = torch.tensor([[10.0, 0.0]])
        target_logits = torch.tensor([[0.0, 10.0]])
        aligned_tokens = torch.tensor([0])
        alignment_mask = torch.tensor([0])
        draft_original = torch.tensor([0])

        accepted, n = heterogeneous_rejection_sample(
            draft_logits, target_logits, aligned_tokens,
            alignment_mask, draft_original, temperature=0.0,
        )
        # Token 0 has p_target=0, so it's rejected and resampled to 1
        assert len(accepted) == 1
        assert accepted[0] == 1  # resampled to target's preference

    def test_multi_token_alignment(self):
        from sped.vocab_agnostic.heterogeneous import heterogeneous_rejection_sample

        # 3 aligned positions, 2 original draft positions
        draft_logits = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
        target_logits = torch.tensor([[3.0, 1.0], [3.0, 1.0], [1.0, 3.0]])
        aligned_tokens = torch.tensor([0, 0, 1])
        alignment_mask = torch.tensor([0, 0, 1])
        draft_original = torch.tensor([0, 1])

        accepted, n = heterogeneous_rejection_sample(
            draft_logits, target_logits, aligned_tokens,
            alignment_mask, draft_original, temperature=1.0,
        )
        assert n >= 1
        assert len(accepted) >= 1


# ── Import tests ─────────────────────────────────────────


class TestVocabImports:
    def test_import_aligner(self):
        from sped.vocab_agnostic import VocabAligner
        assert VocabAligner is not None

    def test_import_decoder(self):
        from sped.vocab_agnostic import HeterogeneousDecoder
        assert HeterogeneousDecoder is not None

    def test_import_rejection(self):
        from sped.vocab_agnostic import heterogeneous_rejection_sample
        assert callable(heterogeneous_rejection_sample)

    def test_import_selector(self):
        from sped.vocab_agnostic import _HybridStrategySelector
        assert _HybridStrategySelector is not None
