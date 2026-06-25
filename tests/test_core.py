"""Tests for core speculative decoding engine (Phase 2)."""

import torch
import pytest
from time import time


# ── Rejection Sampling Tests ─────────────────────────────


class TestRejectionSampling:
    def test_accept_all_when_target_more_confident(self):
        from sped.core.rejection_sampling import rejection_sample

        # Target is more confident than draft for all tokens
        draft_logits = torch.tensor([[[0.0, 0.0, 1.0, 0.0]]])  # token 2
        target_logits = torch.tensor([[[0.0, 0.0, 2.0, 0.0]]])  # token 2
        draft_tokens = torch.tensor([2])

        accepted, n = rejection_sample(draft_logits, target_logits, draft_tokens, temperature=1.0)
        assert len(accepted) == 1
        assert accepted[0] == 2
        assert n == 1

    def test_reject_when_draft_overconfident_greedy(self):
        from sped.core.rejection_sampling import rejection_sample

        # Draft is very confident in a token the target disagrees with
        draft_logits = torch.tensor([[[10.0, 0.0, 0.0, 0.0]]])  # token 0
        target_logits = torch.tensor([[[0.0, 10.0, 0.0, 0.0]]])  # token 1
        draft_tokens = torch.tensor([0])

        # At temperature=0 (greedy), we use raw probs
        accepted, n = rejection_sample(draft_logits, target_logits, draft_tokens, temperature=0.0)
        # token 0 has p_target=0.0 < p_draft=1.0, so rejected
        assert n == 0 or accepted[0] != 0

    def test_losslessness_same_distribution(self):
        from sped.core.rejection_sampling import rejection_sample

        # Identical distributions → must always accept
        logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        draft_tokens = torch.tensor([2])

        # Run many times to check losslessness
        results = []
        for _ in range(100):
            accepted, _ = rejection_sample(logits, logits, draft_tokens, temperature=1.0)
            results.append(accepted[0] if accepted else None)

        # Should always accept the draft token since distributions match
        assert all(r == 2 for r in results), f"Not all accepted: {results}"

    def test_resample_on_rejection(self):
        from sped.core.rejection_sampling import rejection_sample

        # Target completely disagrees with draft
        draft_logits = torch.tensor([[[10.0, 0.0, 0.0, 0.0]]])
        target_logits = torch.tensor([[[0.0, 0.0, 0.0, 10.0]]])
        draft_tokens = torch.tensor([0])

        accepted, n = rejection_sample(draft_logits, target_logits, draft_tokens, temperature=1.0)
        # Token 0 should be rejected, resampled from target distribution → should be token 3
        assert len(accepted) == 1
        assert accepted[0] == 3  # resampled from residual (target - draft)
        assert n == 1

    def test_multiple_tokens_partial_acceptance(self):
        from sped.core.rejection_sampling import rejection_sample

        # Token 0: p_target >= p_draft → always accepted
        # Token 1: p_target < p_draft → may be accepted or rejected probabilistically
        # Token 2: draft says 1, target says 0 → rejects and resamples to 0
        draft_logits = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]])
        target_logits = torch.tensor([[[10.0, 0.0], [10.0, 0.0]]])
        draft_tokens = torch.tensor([0, 1])

        accepted, n = rejection_sample(draft_logits, target_logits, draft_tokens, temperature=0.0)
        # Token 0: always accepted
        assert len(accepted) >= 1
        assert accepted[0] == 0
        # Token 1: draft says 1, target says 0 → p_target=0 < p_draft=1 → always reject
        # If rejected, resample from residual (all mass on token 0)
        if len(accepted) > 1:
            assert accepted[1] == 0  # resampled to target's preferred token


# ── Metrics Collector Tests ──────────────────────────────


class TestMetricsCollector:
    def test_initial_state(self):
        from sped.core.metrics import MetricsCollector

        m = MetricsCollector()
        assert m.acceptance_rate == 0.0
        assert m.tokens_per_second == 0.0

    def test_record_single_step(self):
        from sped.core.metrics import MetricsCollector, StepMetrics

        m = MetricsCollector()
        m.start_step()
        m._phase("draft")
        import time as ttime
        ttime.sleep(0.001)
        m._phase("verify")
        ttime.sleep(0.001)
        m._phase("sampling")
        ttime.sleep(0.001)
        m.end_step(draft_k=5, num_accepted=4, tokens_generated=4)

        assert m.acceptance_rate == 0.8  # 4/5
        assert m.cumulative.total_steps == 1
        assert m.tokens_per_step == 4.0

    def test_multiple_steps(self):
        from sped.core.metrics import MetricsCollector

        m = MetricsCollector()
        for _ in range(10):
            m.start_step()
            m.end_step(draft_k=5, num_accepted=4, tokens_generated=4)

        assert m.acceptance_rate == 0.8
        assert m.cumulative.total_steps == 10
        assert m.cumulative.total_draft_tokens == 50
        assert m.cumulative.total_accepted == 40

    def test_summary_dict(self):
        from sped.core.metrics import MetricsCollector

        m = MetricsCollector()
        m.start_step()
        m.end_step(draft_k=5, num_accepted=3, tokens_generated=3)

        s = m.summary()
        assert "acceptance_rate" in s
        assert "avg_tokens_per_step" in s
        assert "avg_tokens_per_second" in s
        assert "time_breakdown" in s

    def test_reset(self):
        from sped.core.metrics import MetricsCollector

        m = MetricsCollector()
        m.start_step()
        m.end_step(draft_k=5, num_accepted=3, tokens_generated=3)
        assert m.cumulative.total_steps == 1
        m.reset()
        assert m.cumulative.total_steps == 0
        assert m.acceptance_rate == 0.0


# ── KV Cache Manager Tests ───────────────────────────────


class TestKVCacheManager:
    def test_initial_state(self):
        from sped.core.kv_cache import KVCacheManager

        mgr = KVCacheManager(None, max_length=8192, device="cpu")
        assert mgr.base_seq_len == 0
        assert mgr.cache is None
        assert mgr.usage_ratio == 0.0
        assert not mgr.is_full

    def test_rollback_no_cache(self):
        from sped.core.kv_cache import KVCacheManager

        mgr = KVCacheManager(None, max_length=8192, device="cpu")
        mgr.rollback()  # Should not raise


# ── Draft Tree Tests ─────────────────────────────────────


class TestDraftTree:
    def test_single_node(self):
        from sped.core.draft_tree import TreeNode

        node = TreeNode(token_id=42, confidence=1.0, depth=0)
        assert node.token_id == 42
        assert node.confidence == 1.0
        assert node.depth == 0
        assert node.parent is None
        assert node.children == []

    def test_tree_depth(self):
        from sped.core.draft_tree import TreeNode

        root = TreeNode(token_id=0, confidence=1.0, depth=0)
        child = TreeNode(token_id=1, confidence=0.8, depth=1, parent=root)
        root.children.append(child)
        grandchild = TreeNode(token_id=2, confidence=0.6, depth=2, parent=child)
        child.children.append(grandchild)

        assert root.children[0] == child
        assert child.children[0] == grandchild
        assert grandchild.parent == child


# ── Import tests (modules load correctly) ─────────────────


class TestCoreImports:
    def test_import_speculative_decoder(self):
        from sped.core import SpeculativeDecoder
        assert SpeculativeDecoder is not None

    def test_import_verifier(self):
        from sped.core import Verifier
        assert Verifier is not None

    def test_import_rejection_sample(self):
        from sped.core import rejection_sample
        assert callable(rejection_sample)

    def test_import_kv_cache(self):
        from sped.core import KVCacheManager
        assert KVCacheManager is not None

    def test_import_metrics(self):
        from sped.core import MetricsCollector, CumulativeMetrics, StepMetrics
        assert MetricsCollector is not None

    def test_import_draft_tree(self):
        from sped.core import DraftTree, TreeNode
        assert DraftTree is not None
        assert TreeNode is not None


# ── Cross-vocab end-to-end tests ─────────────────────────


class TestCrossVocabSpeculation:
    """Test SpeculativeDecoder with a VocabAligner (cross-vocab path).

    Uses tiny random models so it runs fast. Validates that the
    cross-vocab speculate loop completes without hanging.
    """

    @pytest.fixture
    def models(self):
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer,
        )
        model_id = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.eval()
        return model, tok

    def test_cross_vocab_generates_successfully(self, models):
        """SpeculativeDecoder with VocabAligner should generate text
        without hanging or crashing (regression test for PR #43)."""
        from sped.core.speculative_decoding import SpeculativeDecoder
        from sped.vocab_agnostic.alignment import VocabAligner

        model, tok = models
        # Same model for both draft and target (same vocab, so VocabAligner
        # just passes through — but it exercises the cross-vocab code path)
        aligner = VocabAligner(
            target_tokenizer=tok,
            draft_tokenizer=tok,
            strategy="string",
            target_model=model,
        )
        decoder = SpeculativeDecoder(
            target_model=model, target_tokenizer=tok,
            draft_model=model, draft_tokenizer=tok,
            vocab_aligner=aligner,
            max_draft_tokens=3, device="cpu",
        )

        result = decoder.generate("Hello", max_new_tokens=5, temperature=0.0)
        assert isinstance(result, str)
        assert len(result) > 0
        metrics = decoder.get_metrics()
        assert metrics["total_steps"] > 0

    def test_cross_vocab_hybrid_strategy(self, models):
        """Hybrid alignment should also work through the decoder."""
        from sped.core.speculative_decoding import SpeculativeDecoder
        from sped.vocab_agnostic.alignment import VocabAligner

        model, tok = models
        aligner = VocabAligner(
            target_tokenizer=tok,
            draft_tokenizer=tok,
            strategy="hybrid",
            target_model=model,
        )
        decoder = SpeculativeDecoder(
            target_model=model, target_tokenizer=tok,
            draft_model=model, draft_tokenizer=tok,
            vocab_aligner=aligner,
            max_draft_tokens=2, device="cpu",
        )

        result = decoder.generate("test", max_new_tokens=3, temperature=0.0)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_cross_vocab_probabilistic_strategy(self, models):
        """Probabilistic alignment with target model scoring."""
        from sped.core.speculative_decoding import SpeculativeDecoder
        from sped.vocab_agnostic.alignment import VocabAligner

        model, tok = models
        aligner = VocabAligner(
            target_tokenizer=tok,
            draft_tokenizer=tok,
            strategy="probabilistic",
            target_model=model,
        )
        decoder = SpeculativeDecoder(
            target_model=model, target_tokenizer=tok,
            draft_model=model, draft_tokenizer=tok,
            vocab_aligner=aligner,
            max_draft_tokens=2, device="cpu",
        )

        result = decoder.generate("test", max_new_tokens=3, temperature=0.0)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_cross_vocab_aligner_fallback(self, models):
        """When align() raises an exception, decoder should fall back
        gracefully instead of crashing."""
        from sped.core.speculative_decoding import SpeculativeDecoder
        from sped.vocab_agnostic.alignment import VocabAligner

        model, tok = models
        # Create an aligner that will fail on align()
        class _FailingAligner(VocabAligner):
            def align(self, *args, **kwargs):
                raise RuntimeError("Intentional align failure")

        fail_aligner = _FailingAligner(
            target_tokenizer=tok,
            draft_tokenizer=tok,
            strategy="string",
            target_model=model,
        )
        decoder = SpeculativeDecoder(
            target_model=model, target_tokenizer=tok,
            draft_model=model, draft_tokenizer=tok,
            vocab_aligner=fail_aligner,
            max_draft_tokens=2, device="cpu",
        )

        result = decoder.generate("test", max_new_tokens=3, temperature=0.0)
        assert isinstance(result, str)
        assert len(result) > 0
