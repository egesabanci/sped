"""Tests for online adaptation module (Phase 5)."""

import torch
import pytest


class TinyDraftModel(torch.nn.Module):
    """Minimal LM-like model: embedding → linear head."""
    def __init__(self, vocab_size=100, hidden=16):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.head = torch.nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return type("Out", (), {"logits": self.head(self.embed(x))})()


def _make_adapter(**kwargs):
    """Create an OnlineAdapter with a TinyDraftModel, disabling auto-updates."""
    from sped.adaptation.osd import OnlineAdapter
    opts = dict(update_freq=99999, device="cpu")
    opts.update(kwargs)
    return OnlineAdapter(draft_model=TinyDraftModel(), **opts)


# ── Init Tests ───────────────────────────────────────────


class TestInit:
    def test_defaults(self):
        a = _make_adapter()
        assert a.buffer_size == 256
        assert a.step_count == 0
        assert len(a.buffer) == 0

    def test_custom_params(self):
        a = _make_adapter(lr=5e-6, buffer_size=64, update_freq=8,
                          lr_decay=0.995, kl_reg_weight=0.05)
        assert a.lr == 5e-6
        assert a.buffer_size == 64
        assert a.kl_reg_weight == 0.05

    def test_optimizer_created(self):
        a = _make_adapter()
        assert a.optimizer is not None


# ── Buffer Tests ─────────────────────────────────────────


class TestBuffer:
    def test_observe_appends(self):
        a = _make_adapter()
        a.observe(torch.randint(0, 100, (1, 5)), torch.randint(0, 100, (1, 3)))
        assert len(a.buffer) == 1
        assert a.step_count == 1

    def test_maxlen(self):
        a = _make_adapter(buffer_size=5)
        for _ in range(10):
            a.observe(torch.randint(0, 100, (1, 5)), torch.randint(0, 100, (1, 3)),
                      acceptance_rate=0.5)
        assert len(a.buffer) == 5
        assert a.step_count == 10

    def test_empty_accepted_skips(self):
        a = _make_adapter()
        a.observe(torch.randint(0, 100, (1, 5)), torch.empty(1, 0, dtype=torch.long))
        assert len(a.buffer) == 0
        assert a.step_count == 1

    def test_reservoir_capped(self):
        a = _make_adapter(reservoir_buffer_size=32)
        for _ in range(50):
            a.observe(torch.randint(0, 100, (1, 5)), torch.randint(0, 100, (1, 3)))
        assert len(a.reservoir_buffer) <= 32


# ── Decay Tests ──────────────────────────────────────────


class TestDecay:
    def test_lr_decays(self):
        a = _make_adapter(lr=1e-4, lr_decay=0.5)
        v0 = a._current_lr_value()
        a.update_count = 2
        v2 = a._current_lr_value()
        assert v0 == 1e-4
        assert v2 < v0

    def test_update_freq_decays_and_capped(self):
        a = _make_adapter(update_freq=10, max_update_freq=25)
        a.update_count = 100
        assert a._current_update_freq() <= 25


# ── Rollback Tests (#18) ─────────────────────────────────


class TestRollback:
    def test_not_enough_data(self):
        a = _make_adapter(rollback_window=50)
        for _ in range(10):
            a.recent_acceptance.append(0.5)
        assert not a._should_rollback()

    def test_not_when_improving(self):
        a = _make_adapter(rollback_window=10, rollback_threshold=0.1)
        for rate in [0.3, 0.4, 0.5, 0.6, 0.7]:
            a.recent_acceptance.append(rate)
            if rate > a._best_acceptance:
                a._best_acceptance = rate
        assert not a._should_rollback()

    def test_triggers_on_drop(self):
        a = _make_adapter(rollback_window=10, rollback_threshold=0.2)
        a._best_acceptance = 0.8
        a.step_count = 100
        a._last_rollback_step = 0
        for _ in range(10):
            a.recent_acceptance.append(0.3)
        assert a._should_rollback()

    def test_not_too_frequent(self):
        a = _make_adapter(rollback_window=10)
        a.step_count = 20
        a._last_rollback_step = 15
        for _ in range(10):
            a.recent_acceptance.append(0.1)
        assert not a._should_rollback()

    def test_rollback_clears_buffer(self):
        a = _make_adapter()
        for _ in range(5):
            a.observe(torch.randint(0, 100, (1, 5)), torch.randint(0, 100, (1, 3)),
                      acceptance_rate=0.5)
        a._best_state = {k: v.clone() for k, v in a.draft_model.state_dict().items()}
        a._rollback()
        assert len(a.buffer) == 0


# ── KL Regularization Tests (#18) ────────────────────────


class TestKL:
    def test_same_model_zero(self):
        a = _make_adapter()
        a._save_initial_state()
        kl = a._compute_kl_regularization()
        assert kl.item() < 1e-4

    def test_grows_with_divergence(self):
        a = _make_adapter()
        a._save_initial_state()
        ctx = torch.randint(0, 100, (1, 5))
        acc = torch.randint(0, 100, (1, 3))
        a.observe(ctx, acc)
        before = a._compute_kl_regularization().item()
        for p in a.draft_model.parameters():
            p.data += 0.1 * torch.randn_like(p)
        after = a._compute_kl_regularization().item()
        assert after >= before


# ── Warm-Start Tests (#17) ───────────────────────────────


class TestWarmStart:
    def test_save_initial_state(self):
        a = _make_adapter()
        assert a._initial_state is None
        a._save_initial_state()
        assert a._initial_state is not None

    def test_no_checkpoint(self):
        a = _make_adapter()
        a.warm_start()
        assert a._initial_state is not None
        assert len(a.buffer) == 0

    def test_with_prefill(self):
        a = _make_adapter(buffer_size=16)
        prefill = [(torch.randint(0, 100, (1, 5)), torch.randint(0, 100, (1, 3)))
                   for _ in range(5)]
        a.warm_start(prefill_prompts=prefill)
        assert len(a.buffer) == 5
        assert a._best_state is not None


# ── State Persistence Tests ──────────────────────────────


class TestState:
    def test_state_dict_roundtrip(self):
        a = _make_adapter()
        a.observe(torch.randint(0, 100, (1, 5)), torch.randint(0, 100, (1, 3)))
        state = a.state_dict()
        assert state["step_count"] == 1
        assert len(state["buffer"]) == 1

    def test_load_state_dict(self):
        a1 = _make_adapter()
        a1.observe(torch.randint(0, 100, (1, 5)), torch.randint(0, 100, (1, 3)))
        state = a1.state_dict()

        a2 = _make_adapter()
        assert a2.step_count == 0
        a2.load_state_dict(state)
        assert a2.step_count == 1
        assert len(a2.buffer) == 1

    def test_summary_keys(self):
        a = _make_adapter()
        s = a.summary()
        for k in ("total_steps", "total_updates", "buffer_size", "current_lr", "update_freq"):
            assert k in s

    def test_acceptance_rate_property(self):
        a = _make_adapter()
        assert a.current_acceptance_rate == 0.0
        a.recent_acceptance.extend([0.5, 0.6, 0.7])
        assert abs(a.current_acceptance_rate - 0.6) < 1e-6


# ── Import Tests ─────────────────────────────────────────


class TestImports:
    def test_import_adapter(self):
        from sped.adaptation import OnlineAdapter
        assert OnlineAdapter is not None
