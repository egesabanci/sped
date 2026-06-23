"""Online Speculative Decoding (OSD) — online adaptation of draft model.

Maintains a replay buffer of recent (context, accepted_tokens) examples
and performs lightweight online updates to keep the draft aligned with
the target model's distribution during inference.

Features:
- Warm-start from offline DistillSpec checkpoint (#17)
- Anti-thrash guardrails: LR decay, buffer diversity, rollback, KL reg (#18)
- Reservoir sampling for fair long-term buffer
- Acceptance-rate monitoring with automatic rollback
"""

from pathlib import Path
from typing import Optional
import torch
from collections import deque
from copy import deepcopy
import random


class OnlineAdapter:
    """Online adaptation of draft model during inference.

    Monitors speculation outcomes and applies lightweight gradient updates
    to improve draft quality over time. All guardrails are enabled by
    default to prevent distribution drift.

    Typical integration:
        adapter = OnlineAdapter(draft_model, lr=1e-5)
        adapter.warm_start("./draft-lora")           # #17
        # ... during inference loop ...
        adapter.observe(context_ids, accepted_ids)   # auto-updates
    """

    def __init__(
        self,
        draft_model,
        lr: float = 1e-5,
        buffer_size: int = 256,
        update_freq: int = 16,
        min_update_samples: int = 8,
        device: str = "cuda",
        lr_decay: float = 0.999,
        update_freq_decay: float = 1.0,
        max_update_freq: int = 128,
        grad_clip_norm: float = 1.0,
        kl_reg_weight: float = 0.01,
        rollback_threshold: float = 0.15,
        rollback_window: int = 50,
        reservoir_buffer_size: int = 512,
    ):
        self.draft_model = draft_model
        self.lr = lr
        self.buffer_size = buffer_size
        self.update_freq = update_freq
        self.min_update_samples = min_update_samples
        self.device = device
        self.lr_decay = lr_decay
        self.update_freq_decay = update_freq_decay
        self.max_update_freq = max_update_freq
        self.grad_clip_norm = grad_clip_norm
        self.kl_reg_weight = kl_reg_weight
        self.rollback_threshold = rollback_threshold
        self.rollback_window = rollback_window
        self.reservoir_buffer_size = reservoir_buffer_size

        # Core buffers
        self.buffer = deque(maxlen=buffer_size)
        self.reservoir_buffer: list[tuple] = []  # long-term diverse buffer

        # Tracking
        self.step_count = 0
        self.update_count = 0
        self.recent_acceptance: deque = deque(maxlen=rollback_window)

        # Saved initial weights for KL regularization and rollback
        self._initial_state: Optional[dict] = None
        self._best_state: Optional[dict] = None
        self._best_acceptance: float = 0.0
        self._last_rollback_step: int = 0

        # Optimizer — targets only trainable (LoRA) parameters
        trainable = [p for p in draft_model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=lr)

        # Current LR (decays over time)
        self._current_lr = lr

    # ── Phase 1: Warm-start from DistillSpec Checkpoint (#17) ───────────

    def warm_start(
        self,
        lora_checkpoint_path: Optional[Path] = None,
        prefill_prompts: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        """Initialize adapter from a DistillSpec checkpoint and/or pre-fill buffer.

        Steps:
        1. Save initial model state for KL regularization
        2. Load LoRA adapter if provided
        3. Pre-fill replay buffer with high-quality examples
        4. Store the initial (pre-online) state as anchor for KL reg

        Args:
            lora_checkpoint_path: Path to saved LoRA adapter from DistillSpec.
            prefill_prompts: List of (context_ids, accepted_ids) pairs from
                             distillation validation data.
        """
        # 1. Save initial state (before any loading) as anchor for KL reg
        self._save_initial_state()

        # 2. Load LoRA adapter
        if lora_checkpoint_path is not None:
            path = Path(lora_checkpoint_path)
            if path.exists() and (path / "adapter_config.json").exists():
                from peft import PeftModel
                # Merge existing LoRA + load new adapter
                if hasattr(self.draft_model, "load_adapter"):
                    self.draft_model.load_adapter(str(path))
                else:
                    # Fallback: reload entire model with adapter
                    base = self.draft_model
                    self.draft_model = PeftModel.from_pretrained(
                        base, str(path)
                    )
                print(f"[sped] Warm-started from LoRA adapter: {path}")
            else:
                print(f"[sped] Warning: adapter not found at {path}")

        # 3. Pre-fill replay buffer
        if prefill_prompts:
            for ctx, acc in prefill_prompts:
                self.buffer.append((ctx.cpu(), acc.cpu()))
            print(f"[sped] Pre-filled buffer with {len(prefill_prompts)} examples")

        # 4. Save post-warm state as best initial state
        self._best_state = deepcopy(self.draft_model.state_dict())
        print(f"[sped] OnlineAdapter ready — buffer has {len(self.buffer)}/{self.buffer_size} samples")

    def _save_initial_state(self):
        """Save current model state for KL regularization and rollback."""
        self._initial_state = deepcopy(self.draft_model.state_dict())
        if self._best_state is None:
            self._best_state = deepcopy(self.draft_model.state_dict())

    # ── Phase 2: Core Observation & Update Loop ─────────────────────────

    def observe(
        self,
        context_ids: torch.Tensor,
        accepted_ids: torch.Tensor,
        acceptance_rate: Optional[float] = None,
    ):
        """Store a speculation outcome and trigger online update if due.

        Args:
            context_ids: (1, seq_len) — prefix context before draft.
            accepted_ids: (1, n_accepted) — tokens that were accepted.
            acceptance_rate: Optional per-step acceptance rate for monitoring.
        """
        if accepted_ids.shape[-1] > 0:
            self.buffer.append((context_ids.cpu(), accepted_ids.cpu()))
            self._add_to_reservoir(context_ids, accepted_ids)

        self.step_count += 1

        # Track acceptance rate
        if acceptance_rate is not None:
            self.recent_acceptance.append(acceptance_rate)

        # Check if we need to rollback (acceptance rate dropped)
        if self._should_rollback():
            self._rollback()

        # Trigger online update
        if self.step_count % self._current_update_freq() == 0:
            self._online_update()

    def _add_to_reservoir(self, context_ids, accepted_ids):
        """Add sample to reservoir buffer with equal probability.

        Reservoir sampling ensures the long-term buffer fairly represents
        all past observations, preventing recency bias.
        """
        if len(self.reservoir_buffer) < self.reservoir_buffer_size:
            self.reservoir_buffer.append((context_ids.cpu(), accepted_ids.cpu()))
        else:
            # Random replacement
            idx = random.randint(0, self.step_count - 1)
            if idx < self.reservoir_buffer_size:
                self.reservoir_buffer[idx] = (context_ids.cpu(), accepted_ids.cpu())

    def _current_update_freq(self) -> int:
        """Decay update frequency over time (fewer updates as model stabilizes)."""
        freq = int(self.update_freq * (self.update_freq_decay ** self.update_count))
        return min(max(freq, 1), self.max_update_freq)

    def _current_lr_value(self) -> float:
        """Decayed learning rate."""
        return self.lr * (self.lr_decay ** self.update_count)

    # ── Online Update ───────────────────────────────────────────────────

    def _online_update(self):
        """Perform a lightweight gradient step on the replay buffer.

        Combines:
        - Cross-entropy loss on recent buffer (immediate adaptation)
        - Reservoir buffer (long-term diversity)
        - KL regularization against initial weights (anti-drift)
        """
        if len(self.buffer) < self.min_update_samples:
            return

        self.update_count += 1
        effective_lr = self._current_lr_value()

        # Update optimizer LR
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = effective_lr

        # Sample from recent buffer
        recent_batch = list(self.buffer)[-min(32, len(self.buffer)):]

        # Also sample from reservoir for diversity
        reservoir_batch = []
        if len(self.reservoir_buffer) >= self.min_update_samples:
            reservoir_batch = random.sample(
                self.reservoir_buffer,
                min(16, len(self.reservoir_buffer)),
            )

        batch = recent_batch + reservoir_batch
        total_loss = 0.0
        nll_loss_sum = 0.0
        kl_loss_sum = 0.0

        self.draft_model.train()

        for ctx, accepted in batch:
            ctx = ctx.to(self.device)
            accepted = accepted.to(self.device)

            combined = torch.cat([ctx, accepted], dim=-1)
            outputs = self.draft_model(combined)

            # Next-token prediction loss on accepted tokens
            shift_logits = outputs.logits[:, ctx.shape[-1] - 1 : -1, :]
            shift_labels = accepted

            nll_loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            nll_loss_sum += nll_loss

        avg_nll = nll_loss_sum / len(batch)
        total_loss = avg_nll

        # ── KL regularization against initial weights (#18) ────────────
        if self.kl_reg_weight > 0 and self._initial_state is not None:
            kl_loss = self._compute_kl_regularization()
            kl_loss_sum = kl_loss
            total_loss = avg_nll + self.kl_reg_weight * kl_loss

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping (#18)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.draft_model.parameters() if p.requires_grad],
            self.grad_clip_norm,
        )

        self.optimizer.step()
        self.draft_model.eval()

    def _compute_kl_regularization(self) -> torch.Tensor:
        """Compute KL divergence between current and initial model outputs.

        This penalizes large deviations from the initial (pre-online) weights,
        preventing catastrophic forgetting of the distillation knowledge.
        """
        # Use a tiny sample from the buffer to compute KL
        if len(self.buffer) < 2:
            return torch.tensor(0.0, device=self.device)

        sample_ctx, sample_acc = random.choice(list(self.buffer))
        sample_ctx = sample_ctx.to(self.device)
        sample_acc = sample_acc.to(self.device)
        combined = torch.cat([sample_ctx, sample_acc], dim=-1)

        # Current model logits
        current_outputs = self.draft_model(combined)
        current_logits = current_outputs.logits

        # Initial model logits (run forward on saved state)
        saved_state = self.draft_model.state_dict()
        self.draft_model.load_state_dict(self._initial_state)
        with torch.no_grad():
            initial_outputs = self.draft_model(combined)
            initial_logits = initial_outputs.logits
        self.draft_model.load_state_dict(saved_state)

        # KL divergence (D_KL(current || initial))
        current_probs = torch.log_softmax(current_logits, dim=-1)
        initial_probs = torch.softmax(initial_logits, dim=-1)
        kl = torch.nn.functional.kl_div(
            current_probs, initial_probs,
            reduction="batchmean", log_target=False,
        )
        return kl

    # ── Anti-Thrash: Rollback Mechanism (#18) ───────────────────────────

    def _should_rollback(self) -> bool:
        """Check if acceptance rate has dropped significantly.

        Rollback triggers when the rolling acceptance rate drops by
        more than threshold compared to the best known rate.
        """
        if len(self.recent_acceptance) < self.rollback_window // 2:
            return False

        # Don't rollback too frequently
        if self.step_count - self._last_rollback_step < self.rollback_window:
            return False

        current_rate = sum(self.recent_acceptance) / len(self.recent_acceptance)

        # Update best if improving
        if current_rate > self._best_acceptance:
            self._best_acceptance = current_rate
            self._best_state = deepcopy(self.draft_model.state_dict())
            return False

        # Check for significant drop
        if self._best_acceptance > 0:
            drop = (self._best_acceptance - current_rate) / self._best_acceptance
            return drop > self.rollback_threshold

        return False

    def _rollback(self):
        """Rollback model weights to the best known state."""
        if self._best_state is not None:
            self.draft_model.load_state_dict(self._best_state)
            self._last_rollback_step = self.step_count
            # Re-initialize optimizer
            trainable = [p for p in self.draft_model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(trainable, lr=self.lr)
            # Clear recent buffer
            self.buffer.clear()
            # Restore reservoir
            if hasattr(self, "reservoir_buffer"):
                self.reservoir_buffer = self.reservoir_buffer[-self.reservoir_buffer_size // 2:]

    # ── State Persistence ───────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Serialize adapter state for checkpointing."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "buffer": list(self.buffer),
            "reservoir": self.reservoir_buffer[-self.reservoir_buffer_size:],
            "step_count": self.step_count,
            "update_count": self.update_count,
            "best_acceptance": self._best_acceptance,
            "best_state": self._best_state,
            "initial_state": self._initial_state,
            "recent_acceptance": list(self.recent_acceptance),
        }

    def load_state_dict(self, state: dict):
        """Restore adapter state from checkpoint."""
        self.optimizer.load_state_dict(state["optimizer"])
        self.buffer = deque(state["buffer"], maxlen=self.buffer_size)
        self.reservoir_buffer = list(state.get("reservoir", []))
        self.step_count = state.get("step_count", 0)
        self.update_count = state.get("update_count", 0)
        self._best_acceptance = state.get("best_acceptance", 0.0)
        self._best_state = state.get("best_state")
        self._initial_state = state.get("initial_state")
        self.recent_acceptance = deque(
            state.get("recent_acceptance", []),
            maxlen=self.rollback_window,
        )

    @property
    def current_acceptance_rate(self) -> float:
        """Rolling acceptance rate over recent window."""
        if not self.recent_acceptance:
            return 0.0
        return sum(self.recent_acceptance) / len(self.recent_acceptance)

    @property
    def total_updates(self) -> int:
        return self.update_count

    def summary(self) -> dict:
        """Human-readable status summary."""
        return {
            "total_steps": self.step_count,
            "total_updates": self.update_count,
            "buffer_size": len(self.buffer),
            "reservoir_size": len(self.reservoir_buffer),
            "current_lr": self._current_lr_value(),
            "update_freq": self._current_update_freq(),
            "rollback_window_acceptance": round(self.current_acceptance_rate, 3),
            "best_acceptance": round(self._best_acceptance, 3),
        }
