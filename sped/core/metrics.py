"""Metrics collector for speculative decoding statistics.

Tracks acceptance rate, tokens per step, throughput, and timing breakdowns
in a thread-safe manner. Supports JSON serialization and Rich table output.
"""

from time import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class StepMetrics:
    """Per-step metrics snapshot."""

    draft_k: int = 0
    num_accepted: int = 0
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    align_time_ms: float = 0.0
    sampling_time_ms: float = 0.0
    total_time_ms: float = 0.0
    tokens_generated: int = 0


@dataclass
class CumulativeMetrics:
    """Aggregated metrics across all steps."""

    total_steps: int = 0
    total_draft_tokens: int = 0
    total_accepted: int = 0
    total_tokens_generated: int = 0
    total_draft_time_ms: float = 0.0
    total_verify_time_ms: float = 0.0
    total_align_time_ms: float = 0.0
    total_sampling_time_ms: float = 0.0
    total_generation_time_ms: float = 0.0

    # Rolling window for live stats (last N steps)
    _recent_steps: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposed draft tokens that were accepted."""
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted / self.total_draft_tokens

    @property
    def avg_tokens_per_step(self) -> float:
        """Average number of generated tokens per target forward pass."""
        if self.total_steps == 0:
            return 0.0
        return self.total_tokens_generated / self.total_steps

    @property
    def avg_tokens_per_second(self) -> float:
        """Generation throughput."""
        if self.total_generation_time_ms == 0:
            return 0.0
        return (self.total_tokens_generated / self.total_generation_time_ms) * 1000

    @property
    def speedup_vs_vanilla(self) -> Optional[float]:
        """Estimated speedup vs standard autoregressive decoding.

        Standard AR would use 1 forward pass per token.
        Speculative uses 1 forward pass per N tokens (on average).
        """
        if self.avg_tokens_per_step == 0:
            return None
        return self.avg_tokens_per_step

    def record_step(self, metrics: StepMetrics):
        """Record a single speculation step."""
        self.total_steps += 1
        self.total_draft_tokens += metrics.draft_k
        self.total_accepted += metrics.num_accepted
        self.total_tokens_generated += metrics.tokens_generated
        self.total_draft_time_ms += metrics.draft_time_ms
        self.total_verify_time_ms += metrics.verify_time_ms
        self.total_align_time_ms += metrics.align_time_ms
        self.total_sampling_time_ms += metrics.sampling_time_ms
        self.total_generation_time_ms += metrics.total_time_ms
        self._recent_steps.append(metrics)

    @property
    def recent_acceptance_rate(self) -> float:
        """Running acceptance rate over the last N steps."""
        if not self._recent_steps:
            return 0.0
        total_draft = sum(s.draft_k for s in self._recent_steps)
        total_acc = sum(s.num_accepted for s in self._recent_steps)
        if total_draft == 0:
            return 0.0
        return total_acc / total_draft

    def summary(self) -> dict:
        """Return a JSON-serializable summary dict."""
        return {
            "total_steps": self.total_steps,
            "total_tokens_generated": self.total_tokens_generated,
            "acceptance_rate": round(self.acceptance_rate, 3),
            "avg_tokens_per_step": round(self.avg_tokens_per_step, 2),
            "avg_tokens_per_second": round(self.avg_tokens_per_second, 1),
            "speedup_vs_vanilla": round(self.speedup_vs_vanilla, 2)
            if self.speedup_vs_vanilla is not None
            else None,
            "total_time_seconds": round(self.total_generation_time_ms / 1000, 2),
            "time_breakdown": {
                "draft_pct": round(self.total_draft_time_ms / max(self.total_generation_time_ms, 1) * 100, 1),
                "verify_pct": round(self.total_verify_time_ms / max(self.total_generation_time_ms, 1) * 100, 1),
                "align_pct": round(self.total_align_time_ms / max(self.total_generation_time_ms, 1) * 100, 1),
                "sampling_pct": round(self.total_sampling_time_ms / max(self.total_generation_time_ms, 1) * 100, 1),
            },
            "recent_acceptance_rate": round(self.recent_acceptance_rate, 3),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.summary(), indent=indent)

    def reset(self):
        """Reset all metrics."""
        self.total_steps = 0
        self.total_draft_tokens = 0
        self.total_accepted = 0
        self.total_tokens_generated = 0
        self.total_draft_time_ms = 0.0
        self.total_verify_time_ms = 0.0
        self.total_align_time_ms = 0.0
        self.total_sampling_time_ms = 0.0
        self.total_generation_time_ms = 0.0
        self._recent_steps.clear()


class MetricsCollector:
    """Collects and aggregates speculation statistics.

    Usage:
        collector = MetricsCollector()
        collector.start_step()
        # ... do speculation work ...
        collector.end_step(draft_k=5, num_accepted=3)
        print(collector.summary())
    """

    def __init__(self):
        self.cumulative = CumulativeMetrics()
        self._step_start: Optional[float] = None
        self._current_step = StepMetrics()
        self._phase_start: Optional[float] = None

    def start_step(self):
        """Mark the beginning of a speculation step."""
        self._step_start = time()
        self._current_step = StepMetrics()

    def _phase(self, phase_name: str):
        """Record time for a sub-phase within the current step."""
        now = time()
        if self._phase_start is not None:
            elapsed = (now - self._phase_start) * 1000  # ms
            setattr(self._current_step, f"{phase_name}_ms",
                    getattr(self._current_step, f"{phase_name}_ms", 0.0) + elapsed)

        # Map friendly names to the attribute name
        phase_map = {
            "draft": "draft_time_ms",
            "verify": "verify_time_ms",
            "align": "align_time_ms",
            "sampling": "sampling_time_ms",
            "total": "total_time_ms",
        }
        attr = phase_map.get(phase_name)
        if attr is not None and self._phase_start is not None:
            elapsed = (now - self._phase_start) * 1000
            setattr(self._current_step, attr,
                    getattr(self._current_step, attr, 0.0) + elapsed)
        self._phase_start = now

    def end_step(self, draft_k: int, num_accepted: int, tokens_generated: int):
        """Record the completed step."""
        self._phase("total")
        self._current_step.draft_k = draft_k
        self._current_step.num_accepted = num_accepted
        self._current_step.tokens_generated = tokens_generated
        self.cumulative.record_step(self._current_step)
        self._step_start = None
        self._phase_start = None

    @property
    def acceptance_rate(self) -> float:
        return self.cumulative.acceptance_rate

    @property
    def tokens_per_second(self) -> float:
        return self.cumulative.avg_tokens_per_second

    @property
    def tokens_per_step(self) -> float:
        return self.cumulative.avg_tokens_per_step

    def summary(self) -> dict:
        return self.cumulative.summary()

    def to_json(self) -> str:
        return self.cumulative.to_json()

    def reset(self):
        self.cumulative.reset()
        self._step_start = None
        self._current_step = StepMetrics()
        self._phase_start = None
