"""PEFT-based distillation pipeline (DistillSpec approach).

Aligns a small draft model to a large target model using on-policy
KL divergence distillation with LoRA (PEFT). Enables 1-GPU training
of draft models for multi-billion-parameter targets.
"""

from .distillspec import DistillSpec

__all__ = ["DistillSpec"]
