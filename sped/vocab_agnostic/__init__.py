"""Vocabulary-agnostic speculative decoding algorithms.

Implements the Intel/Weizmann heterogeneous vocabulary approach
(ICML 2025, Oral) for pairing draft and target models with different
tokenizers.

Algorithms:
    1. String-level mapping — decode → re-encode via string roundtrip
    2. Probabilistic mapping — score candidates with target model logits
    3. Hybrid — dynamically selects best strategy per token

Also provides heterogeneous rejection sampling that operates correctly
across vocabulary boundaries.
"""

from .alignment import VocabAligner, _HybridStrategySelector
from .heterogeneous import HeterogeneousDecoder, heterogeneous_rejection_sample

__all__ = [
    "VocabAligner",
    "HeterogeneousDecoder",
    "heterogeneous_rejection_sample",
    "_HybridStrategySelector",
]
