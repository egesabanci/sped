"""Vocabulary-agnostic speculative decoding algorithms.

Implements the Intel/Weizmann heterogeneous vocabulary approach
for pairing draft and target models with different tokenizers.
"""

from .alignment import VocabAligner
from .heterogeneous import HeterogeneousDecoder

__all__ = ["VocabAligner", "HeterogeneousDecoder"]
