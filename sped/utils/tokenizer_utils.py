"""Tokenizer compatibility utilities."""

from typing import Tuple
from transformers import PreTrainedTokenizer


def check_vocab_compatibility(
    tokenizer_a: PreTrainedTokenizer,
    tokenizer_b: PreTrainedTokenizer,
) -> Tuple[bool, float]:
    """Check if two tokenizers share the same vocabulary.

    Returns:
        (compatible: bool, overlap_ratio: float)
    """
    vocab_a = set(tokenizer_a.vocab.keys())
    vocab_b = set(tokenizer_b.vocab.keys())
    overlap = len(vocab_a & vocab_b)
    total = len(vocab_a | vocab_b)
    ratio = overlap / total if total > 0 else 0.0
    return ratio > 0.95, ratio


def estimate_alignment_overhead(
    draft_tokenizer: PreTrainedTokenizer,
    target_tokenizer: PreTrainedTokenizer,
) -> float:
    """Estimate how many target tokens a draft token expands to on average.

    Useful for tuning K (draft length) when vocabularies differ.
    """
    from datasets import load_dataset
    # Quick heuristic: tokenize a few sentences and compare lengths
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Speculative decoding accelerates large language model inference.",
        "Attention is all you need.",
    ]
    ratios = []
    for sent in test_sentences:
        draft_len = len(draft_tokenizer.encode(sent))
        target_len = len(target_tokenizer.encode(sent))
        ratios.append(target_len / max(draft_len, 1))
    return sum(ratios) / len(ratios)
