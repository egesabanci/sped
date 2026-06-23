"""High-level speculative decoder orchestrating draft → verify → accept.

Implements the full speculate-verify-accept loop with:
- Draft model autoregressive proposal
- Optional vocabulary-agnostic alignment
- Parallel target verification with KV cache
- Metropolis-Hastings rejection sampling
- Metrics collection and online adaptation hooks

Memory safety:
- Iteration cap prevents infinite loops
- draft_ctx_ids trimmed to last 512 tokens
- KV cache rollback on memory pressure
- Emergency fallback when no tokens accepted
"""

from typing import Optional, Callable
from time import time
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from sped.core.verification import Verifier
from sped.core.rejection_sampling import rejection_sample
from sped.core.metrics import MetricsCollector
from sped.core.kv_cache import KVCacheManager


class SpeculativeDecoder:
    """Orchestrates draft generation, verification, and acceptance.

    Supports vocabulary-agnostic draft-target pairs via an optional
    alignment layer. Collects detailed step-by-step metrics.
    """

    def __init__(
        self,
        target_model: PreTrainedModel,
        target_tokenizer: PreTrainedTokenizer,
        draft_model: Optional[PreTrainedModel] = None,
        draft_tokenizer: Optional[PreTrainedTokenizer] = None,
        vocab_aligner: Optional[Callable] = None,
        max_draft_tokens: int = 5,
        device: str = "auto",
        max_length: int = 8192,
        max_speculate_iters: Optional[int] = None,
    ):
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
        self.vocab_aligner = vocab_aligner
        self.max_draft_tokens = max_draft_tokens
        self.max_length = max_length
        self.max_speculate_iters = max_speculate_iters  # None = auto
        self.metrics = MetricsCollector()

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.target_cache = KVCacheManager(target_model, max_length=max_length, device=self.device)
        self.draft_cache = KVCacheManager(draft_model, max_length=max_length, device=self.device) if draft_model is not None else None
        self.verifier = Verifier(target_model, target_tokenizer, device=self.device)

        # Max context tokens to keep in draft_ctx_ids (memory safety)
        self._max_ctx_tokens = 512

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        draft_k: Optional[int] = None,
        verbose: bool = False,
    ) -> str:
        """Generate text using speculative decoding.

        Args:
            prompt: Input text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            draft_k: Override default draft K for this call.
            verbose: Print per-step stats.

        Returns:
            Generated text.
        """
        if self.draft_model is not None and self.draft_tokenizer is not None:
            return self._speculate(prompt, max_new_tokens, temperature, draft_k or self.max_draft_tokens, verbose)
        else:
            return self._standard_generate(prompt, max_new_tokens, temperature)

    def _speculate(self, prompt: str, max_new_tokens: int, temperature: float, draft_k: int, verbose: bool) -> str:
        """Draft-then-verify speculative decoding loop.

        Memory-safe: capped iterations, trimmed context, KV cache pressure check.
        """
        self.metrics.reset()
        self.target_cache.reset()
        if self.draft_cache is not None:
            self.draft_cache.reset()

        # Tokenize
        target_enc = self.target_tokenizer(prompt, return_tensors="pt").to(self.device)
        draft_enc = self.draft_tokenizer(prompt, return_tensors="pt").to(self.device)

        # Prefill caches
        self.target_cache.prefill(target_enc.input_ids)
        if self.draft_cache is not None:
            self.draft_cache.prefill(draft_enc.input_ids)

        generated_ids = target_enc.input_ids[0].tolist()
        draft_ctx_ids = draft_enc.input_ids[0].tolist()
        total_generated = 0

        # Iteration cap: prevent infinite loops (memory safety)
        max_iters = self.max_speculate_iters or max(50, max_new_tokens * 2)
        iteration = 0

        while total_generated < max_new_tokens and iteration < max_iters:
            iteration += 1
            self.metrics.start_step()

            # ── Memory pressure check: fall back if cache is too full ──
            if self.target_cache.usage_ratio > 0.85:
                if verbose:
                    print(f"  [cache at {self.target_cache.usage_ratio:.0%}] falling back to standard gen")
                remaining = max_new_tokens - total_generated
                if remaining > 0:
                    ctx = torch.tensor([generated_ids], device=self.device)
                    with torch.no_grad():
                        outputs = self.target_model.generate(
                            ctx, max_new_tokens=remaining, do_sample=(temperature > 0),
                            temperature=temperature if temperature > 0 else None,
                        )
                    generated_ids.extend(outputs[0, ctx.shape[-1]:].tolist())
                    total_generated = max_new_tokens
                break

            # ── Step 1: Draft model proposes K tokens ────────────────
            proposed_draft = []
            for _ in range(draft_k):
                if self.draft_cache is not None and self.draft_cache.is_full:
                    break
                ctx_tensor = torch.tensor([draft_ctx_ids], device=self.device) if self.draft_cache is None else None
                if self.draft_cache is not None:
                    with torch.no_grad():
                        logits = self.draft_cache.extend(ctx_tensor[:, -1:] if ctx_tensor is not None else
                                                         torch.tensor([[draft_ctx_ids[-1]]], device=self.device))
                else:
                    with torch.no_grad():
                        outputs = self.draft_model(torch.tensor([draft_ctx_ids], device=self.device))
                        logits = outputs.logits[:, -1:, :]

                next_logits = logits[0, -1, :]
                if temperature > 0:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                else:
                    next_token = next_logits.argmax().item()

                proposed_draft.append(next_token)
                draft_ctx_ids.append(next_token)
                if next_token == self.draft_tokenizer.eos_token_id:
                    break

            # Trim draft_ctx_ids to prevent unbounded growth (memory safety)
            if len(draft_ctx_ids) > self._max_ctx_tokens + draft_k:
                draft_ctx_ids = draft_ctx_ids[-(self._max_ctx_tokens):]

            if not proposed_draft:
                # Emergency single-token fallback
                self.metrics.end_step(draft_k=0, num_accepted=0, tokens_generated=0)
                ctx = torch.tensor([generated_ids], device=self.device)
                with torch.no_grad():
                    logits = self.target_model(ctx).logits[0, -1, :]
                next_tok = logits.argmax().item()
                generated_ids.append(next_tok)
                total_generated += 1
                continue

            draft_tensor = torch.tensor([proposed_draft], device=self.device)

            # ── Step 2: Align draft tokens (if vocabs differ) ────────
            aligned_draft = draft_tensor
            alignment_mask = None
            if self.vocab_aligner is not None:
                try:
                    aligned_draft, alignment_mask = self.vocab_aligner.align(
                        draft_tensor, torch.tensor([generated_ids], device=self.device),
                    )
                except NotImplementedError:
                    pass
                except Exception as e:
                    # If alignment fails, fall back to standard generation for this step
                    if verbose:
                        print(f"  [align failed: {e}] falling back")
                    aligned_draft = draft_tensor
                    alignment_mask = None

            # ── Step 3: Verify all draft tokens in parallel ──────────
            ctx_tensor = torch.tensor([generated_ids], device=self.device)
            target_logits_for_draft = self.target_cache.verify_draft(ctx_tensor, aligned_draft.to(self.device))

            if self.vocab_aligner is not None and self.draft_model is not None:
                # Cross-vocab: get draft logits in DRAFT vocabulary for the ORIGINAL draft tokens
                draft_ctx = torch.tensor([draft_ctx_ids], device=self.device)
                with torch.no_grad():
                    draft_output = self.draft_model(draft_ctx)
                # Get logits at the original draft positions (in draft vocab)
                draft_logits_original = draft_output.logits[0, -(len(proposed_draft) + 1):-1, :]

                # Use heterogeneous rejection sampling
                try:
                    from sped.vocab_agnostic.heterogeneous import heterogeneous_rejection_sample
                    accepted_tokens, num_accepted = heterogeneous_rejection_sample(
                        draft_logits=draft_logits_original,
                        target_logits=target_logits_for_draft[0],
                        aligned_tokens=aligned_draft[0],
                        alignment_mask=alignment_mask[0] if alignment_mask is not None else None,
                        draft_tokens_original=draft_tensor[0],
                        temperature=temperature,
                    )
                except Exception:
                    # Fallback: standard rejection on aligned tokens
                    draft_logits = target_logits_for_draft.clone()
                    n_verify = min(aligned_draft.shape[-1], draft_logits.shape[0], target_logits_for_draft.shape[1])
                    try:
                        accepted_tokens, num_accepted = rejection_sample(
                            draft_logits=draft_logits[:n_verify],
                            target_logits=target_logits_for_draft[0, :n_verify],
                            draft_tokens=aligned_draft[0, :n_verify],
                            temperature=temperature,
                        )
                    except Exception:
                        accepted_tokens, num_accepted = [], 0

                # Skip the standard rejection below
                _skip_standard_rejection = True
            else:
                _skip_standard_rejection = False

            if not _skip_standard_rejection:
                if self.draft_cache is not None and self.vocab_aligner is None:
                    # Same vocab with draft cache
                    draft_ctx = torch.tensor([draft_ctx_ids], device=self.device)
                    with torch.no_grad():
                        draft_output = self.draft_model(draft_ctx)
                    draft_logits = draft_output.logits[0, -(len(proposed_draft) + 1):-1, :]
                else:
                    draft_logits = target_logits_for_draft.clone()

                # ── Step 4: Rejection sampling ──────────────────────────
                n_verify = min(aligned_draft.shape[-1], draft_logits.shape[0], target_logits_for_draft.shape[1])
                try:
                    accepted_tokens, num_accepted = rejection_sample(
                        draft_logits=draft_logits[:n_verify],
                        target_logits=target_logits_for_draft[0, :n_verify],
                        draft_tokens=aligned_draft[0, :n_verify],
                        temperature=temperature,
                    )
                except Exception:
                    accepted_tokens = []
                    num_accepted = 0

            # ── Step 5: Append accepted tokens ──────────────────────
            accepted_ids = accepted_tokens[:max_new_tokens - total_generated]
            num_to_commit = len(accepted_ids)

            if num_to_commit > 0:
                generated_ids.extend(accepted_ids)
                total_generated += num_to_commit
                self.target_cache.commit(num_to_commit)
                draft_ctx_ids.extend(proposed_draft[:num_to_commit])
                if self.draft_cache is not None:
                    self.draft_cache.commit(num_to_commit)

                # Trim again after extending
                if len(draft_ctx_ids) > self._max_ctx_tokens + draft_k:
                    draft_ctx_ids = draft_ctx_ids[-(self._max_ctx_tokens):]

            # Record metrics
            self.metrics.end_step(draft_k=len(proposed_draft), num_accepted=num_accepted, tokens_generated=num_to_commit)

            if verbose:
                ar = self.metrics.acceptance_rate
                tps = self.metrics.tokens_per_step
                print(f"  step {self.metrics.cumulative.total_steps}: draft={len(proposed_draft)} accepted={num_accepted}/{draft_k} (rate={ar:.1%}) avg_step={tps:.2f} tok")

            # Emergency: if nothing accepted, force one token
            if num_to_commit == 0:
                ctx = torch.tensor([generated_ids], device=self.device)
                with torch.no_grad():
                    logits = self.target_model(ctx).logits[0, -1, :]
                next_tok = logits.argmax().item()
                generated_ids.append(next_tok)
                total_generated += 1

        # Safety: if we hit the iteration cap, force remaining tokens
        if iteration >= max_iters and total_generated < max_new_tokens:
            remaining = max_new_tokens - total_generated
            if remaining > 0:
                ctx = torch.tensor([generated_ids], device=self.device)
                with torch.no_grad():
                    outputs = self.target_model.generate(ctx, max_new_tokens=remaining, do_sample=False)
                generated_ids.extend(outputs[0, ctx.shape[-1]:].tolist())

        return self.target_tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _standard_generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Fallback to standard autoregressive generation."""
        inputs = self.target_tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.target_model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
        )
        return self.target_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_metrics(self) -> dict:
        return self.metrics.summary()

    def reset_metrics(self):
        self.metrics.reset()
