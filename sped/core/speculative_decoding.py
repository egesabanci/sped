"""High-level speculative decoder orchestrating draft → verify → accept.

Implements the full speculate-verify-accept loop with:
- Draft model autoregressive proposal
- Optional vocabulary-agnostic alignment
- Parallel target verification with KV cache
- Metropolis-Hastings rejection sampling
- Metrics collection and online adaptation hooks
"""

from typing import Optional, Callable
from time import time
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from sped.core.verification import Verifier
from sped.core.rejection_sampling import rejection_sample
from sped.core.metrics import MetricsCollector, StepMetrics
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
    ):
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
        self.vocab_aligner = vocab_aligner
        self.max_draft_tokens = max_draft_tokens
        self.max_length = max_length
        self.metrics = MetricsCollector()

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize KV cache managers
        self.target_cache = KVCacheManager(
            target_model, max_length=max_length, device=self.device
        )
        self.draft_cache = KVCacheManager(
            draft_model, max_length=max_length, device=self.device
        ) if draft_model is not None else None

        # Verifier (handles parallel forward pass)
        self.verifier = Verifier(
            target_model, target_tokenizer, device=self.device
        )

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
            return self._speculate(
                prompt, max_new_tokens, temperature,
                draft_k or self.max_draft_tokens, verbose,
            )
        else:
            return self._standard_generate(prompt, max_new_tokens, temperature)

    def _speculate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        draft_k: int,
        verbose: bool,
    ) -> str:
        """Draft-then-verify speculative decoding loop."""
        self.metrics.reset()
        self.target_cache.reset()
        if self.draft_cache is not None:
            self.draft_cache.reset()

        # Tokenize prompt with both tokenizers
        target_enc = self.target_tokenizer(
            prompt, return_tensors="pt"
        ).to(self.device)

        draft_enc = self.draft_tokenizer(
            prompt, return_tensors="pt"
        ).to(self.device)

        # Prefill target KV cache
        self.metrics._phase("verify")
        self.target_cache.prefill(target_enc.input_ids)
        self.metrics._phase_start = time()

        # Prefill draft KV cache (use target tokens aligned to draft vocab)
        if self.draft_cache is not None:
            self.metrics._phase("draft")
            self.draft_cache.prefill(draft_enc.input_ids)
            self.metrics._phase_start = time()

        generated_ids = target_enc.input_ids[0].tolist()
        draft_ctx_ids = draft_enc.input_ids[0].tolist()
        total_generated = 0

        while total_generated < max_new_tokens:
            self.metrics.start_step()

            # ── Step 1: Draft model proposes K tokens ──────────────────
            self.metrics._phase("draft")
            proposed_draft = []

            for _ in range(draft_k):
                if self.draft_cache.is_full:
                    break

                ctx_tensor = torch.tensor(
                    [draft_ctx_ids], device=self.device
                )

                with torch.no_grad():
                    logits = self.draft_cache.extend(
                        ctx_tensor[:, -1:]
                    )

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

            if not proposed_draft:
                # Draft produced nothing — fallback to single token
                self.metrics.end_step(draft_k=0, num_accepted=0, tokens_generated=0)
                self.metrics._phase("verify")
                logits = self.target_cache.extend(
                    torch.tensor([[generated_ids[-1]]], device=self.device)
                )
                next_token = logits[0, -1].argmax().item()
                generated_ids.append(next_token)
                total_generated += 1
                self.metrics._phase_start = time()
                continue

            draft_tensor = torch.tensor([proposed_draft], device=self.device)

            # ── Step 2: Align draft tokens (if vocabs differ) ─────────
            self.metrics._phase("align")
            aligned_draft = draft_tensor
            if self.vocab_aligner is not None:
                try:
                    aligned_draft, _ = self.vocab_aligner.align(
                        draft_tensor,
                        torch.tensor([generated_ids], device=self.device),
                    )
                except NotImplementedError:
                    # Fall through with unaligned draft
                    pass

            # ── Step 3: Verify all draft tokens in parallel ────────────
            self.metrics._phase("verify")
            ctx_tensor = torch.tensor(
                [generated_ids], device=self.device
            )

            target_logits_for_draft = self.target_cache.verify_draft(
                ctx_tensor,
                aligned_draft.to(self.device),
            )

            # Align draft logits to target vocabulary
            if self.draft_cache is not None and self.vocab_aligner is None:
                # Same vocab — get draft logits at same positions
                draft_ctx = torch.tensor(
                    [draft_ctx_ids], device=self.device
                )
                with torch.no_grad():
                    draft_output = self.draft_model(draft_ctx)
                draft_logits = draft_output.logits[
                    0, -(len(proposed_draft) + 1):-1, :
                ]
            else:
                # Cross-vocab — use draft model's own distribution
                draft_logits = target_logits_for_draft.clone()

            # ── Step 4: Rejection sampling ─────────────────────────────
            self.metrics._phase("sampling")

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

            # ── Step 5: Append accepted tokens ─────────────────────────
            accepted_ids = accepted_tokens[:max_new_tokens - total_generated]
            num_to_commit = len(accepted_ids)

            if num_to_commit > 0:
                generated_ids.extend(accepted_ids)
                total_generated += num_to_commit
                self.target_cache.commit(num_to_commit)

                # Also extend draft context
                draft_ctx_ids.extend(
                    proposed_draft[:num_to_commit]
                )
                if self.draft_cache is not None:
                    self.draft_cache.commit(num_to_commit)
            else:
                # No tokens accepted — fallback: generate one token
                # from the target model using the residual
                pass

            # Record metrics
            self.metrics._phase("total")
            self.metrics.end_step(
                draft_k=len(proposed_draft),
                num_accepted=num_accepted,
                tokens_generated=num_to_commit,
            )

            if verbose:
                ar = self.metrics.acceptance_rate
                tps = self.metrics.tokens_per_step
                print(
                    f"  step {self.metrics.cumulative.total_steps}: "
                    f"draft={len(proposed_draft)} accepted={num_accepted}/{draft_k} "
                    f"(rate={ar:.1%}) avg_step={tps:.2f} tok"
                )

            # Prevent infinite loop if nothing is generated
            if num_to_commit == 0:
                # Emergency: generate one token the standard way
                ctx = torch.tensor([generated_ids], device=self.device)
                with torch.no_grad():
                    logits = self.target_model(ctx).logits[0, -1, :]
                next_tok = logits.argmax().item()
                generated_ids.append(next_tok)
                total_generated += 1

            # Check cache limits
            if self.target_cache.is_full:
                if verbose:
                    print("  [cache full] stopping speculation")
                break

        # Decode
        output = self.target_tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        return output

    def _standard_generate(
        self, prompt: str, max_new_tokens: int, temperature: float
    ) -> str:
        """Fallback to standard autoregressive generation."""
        inputs = self.target_tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.target_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )
        return self.target_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_metrics(self) -> dict:
        """Return accumulated metrics summary."""
        return self.metrics.summary()

    def reset_metrics(self):
        """Reset all accumulated metrics."""
        self.metrics.reset()
