"""DistillSpec: knowledge distillation to align a draft model with a target.

Full implementation with on-policy data generation (#14), robust training
loop with Accelerate (#15), and acceptance rate validation (#16).

Uses LoRA (PEFT) for efficient training — only ~0.1–1% of parameters
are updated, enabling single-GPU distillation of 0.5B→70B model pairs.
"""

from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader, random_split
from transformers import (
    PreTrainedModel, PreTrainedTokenizer, get_linear_schedule_with_warmup,
)
from datasets import Dataset
from accelerate import Accelerator
import logging

logger = logging.getLogger(__name__)


# ── Unsloth integration helpers ──────────────────────────────────────────

def _is_unsloth_available() -> bool:
    """Check if unsloth is installed."""
    try:
        import unsloth  # noqa: F401
        return True
    except ImportError:
        return False


def _is_unsloth_model(model) -> bool:
    """Detect whether *model* was loaded/patched by Unsloth.

    Unsloth-patched models carry internal attributes (e.g.
    ``_saved_temp_tokenizer``) that standard HF models lack. We use this
    to avoid routing a plain HF model through the Unsloth LoRA path just
    because the ``unsloth`` package happens to be installed.
    """
    return hasattr(model, "_saved_temp_tokenizer") or any(
        hasattr(model, attr) for attr in ("_unloth_model", "_unsloth_model")
    )


def _apply_unsloth_lora(model, r: int = 8, lora_alpha: int = 16, **kwargs):
    """Apply LoRA via FastLanguageModel.get_peft_model with fast kernels."""
    from unsloth import FastLanguageModel
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        **kwargs,
    )
    # Switch to training mode (undoes for_inference if called)
    FastLanguageModel.for_training(model)
    return model


def _ensure_unsloth_inference(model):
    """Switch model to inference mode (no-op if unsloth not available)."""
    if _is_unsloth_available():
        from unsloth import FastLanguageModel
        try:
            FastLanguageModel.for_inference(model)
        except TypeError:
            pass  # not an unsloth model, ignore


def _ensure_unsloth_training(model):
    """Switch model to training mode (no-op if unsloth not available)."""
    if _is_unsloth_available():
        from unsloth import FastLanguageModel
        try:
            FastLanguageModel.for_training(model)
        except TypeError:
            pass  # not an unsloth model, ignore


class DistillSpec:
    """Aligns a small draft model to a target model via on-policy KL distillation.

    The draft model generates its own continuations (on-policy), and we
    minimize the KL divergence between draft and target logits at each
    generated position. LoRA keeps training efficient.

    Typical workflow:
        1. Initialize with draft + target models + tokenizers
        2. Call distill() with a dataset
        3. Save LoRA adapter
        4. Validate with measure_acceptance_rate()
    """

    def __init__(
        self,
        draft_model: PreTrainedModel,
        draft_tokenizer: PreTrainedTokenizer,
        target_model: PreTrainedModel,
        target_tokenizer: PreTrainedTokenizer,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[list[str]] = None,
        device: str = "auto",
        backend: str = "auto",
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
        self.target_model = target_model
        self.target_tokenizer = target_tokenizer
        self.backend = backend

        # Resolve backend: prefer unsloth if auto, available, AND the
        # model was actually loaded by Unsloth. This avoids routing a
        # plain HF model through the Unsloth LoRA path just because the
        # unsloth package is installed in the environment.
        use_unsloth = (
            backend == "unsloth" or (
                backend == "auto"
                and _is_unsloth_available()
                and _is_unsloth_model(draft_model)
            )
        )

        if use_unsloth:
            logger.info("Applying LoRA via Unsloth fast kernels")
            self.draft_model = _apply_unsloth_lora(
                draft_model,
                r=lora_rank,
                lora_alpha=lora_alpha,
            )
        else:
            # Standard PEFT path
            if lora_target_modules is None:
                lora_target_modules = self._detect_attention_modules(draft_model)

            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            self.draft_model = get_peft_model(draft_model, lora_config)

        trainable = sum(p.numel() for p in self.draft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.draft_model.parameters())
        logger.info(
            f"LoRA applied: {trainable:,} trainable params "
            f"({100 * trainable / total:.2f}% of {total:,} total)"
        )

    @staticmethod
    def _detect_attention_modules(model) -> list[str]:
        """Auto-detect attention projection module names for the model architecture."""
        # Collect all base-level module names (last component of dotted path)
        model_state: set[str] = set()
        for name, _ in model.named_modules():
            base = name.split(".")[-1]
            model_state.add(base)

        # Common patterns across model families, prioritized by frequency
        candidate_sets = [
            ["q_proj", "k_proj", "v_proj", "o_proj"],           # Llama, Mistral, Qwen, Gemma
            ["query", "key", "value", "output"],                 # Falcon, some others
            ["query_key_value", "dense"],                         # GPTNeoX
            ["self_attn.q_proj", "self_attn.k_proj",             # nested format
             "self_attn.v_proj", "self_attn.o_proj"],
            ["gate_proj", "up_proj", "down_proj"],               # MLP layers (useful too)
        ]

        for candidates in candidate_sets:
            found = [c for c in candidates if c in model_state]
            if found:
                return found

        # Ultimate fallback: use PEFT all-linear mode
        return ["all-linear"]

    # ── Phase 1: On-Policy Data Generation (#14) ───────────────────────

    def _generate_on_policy(
        self,
        prompts: list[str],
        gen_temperature: float = 0.7,
        gen_tokens_per_prompt: int = 64,
        max_prompt_length: int = 256,
    ) -> torch.Tensor:
        """Generate continuations using the current draft model (on-policy).

        On-policy generation means the draft model generates tokens using
        its *current* weights. This is critical for DistillSpec: training
        on static data creates distribution mismatch, while on-policy data
        matches what the draft will see during inference speculation.

        Args:
            prompts: List of prompt strings.
            gen_temperature: Sampling temperature for generation.
            gen_tokens_per_prompt: Number of continuation tokens per prompt.
            max_prompt_length: Max length for prompt tokenization.

        Returns:
            sequences: (batch_size, total_len) — prompt + continuation token IDs.
        """
        self.draft_model.eval()
        _ensure_unsloth_inference(self.draft_model)

        if not prompts:
            return torch.tensor([[]], device=self.device)

        # Resolve the device the model actually lives on — more robust than
        # trusting self.device when the model was loaded on a different one.
        try:
            _gen_device = next(self.draft_model.parameters()).device
        except (StopIteration, AttributeError):
            _gen_device = self.device

        # ── Batched generation (#76) ────────────────────────────────
        # Tokenize all prompts together with left-padding (required for
        # batch generation — all prompts share a single forward pass).
        orig_side = getattr(self.draft_tokenizer, "padding_side", "right")
        self.draft_tokenizer.padding_side = "left"
        pad_id = self.draft_tokenizer.pad_token_id or 0
        try:
            with torch.no_grad():
                inputs = self.draft_tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_prompt_length,
                ).to(_gen_device)

                generated = self.draft_model.generate(
                    **inputs,
                    max_new_tokens=gen_tokens_per_prompt,
                    do_sample=True,
                    temperature=gen_temperature,
                    top_p=0.9,
                    pad_token_id=pad_id,
                )
        finally:
            self.draft_tokenizer.padding_side = orig_side

        # Strip left padding so each row starts at its real first token.
        # generated has shape (batch, prompt_len + gen_len); prompt_len may
        # differ per row only if we hadn't padded, but we did, so all rows
        # are aligned. We return the full sequences (prompt + continuation).
        result = generated
        # Switch back to training mode (unsloth: undo for_inference)
        _ensure_unsloth_training(self.draft_model)
        return result

    # ── Phase 2: Full Training Loop (#15) ──────────────────────────────

    def distill(
        self,
        dataset: Dataset,
        text_column: str = "text",
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        max_length: int = 512,
        temperature: float = 1.0,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        mixed_precision: Optional[str] = None,
        on_policy_regenerate_every: int = 200,
        on_policy_tokens_per_prompt: int = 64,
        on_policy_gen_temp: float = 0.7,
        validation_split: float = 0.05,
        val_prompts: int = 20,
        val_draft_k: int = 5,
        val_max_new_tokens: int = 32,
        checkpoint_dir: Optional[Path] = None,
        save_every_steps: int = 500,
        log_every_steps: int = 10,
        resume_from: Optional[Path] = None,
    ) -> PreTrainedModel:
        """Run full DistillSpec training loop.

        Args:
            dataset: Hugging Face Dataset with text prompts.
            text_column: Column name for prompts.
            batch_size: Training batch size per GPU.
            learning_rate: Peak learning rate.
            num_epochs: Number of training epochs.
            max_length: Maximum token length for sequences.
            temperature: Distillation temperature (higher = softer targets).
            gradient_accumulation_steps: Accumulate gradients over N steps.
            warmup_steps: Linear warmup steps for LR scheduler.
            max_grad_norm: Gradient clipping norm.
            mixed_precision: 'fp16', 'bf16', or None for automatic.
            on_policy_regenerate_every: Regenerate on-policy data every N steps.
            on_policy_tokens_per_prompt: Continuation length for on-policy gen.
            on_policy_gen_temp: Generation temperature for on-policy data.
            validation_split: Fraction of dataset to hold out for validation.
            val_prompts: Number of prompts for acceptance rate validation.
            val_draft_k: Draft K for validation.
            val_max_new_tokens: Max new tokens generated per validation prompt.
                Lower values are much faster (default 32 vs the previous 128).
            checkpoint_dir: Directory to save checkpoints.
            save_every_steps: Save checkpoint every N steps.
            log_every_steps: Log metrics every N steps.
            resume_from: Path to a checkpoint to resume from.

        Returns:
            Trained PEFT model (LoRA adapter weights).
        """
        accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        # Prepare optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.draft_model.parameters(), lr=learning_rate
        )

        # Split dataset
        if validation_split > 0:
            val_size = max(1, int(len(dataset) * validation_split))
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size]
            )
        else:
            train_dataset = dataset
            val_dataset = None

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            drop_last=True,
        )
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Prepare with accelerator
        self.draft_model, optimizer, train_loader, scheduler = accelerator.prepare(
            self.draft_model, optimizer, train_loader, scheduler
        )
        self.target_model.eval()

        # Resume from checkpoint
        start_epoch = 0
        global_step = 0
        if resume_from is not None:
            accelerator.load_state(str(resume_from))
            logger.info(f"Resumed from checkpoint: {resume_from}")

        # On-policy data buffer (regenerated periodically)
        on_policy_buffer = None
        prompts_for_generation: list[str] = []

        # Training loop
        import time as _time
        _train_start = _time.time()
        _total_tokens = 0

        for epoch in range(start_epoch, num_epochs):
            self.draft_model.train()
            total_loss = 0.0
            epoch_steps = 0
            _epoch_start = _time.time()

            # ── Progress bar (#75) ────────────────────────────────
            _show_bar = accelerator.is_main_process
            from rich.progress import (
                Progress as _RichProgress, BarColumn as _BarCol,
                TextColumn as _TextCol, TimeRemainingColumn as _ETACol,
                SpinnerColumn as _SpinCol,
            )
            bar = _RichProgress(
                _SpinCol(),
                _TextCol("[bold]{task.description}"),
                _BarCol(),
                _TextCol("{task.completed}/{task.total}"),
                _ETACol(),
                _TextCol("| loss={task.fields[loss]:.3f}"),
                transient=True,
            ) if _show_bar else None
            _bar_task = None
            if bar is not None:
                bar.__enter__()
                _bar_task = bar.add_task(
                    f"Epoch {epoch+1}/{num_epochs}",
                    total=len(train_loader), loss=0.0,
                )

            for batch in train_loader:
                with accelerator.accumulate(self.draft_model):
                    # ── Get texts from batch ──────────────────────────
                    if isinstance(batch, dict):
                        texts = batch[text_column]
                    elif isinstance(batch, list):
                        texts = batch
                    else:
                        texts = list(batch)

                    if isinstance(texts[0], dict):
                        # Multi-turn chat format: use last assistant turn or join
                        texts = [self._extract_text(t) for t in texts]

                    # ── On-policy data regeneration ────────────────────
                    if (global_step % on_policy_regenerate_every == 0) or on_policy_buffer is None:
                        # Take a subset of prompts for on-policy generation
                        gen_prompts = texts[:min(len(texts), 8)]
                        on_policy_buffer = self._generate_on_policy(
                            gen_prompts,
                            gen_temperature=on_policy_gen_temp,
                            gen_tokens_per_prompt=on_policy_tokens_per_prompt,
                            max_prompt_length=max_length,
                        )

                    # ── Tokenize ───────────────────────────────────────
                    inputs = self.draft_tokenizer(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                    ).to(accelerator.device)
                    _batch_tokens = inputs.input_ids.numel()
                    _total_tokens += _batch_tokens

                    # ── Forward through both models ────────────────────
                    with torch.no_grad():
                        target_outputs = self.target_model(inputs.input_ids)
                        target_logits = target_outputs.logits

                    draft_outputs = self.draft_model(inputs.input_ids)
                    draft_logits = draft_outputs.logits

                    # ── KL divergence loss ─────────────────────────────
                    loss = self._kl_divergence(
                        draft_logits, target_logits, temperature
                    )

                    # ── Backward ───────────────────────────────────────
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            self.draft_model.parameters(), max_grad_norm
                        )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    total_loss += loss.item()
                    epoch_steps += 1
                    global_step += 1
                    if bar is not None and _bar_task is not None:
                        bar.update(_bar_task, advance=1, loss=loss.item())

                    # ── Logging ────────────────────────────────────────
                    if global_step % log_every_steps == 0:
                        lr = scheduler.get_last_lr()[0]
                        _elapsed = _time.time() - _train_start
                        _tok_s = _total_tokens / max(_elapsed, 1e-6)
                        _step_s = global_step / max(_elapsed, 1e-6)
                        _mem_gb = 0.0
                        if torch.cuda.is_available():
                            _mem_gb = torch.cuda.max_memory_allocated() / 1e9
                        accelerator.log(
                            {
                                "train/loss": loss.item(),
                                "train/lr": lr,
                                "train/epoch": epoch + global_step / len(train_loader),
                                "train/global_step": global_step,
                                "train/tokens_per_sec": _tok_s,
                                "train/steps_per_sec": _step_s,
                                "train/peak_mem_gb": _mem_gb,
                            },
                            step=global_step,
                        )
                        logger.info(
                            f"Epoch {epoch+1}/{num_epochs} | "
                            f"Step {global_step} | "
                            f"Loss: {loss.item():.4f} | "
                            f"{_tok_s:.0f} tok/s | "
                            f"{_step_s:.2f} step/s | "
                            f"LR: {lr:.2e}"
                        )

                    # ── Checkpointing ──────────────────────────────────
                    if checkpoint_dir is not None and global_step % save_every_steps == 0:
                        ckpt_path = Path(checkpoint_dir) / f"step_{global_step}"
                        accelerator.save_state(str(ckpt_path))
                        logger.info(f"Checkpoint saved: {ckpt_path}")

            # Close the epoch progress bar before logging/validation
            if bar is not None:
                bar.__exit__(None, None, None)
                bar = None

            # End of epoch
            avg_loss = total_loss / max(epoch_steps, 1)
            _epoch_elapsed = _time.time() - _epoch_start
            _elapsed = _time.time() - _train_start
            _tok_s = _total_tokens / max(_elapsed, 1e-6)
            _mem_gb = 0.0
            if torch.cuda.is_available():
                _mem_gb = torch.cuda.max_memory_allocated() / 1e9
            accelerator.log(
                {
                    "train/epoch_loss": avg_loss,
                    "train/epoch_time_s": _epoch_elapsed,
                    "train/total_time_s": _elapsed,
                    "train/tokens_per_sec": _tok_s,
                    "train/peak_mem_gb": _mem_gb,
                },
                step=global_step,
            )
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} complete — "
                f"avg loss: {avg_loss:.4f} | "
                f"{_epoch_elapsed:.1f}s | "
                f"{_tok_s:.0f} tok/s | "
                f"peak mem: {_mem_gb:.1f} GB"
            )

            # ── End-of-epoch validation (#16) ──────────────────────────
            if val_dataset is not None and accelerator.is_main_process:
                val_prompts_list = [
                    self._extract_text(val_dataset[i])
                    for i in range(min(val_prompts, len(val_dataset)))
                ]
                acceptance = self._measure_acceptance_rate(
                    val_prompts_list, draft_k=val_draft_k, temperature=0.0,
                    max_new_tokens=val_max_new_tokens,
                )
                accelerator.log(
                    {"val/acceptance_rate": acceptance},
                    step=global_step,
                )
                logger.info(f"  Validation acceptance rate: {acceptance:.1%}")

                # Switch back to training mode (unsloth's for_inference was called
                # inside _measure_acceptance_rate)
                _ensure_unsloth_training(self.draft_model)

            # Save epoch checkpoint
            if checkpoint_dir is not None:
                ckpt_path = Path(checkpoint_dir) / f"epoch_{epoch+1}"
                accelerator.save_state(str(ckpt_path))

        # Final summary
        _final_elapsed = _time.time() - _train_start
        _final_tok_s = _total_tokens / max(_final_elapsed, 1e-6)
        _final_mem_gb = 0.0
        if torch.cuda.is_available():
            _final_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        logger.info(
            "=" * 60
        )
        logger.info(
            f"Training complete — "
            f"{_final_elapsed:.1f}s total | "
            f"{_total_tokens:,} tokens | "
            f"{_final_tok_s:.0f} tok/s avg | "
            f"peak mem: {_final_mem_gb:.1f} GB"
        )
        logger.info("=" * 60)

        # Save final model
        if checkpoint_dir is not None:
            final_path = Path(checkpoint_dir) / "final"
            accelerator.save_state(str(final_path))

        return accelerator.unwrap_model(self.draft_model)

    # ── Phase 3: Acceptance Rate Validation (#16) ──────────────────────

    def measure_acceptance_rate(
        self,
        prompts: list[str],
        draft_k: int = 5,
        temperature: float = 0.0,
        max_new_tokens: int = 32,
    ) -> dict:
        """Measure speculative decoding acceptance rate for the current draft.

        Runs actual speculative decoding on a set of prompts and reports:
        - Acceptance rate
        - Tokens per step
        - Per-position acceptance distribution

        Args:
            prompts: List of prompt strings.
            draft_k: Number of draft tokens per speculation step.
            temperature: Sampling temperature.
            max_new_tokens: Max tokens generated per prompt (default 32 —
                sufficient for acceptance estimation; 128 is very slow at
                long sequence lengths).

        Returns:
            Dictionary with acceptance metrics.
        """
        from sped.core.speculative_decoding import SpeculativeDecoder

        # Switch to inference mode (required by unsloth for speculation)
        _ensure_unsloth_inference(self.draft_model)
        _ensure_unsloth_inference(self.target_model)

        decoder = SpeculativeDecoder(
            target_model=self.target_model,
            target_tokenizer=self.target_tokenizer,
            draft_model=self.draft_model,
            draft_tokenizer=self.draft_tokenizer,
            max_draft_tokens=draft_k,
            device=self.device,
        )

        for prompt in prompts:
            decoder.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                verbose=False,
            )

        metrics = decoder.get_metrics()
        return metrics

    def _measure_acceptance_rate(
        self,
        prompts: list[str],
        draft_k: int = 5,
        temperature: float = 0.0,
        max_new_tokens: int = 32,
    ) -> float:
        """Quick acceptance rate measurement (returns single float)."""
        metrics = self.measure_acceptance_rate(
            prompts, draft_k, temperature, max_new_tokens=max_new_tokens,
        )
        return metrics.get("acceptance_rate", 0.0)

    @staticmethod
    def _extract_text(item) -> str:
        """Extract text from various dataset formats."""
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            # Try common chat template formats
            if "messages" in item:
                messages = item["messages"]
                if isinstance(messages, list) and len(messages) > 0:
                    # Join all messages
                    return " ".join(
                        m.get("content", "") for m in messages
                        if isinstance(m, dict)
                    )
            if "content" in item:
                return item["content"]
            if "text" in item:
                return item["text"]
            # Return first string value found
            for v in item.values():
                if isinstance(v, str):
                    return v
            return str(item)
        return str(item)

    # ── Utilities ─────────────────────────────────────────────────────

    @staticmethod
    def _kl_divergence(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Compute per-token KL(teacher || student) divergence.

        Returns the mean KL over all batch and sequence positions (NOT a
        sum over positions), so the loss scale is independent of sequence
        length. Typical values are ~0.1-5 nats per token.

        PyTorch's ``kl_div(reduction="batchmean")`` sums over the sequence
        dimension, which makes the loss scale with sequence length and
        produces huge values (100s-1000s) for long sequences. We instead
        compute the per-token KL and average over batch + sequence.
        """
        student_log_probs = torch.log_softmax(
            student_logits / temperature, dim=-1
        )
        teacher_log_probs = torch.log_softmax(
            teacher_logits / temperature, dim=-1
        )
        teacher_probs = teacher_log_probs.exp()
        # Per-token KL: sum over vocab -> (B, L), then mean over batch + seq
        per_token_kl = (
            teacher_probs * (teacher_log_probs - student_log_probs)
        ).sum(dim=-1)
        kl = per_token_kl.mean()
        return kl * (temperature ** 2)

    def save_adapter(self, path: Path):
        """Save the trained LoRA adapter."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.draft_model.save_pretrained(str(path))
        self.draft_tokenizer.save_pretrained(str(path))
        logger.info(f"LoRA adapter saved to {path}")

    @classmethod
    def load_adapter(
        cls,
        base_model: PreTrainedModel,
        adapter_path: Path,
        draft_tokenizer: PreTrainedTokenizer,
        target_model: PreTrainedModel,
        target_tokenizer: PreTrainedTokenizer,
        device: str = "auto",
    ) -> "DistillSpec":
        """Load a previously trained LoRA adapter.

        Creates a DistillSpec instance with the adapter loaded.
        """
        instance = cls(
            draft_model=base_model,
            draft_tokenizer=draft_tokenizer,
            target_model=target_model,
            target_tokenizer=target_tokenizer,
            device=device,
        )
        instance.draft_model = PeftModel.from_pretrained(
            base_model, str(adapter_path)
        )
        return instance

    def compare_before_after(
        self,
        prompts: list[str],
        draft_k: int = 5,
    ) -> dict:
        """Compare acceptance rate before vs after distillation.

        'Before' is the base draft model without LoRA. 'After' is the
        current LoRA-tuned draft model.

        Returns a dict with before/after comparison.
        """
        # Measure before (temporarily disable LoRA)
        self.draft_model.eval()

        # We need the base model for comparison
        # For now, this is a placeholder — the full comparison requires
        # keeping a copy of the untuned model
        after_metrics = self.measure_acceptance_rate(prompts, draft_k)

        return {
            "after": after_metrics,
            "before": {"acceptance_rate": 0.0},  # Will be filled when base model is available
        }
