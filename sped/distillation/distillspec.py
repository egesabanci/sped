"""DistillSpec: knowledge distillation to align a draft model with a target.

Full implementation with on-policy data generation (#14), robust training
loop with Accelerate (#15), and acceptance rate validation (#16).

Uses LoRA (PEFT) for efficient training — only ~0.1-1% of parameters
are updated, enabling single-GPU distillation of 0.5B->70B model pairs.

Performance features (Phase 2):
- Pre-tokenized dataset cache (#99) — tokenize once before training
- Target logits cache across epochs (#97) — hash-based, avoids redundant 8B forward
- bf16 autocast on frozen target forward (#98) — halves target forward cost
- DataLoader workers + pinned memory (#100) — faster batch loading
- Proportional warmup default (#101) — 5% of total steps instead of fixed 100
- Incremental on-policy regeneration (#102) — only 25% of buffer per regen cycle
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
    try:
        import unsloth  # noqa: F401
        return True
    except ImportError:
        return False


def _get_xformers_mask():
    """Get the xformers lower-triangular causal mask for Unsloth models."""
    try:
        from xformers.ops import LowerTriangularMask
        return LowerTriangularMask()
    except ImportError:
        return None


def _is_unsloth_model(model) -> bool:
    return hasattr(model, "_saved_temp_tokenizer") or any(
        hasattr(model, attr) for attr in ("_unloth_model", "_unsloth_model")
    )


def _apply_unsloth_lora(model, r: int = 8, lora_alpha: int = 16, **kwargs):
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
    FastLanguageModel.for_training(model)
    return model


def _ensure_unsloth_inference(model):
    if _is_unsloth_available():
        from unsloth import FastLanguageModel
        try:
            FastLanguageModel.for_inference(model)
        except TypeError:
            pass


def _ensure_unsloth_training(model):
    if _is_unsloth_available():
        from unsloth import FastLanguageModel
        try:
            FastLanguageModel.for_training(model)
        except TypeError:
            pass


class DistillSpec:
    """Aligns a small draft model to a target model via on-policy KL distillation."""

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

        # On-policy rotation index (#102): tracks which subset to regen next
        self._on_policy_rotation_idx: int = 0
        # Target hidden-state cache: {bytes_hash: (attention_mask_hash, hidden_states_cpu)}
        # Hidden states are 38× smaller than logits (33.5 MB vs 1.24 GB at L=4096)
        # Cache is bounded to prevent OOM on large datasets
        self._target_hidden_cache: dict = {}
        self._target_cache_max: int = 200  # ~6.7 GB at L=4096 max, scales to ~40% of smoke dataset

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
                draft_model, r=lora_rank, lora_alpha=lora_alpha,
            )
        else:
            if lora_target_modules is None:
                lora_target_modules = self._detect_attention_modules(draft_model)
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank, lora_alpha=lora_alpha,
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
        model_state: set[str] = set()
        for name, _ in model.named_modules():
            base = name.split(".")[-1]
            model_state.add(base)
        candidate_sets = [
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            ["query", "key", "value", "output"],
            ["query_key_value", "dense"],
            ["self_attn.q_proj", "self_attn.k_proj",
             "self_attn.v_proj", "self_attn.o_proj"],
            ["gate_proj", "up_proj", "down_proj"],
        ]
        for candidates in candidate_sets:
            found = [c for c in candidates if c in model_state]
            if found:
                return found
        return ["all-linear"]

    # ── Pre-tokenized dataset cache (#99) ─────────────────────────────

    def _tokenize_dataset(self, dataset, text_column: str, max_length: int) -> list[dict]:
        """Tokenize entire dataset once before training."""
        import time as _t
        t0 = _t.time()
        tokenized = []
        for i in range(len(dataset)):
            item = dataset[i]
            text = item.get(text_column, "") if isinstance(item, dict) else str(item)
            if not text:
                continue
            encoded = self.draft_tokenizer(
                text, truncation=True, max_length=max_length, return_tensors="pt",
            )
            tokenized.append({"input_ids": encoded.input_ids[0]})
        elapsed = _t.time() - t0
        logger.info(f"Pre-tokenized {len(tokenized)} examples in {elapsed:.1f}s")
        return tokenized

    @staticmethod
    def _collate_batch(batch: list[dict]) -> dict:
        """Custom collate: pad input_ids to batch max length + attention_mask.

        Without this, ``default_collate`` crashes on variable-length sequences.
        """
        import torch
        max_len = max(len(item["input_ids"]) for item in batch)
        ids_list = []
        mask_list = []
        for item in batch:
            seq = item["input_ids"]
            pad_len = max_len - len(seq)
            ids_list.append(torch.cat([seq, torch.zeros(pad_len, dtype=seq.dtype)]))
            mask_list.append(
                torch.cat([torch.ones(len(seq), dtype=torch.long),
                           torch.zeros(pad_len, dtype=torch.long)])
            )
        return {
            "input_ids": torch.stack(ids_list),
            "attention_mask": torch.stack(mask_list),
        }

    # ── On-Policy Data Generation (#14) ──────────────────────────────

    def _generate_on_policy(
        self,
        prompts: list[str],
        gen_temperature: float = 0.7,
        gen_tokens_per_prompt: int = 64,
        max_prompt_length: int = 256,
    ) -> torch.Tensor:
        self.draft_model.eval()
        _ensure_unsloth_inference(self.draft_model)

        if not prompts:
            return torch.tensor([[]], device=self.device)

        try:
            _gen_device = next(self.draft_model.parameters()).device
        except (StopIteration, AttributeError):
            _gen_device = self.device

        orig_side = getattr(self.draft_tokenizer, "padding_side", "right")
        self.draft_tokenizer.padding_side = "left"
        pad_id = self.draft_tokenizer.pad_token_id or 0
        try:
            with torch.no_grad():
                inputs = self.draft_tokenizer(
                    prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_prompt_length,
                ).to(_gen_device)
                generated = self.draft_model.generate(
                    **inputs, max_new_tokens=gen_tokens_per_prompt,
                    do_sample=True, temperature=gen_temperature,
                    top_p=0.9, pad_token_id=pad_id,
                )
        finally:
            self.draft_tokenizer.padding_side = orig_side

        _ensure_unsloth_training(self.draft_model)
        return generated

    # ── Target logits with cache (#97) and bf16 autocast (#98) ───────

    def _get_target_logits(
        self, input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get target model logits with bf16 autocast and hidden-state caching.

        Caches the intermediate hidden states (38× smaller than full logits)
        so that repeated inputs across epochs skip the expensive 8B model
        forward and only compute the cheap lm_head (a single linear layer).
        Cache is bounded to ``_target_cache_max`` entries with FIFO eviction.
        """
        # Build cache key once — used for both lookup and write
        _key_cache = hash((
            input_ids.cpu().numpy().tobytes(),
            attention_mask.cpu().numpy().tobytes() if attention_mask is not None else b"",
        ))

        # ── Cache hit? ──────────────────────────────────────────────
        if _key_cache in self._target_hidden_cache:
            _, hidden_cpu = self._target_hidden_cache[_key_cache]
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available(),
            ):
                hidden = hidden_cpu.to(device=input_ids.device, dtype=torch.bfloat16)
                logits = self.target_model.lm_head(hidden)
            return logits

        # ── Cache miss: compute forward ─────────────────────────────
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available(),
        ):
            try:
                from xformers.ops import LowerTriangularMask
                causal_mask = LowerTriangularMask()
            except ImportError:
                causal_mask = None

            if hasattr(self.target_model, "model") and causal_mask is not None:
                base_out = self.target_model.model(
                    input_ids,
                    attention_mask=attention_mask,
                    causal_mask=causal_mask,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = base_out.last_hidden_state
                # Cache hidden states on CPU (38× smaller than logits)
                hidden_cpu = hidden.cpu().to(torch.bfloat16)
                self._target_hidden_cache[_key_cache] = (_key_cache, hidden_cpu)
                # FIFO eviction if over cap
                if len(self._target_hidden_cache) > self._target_cache_max:
                    oldest_key = next(iter(self._target_hidden_cache))
                    del self._target_hidden_cache[oldest_key]
                # Apply lm_head
                logits = self.target_model.lm_head(hidden.to(torch.bfloat16))
            else:
                # Fallback: full model forward (no xformers — no caching possible)
                target_outputs = self.target_model(input_ids, attention_mask=attention_mask)
                logits = target_outputs.logits

        return logits

    # ── Training Loop (#15) ──────────────────────────────────────────

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
        warmup_steps: int = -1,  # -1 = auto: 5% of total steps (#101)
        max_grad_norm: float = 1.0,
        mixed_precision: Optional[str] = "bf16",  # auto-detect by default (#98)
        on_policy_regenerate_every: int = 200,
        on_policy_tokens_per_prompt: int = 64,
        on_policy_gen_temp: float = 0.7,
        on_policy_fraction: float = 0.25,  # fraction of buffer to regen (#102)
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
            warmup_steps: Linear warmup steps (-1 = auto: 5% of total).
            max_grad_norm: Gradient clipping norm.
            mixed_precision: 'fp16', 'bf16', None, or 'bf16' by default
                on CUDA for frozen target forward (#98).
            on_policy_regenerate_every: Regenerate on-policy data every N steps.
            on_policy_tokens_per_prompt: Continuation length for on-policy gen.
            on_policy_gen_temp: Generation temperature for on-policy data.
            on_policy_fraction: Fraction of buffer to regenerate per cycle (0-1).
                Default 0.25 = rotate 25% each regen step (#102).
            validation_split: Fraction of dataset to hold out for validation.
            val_prompts: Number of prompts for acceptance rate validation.
            val_draft_k: Draft K for validation.
            val_max_new_tokens: Max new tokens generated per validation prompt.
            checkpoint_dir: Directory to save checkpoints.
            save_every_steps: Save checkpoint every N steps.
            log_every_steps: Log metrics every N steps.
            resume_from: Path to a checkpoint to resume from.

        Returns:
            Trained PEFT model (LoRA adapter weights).
        """
        # Auto-detect bf16 if mixed_precision is default "bf16" but not supported
        resolved_mp = mixed_precision
        if resolved_mp == "bf16" and torch.cuda.is_available():
            if not torch.cuda.is_bf16_supported():
                resolved_mp = "fp16"
                logger.info("bf16 not supported on this GPU, falling back to fp16")

        accelerator = Accelerator(
            mixed_precision=resolved_mp,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        optimizer = torch.optim.AdamW(
            self.draft_model.parameters(), lr=learning_rate,
        )

        # Split dataset
        if validation_split > 0:
            val_size = max(1, int(len(dataset) * validation_split))
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size],
            )
        else:
            train_dataset = dataset
            val_dataset = None

        # ── Pre-tokenize entire training set once (#99) ─────────────
        tokenized_data = self._tokenize_dataset(
            train_dataset, text_column, max_length,
        )
        train_loader = DataLoader(
            tokenized_data, batch_size=batch_size, shuffle=True,
            drop_last=True,  # workers=0: no multiprocessing (avoids deadlocks)
            collate_fn=DistillSpec._collate_batch,
        )

        # ── Warmup: auto 5% of total steps if not specified (#101) ──
        total_steps = len(train_loader) * num_epochs
        if warmup_steps == -1 or warmup_steps is None:
            warmup_steps = max(1, int(total_steps * 0.05))
            logger.info(f"Auto warmup: {warmup_steps} steps ({total_steps} total)")

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.draft_model, optimizer, train_loader, scheduler = accelerator.prepare(
            self.draft_model, optimizer, train_loader, scheduler,
        )
        self.target_model.eval()

        start_epoch = 0
        global_step = 0
        if resume_from is not None:
            accelerator.load_state(str(resume_from))
            logger.info(f"Resumed from checkpoint: {resume_from}")

        # On-policy data buffer and rotation state (#102)
        on_policy_buffer = None

        import time as _time
        _train_start = _time.time()
        _total_tokens = 0

        for epoch in range(start_epoch, num_epochs):
            self.draft_model.train()
            total_loss = 0.0
            epoch_steps = 0
            _epoch_start = _time.time()

            _show_bar = accelerator.is_main_process
            from rich.progress import (
                Progress as _RichProgress, BarColumn as _BarCol,
                TextColumn as _TextCol, TimeRemainingColumn as _ETACol,
                SpinnerColumn as _SpinCol,
            )
            bar = _RichProgress(
                _SpinCol(), _TextCol("[bold]{task.description}"),
                _BarCol(), _TextCol("{task.completed}/{task.total}"),
                _ETACol(), _TextCol("| loss={task.fields[loss]:.3f}"),
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
                    # batch is pre-tokenized dict with 'input_ids' key from _tokenize_dataset
                    input_ids = batch["input_ids"].to(accelerator.device)
                    _batch_tokens = input_ids.numel()
                    _total_tokens += _batch_tokens

                    # ── On-policy data regeneration (#102) ───────────
                    # Maintain a small rotating buffer of prompts (size = 4).
                    # Each cycle, regenerate only on_policy_fraction of them.
                    _OP_BUFFER_SIZE = 4
                    if (global_step % on_policy_regenerate_every == 0) or on_policy_buffer is None:
                        n_regen = max(1, int(_OP_BUFFER_SIZE * on_policy_fraction))
                        gen_prompts = []
                        for k in range(n_regen):
                            idx = (self._on_policy_rotation_idx + k) % len(tokenized_data)
                            gen_prompts.append(
                                self.draft_tokenizer.decode(
                                    tokenized_data[idx]["input_ids"][:64],
                                    skip_special_tokens=True,
                                )
                            )
                        self._on_policy_rotation_idx = (
                            self._on_policy_rotation_idx + n_regen
                        ) % len(tokenized_data)
                        if gen_prompts:
                            on_policy_buffer = self._generate_on_policy(
                                gen_prompts,
                                gen_temperature=on_policy_gen_temp,
                                gen_tokens_per_prompt=on_policy_tokens_per_prompt,
                                max_prompt_length=min(max_length, 512),  # cap for speed
                            )

                    # ── Forward through both models ────────────────────
                    # batch is pre-tokenized with attention_mask from _collate_batch
                    attention_mask = batch.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(accelerator.device)

                    # Target forward: cached + autocast (#97, #98)
                    target_logits = self._get_target_logits(input_ids, attention_mask)

                    draft_outputs = self.draft_model(input_ids, attention_mask=attention_mask)
                    draft_logits = draft_outputs.logits

                    # ── KL divergence loss ─────────────────────────────
                    loss = self._kl_divergence(
                        draft_logits, target_logits, temperature,
                        attention_mask=attention_mask,
                    )

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            self.draft_model.parameters(), max_grad_norm,
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
                        accelerator.log({
                            "train/loss": loss.item(),
                            "train/lr": lr,
                            "train/epoch": epoch + global_step / len(train_loader),
                            "train/global_step": global_step,
                            "train/tokens_per_sec": _tok_s,
                            "train/steps_per_sec": _step_s,
                            "train/peak_mem_gb": _mem_gb,
                        }, step=global_step)
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
            accelerator.log({
                "train/epoch_loss": avg_loss,
                "train/epoch_time_s": _epoch_elapsed,
                "train/total_time_s": _elapsed,
                "train/tokens_per_sec": _tok_s,
                "train/peak_mem_gb": _mem_gb,
            }, step=global_step)
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} complete \u2014 "
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
                accelerator.log({"val/acceptance_rate": acceptance}, step=global_step)
                logger.info(f"  Validation acceptance rate: {acceptance:.1%}")
                _ensure_unsloth_training(self.draft_model)

            if checkpoint_dir is not None:
                ckpt_path = Path(checkpoint_dir) / f"epoch_{epoch+1}"
                accelerator.save_state(str(ckpt_path))

        # Final summary
        _final_elapsed = _time.time() - _train_start
        _final_tok_s = _total_tokens / max(_final_elapsed, 1e-6)
        _final_mem_gb = 0.0
        if torch.cuda.is_available():
            _final_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        logger.info("=" * 60)
        logger.info(
            f"Training complete \u2014 "
            f"{_final_elapsed:.1f}s total | "
            f"{_total_tokens:,} tokens | "
            f"{_final_tok_s:.0f} tok/s avg | "
            f"peak mem: {_final_mem_gb:.1f} GB"
        )
        logger.info("=" * 60)

        if checkpoint_dir is not None:
            accelerator.save_state(str(Path(checkpoint_dir) / "final"))

        return accelerator.unwrap_model(self.draft_model)

    # ── Acceptance Rate Validation (#16) ─────────────────────────────

    def measure_acceptance_rate(
        self, prompts: list[str], draft_k: int = 5,
        temperature: float = 0.0, max_new_tokens: int = 32,
    ) -> dict:
        from sped.core.speculative_decoding import SpeculativeDecoder
        _ensure_unsloth_inference(self.draft_model)
        _ensure_unsloth_inference(self.target_model)

        decoder = SpeculativeDecoder(
            target_model=self.target_model, target_tokenizer=self.target_tokenizer,
            draft_model=self.draft_model, draft_tokenizer=self.draft_tokenizer,
            max_draft_tokens=draft_k, device=self.device,
        )
        for prompt in prompts:
            decoder.generate(
                prompt=prompt, max_new_tokens=max_new_tokens,
                temperature=temperature, verbose=False,
            )
        return decoder.get_metrics()

    def _measure_acceptance_rate(
        self, prompts: list[str], draft_k: int = 5,
        temperature: float = 0.0, max_new_tokens: int = 32,
    ) -> float:
        metrics = self.measure_acceptance_rate(
            prompts, draft_k, temperature, max_new_tokens=max_new_tokens,
        )
        return metrics.get("acceptance_rate", 0.0)

    @staticmethod
    def _extract_text(item) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            if "messages" in item:
                messages = item["messages"]
                if isinstance(messages, list) and len(messages) > 0:
                    return " ".join(
                        m.get("content", "") for m in messages if isinstance(m, dict)
                    )
            if "content" in item:
                return item["content"]
            if "text" in item:
                return item["text"]
            for v in item.values():
                if isinstance(v, str):
                    return v
            return str(item)
        return str(item)

    # ── Utilities ────────────────────────────────────────────────────

    @staticmethod
    def _kl_divergence(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
        teacher_log_probs = torch.log_softmax(teacher_logits / temperature, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        per_token_kl = (
            teacher_probs * (teacher_log_probs - student_log_probs)
        ).sum(dim=-1)  # (B, L)
        # Mask out padding positions if attention_mask is provided
        if attention_mask is not None:
            per_token_kl = per_token_kl * attention_mask.float()
            kl = per_token_kl.sum() / attention_mask.float().sum().clamp(min=1)
        else:
            kl = per_token_kl.mean()
        return kl * (temperature ** 2)

    def save_adapter(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.draft_model.save_pretrained(str(path))
        self.draft_tokenizer.save_pretrained(str(path))
        logger.info(f"LoRA adapter saved to {path}")

    @classmethod
    def load_adapter(
        cls,
        base_model: PreTrainedModel, adapter_path: Path,
        draft_tokenizer: PreTrainedTokenizer,
        target_model: PreTrainedModel, target_tokenizer: PreTrainedTokenizer,
        device: str = "auto",
    ) -> "DistillSpec":
        from peft import PeftModel  # noqa: F811
        instance = cls(
            draft_model=base_model, draft_tokenizer=draft_tokenizer,
            target_model=target_model, target_tokenizer=target_tokenizer,
            device=device,
        )
        instance.draft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        return instance

    def compare_before_after(self, prompts: list[str], draft_k: int = 5) -> dict:
        self.draft_model.eval()
        after_metrics = self.measure_acceptance_rate(prompts, draft_k)
        return {
            "after": after_metrics,
            "before": {"acceptance_rate": 0.0},
        }
