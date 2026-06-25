# Autoresearch: Speed up sped training throughput

## Objective
Optimize the DistillSpec training loop for faster throughput on an L4 GPU (22 GB). The workload is knowledge distillation from Qwen3-8B (target, 4-bit) to Qwen3-1.7B (draft, 4-bit + LoRA rank 32). Each step does: target forward (frozen) + draft forward + draft backward (LoRA) + optimizer. The goal is to maximize tokens/second.

## Metrics
- **Primary**: tokens_per_sec (tok/s, higher is better) — throughput of training step
- **Secondary**: vram_gb (GB, lower is better) — peak GPU memory, must stay ≤ 20 GB
- **Secondary**: step_time_ms (ms, lower is better) — wall time per training step

## How to Run
```bash
bash .auto/measure.sh
```
Outputs `METRIC metrics=...` JSON line with primary + secondary metrics.

## Files in Scope
- `sped/distillation/distillspec.py` — training loop, KL loss, data loading, on-policy generation, quantization
- `sped/cli/distil.py` — CLI entry point (controls `--batch-size`, `--draft-dtype`, etc.)
- `sped/serving/unsloth_backend.py` — Unsloth backend integration (LoRA loading, fast inference)
- `sped/utils/unsloth_cache.py` — 4-bit cache resolution

## Off Limits
- Test files (`tests/`)
- Non-training code paths (serve, benchmark, validation metrics)
- Core speculative decoding module (`sped/core/`)
- Models and datasets on disk (not modifying storage paths or file formats)

## Constraints
- Must stay within L4 VRAM (22 GB, target ≤ 20 GB for headroom)
- Training correctness: loss must remain in ~0.5-1.5 range under same conditions
- No new external dependencies (no pip installs unless they're optional detect+use)
- Must work with `--backend unsloth` (all Unsloth patching stays intact)

## What's Been Tried

### ✅ Experiment 1: torch.inference_mode() for target forward (+19%)
`torch.no_grad()` → `torch.inference_mode()` in `_get_target_logits`. Disables more
autograd tracking internals. Safe for frozen models. Baseline: 1128→1339 tok/s.
Committed at `6ce41a0`.

### ✅ Experiment 7: Target hidden-state cache (+124% on hit)
Key insight: hidden states (1, L, 4096) are 38× smaller than logits (1, L, 151936).
Cache stores `target.model(...)` outputs in bf16 on CPU RAM (33.5 MB vs 1.24 GB
each at L=4096). On cache hit: load hidden→GPU→lm_head only (~26ms at L=1480)
instead of full 8B target forward (677ms).

Implementation:
- `_get_target_logits` uses `target.model(..., causal_mask=LowerTriangularMask())`
  + `target.lm_head(...)` split, which matches `target(...)` exactly
- Cache: 200-entry FIFO dict on CPU (~6.7 GB max at L=4096)
- Cache miss: compute + write (same speed as baseline, ~1% overhead)
- Cache hit: 2.2× step speedup (1106ms → 586ms at L=1480)

Limitations:
- Cache key = full batch tensor (incl. padding). bs=2 batches produce different
  keys than bs=1 → cache misses across different batch compositions
- Best for bs=1 with deterministic ordering (eval10: 10 entries → 100% epoch 2 hit)
- Scales to ~200 examples at L=4096 before FIFO eviction reduces hit rate

### ❌ Experiment 2: Disable gradient checkpointing (crash)
Setting `use_gradient_checkpointing=False` in Unsloth's get_peft_model crashes —
the patched LlamaModel_fast_forward expects `_gradient_checkpointing_func`.
Unsloth requires checkpointing. Reverted.

### ❌ Experiment 3: torch.compile on draft model (+3% — not worth it)
`torch.compile(draft, mode='reduce-overhead')` gave only 41ms saving out of
1341ms step time. The backward pass recomputation is the bottleneck, not the
forward. Plus dynamic shapes cause recompilation. Reverted.

### ❌ Experiment 4: torch.compile on target model (2× SLOWER)
Unsloth's patched 4-bit model with custom CUDA kernels is incompatible with
torch.compile. 677ms → 1414ms. Reverted.

### ❌ Experiment 5: bf16 autocast for draft forward (neutral)
Adding autocast to the draft forward didn't help — Unsloth already handles
precision internally. 217ms → 229ms. Not worth it.

### ❌ Experiment 6: fused AdamW for LoRA (-13% regression)
`fused=True` in AdamW regressed throughput (1339→1169 tok/s). The 392 small
LoRA tensors (32K params each) have too much kernel launch overhead per tensor
for the fused kernel to benefit. Reverted.

### Key architectural insight
**Target forward = 48% of step time.** It's bandwidth-bound (8B model weights
read from HBM per token). Hidden-state caching saves the target forward on
cache hit, shifting the bottleneck to draft backward+opt (now 65% of step).

Total practical gains from merged changes:
- Epoch 1: +19% (inference_mode only, no cache hits)
- Epoch 2+: +124% (inference_mode + cache hits on previously seen examples)
