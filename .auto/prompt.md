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

### Baseline (commit 6085867)
- Profile at L=1024: 744ms/step, 1377 tok/s, 10.1 GB VRAM
- Profile at L=2048: 1344ms/step, 1524 tok/s, 12.3 GB VRAM
- Profile at L=4096: 2735ms/step, 1498 tok/s, 16.7 GB VRAM
- Tok/s is FLAT at ~1500 regardless of seq_len — scales linearly with seq_len
- Target forward = 48% of step time (dominant)
- Draft forward = 18% (smaller)
- Backward+optimizer = 34%
- VRAM headroom at L=4096: 5.3 GB

### Potential directions (not yet tried):
1. Disable Unsloth gradient checkpointing to avoid recomputation in backward — VRAM headroom exists
2. Use `torch.inference_mode()` on target forward (slightly faster than `no_grad`)
3. Investigate on-policy buffer — it generates completions but training uses batch data, not generated data (dead code?)
4. `torch.compile` the draft model for fused LoRA kernels
5. Profile native attention ops to find micro-optimizations
6. Reduce Python overhead in training loop (logging, progress bars)
