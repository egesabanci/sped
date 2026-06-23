# sped User Guide

## Overview

**sped** (Speculative Decoding) is a CLI toolkit that makes your large language models run faster by pairing them with tiny draft models. It works with **any** model pair — even models with completely different tokenizers.

### How It Works

```
Standard decoding:     tok1 → tok2 → tok3 → tok4 → ...  (1 forward pass per token)
Speculative decoding:  draft: [tok1, tok2, tok3, tok4]  (4 cheap forward passes)
                       target: verify all 4 in 1 pass → accept 3
                       → 4 tokens for the cost of ~2 passes
```

## Quick Start

### 1. Install

```bash
uv pip install sped
```

Or from source:
```bash
git clone https://github.com/egesabanci/sped
cd sped
uv venv && uv pip install -e .
```

### 2. List recommended pairings

```bash
sped list pairings
```

### 3. Run inference with speculation

```bash
# Same-vocab pair (e.g., Llama 3.2-1B draft → Llama 3.1-70B target)
sped serve run --target meta-llama/Llama-3.1-70B --draft meta-llama/Llama-3.2-1B

# Cross-vocab pair (e.g., Qwen draft → Llama target)
sped serve run --target meta-llama/Llama-3.1-70B --draft Qwen/Qwen2.5-0.5B --align hybrid

# Apple Silicon (MLX backend)
sped serve run --target mlx-community/Llama-3.2-3B --backend mlx
```

### 4. Benchmark

```bash
sped serve run --target meta-llama/Llama-3.1-70B --draft Qwen/Qwen2.5-0.5B --benchmark
```

### 5. Auto-tune draft K

```bash
sped experiment auto-tune --target meta-llama/Llama-3.1-70B --draft Qwen/Qwen2.5-0.5B
```

## Choosing a Draft Model

### Same-Family (Best Acceptance)

Same-family pairs share a tokenizer, so alignment is lossless and acceptance rates are highest.

| Draft | Target | Speedup Potential |
|-------|--------|-------------------|
| Llama-3.2-1B | Llama-3.1-70B | 2–4× |
| Llama-3.2-3B | Llama-3.1-405B | 3–5× |
| Qwen2.5-0.5B | Qwen2.5-72B | 2–4× |
| Gemma-2-2B | Gemma-2-27B | 2–3× |

### Cross-Family (Universal)

Cross-family pairs need vocabulary alignment (`--align hybrid`). Slightly lower acceptance but works with any combination.

| Draft | Target | Notes |
|-------|--------|-------|
| Qwen2.5-0.5B | Llama-3.1-70B | Great tiny draft for big Llama |
| Phi-3-mini | Llama-3.1-70B | Good general-purpose |
| SmolLM2-360M | Qwen2.5-72B | Extremely lightweight |

## Distilling a Draft Model

For best results, align a small draft model to your specific target model:

```bash
sped distil run \
  --draft Qwen/Qwen2.5-0.5B \
  --target meta-llama/Llama-3.1-70B \
  --dataset HuggingFaceH4/ultrachat_200k \
  --lora-rank 8 \
  --epochs 3
```

This trains LoRA adapters on the draft model to better predict the target's outputs. After distillation, use with:

```bash
sped serve run --target meta-llama/Llama-3.1-70B --draft Qwen/Qwen2.5-0.5B --draft-lora ./draft-lora
```

Validate the improvement:

```bash
sped distil validate --draft Qwen/Qwen2.5-0.5B --target meta-llama/Llama-3.1-70B
```

## Running Experiments

Find the optimal hyperparameters for your model pair:

```bash
sped experiment run \
  --target meta-llama/Llama-3.1-70B \
  --draft Qwen/Qwen2.5-0.5B \
  --draft-k-values 3,5,7,10 \
  --temperatures 0.0,0.7 \
  --align-strategies none,hybrid
```

Results are saved to `./experiment-results/results.json` and `./experiment-results/report.html`.

## Backends

| Backend | Flag | Requires | Best For |
|---------|------|----------|----------|
| Hugging Face | `--backend hf` | Nothing | General purpose |
| MLX | `--backend mlx` | `uv pip install mlx-lm` | Apple Silicon Macs |
| vLLM | `--backend vllm` | `uv pip install vllm` | Production serving |

Auto-detection: `--backend auto` picks MLX on Apple Silicon, HF everywhere else.

## Vocabulary Alignment Strategies

| Strategy | Quality | Speed | When to Use |
|----------|---------|-------|-------------|
| `none` | Exact | Fastest | Same tokenizer |
| `string` | Good | Fast | Cross-tokenizer, draft quality high |
| `probabilistic` | Better | Slower | Cross-tokenizer, need accuracy |
| `hybrid` | Best | Medium | Default — dynamic per-token selection |

Use `auto` to let sped decide based on vocabulary overlap.

## Tuning Tips

1. **Start with K=5** and adjust up (K=7–10) if acceptance rate > 60%
2. **Lower acceptance?** Distill the draft first
3. **Temperature 0.0** gives highest acceptance (greedy decoding)
4. **Temperature > 0** reduces acceptance by ~15% but adds diversity
5. **Cross-vocab pairs** benefit most from the hybrid alignment strategy
6. **Auto-tune** saves time — use it before running a full grid

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `sped: command not found` | Activate venv: `source .venv/bin/activate` |
| CUDA out of memory | Use `--device cpu` or quantize with `-q 4bit` |
| High latency | Reduce `--draft-k` or check acceptance rate |
| Low acceptance rate | Distill the draft model or use same-family models |
| MLX not available | `uv pip install mlx-lm` |
