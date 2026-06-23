# ⚡ sped — Universal Speculative Decoding

**sped** pairs **any small draft model** with **any large target model** — even with completely different tokenizers — and accelerates inference **2–5×** with zero loss in output quality.

[![CI](https://github.com/egesabanci/sped/actions/workflows/ci.yml/badge.svg)](https://github.com/egesabanci/sped/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/sped)](https://pypi.org/project/sped/)
[![Python](https://img.shields.io/pypi/pyversions/sped)](https://pypi.org/project/sped/)
[![License](https://img.shields.io/pypi/l/sped)](https://github.com/egesabanci/sped/blob/main/LICENSE)

## Install

```bash
uv pip install sped
```

Or from source:

```bash
git clone https://github.com/egesabanci/sped
cd sped
uv venv && uv pip install -e .
```

## Quick Start

```bash
# List recommended draft-target pairs
sped list pairings

# Run inference with speculation (same-vocab)
sped serve run \
  --target meta-llama/Llama-3.1-70B \
  --draft meta-llama/Llama-3.2-1B

# Cross-vocab (different tokenizers)
sped serve run \
  --target meta-llama/Llama-3.1-70B \
  --draft Qwen/Qwen2.5-0.5B \
  --align hybrid

# Benchmark
sped serve run --target meta-llama/Llama-3.1-70B --draft Qwen/Qwen2.5-0.5B --benchmark

# Distill a draft model via PEFT
sped distil run \
  --draft Qwen/Qwen2.5-0.5B \
  --target meta-llama/Llama-3.1-70B \
  --dataset HuggingFaceH4/ultrachat_200k

# Auto-tune draft K
sped experiment auto-tune --target meta-llama/Llama-3.1-70B --draft Qwen/Qwen2.5-0.5B
```

## Features

| Feature | Description |
|---------|-------------|
| **🧠 Any model pair** | Works across tokenizers via Intel/Weizmann vocabulary-agnostic alignment |
| **⚡ 2–5× speedup** | Lossless — output matches target model distribution exactly |
| **🎯 PEFT distillation** | Align tiny draft models with LoRA in ~1 GPU-hour |
| **🔄 Online adaptation** | Draft improves during inference via OSD |
| **🍎 MLX backend** | Optimized for Apple Silicon (M1–M4) |
| **🏭 vLLM support** | Production-grade serving (optional) |
| **🧪 Experiment runner** | Grid-search, HTML reports, auto-tune |

## How It Works

```
Standard:    tok1 → tok2 → tok3 → tok4 → ...  (1 pass/token, slow)
sped:        draft: [tok1 tok2 tok3 tok4]      (4 cheap passes)
             target: ──────verify──────→ [✓✓✓✗]  (1 expensive pass)
             → same output, 2–5× faster
```

## Documentation

- [CLI Reference](docs/cli.md) — Full command reference
- [User Guide](docs/guide.md) — Tutorials and best practices
- [Architecture](docs/architecture.md) — System design and data flow
- [Contributing](CONTRIBUTING.md) — Development guide

## Backends

```bash
# HF Transformers (default, works everywhere)
sped serve run --target meta-llama/Llama-3.1-70B --backend hf

# MLX (Apple Silicon — 2–3× faster on Mac)
uv pip install mlx-lm
sped serve run --target mlx-community/Llama-3.2-3B --backend mlx

# vLLM (production serving)
uv pip install vllm
sped serve run --target meta-llama/Llama-3.1-70B --backend vllm
```

## License

MIT
