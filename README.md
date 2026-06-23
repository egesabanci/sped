# ⚡ sped — Universal Speculative Decoding

**sped** is a CLI toolkit for universal speculative decoding. It lets you pair
**any small draft model** with **any large target model** — even if they use
completely different tokenizers — and accelerate inference without any loss in
output quality.

## Features

- **🔀 Vocabulary-agnostic** — Draft and target models can use different
  tokenizers. Powered by Intel/Weizmann heterogeneous SD algorithms.
- **🎯 PEFT distillation** — Align any tiny draft model to any target model
  with LoRA in under an hour on a single GPU.
- **🔄 Online adaptation** — Draft keeps improving during inference via OSD.
- **⚡ Universal** — Works with any Hugging Face model out of the box.
- **🧪 Experiment CLI** — Quickly test draft-target pairs, measure acceptance
  rates, and tune hyperparameters.

## Quick Start

```bash
# Install
pip install sped

# Distill a tiny draft to your target
sped distill \
  --draft Qwen/Qwen2.5-0.5B \
  --target meta-llama/Llama-3.1-70B \
  --lora-rank 8

# Run inference with speculation
sped serve \
  --target meta-llama/Llama-3.1-70B \
  --draft ./distilled-draft \
  --speedup 2.5x
```

## How It Works

```
┌─────────────────────────────────────────────────────┐
│  Tiny Draft Model (0.5B)                            │
│    └─ PEFT (LoRA) to mimic target distribution      │
│    └─ Proposes K tokens per step                    │
└──────────┬──────────────────────────────────────────┘
           │ draft tokens (possibly different vocab)
           ▼
┌─────────────────────────────────────────────────────┐
│  Vocab-Agnostic Alignment Layer                     │
│    └─ Maps draft tokens → target token space        │
│    └─ Intel/Weizmann heterogeneous SD algorithms     │
└──────────┬──────────────────────────────────────────┘
           │ aligned draft tokens
           ▼
┌─────────────────────────────────────────────────────┐
│  Large Target Model (70B)                           │
│    └─ Verifies all K tokens in ONE forward pass     │
│    └─ Rejection sampling → lossless acceleration     │
└──────────┬──────────────────────────────────────────┘
           │ online feedback
           ▼
┌─────────────────────────────────────────────────────┐
│  Online Adapter (OSD)                               │
│    └─ Updates LoRA weights based on accept/reject   │
│    └─ Draft gets better over time                   │
└─────────────────────────────────────────────────────┘
```

## Roadmap

- [x] Project scaffolding
- [ ] Core speculative decoding engine
- [ ] Vocabulary-agnostic alignment (Intel/Weizmann)
- [ ] PEFT distillation pipeline (DistillSpec)
- [ ] Online adaptation (OSD)
- [ ] `sped serve` — production inference server
- [ ] `sped experiment` — interactive experiment runner

## License

MIT
