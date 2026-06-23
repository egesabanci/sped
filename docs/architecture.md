# sped Architecture

## System Overview

```
CLI Layer (sped/cli/)
├── main.py          — Typer app, command registration
├── distil.py        — DistillSpec training + validation
├── serve.py         — Multi-backend inference server
├── experiment.py    — Grid-search experiments
├── list_cmd.py      — Model/ adapter listing
└── _experiment_engine.py — Testable experiment logic

Core Layer (sped/core/)
├── speculative_decoding.py — Main speculate-verify-accept loop
├── verification.py          — Parallel draft verification
├── rejection_sampling.py    — Metropolis-Hastings sampling
├── kv_cache.py              — KV cache management
├── metrics.py               — Metrics collection
└── draft_tree.py            — Multi-branch draft trees

Vocab-Agnostic Layer (sped/vocab_agnostic/)
├── alignment.py      — 3 Intel/Weizmann alignment algorithms
└── heterogeneous.py  — Cross-vocabulary rejection sampling

Distillation Layer (sped/distillation/)
└── distillspec.py    — PEFT distillation (DistillSpec)

Adaptation Layer (sped/adaptation/)
└── osd.py            — Online speculative decoding

Serving Layer (sped/serving/)
├── base.py            — Abstract backend interface
├── hf_backend.py      — Hugging Face Transformers
├── mlx_backend.py     — Apple Silicon (MLX)
└── vllm_backend.py    — Production vLLM

Utility Layer (sped/utils/)
└── tokenizer_utils.py — Tokenizer compatibility
```

## Data Flow

### Inference with Speculation

```
User prompt
    │
    ▼
┌──────────────────────────────────────────────────┐
│  CLI (sped/cli/serve.py)                         │
│  • Parse arguments, select backend               │
│  • Load target + draft models                    │
│  • Create SpeculativeDecoder                     │
│  • Run REPL / single / benchmark mode            │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────┐
│  SpeculativeDecoder (sped/core/)                 │
│  ┌─────────────┐   ┌─────────────────────────┐   │
│  │ Draft Model │   │ Target Model            │   │
│  │ (small)     │──▶│ (large, parallel verify)│   │
│  │ proposes K  │   │ checks all K in 1 pass  │   │
│  │ tokens      │   │                         │   │
│  └──────┬──────┘   └──────────┬──────────────┘   │
│         │                     │                   │
│         ▼                     ▼                   │
│  ┌─────────────────────────────────────────┐      │
│  │ VocabAligner (if vocabs differ)         │      │
│  │ Intel/Weizmann alignment strategies     │      │
│  └──────────────────┬──────────────────────┘      │
│                     │                              │
│                     ▼                              │
│  ┌─────────────────────────────────────────┐      │
│  │ Rejection Sampling                      │      │
│  │ Metropolis-Hastings accept/reject rule  │      │
│  │ Lossless: output matches target exactly │      │
│  └──────────────────┬──────────────────────┘      │
│                     │                              │
│                     ▼                              │
│  ┌─────────────────────────────────────────┐      │
│  │ MetricsCollector                        │      │
│  │ Tracks acceptance, tok/s, timing        │      │
│  └─────────────────────────────────────────┘      │
│                     │                              │
│                     ▼                              │
│  ┌─────────────────────────────────────────┐      │
│  │ OnlineAdapter (optional)                │      │
│  │ Updates LoRA weights based on feedback  │      │
│  └─────────────────────────────────────────┘      │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
         Generated text + stats
```

### Training Flow (DistillSpec)

```
Dataset (prompts)
    │
    ▼
┌──────────────────────────────────────────────┐
│  DistillSpec (sped/distillation/)            │
│                                              │
│  1. Draft model generates continuations      │
│     (on-policy, uses current weights)        │
│                                              │
│  2. Target model computes logits at          │
│     each position                            │
│                                              │
│  3. KL divergence loss between               │
│     draft_logits || target_logits            │
│                                              │
│  4. Backprop through LoRA params only        │
│     (PEFT — ~1% of weights trainable)        │
│                                              │
│  5. Periodically validate acceptance rate    │
│     on holdout set                           │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
            LoRA adapter weights
            (saved to ./draft-lora/)
```

## Key Design Decisions

### Why PEFT (LoRA) for Distillation?
- Full fine-tuning of a 1B draft model costs significant GPU time
- LoRA trains only ~3–5M params (0.3–1% of model)
- Achieves 80–90% of full fine-tuning acceptance rate improvement
- Single GPU training feasible for 0.5B → 70B pairs

### Why Vocabulary-Agnostic?
- Most real-world deployments mix model families (e.g., Qwen draft + Llama target)
- Training a draft model from scratch for every target is impractical
- Intel/Weizmann algorithms enable lossless cross-vocab speculation

### Why Multiple Backends?
- Hugging Face: universal, works everywhere
- MLX: 2–3× faster on Apple Silicon (unified memory)
- vLLM: production-grade serving with continuous batching

### Why Online Adaptation?
- Draft quality degrades over time as deployment data shifts
- OSD continuously updates LoRA weights based on live acceptance
- Anti-thrash guardrails prevent catastrophic forgetting
