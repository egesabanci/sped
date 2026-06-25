# sped CLI Reference

## Global Flags

| Flag | Description |
|------|-------------|
| `--help` | Show help message and exit |
| `--install-completion` | Install shell completion |
| `--show-completion` | Show shell completion script |

---

## `sped version`

Show the sped version.

```bash
sped version
# sped v0.1.0
```

---

## `sped info`

Show system information and available hardware.

```bash
sped info
# ╭──────────────────╮
# │ sped System Info │
# ╰──────────────────╯
#   • PyTorch:     2.12.1
#   • CUDA avail:  False
#   • CPU threads: 4
```

---

## `sped distil`

PEFT distillation of a draft model to a target model.

### `sped distil run`

Run full DistillSpec training pipeline.

#### Core flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--draft` | `-d` | required | Draft model ID or path (e.g. `Qwen/Qwen2.5-0.5B`) |
| `--target` | `-t` | required | Target model ID or path |
| `--dataset` | | required | Hugging Face dataset ID or local `save_to_disk` path |
| `--text-column` | | `text` | Column containing text |
| `--split` | | `auto` | Dataset split: auto, train_sft, train_gen, train, or custom |
| `--lora-rank` | `-r` | `8` | LoRA rank (1–64) |
| `--lora-alpha` | `-a` | `16` | LoRA alpha (1–128) |
| `--epochs` | `-e` | `3` | Number of epochs (1–100) |
| `--batch-size` | `-b` | `4` | Batch size per GPU |
| `--learning-rate` | `-lr` | `5e-5` | Learning rate |
| `--max-length` | `-ml` | `512` | Max token length (64–16384) |
| `--temperature` | `-T` | `1.0` | Distillation temperature |
| `--output` | `-o` | `./draft-lora` | Output directory |
| `--device` | | `auto` | Device: auto, cuda, cpu, mps |
| `--backend` | | `auto` | Backend: auto, hf, unsloth |
| `--draft-dtype` | | `bf16` | Draft precision: bf16 (full) or 4bit |

#### Training tuning flags

| Flag | Default | Description |
|------|---------|-------------|
| `--grad-accum` | `1` | Gradient accumulation steps (effective batch = batch_size × this) |
| `--warmup-steps` | `100` | Linear LR warmup steps |
| `--max-grad-norm` | `1.0` | Gradient clipping max norm |
| `--mixed-precision` | none | Mixed precision: bf16, fp16, or none (auto-detect) |

#### On-policy generation flags

| Flag | Default | Description |
|------|---------|-------------|
| `--on-policy-regen-every` | `200` | Regenerate on-policy data every N steps (0 to disable) |
| `--on-policy-tokens` | `64` | Continuation tokens per prompt (on-policy) |
| `--on-policy-temp` | `0.7` | Generation temperature for on-policy data |

#### Validation flags

| Flag | Default | Description |
|------|---------|-------------|
| `--validation-split` | `0.05` | Fraction of dataset for validation (0 to disable) |
| `--val-prompts` | `20` | Number of prompts for validation |
| `--val-draft-k` | `5` | Draft K for validation |
| `--val-max-new-tokens` | `32` | Max new tokens per validation prompt |

#### Logging & checkpointing flags

| Flag | Default | Description |
|------|---------|-------------|
| `--log-every` | `10` | Log metrics every N steps |
| `--save-every` | `500` | Save checkpoint every N steps (requires `--checkpoint-dir`) |
| `--checkpoint-dir` | none | Directory for checkpoints (enables checkpointing) |
| `--resume-from` | none | Path to a checkpoint directory to resume from |

**Examples:**

```bash
# Train draft with Unsloth (bf16 draft, 4-bit target) — smoke test
sped distil run --backend unsloth --draft-dtype bf16 \
  --draft Qwen/Qwen3-0.6B --target Qwen/Qwen3-8B \
  --dataset ./ultrachat_200k_smoke --max-length 256 \
  --validation-split 0 --epochs 1

# Full training with checkpointing and gradient accumulation
sped distil run --backend unsloth \
  --draft Qwen/Qwen3-0.6B --target Qwen/Qwen3-8B \
  --dataset ./ultrachat_200k_formatted --max-length 4096 \
  --batch-size 1 --grad-accum 4 --epochs 3 \
  --checkpoint-dir ./checkpoints --save-every 500

# Resume from a checkpoint
sped distil run ... --resume-from ./checkpoints/epoch_2
```

> **4-bit caching:** the first load with `--backend unsloth` quantizes the
> target model (bf16 → NF4, ~2 min for 8B) and saves it to a
> `{model}-4bit-cache` directory. Subsequent loads use the cache (~26s).

### `sped distil validate`

Validate a trained draft adapter.

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--draft` | `-d` | required | Draft model or adapter path |
| `--target` | `-t` | required | Target model |
| `--num-prompts` | `-n` | `100` | Validation prompts |
| `--draft-k` | `-k` | `5` | Draft tokens per step |
| `--backend` | | `auto` | Backend: auto, hf, unsloth |
| `--draft-lora` | | none | LoRA adapter path to apply on top of draft |
| `--target-4bit` | | false | (Unsloth) Load target in 4-bit |

---

## `sped serve`

Run inference with speculative decoding.

### `sped serve run`

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--target` | `-t` | required | Target model ID or path |
| `--draft` | `-d` | none | Draft model (omit for standard mode) |
| `--draft-lora` | | none | Path to LoRA adapter |
| `--backend` | `-b` | `auto` | Backend: auto, hf, mlx, vllm, unsloth |
| `--align` | | `auto` | Alignment: auto, none, string, probabilistic, hybrid |
| `--draft-k` | `-k` | `5` | Draft tokens per step (1–20) |
| `--temperature` | `-T` | `0.0` | Sampling temperature |
| `--max-new-tokens` | `-n` | `512` | Max tokens per response |
| `--device` | | `auto` | Device: auto, cuda, cpu, mps |
| `--prompt` | `-p` | none | Single prompt mode |
| `--benchmark` | | false | Run benchmark mode |
| `--quantization` | `-q` | none | Quant: 4bit, 8bit |

**REPL Commands** (in interactive mode):
- `/help` — Show commands
- `/stats` — Show cumulative stats
- `/exit` or `/quit` — Quit
- `/clear` — Clear screen

---

## `sped experiment`

Run grid-search experiments and auto-tune.

### `sped experiment run`

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--target` | `-t` | required | Target model |
| `--draft` | `-d` | required | Draft model |
| `--draft-k-values` | | `3,5,7,10` | Comma-separated K values |
| `--temperatures` | | `0.0,0.7` | Comma-separated temperatures |
| `--align-strategies` | | `none,hybrid` | Alignment strategies |
| `--prompts` | `-p` | none | JSONL prompts file |
| `--num-prompts` | `-n` | `10` | Number of prompts |
| `--max-tokens` | `-m` | `128` | Max tokens per prompt |
| `--output` | `-o` | `./experiment-results` | Output directory |

### `sped experiment auto-tune`

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--target` | `-t` | required | Target model |
| `--draft` | `-d` | required | Draft model |
| `--min-k` | | `2` | Minimum K |
| `--max-k` | | `15` | Maximum K |
| `--temperature` | `-T` | `0.0` | Sampling temperature |
| `--align` | | `auto` | Alignment strategy |
| `--num-prompts` | `-n` | `5` | Prompts per evaluation |

---

## `sped list`

### `sped list models`

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--query` | `-q` | none | Filter by name |
| `--limit` | `-l` | `15` | Max results |

### `sped list adapters`

| Flag | Default | Description |
|------|---------|-------------|
| `--path` | `./draft-lora` | Custom adapter path |

### `sped list pairings`

Shows recommended draft-target model pairings. No flags.
