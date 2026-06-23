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

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--draft` | `-d` | required | Draft model ID or path (e.g. `Qwen/Qwen2.5-0.5B`) |
| `--target` | `-t` | required | Target model ID or path |
| `--dataset` | | required | Hugging Face dataset ID |
| `--text-column` | | `text` | Column containing text |
| `--lora-rank` | `-r` | `8` | LoRA rank (1–64) |
| `--lora-alpha` | `-a` | `16` | LoRA alpha (1–128) |
| `--epochs` | `-e` | `3` | Number of epochs (1–100) |
| `--batch-size` | `-b` | `4` | Batch size per GPU |
| `--learning-rate` | `-lr` | `5e-5` | Learning rate |
| `--max-length` | `-ml` | `512` | Max token length (64–8192) |
| `--temperature` | `-T` | `1.0` | Distillation temperature |
| `--output` | `-o` | `./draft-lora` | Output directory |
| `--device` | | `auto` | Device: auto, cuda, cpu, mps |

### `sped distil validate`

Validate a trained draft adapter.

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--draft` | `-d` | required | Draft model or adapter path |
| `--target` | `-t` | required | Target model |
| `--num-prompts` | `-n` | `100` | Validation prompts |
| `--draft-k` | `-k` | `5` | Draft tokens per step |

---

## `sped serve`

Run inference with speculative decoding.

### `sped serve run`

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--target` | `-t` | required | Target model ID or path |
| `--draft` | `-d` | none | Draft model (omit for standard mode) |
| `--draft-lora` | | none | Path to LoRA adapter |
| `--backend` | `-b` | `auto` | Backend: auto, hf, mlx, vllm |
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
