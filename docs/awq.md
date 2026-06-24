# AWQ-Quantized Models with sped

This guide covers using AWQ (Activation-aware Weight Quantization) models with `sped` for efficient inference.

## What is AWQ?

AWQ is a post-training quantization method that reduces model precision from 16-bit to 4-bit while preserving accuracy. This gives:

- **~4x smaller memory footprint** (e.g., 7B model from ~14GB to ~3.5GB)
- **2-3x faster inference** on GPUs
- **Minimal accuracy loss** (< 1% on benchmarks)

## Loading AWQ Models

sped automatically detects AWQ models from their `quantize_config.json`:

```bash
# Auto-detect AWQ from model config
sped serve run --target Qwen/Qwen3-4B-AWQ --device cuda

# Explicit quantize flag (for non-AWQ models)
sped serve run --target Qwen/Qwen3-4B --quantize awq --device cuda
```

### Supported Quantization Methods

| Flag | Method | Library | Auto-detect? |
|------|--------|---------|-------------|
| `--quantize awq` | AWQ | `autoawq` | Yes (from config) |
| `--quantize gptq` | GPTQ | `auto-gptq` | Yes (from config) |
| `--quantize 4bit` | bitsandbytes 4-bit | `bitsandbytes` | No |
| `--quantize 8bit` | bitsandbytes 8-bit | `bitsandbytes` | No |
| `(default)` | None (FP16) | — | — |

### Local AWQ Models

```bash
# Load AWQ model from local path
sped serve run --target /path/to/awq-model/ --device cuda

# Get model info without generating
sped info --model /path/to/awq-model/ --output json
```

## Speculative Decoding with AWQ

Pair an AWQ-quantized target model with a small draft model for 2-5x speedup:

```bash
# Target: 4-bit 4B model, Draft: 0.6B model
sped serve run \
  --target Qwen/Qwen3-4B-AWQ \
  --draft Qwen/Qwen3-0.6B \
  --backend hf \
  --device cuda \
  --draft-k 5 \
  --max-new-tokens 128 \
  --prompt "What is the capital of France?"
```

### Why It Works Well

- **Target model**: Large, accurate, but slow — AWQ reduces its memory 4x
- **Draft model**: Small and fast — runs at near-native speed
- **Speculation**: Draft proposes tokens, target verifies in parallel — net 2-5x speedup
- **Zero quality loss**: Rejection sampling guarantees output matches target model exactly

## Benchmarking Quantized Models

```bash
sped serve run \
  --target Qwen/Qwen3-4B-AWQ \
  --draft Qwen/Qwen3-0.6B \
  --device cuda \
  --benchmark \
  --max-new-tokens 32 \
  --results-dir ./benchmark/
```

## Memory Comparison

| Model | Precision | Size | Speedup vs FP16 |
|-------|-----------|------|-----------------|
| Qwen3-0.6B | FP16 | 1.2 GB | — |
| Qwen3-4B | FP16 | 8.0 GB | 1× |
| Qwen3-4B | 8-bit | 4.0 GB | ~1.2× |
| Qwen3-4B | 4-bit (AWQ) | 2.0 GB | ~2× |

## Finding AWQ Models

Browse models with AWQ support on HuggingFace:

```bash
# List locally cached models
sped list models --local

# Search for AWQ models (requires network)
sped list models --filter awq
```

Popular AWQ models (available on HuggingFace):

- `Qwen/Qwen3-4B-AWQ`
- `Qwen/Qwen2.5-7B-Instruct-AWQ`
- `TheBloke/Llama-2-7B-Chat-AWQ`
- `TheBloke/Mistral-7B-Instruct-v0.2-AWQ`

## Tips for EC2

1. **Always use CUDA** — AWQ requires GPU for the speed benefits
2. **Use `--dtype float16`** for draft model (draft is small, doesn't need quantization)
3. **Set `--log-level info`** to monitor memory usage
4. **Use `--results-dir`** to save all generation outputs
5. **Pair with a much smaller draft** (e.g., 0.5B draft with 4B target)

For production deployment, see the [EC2 guide](ec2.md).
