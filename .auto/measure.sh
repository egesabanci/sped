#!/bin/bash
set -euo pipefail
# Autoresearch measure script for sped training throughput
# Run from /data/sped (or workingDir)
# Outputs METRIC lines parsed by run_experiment

source /data/unsloth_env/bin/activate
cd /data/sped

python3 << 'PYEOF'
import sys, os, time, torch, gc
sys.path.insert(0, '/data/sped')
os.environ['UNSLOTH_USE_NEW_MODEL'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

torch.cuda.empty_cache()
gc.collect()

from unsloth import FastLanguageModel
from datasets import load_from_disk
from sped.distillation.distillspec import DistillSpec

# ── Load models ─────────────────────────────────────────────────────
draft, tok = FastLanguageModel.from_pretrained(
    '/data/models/Qwen3-1.7B-4bit-cache', max_seq_length=4096,
    load_in_4bit=True, device_map='cuda',
)
target, _ = FastLanguageModel.from_pretrained(
    '/data/models/Qwen3-8B-4bit-cache', max_seq_length=4096,
    load_in_4bit=True, device_map='cuda',
)

ds = load_from_disk('/data/ultrachat_200k_eval10')

spec = DistillSpec(
    draft, tok, target, tok,
    lora_rank=32, lora_alpha=32, device='cuda', backend='unsloth',
)

# Pre-tokenize
tokenized = []
for i in range(len(ds)):
    text = ds[i]['text']
    encoded = tok(text, truncation=True, max_length=4096, return_tensors='pt')
    tokenized.append({'input_ids': encoded.input_ids[0]})

# ── Training loop ───────────────────────────────────────────────────
t_start = time.time()
total_tokens = 0
total_steps = 0

opt = torch.optim.AdamW(spec.draft_model.parameters(), lr=5e-5, fused=True)

for step, batch_data in enumerate(tokenized):
    batch = DistillSpec._collate_batch([batch_data])
    input_ids = batch['input_ids'].cuda()
    attn_mask = batch['attention_mask'].cuda()
    n_tokens = attn_mask.sum().item()
    total_tokens += n_tokens
    total_steps += 1

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        tl = target(input_ids, attention_mask=attn_mask).logits
    do = spec.draft_model(input_ids, attention_mask=attn_mask).logits
    loss = DistillSpec._kl_divergence(do, tl, 1.0, attn_mask)
    loss.backward()
    opt.step()
    opt.zero_grad()

t_end = time.time()
elapsed = t_end - t_start
vram_peak = torch.cuda.max_memory_allocated() / 1e9
vram_cur = torch.cuda.memory_allocated() / 1e9
tokens_per_sec = total_tokens / elapsed
step_time_ms = elapsed / total_steps * 1000

print(f"\n=== RESULTS ===")
print(f"Steps: {total_steps}, Tokens: {total_tokens}")
print(f"Time: {elapsed:.1f}s")
print(f"Tokens/sec: {tokens_per_sec:.0f}")
print(f"Step time: {step_time_ms:.0f}ms")
print(f"VRAM peak: {vram_peak:.1f}GB, cur: {vram_cur:.1f}GB")

print(f"METRIC tokens_per_sec={tokens_per_sec:.0f}")
print(f"METRIC step_time_ms={step_time_ms:.0f}")
print(f"METRIC vram_gb={vram_peak:.1f}")
print(f"METRIC mean_seq_len={total_tokens/total_steps:.0f}")
PYEOF
