#!/bin/bash
set -euo pipefail
source /data/unsloth_env/bin/activate
cd /data/sped

# Measure cache-hit throughput (epoch 2+ behavior at batch_size=1)
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

draft, tok = FastLanguageModel.from_pretrained(
    '/data/models/Qwen3-1.7B-4bit-cache', max_seq_length=4096,
    load_in_4bit=True, device_map='cuda',
)
target, _ = FastLanguageModel.from_pretrained(
    '/data/models/Qwen3-8B-4bit-cache', max_seq_length=4096,
    load_in_4bit=True, device_map='cuda',
)
ds = load_from_disk('/data/ultrachat_200k_eval10')

spec = DistillSpec(draft, tok, target, tok, lora_rank=32, lora_alpha=32, device='cuda', backend='unsloth')

# Pre-tokenize
tokenized = []
for i in range(len(ds)):
    text = ds[i]['text']
    encoded = tok(text, truncation=True, max_length=4096, return_tensors='pt')
    tokenized.append({'input_ids': encoded.input_ids[0]})

# Warmup: populate hidden-state cache (simulating epoch 1 cache writes)
for batch_data in tokenized:
    batch = DistillSpec._collate_batch([batch_data])
    _ = spec._get_target_logits(batch['input_ids'].cuda(), batch['attention_mask'].cuda())
print(f"Cache: {len(spec._target_hidden_cache)} entries")

# Timed epoch: all cache hits (simulating epoch 2+)
opt = torch.optim.AdamW(spec.draft_model.parameters(), lr=5e-5)
t_start = time.time()
total_tokens = 0
total_steps = 0

for batch_data in tokenized:
    batch = DistillSpec._collate_batch([batch_data])
    input_ids = batch['input_ids'].cuda()
    attn_mask = batch['attention_mask'].cuda()
    n_tokens = attn_mask.sum().item()
    total_tokens += n_tokens
    total_steps += 1

    # Cache hit: loads hidden from CPU→GPU, computes lm_head only
    tl = spec._get_target_logits(input_ids, attn_mask)
    do = spec.draft_model(input_ids, attention_mask=attn_mask).logits
    loss = DistillSpec._kl_divergence(do, tl, 1.0, attn_mask)
    loss.backward()
    opt.step()
    opt.zero_grad()

t_end = time.time()
elapsed = t_end - t_start
vram = torch.cuda.max_memory_allocated() / 1e9
tok_s = total_tokens / elapsed
step_ms = elapsed / total_steps * 1000
mean_len = total_tokens / total_steps

print(f"\n=== RESULTS (CACHE HIT — epoch 2+, bs=1) ===")
print(f"Steps: {total_steps}, Tokens: {total_tokens}, Time: {elapsed:.1f}s")
print(f"Tokens/sec: {tok_s:.0f}, Step time: {step_ms:.0f}ms, VRAM: {vram:.1f}GB")
print(f"NOTE: target forward skipped via hidden-state cache hit")

print(f"METRIC tokens_per_sec={tok_s:.0f}")
print(f"METRIC step_time_ms={step_ms:.0f}")
print(f"METRIC vram_gb={vram:.1f}")
print(f"METRIC mean_seq_len={mean_len:.0f}")
PYEOF
