#!/bin/bash
set -euo pipefail
source /data/unsloth_env/bin/activate
cd /data/sped

python3 << 'PYEOF'
import sys, os, time, torch, gc
sys.path.insert(0, '/data/sped')
os.environ['UNSLOTH_USE_NEW_MODEL'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings; warnings.filterwarnings('ignore')
import logging; logging.disable(logging.CRITICAL)
torch.cuda.empty_cache(); gc.collect()

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

tokenized = []
for i in range(len(ds)):
    text = ds[i]['text']; encoded = tok(text, truncation=True, max_length=4096, return_tensors='pt')
    tokenized.append({'input_ids': encoded.input_ids[0]})
for batch_data in tokenized:
    batch = DistillSpec._collate_batch([batch_data])
    _ = spec._get_target_logits(batch['input_ids'].cuda(), batch['attention_mask'].cuda())

opt = torch.optim.AdamW(spec.draft_model.parameters(), lr=5e-5)
t_start = time.time()
ttok = 0; st = 0
for batch_data in tokenized:
    batch = DistillSpec._collate_batch([batch_data])
    ids = batch['input_ids'].cuda(); am = batch['attention_mask'].cuda()
    ttok += am.sum().item(); st += 1
    tl = spec._get_target_logits(ids, am)
    do = spec.draft_model(ids, attention_mask=am, use_cache=False).logits
    loss = DistillSpec._kl_divergence(do, tl, 1.0, am); loss.backward()
    opt.step(); opt.zero_grad()
t = time.time() - t_start
tok_s = ttok / t
vram = torch.cuda.max_memory_allocated() / 1e9
step_ms = t / st * 1000

print(f"METRIC tokens_per_sec={tok_s:.0f}")
print(f"METRIC step_time_ms={step_ms:.0f}")
print(f"METRIC vram_gb={vram:.1f}")
print(f"METRIC mean_seq_len={ttok/st:.0f}")
PYEOF
