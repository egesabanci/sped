# Deferred Ideas

## Structural changes (pursue after autoresearch if interested)

### 1. Target hidden-state caching (38× smaller than logit cache)
- Cache target.model(…) → hidden states (1, L, 4096) instead of lm_head(…) → logits (1, L, 151936)
- Hidden states: 33.5 MB/example at L=4096 vs 1.24 GB for logits
- On reuse: cache hit → apply lm_head (fast, ~72ms at L=4096)
- Scales to ~450 examples at 16 GB CPU RAM vs only ~13 for logits
- Requires: separating model forward from lm_head, handling Unsloth patching, LRU cache eviction

### 2. Slimmer target: use early exit layers (first K of 36)
- Forward through only first 20 layers of the 36-layer target
- ~55% of target forward compute
- Distillation quality may degrade (weaker teacher)
- Requires: modifying the target forward to stop at intermediate layer

### 3. Disk-backed target logits cache
- Store pre-computed target logits on NVMe/SSD
- Read asynchronously with background thread (overlap I/O with compute)
- Requires: file format, prefetch logic, ~2.3 TB for 5k examples at full seq_len

### 4. CUDA graph capture for target forward
- If input shapes are fixed, capture target forward as a CUDA graph
- Eliminates Python→CUDA kernel launch overhead
- Requires: static shapes (pre-tokenized, padded), very tricky with 4-bit

## Micro-optimizations (explored, not worth pursuing)
- ❌ fused AdamW (regression for small tensors)
- ❌ torch.compile (breaks with 4-bit or gives tiny gains)
- ❌ bf16 autocast on draft (Unsloth handles precision internally)
- ❌ gradient checkpointing off (crashes Unsloth model)
