# Ollama TurboQuant Fork — Complete Results & Lessons Learned

## Project Summary

Fork of Ollama implementing TurboQuant KV cache compression and related optimizations
for NVIDIA GPUs. Started from the Google TurboQuant paper (ICLR 2026) and iterated
through multiple approaches based on community reference implementations.

**Repository:** https://github.com/nalditopr/ollama/tree/turboquant-support
**Docker:** `gitea.bb.rctechpr.net/localadmin/ollama-turboquant:main-tq6`

---

## What's Working (Production Ready)

### turbo3 KV Cache (Lucien2468 3-bit uniform quantization)
- **Block format:** `d(fp16) + qs[12](3-bit packed)` = 14 bytes / 32 values = 3.5 bpw
- **Quantization:** `round(x/d)` clamped to [-4,3], stored as [0,7] with +4 offset
- **Dequantization:** `(val - 4) * d` — no codebook, no rotation, no WHT
- **Compression:** 4.6x vs FP16
- **Quality:** 24/25 (96%) on full benchmark suite
- **Speed:** 115-134 tok/s generation, 3,000-4,000 tok/s prefill
- **Compatibility:** Works on ALL GPUs (A2000, 5060 Ti, 5090)
- **Usage:** `OLLAMA_KV_CACHE_TYPE=turbo3 OLLAMA_FLASH_ATTENTION=true`

### MXFP4 (NVFP4) Weight Quantization
- **Format:** E8M0 shared exponent + 4-bit mantissa per block of 32
- **Compression:** 3.76x vs FP16 (4.25 bpw)
- **Quality:** 14/15 (93%) on Qwen3.5-35B, 7/7 on basic tests
- **Usage:** Import MXFP4 GGUF files directly, or `ollama create -q MXFP4`

### Best Configuration: MXFP4 Weights + turbo3 KV
- **Model:** Huihui-Qwen3.5-35B-A3B-abliterated MXFP4_MOE
- **VRAM:** 24.3 GB (19.6 weights + 1.7 KV + 3.0 compute)
- **Free:** 7.7 GB for extended context
- **Gen speed:** 115-117 tok/s
- **Prefill:** 3,199-3,840 tok/s
- **Max context:** ~150K tokens
- **NIAH:** Passes through 164K tokens (5,000 docs)

### Sparse V Dequantization Skip
- **What:** Skip V dequant for attention positions with softmax weight < 1e-6
- **Where:** All 3 flash attention kernels (vec, tile, MMA)
- **Impact:** Up to +22% decode speed at 32K+ context
- **Quality:** Zero perplexity impact

### Blackwell / RTX 5090 Tuning
- **fattn-vec:** 256 threads (up from 128)
- **MMVQ:** Dedicated parameter table with 8 warps
- **SM 120:** Full CUDA architecture support in Docker builds

### Docker Image
- **Tag:** `gitea.bb.rctechpr.net/localadmin/ollama-turboquant:main-tq6`
- **CUDA architectures:** SM 75/80/86/89/90/120
- **Base:** CUDA 13.1 runtime Ubuntu 24.04
- **Includes:** All turbo3, MXFP4, sparse V optimizations

---

## Benchmark Results

### NVFP4 + turbo3 KV — Full Quality (Qwen3.5-35B-A3B MXFP4, RTX 5090)

| Category | Score | Details |
|----------|-------|---------|
| Math | 5/5 | 391, 1103, 12, 13, 1024 |
| Factual | 5/5 | Paris, Tokyo, Shakespeare, Mercury, 1945 |
| Logic | 3/3 | Syllogism, primality, affirming consequent |
| Code | 4/4 | is_prime, fibonacci, list comprehension, two-sum |
| NIAH | 3/4 | PASS 100/500/1000 docs, FAIL 2000 (think tags) |
| Coherency | 4/4 | Stack/Queue, Hash Tables, TCP/UDP, Closures |
| **Total** | **24/25 (96%)** | |

### NIAH Stress Test (NVFP4 + turbo3 KV)

| Docs | Tokens | Prefill | Gen | NIAH |
|------|--------|---------|-----|------|
| 100 | 3K | 2,902 t/s | 115 t/s | PASS |
| 500 | 16K | 3,176 t/s | 107 t/s | PASS |
| 1,000 | 32K | 3,796 t/s | 97 t/s | PASS |
| 2,000 | 65K | 3,690 t/s | 85 t/s | PASS |
| 5,000 | 164K | 2,766 t/s | 60 t/s | PASS |
| 10,000 | 262K | 1,933 t/s | 1 t/s | FAIL (context limit) |

### Comparison: turbo3 vs FP16 vs Q4_0 KV

| KV Type | Quality | Prefill | Gen | Compression |
|---------|---------|---------|-----|-------------|
| FP16 | 10/10 | 2,692 t/s | 49 t/s | 1.0x |
| Q4_0 | 10/10 | 3,574 t/s | 131 t/s | 4.0x |
| turbo3 | 10/10 | 3,950 t/s | 134 t/s | 4.6x |

### Speed Benchmarks (Qwen3.5-35B-A3B, Cluster 5060 Ti)

| Test | Tokens | Speed |
|------|--------|-------|
| tg128 | 128 | 76 tok/s |
| tg512 | 512 | 74 tok/s |
| tg2048 | 2,048 | 66 tok/s |
| pp128 | 106 | 630 tok/s |
| pp512 | 384 | 1,389 tok/s |
| pp2048 | 1,498 | 1,647 tok/s |
| pp8192 | 5,954 | 1,537 tok/s |
| pp16384 | 11,895 | 1,320 tok/s |

---

## What's NOT Working

### CPU Offload for 100B+ MoE Models
- **Symptom:** `ggml-cpu.c:2418: fatal error` during compute
- **Affected:** Qwen3.5-122B-A10B, GPT-OSS-120B (any model needing CPU layers)
- **Root cause:** CPU backend crash in MoE expert dispatch, not TurboQuant-related
- **Workaround:** Use models that fit entirely on GPU (≤32 GB)

### turbo4 KV Cache
- **Symptom:** 5/10 quality on Qwen3.5-35B
- **Root cause:** QK=256 block size doesn't divide into head_dim=128
- **Status:** Works only for models with head_dim≥256 (rare)

### NVFP4 Safetensors → GGUF Conversion
- **Symptom:** `nvfp4-pack-quantized format not yet supported`
- **Root cause:** llama.cpp convert script doesn't support RedHatAI's compressed-tensors format
- **Workaround:** Use pre-built MXFP4 GGUFs from noctrex/bartowski

### WHT-Based TurboQuant (Original Paper Approach)
- **Symptom:** Corrupted output on A2000, complex code, slower than simple approach
- **Root cause:** `__constant__` memory collisions on smaller GPUs, head_dim alignment issues
- **Resolution:** Replaced with Lucien2468 simple 3-bit uniform quantization (no WHT)

### MXFP4 as KV Cache Type
- **Symptom:** Crashes — no `set_rows` quantize kernel
- **Status:** Works for weights only, not KV cache
- **Fix needed:** Add MXFP4 quantize-on-write kernel to set_rows.cu

---

## Lessons Learned

### 1. Simple beats complex
The Lucien2468 approach (`round(x/d)` 3-bit uniform) gets 97% of FP16 speed with
~100 lines of code. Our original WHT+centroid approach was 1000+ lines and slower.
The Google paper's PolarQuant + QJL is theoretically optimal but practically
unnecessary for KV cache where the data is already well-distributed.

### 2. Flash attention integration is critical
Disabling FA for turbo3 (animehacker approach) costs 50%+ generation speed.
The Lucien/PR#15 approach enables FA via graph-level F32 cast — almost free.

### 3. Block size must divide head_dim
QK_K=256 blocks don't work with head_dim=128 (most models). QK=32 is universal.
This killed turbo4 and our early QK_K=256 TQ3_0/TQ4_0 implementations.

### 4. `__constant__` memory is dangerous in headers
Static `__constant__` arrays in `.cuh` headers included by multiple `.cu` files
can cause memory corruption on GPUs with smaller constant caches (A2000).
Use `__device__` or local arrays instead.

### 5. CPU offload for MoE is broken
Large MoE models (100B+) crash when layers are offloaded to CPU in our Ollama fork.
This is a CPU backend issue, not quantization-related. Limits 120B models to
multi-GPU setups.

### 6. The `think` parameter matters
Qwen3.5 models use a thinking mode that consumes output tokens. Always use
`"think": false` in the API for benchmarks and NIAH tests. The `/no_think`
prompt prefix doesn't work reliably.

### 7. MXFP4 weight quantization is free performance
MXFP4 (NVFP4) weights save 2-5 GB vs Q4_K_M with minimal quality loss.
On VRAM-constrained setups, this is the difference between fitting and not fitting.

### 8. Docker builds need explicit CUDA architectures
The default NVIDIA CUDA Docker images don't include SM 120 (Blackwell).
Must pass `-DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;120"` explicitly.

### 9. Ollama's `SupportsKVCacheType` is a gatekeeper
Custom KV types must be added to the allowlist in `fs/ggml/ggml.go`.
Missing this causes silent fallback to FP16 with no error message.

### 10. The community moves fast
In one session we went through 5+ reference implementations:
- animehacker/llama-turboquant (WHT + fused Q rotation)
- TheTom/turboquant_plus (sparse V, Metal focus)
- Lucien2468/Ollama-TurboQuant-Integration (simple 3-bit, winner)
- Alberto-Codes/turboquant-vllm (incremental dequant)
- RemizovDenis/turboquant (MoE expert caching)
- peva3/turboquant-h2o-streamingllm (H2O + eviction)
Each had useful ideas; the simplest approach won for production.

---

## Architecture Overview

### Type System (ggml.h)
```
GGML_TYPE_TQ3_0     = 40  [experimental] Weight quant, QK_K=256
GGML_TYPE_TQ4_0     = 41  [experimental] Weight quant, QK_K=256
GGML_TYPE_TQ3_0_WHT = 42  [experimental] WHT KV cache, QK_K=256
GGML_TYPE_TQ4_0_WHT = 43  [experimental] WHT KV cache, QK_K=256
GGML_TYPE_TQ3_KV    = 44  [deprecated]   animehacker KV, QK=32
GGML_TYPE_TURBO3_0  = 45  [production]   Lucien 3-bit KV, QK=32
GGML_TYPE_TURBO4_0  = 46  [experimental] 4-bit KV, QK=256 (head_dim issue)
```

### Key Files Modified
- `ml/backend/ggml/ggml/src/ggml-cuda/dequantize.cuh` — turbo3 dequant
- `ml/backend/ggml/ggml/src/ggml-cuda/cpy-utils.cuh` — turbo3 quantize-on-write
- `ml/backend/ggml/ggml/src/ggml-cuda/set-rows.cu` — KV cache write dispatch
- `ml/backend/ggml/ggml/src/ggml-cuda/fattn-vec.cuh` — sparse V skip
- `ml/backend/ggml/ggml/src/ggml-turbo-quant.c` — CPU reference implementation
- `fs/ggml/ggml.go` — Go type system + SupportsKVCacheType
- `runner/ollamarunner/cache.go` — KV cache type string mapping

### Total Changes
- ~50 files modified across 20+ commits
- ~5,000 lines of new code
- 6 new CUDA template instance files
- Docker build infrastructure (Dockerfile, CI, k8s manifests)
- Quality test suite with NIAH stress testing

---

## Recommended Configuration

### Local (RTX 5090, 32 GB)
```bash
OLLAMA_KV_CACHE_TYPE=turbo3 OLLAMA_FLASH_ATTENTION=true ./ollama serve
# Model: Huihui-Qwen3.5-35B-A3B MXFP4 (24.3 GB VRAM, 150K+ context)
```

### Cluster (A2000 8GB + 5060 Ti 16GB = 24 GB)
```bash
# Docker: gitea.bb.rctechpr.net/localadmin/ollama-turboquant:main-tq6
# Model: Qwen3-Coder-30B-A3B Q4_K_M (19.4 GB, fits both GPUs)
# KV: turbo3 for max context, q4_0 as safe fallback
OLLAMA_KV_CACHE_TYPE=turbo3 OLLAMA_FLASH_ATTENTION=true
```

---

## Reference Implementations Studied

| Repo | Key Contribution | Used? |
|------|-----------------|-------|
| [animehacker/llama-turboquant](https://github.com/animehacker/llama-turboquant) | WHT + fused Q rotation, FA disabled | Ported then replaced |
| [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) | Sparse V skip, Metal kernels | Sparse V adopted |
| [TheTom/llama-cpp-turboquant PR#15](https://github.com/TheTom/llama-cpp-turboquant/pull/15) | Lucien 3-bit uniform, 97% FP16 speed | **Winner — adopted** |
| [Lucien2468/Ollama-TurboQuant-Integration](https://github.com/Lucien2468/Ollama-TurboQuant-Integration) | Simple Ollama integration | Design reference |
| [Alberto-Codes/turboquant-vllm](https://github.com/Alberto-Codes/turboquant-vllm) | Incremental dequant, Triton kernels | Future reference |
| [RemizovDenis/turboquant](https://github.com/RemizovDenis/turboquant) | MoE expert caching, adaptive bitwidth | Future reference |
| [peva3/turboquant-h2o-streamingllm](https://github.com/peva3/turboquant-h2o-streamingllm) | H2O eviction + attention sinks | Future reference |
| [nalditopr/llama-cpp-turboquant](https://github.com/nalditopr/llama-cpp-turboquant) | CUDA port, RotorQuant optimizations | Self-reference |
