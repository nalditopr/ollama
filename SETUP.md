# Ollama TurboQuant Fork — Setup Guide

Custom Ollama build with TurboQuant quantization types and sparse V flash attention optimization.

## Features

- **TQ3_0 / TQ4_0** — TurboQuant quantization types (PolarQuant + QJL sign correction)
- **Sparse V skip** — Skips V dequantization for negligible attention weights in flash attention. At 32K context, ~90% of V dequant is skipped for +22% decode speed. Works with any KV cache quantization.
- **Q4_0 KV cache** — Enables 262K context on 32GB GPUs with models like Qwen3.5 35B-A3B

## Prerequisites

### Linux
```bash
# Ubuntu/Debian
sudo apt install build-essential cmake golang

# CUDA (optional, for GPU acceleration)
# Install from https://developer.nvidia.com/cuda-downloads
```

### Windows
- [Go](https://go.dev/dl/) 1.21+
- [CMake](https://cmake.org/download/)
- [Visual Studio 2022](https://visualstudio.microsoft.com/) (Community, with C++ workload)
- [TDM-GCC](https://github.com/jmeubank/tdm-gcc/releases) or WinLibs MinGW-w64
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (optional)

## Build

### Linux
```bash
git clone https://github.com/YOUR_USER/ollama-turboquant.git
cd ollama-turboquant
chmod +x build_linux.sh
./build_linux.sh
```

### Windows
```cmd
build_ollama.bat
```

## Run

### Start the server
```bash
# Linux
OLLAMA_KV_CACHE_TYPE=q4_0 OLLAMA_FLASH_ATTENTION=true ./ollama serve

# Windows
set OLLAMA_KV_CACHE_TYPE=q4_0
set OLLAMA_FLASH_ATTENTION=true
ollama.exe serve
```

### Pull and run a model
```bash
./ollama pull qwen3.5:35b-a3b
./ollama run qwen3.5:35b-a3b
```

### Max context (262K on 32GB GPU)
```bash
OLLAMA_KV_CACHE_TYPE=q4_0 OLLAMA_FLASH_ATTENTION=true ./ollama serve
# In another terminal:
./ollama run qwen3.5:35b-a3b --ctx-size 262144
```

## Run tests
```bash
# Linux
chmod +x tests/build_test_linux.sh
./tests/build_test_linux.sh

# Windows
tests\build_test.bat
```

## Benchmarks (RTX 5090, Qwen3.5 35B-A3B MoE Q4_K_M)

| Context Fill | Prompt Speed | Gen Speed | NIAH |
|-------------|-------------|-----------|------|
| ~10K tokens | 2,967 tok/s | 63 tok/s | PASS |
| ~68K tokens | 2,712 tok/s | 55 tok/s | PASS |
| ~133K tokens | 2,416 tok/s | 46 tok/s | PASS |
| ~233K tokens | 1,982 tok/s | 39 tok/s | PASS |

## Technical Details

### Sparse V Dequantization
During flash attention, softmax weights are computed before V accumulation. At long context, most weights are near-zero (< 1e-6). The sparse V optimization skips V dequantization entirely for these positions, eliminating ~90% of dequant work at 32K+ context with zero quality loss.

Reference: [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)

### TurboQuant Block Structures
- **TQ3_0**: 3.0625 bpw — 2-bit angle index + 1-bit QJL sign, 256-element super-blocks (98 bytes)
- **TQ4_0**: 4.0625 bpw — 3-bit angle index + 1-bit QJL sign, 256-element super-blocks (130 bytes)

Note: TQ weight quantization is experimental. For production use, standard Q4_K_M weights with Q4_0 KV cache + sparse V skip is recommended. Full TurboQuant KV cache (with WHT rotation per the paper) is future work.
