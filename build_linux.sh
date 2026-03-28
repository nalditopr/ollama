#!/bin/bash
set -e

# Build Ollama with TurboQuant + Sparse V skip on Linux
# Prerequisites: Go 1.21+, GCC/Clang, CMake 3.22+, CUDA Toolkit (optional)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Detect CUDA
CUDA_FLAGS=""
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9.]+')
    echo "=== CUDA $CUDA_VERSION detected ==="
    CUDA_FLAGS="-DGGML_CUDA=ON"
else
    echo "=== No CUDA found, building CPU-only ==="
fi

# Step 1: CMake configure
echo "=== Step 1: CMake Configure ==="
cmake -B build $CUDA_FLAGS

# Step 2: CMake build
echo "=== Step 2: CMake Build ==="
cmake --build build --config Release -j$(nproc)

# Step 3: Go build
echo "=== Step 3: Go Build ==="
CGO_ENABLED=1 go build -o ollama .

echo "=== BUILD COMPLETE ==="
./ollama --version
echo ""
echo "Usage:"
echo "  # Start server"
echo "  OLLAMA_MODELS=/path/to/models ./ollama serve"
echo ""
echo "  # Pull and run a model"
echo "  ./ollama pull qwen3.5:35b-a3b"
echo "  ./ollama run qwen3.5:35b-a3b"
echo ""
echo "  # Run with Q4_0 KV cache for max context"
echo "  OLLAMA_KV_CACHE_TYPE=q4_0 OLLAMA_FLASH_ATTENTION=true ./ollama serve"
