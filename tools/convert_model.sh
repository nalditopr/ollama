#!/bin/bash
#
# Convert a model to TurboQuant format using Ollama's built-in quantization pipeline.
#
# Usage:
#   ./tools/convert_model.sh <model-name> <quant-type>
#
# Examples:
#   ./tools/convert_model.sh llama3.1:8b TQ4_0
#   ./tools/convert_model.sh llama3.1:8b TQ3_0
#   ./tools/convert_model.sh mistral:7b TQ4_0
#
# Prerequisites:
#   - The Ollama server must be running (ollama serve)
#   - The source model must already be pulled (ollama pull <model>)
#   - The source model should be F16 or F32 for best results
#
# How it works:
#   Ollama's `create` command with the -q flag triggers the server-side
#   quantization pipeline (server/quantization.go). The pipeline:
#     1. Reads the source model's GGUF tensors
#     2. For each quantizable tensor (2D+ weights, excluding norms/biases):
#        - Dequantizes to float32
#        - Re-quantizes to the target TQ type
#     3. Sensitive tensors (output.weight, token_embd.weight) are kept at
#        higher precision (Q6_K / Q8_0)
#     4. Attention value projections and early/late ffn_down layers get
#        bumped to higher precision for quality preservation
#     5. Writes the output GGUF with updated tensor types and metadata
#

set -euo pipefail

MODEL="${1:?Usage: $0 <model-name> <quant-type>}"
QUANT="${2:?Usage: $0 <model-name> <quant-type>}"

# Validate quant type
case "$QUANT" in
    TQ3_0|TQ4_0) ;;
    *)
        echo "Error: unsupported quant type '$QUANT'. Use TQ3_0 or TQ4_0."
        exit 1
        ;;
esac

# Derive output model name
# e.g., llama3.1:8b -> llama3.1:8b-tq4_0
BASE_NAME="${MODEL%%:*}"
TAG="${MODEL#*:}"
if [ "$TAG" = "$MODEL" ]; then
    TAG="latest"
fi
QUANT_LOWER=$(echo "$QUANT" | tr '[:upper:]' '[:lower:]')
OUTPUT_NAME="${BASE_NAME}:${TAG}-${QUANT_LOWER}"

echo "=== TurboQuant Model Conversion ==="
echo "  Source:  $MODEL"
echo "  Target:  $OUTPUT_NAME"
echo "  Quant:   $QUANT"
echo ""

# Check that ollama is available
if ! command -v ollama &>/dev/null; then
    echo "Error: 'ollama' command not found. Build and install first:"
    echo "  cd $(dirname "$0")/.. && go build -o ollama . && export PATH=\$PWD:\$PATH"
    exit 1
fi

# Check that the source model exists
if ! ollama show "$MODEL" &>/dev/null; then
    echo "Error: model '$MODEL' not found. Pull it first:"
    echo "  ollama pull $MODEL"
    exit 1
fi

echo "Creating quantized model..."
ollama create "$OUTPUT_NAME" --from "$MODEL" -q "$QUANT"

echo ""
echo "Done. Run the model with:"
echo "  ollama run $OUTPUT_NAME"
