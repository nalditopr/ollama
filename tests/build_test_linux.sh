#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=== Building test_turboquant ==="
gcc -O2 -Wall -Wextra -o tests/test_turboquant tests/test_turboquant.c -lm
tests/test_turboquant

echo ""
echo "=== Building bench_turboquant ==="
gcc -O2 -Wall -Wextra -o tests/bench_turboquant tests/bench_turboquant.c -lm
tests/bench_turboquant
