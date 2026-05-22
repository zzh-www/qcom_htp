#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== correctness ci ==="
"$ROOT_DIR/tests/qnn_kernel_e2e/run_correctness.sh"

echo "=== performance ci ==="
"$ROOT_DIR/tests/qnn_kernel_e2e/run_performance.sh"
