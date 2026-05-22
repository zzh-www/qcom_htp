#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

if [ "$#" -ne 0 ]; then
  echo "Usage: tests/qnn_kernel_e2e/run_performance.sh" >&2
  exit 2
fi

echo "=== qnn performance ci: u8i8 canonical profile/chain8 ==="
"$ROOT_DIR/tests/qnn_kernel_e2e/performance/test_u8i8_e2e.sh"

echo "=== qnn performance ci: w4a8 per-channel canonical profile/chain8 ==="
"$ROOT_DIR/tests/qnn_kernel_e2e/performance/test_w4a8_per_channel_e2e.sh"

echo "=== qnn performance ci: w8a16 canonical profile/chain8 ==="
"$ROOT_DIR/tests/qnn_kernel_e2e/performance/test_w8a16_e2e.sh"

echo "=== qnn performance ci: w4a16 per-channel canonical profile/chain8 ==="
"$ROOT_DIR/tests/qnn_kernel_e2e/performance/test_w4a16_per_channel_e2e.sh"
