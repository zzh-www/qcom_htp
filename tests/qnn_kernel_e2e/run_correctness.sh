#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

if [ "$#" -ne 0 ]; then
  echo "Usage: tests/qnn_kernel_e2e/run_correctness.sh" >&2
  exit 2
fi

echo "=== qnn correctness ci: u8i8 custom/native exactness ==="
"$ROOT_DIR/tests/qnn_kernel_e2e/correctness/test_u8i8_native_match_e2e.sh"

echo "=== qnn correctness ci: w4a8 per-channel custom/native exactness ==="
"$ROOT_DIR/tests/qnn_kernel_e2e/correctness/test_w4a8_per_channel_native_match_e2e.sh"

echo "=== qnn correctness ci: w4a8 lpbq custom/native exactness ==="
"$ROOT_DIR/tests/qnn_kernel_e2e/correctness/test_w4a8_lpbq_native_match_e2e.sh"

echo "=== qnn correctness ci: w8a16 per-channel custom/native exactness ==="
"$ROOT_DIR/tests/qnn_kernel_e2e/correctness/test_w8a16_per_channel_native_match_e2e.sh"

echo "=== qnn correctness ci: w4a16 per-channel custom/native exactness ==="
"$ROOT_DIR/tests/qnn_kernel_e2e/correctness/test_w4a16_per_channel_native_match_e2e.sh"

echo "=== qnn correctness ci: w4a16 lpbq custom/native exactness ==="
"$ROOT_DIR/tests/qnn_kernel_e2e/correctness/test_w4a16_lpbq_native_match_e2e.sh"

echo "=== qnn correctness ci: w4a16 per-channel chain1 precision ==="
"$ROOT_DIR/tests/qnn_kernel_e2e/correctness/test_w4a16_per_channel_chain1_e2e.sh"

echo "=== qnn correctness ci: w16a16 == two w8a16 (byte decomposition gate) ==="
"$ROOT_DIR/tests/qnn_kernel_e2e/correctness/test_w16a16_two_w8a16_e2e.sh"

echo "=== handwritten hmx matmul correctness ci ==="
"$ROOT_DIR/tests/qnn_kernel_e2e/correctness/test_handwritten_hmx_matmul_e2e.sh"

echo "=== gdn triangular-inverse (GDNSolveHVXMixHMX) precision gate ==="
"$ROOT_DIR/tests/qnn_kernel_e2e/correctness/test_gdn_solve_e2e.sh"

echo "=== gdn triangular-inverse (GDNSolveHMX pure-HMX) precision gate ==="
"$ROOT_DIR/tests/qnn_kernel_e2e/correctness/test_gdn_pure_hmx_e2e.sh"
