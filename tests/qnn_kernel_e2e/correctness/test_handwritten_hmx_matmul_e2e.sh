#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT_ROOT="${HANDWRITTEN_HMX_MATMUL_OUT_ROOT:-${KERNEL_E2E_OUT_ROOT:-/tmp/qcom_htp_kernel_ci}/handwritten_hmx_matmul}"

OUT_ROOT="$OUT_ROOT" \
DEVICE="${DEVICE:-oneplus}" \
ARTIFACT_ONLY="${ARTIFACT_ONLY:-0}" \
    "$ROOT_DIR/tests/qnn_kernel_e2e/handwritten_hmx_matmul/run_all.sh"
