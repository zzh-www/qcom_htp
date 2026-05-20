#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/example/handwritten_hmx_matmul/tutorial_w4a16_qnn_kernel/build}"
DEVICE="${DEVICE:-oneplus}"
REMOTE_DIR="${REMOTE_DIR:-w4a16_qnn_kernel_tutorial}"
JSON_OUT="${JSON_OUT:-$BUILD_DIR/device_result.json}"

cd "$ROOT_DIR"
uv run python scripts/run_w4a16_qnn_kernel_tutorial_device.py \
  --build-dir "$BUILD_DIR" \
  --device "$DEVICE" \
  --remote-dir "$REMOTE_DIR" \
  --json-out "$JSON_OUT"
