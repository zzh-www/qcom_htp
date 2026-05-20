#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
ARTIFACT="${ARTIFACT:-/tmp/handwritten_hmx_matmul_gate_w4a16_device_diag/w4a16}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/example/handwritten_hmx_matmul/tutorial_w4a16_qnn_kernel/build}"
MEASURE_REPEATS="${MEASURE_REPEATS:-1}"
CHAIN_STEPS="${CHAIN_STEPS:-8}"
PRE_CLEAR_ACC="${PRE_CLEAR_ACC:-1}"
NATIVE_WRAPPER_PREFETCH="${NATIVE_WRAPPER_PREFETCH:-0}"
PRELOAD_HMX_IDENTITY_BIAS="${PRELOAD_HMX_IDENTITY_BIAS:-0}"
OUTPUT_SEED_MODE="${OUTPUT_SEED_MODE:-prepared}"
ACTIVATION_RAW_OVERRIDE="${ACTIVATION_RAW_OVERRIDE:-}"
PACKED_WEIGHT_BYTE_OFFSET="${PACKED_WEIGHT_BYTE_OFFSET:-0}"
PACKED_WEIGHT_RAW_OVERRIDE="${PACKED_WEIGHT_RAW_OVERRIDE:-}"
FOLDED_BIAS_BYTE_OFFSET="${FOLDED_BIAS_BYTE_OFFSET:-0}"
FOLDED_BIAS_RAW_OVERRIDE="${FOLDED_BIAS_RAW_OVERRIDE:-}"
EXTRA_WORD_OVERRIDES="${EXTRA_WORD_OVERRIDES:-}"
MASK_WORD_OVERRIDES="${MASK_WORD_OVERRIDES:-}"
U16_SAMPLE_INDICES="${U16_SAMPLE_INDICES:-}"
INTERNAL_U16_SAMPLES="${INTERNAL_U16_SAMPLES:-}"
FOCUSED_SAMPLE_LOG="${FOCUSED_SAMPLE_LOG:-0}"
KERNEL_ENTRY="${KERNEL_ENTRY:-deep}"

extra_args=()
if [[ "$PRE_CLEAR_ACC" == "0" || "$PRE_CLEAR_ACC" == "false" ]]; then
  extra_args+=(--no-pre-clear-acc)
fi
if [[ "$NATIVE_WRAPPER_PREFETCH" == "1" || "$NATIVE_WRAPPER_PREFETCH" == "true" ]]; then
  extra_args+=(--native-wrapper-prefetch)
fi
if [[ "$PRELOAD_HMX_IDENTITY_BIAS" == "1" || "$PRELOAD_HMX_IDENTITY_BIAS" == "true" ]]; then
  extra_args+=(--preload-hmx-identity-bias)
fi
if [[ "$FOCUSED_SAMPLE_LOG" == "1" || "$FOCUSED_SAMPLE_LOG" == "true" ]]; then
  extra_args+=(--focused-sample-log)
fi
if [[ -n "$EXTRA_WORD_OVERRIDES" ]]; then
  for item in $EXTRA_WORD_OVERRIDES; do
    extra_args+=(--extra-word-override "$item")
  done
fi
if [[ -n "$MASK_WORD_OVERRIDES" ]]; then
  for item in $MASK_WORD_OVERRIDES; do
    extra_args+=(--mask-word-override "$item")
  done
fi
if [[ -n "$U16_SAMPLE_INDICES" ]]; then
  for item in $U16_SAMPLE_INDICES; do
    extra_args+=(--u16-sample-index "$item")
  done
fi
if [[ -n "$INTERNAL_U16_SAMPLES" ]]; then
  for item in $INTERNAL_U16_SAMPLES; do
    extra_args+=(--internal-u16-sample "$item")
  done
fi
if [[ -n "$ACTIVATION_RAW_OVERRIDE" ]]; then
  extra_args+=(--activation-raw-override "$ACTIVATION_RAW_OVERRIDE")
fi
if [[ -n "$PACKED_WEIGHT_RAW_OVERRIDE" ]]; then
  extra_args+=(--packed-weight-raw-override "$PACKED_WEIGHT_RAW_OVERRIDE")
fi
if [[ -n "$FOLDED_BIAS_RAW_OVERRIDE" ]]; then
  extra_args+=(--folded-bias-raw-override "$FOLDED_BIAS_RAW_OVERRIDE")
fi

cd "$ROOT_DIR"
uv run python scripts/build_w4a16_qnn_kernel_tutorial.py \
  --artifact "$ARTIFACT" \
  --out-dir "$OUT_DIR" \
  --measure-repeats "$MEASURE_REPEATS" \
  --chain-steps "$CHAIN_STEPS" \
  --output-seed-mode "$OUTPUT_SEED_MODE" \
  --packed-weight-byte-offset "$PACKED_WEIGHT_BYTE_OFFSET" \
  --folded-bias-byte-offset "$FOLDED_BIAS_BYTE_OFFSET" \
  --kernel-entry "$KERNEL_ENTRY" \
  "${extra_args[@]}"
