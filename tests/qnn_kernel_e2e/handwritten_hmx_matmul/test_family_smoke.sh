#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <family>" >&2
  exit 2
fi

FAMILY="$1"
ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/tmp/handwritten_hmx_matmul_gate}"
DEVICE="${DEVICE:-oneplus}"
ARTIFACT_ONLY="${ARTIFACT_ONLY:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
OUT_DIR="$OUT_ROOT/$FAMILY"

case "$FAMILY" in
  u8i8|w4a8|w8a16|w4a16) ;;
  *)
    echo "unknown handwritten HMX MatMul family: $FAMILY" >&2
    exit 2
    ;;
esac

cd "$ROOT_DIR"

if [[ "$ARTIFACT_ONLY" == "1" ]]; then
  uv run python scripts/check_handwritten_runtime_artifact.py "$OUT_DIR"
  exit 0
fi

if [[ "$SKIP_BUILD" != "1" ]]; then
  example/handwritten_hmx_matmul/build_host.sh
  example/handwritten_hmx_matmul/build_android.sh
fi

uv run python example/handwritten_hmx_matmul/run_owned_smoke.py \
  --family "$FAMILY" --device "$DEVICE" --out-dir "$OUT_DIR"
uv run python scripts/check_handwritten_runtime_artifact.py \
  "$OUT_DIR" --require-device
