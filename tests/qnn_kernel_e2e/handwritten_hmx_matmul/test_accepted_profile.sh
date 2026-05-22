#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"
ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/tmp/handwritten_hmx_matmul_gate}"
ARTIFACT_ONLY="${ARTIFACT_ONLY:-0}"

case "$PROFILE" in
  u8i8|w4a8|w8a16) ;;
  *)
    echo "profile is not accepted by the current handwritten gate: $PROFILE" >&2
    exit 2
    ;;
esac

if [[ "$ARTIFACT_ONLY" == "1" ]]; then
  echo "accepted profile checks require device-backed promotion evidence" >&2
  exit 2
fi

cd "$ROOT_DIR"

uv run python scripts/check_handwritten_accepted_profile.py \
  --artifact-root "$OUT_ROOT" \
  --profile "$PROFILE"
