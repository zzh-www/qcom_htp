#!/usr/bin/env bash
# run_v9_desc_dump.sh — V9 descriptor-dump runner.
#
# Prereqs:
#   bash build.sh        with EXTRA_DEFS="-DV9_USE_NATIVE_KERNEL -DV9_NATIVE_SINGLE_CALL -DV9_NATIVE_V73DEEP -DV9_C8_ALIGNMENT_TEST -DV9_DESC_DUMP"
#   bash build_x86.sh    with same EXTRA_DEFS
#
# Output: $OUT_DIR/device_out/out.raw — fp32 file with M*N cells (scale=1, zp=0
# means each fp32 cell is the original u8). First 5 rows × 128 cells encode the
# descriptor dump (see scripts/parse_v73deep_desc_dump.py).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

OUT_DIR_DEFAULT="$SCRIPT_DIR/phase1_validation/v73deep_desc_dump"
export OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"
export WT_LAYOUT="${WT_LAYOUT:-kmaj}"
export CHAIN="${CHAIN:-8}"
export M="${M:-256}"
export K="${K:-256}"
export N="${N:-256}"

bash "$SCRIPT_DIR/run_v8c8_chain.sh" "$@"

echo
echo "=== parse descriptor dump ==="
python3 "$ROOT_DIR/scripts/parse_v73deep_desc_dump.py" \
    "$OUT_DIR/device_out/out.raw" --N "$N"
