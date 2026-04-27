#!/usr/bin/env bash
# sweep_v8c8_shapes.sh — Single-instance V8C8 framework smoke across shapes.
#
# Validates: ONNX gen → qairt-converter → ctxgen → device run → row-major output.
# Kernel body is still NOOP-with-marker, so only checks framework plumbing
# (auto-inserted q::*InputSlice/ForceFormat_Crouton/weights_to_vtcm + our
# BbbKMajor + UntileToRowMajor + auto-folded Reshape).
#
# VTCM budget per square S³ ≈ 3·S² + 8·S bytes (act crouton + combined wt+bias
# static + tile-layout output). 1024³ ≈ 3 MiB OK; 2048³ ≈ 12 MiB likely too big
# without multi-instance split.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

SHAPES="${SHAPES:-32 64 128 256 512 1024}"
SUMMARY="$SCRIPT_DIR/phase1_validation/v8c8_sweep_summary.txt"
mkdir -p "$(dirname "$SUMMARY")"
: > "$SUMMARY"

printf "%-7s %-15s %-12s %-10s %-12s %-10s\n" \
    "shape" "ctx_nodes" "device" "out_size" "out_match" "marker" | tee -a "$SUMMARY"
printf "%-7s %-15s %-12s %-10s %-12s %-10s\n" \
    "-----" "---------" "------" "--------" "---------" "------" | tee -a "$SUMMARY"

for S in $SHAPES; do
    OUT_DIR="$SCRIPT_DIR/phase1_validation/v8c8_sweep_${S}"
    M=$S K=$S N=$S OUT_DIR="$OUT_DIR" \
        bash "$SCRIPT_DIR/run_v8c8_phase2.sh" > "$OUT_DIR.log" 2>&1 || true

    # Parse outcomes
    NODES="?"
    DEVICE="FAIL"
    OUT_SZ="0"
    OUT_MATCH="?"
    MARKER="?"

    if [ -f "$OUT_DIR/ctx/v8c8_bottom_mapping.json" ]; then
        NODES=$(python3 -c "
import json
d=json.load(open('$OUT_DIR/ctx/v8c8_bottom_mapping.json'))
print(len(d['graph']['nodes']))
" 2>/dev/null || echo "?")
    fi

    if grep -q "Finished Executing Graphs" "$OUT_DIR/run.log" 2>/dev/null; then
        DEVICE="OK"
    fi

    if [ -s "$OUT_DIR/device_out/out.raw" ]; then
        OUT_SZ=$(stat -c%s "$OUT_DIR/device_out/out.raw")
        EXPECTED=$((S * S * 4))
        if [ "$OUT_SZ" == "$EXPECTED" ]; then
            OUT_MATCH="OK"
        else
            OUT_MATCH="want=$EXPECTED"
        fi
        # Check first u8 cell == 0xa5 (after fp32 dequant: round to int)
        MARKER=$(python3 -c "
import numpy as np
b = np.fromfile('$OUT_DIR/device_out/out.raw', dtype=np.float32)
if b.size >= 16:
    u8 = np.round(b[:16]).astype(int).tolist()
    print('OK' if u8[0]==0xa5 and u8[15]==0x5a else 'BAD' + str(u8[:4]))
else:
    print('SHORT')
" 2>/dev/null || echo "?")
    fi

    printf "%-7s %-15s %-12s %-10s %-12s %-10s\n" \
        "${S}^3" "$NODES" "$DEVICE" "$OUT_SZ" "$OUT_MATCH" "$MARKER" | tee -a "$SUMMARY"
done

echo
echo "summary saved to $SUMMARY"
