#!/usr/bin/env bash
#
# build.sh — compile the device-side matmul bench as a v75 CDSP .so.
# Loaded by run_main_on_hexagon on a physical SM8650 (Unsigned PD).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD="$SCRIPT_DIR/build"
mkdir -p "$BUILD"

source "$ROOT_DIR/scripts/env.sh" >/dev/null

HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
INT16="$ROOT_DIR/example/hmx_matmul_int16"

OUT="$BUILD/libbench_matmul_device.so"

hexagon-clang -mv75 -O2 \
    -mhvx -mhvx-length=128B \
    -mhmx \
    -shared -fPIC \
    -I "$HEXAGON_SDK/incs" \
    -I "$HEXAGON_SDK/incs/stddef" \
    -I "$HEXAGON_SDK/rtos/qurt/computev75/include/qurt" \
    -I "$INT16" \
    "$SCRIPT_DIR/bench_matmul_device.c" \
    "$INT16/int16_matmul_hmx.c" \
    "$INT16/int16_matmul_ref.c" \
    -o "$OUT"

echo "  -> $OUT ($(wc -c < "$OUT") bytes)"

# probe_subbyte_device — silicon-side replica of probe_subbyte_full sim probe.
OUT_PROBE="$BUILD/libprobe_subbyte_device.so"
hexagon-clang -mv75 -O2 \
    -mhvx -mhvx-length=128B \
    -mhmx \
    -shared -fPIC \
    -I "$HEXAGON_SDK/incs" \
    -I "$HEXAGON_SDK/incs/stddef" \
    -I "$HEXAGON_SDK/rtos/qurt/computev75/include/qurt" \
    "$SCRIPT_DIR/probe_subbyte_device.c" \
    -o "$OUT_PROBE"
echo "  -> $OUT_PROBE ($(wc -c < "$OUT_PROBE") bytes)"

# probe_dualacc_device — isolates :above + mxswapacc semantics.
OUT_DA="$BUILD/libprobe_dualacc_device.so"
hexagon-clang -mv75 -O2 \
    -mhvx -mhvx-length=128B -mhmx \
    -shared -fPIC \
    -I "$HEXAGON_SDK/incs" \
    -I "$HEXAGON_SDK/incs/stddef" \
    -I "$HEXAGON_SDK/rtos/qurt/computev75/include/qurt" \
    "$SCRIPT_DIR/probe_dualacc_device.c" \
    -o "$OUT_DA"
echo "  -> $OUT_DA ($(wc -c < "$OUT_DA") bytes)"

# probe_pipeline_device — isolates :cm / Rt_wt=0x3FF / pair-mode pipelining.
OUT_PIPE="$BUILD/libprobe_pipeline_device.so"
hexagon-clang -mv75 -O2 \
    -mhvx -mhvx-length=128B \
    -mhmx \
    -shared -fPIC \
    -I "$HEXAGON_SDK/incs" \
    -I "$HEXAGON_SDK/incs/stddef" \
    -I "$HEXAGON_SDK/rtos/qurt/computev75/include/qurt" \
    "$SCRIPT_DIR/probe_pipeline_device.c" \
    -o "$OUT_PIPE"
echo "  -> $OUT_PIPE ($(wc -c < "$OUT_PIPE") bytes)"

# probe_cm_row_major — tests whether :cm can consume row-major activation.
OUT_CM="$BUILD/libprobe_cm_row_major.so"
hexagon-clang -mv75 -O2 \
    -mhvx -mhvx-length=128B \
    -mhmx \
    -shared -fPIC \
    -I "$HEXAGON_SDK/incs" \
    -I "$HEXAGON_SDK/incs/stddef" \
    -I "$HEXAGON_SDK/rtos/qurt/computev75/include/qurt" \
    "$SCRIPT_DIR/probe_cm_row_major.c" \
    -o "$OUT_CM"
echo "  -> $OUT_CM ($(wc -c < "$OUT_CM") bytes)"

# probe_cm_weight_layout — varies weight patterns + layouts under :cm to
# pin down the weight-tile byte layout HMX expects.
OUT_CMW="$BUILD/libprobe_cm_weight_layout.so"
hexagon-clang -mv75 -O2 \
    -mhvx -mhvx-length=128B \
    -mhmx \
    -shared -fPIC \
    -I "$HEXAGON_SDK/incs" \
    -I "$HEXAGON_SDK/incs/stddef" \
    -I "$HEXAGON_SDK/rtos/qurt/computev75/include/qurt" \
    "$SCRIPT_DIR/probe_cm_weight_layout.c" \
    -o "$OUT_CMW"
echo "  -> $OUT_CMW ($(wc -c < "$OUT_CMW") bytes)"

# probe_cm_readback — pins down the dual-scale readback layout under :cm.
OUT_CMR="$BUILD/libprobe_cm_readback.so"
hexagon-clang -mv75 -O2 \
    -mhvx -mhvx-length=128B \
    -mhmx \
    -shared -fPIC \
    -I "$HEXAGON_SDK/incs" \
    -I "$HEXAGON_SDK/incs/stddef" \
    -I "$HEXAGON_SDK/rtos/qurt/computev75/include/qurt" \
    "$SCRIPT_DIR/probe_cm_readback.c" \
    -o "$OUT_CMR"
echo "  -> $OUT_CMR ($(wc -c < "$OUT_CMR") bytes)"

# probe_sat_ub — verifies :after:cm:sat.ub single-byte HMX readback,
# primarily whether it fills all 32 output rows (vs the 16-of-32 limitation
# found on :after.uh acc:2x1). Drives Phase 3D.4 V8 replica decision.
OUT_SU="$BUILD/libprobe_sat_ub.so"
hexagon-clang -mv75 -O2 \
    -mhvx -mhvx-length=128B -mhmx \
    -shared -fPIC \
    -I "$HEXAGON_SDK/incs" \
    -I "$HEXAGON_SDK/incs/stddef" \
    -I "$HEXAGON_SDK/rtos/qurt/computev75/include/qurt" \
    "$SCRIPT_DIR/probe_sat_ub.c" \
    -o "$OUT_SU"
echo "  -> $OUT_SU ($(wc -c < "$OUT_SU") bytes)"

# probe_cm_singlecell — single (m,n) cell sweep to definitively map
# acc[m][n] → readback indices (lo + hi) under :cm.
OUT_CMS="$BUILD/libprobe_cm_singlecell.so"
hexagon-clang -mv75 -O2 \
    -mhvx -mhvx-length=128B \
    -mhmx \
    -shared -fPIC \
    -I "$HEXAGON_SDK/incs" \
    -I "$HEXAGON_SDK/incs/stddef" \
    -I "$HEXAGON_SDK/rtos/qurt/computev75/include/qurt" \
    "$SCRIPT_DIR/probe_cm_singlecell.c" \
    -o "$OUT_CMS"
echo "  -> $OUT_CMS ($(wc -c < "$OUT_CMS") bytes)"
