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
