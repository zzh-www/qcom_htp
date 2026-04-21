#!/usr/bin/env bash
# run_dualacc_probe.sh — run probe_dualacc_device on SM8650 v75 cDSP.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
DEVICE_DIR="probe_dualacc"
SO_FILE="$BUILD_DIR/libprobe_dualacc_device.so"
RUN_MAIN="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon"
RUN_SKEL="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/hexagon_toolv87_v75/librun_main_on_hexagon_skel.so"
RESULT="$BUILD_DIR/probe_dualacc_result.txt"
[ -f "$SO_FILE" ] || { echo "Missing $SO_FILE"; exit 1; }
push() { ssh oneplus "cat > $2" < "$1"; }
shrun() { ssh oneplus "$@"; }
pull() { ssh oneplus "cat $1" > "$2"; }
shrun "mkdir -p $DEVICE_DIR"
push "$SO_FILE"  "$DEVICE_DIR/$(basename "$SO_FILE")"
push "$RUN_MAIN" "$DEVICE_DIR/$(basename "$RUN_MAIN")"
push "$RUN_SKEL" "$DEVICE_DIR/$(basename "$RUN_SKEL")"
shrun "chmod +x $DEVICE_DIR/run_main_on_hexagon"
shrun "cd $DEVICE_DIR && rm -f probe_dualacc_result.txt && DSP_LIBRARY_PATH=\"\$(pwd)\" LD_LIBRARY_PATH=\"\$(pwd)\":/vendor/lib64 timeout 30 ./run_main_on_hexagon 3 libprobe_dualacc_device.so" 2>&1 || true
pull "$DEVICE_DIR/probe_dualacc_result.txt" "$RESULT" 2>&1 || true
echo "=== Result ==="
[ -f "$RESULT" ] && cat "$RESULT"
