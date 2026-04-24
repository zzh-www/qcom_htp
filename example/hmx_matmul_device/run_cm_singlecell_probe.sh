#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
DEVICE_DIR_SSH="probe_cm_singlecell"

SO_FILE="$BUILD_DIR/libprobe_cm_singlecell.so"
RUN_MAIN="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon"
RUN_MAIN_SKEL="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/hexagon_toolv87_v75/librun_main_on_hexagon_skel.so"
RESULT_LOCAL="$BUILD_DIR/probe_cm_singlecell_result.txt"

[ -f "$SO_FILE" ] || { echo "Missing $SO_FILE"; exit 1; }
ssh oneplus "mkdir -p $DEVICE_DIR_SSH"
ssh oneplus "cat > $DEVICE_DIR_SSH/$(basename "$SO_FILE")" < "$SO_FILE"
ssh oneplus "cat > $DEVICE_DIR_SSH/$(basename "$RUN_MAIN")" < "$RUN_MAIN"
ssh oneplus "cat > $DEVICE_DIR_SSH/$(basename "$RUN_MAIN_SKEL")" < "$RUN_MAIN_SKEL"
ssh oneplus "chmod +x $DEVICE_DIR_SSH/run_main_on_hexagon"
ssh oneplus "cd $DEVICE_DIR_SSH && rm -f probe_cm_singlecell_result.txt && \
    DSP_LIBRARY_PATH=\"\$(pwd)\" LD_LIBRARY_PATH=\"\$(pwd)\":/vendor/lib64 \
    timeout 60 ./run_main_on_hexagon 3 libprobe_cm_singlecell.so" 2>&1 \
    || echo "(run command exited non-zero)"
sleep 1
ssh oneplus "cat $DEVICE_DIR_SSH/probe_cm_singlecell_result.txt" > "$RESULT_LOCAL" 2>&1 \
    || echo "(no result file)"
[ -f "$RESULT_LOCAL" ] && cat "$RESULT_LOCAL"
