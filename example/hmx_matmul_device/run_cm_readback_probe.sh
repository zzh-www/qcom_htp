#!/usr/bin/env bash
#
# run_cm_readback_probe.sh — runs probe_cm_readback.so to map the
# `:after.uh acc:2x1` readback bytes to (m,n) accumulator cells when
# the MAC is issued via `:cm` row-major activation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
DEVICE_DIR_ADB="/data/local/tmp/probe_cm_readback"
DEVICE_DIR_SSH="probe_cm_readback"
DEVICE_DIR="$DEVICE_DIR_ADB"

SO_FILE="$BUILD_DIR/libprobe_cm_readback.so"
RUN_MAIN="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon"
RUN_MAIN_SKEL="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/hexagon_toolv87_v75/librun_main_on_hexagon_skel.so"
RESULT_LOCAL="$BUILD_DIR/probe_cm_readback_result.txt"

[ -f "$SO_FILE" ]   || { echo "Missing $SO_FILE — run build.sh first"; exit 1; }

ADB=""
TRANSPORT=""
if adb devices 2>/dev/null | grep -q "device$"; then
    TRANSPORT=adb; ADB=adb
elif ssh -o ConnectTimeout=3 oneplus true 2>/dev/null; then
    TRANSPORT=ssh
    DEVICE_DIR="$DEVICE_DIR_SSH"
else
    echo "ERROR: no device reachable."; exit 2
fi

push()  { if [ "$TRANSPORT" = adb ]; then "$ADB" push "$1" "$2" >/dev/null
          else ssh oneplus "cat > $2" < "$1" ; fi ; }
shrun() { if [ "$TRANSPORT" = adb ]; then "$ADB" shell "$@"
          else ssh oneplus "$@" ; fi ; }
pull()  { if [ "$TRANSPORT" = adb ]; then "$ADB" pull "$1" "$2" >/dev/null
          else ssh oneplus "cat $1" > "$2" ; fi ; }

shrun "mkdir -p $DEVICE_DIR"
push "$SO_FILE" "$DEVICE_DIR/$(basename "$SO_FILE")"
push "$RUN_MAIN" "$DEVICE_DIR/$(basename "$RUN_MAIN")"
push "$RUN_MAIN_SKEL" "$DEVICE_DIR/$(basename "$RUN_MAIN_SKEL")"
shrun "chmod +x $DEVICE_DIR/run_main_on_hexagon"

shrun "cd $DEVICE_DIR && rm -f probe_cm_readback_result.txt && \
    DSP_LIBRARY_PATH=\"\$(pwd)\" LD_LIBRARY_PATH=\"\$(pwd)\":/vendor/lib64 \
    timeout 60 ./run_main_on_hexagon 3 libprobe_cm_readback.so" 2>&1 \
    || echo "(run command exited non-zero; probe may still have completed)"

sleep 1
pull "$DEVICE_DIR/probe_cm_readback_result.txt" "$RESULT_LOCAL" 2>&1 \
    || echo "(no result file on device)"

if [ -f "$RESULT_LOCAL" ]; then cat "$RESULT_LOCAL"; fi
