#!/usr/bin/env bash
#
# run_cm_weight_probe.sh — runs probe_cm_weight_layout.so on SM8650 v75
# cDSP to pin down the HMX weight-tile byte layout under `:cm` activation.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
DEVICE_DIR_ADB="/data/local/tmp/probe_cm_weight_layout"
DEVICE_DIR_SSH="probe_cm_weight_layout"
DEVICE_DIR="$DEVICE_DIR_ADB"

SO_FILE="$BUILD_DIR/libprobe_cm_weight_layout.so"
RUN_MAIN="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon"
RUN_MAIN_SKEL="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/hexagon_toolv87_v75/librun_main_on_hexagon_skel.so"
RESULT_LOCAL="$BUILD_DIR/probe_cm_weight_layout_result.txt"

[ -f "$SO_FILE" ]   || { echo "Missing $SO_FILE — run build.sh first"; exit 1; }
[ -f "$RUN_MAIN" ]  || { echo "Missing $RUN_MAIN"; exit 1; }
[ -f "$RUN_MAIN_SKEL" ] || { echo "Missing $RUN_MAIN_SKEL"; exit 1; }

ADB=""
TRANSPORT=""
if adb devices 2>/dev/null | grep -q "device$"; then
    TRANSPORT=adb
    ADB=adb
elif [ -x "/mnt/c/Program Files/android_platform_tools/platform-tools/adb.exe" ] \
      && "/mnt/c/Program Files/android_platform_tools/platform-tools/adb.exe" devices 2>/dev/null | grep -q "device$"; then
    TRANSPORT=adb
    ADB="/mnt/c/Program Files/android_platform_tools/platform-tools/adb.exe"
elif ssh -o ConnectTimeout=3 oneplus true 2>/dev/null; then
    TRANSPORT=ssh
    DEVICE_DIR="$DEVICE_DIR_SSH"
else
    echo "ERROR: no device reachable."
    exit 2
fi

push()  { if [ "$TRANSPORT" = adb ]; then "$ADB" push "$1" "$2" >/dev/null
          else ssh oneplus "cat > $2" < "$1" ; fi ; }
shrun() { if [ "$TRANSPORT" = adb ]; then "$ADB" shell "$@"
          else ssh oneplus "$@" ; fi ; }
pull()  { if [ "$TRANSPORT" = adb ]; then "$ADB" pull "$1" "$2" >/dev/null
          else ssh oneplus "cat $1" > "$2" ; fi ; }

echo "[transport] $TRANSPORT"
shrun "mkdir -p $DEVICE_DIR"
push "$SO_FILE" "$DEVICE_DIR/$(basename "$SO_FILE")"
push "$RUN_MAIN" "$DEVICE_DIR/$(basename "$RUN_MAIN")"
push "$RUN_MAIN_SKEL" "$DEVICE_DIR/$(basename "$RUN_MAIN_SKEL")"
shrun "chmod +x $DEVICE_DIR/run_main_on_hexagon"

if [ "$TRANSPORT" = adb ]; then "$ADB" logcat -c ; fi

echo "=== Run on cDSP (60s timeout) ==="
shrun "cd $DEVICE_DIR && rm -f probe_cm_weight_layout_result.txt && \
    DSP_LIBRARY_PATH=\"\$(pwd)\" LD_LIBRARY_PATH=\"\$(pwd)\":/vendor/lib64 \
    timeout 60 ./run_main_on_hexagon 3 libprobe_cm_weight_layout.so" \
    2>&1 || echo "(run command exited non-zero; probe may still have completed)"

sleep 1
pull "$DEVICE_DIR/probe_cm_weight_layout_result.txt" "$RESULT_LOCAL" 2>&1 \
    || echo "(no result file on device)"

echo ""
if [ -f "$RESULT_LOCAL" ]; then
    echo "=== Local result ==="
    cat "$RESULT_LOCAL"
fi
