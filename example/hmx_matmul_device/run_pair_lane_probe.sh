#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
DEVICE_DIR_ADB="/data/local/tmp/probe_pair_lane"
DEVICE_DIR_SSH="probe_pair_lane"
DEVICE_DIR="$DEVICE_DIR_ADB"

SO_FILE="$BUILD_DIR/libprobe_pair_lane.so"
RUN_MAIN="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon"
RUN_MAIN_SKEL="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/hexagon_toolv87_v75/librun_main_on_hexagon_skel.so"
RESULT_LOCAL="$BUILD_DIR/probe_pair_lane_result.txt"

[ -f "$SO_FILE" ]   || { echo "Missing $SO_FILE — run build.sh first"; exit 1; }

TRANSPORT=""
if adb devices 2>/dev/null | grep -q "device$"; then
    TRANSPORT=adb
elif ssh -o ConnectTimeout=3 oneplus true 2>/dev/null; then
    TRANSPORT=ssh
    DEVICE_DIR="$DEVICE_DIR_SSH"
else
    echo "ERROR: no device reachable."; exit 2
fi

push()  { if [ "$TRANSPORT" = adb ]; then adb push "$1" "$2" >/dev/null
          else ssh oneplus "cat > $2" < "$1" ; fi ; }
shrun() { if [ "$TRANSPORT" = adb ]; then adb shell "$@"
          else ssh oneplus "$@" ; fi ; }
pull()  { if [ "$TRANSPORT" = adb ]; then adb pull "$1" "$2" >/dev/null
          else ssh oneplus "cat $1" > "$2" ; fi ; }

shrun "mkdir -p $DEVICE_DIR"
push "$SO_FILE" "$DEVICE_DIR/$(basename "$SO_FILE")"
push "$RUN_MAIN" "$DEVICE_DIR/$(basename "$RUN_MAIN")"
push "$RUN_MAIN_SKEL" "$DEVICE_DIR/$(basename "$RUN_MAIN_SKEL")"
shrun "chmod +x $DEVICE_DIR/run_main_on_hexagon"
shrun "cd $DEVICE_DIR && rm -f probe_pair_lane_result.txt && \
    DSP_LIBRARY_PATH=\"\$(pwd)\" LD_LIBRARY_PATH=\"\$(pwd)\":/vendor/lib64 \
    timeout 60 ./run_main_on_hexagon 3 libprobe_pair_lane.so" 2>&1 \
    || echo "(run command exited non-zero; probe may still have completed)"
sleep 1
pull "$DEVICE_DIR/probe_pair_lane_result.txt" "$RESULT_LOCAL" 2>&1 || echo "(no result)"
if [ -f "$RESULT_LOCAL" ]; then cat "$RESULT_LOCAL"; fi
