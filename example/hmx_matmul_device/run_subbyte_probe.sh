#!/usr/bin/env bash
#
# run_subbyte_probe.sh — push probe_subbyte_device.so to the SM8650
# and execute it on the cDSP, then pull the result file back.
#
# Prerequisites:
#   - ssh oneplus reachable  OR  adb device connected
#   - `bash build.sh` run first
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
HEXAGON_SDK="$ROOT_DIR/tools/hexagon-sdk"
DEVICE_DIR="/data/local/tmp/probe_subbyte"

SO_FILE="$BUILD_DIR/libprobe_subbyte_device.so"
RUN_MAIN="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon"
RUN_MAIN_SKEL="$HEXAGON_SDK/libs/run_main_on_hexagon/ship/hexagon_toolv87_v75/librun_main_on_hexagon_skel.so"
RESULT_LOCAL="$BUILD_DIR/probe_subbyte_result.txt"

[ -f "$SO_FILE" ]   || { echo "Missing $SO_FILE — run build.sh first"; exit 1; }
[ -f "$RUN_MAIN" ]  || { echo "Missing $RUN_MAIN"; exit 1; }
[ -f "$RUN_MAIN_SKEL" ] || { echo "Missing $RUN_MAIN_SKEL"; exit 1; }

# Pick transport. Try (in order):
#   1. ssh oneplus             — tunneled sshd to Android
#   2. adb (Linux)             — Linux adb server + USB
#   3. adb.exe (Windows, WSL)  — Windows adb.exe via /mnt/c
ADB=""
TRANSPORT=""
# Prefer adb (shell user can write /data/local/tmp). ssh falls back.
if adb devices 2>/dev/null | grep -q "device$"; then
    TRANSPORT=adb
    ADB=adb
elif [ -x "/mnt/c/Program Files/android_platform_tools/platform-tools/adb.exe" ] \
      && "/mnt/c/Program Files/android_platform_tools/platform-tools/adb.exe" devices 2>/dev/null | grep -q "device$"; then
    TRANSPORT=adb
    ADB="/mnt/c/Program Files/android_platform_tools/platform-tools/adb.exe"
elif ssh -o ConnectTimeout=3 oneplus true 2>/dev/null; then
    TRANSPORT=ssh
else
    echo "ERROR: no device reachable. Tried:"
    echo "  ssh oneplus              : refused"
    echo "  adb devices (Linux)      : no device"
    echo "  adb.exe devices (Windows): no device"
    echo ""
    echo "Please connect the phone via USB with debugging enabled, then re-run."
    exit 2
fi

echo "[transport] $TRANSPORT"

push()   { if [ "$TRANSPORT" = adb ]; then "$ADB" push "$1" "$2" >/dev/null
           else scp -q "$1" "oneplus:$2" ; fi ; }
shrun()  { if [ "$TRANSPORT" = adb ]; then "$ADB" shell "$@"
           else ssh oneplus "$@" ; fi ; }
pull()   { if [ "$TRANSPORT" = adb ]; then "$ADB" pull "$1" "$2" >/dev/null
           else scp -q "oneplus:$1" "$2" ; fi ; }

echo "=== Push files ==="
shrun "mkdir -p $DEVICE_DIR"
push "$SO_FILE" "$DEVICE_DIR/"
push "$RUN_MAIN" "$DEVICE_DIR/"
push "$RUN_MAIN_SKEL" "$DEVICE_DIR/"
shrun "chmod +x $DEVICE_DIR/run_main_on_hexagon"

if [ "$TRANSPORT" = adb ]; then "$ADB" logcat -c ; fi

echo ""
echo "=== Run on cDSP (30s timeout) ==="
shrun "cd $DEVICE_DIR && rm -f probe_subbyte_result.txt && \
    DSP_LIBRARY_PATH=\"$DEVICE_DIR\" LD_LIBRARY_PATH=$DEVICE_DIR \
    timeout 30 ./run_main_on_hexagon 3 libprobe_subbyte_device.so" \
    2>&1 || echo "(run command exited non-zero; probe may still have completed)"

sleep 1

echo ""
echo "=== Pull result file ==="
pull "$DEVICE_DIR/probe_subbyte_result.txt" "$RESULT_LOCAL" 2>&1 \
    || echo "(no result file on device)"

echo ""
if [ -f "$RESULT_LOCAL" ]; then
    echo "=== Local result ($RESULT_LOCAL) ==="
    cat "$RESULT_LOCAL"
else
    echo "=== FARF logcat (fallback, only for adb transport) ==="
    if [ "$TRANSPORT" = adb ]; then
        "$ADB" logcat -d -v brief | grep "\[DU\]" | sed 's/.*\[DU\]: //' | tail -80
    else
        echo "(result file missing; ssh transport — no logcat capture)"
    fi
fi
