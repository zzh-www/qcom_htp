#!/usr/bin/env bash
#
# run_on_device.sh — push Phase 3 OpPackage + probe host binary, execute.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env.sh" >/dev/null

DEVICE="${DEVICE:-oneplus}"
CONNECT="${CONNECT:-ssh}"
ARCH="${ARCH:-v75}"
SHAPE="${SHAPE:-32,32,32}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device|-d)  DEVICE="$2";  shift 2 ;;
        --connect|-c) CONNECT="$2"; shift 2 ;;
        --arch|-a)    ARCH="$2";    shift 2 ;;
        --shape)      SHAPE="$2";   shift 2 ;;
        -h|--help) sed -n '3,15p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done
IFS=',' read -r SHAPE_M SHAPE_K SHAPE_N <<<"$SHAPE"

if [ -z "${DEVICE_DIR:-}" ]; then
    case "$CONNECT" in
        ssh) DEVICE_DIR="~/qnn_run" ;;
        adb) DEVICE_DIR="/data/local/tmp/qnn_run" ;;
    esac
fi

remote_exec() {
    case "$CONNECT" in
        ssh) ssh "$DEVICE" "$1" ;;
        adb) adb -s "$DEVICE" shell "$1" ;;
    esac
}
remote_push() {
    case "$CONNECT" in
        ssh) ssh "$DEVICE" "cat > $DEVICE_DIR/$2" < "$1" ;;
        adb) adb -s "$DEVICE" push "$1" "$DEVICE_DIR/$2" >/dev/null ;;
    esac
}

HTP_SO="$SCRIPT_DIR/build/hexagon-$ARCH/libQnnHmxMatMulPhase3_htp.so"
CPU_SO="$SCRIPT_DIR/build/aarch64/libQnnHmxMatMulPhase3_cpu.so"
HOST_BIN="$SCRIPT_DIR/build/aarch64/run_phase3_probe"

for f in "$HTP_SO" "$CPU_SO" "$HOST_BIN"; do
    [ -f "$f" ] || { echo "missing $f — run build.sh first" >&2; exit 1; }
done

echo "=== push libs ==="
remote_exec "mkdir -p $DEVICE_DIR"
remote_push "$HTP_SO"   "libQnnHmxMatMulPhase3_htp.so"
remote_push "$CPU_SO"   "libQnnHmxMatMulPhase3_cpu.so"
remote_push "$HOST_BIN" "run_phase3_probe"
remote_exec "chmod +x $DEVICE_DIR/run_phase3_probe"

echo "=== execute: $SHAPE_M x $SHAPE_K x $SHAPE_N ==="
remote_exec "cd $DEVICE_DIR && \
    LD_LIBRARY_PATH=.:/vendor/lib64 ADSP_LIBRARY_PATH=. \
    ./run_phase3_probe $SHAPE_M $SHAPE_K $SHAPE_N" 2>&1
