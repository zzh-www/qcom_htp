#!/usr/bin/env bash
#
# run_on_device.sh — push OpPackage + host test to device, execute, report.
#
# Flags (env fallbacks in parens):
#   --device   DEVICE   device id / ssh host alias / adb serial (DEVICE, default oneplus)
#   --connect  CONNECT  ssh | adb                               (CONNECT, default ssh)
#   --arch     ARCH     HTP arch                                (ARCH, default v75)
#   --shape    M,K,N    matmul dimensions                        (SHAPE, default 32,32,32)
#
# Prerequisites: scripts/env.sh sourced; ~/qnn_run (or /data/local/tmp/qnn_run)
# pre-populated with QNN runtime (same as qnn_matmul_profile expects).

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
    # <local_path> <dst_basename>
    case "$CONNECT" in
        ssh) ssh "$DEVICE" "cat > $DEVICE_DIR/$2" < "$1" ;;
        adb) adb -s "$DEVICE" push "$1" "$DEVICE_DIR/$2" >/dev/null ;;
    esac
}

HTP_SO="$SCRIPT_DIR/build/hexagon-$ARCH/libQnnHmxInt4MatMul_htp.so"
CPU_SO="$SCRIPT_DIR/build/aarch64/libQnnHmxInt4MatMul_cpu.so"
HOST_BIN="$SCRIPT_DIR/build/aarch64/run_int4_matmul"

for f in "$HTP_SO" "$CPU_SO" "$HOST_BIN"; do
    [ -f "$f" ] || { echo "missing $f — run build.sh first" >&2; exit 1; }
done

echo "=== push libs ==="
remote_exec "mkdir -p $DEVICE_DIR"
remote_push "$HTP_SO"   "libQnnHmxInt4MatMul_htp.so"
remote_push "$CPU_SO"   "libQnnHmxInt4MatMul_cpu.so"
remote_push "$HOST_BIN" "run_int4_matmul"
remote_exec "chmod +x $DEVICE_DIR/run_int4_matmul"

echo "=== execute: $SHAPE_M x $SHAPE_K x $SHAPE_N ==="
remote_exec "cd $DEVICE_DIR && \
    LD_LIBRARY_PATH=.:/vendor/lib64 ADSP_LIBRARY_PATH=. \
    ./run_int4_matmul $SHAPE_M $SHAPE_K $SHAPE_N" 2>&1
