#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$ROOT_DIR/scripts/env.sh" >/dev/null
DEVICE="${DEVICE:-oneplus}"
SHAPE="${SHAPE:-32,32,32}"
while [[ $# -gt 0 ]]; do case "$1" in --shape) SHAPE="$2"; shift 2 ;; *) shift ;; esac; done
IFS=',' read -r M K N <<<"$SHAPE"
DEVICE_DIR="~/qnn_run"
ssh "$DEVICE" "cat > $DEVICE_DIR/libQnnHmxMatMulPhase3_htp.so" < build/hexagon-v75/libQnnHmxMatMulPhase3_htp.so
ssh "$DEVICE" "cat > $DEVICE_DIR/libQnnHmxMatMulPhase3_cpu.so" < build/aarch64/libQnnHmxMatMulPhase3_cpu.so
ssh "$DEVICE" "cat > $DEVICE_DIR/run_matmul_v6" < build/aarch64/run_matmul_v6
ssh "$DEVICE" "chmod +x $DEVICE_DIR/run_matmul_v6 && cd $DEVICE_DIR && LD_LIBRARY_PATH=.:/vendor/lib64 ADSP_LIBRARY_PATH=. ./run_matmul_v6 $M $K $N"
