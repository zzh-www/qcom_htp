#!/usr/bin/env bash
# Push Phase-A artifacts to device, run qnn-net-run with optrace, pull logs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE="${DEVICE:-oneplus}"
DEV_DIR="~/qnn_run/phaseA"

ssh "$DEVICE" "mkdir -p $DEV_DIR $DEV_DIR/runtime_inputs_u8"

# 1. context binary
scp -q "$SCRIPT_DIR/ctx_out/matmul_native_ctx.bin" "$DEVICE:$DEV_DIR/"

# 2. runtime input + list (file list is relative to DEV_DIR at execute time)
scp -q "$SCRIPT_DIR/runtime_inputs_u8/a.raw" "$DEVICE:$DEV_DIR/runtime_inputs_u8/"
printf 'input_0:=runtime_inputs_u8/a.raw\n' > "$SCRIPT_DIR/input_list_dev.txt"
scp -q "$SCRIPT_DIR/input_list_dev.txt" "$DEVICE:$DEV_DIR/"

# 3. backend extension configs (match graph_name = "model")
scp -q "$SCRIPT_DIR/htp_config.json" "$DEVICE:$DEV_DIR/"
scp -q "$SCRIPT_DIR/htp_backend_ext.json" "$DEVICE:$DEV_DIR/"

# 4. execute with detailed profiling + optrace
ssh "$DEVICE" "cd $DEV_DIR && LD_LIBRARY_PATH=../:/vendor/lib64 ADSP_LIBRARY_PATH=../ \
    ../qnn-net-run \
        --backend ../libQnnHtp.so \
        --retrieve_context matmul_native_ctx.bin \
        --input_list input_list_dev.txt \
        --profiling_level detailed \
        --profiling_option optrace \
        --output_dir out \
        --config_file htp_config.json \
        --perf_profile burst 2>&1 | tail -40"

# 5. pull results
mkdir -p "$SCRIPT_DIR/device_out"
scp -q -r "$DEVICE:$DEV_DIR/out" "$SCRIPT_DIR/device_out/" 2>/dev/null || true
echo "=== pulled ==="
find "$SCRIPT_DIR/device_out" -type f -printf "  %p (%s bytes)\n"
