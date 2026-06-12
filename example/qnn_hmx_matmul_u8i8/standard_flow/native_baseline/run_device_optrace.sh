#!/usr/bin/env bash
# Push Phase-A artifacts to device, run qnn-net-run with native I/O + optrace,
# pull logs, and decode the standard optrace artifact set.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
DEVICE="${DEVICE:-oneplus}"
DEV_DIR="~/qnn_run/phaseA"
source "$ROOT_DIR/scripts/dssh.sh"

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
        --use_native_input_files \
        --use_native_output_files \
        --perf_profile burst 2>&1 | tail -40"

# 5. pull results
mkdir -p "$SCRIPT_DIR/device_out"
ssh "$DEVICE" "cat $DEV_DIR/out/qnn-profiling-data_0.log 2>/dev/null" \
    > "$SCRIPT_DIR/device_out/qnn-profiling-data_0.log" 2>/dev/null || true
ssh "$DEVICE" "cat $DEV_DIR/out/qnn-profiling-data_2.log 2>/dev/null" \
    > "$SCRIPT_DIR/device_out/qnn-profiling-data_2.log" 2>/dev/null || true
ssh "$DEVICE" "cat $DEV_DIR/out/Result_0/output_0_native.raw 2>/dev/null || \
    cat $DEV_DIR/out/Result_0/output_0.raw 2>/dev/null || \
    cat $DEV_DIR/out/Result_0/Y_native.raw 2>/dev/null || \
    cat $DEV_DIR/out/Result_0/Y.raw 2>/dev/null" \
    > "$SCRIPT_DIR/device_out/out.raw" 2>/dev/null || true

if [ "${DECODE_OPTRACE:-1}" = "1" ]; then
    PROFILE_LOG="$SCRIPT_DIR/device_out/qnn-profiling-data_2.log"
    [ -s "$PROFILE_LOG" ] || PROFILE_LOG="$SCRIPT_DIR/device_out/qnn-profiling-data_0.log"
    SCHEMATIC="$(find "$SCRIPT_DIR/ctx_out" -maxdepth 1 -name '*schematic.bin' -print -quit 2>/dev/null || true)"
    if [ -s "$PROFILE_LOG" ] && [ -n "$SCHEMATIC" ]; then
        python "$ROOT_DIR/scripts/decode_qnn_optrace.py" "$SCRIPT_DIR" \
            --profile-log "$PROFILE_LOG" \
            --schematic "$SCHEMATIC" || {
            [ "${STRICT_OPTRACE:-0}" = "1" ] && exit 1
            echo "  [warn] optrace decode failed; raw log kept in $SCRIPT_DIR/device_out" >&2
        }
    else
        echo "  [warn] missing profile log or schematic; optrace decode skipped" >&2
    fi
fi

echo "=== pulled ==="
find "$SCRIPT_DIR/device_out" -type f -printf "  %p (%s bytes)\n"
