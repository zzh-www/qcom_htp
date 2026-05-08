#!/usr/bin/env bash
# Native baseline chain runner: generate, convert, ctxgen, push to device,
# run with optrace, pull profile log + decode chrometrace.
#
# Mirror of custom_u8i8/run_u8i8_chain.sh so native and custom chains can be
# profile-compared with the same chain methodology.
#
# Usage:
#   SIZE=256 CHAIN=8 OUT_NAME=s256_chain8 bash run_native_chain.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$ROOT_DIR/scripts/env.sh" >/dev/null
source "$ROOT_DIR/scripts/qairt_quant_flow.sh"
source "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH=$QNN_SDK_ROOT/lib/python
export PATH=$ANDROID_NDK_ROOT:$PATH
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}

DEVICE="${DEVICE:-oneplus}"
SIZE="${SIZE:-256}"
CHAIN="${CHAIN:-8}"
MODE="${MODE:-chain}"        # chain | independent
SHARED_W="${SHARED_W:-0}"    # 1 = share W across all matmuls, 0 = per-op W
OUT_NAME="${OUT_NAME:-s${SIZE}_${MODE}${CHAIN}}"
OUT_DIR="$SCRIPT_DIR/$OUT_NAME"

echo "=== native ${MODE}: size=$SIZE chain=$CHAIN shared_w=$SHARED_W -> $OUT_DIR ==="

SHARED_W_FLAG=""
[ "$SHARED_W" = "1" ] && SHARED_W_FLAG="--shared_w"
python "$SCRIPT_DIR/gen_matmul_onnx_chain.py" --size "$SIZE" --chain "$CHAIN" --mode "$MODE" $SHARED_W_FLAG --out "$OUT_NAME"

cd "$OUT_DIR"
cp -f "$SCRIPT_DIR/htp_config.json" "$SCRIPT_DIR/htp_backend_ext.json" .
LAYOUT_FLAGS=()
if [ "$MODE" = "chain" ]; then
    LAYOUT_FLAGS+=(--source_model_input_layout A NONTRIVIAL)
    LAYOUT_FLAGS+=(--desired_input_layout A NONTRIVIAL)
    LAYOUT_FLAGS+=(--source_model_output_layout Y NONTRIVIAL)
    LAYOUT_FLAGS+=(--desired_output_layout Y NONTRIVIAL)
else
    for i in $(seq 0 $((CHAIN - 1))); do
        IN_NAME="A"
        if [ "$i" != "0" ]; then IN_NAME="A_${i}"; fi
        OUT_NAME="Y_${i}"
        LAYOUT_FLAGS+=(--source_model_input_layout "$IN_NAME" NONTRIVIAL)
        LAYOUT_FLAGS+=(--desired_input_layout "$IN_NAME" NONTRIVIAL)
        LAYOUT_FLAGS+=(--source_model_output_layout "$OUT_NAME" NONTRIVIAL)
        LAYOUT_FLAGS+=(--desired_output_layout "$OUT_NAME" NONTRIVIAL)
    done
fi

echo "=== qairt-converter -> qairt-quantizer ==="
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter \
    -i model.onnx \
    --quantization_overrides quant_overrides.json \
    "${LAYOUT_FLAGS[@]}" \
    -o model_encoded.dlc 2>&1 | tee _convert.log | tail -3
qairt_quantize_encoded_dlc model_encoded.dlc model.dlc 8 8 32 0 _quantize.log
tail -3 _quantize.log

echo "=== qnn-context-binary-generator ==="
rm -rf ctx
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator \
    --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
    --dlc_path model.dlc \
    --binary_file matmul_native_ctx \
    --output_dir ctx \
    --config_file htp_config.json \
    --profiling_level detailed --profiling_option optrace \
    --save_backend_op_mapping 2>&1 | tee _ctxgen.log | tail -3
# schematic.bin lands in CWD; sometimes named model_schematic.bin
SCHEMATIC=""
[ -f schematic.bin ] && SCHEMATIC=schematic.bin
[ -z "$SCHEMATIC" ] && [ -f model_schematic.bin ] && SCHEMATIC=model_schematic.bin
[ -z "$SCHEMATIC" ] && { echo "  [warn] no schematic.bin found — chrometrace decode will fail"; }

echo "  --- lowered graph node summary ---"
python3 -c "
import json
from collections import Counter
from pathlib import Path
paths = sorted(Path('ctx').glob('*bottom_mapping.json'))
d = json.load(open(paths[0]))
n = d['graph']['nodes']
print(f'  total nodes: {len(n)}')
for t,c in Counter(x.get('type','?') for x in n.values()).most_common():
    print(f'    {c:>3}  {t}')
" || true

echo "=== push + run on device ==="
DEV_DIR="qnn_run/native_${OUT_NAME}"
ssh "$DEVICE" "mkdir -p $DEV_DIR/runtime_inputs_u8"
ssh "$DEVICE" "cat > $DEV_DIR/matmul_native_ctx.bin"   < ctx/matmul_native_ctx.bin
ssh "$DEVICE" "cat > $DEV_DIR/htp_config.json"         < "$SCRIPT_DIR/htp_config.json"
ssh "$DEVICE" "cat > $DEV_DIR/htp_backend_ext.json"    < "$SCRIPT_DIR/htp_backend_ext.json"
# Push all runtime input .raw files (chain mode = 1 file, independent = N files).
for f in runtime_inputs_u8/*.raw; do
    ssh "$DEVICE" "cat > $DEV_DIR/runtime_inputs_u8/$(basename "$f")" < "$f"
done
# Push the gen-script-produced runtime_input_list.txt (handles single-line
# multi-input format for independent mode).
ssh "$DEVICE" "cat > $DEV_DIR/input_list.txt" < runtime_input_list.txt

OUTPUT_FLAGS=""
if [ "${NATIVE_OUTPUT:-1}" = "1" ]; then
    OUTPUT_FLAGS="--use_native_output_files"
fi
ssh "$DEVICE" "cd $DEV_DIR && rm -rf out && \
    LD_LIBRARY_PATH=../:/vendor/lib64 ADSP_LIBRARY_PATH=../ \
    ../qnn-net-run \
      --backend ../libQnnHtp.so \
      --retrieve_context matmul_native_ctx.bin \
      --input_list input_list.txt \
      --profiling_level detailed --profiling_option optrace \
      --output_dir out \
      --config_file htp_config.json \
      --use_native_input_files \
      $OUTPUT_FLAGS \
      --num_inferences 3 \
      --perf_profile burst 2>&1 | tail -3" > _run.log

mkdir -p device_out
ssh "$DEVICE" "cat $DEV_DIR/out/qnn-profiling-data_0.log" > device_out/qnn-profiling-data_0.log
ssh "$DEVICE" "cat $DEV_DIR/out/Result_0/Y_native.raw 2>/dev/null || \
    cat $DEV_DIR/out/Result_0/Y.raw 2>/dev/null || \
    cat $DEV_DIR/out/Result_0/output_0_native.raw 2>/dev/null || \
    cat $DEV_DIR/out/Result_0/output_0.raw 2>/dev/null" \
    > device_out/Y.raw 2>/dev/null || true

echo "=== decode profile (text) ==="
LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang \
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-profile-viewer \
    --reader $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpProfilingReader.so \
    --input_log device_out/qnn-profiling-data_0.log \
    > device_out/profile.txt 2>&1

if [ "${DECODE_OPTRACE:-1}" = "1" ]; then
    echo "=== decode optrace artifacts ==="
    python "$ROOT_DIR/scripts/decode_qnn_optrace.py" "$OUT_DIR" || {
        [ "${STRICT_OPTRACE:-0}" = "1" ] && exit 1
        echo "  [warn] optrace decode failed; raw log kept in $OUT_DIR/device_out" >&2
    }
fi

CHECK_ARGS=("$OUT_DIR" --require-native-io --require-layout-flags --reject-float-io)
[ "${STRICT_ARTIFACT_STANDARD:-1}" = "0" ] && CHECK_ARGS+=(--warn-only)
python "$ROOT_DIR/scripts/check_qnn_artifact_standard.py" "${CHECK_ARGS[@]}"

echo "=== iter 3 (steady) per-MatMul cyc ==="
awk '/Number of HVX threads used : 4  count/{n++} n==3' device_out/profile.txt | grep -E "MatMul_|Accelerator \(execute\) time \(cycles\)|Accelerator \(execute\) time :|Input OpId|Output OpId" | head -20

echo "=== done. artefacts in $OUT_DIR ==="
