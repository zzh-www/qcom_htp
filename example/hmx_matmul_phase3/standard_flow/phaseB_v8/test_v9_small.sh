#!/usr/bin/env bash
# test_v9_small.sh — end-to-end V9 graph test at small shape, byte-compare
# vs V8 at same shape (both should produce identical tile-layout output).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$ROOT_DIR/scripts/env.sh" >/dev/null
source "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH=$QNN_SDK_ROOT/lib/python PATH=$ANDROID_NDK_ROOT:$PATH
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}

CPL="$SCRIPT_DIR/gen_out/HmxMatMulPhase3Package_Converter_Op_Package/ConverterOpPackage/libConverterOpPackage.so"
X86_PKG="$(cd "$SCRIPT_DIR/../../build/x86_64-linux-clang" && pwd)/libQnnHmxMatMulPhase3.so"
cd "$SCRIPT_DIR"

M=${M:-32}; K=${K:-128}; N=${N:-128}
WORK="v9_small_test"
mkdir -p "$WORK"

# ---- (1) V9 ----
echo "=== V9 (M=$M K=$K N=$N) ==="
python gen_v9_test.py --M $M --K $K --N $N > /dev/null

$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter \
    -i v9_model.onnx --op_package_config MatMulV8Package.xml --converter_op_package_lib "$CPL" \
    --quantization_overrides quant_overrides.json \
    --source_model_input_layout act_raw NONTRIVIAL --source_model_output_layout out NONTRIVIAL \
    --desired_input_layout act_raw NONTRIVIAL --desired_output_layout out NONTRIVIAL \
    -o "$WORK/v9.dlc" > "$WORK/v9_convert.log" 2>&1

rm -rf "$WORK/v9_ctx" && mkdir "$WORK/v9_ctx"
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator \
    --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
    --dlc_path "$WORK/v9.dlc" \
    --op_packages "$X86_PKG:HmxMatMulPhase3InterfaceProvider" \
    --binary_file v9_ctx --output_dir "$WORK/v9_ctx" \
    --config_file htp_config.json > "$WORK/v9_ctxgen.log" 2>&1

# Push and run V9
ssh oneplus "mkdir -p ~/qnn_run/phaseB/runtime_inputs_v9"
ssh oneplus "cat > ~/qnn_run/phaseB/v9_ctx.bin" < "$WORK/v9_ctx/v9_ctx.bin"
ssh oneplus "cat > ~/qnn_run/phaseB/runtime_inputs_v9/act.raw"     < runtime_inputs_v9/act.raw
ssh oneplus "cat > ~/qnn_run/phaseB/runtime_inputs_v9/bias.raw"    < runtime_inputs_v9/bias.raw
ssh oneplus "cat > ~/qnn_run/phaseB/runtime_inputs_v9/scratch.raw" < runtime_inputs_v9/scratch.raw
ssh oneplus "cat > ~/qnn_run/phaseB/input_list_v9.txt"             < input_list_v9.txt

ssh oneplus "cd ~/qnn_run/phaseB && rm -rf out_v9 && \
    LD_LIBRARY_PATH=../:.:/vendor/lib64 ADSP_LIBRARY_PATH=../ \
    ../qnn-net-run --backend ../libQnnHtp.so --retrieve_context v9_ctx.bin \
      --op_packages ../libQnnHmxMatMulPhase3_cpu.so:HmxMatMulPhase3InterfaceProvider:CPU,../libQnnHmxMatMulPhase3_htp.so:HmxMatMulPhase3InterfaceProvider:HTP \
      --input_list input_list_v9.txt --profiling_level basic --config_file htp_config.json \
      --output_dir out_v9 --use_native_input_files --use_native_output_files --num_inferences 1 2>&1 | tail -3" \
    > "$WORK/v9_run.log" 2>&1

ssh oneplus "cat ~/qnn_run/phaseB/out_v9/Result_0/out_native.raw" > "$WORK/v9_out.raw"
echo "  V9 out: $(wc -c < $WORK/v9_out.raw) bytes"

# ---- (2) V8 at same shape ----
echo "=== V8 reference (M=$M K=$K N=$N) ==="
python gen_v8_graph.py --M $M --K $K --N $N > "$WORK/v8_gen.log" 2>&1
grep "M_TILE" "$WORK/v8_gen.log"

$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter \
    -i v8_model.onnx --op_package_config MatMulV8Package.xml --converter_op_package_lib "$CPL" \
    --quantization_overrides quant_overrides.json \
    --source_model_input_layout act_raw NONTRIVIAL --source_model_output_layout out NONTRIVIAL \
    --desired_input_layout act_raw NONTRIVIAL --desired_output_layout out NONTRIVIAL \
    -o "$WORK/v8.dlc" > "$WORK/v8_convert.log" 2>&1

rm -rf "$WORK/v8_ctx" && mkdir "$WORK/v8_ctx"
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator \
    --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
    --dlc_path "$WORK/v8.dlc" \
    --op_packages "$X86_PKG:HmxMatMulPhase3InterfaceProvider" \
    --binary_file v8_ctx --output_dir "$WORK/v8_ctx" \
    --config_file htp_config.json > "$WORK/v8_ctxgen.log" 2>&1

# Push and run V8 (re-uses runtime_inputs_v9 since V9 uses identical act/bias)
# Note: gen_v8_graph creates runtime_inputs_u8 which is identical-pattern to runtime_inputs_v9
ssh oneplus "cat > ~/qnn_run/phaseB/v8_ctx.bin" < "$WORK/v8_ctx/v8_ctx.bin"
ssh oneplus "mkdir -p ~/qnn_run/phaseB/runtime_inputs_u8"
ssh oneplus "cat > ~/qnn_run/phaseB/runtime_inputs_u8/act.raw"     < runtime_inputs_u8/act.raw
ssh oneplus "cat > ~/qnn_run/phaseB/runtime_inputs_u8/bias.raw"    < runtime_inputs_u8/bias.raw
ssh oneplus "cat > ~/qnn_run/phaseB/runtime_inputs_u8/scratch.raw" < runtime_inputs_u8/scratch.raw
ssh oneplus "cat > ~/qnn_run/phaseB/input_list.txt"                < input_list.txt

ssh oneplus "cd ~/qnn_run/phaseB && rm -rf out_v8 && \
    LD_LIBRARY_PATH=../:.:/vendor/lib64 ADSP_LIBRARY_PATH=../ \
    ../qnn-net-run --backend ../libQnnHtp.so --retrieve_context v8_ctx.bin \
      --op_packages ../libQnnHmxMatMulPhase3_cpu.so:HmxMatMulPhase3InterfaceProvider:CPU,../libQnnHmxMatMulPhase3_htp.so:HmxMatMulPhase3InterfaceProvider:HTP \
      --input_list input_list.txt --profiling_level basic --config_file htp_config.json \
      --output_dir out_v8 --use_native_input_files --use_native_output_files --num_inferences 1 2>&1 | tail -3" \
    > "$WORK/v8_run.log" 2>&1

ssh oneplus "cat ~/qnn_run/phaseB/out_v8/Result_0/out_native.raw" > "$WORK/v8_out.raw"
echo "  V8 out: $(wc -c < $WORK/v8_out.raw) bytes"

# ---- (3) compare ----
python3 - <<PY
import numpy as np
v9 = np.fromfile("$WORK/v9_out.raw", dtype=np.uint8)
v8 = np.fromfile("$WORK/v8_out.raw", dtype=np.uint8)
print(f"V9: {v9.shape}, V8: {v8.shape}")
if v9.size != v8.size:
    print(f"SIZE MISMATCH"); exit(1)
diff = (v9 != v8).sum()
print(f"diffs = {diff} / {v9.size}", "PASS" if diff == 0 else "FAIL")
if diff > 0:
    d = np.where(v9 != v8)[0]
    print(f"  first diff at byte {d[0]}: v9={v9[d[0]]} v8={v8[d[0]]}")
    print(f"  max abs delta = {np.abs(v9.astype(int) - v8.astype(int)).max()}")
PY
