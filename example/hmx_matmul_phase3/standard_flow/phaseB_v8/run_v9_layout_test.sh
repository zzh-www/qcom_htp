#!/usr/bin/env bash
# run_v9_layout_test.sh — verify V9_KERNEL_HMX bit-exact under non-saturating
# inputs (output[m,n] = m & 0xF). Detects Crouton_8 byte-mapping bugs
# hidden by saturation in the default (random-input) test.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$ROOT_DIR/scripts/env.sh" >/dev/null
source "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH=$QNN_SDK_ROOT/lib/python
export PATH=$ANDROID_NDK_ROOT:$PATH
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}

DEVICE="${DEVICE:-oneplus}"
M="${M:-256}"; K="${K:-256}"; N="${N:-256}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/phase1_validation/v9_layout_test}"
export OUT_DIR M K N
mkdir -p "$OUT_DIR"

CPL="$SCRIPT_DIR/gen_out/HmxMatMulPhase3Package_Converter_Op_Package/ConverterOpPackage/libConverterOpPackage.so"
X86_PKG="$(cd "$SCRIPT_DIR/../../build/x86_64-linux-clang" && pwd)/libQnnHmxMatMulPhase3.so"
PKG_HTP="$SCRIPT_DIR/../../build/hexagon-v75/libQnnHmxMatMulPhase3_htp.so"
PKG_CPU="$SCRIPT_DIR/../../build/aarch64/libQnnHmxMatMulPhase3_cpu.so"

cd "$SCRIPT_DIR"

echo "=== [1/5] gen_v8c8_test.py --mode layout_test ${M}×${K}×${N} ==="
python gen_v8c8_test.py --M "$M" --K "$K" --N "$N" --mode layout_test -o "$OUT_DIR/v8c8.onnx" 2>&1 | tail -3

echo "=== [2/5] qairt-converter ==="
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter \
    -i "$OUT_DIR/v8c8.onnx" \
    --op_package_config MatMulV8Package.xml \
    --converter_op_package_lib "$CPL" \
    --quantization_overrides quant_overrides.json \
    --source_model_input_layout act_raw NONTRIVIAL \
    --source_model_output_layout out NONTRIVIAL \
    --desired_input_layout act_raw NONTRIVIAL \
    --desired_output_layout out NONTRIVIAL \
    -o "$OUT_DIR/v8c8.dlc" 2>&1 | tee "$OUT_DIR/convert.log" | tail -3

echo "=== [3/5] qnn-context-binary-generator ==="
rm -rf "$OUT_DIR/ctx"
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator \
    --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
    --dlc_path "$OUT_DIR/v8c8.dlc" \
    --op_packages "$X86_PKG:HmxMatMulPhase3InterfaceProvider" \
    --binary_file v8c8_ctx --output_dir "$OUT_DIR/ctx" \
    --config_file htp_config.json 2>&1 | tee "$OUT_DIR/ctxgen.log" | tail -3

echo "=== [4/5] push and run on device ==="
ssh "$DEVICE" "mkdir -p qnn_run/phaseB_c8/runtime_inputs_u8"
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/v8c8_ctx.bin" < "$OUT_DIR/ctx/v8c8_ctx.bin"
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/runtime_inputs_u8/act_v8c8.raw" < runtime_inputs_u8/act_v8c8.raw
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/htp_config.json" < htp_config.json
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/htp_backend_ext.json" < htp_backend_ext.json
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/libQnnHmxMatMulPhase3_htp.so" < "$PKG_HTP"
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/libQnnHmxMatMulPhase3_cpu.so" < "$PKG_CPU"
ssh "$DEVICE" "cat > qnn_run/libQnnHmxMatMulPhase3_htp.so" < "$PKG_HTP"
ssh "$DEVICE" "cat > qnn_run/libQnnHmxMatMulPhase3_cpu.so" < "$PKG_CPU"
echo 'act_raw:=runtime_inputs_u8/act_v8c8.raw' > "$OUT_DIR/input_list.txt"
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/input_list.txt" < "$OUT_DIR/input_list.txt"

ssh "$DEVICE" "cd qnn_run/phaseB_c8 && rm -rf out && \
    LD_LIBRARY_PATH=../:.:/vendor/lib64 ADSP_LIBRARY_PATH=../ \
    ../qnn-net-run \
      --backend ../libQnnHtp.so \
      --retrieve_context v8c8_ctx.bin \
      --op_packages ../libQnnHmxMatMulPhase3_cpu.so:HmxMatMulPhase3InterfaceProvider:CPU,../libQnnHmxMatMulPhase3_htp.so:HmxMatMulPhase3InterfaceProvider:HTP \
      --input_list input_list.txt \
      --output_dir out \
      --config_file htp_config.json \
      --use_native_input_files \
      --num_inferences 1 \
      --perf_profile burst 2>&1" > "$OUT_DIR/run.log" 2>&1 || true
tail -10 "$OUT_DIR/run.log"

mkdir -p "$OUT_DIR/device_out"
ssh "$DEVICE" 'cat qnn_run/phaseB_c8/out/Result_0/out.raw 2>/dev/null' > "$OUT_DIR/device_out/out.raw" 2>/dev/null || true

echo "=== [5/5] compare to reference (output[m,n] = m & 0xF) ==="
python3 - <<'PY'
import numpy as np, os
M, N = int(os.environ["M"]), int(os.environ["N"])
b = np.fromfile(os.environ["OUT_DIR"] + "/device_out/out.raw", dtype=np.float32)
dev = np.round(b).astype(int).clip(0,255).astype(np.uint8).reshape(M, N)
ref = np.load(os.environ["OUT_DIR"] + "/v8c8.onnx.out_ref_u8.npy")

print(f"  ref[:8, 0]   = {ref[:8, 0].tolist()}")
print(f"  dev[:8, 0]   = {dev[:8, 0].tolist()}")
print(f"  ref[8:16, 0] = {ref[8:16, 0].tolist()}")
print(f"  dev[8:16, 0] = {dev[8:16, 0].tolist()}")
print(f"  ref[32:40,0] = {ref[32:40, 0].tolist()}")
print(f"  dev[32:40,0] = {dev[32:40, 0].tolist()}")
print()

match = (dev == ref).sum()
total = M * N
print(f"Bit-exact: {match}/{total} ({100*match/total:.2f}%)")

if match < total:
    diff_rows = (dev != ref).any(axis=1)
    bad = np.where(diff_rows)[0]
    print(f"Rows with any diff: {diff_rows.sum()}/{M}, first 16: {bad[:16].tolist()}")
    # Try to detect a permutation: dev[r] should equal ref[?]
    print()
    print("Detecting per-row permutation (looking for matching reference rows):")
    for r in [0, 1, 7, 8, 15, 16, 31, 32, 33]:
        if r < M:
            dev_row = dev[r]
            # Find ref rows that match this dev row
            match_rows = [m for m in range(M) if np.array_equal(ref[m], dev_row)]
            print(f"  device row {r} (first cell={dev_row[0]}) matches ref rows: {match_rows[:8]}")
PY
echo "=== done ==="
