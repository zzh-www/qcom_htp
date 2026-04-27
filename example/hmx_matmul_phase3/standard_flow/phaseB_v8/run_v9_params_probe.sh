#!/usr/bin/env bash
# run_v9_params_probe.sh — Step 5.1 set_hmx_params_conv1x1 characterization probe.
#
# Builds with -DV9_C8_ALIGNMENT_TEST -DV9_PARAMS_PROBE, generates the standard
# v8c8 256³ ONNX, runs on device, and decodes the first 16 rows of output
# (16 cases × 128 bytes/row) into the dlsym'd-native-function-output bytes
# for each (arg1..arg5) combination.
#
# Result tells us which arg bits flip which descriptor bytes — characterizes
# the args without manually transliterating ~70-line set_hmx_params_conv1x1
# disasm.
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
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/phase1_validation/v9_params_probe}"
export OUT_DIR M K N
mkdir -p "$OUT_DIR"

CPL="$SCRIPT_DIR/gen_out/HmxMatMulPhase3Package_Converter_Op_Package/ConverterOpPackage/libConverterOpPackage.so"
X86_PKG="$(cd "$SCRIPT_DIR/../../build/x86_64-linux-clang" && pwd)/libQnnHmxMatMulPhase3.so"
PKG_HTP="$SCRIPT_DIR/../../build/hexagon-v75/libQnnHmxMatMulPhase3_htp.so"
PKG_CPU="$SCRIPT_DIR/../../build/aarch64/libQnnHmxMatMulPhase3_cpu.so"

cd "$SCRIPT_DIR"

echo "=== [0/5] (re)build with V9_PARAMS_PROBE ==="
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_PARAMS_PROBE" \
    bash "$SCRIPT_DIR/../../build.sh" 2>&1 | tail -5
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_PARAMS_PROBE" \
    bash "$SCRIPT_DIR/../../build_x86.sh" 2>&1 | tail -5

echo "=== [1/5] gen_v8c8_test.py ${M}×${K}×${N} ==="
python gen_v8c8_test.py --M "$M" --K "$K" --N "$N" -o "$OUT_DIR/v8c8.onnx" 2>&1 | tail -3

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

ssh "$DEVICE" 'logcat -c 2>/dev/null || true'
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

echo "  --- last 20 lines of run.log ---"
tail -20 "$OUT_DIR/run.log"

mkdir -p "$OUT_DIR/device_out"
ssh "$DEVICE" 'cat qnn_run/phaseB_c8/out/Result_0/out.raw 2>/dev/null' \
    > "$OUT_DIR/device_out/out.raw" 2>/dev/null || true

echo "=== [5/5] decode 16 cases × 128 bytes ==="
if [ ! -s "$OUT_DIR/device_out/out.raw" ]; then
    echo "  output file empty — kernel likely never ran"
    exit 1
fi
python3 - <<'PY'
import numpy as np, os, struct
M = int(os.environ["M"])
N = int(os.environ["N"])
b = np.fromfile(os.environ["OUT_DIR"] + "/device_out/out.raw", dtype=np.float32)
print(f"  raw output size: {b.nbytes} B  cells={b.size}")
img = np.round(b).astype(np.int64).clip(0,255).astype(np.uint8).reshape(M, N)
print(f"  decoded as [{M},{N}] u8")
print()
print("  ┌─ probe table ─────────────────────────────────────────────")
print("  │ idx  arg1   arg2   arg3   arg4   arg5    → 0x40 desc bytes (hex)")
print("  ├───────────────────────────────────────────────────────────")
for i in range(16):
    if i >= M:
        break
    row = bytes(img[i, :128].tolist())
    a1, a2, a3, a4, a5 = struct.unpack_from("<5I", row, 0)
    magic, = struct.unpack_from("<I", row, 20)
    if magic != 0xDEADBEEF:
        print(f"  │ {i:2d}: BAD magic {magic:#x} (kernel may not have written this row)")
        continue
    desc = row[24:88]
    desc_hex = " ".join(f"{b:02x}" for b in desc)
    print(f"  │ {i:2d}: {a1:#06x} {a2:#06x} {a3:#06x} {a4:#06x} {a5:#06x}")
    # Group desc as 16 dwords (4 bytes ea)
    desc_dw = struct.unpack_from("<16I", desc, 0)
    for off, dw in enumerate(desc_dw):
        if dw != 0:
            print(f"  │      +0x{off*4:02x}: 0x{dw:08x}")
print("  └───────────────────────────────────────────────────────────")
PY

echo "=== done. artefacts in $OUT_DIR ==="
