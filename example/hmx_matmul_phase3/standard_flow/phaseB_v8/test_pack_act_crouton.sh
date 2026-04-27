#!/usr/bin/env bash
# Test PackActCrouton at multiple (M, K) shapes; verify bit-exact against
# Python reference.
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

# (M, K) cases
SHAPES="${SHAPES:-32,128 64,128 32,256 128,256 32,512 256,512 32,1024 512,1024}"
mkdir -p pack_act_out

for shape in $SHAPES; do
    M="${shape%,*}"; K="${shape#*,}"
    echo "=== (M=$M, K=$K) ==="

    python gen_pack_act_crouton_test.py --M $M --K $K > /dev/null

    OUT_DIR="pack_act_out/M${M}_K${K}"; mkdir -p "$OUT_DIR"

    $QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter \
        -i pack_act_test.onnx \
        --op_package_config MatMulV8Package.xml \
        --converter_op_package_lib "$CPL" \
        --source_model_input_layout act_raw NONTRIVIAL \
        --source_model_output_layout crouton_out NONTRIVIAL \
        --desired_input_layout act_raw NONTRIVIAL \
        --desired_output_layout crouton_out NONTRIVIAL \
        -o "$OUT_DIR/test.dlc" 2>&1 > "$OUT_DIR/convert.log" || { echo "  [FAIL convert]"; tail -5 "$OUT_DIR/convert.log"; continue; }

    rm -rf "$OUT_DIR/ctx"
    $QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator \
        --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
        --dlc_path "$OUT_DIR/test.dlc" \
        --op_packages "$X86_PKG:HmxMatMulPhase3InterfaceProvider" \
        --binary_file test_ctx --output_dir "$OUT_DIR/ctx" \
        --config_file htp_config.json 2>&1 > "$OUT_DIR/ctxgen.log" || { echo "  [FAIL ctxgen]"; tail -5 "$OUT_DIR/ctxgen.log"; continue; }

    ssh oneplus "mkdir -p ~/qnn_run/spike/runtime_inputs_pack_act"
    ssh oneplus "cat > ~/qnn_run/spike/test_ctx.bin" < "$OUT_DIR/ctx/test_ctx.bin"
    ssh oneplus "cat > ~/qnn_run/spike/runtime_inputs_pack_act/act.raw" < runtime_inputs_pack_act/act.raw
    ssh oneplus "cat > ~/qnn_run/spike/input_list_pack_act.txt" < input_list_pack_act.txt

    ssh oneplus "cd ~/qnn_run/spike && rm -rf out && \
        LD_LIBRARY_PATH=../:.:/vendor/lib64 ADSP_LIBRARY_PATH=../ \
        ../qnn-net-run --backend ../libQnnHtp.so --retrieve_context test_ctx.bin \
          --op_packages ../libQnnHmxMatMulPhase3_cpu.so:HmxMatMulPhase3InterfaceProvider:CPU,../libQnnHmxMatMulPhase3_htp.so:HmxMatMulPhase3InterfaceProvider:HTP \
          --input_list input_list_pack_act.txt --profiling_level basic --config_file htp_config.json \
          --output_dir out --use_native_input_files --use_native_output_files --num_inferences 1 2>&1 | tail -2" \
        > "$OUT_DIR/run.log" 2>&1

    ssh oneplus "cat ~/qnn_run/spike/out/Result_0/crouton_out_native.raw" > "$OUT_DIR/skel.raw"

    python3 - <<PY
import numpy as np
ref = np.fromfile("pack_act_ref.raw", dtype=np.uint8)
skel = np.fromfile("${OUT_DIR}/skel.raw", dtype=np.uint8)
n = int((skel != ref).sum())
print(f"  M=$M K=$K → diffs={n}/{ref.size}", "PASS" if n == 0 else "FAIL")
if n > 0:
    d = np.where(skel != ref)[0]
    print(f"  first divergence at byte {d[0]}; skel={skel[d[0]]:02x} ref={ref[d[0]]:02x}")
PY
done
echo "=== Sweep done ==="
