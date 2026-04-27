#!/usr/bin/env bash
# run_v8c8_phase2.sh — V8 C8 alignment Phase 2 device test.
#
# Builds artifacts for gen_v8c8_chain.py --chain "${CHAIN:-8}" (256³ NOOP body) using the
# Crouton_8 + Indirect signature on in[0], runs on device, pulls back
# the marker bytes from output[0..15] to confirm the kernel actually
# ran.
#
# Prereqs:
#   - bash build.sh        with EXTRA_DEFS=-DV9_C8_ALIGNMENT_TEST
#   - bash build_x86.sh    with EXTRA_DEFS=-DV9_C8_ALIGNMENT_TEST
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$ROOT_DIR/scripts/env.sh" >/dev/null
source "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH=$QNN_SDK_ROOT/lib/python
export PATH=$ANDROID_NDK_ROOT:$PATH
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}

DEVICE="${DEVICE:-oneplus}"
M="${M:-256}"
K="${K:-256}"
N="${N:-256}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/phase1_validation/v8c8_test}"
export OUT_DIR M K N
mkdir -p "$OUT_DIR"

CPL="$SCRIPT_DIR/gen_out/HmxMatMulPhase3Package_Converter_Op_Package/ConverterOpPackage/libConverterOpPackage.so"
X86_PKG="$(cd "$SCRIPT_DIR/../../build/x86_64-linux-clang" && pwd)/libQnnHmxMatMulPhase3.so"
PKG_HTP="$SCRIPT_DIR/../../build/hexagon-v75/libQnnHmxMatMulPhase3_htp.so"
PKG_CPU="$SCRIPT_DIR/../../build/aarch64/libQnnHmxMatMulPhase3_cpu.so"

cd "$SCRIPT_DIR"

CHAIN_N="${CHAIN:-8}"
MODE_NAME="${MODE:-chain}"
echo "=== [1/5] gen_v8c8_chain.py --chain $CHAIN_N --mode $MODE_NAME ${M}×${K}×${N} ==="
python gen_v8c8_chain.py --chain "$CHAIN_N" --mode "$MODE_NAME" \
    --M "$M" --K "$K" --N "$N" -o "$OUT_DIR/v8c8.onnx"

# Build qairt-converter input/output layout flags. Independent mode has
# multiple input/output tensors (act_raw, act_raw_1, ..., out_0, out_1, ...).
LAYOUT_FLAGS=()
if [ "$MODE_NAME" = "chain" ]; then
    LAYOUT_FLAGS+=(--source_model_input_layout  act_raw NONTRIVIAL
                   --desired_input_layout       act_raw NONTRIVIAL
                   --source_model_output_layout out     NONTRIVIAL
                   --desired_output_layout      out     NONTRIVIAL)
else
    for i in $(seq 0 $((CHAIN_N - 1))); do
        IN_NAME="act_raw"; [ "$i" != "0" ] && IN_NAME="act_raw_${i}"
        OUT_NAME="out_${i}"
        LAYOUT_FLAGS+=(--source_model_input_layout  "$IN_NAME"  NONTRIVIAL
                       --desired_input_layout       "$IN_NAME"  NONTRIVIAL
                       --source_model_output_layout "$OUT_NAME" NONTRIVIAL
                       --desired_output_layout      "$OUT_NAME" NONTRIVIAL)
    done
fi

echo "=== [2/5] qairt-converter ==="
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter \
    -i "$OUT_DIR/v8c8.onnx" \
    --op_package_config MatMulV8Package.xml \
    --converter_op_package_lib "$CPL" \
    --quantization_overrides "$OUT_DIR/quant_overrides.json" \
    "${LAYOUT_FLAGS[@]}" \
    -o "$OUT_DIR/v8c8.dlc" 2>&1 | tee "$OUT_DIR/convert.log" | tail -3

echo "=== [3/5] qnn-context-binary-generator ==="
rm -rf "$OUT_DIR/ctx"
# --profiling_level detailed + --profiling_option optrace is what makes
# ctxgen actually emit schematic.bin (alongside the bottom_mapping.json).
# Without these two flags, schematic is processed in-memory and discarded —
# so optrace decode on the runtime profile log later fails.
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator \
    --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
    --dlc_path "$OUT_DIR/v8c8.dlc" \
    --op_packages "$X86_PKG:HmxMatMulPhase3InterfaceProvider" \
    --binary_file v8c8_ctx --output_dir "$OUT_DIR/ctx" \
    --config_file htp_config.json \
    --profiling_level detailed --profiling_option optrace \
    --save_backend_op_mapping 2>&1 | tee "$OUT_DIR/ctxgen.log" | tail -3
# Schematic lands in CWD (which is SCRIPT_DIR after our cd above) under
# the input model basename → "v8c8_schematic.bin". Move into OUT_DIR/ctx
# for self-contained per-shape artifacts.
if [ -f "$SCRIPT_DIR/v8c8_schematic.bin" ]; then
    mv "$SCRIPT_DIR/v8c8_schematic.bin" "$OUT_DIR/ctx/v8c8_schematic.bin"
elif [ -f "v8c8_schematic.bin" ]; then
    mv v8c8_schematic.bin "$OUT_DIR/ctx/v8c8_schematic.bin"
fi
ls -la "$OUT_DIR/ctx/" | grep -i schem || echo "  (warn: no schematic.bin produced)"

echo "  --- ctxgen node summary ---"
python3 -c "
import json
d = json.load(open('$OUT_DIR/ctx/v8c8_ctx_bottom_mapping.json'))
from collections import Counter
print(Counter(n['type'] for n in d['graph']['nodes'].values()))
" || true

echo "=== [4/5] push and run on device ==="
ssh "$DEVICE" "mkdir -p qnn_run/phaseB_c8/runtime_inputs_u8"
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/v8c8_ctx.bin" < "$OUT_DIR/ctx/v8c8_ctx.bin"
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/htp_config.json" < htp_config.json
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/htp_backend_ext.json" < htp_backend_ext.json
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/libQnnHmxMatMulPhase3_htp.so" < "$PKG_HTP"
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/libQnnHmxMatMulPhase3_cpu.so" < "$PKG_CPU"

# Push all act .raw files from runtime_inputs_u8/ (chain = 1 file, independent = N files)
for f in runtime_inputs_u8/act_v8c8*.raw; do
    [ -f "$f" ] || continue
    ssh "$DEVICE" "cat > qnn_run/phaseB_c8/runtime_inputs_u8/$(basename "$f")" < "$f"
done

# Build input_list.txt — single line with all inputs separated by spaces
if [ "$MODE_NAME" = "chain" ]; then
    echo 'act_raw:=runtime_inputs_u8/act_v8c8.raw' > "$OUT_DIR/input_list.txt"
else
    line="act_raw:=runtime_inputs_u8/act_v8c8.raw"
    for i in $(seq 1 $((CHAIN_N - 1))); do
        line="$line act_raw_${i}:=runtime_inputs_u8/act_v8c8_${i}.raw"
    done
    echo "$line" > "$OUT_DIR/input_list.txt"
fi
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/input_list.txt" < "$OUT_DIR/input_list.txt"

# Clear logcat just before the run so we can capture device messages
ssh "$DEVICE" 'logcat -c 2>/dev/null || true'

# Replace the parent-dir pkg .so with our freshly-built ones so the
# proven `../libQnn...` path picks up the C8-aligned signatures.
ssh "$DEVICE" "cat > qnn_run/libQnnHmxMatMulPhase3_htp.so" < "$PKG_HTP"
ssh "$DEVICE" "cat > qnn_run/libQnnHmxMatMulPhase3_cpu.so" < "$PKG_CPU"

ssh "$DEVICE" "cd qnn_run/phaseB_c8 && rm -rf out && \
    LD_LIBRARY_PATH=../:.:/vendor/lib64 ADSP_LIBRARY_PATH=../ \
    ../qnn-net-run \
      --backend ../libQnnHtp.so \
      --retrieve_context v8c8_ctx.bin \
      --op_packages ../libQnnHmxMatMulPhase3_cpu.so:HmxMatMulPhase3InterfaceProvider:CPU,../libQnnHmxMatMulPhase3_htp.so:HmxMatMulPhase3InterfaceProvider:HTP \
      --input_list input_list.txt \
      --profiling_level detailed --profiling_option optrace \
      --output_dir out \
      --config_file htp_config.json \
      --use_native_input_files \
      --num_inferences 3 \
      --perf_profile burst 2>&1" > "$OUT_DIR/run.log" 2>&1 || true

echo "  --- last 30 lines of run.log ---"
tail -30 "$OUT_DIR/run.log"

echo "  --- DSP-side logcat (last 60s, filtered) ---"
ssh "$DEVICE" 'logcat -d -t 1000 2>&1' \
    | grep -iE 'qnn|adsprpc|dsp|skel|fastrpc|ssr|sub_pd|crash|fault|err' \
    | grep -v 'avc:' \
    | tail -40 \
    > "$OUT_DIR/logcat.txt" || true
head -40 "$OUT_DIR/logcat.txt" || true

echo "=== [5/5] pull and decode output ==="
mkdir -p "$OUT_DIR/device_out"
ssh "$DEVICE" "ls qnn_run/phaseB_c8/out/Result_0/ 2>&1" | tee "$OUT_DIR/device_out/result_listing.txt"
ssh "$DEVICE" 'cat qnn_run/phaseB_c8/out/Result_0/out.raw 2>/dev/null' \
    > "$OUT_DIR/device_out/out.raw" 2>/dev/null || true
if [ -s "$OUT_DIR/device_out/out.raw" ]; then
    echo "  --- decode output (fp32-dequantized rank-3 row-major) ---"
    python3 - <<'PY'
import numpy as np, os
b = np.fromfile(os.environ["OUT_DIR"] + "/device_out/out.raw", dtype=np.float32)
print(f"  size: {b.nbytes} bytes  ({b.size} fp32 cells)")
M, N = int(os.environ["M"]), int(os.environ["N"])
expected_cells = M * N
print(f"  expected: {M}×{N} = {expected_cells} u8 cells = {expected_cells*4} fp32 bytes")
print(f"  shape match: {b.size == expected_cells}")

# After UntileToRowMajor + Reshape, marker (BbbKMajor wrote at out_buf[0..15]
# in tile-layout) lands at row-major positions out[0..15] (tile mt=0,nt=0,
# row 0, cols 0..15). Decode marker.
u8 = np.round(b[:16]).astype(int).tolist()
print(f"  marker[0..15]: {u8}")
print(f"    [0]  magic        = 0x{u8[0]:02x} (expect 0xa5)")
print(f"    [1]  in[0] layout = {u8[1]} (Crouton_8 = 9)")
print(f"    [2]  out  layout  = {u8[2]} (Flat4 = 2)")
print(f"    [3]  num_inputs   = {u8[3]}")
print(f"    [4]  output rank  = {u8[4]}")
print(f"    [5..8] dims       = {u8[5:9]}")
print(f"    [15] tail magic   = 0x{u8[15]:02x} (expect 0x5a)")
PY
else
    echo "  output file empty or missing — kernel likely never ran"
fi

echo "=== done. artefacts in $OUT_DIR ==="
