#!/usr/bin/env bash
#
# Generate and run a native W4A16 Conv graph with a diagnostic QHPI tensor dump
# immediately after the Conv.  The output directory keeps converter logs,
# ctxgen mapping, device output, decoded optrace, and parsed dump JSON together.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
EXAMPLE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env.sh" >/dev/null
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/qairt_quant_flow.sh"
# shellcheck disable=SC1091
source "$ROOT_DIR/.venv/bin/activate"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/dssh.sh"

export PYTHONPATH="$QNN_SDK_ROOT/lib/python"
export PATH="$ANDROID_NDK_ROOT:$PATH"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"

DEVICE="${DEVICE:-oneplus}"
NATIVE_DIR="${NATIVE_DIR:-$ROOT_DIR/example/qnn_matmul_profile/output_codex_native_w4a16_same_custom_256}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/out/native_conv_tensor_dump_256}"
SKIP_DEVICE="${SKIP_DEVICE:-0}"
NUM_INFERENCES="${NUM_INFERENCES:-3}"
PACK_4BIT_WEIGHTS="${PACK_4BIT_WEIGHTS:-1}"
NATIVE_DIR="$(realpath -m "$NATIVE_DIR")"
OUT_DIR="$(realpath -m "$OUT_DIR")"

mkdir -p "$OUT_DIR"
cd "$SCRIPT_DIR"
cp htp_config.json "$OUT_DIR/htp_config.json"
cp htp_backend_ext.json "$OUT_DIR/htp_backend_ext.json"

X86_PKG="$EXAMPLE_DIR/build/x86_64-linux-clang/libQnnHmxMatMulW4A16.so"
PKG_HTP="$EXAMPLE_DIR/build/hexagon-v75/libQnnHmxMatMulW4A16_htp.so"
PKG_CPU="$EXAMPLE_DIR/build/aarch64/libQnnHmxMatMulW4A16_cpu.so"
CPL="$SCRIPT_DIR/converter/build/libConverterOpPackage.so"

for f in "$X86_PKG" "$PKG_HTP" "$PKG_CPU"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f; run $EXAMPLE_DIR/build.sh and build_x86.sh first" >&2
        exit 1
    fi
done

echo "=== [1/5] generate native Conv + HmxW4A16TensorDump graph ==="
python gen_native_conv_tensor_dump.py \
    --native-dir "$NATIVE_DIR" \
    --out-dir "$OUT_DIR"

echo "=== [2/5] build converter op package ==="
mkdir -p "$SCRIPT_DIR/converter/build"
"${CXX:-clang++}" -std=c++17 -O2 -shared -fPIC \
    -I "$QNN_SDK_ROOT/include/QNN" \
    -o "$CPL" \
    "$SCRIPT_DIR/converter/ConverterOpPackage.cpp"
echo "  -> $CPL"

echo "=== [3/5] qairt-converter -> qairt-quantizer ==="
"$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter" \
    -i "$OUT_DIR/native_conv_tensor_dump.onnx" \
    --target_backend HTP \
    --enable_framework_trace \
    --preserve_io_datatype A Y \
    --op_package_config QnnHmxMatMulW4A16Package.xml \
    --converter_op_package_lib "$CPL" \
    --quantization_overrides "$OUT_DIR/quant_overrides.json" \
    -o "$OUT_DIR/native_conv_tensor_dump_encoded.dlc" \
    2>&1 | tee "$OUT_DIR/convert.log" | tail -8
qairt_quantize_encoded_dlc \
    "$OUT_DIR/native_conv_tensor_dump_encoded.dlc" \
    "$OUT_DIR/native_conv_tensor_dump.dlc" \
    16 4 32 "$PACK_4BIT_WEIGHTS" \
    "$OUT_DIR/quantize.log"
tail -8 "$OUT_DIR/quantize.log"

echo "=== [4/5] qnn-context-binary-generator ==="
rm -rf "$OUT_DIR/ctx"
mkdir -p "$OUT_DIR/ctx"
rm -f "$OUT_DIR"/*_schematic.bin "$OUT_DIR"/schematic.bin "$OUT_DIR"/model_schematic.bin
(
    cd "$OUT_DIR"
    "$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator" \
        --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
        --dlc_path "$OUT_DIR/native_conv_tensor_dump.dlc" \
        --op_packages "$X86_PKG:QnnHmxMatMulW4A16InterfaceProvider" \
        --binary_file native_conv_tensor_dump_ctx \
        --output_dir "$OUT_DIR/ctx" \
        --config_file "$OUT_DIR/htp_config.json" \
        --profiling_level detailed \
        --profiling_option optrace \
        --save_backend_op_mapping
) 2>&1 | tee "$OUT_DIR/ctxgen.log" | tail -8

find "$OUT_DIR" -maxdepth 1 -name '*schematic.bin' -type f -exec mv {} "$OUT_DIR/ctx/" \;

MAPPING_JSON="$OUT_DIR/ctx/native_conv_tensor_dump_ctx_bottom_mapping.json"
if [ ! -f "$MAPPING_JSON" ]; then
    MAPPING_JSON="$(find "$OUT_DIR/ctx" -maxdepth 1 -name '*bottom_mapping.json' -print -quit)"
fi
if [ -n "$MAPPING_JSON" ] && [ -f "$MAPPING_JSON" ]; then
    python3 - "$MAPPING_JSON" <<'PY' || true
import json
import sys
from collections import Counter

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
graph = data.get("graph", {})
nodes = graph.get("nodes", {})
tensors = graph.get("tensors", {})
print("  ctx nodes:", Counter(n.get("type") for n in nodes.values()))
for name, node in nodes.items():
    typ = str(node.get("type", ""))
    if typ == "q::ConvLayer_s1.opt" or "HmxW4A16TensorDump" in typ:
        print(f"  boundary: {name}: {typ}")
        for label, tensor_ids in (
            ("inputs", node.get("input_names", [])),
            ("outputs", node.get("output_names", [])),
        ):
            for idx, tensor_id in enumerate(tensor_ids):
                tensor = tensors.get(tensor_id, {})
                print(
                    f"    {label}[{idx}] {tensor_id}: "
                    f"data_type={tensor.get('data_type')} dims={tensor.get('dims')}"
                )
PY
fi

if [ "$SKIP_DEVICE" = "1" ]; then
    echo "=== done: ctx artifacts in $OUT_DIR ==="
    exit 0
fi

echo "=== [5/5] device run on $DEVICE ==="
REMOTE="qnn_run/native_w4a16_tensor_dump"
ssh "$DEVICE" "mkdir -p $REMOTE"
ssh "$DEVICE" "cat > $REMOTE/native_conv_tensor_dump_ctx.bin" < "$OUT_DIR/ctx/native_conv_tensor_dump_ctx.bin"
ssh "$DEVICE" "cat > $REMOTE/htp_config.json" < htp_config.json
ssh "$DEVICE" "cat > $REMOTE/htp_backend_ext.json" < htp_backend_ext.json
ssh "$DEVICE" "cat > $REMOTE/input_A.raw" < "$OUT_DIR/input_A.raw"
ssh "$DEVICE" "cat > $REMOTE/input_list.txt" < "$OUT_DIR/input_list.txt"
ssh "$DEVICE" "cat > qnn_run/libQnnHmxMatMulW4A16_htp.so" < "$PKG_HTP"
ssh "$DEVICE" "cat > qnn_run/libQnnHmxMatMulW4A16_cpu.so" < "$PKG_CPU"

RUN_STATUS=0
ssh "$DEVICE" "cd $REMOTE && rm -rf out && LD_LIBRARY_PATH=../:.:/vendor/lib64 ADSP_LIBRARY_PATH=../ ../qnn-net-run --backend ../libQnnHtp.so --retrieve_context native_conv_tensor_dump_ctx.bin --op_packages ../libQnnHmxMatMulW4A16_cpu.so:QnnHmxMatMulW4A16InterfaceProvider:CPU,../libQnnHmxMatMulW4A16_htp.so:QnnHmxMatMulW4A16InterfaceProvider:HTP --input_list input_list.txt --profiling_level detailed --profiling_option optrace --output_dir out --config_file htp_config.json --use_native_output_files --num_inferences $NUM_INFERENCES --perf_profile burst 2>&1" \
    > "$OUT_DIR/run.log" 2>&1 || RUN_STATUS=$?

tail -50 "$OUT_DIR/run.log"
rm -rf "$OUT_DIR/device_out"
mkdir -p "$OUT_DIR/device_out"
ssh "$DEVICE" "find $REMOTE/out -maxdepth 2 -type f -print 2>/dev/null" > "$OUT_DIR/device_out/remote_files.txt" 2>/dev/null || true
ssh "$DEVICE" "cat $REMOTE/out/Result_0/D_native.raw 2>/dev/null || cat $REMOTE/out/Result_0/D.raw 2>/dev/null || cat $REMOTE/out/Result_0/Y_native.raw 2>/dev/null || cat $REMOTE/out/Result_0/Y.raw 2>/dev/null" > "$OUT_DIR/device_out/D.raw" 2>/dev/null || true
ssh "$DEVICE" "cat $REMOTE/out/qnn-profiling-data_0.log 2>/dev/null" > "$OUT_DIR/device_out/qnn-profiling-data_0.log" 2>/dev/null || true

if [ "${DECODE_OPTRACE:-1}" = "1" ]; then
    echo "=== decode optrace artifacts ==="
    python "$ROOT_DIR/scripts/decode_qnn_optrace.py" "$OUT_DIR" || {
        [ "${STRICT_OPTRACE:-0}" = "1" ] && exit 1
        echo "  [warn] optrace decode failed; raw log kept in $OUT_DIR/device_out" >&2
    }
fi

if [ -s "$OUT_DIR/device_out/D.raw" ]; then
    BYTES="$(wc -c < "$OUT_DIR/device_out/D.raw")"
    DUMP_DTYPE="u16"
    if [ "$BYTES" -eq 262144 ]; then
        DUMP_DTYPE="f32"
    fi
    python "$ROOT_DIR/scripts/parse_w4a16_tensor_dump.py" \
        "$OUT_DIR/device_out/D.raw" \
        --cols 256 \
        --dtype "$DUMP_DTYPE" \
        --json \
        > "$OUT_DIR/device_out/tensor_dump.json"
    python "$ROOT_DIR/scripts/parse_w4a16_tensor_dump.py" \
        "$OUT_DIR/device_out/D.raw" \
        --cols 256 \
        --dtype "$DUMP_DTYPE" \
        > "$OUT_DIR/device_out/tensor_dump.txt"
    sed -n '1,80p' "$OUT_DIR/device_out/tensor_dump.txt"
else
    echo "ERROR: missing dumped D.raw under $OUT_DIR/device_out" >&2
    RUN_STATUS=1
fi

if [ "$RUN_STATUS" != "0" ]; then
    echo "ERROR: device run failed (run=$RUN_STATUS)" >&2
    exit 1
fi

echo "=== done: artifacts in $OUT_DIR ==="
