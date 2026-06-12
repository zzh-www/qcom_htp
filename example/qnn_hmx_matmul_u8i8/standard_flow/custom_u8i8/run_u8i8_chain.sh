#!/usr/bin/env bash
#
# Generate, convert, ctxgen, and optionally run the custom u8/i8 HMX MatMul
# chain.  Set SKIP_DEVICE=1 to stop after context-binary generation.

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
M="${M:-256}"
K="${K:-256}"
N="${N:-256}"
CHAIN="${CHAIN:-8}"
MODE="${MODE:-chain}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/out/u8i8_${MODE}_${M}}"
SKIP_DEVICE="${SKIP_DEVICE:-0}"
NATIVE_OUTPUT="${NATIVE_OUTPUT:-1}"
VERIFY_REF="${VERIFY_REF:-1}"

mkdir -p "$OUT_DIR"
cd "$SCRIPT_DIR"

X86_PKG="$EXAMPLE_DIR/build/x86_64-linux-clang/libQnnHmxMatMulU8I8.so"
PKG_HTP="$EXAMPLE_DIR/build/hexagon-v75/libQnnHmxMatMulU8I8_htp.so"
PKG_CPU="$EXAMPLE_DIR/build/aarch64/libQnnHmxMatMulU8I8_cpu.so"
CPL="$SCRIPT_DIR/converter/build/libConverterOpPackage.so"

for f in "$X86_PKG" "$PKG_HTP" "$PKG_CPU"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f; run $EXAMPLE_DIR/build.sh and build_x86.sh first" >&2
        exit 1
    fi
done

echo "=== [1/4] generate ONNX: HmxU8I8ToU8MatMul x $CHAIN ($MODE, ${M}x${K}x${N}) ==="
python gen_u8i8_chain.py \
    --chain "$CHAIN" --mode "$MODE" \
    --M "$M" --K "$K" --N "$N" \
    ${GEN_EXTRA_ARGS:-} \
    -o "$OUT_DIR/u8i8.onnx"

echo "=== [2/4] build converter op package ==="
mkdir -p "$SCRIPT_DIR/converter/build"
"${CXX:-clang++}" -std=c++17 -O2 -shared -fPIC \
    -I "$QNN_SDK_ROOT/include/QNN" \
    -o "$CPL" \
    "$SCRIPT_DIR/converter/ConverterOpPackage.cpp"
echo "  -> $CPL"

LAYOUT_FLAGS=()
if [ "$MODE" = "chain" ]; then
    LAYOUT_FLAGS+=(--source_model_input_layout act_raw NONTRIVIAL)
    LAYOUT_FLAGS+=(--desired_input_layout act_raw NONTRIVIAL)
    LAYOUT_FLAGS+=(--source_model_output_layout out NONTRIVIAL)
    LAYOUT_FLAGS+=(--desired_output_layout out NONTRIVIAL)
else
    for i in $(seq 0 $((CHAIN - 1))); do
        IN_NAME="act_raw"
        if [ "$i" != "0" ]; then IN_NAME="act_raw_${i}"; fi
        OUT_NAME="out_${i}"
        LAYOUT_FLAGS+=(--source_model_input_layout "$IN_NAME" NONTRIVIAL)
        LAYOUT_FLAGS+=(--desired_input_layout "$IN_NAME" NONTRIVIAL)
        LAYOUT_FLAGS+=(--source_model_output_layout "$OUT_NAME" NONTRIVIAL)
        LAYOUT_FLAGS+=(--desired_output_layout "$OUT_NAME" NONTRIVIAL)
    done
fi

echo "=== [3/4] qairt-converter -> qairt-quantizer ==="
"$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter" \
    -i "$OUT_DIR/u8i8.onnx" \
    --op_package_config QnnHmxMatMulU8I8Package.xml \
    --converter_op_package_lib "$CPL" \
    --quantization_overrides "$OUT_DIR/quant_overrides.json" \
    "${LAYOUT_FLAGS[@]}" \
    -o "$OUT_DIR/u8i8_encoded.dlc" \
    2>&1 | tee "$OUT_DIR/convert.log" | tail -5
qairt_quantize_encoded_dlc \
    "$OUT_DIR/u8i8_encoded.dlc" \
    "$OUT_DIR/u8i8.dlc" \
    8 8 32 0 \
    "$OUT_DIR/quantize.log"
tail -5 "$OUT_DIR/quantize.log"

echo "=== [4/4] qnn-context-binary-generator ==="
rm -rf "$OUT_DIR/ctx"
rm -f ./*_schematic.bin ./schematic.bin ./model_schematic.bin
"$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --dlc_path "$OUT_DIR/u8i8.dlc" \
    --op_packages "$X86_PKG:QnnHmxMatMulU8I8InterfaceProvider" \
    --binary_file u8i8_ctx \
    --output_dir "$OUT_DIR/ctx" \
    --config_file htp_config.json \
    --profiling_level detailed \
    --profiling_option optrace \
    --save_backend_op_mapping \
    2>&1 | tee "$OUT_DIR/ctxgen.log" | tail -5
for f in "$SCRIPT_DIR"/*_schematic.bin "$SCRIPT_DIR"/schematic.bin "$SCRIPT_DIR"/model_schematic.bin; do
    [ -f "$f" ] || continue
    mv "$f" "$OUT_DIR/ctx/$(basename "$f")"
done

MAPPING_JSON="$OUT_DIR/ctx/u8i8_ctx_bottom_mapping.json"
if [ ! -f "$MAPPING_JSON" ]; then
    MAPPING_JSON="$(find "$OUT_DIR/ctx" -maxdepth 1 -name '*bottom_mapping.json' -print -quit)"
fi
python3 - "$MAPPING_JSON" <<'PY' || true
import json
import sys
from collections import Counter

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print("  ctx nodes:", Counter(n["type"] for n in data["graph"]["nodes"].values()))
PY

if [ "$SKIP_DEVICE" = "1" ]; then
    echo "=== done: ctx artifacts in $OUT_DIR ==="
    exit 0
fi

echo "=== device run on $DEVICE ==="
REMOTE="qnn_run/custom_u8i8"
ssh "$DEVICE" "mkdir -p $REMOTE/runtime_inputs_u8"
ssh "$DEVICE" "cat > $REMOTE/u8i8_ctx.bin" < "$OUT_DIR/ctx/u8i8_ctx.bin"
ssh "$DEVICE" "cat > $REMOTE/htp_config.json" < htp_config.json
ssh "$DEVICE" "cat > $REMOTE/htp_backend_ext.json" < htp_backend_ext.json
ssh "$DEVICE" "cat > qnn_run/libQnnHmxMatMulU8I8_htp.so" < "$PKG_HTP"
ssh "$DEVICE" "cat > qnn_run/libQnnHmxMatMulU8I8_cpu.so" < "$PKG_CPU"

for f in "$OUT_DIR"/runtime_inputs_u8/act_u8i8*.raw; do
    [ -f "$f" ] || continue
    ssh "$DEVICE" "cat > $REMOTE/runtime_inputs_u8/$(basename "$f")" < "$f"
done

if [ -f "$OUT_DIR/runtime_input_list.txt" ]; then
    cp "$OUT_DIR/runtime_input_list.txt" "$OUT_DIR/input_list.txt"
elif [ "$MODE" = "chain" ]; then
    echo "act_raw:=runtime_inputs_u8/act_u8i8.raw" > "$OUT_DIR/input_list.txt"
else
    line="act_raw:=runtime_inputs_u8/act_u8i8.raw"
    for i in $(seq 1 $((CHAIN - 1))); do
        line="$line act_raw_${i}:=runtime_inputs_u8/act_u8i8_${i}.raw"
    done
    echo "$line" > "$OUT_DIR/input_list.txt"
fi
ssh "$DEVICE" "cat > $REMOTE/input_list.txt" < "$OUT_DIR/input_list.txt"
RUN_STATUS=0
OUTPUT_FLAGS=""
if [ "$NATIVE_OUTPUT" = "1" ]; then
    OUTPUT_FLAGS="--use_native_output_files"
fi
ssh "$DEVICE" "cd $REMOTE && rm -rf out && LD_LIBRARY_PATH=../:.:/vendor/lib64 ADSP_LIBRARY_PATH=../ ../qnn-net-run --backend ../libQnnHtp.so --retrieve_context u8i8_ctx.bin --op_packages ../libQnnHmxMatMulU8I8_cpu.so:QnnHmxMatMulU8I8InterfaceProvider:CPU,../libQnnHmxMatMulU8I8_htp.so:QnnHmxMatMulU8I8InterfaceProvider:HTP --input_list input_list.txt --profiling_level detailed --profiling_option optrace --output_dir out --config_file htp_config.json --use_native_input_files $OUTPUT_FLAGS --num_inferences 3 --perf_profile burst 2>&1" \
    > "$OUT_DIR/run.log" 2>&1 || RUN_STATUS=$?

tail -40 "$OUT_DIR/run.log"
mkdir -p "$OUT_DIR/device_out"
if [ "$NATIVE_OUTPUT" = "1" ]; then
    ssh "$DEVICE" "cat $REMOTE/out/Result_0/out_native.raw 2>/dev/null || cat $REMOTE/out/Result_0/out.raw 2>/dev/null" > "$OUT_DIR/device_out/out.raw" 2>/dev/null || true
else
    ssh "$DEVICE" "cat $REMOTE/out/Result_0/out.raw 2>/dev/null || cat $REMOTE/out/Result_0/out_native.raw 2>/dev/null" > "$OUT_DIR/device_out/out.raw" 2>/dev/null || true
fi
ssh "$DEVICE" "cat $REMOTE/out/qnn-profiling-data_0.log 2>/dev/null" > "$OUT_DIR/device_out/qnn-profiling-data_0.log" 2>/dev/null || true

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

VERIFY_STATUS=0
if [ "$VERIFY_REF" = "1" ]; then
python3 - "$OUT_DIR" "$M" <<'PY' || VERIFY_STATUS=$?
import sys
from pathlib import Path

import numpy as np

out_dir = Path(sys.argv[1])
shape = int(sys.argv[2])
raw = out_dir / "device_out" / "out.raw"
refs = list(out_dir.glob("*.out_ref_u8.npy"))
if not raw.exists() or not refs:
    print("  bit-exact: failed (missing output or reference)")
    raise SystemExit(2)
ref = np.load(refs[0])
raw_size = raw.stat().st_size
q_bytes = ref.size * ref.dtype.itemsize
f_bytes = ref.size * np.dtype(np.float32).itemsize
if raw_size == q_bytes:
    out_u8 = np.fromfile(raw, dtype=ref.dtype).reshape(ref.shape)
elif raw_size == f_bytes:
    out = np.fromfile(raw, dtype=np.float32).reshape(ref.shape)
    out_u8 = np.round(out).astype(ref.dtype)
else:
    print(f"  bit-exact: failed ({raw} has unexpected size {raw_size})")
    raise SystemExit(2)
ok = int((out_u8 == ref).sum())
print(f"  bit-exact: {ok}/{out_u8.size}")
if ok != out_u8.size:
    raise SystemExit(3)
PY
else
    echo "  bit-exact: skipped local reference check (VERIFY_REF=0)"
fi

if [ "$RUN_STATUS" != "0" ] || [ "$VERIFY_STATUS" != "0" ]; then
    echo "ERROR: device validation failed (run=$RUN_STATUS verify=$VERIFY_STATUS)" >&2
    exit 1
fi

echo "=== done: artifacts in $OUT_DIR ==="
