#!/usr/bin/env bash
#
# Generate, convert, ctxgen, and optionally run the custom w16a16 HMX MatMul
# chain.  Set SKIP_DEVICE=1 to stop after context-binary generation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
EXAMPLE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env.sh" >/dev/null
# shellcheck disable=SC1091
source "$ROOT_DIR/.venv/bin/activate"

export PYTHONPATH="$QNN_SDK_ROOT/lib/python"
export PATH="$ANDROID_NDK_ROOT:$PATH"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"

DEVICE="${DEVICE:-oneplus}"
M="${M:-256}"
K="${K:-256}"
N="${N:-256}"
CHAIN="${CHAIN:-8}"
MODE="${MODE:-chain}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/out/w16a16_${MODE}_${M}}"
SKIP_DEVICE="${SKIP_DEVICE:-0}"

mkdir -p "$OUT_DIR"
cd "$SCRIPT_DIR"

X86_PKG="$EXAMPLE_DIR/build/x86_64-linux-clang/libQnnHmxMatMulW16A16.so"
PKG_HTP="$EXAMPLE_DIR/build/hexagon-v75/libQnnHmxMatMulW16A16_htp.so"
PKG_CPU="$EXAMPLE_DIR/build/aarch64/libQnnHmxMatMulW16A16_cpu.so"
CPL="$SCRIPT_DIR/converter/build/libConverterOpPackage.so"

for f in "$X86_PKG" "$PKG_HTP" "$PKG_CPU"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f; run $EXAMPLE_DIR/build.sh and build_x86.sh first" >&2
        exit 1
    fi
done

echo "=== [1/4] generate ONNX: HmxU16I16ToU16MatMul x $CHAIN ($MODE, ${M}x${K}x${N}) ==="
python gen_w16a16_chain.py \
    --chain "$CHAIN" --mode "$MODE" \
    --M "$M" --K "$K" --N "$N" \
    --bias-scale "${BIAS_SCALE:-512.0}" \
    ${GEN_EXTRA_ARGS:-} \
    -o "$OUT_DIR/w16a16.onnx"

echo "=== [2/4] build converter op package ==="
mkdir -p "$SCRIPT_DIR/converter/build"
"${CXX:-clang++}" -std=c++17 -O2 -shared -fPIC \
    -I "$QNN_SDK_ROOT/include/QNN" \
    -o "$CPL" \
    "$SCRIPT_DIR/converter/ConverterOpPackage.cpp"
echo "  -> $CPL"

LAYOUT_FLAGS=()
if [ "$MODE" = "chain" ] || [ "$MODE" = "direct" ] || [ "$MODE" = "direct_flat" ]; then
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

echo "=== [3/4] qairt-converter ==="
"$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter" \
    -i "$OUT_DIR/w16a16.onnx" \
    --op_package_config QnnHmxMatMulW16A16Package.xml \
    --converter_op_package_lib "$CPL" \
    --quantization_overrides "$OUT_DIR/quant_overrides.json" \
    "${LAYOUT_FLAGS[@]}" \
    -o "$OUT_DIR/w16a16.dlc" \
    2>&1 | tee "$OUT_DIR/convert.log" | tail -5

echo "=== [4/4] qnn-context-binary-generator ==="
rm -rf "$OUT_DIR/ctx"
"$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator" \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --dlc_path "$OUT_DIR/w16a16.dlc" \
    --op_packages "$X86_PKG:QnnHmxMatMulW16A16InterfaceProvider" \
    --binary_file w16a16_ctx \
    --output_dir "$OUT_DIR/ctx" \
    --config_file htp_config.json \
    --profiling_level detailed \
    --profiling_option optrace \
    --save_backend_op_mapping \
    2>&1 | tee "$OUT_DIR/ctxgen.log" | tail -5

if [ -f "$SCRIPT_DIR/w16a16_schematic.bin" ]; then
    mv "$SCRIPT_DIR/w16a16_schematic.bin" "$OUT_DIR/ctx/w16a16_schematic.bin"
fi

MAPPING_JSON="$OUT_DIR/ctx/w16a16_ctx_bottom_mapping.json"
if [ ! -f "$MAPPING_JSON" ]; then
    MAPPING_JSON="$OUT_DIR/ctx/w16a16_bottom_mapping.json"
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
REMOTE="qnn_run/custom_w16a16"
ssh "$DEVICE" "mkdir -p $REMOTE/runtime_inputs_u8"
ssh "$DEVICE" "cat > $REMOTE/w16a16_ctx.bin" < "$OUT_DIR/ctx/w16a16_ctx.bin"
ssh "$DEVICE" "cat > $REMOTE/htp_config.json" < htp_config.json
ssh "$DEVICE" "cat > $REMOTE/htp_backend_ext.json" < htp_backend_ext.json
ssh "$DEVICE" "cat > qnn_run/libQnnHmxMatMulW16A16_htp.so" < "$PKG_HTP"
ssh "$DEVICE" "cat > qnn_run/libQnnHmxMatMulW16A16_cpu.so" < "$PKG_CPU"

for f in "$OUT_DIR"/runtime_inputs_u8/act_w16a16*.raw; do
    [ -f "$f" ] || continue
    ssh "$DEVICE" "cat > $REMOTE/runtime_inputs_u8/$(basename "$f")" < "$f"
done

if [ "$MODE" = "chain" ] || [ "$MODE" = "direct" ] || [ "$MODE" = "direct_flat" ]; then
    echo "act_raw:=runtime_inputs_u8/act_w16a16.raw" > "$OUT_DIR/input_list.txt"
else
    line="act_raw:=runtime_inputs_u8/act_w16a16.raw"
    for i in $(seq 1 $((CHAIN - 1))); do
        line="$line act_raw_${i}:=runtime_inputs_u8/act_w16a16_${i}.raw"
    done
    echo "$line" > "$OUT_DIR/input_list.txt"
fi
ssh "$DEVICE" "cat > $REMOTE/input_list.txt" < "$OUT_DIR/input_list.txt"
RUN_STATUS=0
ssh "$DEVICE" "cd $REMOTE && rm -rf out && LD_LIBRARY_PATH=../:.:/vendor/lib64 ADSP_LIBRARY_PATH=../ ../qnn-net-run --backend ../libQnnHtp.so --retrieve_context w16a16_ctx.bin --op_packages ../libQnnHmxMatMulW16A16_cpu.so:QnnHmxMatMulW16A16InterfaceProvider:CPU,../libQnnHmxMatMulW16A16_htp.so:QnnHmxMatMulW16A16InterfaceProvider:HTP --input_list input_list.txt --profiling_level detailed --profiling_option optrace --output_dir out --config_file htp_config.json --use_native_input_files --num_inferences 3 --perf_profile burst 2>&1" \
    > "$OUT_DIR/run.log" 2>&1 || RUN_STATUS=$?

tail -40 "$OUT_DIR/run.log"
mkdir -p "$OUT_DIR/device_out"
ssh "$DEVICE" "cat $REMOTE/out/Result_0/out.raw 2>/dev/null" > "$OUT_DIR/device_out/out.raw" 2>/dev/null || true
ssh "$DEVICE" "cat $REMOTE/out/qnn-profiling-data_0.log 2>/dev/null" > "$OUT_DIR/device_out/qnn-profiling-data_0.log" 2>/dev/null || true

if [ "${DECODE_OPTRACE:-1}" = "1" ]; then
    echo "=== decode optrace artifacts ==="
    python "$ROOT_DIR/scripts/decode_qnn_optrace.py" "$OUT_DIR" || {
        [ "${STRICT_OPTRACE:-0}" = "1" ] && exit 1
        echo "  [warn] optrace decode failed; raw log kept in $OUT_DIR/device_out" >&2
    }
fi

VERIFY_STATUS=0
python3 - "$OUT_DIR" "$M" <<'PY' || VERIFY_STATUS=$?
import sys
from pathlib import Path

import numpy as np

out_dir = Path(sys.argv[1])
shape = int(sys.argv[2])
raw = out_dir / "device_out" / "out.raw"
refs = list(out_dir.glob("*.out_ref_u*.npy"))
if not raw.exists() or not refs:
    print("  bit-exact: failed (missing output or reference)")
    raise SystemExit(2)
out = np.fromfile(raw, dtype=np.float32).reshape(shape, shape)
ref = np.load(refs[0])
max_value = np.iinfo(ref.dtype).max
out_q = np.clip(np.round(out), 0, max_value).astype(ref.dtype)
ok = int((out_q == ref).sum())
print(f"  bit-exact: {ok}/{out_q.size}")
if ok != out_q.size:
    raise SystemExit(3)
PY

if [ "$RUN_STATUS" != "0" ] || [ "$VERIFY_STATUS" != "0" ]; then
    echo "ERROR: device validation failed (run=$RUN_STATUS verify=$VERIFY_STATUS)" >&2
    exit 1
fi

echo "=== done: artifacts in $OUT_DIR ==="
