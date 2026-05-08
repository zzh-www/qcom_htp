#!/usr/bin/env bash
#
# Export a QNN-native W4A16 Conv DLC through the explicit QAIRT quantizer flow:
#   ONNX -> float DLC with preserved public layout -> quantized W4A16 DLC.
#
# The default W4 policy is packed DLC storage, so W is reported as sFxp_4 by
# qairt-dlc-info. Set PACK_4BIT_WEIGHTS=0 only for deliberate carrier probes.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env.sh" >/dev/null
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/qairt_quant_flow.sh"

SHAPE="${SHAPE:-256,256,256}"
OUT_DIR="${OUT_DIR:-}"
PACK_4BIT_WEIGHTS="${PACK_4BIT_WEIGHTS:-1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --shape) SHAPE="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --pack-4bit) PACK_4BIT_WEIGHTS=1; shift ;;
        --no-pack-4bit) PACK_4BIT_WEIGHTS=0; shift ;;
        --help|-h)
            sed -n '3,34p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

case "$PACK_4BIT_WEIGHTS" in
    0|1) ;;
    *) echo "invalid PACK_4BIT_WEIGHTS=$PACK_4BIT_WEIGHTS (need 0|1)" >&2; exit 2 ;;
esac

IFS=',' read -r SHAPE_M SHAPE_K SHAPE_N <<<"$SHAPE"
if [ -z "${SHAPE_M:-}" ] || [ -z "${SHAPE_K:-}" ] || [ -z "${SHAPE_N:-}" ]; then
    echo "bad --shape '$SHAPE' (expected M,K,N)" >&2
    exit 2
fi

SCRIPT_DIR="$ROOT_DIR/example/qnn_matmul_profile"
if [ -z "$OUT_DIR" ]; then
    OUT_DIR="$SCRIPT_DIR/output_native_w4a16_qairt_quantizer_${SHAPE_M}"
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

QNN="$QNN_SDK_ROOT"
LIBCXX_DIR="$ROOT_DIR/.libcxx_shim/usr/lib/x86_64-linux-gnu"
if [ -f "$LIBCXX_DIR/libc++.so.1" ]; then
    export LD_LIBRARY_PATH="$LIBCXX_DIR:$QNN/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="$QNN/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="$QNN/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$QNN/bin/x86_64-linux-clang:$PATH"

echo "=== generate ONNX + calibration input (${SHAPE_M}x${SHAPE_K}x${SHAPE_N}) ==="
python "$SCRIPT_DIR/gen_native_w4a16_conv.py" "$OUT_DIR" \
    --m "$SHAPE_M" --k "$SHAPE_K" --n "$SHAPE_N"

qairt_make_abs_input_list "$OUT_DIR/input_list.txt" "$OUT_DIR/input_list_abs.txt"

echo "=== qairt-converter: ONNX -> float DLC ==="
qairt-converter \
    -i "$OUT_DIR/conv.onnx" \
    --target_backend HTP \
    --enable_framework_trace \
    --source_model_input_layout A NONTRIVIAL \
    --desired_input_layout A NONTRIVIAL \
    --source_model_output_layout Y NONTRIVIAL \
    --desired_output_layout Y NONTRIVIAL \
    -o "$OUT_DIR/conv_float.dlc" \
    > "$OUT_DIR/convert_float.log" 2>&1

echo "=== qairt-quantizer: float DLC -> W4A16 DLC ==="
qairt_quantize_dlc \
    "$OUT_DIR/conv_float.dlc" \
    "$OUT_DIR/conv_w4a16_quantized.dlc" \
    "$OUT_DIR/input_list_abs.txt" \
    16 4 32 "$PACK_4BIT_WEIGHTS" \
    "$OUT_DIR/quantize.log"

qairt-dlc-info -i "$OUT_DIR/conv_w4a16_quantized.dlc" > "$OUT_DIR/dlc_info.txt" 2>&1

echo "=== done ==="
echo "DLC: $OUT_DIR/conv_w4a16_quantized.dlc"
echo "Info: $OUT_DIR/dlc_info.txt"
grep -E "Quantizer command|A \\(data type|W \\(data type|Y \\(data type|W encoding|pack_4_bit_weights" \
    "$OUT_DIR/dlc_info.txt" || true
