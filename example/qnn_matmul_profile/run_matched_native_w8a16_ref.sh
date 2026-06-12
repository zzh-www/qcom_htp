#!/usr/bin/env bash
#
# Generate and run a QNN-native W8A16 MatMul chain matched to an existing
# custom w8a16 artifact.  The native reference uses the custom artifact's exact
# runtime input, logical W matrix, and chain length.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/example/qnn_matmul_profile"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env.sh" >/dev/null
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/qairt_quant_flow.sh"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/dssh.sh"

DEVICE="${DEVICE:-oneplus}"
ARCH="${ARCH:-v75}"
CUSTOM_DIR=""
OUT_DIR=""
CHAIN="${CHAIN:-0}"
NUM_INFERENCES="${NUM_INFERENCES:-3}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --custom-dir) CUSTOM_DIR="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --chain) CHAIN="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        --help|-h)
            sed -n '3,24p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$CUSTOM_DIR" ]; then
    CUSTOM_DIR="$SCRIPT_DIR/output_w8a16_aligned_e2e_256"
fi
if [ -z "$OUT_DIR" ]; then
    OUT_DIR="$SCRIPT_DIR/output_w8a16_native_ref_e2e_256"
fi
CUSTOM_DIR="$(cd "$CUSTOM_DIR" && pwd)"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

QNN="$QNN_SDK_ROOT"
case "$ARCH" in
    v73) SOC_ID="${SOC_ID:-43}" ;;
    v75) SOC_ID="${SOC_ID:-57}" ;;
    v79) SOC_ID="${SOC_ID:-69}" ;;
    v81) SOC_ID="${SOC_ID:-87}" ;;
    *) SOC_ID="${SOC_ID:-57}" ;;
esac

export PYTHONPATH="$QNN/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$QNN/bin/x86_64-linux-clang:$PATH"

cat > "$OUT_DIR/htp_config.json" <<EOF
{
  "backend_extensions": {
    "shared_library_path": "$QNN/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so",
    "config_file_path": "$OUT_DIR/htp_backend_ext.json"
  }
}
EOF
cat > "$OUT_DIR/htp_config_device.json" <<'EOF'
{
  "backend_extensions": {
    "shared_library_path": "../libQnnHtpNetRunExtensions.so",
    "config_file_path": "htp_backend_ext.json"
  }
}
EOF
cat > "$OUT_DIR/htp_backend_ext.json" <<EOF
{
  "devices": [
    {
      "dsp_arch": "$ARCH",
      "soc_id": $SOC_ID,
      "pd_session": "unsigned",
      "cores": [
        {"core_id": 0, "perf_profile": "burst", "rpc_control_latency": 100}
      ]
    }
  ]
}
EOF

GEN_ARGS=(--custom-dir "$CUSTOM_DIR" --out-dir "$OUT_DIR")
if [ "$CHAIN" != "0" ]; then
    GEN_ARGS+=(--chain "$CHAIN")
fi
python3 "$SCRIPT_DIR/gen_matched_native_w8a16.py" "${GEN_ARGS[@]}"

echo "=== qairt-converter -> qairt-quantizer (w8a16 matched native) ==="
"$QNN/bin/x86_64-linux-clang/qairt-converter" \
    -i "$OUT_DIR/matmul.onnx" \
    --target_backend HTP \
    --enable_framework_trace \
    --quantization_overrides "$OUT_DIR/quant_overrides.json" \
    --source_model_input_layout A NONTRIVIAL \
    --desired_input_layout A NONTRIVIAL \
    --source_model_output_layout Y NONTRIVIAL \
    --desired_output_layout Y NONTRIVIAL \
    -o "$OUT_DIR/matmul_encoded.dlc" \
    > "$OUT_DIR/_convert.log" 2>&1
qairt_quantize_encoded_dlc \
    "$OUT_DIR/matmul_encoded.dlc" \
    "$OUT_DIR/matmul.dlc" \
    16 8 32 0 \
    "$OUT_DIR/_quantize.log"

echo "=== qnn-context-binary-generator ==="
rm -rf "$OUT_DIR/ctx"
( cd "$OUT_DIR" && "$QNN/bin/x86_64-linux-clang/qnn-context-binary-generator" \
    --backend "$QNN/lib/x86_64-linux-clang/libQnnHtp.so" \
    --dlc_path "$OUT_DIR/matmul.dlc" \
    --binary_file matmul_native_ctx \
    --output_dir ctx \
    --config_file "$OUT_DIR/htp_config.json" \
    --profiling_level detailed \
    --profiling_option optrace \
    --save_backend_op_mapping \
    > "$OUT_DIR/_ctxgen.log" 2>&1 )
mkdir -p "$OUT_DIR/ctx"
for f in "$OUT_DIR"/*schematic.bin "$OUT_DIR"/schematic.bin; do
    [ -f "$f" ] || continue
    mv "$f" "$OUT_DIR/ctx/$(basename "$f")"
done

echo "=== device run on $DEVICE ==="
REMOTE="qnn_run/native_w8a16_matched"
ssh "$DEVICE" "mkdir -p $REMOTE/runtime_inputs_native"
ssh "$DEVICE" "cat > $REMOTE/matmul_native_ctx.bin" < "$OUT_DIR/ctx/matmul_native_ctx.bin"
ssh "$DEVICE" "cat > $REMOTE/htp_config.json" < "$OUT_DIR/htp_config_device.json"
ssh "$DEVICE" "cat > $REMOTE/htp_backend_ext.json" < "$OUT_DIR/htp_backend_ext.json"
ssh "$DEVICE" "cat > $REMOTE/input_list.txt" < "$OUT_DIR/runtime_input_list.txt"
ssh "$DEVICE" "cat > $REMOTE/runtime_inputs_native/A.raw" < "$OUT_DIR/runtime_inputs_native/A.raw"
ssh "$DEVICE" "cd $REMOTE && rm -rf out && \
    LD_LIBRARY_PATH=../:.:/vendor/lib64 ADSP_LIBRARY_PATH=../ \
    ../qnn-net-run \
      --backend ../libQnnHtp.so \
      --retrieve_context matmul_native_ctx.bin \
      --input_list input_list.txt \
      --profiling_level detailed \
      --profiling_option optrace \
      --output_dir out \
      --config_file htp_config.json \
      --use_native_input_files \
      --use_native_output_files \
      --num_inferences $NUM_INFERENCES \
      --perf_profile burst 2>&1" \
      > "$OUT_DIR/_run.log" 2>&1

mkdir -p "$OUT_DIR/device_out"
ssh "$DEVICE" "cat $REMOTE/out/qnn-profiling-data_0.log" > "$OUT_DIR/device_out/qnn-profiling-data_0.log"
ssh "$DEVICE" "cat $REMOTE/out/Result_0/Y_native.raw 2>/dev/null || cat $REMOTE/out/Result_0/Y.raw" \
    > "$OUT_DIR/device_out/Y.raw"

python "$ROOT_DIR/scripts/decode_qnn_optrace.py" "$OUT_DIR"
python "$ROOT_DIR/scripts/check_qnn_artifact_standard.py" \
    "$OUT_DIR" --require-native-io --require-layout-flags --reject-float-io

mkdir -p "$OUT_DIR/analysis"
python3 - "$CUSTOM_DIR" "$OUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

custom_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
custom = np.fromfile(custom_dir / "device_out" / "out.raw", dtype=np.uint16)
native = np.fromfile(out_dir / "device_out" / "Y.raw", dtype=np.uint16)
if custom.size != native.size:
    raise SystemExit(f"size mismatch: custom={custom.size}, native={native.size}")
diff = np.abs(custom.astype(np.int64) - native.astype(np.int64))
summary = {
    "family": "w8a16",
    "custom_dir": str(custom_dir),
    "native_dir": str(out_dir),
    "exact": int((custom == native).sum()),
    "total": int(custom.size),
    "maxdiff": int(diff.max()) if diff.size else 0,
    "mean_absdiff": float(diff.mean()) if diff.size else 0.0,
}
(out_dir / "analysis" / "matched_native_compare.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
(out_dir / "analysis" / "matched_native_compare.txt").write_text(
    (
        "w8a16 matched native compare\n"
        f"custom: {custom_dir}\n"
        f"native: {out_dir}\n"
        f"exact: {summary['exact']}/{summary['total']}\n"
        f"maxdiff: {summary['maxdiff']}\n"
        f"mean_absdiff: {summary['mean_absdiff']:.6f}\n"
    ),
    encoding="utf-8",
)
print(f"matched-native exact: {summary['exact']}/{summary['total']} maxdiff={summary['maxdiff']}")
PY

echo "=== done: $OUT_DIR ==="
