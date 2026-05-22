#!/usr/bin/env bash
#
# Generate a float-derived u8i8 Python/QNN case, run QNN Native, then run the
# custom HMX op with the recovered HTP-prepare bias sidecar rule.  Extracted
# native sidecar injection is retained as a diagnostic override only.  The final
# gate is same-hardware custom/native exactness.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DEVICE="${DEVICE:-oneplus}"
OUT_ROOT="${OUT_ROOT:-/tmp/qcom_htp_u8i8_python_case_custom_native_match}"
CASE_NAME="${CASE_NAME:-normal_random}"
M="${M:-256}"
K="${K:-256}"
N="${N:-256}"
CHAIN="${CHAIN:-1}"
BUILD_PACKAGES="${BUILD_PACKAGES:-0}"
USE_NATIVE_BIAS_RECORD="${USE_NATIVE_BIAS_RECORD:-0}"

CASE_ROOT="$OUT_ROOT/cases"
CASE_DIR="$CASE_ROOT/u8i8/$CASE_NAME"
NATIVE_DIR="$OUT_ROOT/native_$CASE_NAME"
CUSTOM_DIR="$OUT_ROOT/custom_$CASE_NAME"
ANALYSIS_DIR="$OUT_ROOT/analysis"
BIAS_RECORD="$ANALYSIS_DIR/native_bias_record_${CASE_NAME}.raw"

mkdir -p "$OUT_ROOT" "$ANALYSIS_DIR"

echo "=== generate u8i8 Python/QNN case: $CASE_NAME ${M}x${K}x${N} ==="
uv run python "$ROOT_DIR/scripts/generate_qnn_matmul_python_cases.py" \
    --families u8i8 \
    --cases "$CASE_NAME" \
    --m "$M" --k "$K" --n "$N" \
    --out-root "$CASE_ROOT"

echo "=== run QNN Native case on $DEVICE ==="
uv run python "$ROOT_DIR/scripts/run_qnn_python_case_native.py" \
    --case-dir "$CASE_DIR" \
    --out-dir "$NATIVE_DIR" \
    --device "$DEVICE"

echo "=== extract native u8i8 bias/control sidecar ==="
uv run python "$ROOT_DIR/scripts/analyze_u8i8_native_bias_record.py" \
    --case-dir "$CASE_DIR" \
    --native-context-bin "$NATIVE_DIR/ctx/case_native_ctx.bin" \
    --out-raw "$BIAS_RECORD" \
    | tee "$ANALYSIS_DIR/native_bias_record_${CASE_NAME}.txt"

GEN_EXTRA_ARGS_VALUE="--case-dir $CASE_DIR"
if [ "$USE_NATIVE_BIAS_RECORD" = "1" ]; then
    echo "=== run custom u8i8 with extracted native sidecar on $DEVICE ==="
    GEN_EXTRA_ARGS_VALUE="$GEN_EXTRA_ARGS_VALUE --bias-record-raw $BIAS_RECORD"
else
    echo "=== run custom u8i8 with generated HTP-prepare bias sidecar on $DEVICE ==="
fi
BUILD_PACKAGES="$BUILD_PACKAGES" \
VERIFY_REF=0 \
M="$M" K="$K" N="$N" CHAIN="$CHAIN" MODE=chain \
OUT_DIR="$CUSTOM_DIR" DEVICE="$DEVICE" \
GEN_EXTRA_ARGS="$GEN_EXTRA_ARGS_VALUE" \
    bash "$ROOT_DIR/example/qnn_hmx_matmul_u8i8/standard_flow/custom_u8i8/run_u8i8_chain.sh"

echo "=== verify custom sidecar equals QNN Native final sidecar ==="
uv run python "$ROOT_DIR/scripts/analyze_u8i8_native_bias_record.py" \
    --case-dir "$CASE_DIR" \
    --native-context-bin "$NATIVE_DIR/ctx/case_native_ctx.bin" \
    --custom-onnx "$CUSTOM_DIR/u8i8.onnx" \
    | tee "$ANALYSIS_DIR/custom_bias_record_${CASE_NAME}.txt"

uv run python - "$ANALYSIS_DIR/custom_bias_record_${CASE_NAME}.txt" "$N" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
n = int(sys.argv[2])
total = (n // 32) * 256
half = total // 2
checks = {
    "native vs custom bytes": (total, total),
    "native vs custom scale/control bytes": (half, half),
    "native vs custom effective bytes": (half, half),
}
for label, expected in checks.items():
    match = re.search(rf"{re.escape(label)}: (\d+)/(\d+)", text)
    if not match:
        raise SystemExit(f"missing check: {label}")
    got = (int(match.group(1)), int(match.group(2)))
    print(f"{label}: {got[0]}/{got[1]}")
    if got != expected:
        raise SystemExit(f"{label} mismatch: got {got}, expected {expected}")
PY

echo "=== compare custom/native exactness ==="
uv run python - "$OUT_ROOT" "$NATIVE_DIR" "$CUSTOM_DIR" "$M" "$N" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

out_root = Path(sys.argv[1])
native_dir = Path(sys.argv[2])
custom_dir = Path(sys.argv[3])
m = int(sys.argv[4])
n = int(sys.argv[5])

native = np.load(native_dir / "analysis" / "native_output_mn.npy").astype(np.uint8).reshape(m, n)
custom = np.fromfile(custom_dir / "device_out" / "out.raw", dtype=np.uint8).reshape(m, n)
diff = custom.astype(np.int16) - native.astype(np.int16)
exact = int((diff == 0).sum())
total = int(diff.size)
summary = {
    "comparison_scope": "same_hardware_custom_hmx_vs_qnn_native_htp",
    "exact": exact,
    "total": total,
    "maxabs": int(np.abs(diff).max()) if total else 0,
    "diff_counts": {str(int(v)): int((diff == v).sum()) for v in np.unique(diff)},
    "native_dir": str(native_dir),
    "custom_dir": str(custom_dir),
}
analysis_dir = out_root / "analysis"
analysis_dir.mkdir(parents=True, exist_ok=True)
(analysis_dir / "custom_native_compare.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
if exact != total:
    raise SystemExit(1)
PY

echo "PASS: u8i8 Python-case custom/native exactness ($CASE_NAME)"
