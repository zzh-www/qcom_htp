#!/usr/bin/env bash
#
# Generate float-derived w4a8 per-channel Python/QNN cases, run QNN Native, then
# run the custom HMX op with the recovered HTP-prepare A8 sidecar rule.  The
# final gate is same-hardware custom/native exactness for every selected case.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DEVICE="${DEVICE:-oneplus}"
OUT_ROOT="${OUT_ROOT:-/tmp/qcom_htp_w4a8_python_case_custom_native_match}"
CASES="${W4A8_MATCH_CASES:-normal_random zp_neutral positive_boundary negative_boundary single_k_impulse bias_only scale_only}"
M="${M:-256}"
K="${K:-256}"
N="${N:-256}"
CHAIN="${CHAIN:-1}"
BUILD_PACKAGES="${BUILD_PACKAGES:-1}"

if [ "$CHAIN" != "1" ]; then
    echo "ERROR: w4a8 per-channel native_match currently supports CHAIN=1 only" >&2
    exit 2
fi

CASE_ROOT="$OUT_ROOT/cases"
ANALYSIS_ROOT="$OUT_ROOT/analysis"
mkdir -p "$OUT_ROOT" "$ANALYSIS_ROOT"

if [ "$BUILD_PACKAGES" = "1" ]; then
    echo "=== build non-LPBQ w4a8 per-channel package ==="
    bash "$ROOT_DIR/example/qnn_hmx_matmul_w4a8/build.sh"
    bash "$ROOT_DIR/example/qnn_hmx_matmul_w4a8/build_x86.sh"
fi

echo "=== generate w4a8 per-channel Python/QNN cases: ${M}x${K}x${N} ==="
uv run python "$ROOT_DIR/scripts/generate_qnn_matmul_python_cases.py" \
    --families w4a8_per_channel \
    --cases $CASES \
    --m "$M" --k "$K" --n "$N" \
    --out-root "$CASE_ROOT"

for case_name in $CASES; do
    CASE_DIR="$CASE_ROOT/w4a8_per_channel/$case_name"
    NATIVE_DIR="$OUT_ROOT/native_$case_name"
    CUSTOM_DIR="$OUT_ROOT/custom_$case_name"
    BIAS_RECORD="$ANALYSIS_ROOT/native_bias_record_${case_name}.raw"

    echo "=== run QNN Native w4a8_per_channel case=$case_name on $DEVICE ==="
    uv run python "$ROOT_DIR/scripts/run_qnn_python_case_native.py" \
        --case-dir "$CASE_DIR" \
        --out-dir "$NATIVE_DIR" \
        --device "$DEVICE"

    echo "=== sync custom bias_q to QNN Native quantized DLC B case=$case_name ==="
    uv run python - "$CASE_DIR" "$NATIVE_DIR/analysis/dlc_bias_q_int32.npy" <<'PY'
import sys
from pathlib import Path

import numpy as np

case_dir = Path(sys.argv[1])
dlc_bias_path = Path(sys.argv[2])
if not dlc_bias_path.exists():
    raise SystemExit(f"missing native DLC bias export: {dlc_bias_path}")
dst_npy = case_dir / "bias_q_int32.npy"
dst_raw = case_dir / "bias_q_int32.raw"
old = np.load(dst_npy).astype(np.int32)
new = np.load(dlc_bias_path).astype(np.int32)
if old.shape != new.shape:
    raise SystemExit(f"DLC bias shape mismatch: {new.shape} vs {old.shape}")
delta = new - old
print(
    "DLC B vs generated bias_q:",
    {str(int(v)): int((delta == v).sum()) for v in np.unique(delta)},
)
np.save(dst_npy, new)
new.tofile(dst_raw)
PY

    echo "=== extract native w4a8 bias/control sidecar case=$case_name ==="
    uv run python "$ROOT_DIR/scripts/analyze_u8i8_native_bias_record.py" \
        --case-dir "$CASE_DIR" \
        --native-context-bin "$NATIVE_DIR/ctx/case_native_ctx.bin" \
        --out-raw "$BIAS_RECORD" \
        | tee "$ANALYSIS_ROOT/native_bias_record_${case_name}.txt"

    echo "=== run custom w4a8 with generated HTP-prepare sidecar case=$case_name on $DEVICE ==="
    BUILD_PACKAGES="$BUILD_PACKAGES" \
    VERIFY_REF=0 \
    M="$M" K="$K" N="$N" CHAIN=1 MODE=chain OP_INPUT_LAYOUT=native \
    OUT_DIR="$CUSTOM_DIR" DEVICE="$DEVICE" \
    GEN_EXTRA_ARGS="--case-dir $CASE_DIR" \
        bash "$ROOT_DIR/example/qnn_hmx_matmul_w4a8/standard_flow/custom_w4a8/run_w4a8_chain.sh"

    echo "=== verify custom sidecar equals QNN Native final sidecar case=$case_name ==="
    uv run python "$ROOT_DIR/scripts/analyze_u8i8_native_bias_record.py" \
        --case-dir "$CASE_DIR" \
        --native-context-bin "$NATIVE_DIR/ctx/case_native_ctx.bin" \
        --custom-onnx "$CUSTOM_DIR/w4a8.onnx" \
        | tee "$ANALYSIS_ROOT/custom_bias_record_${case_name}.txt"

    uv run python - "$ANALYSIS_ROOT/custom_bias_record_${case_name}.txt" "$N" <<'PY'
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

    echo "=== compare custom/native exactness case=$case_name ==="
    uv run python - "$OUT_ROOT" "$case_name" "$NATIVE_DIR" "$CUSTOM_DIR" "$M" "$N" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

out_root = Path(sys.argv[1])
case_name = sys.argv[2]
native_dir = Path(sys.argv[3])
custom_dir = Path(sys.argv[4])
m = int(sys.argv[5])
n = int(sys.argv[6])

native = np.load(native_dir / "analysis" / "native_output_mn.npy").astype(np.uint8).reshape(m, n)
custom = np.fromfile(custom_dir / "device_out" / "out.raw", dtype=np.uint8).reshape(m, n)
diff = custom.astype(np.int16) - native.astype(np.int16)
exact = int((diff == 0).sum())
total = int(diff.size)
summary = {
    "comparison_scope": "same_hardware_custom_hmx_vs_qnn_native_htp",
    "family": "w4a8_per_channel",
    "case": case_name,
    "exact": exact,
    "total": total,
    "maxabs": int(np.abs(diff).max()) if total else 0,
    "diff_counts": {str(int(v)): int((diff == v).sum()) for v in np.unique(diff)},
    "native_dir": str(native_dir),
    "custom_dir": str(custom_dir),
}
analysis_dir = out_root / "analysis"
analysis_dir.mkdir(parents=True, exist_ok=True)
(analysis_dir / f"custom_native_compare_{case_name}.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, indent=2))
if exact != total:
    raise SystemExit(1)
PY
done

uv run python - "$OUT_ROOT" $CASES <<'PY'
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
cases = sys.argv[2:]
items = []
for case_name in cases:
    path = out_root / "analysis" / f"custom_native_compare_{case_name}.json"
    items.append(json.loads(path.read_text(encoding="utf-8")))
summary = {
    "comparison_scope": "same_hardware_custom_hmx_vs_qnn_native_htp",
    "family": "w4a8_per_channel",
    "cases": cases,
    "passed": all(item["exact"] == item["total"] for item in items),
    "results": items,
}
(out_root / "analysis" / "custom_native_compare_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, indent=2))
if not summary["passed"]:
    raise SystemExit(1)
PY

echo "PASS: w4a8 per-channel Python-case custom/native exactness ($CASES)"
