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
W4_ENCODING="${W4_ENCODING:-symmetric}"
LPBQ_BUILD_FLAG=0
FAMILY_LABEL="w4a8_per_channel"
CASE_FAMILY="w4a8_per_channel"
if [ "$W4_ENCODING" = "lpbq" ]; then
    LPBQ_BUILD_FLAG=1
    FAMILY_LABEL="w4a8_lpbq"
    CASE_FAMILY="w4a8_lpbq"
fi

if [ "$CHAIN" != "1" ]; then
    echo "ERROR: w4a8 per-channel native_match currently supports CHAIN=1 only" >&2
    exit 2
fi

CASE_ROOT="$OUT_ROOT/cases"
ANALYSIS_ROOT="$OUT_ROOT/analysis"
mkdir -p "$OUT_ROOT" "$ANALYSIS_ROOT"

if [ "$BUILD_PACKAGES" = "1" ]; then
    echo "=== build $FAMILY_LABEL package ==="
    LPBQ_ONLY="$LPBQ_BUILD_FLAG" bash "$ROOT_DIR/example/qnn_hmx_matmul_w4a8/build.sh"
    LPBQ_ONLY="$LPBQ_BUILD_FLAG" bash "$ROOT_DIR/example/qnn_hmx_matmul_w4a8/build_x86.sh"
fi

echo "=== generate $FAMILY_LABEL Python/QNN cases: ${M}x${K}x${N} ==="
uv run python "$ROOT_DIR/scripts/generate_qnn_matmul_python_cases.py" \
    --families "$CASE_FAMILY" \
    --cases $CASES \
    --m "$M" --k "$K" --n "$N" \
    --out-root "$CASE_ROOT"

for case_name in $CASES; do
    CASE_DIR="$CASE_ROOT/$CASE_FAMILY/$case_name"
    NATIVE_DIR="$OUT_ROOT/native_$case_name"
    CUSTOM_DIR="$OUT_ROOT/custom_$case_name"
    BIAS_RECORD="$ANALYSIS_ROOT/native_bias_record_${case_name}.raw"

    echo "=== run QNN Native w4a8_per_channel case=$case_name on $DEVICE ==="
    uv run python "$ROOT_DIR/scripts/run_qnn_python_case_native.py" \
        --case-dir "$CASE_DIR" \
        --out-dir "$NATIVE_DIR" \
        --device "$DEVICE" \
        --w4-encoding "$W4_ENCODING"

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
    M="$M" K="$K" N="$N" CHAIN=1 MODE=chain OP_INPUT_LAYOUT=native W4_ENCODING="$W4_ENCODING" \
    OUT_DIR="$CUSTOM_DIR" DEVICE="$DEVICE" \
    GEN_EXTRA_ARGS="--case-dir $CASE_DIR" \
        bash "$ROOT_DIR/example/qnn_hmx_matmul_w4a8/standard_flow/custom_w4a8/run_w4a8_chain.sh"

    if [ "$W4_ENCODING" = "lpbq" ]; then
        echo "=== verify native/custom LPBQ metadata case=$case_name ==="
        uv run python - "$NATIVE_DIR/quant_overrides.json" "$CUSTOM_DIR/quant_overrides.json" "$CUSTOM_DIR/native_io.json" <<'PY'
import json
import sys
from pathlib import Path

def check_native_lpbq(path):
    overrides = json.loads(Path(path).read_text(encoding="utf-8"))
    if overrides.get("version") != "1.0.0":
        raise SystemExit(f"{path}: LPBQ override version mismatch: {overrides.get('version')!r}")
    params = overrides.get("param_encodings", [])
    weight = next((enc for enc in params if isinstance(enc, dict) and enc.get("name") in {"W", "weight"}), None)
    if weight is None:
        raise SystemExit(f"{path}: missing LPBQ weight encoding")
    expected = {
        "enc_type": "LPBQ",
        "dtype": "INT",
        "bw": 8,
        "compressed_bw": 4,
        "block_size": 32,
        "is_sym": True,
    }
    for key, value in expected.items():
        if weight.get(key) != value:
            raise SystemExit(f"{path}: LPBQ weight {key}={weight.get(key)!r}, expected {value!r}")
    scales = weight.get("scale")
    per_block = weight.get("per_block_int_scale")
    if not isinstance(scales, list) or not scales:
        raise SystemExit(f"{path}: missing LPBQ scale list")
    if not isinstance(per_block, list) or len(per_block) != len(scales):
        raise SystemExit(f"{path}: LPBQ per_block_int_scale channel count mismatch")
    flat = [int(v) for row in per_block for v in (row if isinstance(row, list) else [row])]
    if len(set(flat)) <= 1:
        raise SystemExit(f"{path}: LPBQ per_block_int_scale is degenerate: {sorted(set(flat))}")
    print(f"{path}: LPBQ metadata ok channels={len(scales)} block_size={weight['block_size']} block_scales={sorted(set(flat))}")

def check_custom_two_op(override_path, native_io_path):
    overrides = json.loads(Path(override_path).read_text(encoding="utf-8"))
    param_encodings = overrides.get("param_encodings", {})
    if isinstance(param_encodings, dict):
        params = set(param_encodings)
    else:
        params = {enc.get("name") for enc in param_encodings if isinstance(enc, dict)}
    missing = {"weight_lpbq", "weight_lpbq_per_block_int_scale"} - params
    if missing:
        raise SystemExit(f"{override_path}: missing custom LPBQ params {sorted(missing)}")
    info = json.loads(Path(native_io_path).read_text(encoding="utf-8"))
    if not info.get("lpbq_expand_custom_op"):
        raise SystemExit(f"{native_io_path}: LPBQ expand custom op flag is not set")
    if info.get("lpbq_matmul_op") != "HmxU8I8ToU8MatMul":
        raise SystemExit(f"{native_io_path}: unexpected LPBQ matmul op {info.get('lpbq_matmul_op')!r}")
    print(f"{native_io_path}: custom LPBQ two-op metadata ok op={info['lpbq_matmul_op']}")

check_native_lpbq(sys.argv[1])
check_custom_two_op(sys.argv[2], sys.argv[3])
PY
    fi

    echo "=== verify custom sidecar equals QNN Native final sidecar case=$case_name ==="
    uv run python "$ROOT_DIR/scripts/analyze_u8i8_native_bias_record.py" \
        --case-dir "$CASE_DIR" \
        --native-context-bin "$NATIVE_DIR/ctx/case_native_ctx.bin" \
        --custom-onnx "$CUSTOM_DIR/w4a8.onnx" \
        | tee "$ANALYSIS_ROOT/custom_bias_record_${case_name}.txt"

    if [ "$W4_ENCODING" != "lpbq" ]; then
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
    else
        echo "  LPBQ sidecar byte exactness is diagnostic; final custom/native output exactness is the gate."
    fi

    echo "=== compare custom/native exactness case=$case_name ==="
    uv run python - "$OUT_ROOT" "$case_name" "$NATIVE_DIR" "$CUSTOM_DIR" "$M" "$N" "$FAMILY_LABEL" <<'PY'
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
family_label = sys.argv[7]

native = np.load(native_dir / "analysis" / "native_output_mn.npy").astype(np.uint8).reshape(m, n)
custom = np.fromfile(custom_dir / "device_out" / "out.raw", dtype=np.uint8).reshape(m, n)
diff = custom.astype(np.int16) - native.astype(np.int16)
exact = int((diff == 0).sum())
total = int(diff.size)
summary = {
    "comparison_scope": "same_hardware_custom_hmx_vs_qnn_native_htp",
    "family": family_label,
    "ci_family": family_label,
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

uv run python - "$OUT_ROOT" "$FAMILY_LABEL" $CASES <<'PY'
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
family_label = sys.argv[2]
cases = sys.argv[3:]
items = []
for case_name in cases:
    path = out_root / "analysis" / f"custom_native_compare_{case_name}.json"
    items.append(json.loads(path.read_text(encoding="utf-8")))
summary = {
    "comparison_scope": "same_hardware_custom_hmx_vs_qnn_native_htp",
    "family": family_label,
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

echo "PASS: $FAMILY_LABEL Python-case custom/native exactness ($CASES)"
