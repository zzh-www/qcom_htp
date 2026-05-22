#!/usr/bin/env bash
#
# Generate float-derived w4a16 per-channel Python/QNN cases, run QNN Native,
# then run the custom HMX op with the recovered generated A16 sidecar rule.
# The final gate is same-hardware custom/native exactness for each case.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

DEVICE="${DEVICE:-oneplus}"
OUT_ROOT="${OUT_ROOT:-/tmp/qcom_htp_w4a16_python_case_custom_native_match}"
CASES="${W4A16_MATCH_CASES:-normal_random zp_neutral positive_boundary negative_boundary single_k_impulse bias_only scale_only}"
M="${M:-256}"
K="${K:-256}"
N="${N:-256}"
CHAIN="${CHAIN:-1}"
BUILD_PACKAGES="${BUILD_PACKAGES:-1}"
SIDECAR_SOURCE="${W4A16_SIDECAR_SOURCE:-generated}"

if [ "$CHAIN" != "1" ]; then
    echo "ERROR: w4a16 per-channel native_match currently supports CHAIN=1 only" >&2
    exit 2
fi

CASE_ROOT="$OUT_ROOT/cases"
ANALYSIS_ROOT="$OUT_ROOT/analysis"
mkdir -p "$OUT_ROOT" "$ANALYSIS_ROOT"

if [ "$BUILD_PACKAGES" = "1" ]; then
    echo "=== build w4a16 package ==="
    EXTRA_DEFS="-UHMX_W4A16_SKIP_KERNEL -DHMX_W4A16_ALLOW_UNVALIDATED_KERNEL" \
        bash "$ROOT_DIR/example/qnn_hmx_matmul_w4a16/build.sh"
    EXTRA_DEFS="-UHMX_W4A16_SKIP_KERNEL -DHMX_W4A16_ALLOW_UNVALIDATED_KERNEL" \
        bash "$ROOT_DIR/example/qnn_hmx_matmul_w4a16/build_x86.sh"
fi

echo "=== generate w4a16 per-channel Python/QNN cases: ${M}x${K}x${N} ==="
uv run python "$ROOT_DIR/scripts/generate_qnn_matmul_python_cases.py" \
    --families w4a16_per_channel \
    --cases $CASES \
    --m "$M" --k "$K" --n "$N" \
    --out-root "$CASE_ROOT"

for case_name in $CASES; do
    CASE_DIR="$CASE_ROOT/w4a16_per_channel/$case_name"
    NATIVE_DIR="$OUT_ROOT/native_$case_name"
    CUSTOM_DIR="$OUT_ROOT/custom_$case_name"
    BIAS_RECORD="$ANALYSIS_ROOT/native_bias_record_${case_name}.raw"
    CUSTOM_SIDECAR="$ANALYSIS_ROOT/custom_a16_sidecar_${case_name}.raw"

    echo "=== run QNN Native w4a16_per_channel case=$case_name on $DEVICE ==="
    uv run python "$ROOT_DIR/scripts/run_qnn_python_case_native.py" \
        --case-dir "$CASE_DIR" \
        --out-dir "$NATIVE_DIR" \
        --device "$DEVICE"

    echo "=== sync custom qparams/bias_q to QNN Native quantized DLC case=$case_name ==="
    uv run python - "$CASE_DIR" "$NATIVE_DIR/analysis/dlc_bias_q_int32.npy" "$NATIVE_DIR/case.dlc" "$ROOT_DIR/tools/qnn-sdk" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

case_dir = Path(sys.argv[1])
dlc_bias_path = Path(sys.argv[2])
dlc_path = Path(sys.argv[3])
qnn_root = Path(sys.argv[4])
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

sys.path.insert(0, str(qnn_root / "lib/python"))
from qti.aisw.dlc_utils import snpe_dlc_utils

reader = snpe_dlc_utils.modeltools.IrDlcReader()
reader.open(str(dlc_path))
try:
    graph = reader.get_ir_graph(next(iter(reader.get_ir_graph_names())))
    conv = None
    for op in graph.get_ops():
        name = op.name() if callable(getattr(op, "name", None)) else op.name
        if name == "conv1x1":
            conv = op
            break
    if conv is None:
        raise SystemExit(f"{dlc_path}: missing conv1x1 op")
    tensors = list(conv.inputs()) + list(conv.outputs())
    by_name = {
        (t.name() if callable(getattr(t, "name", None)) else t.name): t
        for t in tensors
    }
    act_scale = float(by_name["A_0231"].get_encoding().encInfo.scale)
    out_scale = float(by_name["Y_0231"].get_encoding().encInfo.scale)
    weight_scales = np.array(
        [float(enc.scale) for enc in by_name["W"].get_encoding().axisEncInfo.encInfos],
        dtype=np.float32,
    )
finally:
    reader.close()

meta_path = case_dir / "case.json"
meta = json.loads(meta_path.read_text(encoding="utf-8"))
old_act = float(meta["qparams"]["activation"]["scale"])
old_out = float(meta["qparams"]["output"]["scale"])
meta["qparams"]["activation"]["scale"] = act_scale
meta["qparams"]["activation"]["min"] = (0 - int(meta["qparams"]["activation"]["zero_point"])) * act_scale
meta["qparams"]["activation"]["max"] = ((1 << int(meta["qparams"]["activation"]["bitwidth"])) - 1 - int(meta["qparams"]["activation"]["zero_point"])) * act_scale
meta["qparams"]["output"]["scale"] = out_scale
meta["qparams"]["output"]["min"] = (0 - int(meta["qparams"]["output"]["zero_point"])) * out_scale
meta["qparams"]["output"]["max"] = ((1 << int(meta["qparams"]["output"]["bitwidth"])) - 1 - int(meta["qparams"]["output"]["zero_point"])) * out_scale
meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
np.save(case_dir / "weight_scale.npy", weight_scales)
weight_scales.tofile(case_dir / "weight_scale.raw")
print(
    "DLC encoding scale sync:",
    {
        "act_delta": act_scale - old_act,
        "out_delta": out_scale - old_out,
        "weight_scale_shape": list(weight_scales.shape),
    },
)
PY

    echo "=== extract native w4a16 bias/control sidecar case=$case_name ==="
    uv run python "$ROOT_DIR/scripts/analyze_a16_native_bias_record.py" \
        --case-dir "$CASE_DIR" \
        --native-context-bin "$NATIVE_DIR/ctx/case_native_ctx.bin" \
        --out-raw "$BIAS_RECORD" \
        | tee "$ANALYSIS_ROOT/native_bias_record_${case_name}.txt"

    echo "=== build custom w4a16 A16 sidecar source=$SIDECAR_SOURCE case=$case_name ==="
    uv run python "$ROOT_DIR/scripts/build_w8a16_custom_a16_sidecar.py" \
        --case-dir "$CASE_DIR" \
        --native-sidecar-raw "$BIAS_RECORD" \
        --source "$SIDECAR_SOURCE" \
        --out-raw "$CUSTOM_SIDECAR" \
        --json-out "$ANALYSIS_ROOT/custom_a16_sidecar_${case_name}.json"

    echo "=== run custom w4a16 sidecar_source=$SIDECAR_SOURCE case=$case_name on $DEVICE ==="
    VERIFY_NATIVE_RAW="$NATIVE_DIR/device_out/Y.raw" \
    VERIFY_NATIVE_TRANSPOSE=1 \
    VERIFY_ABS_TOL=0 \
    ANALYZE_W4A16=0 \
    M="$M" K="$K" N="$N" CHAIN=1 MODE=chain_qdq OP_INPUT_LAYOUT=tiled \
    OUT_DIR="$CUSTOM_DIR" DEVICE="$DEVICE" \
    GEN_EXTRA_ARGS="--case-dir $CASE_DIR --a16-bias-sidecar-raw $CUSTOM_SIDECAR" \
        bash "$ROOT_DIR/example/qnn_hmx_matmul_w4a16/standard_flow/custom_w4a16/run_w4a16_chain.sh"

    echo "=== compare custom sidecar against QNN Native final sidecar case=$case_name ==="
    uv run python "$ROOT_DIR/scripts/analyze_a16_native_bias_record.py" \
        --case-dir "$CASE_DIR" \
        --native-context-bin "$NATIVE_DIR/ctx/case_native_ctx.bin" \
        --custom-onnx "$CUSTOM_DIR/w4a16.onnx" \
        | tee "$ANALYSIS_ROOT/custom_bias_record_${case_name}.txt"

    echo "=== compare custom/native exactness case=$case_name ==="
    uv run python - "$OUT_ROOT" "$case_name" "$NATIVE_DIR" "$CUSTOM_DIR" "$M" "$N" "$SIDECAR_SOURCE" <<'PY'
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
sidecar_source = sys.argv[7]

native = np.load(native_dir / "analysis" / "native_output_mn.npy").astype(np.uint16).reshape(m, n)
custom = np.fromfile(custom_dir / "device_out" / "out.raw", dtype="<u2").reshape(m, n)
diff = custom.astype(np.int64) - native.astype(np.int64)
exact = int((diff == 0).sum())
total = int(diff.size)
summary = {
    "comparison_scope": "same_hardware_custom_hmx_vs_qnn_native_htp",
    "family": "w4a16_per_channel",
    "case": case_name,
    "exact": exact,
    "total": total,
    "maxabs": int(np.abs(diff).max()) if total else 0,
    "diff_counts": {str(int(v)): int((diff == v).sum()) for v in np.unique(diff)},
    "custom_sidecar_source": sidecar_source,
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
    "family": "w4a16_per_channel",
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

echo "PASS: w4a16 per-channel Python-case custom/native exactness ($CASES)"
