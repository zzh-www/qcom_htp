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
W4_ENCODING="${W4_ENCODING:-symmetric}"
LPBQ_BUILD_FLAG=0
FAMILY_LABEL="w4a16_per_channel"
CASE_FAMILY="w4a16_per_channel"
CUSTOM_VERIFY_REF=1
CUSTOM_MODE=chain_qdq
CUSTOM_OP_INPUT_LAYOUT=tiled
if [ "$W4_ENCODING" = "lpbq" ]; then
    LPBQ_BUILD_FLAG=1
    FAMILY_LABEL="w4a16_lpbq"
    CASE_FAMILY="w4a16_lpbq"
    CUSTOM_VERIFY_REF=0
    CUSTOM_MODE=direct
    CUSTOM_OP_INPUT_LAYOUT=tiled
fi

if [ "$CHAIN" != "1" ]; then
    echo "ERROR: w4a16 per-channel native_match currently supports CHAIN=1 only" >&2
    exit 2
fi

CASE_ROOT="$OUT_ROOT/cases"
ANALYSIS_ROOT="$OUT_ROOT/analysis"
mkdir -p "$OUT_ROOT" "$ANALYSIS_ROOT"

if [ "$BUILD_PACKAGES" = "1" ]; then
    echo "=== build $FAMILY_LABEL package ==="
    LPBQ_ONLY="$LPBQ_BUILD_FLAG" \
    EXTRA_DEFS="-UHMX_W4A16_SKIP_KERNEL -DHMX_W4A16_ALLOW_UNVALIDATED_KERNEL" \
        bash "$ROOT_DIR/example/qnn_hmx_matmul_w4a16/build.sh"
    LPBQ_ONLY="$LPBQ_BUILD_FLAG" \
    EXTRA_DEFS="-UHMX_W4A16_SKIP_KERNEL -DHMX_W4A16_ALLOW_UNVALIDATED_KERNEL" \
        bash "$ROOT_DIR/example/qnn_hmx_matmul_w4a16/build_x86.sh"
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
    CUSTOM_SIDECAR="$ANALYSIS_ROOT/custom_a16_sidecar_${case_name}.raw"

    echo "=== run QNN Native w4a16_per_channel case=$case_name on $DEVICE ==="
    uv run python "$ROOT_DIR/scripts/run_qnn_python_case_native.py" \
        --case-dir "$CASE_DIR" \
        --out-dir "$NATIVE_DIR" \
        --device "$DEVICE" \
        --w4-encoding "$W4_ENCODING"

    echo "=== sync custom qparams/bias_q to QNN Native quantized DLC case=$case_name ==="
    uv run python - "$CASE_DIR" "$NATIVE_DIR/analysis/dlc_bias_q_int32.npy" "$NATIVE_DIR/case.dlc" "$ROOT_DIR/tools/qnn-sdk" "$W4_ENCODING" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

case_dir = Path(sys.argv[1])
dlc_bias_path = Path(sys.argv[2])
dlc_path = Path(sys.argv[3])
qnn_root = Path(sys.argv[4])
w4_encoding = sys.argv[5]
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
    if w4_encoding == "lpbq":
        weight_scales = np.load(case_dir / "weight_scale.npy").astype(np.float32)
    else:
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
    VERIFY_REF="$CUSTOM_VERIFY_REF" \
    ANALYZE_W4A16=0 \
    M="$M" K="$K" N="$N" CHAIN=1 MODE="$CUSTOM_MODE" OP_INPUT_LAYOUT="$CUSTOM_OP_INPUT_LAYOUT" W4_ENCODING="$W4_ENCODING" \
    OUT_DIR="$CUSTOM_DIR" DEVICE="$DEVICE" \
    GEN_EXTRA_ARGS="--case-dir $CASE_DIR --a16-bias-sidecar-raw $CUSTOM_SIDECAR" \
        bash "$ROOT_DIR/example/qnn_hmx_matmul_w4a16/standard_flow/custom_w4a16/run_w4a16_chain.sh"

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
    if info.get("lpbq_matmul_op") != "HmxU16I8ToU16MatMul":
        raise SystemExit(f"{native_io_path}: unexpected LPBQ matmul op {info.get('lpbq_matmul_op')!r}")
    print(f"{native_io_path}: custom LPBQ two-op metadata ok op={info['lpbq_matmul_op']}")

check_native_lpbq(sys.argv[1])
check_custom_two_op(sys.argv[2], sys.argv[3])
PY
    fi

    echo "=== compare custom sidecar against QNN Native final sidecar case=$case_name ==="
    uv run python "$ROOT_DIR/scripts/analyze_a16_native_bias_record.py" \
        --case-dir "$CASE_DIR" \
        --native-context-bin "$NATIVE_DIR/ctx/case_native_ctx.bin" \
        --custom-onnx "$CUSTOM_DIR/w4a16.onnx" \
        | tee "$ANALYSIS_ROOT/custom_bias_record_${case_name}.txt"

    echo "=== compare custom/native exactness case=$case_name ==="
    uv run python - "$OUT_ROOT" "$case_name" "$NATIVE_DIR" "$CUSTOM_DIR" "$M" "$N" "$SIDECAR_SOURCE" "$FAMILY_LABEL" <<'PY'
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
family_label = sys.argv[8]

native = np.load(native_dir / "analysis" / "native_output_mn.npy").astype(np.uint16).reshape(m, n)
custom = np.fromfile(custom_dir / "device_out" / "out.raw", dtype="<u2").reshape(m, n)
diff = custom.astype(np.int64) - native.astype(np.int64)
exact = int((diff == 0).sum())
total = int(diff.size)
summary = {
    "comparison_scope": "same_hardware_custom_hmx_vs_qnn_native_htp",
    "family": family_label,
    "case_family": family_label,
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
