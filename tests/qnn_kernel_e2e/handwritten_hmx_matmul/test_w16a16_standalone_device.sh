#!/usr/bin/env bash
# Device CI gate for the QNN-free standalone W16A16 HMX MatMul (handwriting route).
#
# Runs the byte-verified kernel body on REAL CDSP via run_main_on_hexagon (zero
# QNN) and asserts the recovered linear output is byte-exact to the QNN native
# Y.raw (row4 deblock diff == 0). This is the device-backed counterpart of
# test_w16a16_standalone.sh, matching the device-body coverage the other active
# families get.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/tmp/handwritten_hmx_matmul_gate}"
OUT_DIR="$OUT_ROOT/w16a16_standalone"
DEVICE="${DEVICE:-oneplus}"
cd "$ROOT_DIR"
mkdir -p "$OUT_DIR"

# Reuse the artifact prepared by test_w16a16_standalone.sh; prepare if absent.
if [[ ! -f "$OUT_DIR/artifact/prepared_state/mask_control.raw" ]]; then
  uv run python example/handwritten_hmx_matmul/run_owned_smoke.py \
    --family w16a16 --out-dir "$OUT_DIR/artifact"
fi

echo "=== [w16a16-standalone-device] run kernel body on real CDSP ==="
uv run python scripts/run_w16a16_standalone_device.py \
  --artifact "$OUT_DIR/artifact" --device "$DEVICE" \
  --json-out "$OUT_DIR/standalone_device.json"

uv run python - "$OUT_DIR/standalone_device.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("byte_exact") is True and d.get("row4_diff_bytes") == 0, \
    f"device standalone NOT byte-exact: {d}"
print(f"  device standalone: BYTE-EXACT row4_diff={d.get('row4_diff_bytes')}/{d.get('total_bytes')}")
PY
echo "w16a16 standalone (QNN-free) on device: BYTE-EXACT"
