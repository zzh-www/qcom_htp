#!/usr/bin/env bash
# CI gate for the QNN-free standalone W16A16 HMX MatMul (handwriting route).
#
# Prepares the owned w16a16 artifact, then runs the byte-verified kernel body
# bare in hexagon-sim (zero QNN) via two independent harnesses and asserts the
# output is byte-exact to the QNN native Y.raw:
#
#   1. scripts/run_handwritten_artifact_body_sim.py --family w16a16
#        (the generic body-sim runner; split-N128 deblock)
#   2. scripts/run_w16a16_standalone_kernel.py
#        (the dedicated op-faithful single-surface driver)
#
# Both depend on the corrected conv1x1_words mask emulation (arg5=0x80 dilate
# word6=0x3ff). Sim-only -> portable; no device required.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-/tmp/handwritten_hmx_matmul_gate}"
OUT_DIR="$OUT_ROOT/w16a16_standalone"
cd "$ROOT_DIR"
mkdir -p "$OUT_DIR"

echo "=== [w16a16-standalone] prepare owned artifact ==="
uv run python example/handwritten_hmx_matmul/run_owned_smoke.py \
  --family w16a16 --out-dir "$OUT_DIR/artifact"

echo "=== [w16a16-standalone] generic body-sim (byte-exact vs native) ==="
uv run python scripts/run_handwritten_artifact_body_sim.py \
  --family w16a16 --artifact "$OUT_DIR/artifact" \
  --json-out "$OUT_DIR/artifact_body_w16a16.json"
uv run python - "$OUT_DIR/artifact_body_w16a16.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))["result"]
status = d.get("exactness_status")
matches = d.get("checksum_matches_native_raw")
assert status == "byte_exact_checksum" and matches is True, \
    f"generic body-sim NOT byte-exact: status={status} matches={matches}"
print(f"  generic body-sim: {status} matches_native={matches}")
PY

echo "=== [w16a16-standalone] dedicated standalone driver (byte-exact vs native) ==="
uv run python scripts/run_w16a16_standalone_kernel.py \
  --artifact "$OUT_DIR/artifact" --json-out "$OUT_DIR/standalone.json"
uv run python - "$OUT_DIR/standalone.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("byte_exact") is True and d.get("diff_bytes") == 0, \
    f"standalone NOT byte-exact: byte_exact={d.get('byte_exact')} diff_bytes={d.get('diff_bytes')}"
print(f"  standalone: BYTE-EXACT out={d.get('output_checksum')} native={d.get('native_checksum')}")
PY

echo "w16a16 standalone (QNN-free): ALL BYTE-EXACT"
