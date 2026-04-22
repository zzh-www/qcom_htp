#!/usr/bin/env bash
# test_hmx_matmul_w4a16.sh — sim E2E test for the w4a16 HMX kernel.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/scripts/env.sh"
EX="$ROOT_DIR/example/hmx_matmul_qnn"
H2_INSTALL="$ROOT_DIR/tools/h2-install"
[ -x "$H2_INSTALL/bin/booter" ] || { echo "ERROR: H2 booter missing"; exit 2; }
echo "=== Build ==="
bash "$EX/build_sim.sh"
echo "=== Run on simulator ==="
LOG="$(mktemp)"; trap 'rm -f "$LOG"' EXIT
hexagon-sim --mv75 --mhmx 1 --simulated_returnval \
    -- "$H2_INSTALL/bin/booter" \
       --ext_power 1 --use_ext 1 --fence_hi 0xfe000000 \
       "$EX/test_w4a16_matmul_sim" 2>&1 | tee "$LOG"
pass=$(grep -c '\[PASS\]' "$LOG" || true)
fail=$(grep -c '\[FAIL\]' "$LOG" || true)
echo "  PASS=$pass  FAIL=$fail"
[ "$pass" -eq 3 ] && [ "$fail" -eq 0 ] && { echo "PASS: w4a16 sim (3/3)"; exit 0; }
echo "FAIL" >&2; exit 1
