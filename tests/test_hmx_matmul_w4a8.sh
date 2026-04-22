#!/usr/bin/env bash
#
# test_hmx_matmul_w4a8.sh — E2E sim test for the w4a8 HMX kernel.
# Runs on hexagon-sim + H2 booter. Matches pattern of test_hmx_matmul_int16.sh.
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env.sh"

EX="$ROOT_DIR/example/hmx_matmul_w4a8"
H2_INSTALL="$ROOT_DIR/tools/h2-install"

if [ ! -x "$H2_INSTALL/bin/booter" ]; then
    echo "ERROR: H2 booter not found at $H2_INSTALL/bin/booter" >&2
    exit 2
fi

echo "=== Build ==="
bash "$EX/build_sim.sh"

echo ""
echo "=== Run on simulator ==="
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

hexagon-sim --mv75 --mhmx 1 --simulated_returnval \
    -- "$H2_INSTALL/bin/booter" \
       --ext_power 1 --use_ext 1 --fence_hi 0xfe000000 \
       "$EX/test_w4a8_matmul_sim" 2>&1 | tee "$LOG"

echo ""
echo "=== Validate ==="
pass=$(grep -c '\[PASS\]' "$LOG" || true)
fail=$(grep -c '\[FAIL\]' "$LOG" || true)

echo "  PASS lines: $pass   FAIL lines: $fail"
if [ "$pass" -eq 3 ] && [ "$fail" -eq 0 ]; then
    echo "PASS: HMX w4a8 matmul E2E (3/3 scenarios bit-exact)"
    exit 0
fi
echo "FAIL: expected 3 PASS / 0 FAIL" >&2
exit 1
