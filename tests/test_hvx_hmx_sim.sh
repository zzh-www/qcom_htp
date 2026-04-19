#!/usr/bin/env bash
#
# test_hvx_hmx_sim.sh — E2E test: compile test_hvx_hmx.c, run on hexagon-sim,
# validate that it prints "Results: 3 PASS / 0 FAIL".
#
# Prereqs:
#   1. bash install.sh                                          (SDKs)
#   2. bash docs/hexagon-tutorial/ch01-simulator-setup/install_tools.sh
#      (builds H2 hypervisor booter used by the simulator)
#
# Exit: 0 on pass, 1 on simulation mismatch, 2 on missing prereq.
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env.sh"

SRC="$ROOT_DIR/docs/hexagon-tutorial/ch01-simulator-setup/test_hvx_hmx.c"
H2_INSTALL="$ROOT_DIR/tools/h2-install"

if [ -L "$H2_INSTALL" ]; then
    H2_ROOT="$(dirname "$(readlink -f "$H2_INSTALL")")"
else
    H2_ROOT="$(cd "$H2_INSTALL/.." && pwd)"
fi
H2_KERNEL_INC="$H2_ROOT/kernel/include"

# ---- Preconditions ----
for tool in hexagon-clang hexagon-sim; do
    command -v "$tool" >/dev/null || {
        echo "ERROR: $tool not on PATH (source env.sh / run install.sh)" >&2
        exit 2
    }
done
[ -f "$SRC" ] || { echo "ERROR: source not found: $SRC" >&2; exit 2; }
if [ ! -x "$H2_INSTALL/bin/booter" ]; then
    cat >&2 <<EOF
ERROR: H2 hypervisor booter not found at $H2_INSTALL/bin/booter
       Build it first:
         bash $ROOT_DIR/docs/hexagon-tutorial/ch01-simulator-setup/install_tools.sh
EOF
    exit 2
fi

# ---- Compile ----
WORK="$(mktemp -d)"
cleanup_work() { rm -rf "$WORK"; }
trap cleanup_work EXIT
BIN="$WORK/test_hvx_hmx"

echo "=== Compile ==="
hexagon-clang -O2 -mv75 \
    -mhvx -mhvx-length=128B \
    -mhmx \
    -DARCHV=75 \
    -I "$H2_INSTALL/include" \
    -I "$H2_KERNEL_INC" \
    -moslib=h2 \
    -Wl,-L,"$H2_INSTALL/lib" \
    -Wl,--section-start=.start=0x02000000 \
    -o "$BIN" "$SRC"
echo "  -> $BIN"

# ---- Simulate ----
echo ""
echo "=== Simulate ==="
LOG="$WORK/sim.log"
hexagon-sim --mv75 --mhmx 1 --simulated_returnval \
    -- "$H2_INSTALL/bin/booter" \
       --ext_power 1 \
       --use_ext 1 \
       --fence_hi 0xfe000000 \
       "$BIN" 2>&1 | tee "$LOG"

# ---- Validate ----
# Two-stage check:
#   1. Summary line must be exactly "Results: 3 PASS / 0 FAIL".
#   2. The deterministic program-output section must diff-clean against
#      tests/golden/sim_output.txt (catches regressions in intermediate
#      HVX/HMX values, not just the final count).
echo ""
echo "=== Validate ==="
GOLDEN="$ROOT_DIR/tests/golden/sim_output.txt"
[ -f "$GOLDEN" ] || { echo "ERROR: golden not found: $GOLDEN" >&2; exit 2; }

ACTUAL="$WORK/actual.txt"
sed -n '/Chapter 1: HVX/,/Results: [0-9]* PASS/p' "$LOG" > "$ACTUAL"

fail() {
    KEEP_LOG="$ROOT_DIR/tests/last-failure.log"
    KEEP_ACTUAL="$ROOT_DIR/tests/last-failure-actual.txt"
    cp "$LOG" "$KEEP_LOG"
    cp "$ACTUAL" "$KEEP_ACTUAL"
    trap - EXIT
    rm -rf "$WORK"
    echo "FAIL: $1" >&2
    echo "      full log:    $KEEP_LOG" >&2
    echo "      extracted:   $KEEP_ACTUAL" >&2
    echo "      golden:      $GOLDEN" >&2
    exit 1
}

if ! grep -Eq '^[[:space:]]*Results:[[:space:]]+3 PASS[[:space:]]+/[[:space:]]+0 FAIL' "$LOG"; then
    fail "simulator output did not contain 'Results: 3 PASS / 0 FAIL'"
fi

if ! diff -u "$GOLDEN" "$ACTUAL"; then
    fail "simulator output diverged from golden (see diff above)"
fi

echo "PASS: HVX/HMX simulator E2E (3/3 subtests, diff-clean vs golden)"
exit 0
