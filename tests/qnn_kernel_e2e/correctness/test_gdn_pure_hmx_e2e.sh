#!/usr/bin/env bash
#
# GDN triangular-inverse PURE-HMX (GDNSolveHMX) E2E precision gate.
#
# Builds the shipping pure-HMX baremetal solve (production default = lean + wt-cache + lever-B, the
# `-DGDNBM_GDN_PURE_SOLVE` flag), runs it on the v75 device with the DETERMINISTIC golden input
# (run_w16a16_head_phase4.py seed=1, scale=0.05, cv-block GP_CVIO contract that the production
# default consumes), and asserts:
#   - oc  = mean(|T_dev - fp64 inv(I-A)|)/mean(|fp64 inv(I-A)|)  <  GDN_PURE_OC_GATE  (default 4e-2)
#   - PACKCHK == 0  (HVX weight-pack byte-exact vs the scalar packer == bit-exact-to-native feed)
#
# Gates on PRECISION (oc) + bit-exact feed (PACKCHK), NOT wall (wall drifts with device thermal state
# and is not a deterministic CI signal).  Input is fixed (seed=1) so the golden fp64 reference and the
# device output are both deterministic; no random/thermal dependence in the gate signal.
#
# This is the pure-HMX sibling of test_gdn_solve_e2e.sh (which gates the HVXMixHMX route).  The two
# routes use different I/O contracts (pure-HMX = cv-block, HVXMixHMX = linear), so this gate reuses
# the pure-HMX route's own deterministic golden+oc harness (run_w16a16_head_phase4.py) rather than the
# linear gdn_solve_oc_check.py.
#
# Best implementation + numbers + reproduce: docs/gdn_inverse_pure_hmx.md
# Env: DSSH_HOST / DEVICE = target device (default oneplus via scripts/dssh.sh); ARTIFACT_ONLY=1 = build only.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

BM="example/gdn_native/baremetal"
FLAGS="-DGDNBM_GDN_PURE_SOLVE"
GATE="${GDN_PURE_OC_GATE:-4e-2}"

echo "=== gdn pure-hmx e2e: build (production default -DGDNBM_GDN_PURE_SOLVE = lean+wt-cache+B) ==="
( cd "$BM" && EXTRA_DEFS="$FLAGS" bash build.sh )
test -f "$BM/build/libgdnbm_skel.so" && test -f "$BM/build/gdnbm" || { echo "FAIL: build artifacts missing"; exit 1; }

if [ "${ARTIFACT_ONLY:-0}" = "1" ]; then
  echo "ARTIFACT_ONLY=1: build OK, skipping device run."
  exit 0
fi

echo "=== gdn pure-hmx e2e: deploy + run on device (deterministic seed=1, cv-block contract) ==="
# shellcheck disable=SC1091
source scripts/dssh.sh
dssh_open "${DSSH_HOST:-${DEVICE:-oneplus}}"

# run_w16a16_head_phase4.py: builds the fixed seed=1 cv-block A, deploys gdnbm + the golden input,
# runs the pure-HMX solve on device, and prints "H=32 oc mean=..." (its own fp64 golden) plus the
# QNN-aligned breakdown line carrying "PACKCHK=N(0=ok)".  --reps 4 = steady-state (rep1 = cold).
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
uv run python scripts/run_w16a16_head_phase4.py \
  --deploy --threads 4 --heads 32 --scale 0.05 --reps 4 2>&1 | tee "$TMP/run.log"

echo "=== gdn pure-hmx e2e: precision + bit-exact-feed gate ==="
# oc: "H=32 oc mean=4.238e-03 max=... min=..."
OC="$(grep -oE 'oc mean=[0-9.eE+-]+' "$TMP/run.log" | head -1 | sed 's/oc mean=//')"
# PACKCHK: "... PACKCHK=0(0=ok)"
PACKCHK="$(grep -oE 'PACKCHK=[0-9-]+' "$TMP/run.log" | head -1 | sed 's/PACKCHK=//')"

test -n "$OC" || { echo "FAIL: could not parse 'oc mean=' from device run"; exit 1; }
test -n "$PACKCHK" || { echo "FAIL: could not parse 'PACKCHK=' from device run"; exit 1; }

OK_OC="$(python3 -c "import sys; sys.exit(0 if float('$OC') < float('$GATE') else 1)" && echo 1 || echo 0)"
if [ "$OK_OC" != "1" ]; then
  echo "FAIL: oc = $OC  (gate < $GATE)"; exit 1
fi
if [ "$PACKCHK" != "0" ]; then
  echo "FAIL: PACKCHK = $PACKCHK  (expected 0 = HVX wt-pack byte-exact vs scalar / bit-exact-to-native feed)"; exit 1
fi

echo "gdn pure-hmx e2e: oc = $OC  (gate < $GATE)  PACKCHK = $PACKCHK"
echo "gdn pure-hmx e2e: PASS"
