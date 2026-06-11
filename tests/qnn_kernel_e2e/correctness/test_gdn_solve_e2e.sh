#!/usr/bin/env bash
#
# GDN triangular-inverse (GDNSolveHVXMixHMX) E2E precision gate.
#
# Builds the shipping baremetal solve (7-flag default), runs it on the v75 device, and asserts the
# per-head T = (I-A)^-1 matches fp64 inv(I-A) within the oc gate.  Gates on PRECISION (oc), not wall
# (wall drifts with device thermal state and is not a deterministic CI signal).
#
# Best implementation + numbers + reproduce: docs/gdn_inverse.md
# Env: DSSH_HOST / DEVICE = target device (default via scripts/dssh.sh); ARTIFACT_ONLY=1 = build only.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

BM="example/gdn_native/baremetal"
FLAGS="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL -DGDN_BR_SBOOST -DGDN_BR_DIAG_I16 -DGDN_BR_REQ_FUSE -DGDN_BR_FBOOST"
GATE="${GDN_OC_GATE:-1.05e-2}"
SA="2.770166930875267e-05"; ST="6.103701895199438e-05"

echo "=== gdn solve e2e: build (shipping 7-flag) ==="
( cd "$BM" && EXTRA_DEFS="$FLAGS" bash build.sh )
test -f "$BM/build/libgdnbm_skel.so" && test -f "$BM/build/gdnbm" || { echo "FAIL: build artifacts missing"; exit 1; }
test -f "$BM/A_u16_h32.raw" || { echo "FAIL: golden input A_u16_h32.raw missing"; exit 1; }

if [ "${ARTIFACT_ONLY:-0}" = "1" ]; then
  echo "ARTIFACT_ONLY=1: build OK, skipping device run."
  exit 0
fi

echo "=== gdn solve e2e: deploy + run on device ==="
# shellcheck disable=SC1091
source scripts/dssh.sh
dssh_open "${DSSH_HOST:-${DEVICE:-oneplus}}"
W="$(dssh 'echo $HOME/gdnbm_run')"
dssh "mkdir -p $W"
dssh "cat > $W/libgdnbm_skel.so" < "$BM/build/libgdnbm_skel.so"
dssh "cat > $W/gdnbm" < "$BM/build/gdnbm"; dssh "chmod +x $W/gdnbm"
dssh "cat > $W/A_u16_h32.raw" < "$BM/A_u16_h32.raw"
dssh "pkill -9 gdnbm 2>/dev/null || true; cd $W && GDNBM_REPS=4 \
  LD_LIBRARY_PATH=$W:/vendor/lib64:/system/lib64 \
  ADSP_LIBRARY_PATH='$W;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' \
  ./gdnbm 4 A_u16_h32.raw T.raw 32 256 32768 32768 $SA $ST" | grep -E "rc=|wall=" || true

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
dssh "cat $W/T.raw" > "$TMP/T.raw"

echo "=== gdn solve e2e: precision gate (oc vs fp64) ==="
python3 scripts/gdn_solve_oc_check.py "$BM/A_u16_h32.raw" "$TMP/T.raw" \
  --sA "$SA" --sT "$ST" --gate "$GATE"
echo "gdn solve e2e: PASS"
