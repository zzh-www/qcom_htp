#!/usr/bin/env bash
# W16_N8 act-cache bit-exact gate: A (FUSEDEP, no actcache, oracle) vs C (FUSEDEP + actcache).
# Same deterministic input + scales; pull both 32-head T; md5 / cmp / oc.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"; BM="$ROOT/example/gdn_native/baremetal"
SHIP="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL -DGDN_BR_SBOOST -DGDN_BR_DIAG_I16 -DGDN_BR_REQ_FUSE -DGDN_BR_FBOOST"
BASE="$SHIP -DGDN_BR_W16 -DGDN_BR_W16_N8 -DGDN_BR_W16_N8_FUSEDEP"
SCALES="2.770166930875267e-05 6.103701895199438e-05"
source "$ROOT/scripts/dssh.sh"; dssh_open

build() { ( cd "$BM" && EXTRA_DEFS="$1" bash build.sh ) >/tmp/b_$2.log 2>&1 || { echo "BUILD FAIL $2"; tail -20 /tmp/b_$2.log; exit 1; }
  mkdir -p /tmp/w16ac_$2; cp "$BM/build/libgdnbm_skel.so" "$BM/build/gdnbm" /tmp/w16ac_$2/; }
echo "=== build A (FUSEDEP, no actcache) ==="; build "$BASE" A
echo "=== build C (FUSEDEP + actcache)  ==="; build "$BASE -DGDN_BR_W16_N8_ACTCACHE" C

for V in A C; do W="\$HOME/w16ac_$V"; dssh "mkdir -p $W"
  dssh "cat > $W/libgdnbm_skel.so" < /tmp/w16ac_$V/libgdnbm_skel.so
  dssh "cat > $W/gdnbm" < /tmp/w16ac_$V/gdnbm; dssh "chmod +x $W/gdnbm"
  dssh "cat > $W/A_u16_h32.raw" < "$BM/A_u16_h32.raw"
done

run() { local V="$1" W="\$HOME/w16ac_$1"
  dssh "cd $W && GDNBM_REPS=4 LD_LIBRARY_PATH=\$PWD:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH=\"\$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" ./gdnbm 4 A_u16_h32.raw T_${V}.raw 32 256 32768 32768 $SCALES 2>&1" | tail -8
  dssh "cat $W/T_${V}.raw" > /tmp/w16ac_T_$V.raw 2>/dev/null; }
echo "=== run A ==="; run A
echo "=== run C ==="; run C

echo ""; echo "### bit-exact ###"
md5sum /tmp/w16ac_T_A.raw /tmp/w16ac_T_C.raw
if cmp -s /tmp/w16ac_T_A.raw /tmp/w16ac_T_C.raw; then echo "cmp: IDENTICAL (byte-exact)"; else echo "cmp: DIFFER!!"; cmp /tmp/w16ac_T_A.raw /tmp/w16ac_T_C.raw | head; fi
python3 - <<PY
import numpy as np
a=np.fromfile('/tmp/w16ac_T_A.raw',dtype=np.uint16); c=np.fromfile('/tmp/w16ac_T_C.raw',dtype=np.uint16)
print(f"len A={a.size} C={c.size}  equal={np.array_equal(a,c)}  max|d|={int(np.abs(a.astype(np.int32)-c.astype(np.int32)).max()) if a.size==c.size else 'NA'}")
PY
echo "### oc (gate 4e-2) ###"
for V in A C; do echo -n "$V: "; python3 scripts/gdn_solve_oc_check.py "$BM/A_u16_h32.raw" /tmp/w16ac_T_$V.raw --gate 4e-2 2>&1 | grep -iE 'oc|pass|fail' | head -2 | tr '\n' ' '; echo; done
