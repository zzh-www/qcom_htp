#!/usr/bin/env bash
# Capture per-thread STRUCTURE traces (magic 0x47545203 blob, single rep, GDN_BR_TRACE/GP_TRACE) for the
# three GDN-solve implementations, into /tmp/{ship,w16n8,purehmx}_trace.raw, for the aggregate renderer
# scripts/gdn_3impl_aggregate_timeline.py.  Single rep -> trace-perturbed; read STRUCTURE not absolute wall.
#   (0) SHIP u8i8      : 7 SHIP flags, no W16
#   (1) W16_N8 ARES    : SHIP + W16 + W16_N8 + FUSEDEP + ACTCACHE + ARES
#   (2) pure-HMX       : GDNBM_GDN_PURE_SOLVE + GP_TRACE (own gdn_pure_solve.cpp)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/example/gdn_native/baremetal"
source "$ROOT/scripts/dssh.sh"; dssh_open oneplus >/dev/null
W=$(dssh 'echo $HOME/gdnbm_run')
RUNARGS="A_u16_h32.raw T.raw 32 256 32768 32768 2.770166930875267e-05 6.103701895199438e-05"
SHIP="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL -DGDN_BR_SBOOST -DGDN_BR_DIAG_I16 -DGDN_BR_REQ_FUSE -DGDN_BR_FBOOST"

cap() { # name  extra_defs
  local name="$1"; shift
  echo "=== build+run [$name] ==="
  EXTRA_DEFS="$*" bash build.sh >/dev/null 2>&1 || { echo "BUILD FAIL $name"; return 1; }
  dssh "cat > $W/libgdnbm_skel.so" < build/libgdnbm_skel.so
  dssh "cat > $W/gdnbm" < build/gdnbm; dssh "chmod +x $W/gdnbm"
  dssh "cd $W && GDNBM_REPS=1 LD_LIBRARY_PATH=$W:/vendor/lib64:/system/lib64 \
    ADSP_LIBRARY_PATH='$W;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' \
    ./gdnbm 4 $RUNARGS 2>&1" | grep -iE 'PIPE TRACE|PUREHMX |solve rc|TRACE:' | head
  dssh "cat $W/T.raw" > "/tmp/${name}_trace.raw"
  echo "  -> /tmp/${name}_trace.raw ($(stat -c%s /tmp/${name}_trace.raw) bytes)"
}

cap ship    "$SHIP -DGDN_BR_TRACE"

# ---- W16_N8 ARES needs the EXTENDED A input (natural-A + cv-block tail). Feeding plain natural A makes the
# resident bulk-load read past the FastRPC buffer -> rc=0x8000040d. So: (1) build+run the DUMP variant to
# emit the cv-block A, (2) assemble A_ares.raw (natural + cv-block), (3) run ARES+TRACE with A_ares.raw.
ARES_BASE="$SHIP -DGDN_BR_W16 -DGDN_BR_W16_N8 -DGDN_BR_W16_N8_FUSEDEP -DGDN_BR_W16_N8_ACTCACHE"
echo "=== build+run [w16n8 DUMP] (emit cv-block A) ==="
EXTRA_DEFS="$ARES_BASE -DGDN_BR_W16_N8_ACVRES_DUMP" bash build.sh >/dev/null 2>&1 || { echo "DUMP BUILD FAIL"; exit 1; }
dssh "cat > $W/libgdnbm_skel.so" < build/libgdnbm_skel.so
dssh "cat > $W/gdnbm" < build/gdnbm; dssh "chmod +x $W/gdnbm"
dssh "cd $W && GDNBM_REPS=1 LD_LIBRARY_PATH=$W:/vendor/lib64:/system/lib64 \
  ADSP_LIBRARY_PATH='$W;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' \
  ./gdnbm 4 A_u16_h32.raw T_dump.raw 32 256 32768 32768 2.770166930875267e-05 6.103701895199438e-05 2>&1" | grep -iE 'solve rc' | head -1
dssh "cat $W/T_dump.raw" > /tmp/T_dump.raw
NATA="$ROOT/example/gdn_native/baremetal/A_u16_h32.raw" python3 - <<'PY'
import numpy as np, os
H=32; CC=256*256
dump=np.fromfile("/tmp/T_dump.raw", dtype="<u2")
nat =np.fromfile(os.environ["NATA"], dtype="<u2")
cv=np.zeros((H,6*4096),dtype="<u2")
for h in range(H): cv[h]=dump[h*CC : h*CC + 6*4096]
np.concatenate([nat, cv.ravel()]).tofile("/tmp/A_ares.raw")
print("  assembled /tmp/A_ares.raw")
PY
dssh "cat > $W/A_ares.raw" < /tmp/A_ares.raw
echo "=== build+run [w16n8 ARES+TRACE] (A_ares.raw) ==="
EXTRA_DEFS="$ARES_BASE -DGDN_BR_W16_N8_ARES -DGDN_BR_TRACE" bash build.sh >/dev/null 2>&1 || { echo "ARES BUILD FAIL"; exit 1; }
dssh "cat > $W/libgdnbm_skel.so" < build/libgdnbm_skel.so
dssh "cat > $W/gdnbm" < build/gdnbm; dssh "chmod +x $W/gdnbm"
dssh "cd $W && GDNBM_REPS=1 LD_LIBRARY_PATH=$W:/vendor/lib64:/system/lib64 \
  ADSP_LIBRARY_PATH='$W;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' \
  ./gdnbm 4 A_ares.raw T.raw 32 256 32768 32768 2.770166930875267e-05 6.103701895199438e-05 2>&1" | grep -iE 'solve rc|graph-wall' | head -2
dssh "cat $W/T.raw" > /tmp/w16n8_trace.raw
echo "  -> /tmp/w16n8_trace.raw ($(stat -c%s /tmp/w16n8_trace.raw) bytes)"

cap purehmx "-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL -DGDNBM_GDN_PURE_SOLVE -DGP_TRACE"

# P3.2: auto-render the canonical aggregate timeline (htp_timeline.py, the ONE tool; §5 stage spec) from
# the three trace blobs just captured. Additive — the .raw blobs above are still written as before.
OUT="${TIMELINE_SVG:-/tmp/htp_perf/3impl_timeline.svg}"; mkdir -p "$(dirname "$OUT")"
if [ -s /tmp/ship_trace.raw ] && [ -s /tmp/w16n8_trace.raw ] && [ -s /tmp/purehmx_trace.raw ]; then
  python3 "$ROOT/scripts/htp_timeline.py" aggregate /tmp/ship_trace.raw /tmp/w16n8_trace.raw /tmp/purehmx_trace.raw "$OUT" \
    && echo "  -> aggregate timeline $OUT (SHIP+ARES+pure-HMX, §5 canonical)"
fi

echo "DONE"
