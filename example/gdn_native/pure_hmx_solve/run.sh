#!/usr/bin/env bash
# Build + run the isolated pure-HMX (all-w16a16) GDN-inverse schedule bench on device, render timeline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT/example/gdn_native/baremetal"
EXTRA_DEFS="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL -DGDNBM_PURE_HMX_SOLVE" bash build.sh
source "$ROOT/scripts/dssh.sh"; dssh_open oneplus
W=$(dssh 'echo $HOME/gdnbm_run')
dssh "cat > $W/libgdnbm_skel.so" < build/libgdnbm_skel.so
dssh "cat > $W/gdnbm" < build/gdnbm; dssh "chmod +x $W/gdnbm"
dssh "cd $W && GDNBM_REPS=3 LD_LIBRARY_PATH=$W:/vendor/lib64:/system/lib64 \
  ADSP_LIBRARY_PATH='$W;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' \
  ./gdnbm 4 A_u16_h32.raw T.raw 32 256 32768 32768 2.770166930875267e-05 6.103701895199438e-05 2>&1" \
  | grep -iE 'PUREHMX |solve rc'
dssh "cat $W/T.raw" > /tmp/tl_phs.raw
echo "=== pure-HMX per-thread timeline (32-head) ==="
GDN_TL_COARSE=1 uv run --project "$ROOT" python "$ROOT/scripts/gdn_pipe_timeline.py" /tmp/tl_phs.raw 116 | head -11
