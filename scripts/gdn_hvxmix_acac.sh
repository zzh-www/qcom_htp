#!/usr/bin/env bash
# ACAC same-thermal-window paired runner for the GDNSolveHVXMixHMX baremetal harness.
# Builds A-variant and C-variant binaries, deploys both, runs reps8 ACAC interleaved x3 windows.
# metric = stats[0] graph-wall (32-head TOTAL, VTCM-only), reps2-8 median per run.
#
# usage: gdn_hvxmix_acac.sh "<A_EXTRA_DEFS>" "<C_EXTRA_DEFS>" <tag>
# env:   WINDOWS (default 3)  REPS (default 8)  THREADS (default 4)  PULL_T=1 to fetch T per variant
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BM="$ROOT/example/gdn_native/baremetal"
A_DEFS="$1"; C_DEFS="$2"; TAG="${3:-acac}"
WINDOWS="${WINDOWS:-3}"; REPS="${REPS:-8}"; THREADS="${THREADS:-4}"
SHIP="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL -DGDN_BR_SBOOST -DGDN_BR_DIAG_I16 -DGDN_BR_REQ_FUSE -DGDN_BR_FBOOST"

source "$ROOT/scripts/dssh.sh"; dssh_open

build_variant() {  # $1=defs $2=outdir
  local defs="$1" od="$2"
  ( cd "$BM" && EXTRA_DEFS="$defs" bash build.sh ) >/tmp/build_$TAG.log 2>&1 || { echo "BUILD FAIL ($od)"; tail -20 /tmp/build_$TAG.log; exit 1; }
  mkdir -p "$od"; cp "$BM/build/libgdnbm_skel.so" "$od/"; cp "$BM/build/gdnbm" "$od/"
}

echo "=== build A: $SHIP $A_DEFS"
build_variant "$SHIP $A_DEFS" /tmp/${TAG}_A
echo "=== build C: $SHIP $C_DEFS"
build_variant "$SHIP $C_DEFS" /tmp/${TAG}_C

# deploy both into separate device dirs
WA="\$HOME/gdnbm_${TAG}_A"; WC="\$HOME/gdnbm_${TAG}_C"
for V in A C; do
  W=$([ $V = A ] && echo "$WA" || echo "$WC")
  src=/tmp/${TAG}_${V}
  dssh "mkdir -p $W"
  dssh "cat > $W/libgdnbm_skel.so" < "$src/libgdnbm_skel.so"
  dssh "cat > $W/gdnbm" < "$src/gdnbm"; dssh "chmod +x $W/gdnbm"
  dssh "cat > $W/A_u16_h32.raw" < "$BM/A_u16_h32.raw"
done

run_one() {  # $1=A|C  -> prints "wall_med min max n" + writes /tmp/${TAG}_${V}_T_win${win}.raw if PULL_T
  local V="$1" win="$2"
  local W=$([ $V = A ] && echo "$WA" || echo "$WC")
  local out
  out=$(dssh "cd $W && GDNBM_REPS=$REPS LD_LIBRARY_PATH=\$PWD:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH=\"\$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" ./gdnbm $THREADS A_u16_h32.raw T_${V}.raw 32 256 32768 32768 3.05e-5 3.05e-5 2>&1")
  echo "$out" > /tmp/${TAG}_${V}_win${win}.log
  # parse reps2-N walls
  local walls
  walls=$(echo "$out" | grep -oE "rep=[0-9]+/[0-9]+ +wall=[0-9]+" | sed -E 's/rep=([0-9]+).*wall=([0-9]+)/\1 \2/')
  python3 - "$win" "$V" <<PY
import sys
win, V = sys.argv[1], sys.argv[2]
rows=[]
for ln in '''$walls'''.strip().splitlines():
    r,w=ln.split(); rows.append((int(r),int(w)))
steady=sorted(w for r,w in rows if r>=2)
if not steady:
    print(f"{V} win{win}: NO_WALL", file=sys.stderr); print("0 0 0 0"); sys.exit()
n=len(steady)
med=steady[n//2] if n%2 else (steady[n//2-1]+steady[n//2])//2
print(f"{med} {steady[0]} {steady[-1]} {n}")
PY
}

echo ""
echo "### ACAC x$WINDOWS windows, REPS=$REPS, THREADS=$THREADS  (metric=stats[0] reps2-$REPS median) ###"
declare -a AW CW
for win in $(seq 1 $WINDOWS); do
  # A C A C interleaved within the window
  ra1=$(run_one A $win); rc1=$(run_one C ${win}b)
  amed=$(echo $ra1 | cut -d' ' -f1); cmed=$(echo $rc1 | cut -d' ' -f1)
  AW+=($amed); CW+=($cmed)
  if [ "$amed" -gt 0 ] && [ "$cmed" -gt 0 ]; then
    delta=$(python3 -c "print(f'{($cmed-$amed)/$amed*100:+.2f}')")
  else delta="NA"; fi
  echo "win$win: A=$ra1 | C=$rc1 | dWall=${delta}%"
  if [ "${PULL_T:-0}" = "1" ] && [ "$win" = "1" ]; then
    dssh "cat $WA/T_A.raw" > /tmp/${TAG}_T_A.raw 2>/dev/null
    dssh "cat $WC/T_C.raw" > /tmp/${TAG}_T_C.raw 2>/dev/null
  fi
done

# P3.2: emit the canonical §7 perf report for the A-variant (win1 log) — additive, the A/C delta above
# is unchanged. SHIP/ARES = the "hvxmix" §4 stats mapping; P=THREADS.
if [ -s "/tmp/${TAG}_A_win1.log" ]; then
  python3 "$ROOT/scripts/htp_harness_report.py" "/tmp/${TAG}_A_win1.log" hvxmix "$THREADS" "/tmp/htp_perf/${TAG}_A" \
    && echo "  -> §7 perf report /tmp/htp_perf/${TAG}_A/report.txt"
fi

echo ""
python3 - <<PY
A=[${AW[@]/%/,}]; C=[${CW[@]/%/,}]
A=[x for x in A if x>0]; C=[x for x in C if x>0]
def med(v):
    v=sorted(v); n=len(v); return v[n//2] if n%2 else (v[n//2-1]+v[n//2])//2
if A and C:
    am,cm=med(A),med(C)
    print(f"3-WINDOW MERGED: A_med={am} C_med={cm}  dWall={ (cm-am)/am*100:+.2f}%")
    print(f"  A wins={A}  C wins={C}")
PY
