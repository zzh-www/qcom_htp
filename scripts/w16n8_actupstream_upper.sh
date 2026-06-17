#!/usr/bin/env bash
# W16_N8 act-upstream UPPER-BOUND ACAC (phase 1, cheapest kill of the int16 magnitude).
# A = W16_N8 base (actcache 态, production) ; C = base + -DGDN_BR_A_PRECROUTON (skip A fold_quant+pack).
# ⚠️ T is WRONG under C (standalone feeds no correct cv) -> this is the wall-delta UPPER BOUND ONLY.
# Same SHIP+FUSEDEP+ACTCACHE base, only -DGDN_BR_A_PRECROUTON toggles. 3 windows, reps2-8 median,
# A C interleaved within each window (same thermal window). metric = stats[0] 32-head graph-wall.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"; BM="$ROOT/example/gdn_native/baremetal"
SHIP="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL -DGDN_BR_SBOOST -DGDN_BR_DIAG_I16 -DGDN_BR_REQ_FUSE -DGDN_BR_FBOOST"
BASE="$SHIP -DGDN_BR_W16 -DGDN_BR_W16_N8 -DGDN_BR_W16_N8_FUSEDEP -DGDN_BR_W16_N8_ACTCACHE"
SCALES="2.770166930875267e-05 6.103701895199438e-05"
WINDOWS="${WINDOWS:-3}"; REPS="${REPS:-8}"; THREADS="${THREADS:-4}"
source "$ROOT/scripts/dssh.sh"; dssh_open

build() { ( cd "$BM" && EXTRA_DEFS="$1" bash build.sh ) >/tmp/wup_$2.log 2>&1 || { echo "BUILD FAIL $2"; tail -20 /tmp/wup_$2.log; exit 1; }
  mkdir -p /tmp/wup_$2; cp "$BM/build/libgdnbm_skel.so" "$BM/build/gdnbm" /tmp/wup_$2/; }
echo "=== build A (actcache base)      ==="; build "$BASE" A
echo "=== build C (base + PRECROUTON)  ==="; build "$BASE -DGDN_BR_A_PRECROUTON" C

for V in A C; do W="\$HOME/wup_$V"; dssh "mkdir -p $W"
  dssh "cat > $W/libgdnbm_skel.so" < /tmp/wup_$V/libgdnbm_skel.so
  dssh "cat > $W/gdnbm" < /tmp/wup_$V/gdnbm; dssh "chmod +x $W/gdnbm"
  dssh "cat > $W/A_u16_h32.raw" < "$BM/A_u16_h32.raw"
done

run_one() {  # $1=V  -> "med min max n"
  local V="$1" W="\$HOME/wup_$1" out
  out=$(dssh "cd $W && GDNBM_REPS=$REPS LD_LIBRARY_PATH=\$PWD:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH=\"\$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" ./gdnbm $THREADS A_u16_h32.raw T_${V}.raw 32 256 32768 32768 $SCALES 2>&1")
  local walls; walls=$(echo "$out" | grep -oE "rep=[0-9]+/[0-9]+ +wall=[0-9]+" | sed -E 's/rep=([0-9]+).*wall=([0-9]+)/\1 \2/')
  python3 - <<PY
rows=[]
for ln in '''$walls'''.strip().splitlines():
    r,w=ln.split(); rows.append((int(r),int(w)))
st=sorted(w for r,w in rows if r>=2)
if not st: print("0 0 0 0")
else:
    n=len(st); med=st[n//2] if n%2 else (st[n//2-1]+st[n//2])//2
    print(f"{med} {st[0]} {st[-1]} {n}")
PY
}

echo ""; echo "### ACAC x$WINDOWS, REPS=$REPS, THREADS=$THREADS (metric stats[0] reps2-$REPS median) ###"
echo "### ⚠️ UPPER BOUND: C drops A act path, T is WRONG -> wall口径 only ###"
declare -a AW CW
for win in $(seq 1 $WINDOWS); do
  ra=$(run_one A); rc=$(run_one C)
  am=$(echo $ra|cut -d' ' -f1); cm=$(echo $rc|cut -d' ' -f1)
  AW+=($am); CW+=($cm)
  if [ "$am" -gt 0 ] && [ "$cm" -gt 0 ]; then d=$(python3 -c "print(f'{($cm-$am)/$am*100:+.2f}')"); else d=NA; fi
  echo "win$win: A=$ra | C=$rc | dWall=${d}%"
done
echo ""
python3 - <<PY
A=[x for x in [${AW[@]/%/,}] if x>0]; C=[x for x in [${CW[@]/%/,}] if x>0]
def med(v): v=sorted(v); n=len(v); return v[n//2] if n%2 else (v[n//2-1]+v[n//2])//2
if A and C:
    am,cm=med(A),med(C)
    print(f"3-WINDOW MERGED: A_med={am} C_med={cm}  dWall(UPPER)={(cm-am)/am*100:+.2f}%  (C=PRECROUTON vs A=actcache base)")
    print(f"  A wins={A}  C wins={C}")
PY
