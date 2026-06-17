#!/usr/bin/env bash
# W16_N8 ACTCV (path b真兑现: fused fold+quant+crouton, no nat round-trip) bit-exact + wall gate.
# A = W16_N8 actcache base (oracle) ; C = base + -DGDN_BR_W16_N8_ACTCV.
# bit-exact: 32-head 4MB T md5 逐字节相同 (fold/quant/crouton 确定性逐元素, 同 LUT/scale => 必 bit-exact)。
# wall: ACAC reps2-8 median, A C 同热窗背靠背配对。 metric = stats[0] 32-head graph-wall。
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"; BM="$ROOT/example/gdn_native/baremetal"
SHIP="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL -DGDN_BR_SBOOST -DGDN_BR_DIAG_I16 -DGDN_BR_REQ_FUSE -DGDN_BR_FBOOST"
BASE="$SHIP -DGDN_BR_W16 -DGDN_BR_W16_N8 -DGDN_BR_W16_N8_FUSEDEP -DGDN_BR_W16_N8_ACTCACHE"
SCALES="2.770166930875267e-05 6.103701895199438e-05"
WINDOWS="${WINDOWS:-3}"; REPS="${REPS:-8}"; THREADS="${THREADS:-4}"
source "$ROOT/scripts/dssh.sh"; dssh_open

build() { ( cd "$BM" && EXTRA_DEFS="$1" bash build.sh ) >/tmp/cv_$2.log 2>&1 || { echo "BUILD FAIL $2"; tail -20 /tmp/cv_$2.log; exit 1; }
  mkdir -p /tmp/cv_$2; cp "$BM/build/libgdnbm_skel.so" "$BM/build/gdnbm" /tmp/cv_$2/; }
echo "=== build A (actcache base oracle) ==="; build "$BASE" A
echo "=== build C (base + ACTCV)         ==="; build "$BASE -DGDN_BR_W16_N8_ACTCV" C

for V in A C; do W="\$HOME/cv_$V"; dssh "mkdir -p $W"
  dssh "cat > $W/libgdnbm_skel.so" < /tmp/cv_$V/libgdnbm_skel.so
  dssh "cat > $W/gdnbm" < /tmp/cv_$V/gdnbm; dssh "chmod +x $W/gdnbm"
  dssh "cat > $W/A_u16_h32.raw" < "$BM/A_u16_h32.raw"
done

run_t() {  # produce T_$1.raw + md5
  local V="$1" W="\$HOME/cv_$1"
  dssh "cd $W && GDNBM_REPS=2 LD_LIBRARY_PATH=\$PWD:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH=\"\$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" ./gdnbm $THREADS A_u16_h32.raw T_${V}.raw 32 256 32768 32768 $SCALES >/dev/null 2>&1; md5sum T_${V}.raw; ls -l T_${V}.raw | awk '{print \$5}'"
}
run_wall() {  # $1=V  -> median
  local V="$1" W="\$HOME/cv_$1" out
  out=$(dssh "cd $W && GDNBM_REPS=$REPS LD_LIBRARY_PATH=\$PWD:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH=\"\$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" ./gdnbm $THREADS A_u16_h32.raw T_${V}.raw 32 256 32768 32768 $SCALES 2>&1")
  local walls; walls=$(echo "$out" | grep -oE "rep=[0-9]+/[0-9]+ +wall=[0-9]+" | sed -E 's/rep=([0-9]+).*wall=([0-9]+)/\1 \2/')
  python3 -c "
rows=[l.split() for l in '''$walls'''.strip().splitlines()]
st=sorted(int(w) for r,w in rows if int(r)>=2)
print(st[len(st)//2] if len(st)%2 else (st[len(st)//2-1]+st[len(st)//2])//2 if st else 0)"
}

echo ""; echo "### BIT-EXACT (32-head 4MB T md5) ###"
MA=$(run_t A); MC=$(run_t C)
echo "A: $MA"; echo "C: $MC"
ha=$(echo "$MA"|head -1|awk '{print $1}'); hc=$(echo "$MC"|head -1|awk '{print $1}')
if [ "$ha" = "$hc" ]; then echo "MD5 MATCH ($ha) -> T 逐字节相同 BIT-EXACT"; else echo "MD5 MISMATCH A=$ha C=$hc -> 实现错, 先修"; fi

echo ""; echo "### WALL ACAC x$WINDOWS REPS=$REPS THREADS=$THREADS (stats[0] reps2-$REPS median) ###"
declare -a AW CW
for win in $(seq 1 $WINDOWS); do
  am=$(run_wall A); cm=$(run_wall C); AW+=($am); CW+=($cm)
  d=$(python3 -c "print(f'{($cm-$am)/$am*100:+.2f}')" 2>/dev/null || echo NA)
  echo "win$win: A=$am C=$cm dWall=${d}%"
done
python3 -c "
A=sorted([${AW[@]/%/,}]); C=sorted([${CW[@]/%/,}])
def m(v): n=len(v); return v[n//2] if n%2 else (v[n//2-1]+v[n//2])//2
am,cm=m(A),m(C); print(f'MERGED: A={am} C={cm} dWall={(cm-am)/am*100:+.2f}% (C=ACTCV vs A=actcache base)')"
