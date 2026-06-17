#!/usr/bin/env bash
# EXP1: ARES + pure-HMX 4-point P-sweep (P=1,2,3,4). 钉死 contention(c) vs 固定串行(b).
# 每点抓 wall(stats[0]) / feed_Σ(stats[4]) / cbusy(stats[3]) / lmax(stats[8]) (ARES path).
# pure-HMX path stats 语义不同(stats[4]=spin_Σ, stats[11]=lmax) — 记 wall + raw stats, 拟合只用 wall。
# 口径: 32-head TOTAL wall, VTCM-only, reps2-N median (绝不取 min). authoritative scales, A_u16_h32.raw.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"; BM="$ROOT/example/gdn_native/baremetal"
SHIP="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL -DGDN_BR_SBOOST -DGDN_BR_DIAG_I16 -DGDN_BR_REQ_FUSE -DGDN_BR_FBOOST"
ARES_BASE="$SHIP -DGDN_BR_W16 -DGDN_BR_W16_N8 -DGDN_BR_W16_N8_FUSEDEP -DGDN_BR_W16_N8_ACTCACHE -DGDN_BR_W16_N8_ARES"
PURE="-DGDNBM_GDN_PURE_SOLVE"
SCALES="2.770166930875267e-05 6.103701895199438e-05"
H=32; C=256
REPS="${REPS:-8}"
OUT="${OUT:-$ROOT/Agent/current/perf_ares_bserial_pin.txt}"
source "$ROOT/scripts/dssh.sh"; dssh_open

build() { ( cd "$BM" && EXTRA_DEFS="$1" bash build.sh ) >/tmp/ps_$2.log 2>&1 || { echo "BUILD FAIL $2"; tail -25 /tmp/ps_$2.log; exit 1; }
  mkdir -p /tmp/ps_$2; cp "$BM/build/libgdnbm_skel.so" "$BM/build/gdnbm" /tmp/ps_$2/; }
deploy() { local V="$1" W="\$HOME/ps_$1"; dssh "mkdir -p $W"
  dssh "cat > $W/libgdnbm_skel.so" < /tmp/ps_$V/libgdnbm_skel.so
  dssh "cat > $W/gdnbm" < /tmp/ps_$V/gdnbm; dssh "chmod +x $W/gdnbm"; }

# run_dev VARIANT P Ain Tout reps  -> stdout
run_dev() { local V="$1" P="$2" Ain="$3" Tout="$4" rp="${5:-1}" W="\$HOME/ps_$1"
  dssh "cd $W && GDNBM_REPS=$rp LD_LIBRARY_PATH=\$PWD:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH=\"\$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" ./gdnbm $P $Ain $Tout $H $C 32768 32768 $SCALES 2>&1"; }

parse_med() { printf '%s\n' "$1" | python3 -c '
import sys,re
rows=[(int(m.group(1)),int(m.group(2))) for ln in sys.stdin for m in [re.search(r"rep=(\d+)/\d+\s+wall=(\d+)",ln)] if m]
st=sorted(w for r,w in rows if r>=2)
print((st[len(st)//2] if len(st)%2 else (st[len(st)//2-1]+st[len(st)//2])//2) if st else 0)'; }
# spread / raw stats[0..11] of LAST rep (steady)
last_raw() { printf '%s\n' "$1" | grep 'raw stats\[0\.\.11\]' | tail -1; }
clk() { printf '%s\n' "$1" | grep -i 'clock self-check' | tail -1; }

echo "" | tee -a "$OUT"
echo "================================================================================" | tee -a "$OUT"
echo "EXP1: P-sweep (ARES + pure-HMX, P=1..4)  REPS=$REPS  $(date '+%F %T')" | tee -a "$OUT"
echo "================================================================================" | tee -a "$OUT"

echo "=== build ARES ===" ; build "$ARES_BASE" ARES ; deploy ARES
echo "=== build PURE ===" ; build "$PURE" PURE ; deploy PURE
dssh "cat > \$HOME/ps_ARES/A_u16_h32.raw" < "$BM/A_u16_h32.raw"
# DUMP run to build extended A_ares.raw (natural||cv-block), reuse 8win recipe
build "$ARES_BASE -DGDN_BR_W16_N8_ACVRES_DUMP" DUMP ; deploy DUMP
dssh "cat > \$HOME/ps_DUMP/A_u16_h32.raw" < "$BM/A_u16_h32.raw"
run_dev DUMP 1 A_u16_h32.raw T_dump.raw 1 >/tmp/ps_dump.log 2>&1
dssh "cat \$HOME/ps_DUMP/T_dump.raw" > /tmp/ps_T_dump.raw
python3 - <<'PY'
import numpy as np
H=32; HEAD_U16=256*256
dump=np.fromfile("/tmp/ps_T_dump.raw","<u2")
nat =np.fromfile("/home/zzh/work/qcom_htp/example/gdn_native/baremetal/A_u16_h32.raw","<u2")
assert dump.size==H*HEAD_U16, dump.size
cv=np.zeros((H,6*4096),"<u2")
for h in range(H): cv[h]=dump[h*HEAD_U16:h*HEAD_U16+6*4096]
np.concatenate([nat,cv.ravel()]).tofile("/tmp/ps_A_ares.raw")
PY
dssh "cat > \$HOME/ps_ARES/A_ares.raw" < /tmp/ps_A_ares.raw

for P in 1 2 3 4; do
  ao="$(run_dev ARES $P A_ares.raw    T_a.raw $REPS)"
  po="$(run_dev PURE $P A_u16_h32.raw T_p.raw $REPS)"
  am=$(parse_med "$ao"); pm=$(parse_med "$po")
  echo "" | tee -a "$OUT"
  echo "--- P=$P (reps2-$REPS median) ---" | tee -a "$OUT"
  echo "ARES wall=$am  | $(clk "$ao")" | tee -a "$OUT"
  echo "ARES $(last_raw "$ao")" | tee -a "$OUT"
  echo "PURE wall=$pm  | $(clk "$po")" | tee -a "$OUT"
  echo "PURE $(last_raw "$po")" | tee -a "$OUT"
done

echo "" | tee -a "$OUT"
echo "(EXP1 P-sweep done; fit computed in followup python)" | tee -a "$OUT"
