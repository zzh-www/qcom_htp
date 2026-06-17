#!/usr/bin/env bash
# EXP2: ACVRES head-load 48KB/head copy 真实 wall 占用 (gated GDN_BR_HEADLOAD_PROBE, 默认 OFF)。
# 量 Σ(全 32-head copy span, PCYCLE) + per-head + 折到 wall。回答 -8.2% 归因: 几万拷贝指令 vs prep 阻塞。
# 4 producers 并行 -> wall 贡献 ≈ Σ/4 (8 heads/producer 串行在该 producer 内)。
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"; BM="$ROOT/example/gdn_native/baremetal"
SHIP="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL -DGDN_BR_SBOOST -DGDN_BR_DIAG_I16 -DGDN_BR_REQ_FUSE -DGDN_BR_FBOOST"
BASE="$SHIP -DGDN_BR_W16 -DGDN_BR_W16_N8 -DGDN_BR_W16_N8_FUSEDEP -DGDN_BR_W16_N8_ACTCACHE"
SCALES="2.770166930875267e-05 6.103701895199438e-05"
H=32; C=256; THREADS="${THREADS:-4}"; REPS="${REPS:-8}"
OUT="${OUT:-$ROOT/Agent/current/w16n8_headload_probe.txt}"
source "$ROOT/scripts/dssh.sh"; dssh_open

build() { ( cd "$BM" && EXTRA_DEFS="$1" bash build.sh ) >/tmp/hl_$2.log 2>&1 || { echo "BUILD FAIL $2"; tail -25 /tmp/hl_$2.log; exit 1; }
  mkdir -p /tmp/hl_$2; cp "$BM/build/libgdnbm_skel.so" "$BM/build/gdnbm" /tmp/hl_$2/; }
echo "=== build DUMP ==="; build "$BASE -DGDN_BR_W16_N8_ACVRES_DUMP" DUMP
echo "=== build PROBE (ACVRES + HEADLOAD_PROBE) ==="; build "$BASE -DGDN_BR_W16_N8_ACVRES -DGDN_BR_HEADLOAD_PROBE" PROBE

deploy() { local V="$1" W="\$HOME/hl_$1"; dssh "mkdir -p $W"
  dssh "cat > $W/libgdnbm_skel.so" < /tmp/hl_$V/libgdnbm_skel.so
  dssh "cat > $W/gdnbm" < /tmp/hl_$V/gdnbm; dssh "chmod +x $W/gdnbm"; }
for V in DUMP PROBE; do deploy "$V"; dssh "cat > \$HOME/hl_$V/A_u16_h32.raw" < "$BM/A_u16_h32.raw"; done

run_dev() { local V="$1" Ain="$2" Tout="$3" rp="${4:-1}" W="\$HOME/hl_$1"
  dssh "cd $W && GDNBM_REPS=$rp LD_LIBRARY_PATH=\$PWD:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH=\"\$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" ./gdnbm $THREADS $Ain $Tout $H $C 32768 32768 $SCALES 2>&1"; }

# build extended A for ACVRES
run_dev DUMP A_u16_h32.raw T_dump.raw 1 >/tmp/hl_dump.log 2>&1
dssh "cat \$HOME/hl_DUMP/T_dump.raw" > /tmp/T_dump.raw
python3 - <<'PY'
import numpy as np
H=32; HEAD_U16=256*256
dump=np.fromfile("/tmp/T_dump.raw","<u2"); nat=np.fromfile("/home/zzh/work/qcom_htp/example/gdn_native/baremetal/A_u16_h32.raw","<u2")
cv=np.zeros((H,6*4096),"<u2")
for h in range(H): cv[h]=dump[h*HEAD_U16:h*HEAD_U16+6*4096]
np.concatenate([nat,cv.ravel()]).tofile("/tmp/A_ares.raw")
PY
dssh "cat > \$HOME/hl_PROBE/A_ares.raw" < /tmp/A_ares.raw

echo "### EXP2 HEADLOAD PROBE (REPS=$REPS) ###" | tee "$OUT"
for r in 1 2 3; do
  echo "--- run $r ---" | tee -a "$OUT"
  run_dev PROBE A_ares.raw T_p.raw $REPS 2>&1 | grep -E "HEADLOAD:|wall=" | tee -a "$OUT"
done
echo "" | tee -a "$OUT"
echo "NOTE: total_cyc = Σ全 32-head copy span; 4 producers 并行 (8 head/producer 串行) -> wall 贡献 ≈ total/4。" | tee -a "$OUT"
echo "(written to $OUT)"
