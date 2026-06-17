#!/usr/bin/env bash
# EXP3: DIAG fwd-subst cap-test 重验 (当前规模, DIAG Σ 已 2.5M, 旧 142.9K-era 测 stale).
# base = ARES; cap = ARES + -DGDN_BR_FWD_ROWS=4 (内层砍到 4/64 = 1/16, T garbage 但 mxmem timing 数据无关).
# 同热窗 ACAC 配对, P=4. cap>=5% => 精确对角进了 b_serial(真串行); cap~0 => SMT 藏住.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"; BM="$ROOT/example/gdn_native/baremetal"
SHIP="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL -DGDN_BR_SBOOST -DGDN_BR_DIAG_I16 -DGDN_BR_REQ_FUSE -DGDN_BR_FBOOST"
ARES_BASE="$SHIP -DGDN_BR_W16 -DGDN_BR_W16_N8 -DGDN_BR_W16_N8_FUSEDEP -DGDN_BR_W16_N8_ACTCACHE -DGDN_BR_W16_N8_ARES"
SCALES="2.770166930875267e-05 6.103701895199438e-05"
H=32; C=256; P="${P:-4}"
REPS="${REPS:-8}"; WINDOWS="${WINDOWS:-5}"
OUT="${OUT:-$ROOT/Agent/current/perf_ares_bserial_pin.txt}"
source "$ROOT/scripts/dssh.sh"; dssh_open

build() { ( cd "$BM" && EXTRA_DEFS="$1" bash build.sh ) >/tmp/fc_$2.log 2>&1 || { echo "BUILD FAIL $2"; tail -25 /tmp/fc_$2.log; exit 1; }
  mkdir -p /tmp/fc_$2; cp "$BM/build/libgdnbm_skel.so" "$BM/build/gdnbm" /tmp/fc_$2/; }
deploy() { local V="$1" W="\$HOME/fc_$1"; dssh "mkdir -p $W"
  dssh "cat > $W/libgdnbm_skel.so" < /tmp/fc_$V/libgdnbm_skel.so
  dssh "cat > $W/gdnbm" < /tmp/fc_$V/gdnbm; dssh "chmod +x $W/gdnbm"; }
run_dev() { local V="$1" Ain="$2" Tout="$3" rp="${4:-1}" W="\$HOME/fc_$1"
  dssh "cd $W && GDNBM_REPS=$rp LD_LIBRARY_PATH=\$PWD:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH=\"\$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" ./gdnbm $P $Ain $Tout $H $C 32768 32768 $SCALES 2>&1"; }
parse_med() { printf '%s\n' "$1" | python3 -c '
import sys,re
rows=[(int(m.group(1)),int(m.group(2))) for ln in sys.stdin for m in [re.search(r"rep=(\d+)/\d+\s+wall=(\d+)",ln)] if m]
st=sorted(w for r,w in rows if r>=2)
print((st[len(st)//2] if len(st)%2 else (st[len(st)//2-1]+st[len(st)//2])//2) if st else 0)'; }
feedsig() { printf '%s\n' "$1" | grep 'raw stats\[0\.\.11\]' | tail -1 | awk '{print $7}'; }

echo "" | tee -a "$OUT"
echo "================================================================================" | tee -a "$OUT"
echo "EXP3: DIAG fwd-subst cap-test (-DGDN_BR_FWD_ROWS=4 = 1/16)  P=$P REPS=$REPS WIN=$WINDOWS  $(date '+%F %T')" | tee -a "$OUT"
echo "================================================================================" | tee -a "$OUT"

build "$ARES_BASE" BASE ; deploy BASE
build "$ARES_BASE -DGDN_BR_FWD_ROWS=4" CAP ; deploy CAP
build "$ARES_BASE -DGDN_BR_W16_N8_ACVRES_DUMP" DUMP ; deploy DUMP
dssh "cat > \$HOME/fc_DUMP/A_u16_h32.raw" < "$BM/A_u16_h32.raw"
run_dev DUMP A_u16_h32.raw T_dump.raw 1 >/tmp/fc_dump.log 2>&1
dssh "cat \$HOME/fc_DUMP/T_dump.raw" > /tmp/fc_T_dump.raw
python3 - <<'PY'
import numpy as np
H=32; HEAD_U16=256*256
dump=np.fromfile("/tmp/fc_T_dump.raw","<u2")
nat =np.fromfile("/home/zzh/work/qcom_htp/example/gdn_native/baremetal/A_u16_h32.raw","<u2")
cv=np.zeros((H,6*4096),"<u2")
for h in range(H): cv[h]=dump[h*HEAD_U16:h*HEAD_U16+6*4096]
np.concatenate([nat,cv.ravel()]).tofile("/tmp/fc_A_ares.raw")
PY
for V in BASE CAP; do dssh "cat > \$HOME/fc_$V/A_ares.raw" < /tmp/fc_A_ares.raw; done

declare -a BW CW BF CF
for win in $(seq 1 $WINDOWS); do
  bo="$(run_dev BASE A_ares.raw T_b.raw $REPS)"
  co="$(run_dev CAP  A_ares.raw T_c.raw $REPS)"
  bm=$(parse_med "$bo"); cm=$(parse_med "$co")
  bf=$(feedsig "$bo"); cf=$(feedsig "$co")
  BW+=($bm); CW+=($cm); BF+=($bf); CF+=($cf)
  d=$(python3 -c "print(f'{($cm-$bm)/$bm*100:+.2f}')")
  echo "win$win: BASE wall=$bm feed_Σ=$bf | CAP wall=$cm feed_Σ=$cf | Δwall=$d%" | tee -a "$OUT"
done

python3 - "$OUT" "${BW[@]}" "|" "${CW[@]}" "|" "${BF[@]}" "|" "${CF[@]}" <<'PY' | tee -a "$OUT"
import sys
rest=sys.argv[2:]; g=[[],[],[],[]]; gi=0
for t in rest:
    if t=="|": gi+=1; continue
    g[gi].append(int(t))
BW,CW,BF,CF=g
def med(v):
    v=sorted(v); n=len(v); return v[n//2] if n%2 else (v[n//2-1]+v[n//2])//2
pds=sorted((c-b)/b*100 for b,c in zip(BW,CW)); n=len(pds)
pm=pds[n//2] if n%2 else (pds[n//2-1]+pds[n//2])/2
print("\n=== EXP3 RESULT (fwd-subst cap to 1/16, P=4) ===")
print(f"  BASE wall median={med(BW)}  CAP wall median={med(CW)}")
print(f"  Δwall same-window paired: median={pm:+.2f}%  range=[{pds[0]:+.2f},{pds[-1]:+.2f}]  n={n}")
print(f"  BASE feed_Σ median={med(BF)/1e6:.3f}M  CAP feed_Σ median={med(CF)/1e6:.3f}M  Δfeed={(med(CF)-med(BF))/med(BF)*100:+.1f}%")
verdict = "精确对角进了 b_serial (真串行, 出路=换 Taylor 丢精度)" if abs(pm)>=5 else "SMT 藏住 (b_serial 主要是 HW-pack + gather-争用, 非对角)"
print(f"  => |Δwall|={abs(pm):.1f}%  =>  {verdict}")
PY
echo "(EXP3 done)" | tee -a "$OUT"
