#!/usr/bin/env bash
# EXP3: ARES alias 62K 残留可消否 — FASTPTR (prep 一次性写 6 指针, 运行时纯 indexed load) vs ARES (per-call key2slot+offset)。
# 同窗 ACAC wall-delta + bit-exact 守门 (T md5 不变)。回答: 62K 是不可消结构开销 还是 可优化 per-call 分支。
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"; BM="$ROOT/example/gdn_native/baremetal"
SHIP="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL -DGDN_BR_SBOOST -DGDN_BR_DIAG_I16 -DGDN_BR_REQ_FUSE -DGDN_BR_FBOOST"
BASE="$SHIP -DGDN_BR_W16 -DGDN_BR_W16_N8 -DGDN_BR_W16_N8_FUSEDEP -DGDN_BR_W16_N8_ACTCACHE"
SCALES="2.770166930875267e-05 6.103701895199438e-05"
H=32; C=256; THREADS="${THREADS:-4}"; REPS="${REPS:-8}"; WINDOWS="${WINDOWS:-6}"
OUT="${OUT:-$ROOT/Agent/current/w16n8_alias62k.txt}"
source "$ROOT/scripts/dssh.sh"; dssh_open

build() { ( cd "$BM" && EXTRA_DEFS="$1" bash build.sh ) >/tmp/fp_$2.log 2>&1 || { echo "BUILD FAIL $2"; tail -25 /tmp/fp_$2.log; exit 1; }
  mkdir -p /tmp/fp_$2; cp "$BM/build/libgdnbm_skel.so" "$BM/build/gdnbm" /tmp/fp_$2/; }
echo "=== build DUMP    ==="; build "$BASE -DGDN_BR_W16_N8_ACVRES_DUMP" DUMP
echo "=== build ARES    ==="; build "$BASE -DGDN_BR_W16_N8_ARES" ARES
echo "=== build FASTPTR ==="; build "$BASE -DGDN_BR_W16_N8_ARES -DGDN_BR_W16_N8_ARES_FASTPTR" FASTPTR

deploy() { local V="$1" W="\$HOME/fp_$1"; dssh "mkdir -p $W"
  dssh "cat > $W/libgdnbm_skel.so" < /tmp/fp_$V/libgdnbm_skel.so
  dssh "cat > $W/gdnbm" < /tmp/fp_$V/gdnbm; dssh "chmod +x $W/gdnbm"; }
for V in DUMP ARES FASTPTR; do deploy "$V"; dssh "cat > \$HOME/fp_$V/A_u16_h32.raw" < "$BM/A_u16_h32.raw"; done

run_dev() { local V="$1" Ain="$2" Tout="$3" rp="${4:-1}" W="\$HOME/fp_$1"
  dssh "cd $W && GDNBM_REPS=$rp LD_LIBRARY_PATH=\$PWD:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH=\"\$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" ./gdnbm $THREADS $Ain $Tout $H $C 32768 32768 $SCALES 2>&1"; }

run_dev DUMP A_u16_h32.raw T_dump.raw 1 >/tmp/fp_dump.log 2>&1
dssh "cat \$HOME/fp_DUMP/T_dump.raw" > /tmp/T_dump.raw
python3 - <<'PY'
import numpy as np
H=32; HEAD_U16=256*256
dump=np.fromfile("/tmp/T_dump.raw","<u2"); nat=np.fromfile("/home/zzh/work/qcom_htp/example/gdn_native/baremetal/A_u16_h32.raw","<u2")
cv=np.zeros((H,6*4096),"<u2")
for h in range(H): cv[h]=dump[h*HEAD_U16:h*HEAD_U16+6*4096]
np.concatenate([nat,cv.ravel()]).tofile("/tmp/A_ares.raw")
PY
for V in ARES FASTPTR; do dssh "cat > \$HOME/fp_$V/A_ares.raw" < /tmp/A_ares.raw; done

echo "### EXP3 alias62k FASTPTR vs ARES ###" | tee "$OUT"
# bit-exact gate: FASTPTR T md5 == ARES T md5
run_dev ARES    A_ares.raw T_ares.raw 1 >/tmp/fp_a.log 2>&1
run_dev FASTPTR A_ares.raw T_fp.raw   1 >/tmp/fp_f.log 2>&1
dssh "cat \$HOME/fp_ARES/T_ares.raw"  > /tmp/T_ares.raw
dssh "cat \$HOME/fp_FASTPTR/T_fp.raw" > /tmp/T_fp.raw
echo "bit-exact (FASTPTR T == ARES T):" | tee -a "$OUT"
md5sum /tmp/T_ares.raw /tmp/T_fp.raw | tee -a "$OUT"
cmp -s /tmp/T_ares.raw /tmp/T_fp.raw && echo "  cmp: IDENTICAL (PASS)" | tee -a "$OUT" || echo "  cmp: DIFFER (FAIL)" | tee -a "$OUT"

parse_med() { printf '%s\n' "$1" | python3 -c '
import sys,re
rows=[(int(m.group(1)),int(m.group(2))) for ln in sys.stdin for m in [re.search(r"rep=(\d+)/\d+\s+wall=(\d+)",ln)] if m]
st=sorted(w for r,w in rows if r>=2)
print((st[len(st)//2] if len(st)%2 else (st[len(st)//2-1]+st[len(st)//2])//2) if st else 0)'; }

declare -a AW FW
echo "### ACAC x$WINDOWS REPS=$REPS ###" | tee -a "$OUT"
for win in $(seq 1 $WINDOWS); do
  am=$(parse_med "$(run_dev ARES    A_ares.raw T_wa.raw $REPS)")
  fm=$(parse_med "$(run_dev FASTPTR A_ares.raw T_wf.raw $REPS)")
  AW+=($am); FW+=($fm)
  d=$(python3 -c "print(f'{($fm-$am)/$am*100:+.2f}')" 2>/dev/null || echo NA)
  echo "win$win: ARES=$am FASTPTR=$fm dWall=${d}%" | tee -a "$OUT"
done
python3 - "${AW[@]/%/,}" "|" "${FW[@]/%/,}" <<'PY' | tee -a "$OUT"
import sys
g=[[],[]]; gi=0
for tok in sys.argv[1:]:
    if tok=="|": gi+=1; continue
    for x in tok.split(","):
        if x: g[gi].append(int(x))
A,F=g
def med(v): v=sorted(v); n=len(v); return v[n//2] if n%2 else (v[n//2-1]+v[n//2])//2
def rej(v): m=med(v); return [x for x in v if x<=1.15*m]
A2,F2=rej(A),rej(F); am,fm=med(A2),med(F2)
print(f"\n  ARES med={am}  FASTPTR med={fm}  dWall={(fm-am)/am*100:+.2f}%")
ds=sorted((f-a)/a*100 for a,f in zip(A,F))
n=len(ds); m=ds[n//2] if n%2 else (ds[n//2-1]+ds[n//2])/2
print(f"  same-window paired Δ median={m:+.2f}%  range=[{ds[0]:+.2f},{ds[-1]:+.2f}]")
PY
echo "(written to $OUT)"
