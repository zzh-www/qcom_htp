#!/usr/bin/env bash
# EXP1: ARES / PRECROUTON / ACVRES — ≥8-window同窗 ACAC, IQR + 撞窗离群剔除, 钉死地板 + 收 spread。
#
# 三档严格同窗轮转 (每窗内背靠背 ARES->PRECR->ACVRES, reps2-N median)。
# 离群剔除标准: 某档某窗 median > 1.15× 该档全窗中位 -> 剔除该窗该档 (撞窗 artifact)。
# 报: 各档 median + IQR(Q1/Q3) + 同窗配对 (ARES-vs-PRECR, ARES-vs-ACVRES)。
#
# build base = SHIP + W16 + W16_N8 + FUSEDEP + ACTCACHE (生产高精度档)。
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"; BM="$ROOT/example/gdn_native/baremetal"
SHIP="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL -DGDN_BR_SBOOST -DGDN_BR_DIAG_I16 -DGDN_BR_REQ_FUSE -DGDN_BR_FBOOST"
BASE="$SHIP -DGDN_BR_W16 -DGDN_BR_W16_N8 -DGDN_BR_W16_N8_FUSEDEP -DGDN_BR_W16_N8_ACTCACHE"
SCALES="2.770166930875267e-05 6.103701895199438e-05"
H=32; C=256
WINDOWS="${WINDOWS:-9}"; REPS="${REPS:-8}"; THREADS="${THREADS:-4}"
OUT="${OUT:-$ROOT/Agent/current/w16n8_ares_8win.txt}"
source "$ROOT/scripts/dssh.sh"; dssh_open

build() { ( cd "$BM" && EXTRA_DEFS="$1" bash build.sh ) >/tmp/a8_$2.log 2>&1 || { echo "BUILD FAIL $2"; tail -25 /tmp/a8_$2.log; exit 1; }
  mkdir -p /tmp/a8_$2; cp "$BM/build/libgdnbm_skel.so" "$BM/build/gdnbm" /tmp/a8_$2/; }
echo "=== build DUMP  ==="; build "$BASE -DGDN_BR_W16_N8_ACVRES_DUMP" DUMP
echo "=== build BASE  ==="; build "$BASE" BASE
echo "=== build ACVRES==="; build "$BASE -DGDN_BR_W16_N8_ACVRES" ACVRES
echo "=== build ARES  ==="; build "$BASE -DGDN_BR_W16_N8_ARES" ARES
echo "=== build PRECR ==="; build "$BASE -DGDN_BR_A_PRECROUTON" PRECR

deploy() { local V="$1" W="\$HOME/a8_$1"; dssh "mkdir -p $W"
  dssh "cat > $W/libgdnbm_skel.so" < /tmp/a8_$V/libgdnbm_skel.so
  dssh "cat > $W/gdnbm" < /tmp/a8_$V/gdnbm; dssh "chmod +x $W/gdnbm"; }
for V in DUMP BASE ACVRES ARES PRECR; do deploy "$V"; done
for V in DUMP BASE ACVRES ARES PRECR; do dssh "cat > \$HOME/a8_$V/A_u16_h32.raw" < "$BM/A_u16_h32.raw"; done

run_dev() { local V="$1" Ain="$2" Tout="$3" rp="${4:-1}" W="\$HOME/a8_$1"
  dssh "cd $W && GDNBM_REPS=$rp LD_LIBRARY_PATH=\$PWD:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH=\"\$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" ./gdnbm $THREADS $Ain $Tout $H $C 32768 32768 $SCALES 2>&1"; }

# build extended A (natural || cv-block) for ARES/ACVRES alias
run_dev DUMP A_u16_h32.raw T_dump.raw 1 >/tmp/a8_dump.log 2>&1
dssh "cat \$HOME/a8_DUMP/T_dump.raw" > /tmp/T_dump.raw
python3 - <<'PY'
import numpy as np
H=32; HEAD_U16=256*256
dump=np.fromfile("/tmp/T_dump.raw","<u2")
nat =np.fromfile("/home/zzh/work/qcom_htp/example/gdn_native/baremetal/A_u16_h32.raw","<u2")
assert dump.size==H*HEAD_U16, dump.size
cv=np.zeros((H,6*4096),"<u2")
for h in range(H): cv[h]=dump[h*HEAD_U16:h*HEAD_U16+6*4096]
np.concatenate([nat,cv.ravel()]).tofile("/tmp/A_ares.raw")
PY
for V in ACVRES ARES; do dssh "cat > \$HOME/a8_$V/A_ares.raw" < /tmp/A_ares.raw; done

# bit-exact gate (ARES T == BASE T)
run_dev BASE A_u16_h32.raw T_base.raw 1 >/tmp/a8_base.log 2>&1
run_dev ARES A_ares.raw    T_ares.raw 1 >/tmp/a8_ares.log 2>&1
dssh "cat \$HOME/a8_BASE/T_base.raw" > /tmp/T_base.raw
dssh "cat \$HOME/a8_ARES/T_ares.raw" > /tmp/T_ares.raw
BX="FAIL"; cmp -s /tmp/T_base.raw /tmp/T_ares.raw && BX="PASS(IDENTICAL)"

parse_med() { printf '%s\n' "$1" | python3 -c '
import sys,re
rows=[(int(m.group(1)),int(m.group(2))) for ln in sys.stdin for m in [re.search(r"rep=(\d+)/\d+\s+wall=(\d+)",ln)] if m]
st=sorted(w for r,w in rows if r>=2)
print((st[len(st)//2] if len(st)%2 else (st[len(st)//2-1]+st[len(st)//2])//2) if st else 0)'; }

declare -a AW PW VW
echo "### EXP1 8win ACAC (ARES/PRECR/ACVRES same-window) REPS=$REPS THREADS=$THREADS ###" | tee "$OUT"
echo "bit-exact ARES-vs-BASE: $BX" | tee -a "$OUT"
md5sum /tmp/T_base.raw /tmp/T_ares.raw | tee -a "$OUT"
for win in $(seq 1 $WINDOWS); do
  am=$(parse_med "$(run_dev ARES   A_ares.raw    T_wa.raw $REPS)")
  pm=$(parse_med "$(run_dev PRECR  A_u16_h32.raw T_wp.raw $REPS)")
  vm=$(parse_med "$(run_dev ACVRES A_ares.raw    T_wv.raw $REPS)")
  AW+=($am); PW+=($pm); VW+=($vm)
  echo "win$win: ARES=$am PRECR=$pm ACVRES=$vm" | tee -a "$OUT"
done

python3 - "$OUT" "${AW[@]/%/,}" "|" "${PW[@]/%/,}" "|" "${VW[@]/%/,}" <<'PY' | tee -a "$OUT"
import sys
out=sys.argv[1]; rest=sys.argv[2:]
g=[[],[],[]]; gi=0
for tok in rest:
    if tok=="|": gi+=1; continue
    for x in tok.split(","):
        if x: g[gi].append(int(x))
A,P,V=g
def med(v):
    v=sorted(v); n=len(v); return v[n//2] if n%2 else (v[n//2-1]+v[n//2])//2
def q(v,f):
    v=sorted(v); import math; i=f*(len(v)-1); lo=int(i); return v[lo] if lo==i else v[lo]+(v[lo+1]-v[lo])*(i-lo)
def reject(v,name):
    m=med(v); keep=[x for x in v if x<=1.15*m]; drop=[x for x in v if x>1.15*m]
    if drop: print(f"  [{name}] dropped撞窗 (>1.15x med {m}): {drop}")
    return keep
print("\n===== EXP1 RESULT (撞窗剔除 >1.15x档中位; IQR Q1/Q3) =====")
res={}
for name,v in [("ARES",A),("PRECR",P),("ACVRES",V)]:
    vk=reject(v,name); m=med(vk); q1=q(vk,0.25); q3=q(vk,0.75)
    res[name]=(m,q1,q3,vk)
    print(f"  {name}: median={m}  IQR=[{q1:.0f},{q3:.0f}]  IQRwidth={(q3-q1)/m*100:.1f}%  best={min(vk)}  n={len(vk)}  raw={sorted(vk)}")
# paired same-window (only windows surviving in BOTH)
def paired(a,b,va,vb,ma):
    ma2=med([x for x in va if x<=1.15*med(va)]); mb2=med([x for x in vb if x<=1.15*med(vb)])
    # same-window paired delta on surviving common windows
    pairs=[(x,y) for x,y in zip(va,vb) if x<=1.15*med(va) and y<=1.15*med(vb)]
    if pairs:
        ds=sorted((x-y)/y*100 for x,y in pairs)
        n=len(ds); mededit=ds[n//2] if n%2 else (ds[n//2-1]+ds[n//2])/2
        print(f"  {a}-vs-{b} same-window paired Δ: median={mededit:+.2f}%  range=[{ds[0]:+.2f},{ds[-1]:+.2f}]  npairs={n}")
print()
paired("ARES","PRECR",A,P,res)
paired("ARES","ACVRES",A,V,res)
SHIP=1925000
am=res["ARES"][0]
print(f"\n  ARES median vs SHIP(u8i8 {SHIP}): {(am-SHIP)/SHIP*100:+.1f}%")
print(f"  -> +25% target = {int(SHIP*1.25)}  +27% = {int(SHIP*1.27)}  +30% = {int(SHIP*1.30)}  +32% = {int(SHIP*1.32)}")
PY
echo "" | tee -a "$OUT"
echo "(written to $OUT)"
