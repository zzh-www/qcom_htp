#!/usr/bin/env bash
# W16_N8 full GP_RESIDENT (ARES) — 照搬 pure-HMX gdn_pure_solve.cpp GP_RESIDENT 完整模式。
#
# 与 ACVRES 的区别 (这是真收口):
#   ACVRES = host cv-block + DSP 每-head 48KB DDR->VTCM 平拷 (head-load) + 指针直返。只方差收敛, mean 未降。
#   ARES   = host cv-block 全 32-head 一次性常驻 VTCM 共享尾区 (vbase+P*0xA0000, 1.5MB) + alias 零拷。
#            消掉那 285K/head-load 平拷 -> 兑现 PRECROUTON 实测的 ~2.50M 地板, 带正确 T。
#
# build (同 SHIP+FUSEDEP+ACTCACHE base):
#   DUMP       = base + -DGDN_BR_W16_N8_ACVRES_DUMP : 正常求解, head 末把 6 个 cv-block A 面拷进 T (cv-block 源)。
#   BASE       = base (actcache, 生产高精度档)        : 正确 T 参照 (oracle) + wall A。
#   ACVRES     = base + -DGDN_BR_W16_N8_ACVRES        : 平拷+指针直返 (半截), wall 对照。
#   ARES       = base + -DGDN_BR_W16_N8_ARES          : 全常驻 + alias 零拷 (真收口), wall C。
#   PRECROUTON = base + -DGDN_BR_A_PRECROUTON         : act-zero 地板 (T 错, 仅 wall 口径), wall 锚。
#
# 门:
#   (1) bit-exact 硬门: ARES T md5 == BASE T md5 (整 32-head 4MB 逐字节) + max|d|=0 + oc 相同。
#   (2) wall 门 (ACAC 同热窗背靠背配对, reps2-N median, N+ 窗, 带 spread + BASE best-window):
#       C(ARES) vs PRECROUTON 同窗 -> 验是否真吃到 ~2.50M (producer 路应等价)。
#       C(ARES) vs ACVRES     同窗 -> 验 -10.2% (消 head-load)。
#       不要用 noisy BASE median 报百分比。
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"; BM="$ROOT/example/gdn_native/baremetal"
SHIP="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL -DGDN_BR_SBOOST -DGDN_BR_DIAG_I16 -DGDN_BR_REQ_FUSE -DGDN_BR_FBOOST"
BASE="$SHIP -DGDN_BR_W16 -DGDN_BR_W16_N8 -DGDN_BR_W16_N8_FUSEDEP -DGDN_BR_W16_N8_ACTCACHE"
SCALES="2.770166930875267e-05 6.103701895199438e-05"
H=32; C=256
WINDOWS="${WINDOWS:-4}"; REPS="${REPS:-8}"; THREADS="${THREADS:-4}"
source "$ROOT/scripts/dssh.sh"; dssh_open

build() { ( cd "$BM" && EXTRA_DEFS="$1" bash build.sh ) >/tmp/ares_$2.log 2>&1 || { echo "BUILD FAIL $2"; tail -25 /tmp/ares_$2.log; exit 1; }
  mkdir -p /tmp/ares_$2; cp "$BM/build/libgdnbm_skel.so" "$BM/build/gdnbm" /tmp/ares_$2/; }
echo "=== build DUMP       ==="; build "$BASE -DGDN_BR_W16_N8_ACVRES_DUMP" DUMP
echo "=== build BASE       ==="; build "$BASE" BASE
echo "=== build ACVRES     ==="; build "$BASE -DGDN_BR_W16_N8_ACVRES" ACVRES
echo "=== build ARES       ==="; build "$BASE -DGDN_BR_W16_N8_ARES" ARES
echo "=== build PRECROUTON ==="; build "$BASE -DGDN_BR_A_PRECROUTON" PRECR

deploy() { local V="$1" W="\$HOME/ares_$1"; dssh "mkdir -p $W"
  dssh "cat > $W/libgdnbm_skel.so" < /tmp/ares_$V/libgdnbm_skel.so
  dssh "cat > $W/gdnbm" < /tmp/ares_$V/gdnbm; dssh "chmod +x $W/gdnbm"; }
for V in DUMP BASE ACVRES ARES PRECR; do deploy "$V"; done
for V in DUMP BASE ACVRES ARES PRECR; do dssh "cat > \$HOME/ares_$V/A_u16_h32.raw" < "$BM/A_u16_h32.raw"; done

run_dev() {  # $1=V $2=Ainput $3=Tout $4=reps -> stdout
  local V="$1" Ain="$2" Tout="$3" rp="${4:-1}" W="\$HOME/ares_$1"
  dssh "cd $W && GDNBM_REPS=$rp LD_LIBRARY_PATH=\$PWD:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH=\"\$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" ./gdnbm $THREADS $Ain $Tout $H $C 32768 32768 $SCALES 2>&1"; }

echo ""; echo "### STEP 1: DUMP run -> extract host cv-block A, build extended A_ares.raw ###"
run_dev DUMP A_u16_h32.raw T_dump.raw 1 >/tmp/ares_dump.log 2>&1
grep -E "rc=|wrote" /tmp/ares_dump.log | head -3
dssh "cat \$HOME/ares_DUMP/T_dump.raw" > /tmp/T_dump.raw
python3 - <<'PY'
import numpy as np
H=32; CC=256*256; HEAD_U16=CC
dump=np.fromfile("/tmp/T_dump.raw", dtype="<u2")
nat =np.fromfile("/home/zzh/work/qcom_htp/example/gdn_native/baremetal/A_u16_h32.raw", dtype="<u2")
assert dump.size==H*HEAD_U16, dump.size
cv = np.zeros((H, 6*4096), dtype="<u2")
for h in range(H): cv[h] = dump[h*HEAD_U16 : h*HEAD_U16 + 6*4096]
ext = np.concatenate([nat, cv.ravel()])
ext.tofile("/tmp/A_ares.raw")
print(f"  natural A u16={nat.size}  cv-block u16={cv.size}  extended u16={ext.size}  ({ext.nbytes} bytes)")
PY
for V in ACVRES ARES; do dssh "cat > \$HOME/ares_$V/A_ares.raw" < /tmp/A_ares.raw; done

echo ""; echo "### STEP 2: BASE (oracle T) + ARES -> bit-exact gate ###"
run_dev BASE A_u16_h32.raw T_base.raw 1 >/tmp/ares_base.log 2>&1
run_dev ARES A_ares.raw    T_ares.raw 1 >/tmp/ares_ares.log 2>&1
grep -E "rc=" /tmp/ares_base.log | head -1
grep -E "rc=" /tmp/ares_ares.log | head -1
dssh "cat \$HOME/ares_BASE/T_base.raw" > /tmp/T_base.raw
dssh "cat \$HOME/ares_ARES/T_ares.raw" > /tmp/T_ares.raw
echo "  md5:"; md5sum /tmp/T_base.raw /tmp/T_ares.raw
if cmp -s /tmp/T_base.raw /tmp/T_ares.raw; then echo "  cmp: IDENTICAL (bit-exact PASS)"; else echo "  cmp: DIFFER (bit-exact FAIL)"; cmp /tmp/T_base.raw /tmp/T_ares.raw | head; fi
python3 - <<'PY'
import numpy as np
b=np.fromfile("/tmp/T_base.raw","<u2"); c=np.fromfile("/tmp/T_ares.raw","<u2")
print(f"  numpy: array_equal={np.array_equal(b,c)}  max|d|={int(np.abs(b.astype(np.int32)-c.astype(np.int32)).max())}")
PY

[ "${BITEXACT_ONLY:-0}" = "1" ] && { echo "(BITEXACT_ONLY=1, skip wall)"; exit 0; }

parse_med() { printf '%s\n' "$1" | python3 -c '
import sys,re
rows=[]
for ln in sys.stdin:
    m=re.search(r"rep=(\d+)/\d+\s+wall=(\d+)", ln)
    if m: rows.append((int(m.group(1)),int(m.group(2))))
st=sorted(w for r,w in rows if r>=2)
if not st: print("0 0 0 0")
else:
    n=len(st); med=st[n//2] if n%2 else (st[n//2-1]+st[n//2])//2
    print(f"{med} {st[0]} {st[-1]} {n}")
'; }

echo ""; echo "### STEP 3a: wall ACAC C(ARES) vs PRECROUTON (act-zero 地板) x$WINDOWS REPS=$REPS ###"
declare -a CW PW
for win in $(seq 1 $WINDOWS); do
  lc=$(run_dev ARES  A_ares.raw      T_wc.raw $REPS); rc=$(parse_med "$lc")
  lp=$(run_dev PRECR A_u16_h32.raw   T_wp.raw $REPS); rp=$(parse_med "$lp")
  cm=$(echo $rc|cut -d' ' -f1); pm=$(echo $rp|cut -d' ' -f1)
  CW+=($cm); PW+=($pm)
  if [ "$pm" -gt 0 ] && [ "$cm" -gt 0 ]; then d=$(python3 -c "print(f'{($cm-$pm)/$pm*100:+.2f}')"); else d=NA; fi
  echo "win$win: C(ARES)=$rc | P(PRECR地板)=$rp | C-vs-P=${d}%"
done

echo ""; echo "### STEP 3b: wall ACAC C(ARES) vs ACVRES (head-load 半截) x$WINDOWS REPS=$REPS ###"
declare -a CW2 VW
for win in $(seq 1 $WINDOWS); do
  lc=$(run_dev ARES   A_ares.raw T_wc2.raw $REPS); rc=$(parse_med "$lc")
  lv=$(run_dev ACVRES A_ares.raw T_wv.raw  $REPS); rv=$(parse_med "$lv")
  cm=$(echo $rc|cut -d' ' -f1); vm=$(echo $rv|cut -d' ' -f1)
  CW2+=($cm); VW+=($vm)
  if [ "$vm" -gt 0 ] && [ "$cm" -gt 0 ]; then d=$(python3 -c "print(f'{($cm-$vm)/$vm*100:+.2f}')"); else d=NA; fi
  echo "win$win: C(ARES)=$rc | V(ACVRES)=$rv | C-vs-V=${d}%"
done

echo ""; echo "### STEP 3c: BASE best-window (noisy, NOT for %; reference anchor only) x$WINDOWS ###"
declare -a BW
for win in $(seq 1 $WINDOWS); do
  lb=$(run_dev BASE A_u16_h32.raw T_wbb.raw $REPS); rb=$(parse_med "$lb")
  bm=$(echo $rb|cut -d' ' -f1); BW+=($bm); echo "win$win: BASE=$rb"
done

echo ""
python3 - <<PY
def med(v): v=sorted(v); n=len(v); return v[n//2] if n%2 else (v[n//2-1]+v[n//2])//2
def spread(v):
    v=sorted(v); return 0.0 if not v or med(v)==0 else (v[-1]-v[0])/med(v)*100
C =[x for x in [${CW[@]/%/,}]  if x>0]; P=[x for x in [${PW[@]/%/,}] if x>0]
C2=[x for x in [${CW2[@]/%/,}] if x>0]; V=[x for x in [${VW[@]/%/,}] if x>0]
B =[x for x in [${BW[@]/%/,}]  if x>0]
SHIP=1925000
print("===== MERGED (medians, spread, paired same-window) =====")
if C and P:
    cm,pm=med(C),med(P); print(f"ARES vs PRECROUTON: C_med={cm} (spread {spread(C):.1f}%) P_med={pm} (spread {spread(P):.1f}%)  C-vs-P={(cm-pm)/pm*100:+.2f}%  [到地板? 应≈0]")
if C2 and V:
    cm,vm=med(C2),med(V); print(f"ARES vs ACVRES:     C_med={cm} (spread {spread(C2):.1f}%) V_med={vm} (spread {spread(V):.1f}%)  C-vs-V={(cm-vm)/vm*100:+.2f}%  [目标 -10.2%]")
if B:
    print(f"BASE best-window={min(B)} median={med(B)} (spread {spread(B):.1f}%)  [noisy, anchor only, NOT for %]")
if C:
    cm=med(C); print(f"W16_N8 ARES vs SHIP(u8i8 {SHIP}): {(cm-SHIP)/SHIP*100:+.1f}%   (target ~+30%)")
print(f"  C wins (ARES)={C}")
print(f"  P wins (PRECR)={P}")
print(f"  V wins (ACVRES)={V}")
PY
