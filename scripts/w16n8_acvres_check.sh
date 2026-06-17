#!/usr/bin/env bash
# W16_N8 act 上游协同 真兑现 (ACVRES) — 照搬 pure-HMX GP_CVIO/GP_RESIDENT 的 host cv-block A 数据流。
#
# 数据流 (= pure-HMX GP_CVIO):
#   host 一次性把 A 的 6 个 distinct merge A-act 块准备成 cv-block (fold+quant+crouton_pos 序),
#   DSP head-load 平拷进 VTCM ACTCACHE 槽 (零 fold / 零 quant / 零 vgather), merge get_act_A = 指针直返。
#   host cv-block 准备 = DSP-dump (一次性非计时, 用与求解内完全相同的 HVX fold+crouton 路 => 逐字节 bit-exact),
#   完全对应 pure-HMX 的 "host 离线把 A 准备成 cv-block, 计时循环只平拷"。
#
# 3 个 build (同 SHIP+FUSEDEP+ACTCACHE base):
#   DUMP = base + -DGDN_BR_W16_N8_ACVRES_DUMP : 正常求解, head 末把 6 个 cv-block A 面拷进 T (T=dump 载体)。
#   BASE = base (actcache, 生产高精度档)       : 正确 T 参照 (oracle) + wall A。
#   ACVRES = base + -DGDN_BR_W16_N8_ACVRES     : 扩展 A 输入 (natural || cv-block), 平拷+指针直返, wall C。
#
# 门: (1) bit-exact 硬门 = ACVRES T md5 == BASE T md5 (整 32-head 4MB 逐字节) + oc 相同。
#     (2) wall 门 = ACAC 3 窗 reps2-8 median, C(ACVRES) vs A(BASE), 目标 -12.66% (真兑现, T 正确)。
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"; BM="$ROOT/example/gdn_native/baremetal"
SHIP="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL -DGDN_BR_SBOOST -DGDN_BR_DIAG_I16 -DGDN_BR_REQ_FUSE -DGDN_BR_FBOOST"
BASE="$SHIP -DGDN_BR_W16 -DGDN_BR_W16_N8 -DGDN_BR_W16_N8_FUSEDEP -DGDN_BR_W16_N8_ACTCACHE"
SCALES="2.770166930875267e-05 6.103701895199438e-05"
H=32; C=256
WINDOWS="${WINDOWS:-3}"; REPS="${REPS:-8}"; THREADS="${THREADS:-4}"
source "$ROOT/scripts/dssh.sh"; dssh_open

build() { ( cd "$BM" && EXTRA_DEFS="$1" bash build.sh ) >/tmp/acv_$2.log 2>&1 || { echo "BUILD FAIL $2"; tail -25 /tmp/acv_$2.log; exit 1; }
  mkdir -p /tmp/acv_$2; cp "$BM/build/libgdnbm_skel.so" "$BM/build/gdnbm" /tmp/acv_$2/; }
echo "=== build DUMP   ==="; build "$BASE -DGDN_BR_W16_N8_ACVRES_DUMP" DUMP
echo "=== build BASE   ==="; build "$BASE" BASE
echo "=== build ACVRES ==="; build "$BASE -DGDN_BR_W16_N8_ACVRES" ACVRES

# deploy each variant into its own dir on device
deploy() {  # $1=V
  local V="$1" W="\$HOME/acv_$1"; dssh "mkdir -p $W"
  dssh "cat > $W/libgdnbm_skel.so" < /tmp/acv_$V/libgdnbm_skel.so
  dssh "cat > $W/gdnbm" < /tmp/acv_$V/gdnbm; dssh "chmod +x $W/gdnbm"
}
for V in DUMP BASE ACVRES; do deploy "$V"; done
# natural A (32-head, 4MB) into all dirs
for V in DUMP BASE ACVRES; do dssh "cat > \$HOME/acv_$V/A_u16_h32.raw" < "$BM/A_u16_h32.raw"; done

run_dev() {  # $1=V $2=Ainput $3=Tout $4=reps  -> stdout
  local V="$1" Ain="$2" Tout="$3" rp="${4:-1}" W="\$HOME/acv_$1"
  dssh "cd $W && GDNBM_REPS=$rp LD_LIBRARY_PATH=\$PWD:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH=\"\$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" ./gdnbm $THREADS $Ain $Tout $H $C 32768 32768 $SCALES 2>&1"
}

echo ""; echo "### STEP 1: DUMP run -> extract host cv-block A, build extended A_acvres.raw ###"
run_dev DUMP A_u16_h32.raw T_dump.raw 1 >/tmp/acv_dump.log 2>&1
grep -E "rc=|wrote" /tmp/acv_dump.log | head -3
dssh "cat \$HOME/acv_DUMP/T_dump.raw" > /tmp/T_dump.raw
python3 - <<'PY'
import numpy as np
H=32; CC=256*256; HEAD_U16=CC  # natural T head = 256*256 u16 = 131072 u16
dump=np.fromfile("/tmp/T_dump.raw", dtype="<u2")   # H*131072 u16
nat =np.fromfile("/home/zzh/work/qcom_htp/example/gdn_native/baremetal/A_u16_h32.raw", dtype="<u2")  # H*131072 u16
assert dump.size==H*HEAD_U16, dump.size
# per head: first 6*4096 u16 = the 6 cv-block A surfaces (8KB each). append after natural A.
cv = np.zeros((H, 6*4096), dtype="<u2")
for h in range(H):
    cv[h] = dump[h*HEAD_U16 : h*HEAD_U16 + 6*4096]
ext = np.concatenate([nat, cv.ravel()])
ext.tofile("/tmp/A_acvres.raw")
print(f"  natural A u16={nat.size}  cv-block u16={cv.size}  extended u16={ext.size}  ({ext.nbytes} bytes)")
PY
dssh "cat > \$HOME/acv_ACVRES/A_acvres.raw" < /tmp/A_acvres.raw

echo ""; echo "### STEP 2: BASE (oracle T) + ACVRES (cv-block) single-rep -> bit-exact gate ###"
run_dev BASE   A_u16_h32.raw  T_base.raw   1 >/tmp/acv_base.log 2>&1
run_dev ACVRES A_acvres.raw   T_acvres.raw 1 >/tmp/acv_acv.log 2>&1
grep -E "rc=" /tmp/acv_base.log | head -1
grep -E "rc=" /tmp/acv_acv.log  | head -1
dssh "cat \$HOME/acv_BASE/T_base.raw"     > /tmp/T_base.raw
dssh "cat \$HOME/acv_ACVRES/T_acvres.raw" > /tmp/T_acvres.raw
echo "  md5:"; md5sum /tmp/T_base.raw /tmp/T_acvres.raw
if cmp -s /tmp/T_base.raw /tmp/T_acvres.raw; then echo "  cmp: IDENTICAL (bit-exact PASS)"; else echo "  cmp: DIFFER (bit-exact FAIL)"; cmp /tmp/T_base.raw /tmp/T_acvres.raw | head; fi
python3 - <<'PY'
import numpy as np
b=np.fromfile("/tmp/T_base.raw","<u2"); c=np.fromfile("/tmp/T_acvres.raw","<u2")
print(f"  numpy: array_equal={np.array_equal(b,c)}  max|d|={int(np.abs(b.astype(np.int32)-c.astype(np.int32)).max())}")
PY

[ "${BITEXACT_ONLY:-0}" = "1" ] && { echo "(BITEXACT_ONLY=1, skip wall)"; exit 0; }

echo ""; echo "### STEP 3: wall ACAC x$WINDOWS REPS=$REPS THREADS=$THREADS (A=BASE vs C=ACVRES, reps2-$REPS median) ###"
parse_med() {  # $1 = log text -> "med min max n"
  printf '%s\n' "$1" | python3 -c '
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
'
}
declare -a AW CW
for win in $(seq 1 $WINDOWS); do
  la=$(run_dev BASE   A_u16_h32.raw T_wb.raw  $REPS); ra=$(parse_med "$la")
  lc=$(run_dev ACVRES A_acvres.raw  T_wc.raw  $REPS); rc=$(parse_med "$lc")
  am=$(echo $ra|cut -d' ' -f1); cm=$(echo $rc|cut -d' ' -f1)
  AW+=($am); CW+=($cm)
  if [ "$am" -gt 0 ] && [ "$cm" -gt 0 ]; then d=$(python3 -c "print(f'{($cm-$am)/$am*100:+.2f}')"); else d=NA; fi
  echo "win$win: A(BASE)=$ra | C(ACVRES)=$rc | dWall=${d}%"
done
echo ""
python3 - <<PY
A=[x for x in [${AW[@]/%/,}] if x>0]; C=[x for x in [${CW[@]/%/,}] if x>0]
def med(v): v=sorted(v); n=len(v); return v[n//2] if n%2 else (v[n//2-1]+v[n//2])//2
if A and C:
    am,cm=med(A),med(C)
    print(f"{len(A)}-WINDOW MERGED: A_med={am} C_med={cm}  dWall={(cm-am)/am*100:+.2f}%  (C=ACVRES cv-block vs A=actcache base)")
    print(f"  A wins={A}  C wins={C}")
PY
