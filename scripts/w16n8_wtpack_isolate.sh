#!/usr/bin/env bash
# 实验 A: ARES 态剥离纯 wt-pack vs Tdiag-act-pack。
#
# 背景: analyzer 揪出 W16_N8 纯 wt-pack ~5100/+27% vs pure-HMX 4026 (dev 旧说"贴 4026"未剥离实测)。
# trace stage-8 PACK 桶把 wt-pack + Tdiag-act-pack 混在一起 (=4871 mixed)。本实验两条正交方法剥离:
#
# 方法1 (cap wall-delta, 真 wall 口径): ARES 态 off-diag A-act 已 alias 跳过, 所以现成 cap 旗能干净剥离:
#   +CAP_PACK_W -> 只 cap gdn_w16_pack_wt = 纯 wt-pack (off-diag T + diag-S final)。
#   +CAP_PACK_A -> ARES 态只剩 Tdiag-act-pack (off-diag A 已 alias)。
#   wall-delta(BASE - CAP) = 各自对 WALL 的真实贡献 (注: 4 producer 并行, wall-delta != Σ-work)。
#
# 方法2 (PACKCNT Σ-work + per-inst, 同 pure-HMX 4026 口径): +PACKCNT clean 计数器在 gdn_w16_pack_wt /
#   gdn_w16_pack_act 内分别累 Σ-work(C15:14, 全 4 线程)+ 调用数 -> per-inst = Σ÷n, 直接对 pure-HMX 4026。
#   这是回答 "纯 wt-pack per-inst 是 ~5100/+27% 还是贴 4026" 的决定性数据 (同口径)。
#
# (cap 破坏正确性, 只测 wall 不验 oc。PACKCNT 不破坏正确性, 但本实验只取其 Σ/per-inst。)
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"; BM="$ROOT/example/gdn_native/baremetal"
OUT="$ROOT/Agent/current/w16n8_wtpack_isolate.txt"
SHIP="-DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL -DGDN_BR_SBOOST -DGDN_BR_DIAG_I16 -DGDN_BR_REQ_FUSE -DGDN_BR_FBOOST"
BASE="$SHIP -DGDN_BR_W16 -DGDN_BR_W16_N8 -DGDN_BR_W16_N8_FUSEDEP -DGDN_BR_W16_N8_ACTCACHE -DGDN_BR_W16_N8_ARES"
SCALES="2.770166930875267e-05 6.103701895199438e-05"
H=32; C=256
WINDOWS="${WINDOWS:-5}"; REPS="${REPS:-8}"; THREADS="${THREADS:-4}"
source "$ROOT/scripts/dssh.sh"; dssh_open

build() { ( cd "$BM" && EXTRA_DEFS="$1" bash build.sh ) >/tmp/wtiso_$2.log 2>&1 || { echo "BUILD FAIL $2"; tail -25 /tmp/wtiso_$2.log; exit 1; }
  mkdir -p /tmp/wtiso_$2; cp "$BM/build/libgdnbm_skel.so" "$BM/build/gdnbm" /tmp/wtiso_$2/; }
echo "=== build DUMP    (cv-block dumper)  ==="; build "$BASE -DGDN_BR_W16_N8_ACVRES_DUMP" DUMP
echo "=== build BASE    (ARES)            ==="; build "$BASE" BASE
echo "=== build CAPW    (BASE +CAP_PACK_W)==="; build "$BASE -DGDN_BR_CAP_PACK_W" CAPW
echo "=== build CAPA    (BASE +CAP_PACK_A)==="; build "$BASE -DGDN_BR_CAP_PACK_A" CAPA
echo "=== build PACKCNT (BASE +PACKCNT)   ==="; build "$BASE -DGDN_BR_W16_PACKCNT" PCNT

deploy() { local V="$1" W="\$HOME/wtiso_$1"; dssh "mkdir -p $W"
  dssh "cat > $W/libgdnbm_skel.so" < /tmp/wtiso_$V/libgdnbm_skel.so
  dssh "cat > $W/gdnbm" < /tmp/wtiso_$V/gdnbm; dssh "chmod +x $W/gdnbm"
  dssh "cat > $W/A_u16_h32.raw" < "$BM/A_u16_h32.raw"; }
for V in DUMP BASE CAPW CAPA PCNT; do deploy "$V"; done

# ARES 态需要 extended A (Au natural + 32-head cv-block tail)。DUMP build 正常求解并把 6 个 cv-block A
# 面拷进 T；host 拼接成 A_ares.raw (照搬 ARES harness)。
echo "### STEP0: DUMP -> build A_ares.raw (extended A for ARES) ###"
dssh "cd \$HOME/wtiso_DUMP && GDNBM_REPS=1 LD_LIBRARY_PATH=\$PWD:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH=\"\$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" ./gdnbm $THREADS A_u16_h32.raw T_dump.raw $H $C 32768 32768 $SCALES 2>&1" | grep -E "rc=" | head -1
dssh "cat \$HOME/wtiso_DUMP/T_dump.raw" > /tmp/wtiso_T_dump.raw
python3 - <<'PY'
import numpy as np
H=32; HEAD_U16=256*256
dump=np.fromfile("/tmp/wtiso_T_dump.raw", dtype="<u2")
nat =np.fromfile("/home/zzh/work/qcom_htp/example/gdn_native/baremetal/A_u16_h32.raw", dtype="<u2")
assert dump.size==H*HEAD_U16, dump.size
cv = np.zeros((H, 6*4096), dtype="<u2")
for h in range(H): cv[h] = dump[h*HEAD_U16 : h*HEAD_U16 + 6*4096]
ext = np.concatenate([nat, cv.ravel()]); ext.tofile("/tmp/wtiso_A_ares.raw")
print(f"  extended A u16={ext.size} ({ext.nbytes} bytes)")
PY
for V in BASE CAPW CAPA PCNT; do dssh "cat > \$HOME/wtiso_$V/A_ares.raw" < /tmp/wtiso_A_ares.raw; done

run_dev() {  # $1=V $2=reps -> stdout
  local V="$1" rp="${2:-1}" W="\$HOME/wtiso_$1"
  dssh "cd $W && GDNBM_REPS=$rp LD_LIBRARY_PATH=\$PWD:/vendor/lib64:/system/lib64 ADSP_LIBRARY_PATH=\"\$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp\" ./gdnbm $THREADS A_ares.raw T_out.raw $H $C 32768 32768 $SCALES 2>&1"; }

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

: > "$OUT"
{
echo "=================================================================="
echo " 实验 A: W16_N8 ARES 态剥离纯 wt-pack vs Tdiag-act-pack"
echo " $(date)  WINDOWS=$WINDOWS REPS=$REPS THREADS=$THREADS"
echo "=================================================================="
echo ""
echo "### 方法2 (决定性, 同 pure-HMX 4026 口径): PACKCNT clean Σ-work + per-inst ###"
} >> "$OUT"
pcl=$(run_dev PCNT 4)
echo "$pcl" | grep -E "wall=|PACKCNT" | head -8 | tee -a "$OUT"
echo "" >> "$OUT"

echo ""; echo "### 方法1 (cap wall-delta, 真 wall 口径) ACAC paired x$WINDOWS ###" | tee -a "$OUT"
declare -a BW WW AW
for win in $(seq 1 $WINDOWS); do
  lb=$(run_dev BASE $REPS); rb=$(parse_med "$lb"); bm=$(echo $rb|cut -d' ' -f1)
  lw=$(run_dev CAPW $REPS); rw=$(parse_med "$lw"); wm=$(echo $rw|cut -d' ' -f1)
  la=$(run_dev CAPA $REPS); ra=$(parse_med "$la"); am=$(echo $ra|cut -d' ' -f1)
  BW+=($bm); WW+=($wm); AW+=($am)
  dw=$(python3 -c "print(f'{$bm-$wm}' if $bm>0 and $wm>0 else 'NA')")
  da=$(python3 -c "print(f'{$bm-$am}' if $bm>0 and $am>0 else 'NA')")
  echo "win$win: BASE=$rb | CAPW=$rw (Δwt=$dw) | CAPA=$ra (Δact_Tdiag=$da)" | tee -a "$OUT"
done

echo "" | tee -a "$OUT"
python3 - <<PY | tee -a "$OUT"
def med(v): v=sorted(v); n=len(v); return v[n//2] if n%2 else (v[n//2-1]+v[n//2])//2
def spread(v):
    v=sorted(v); return 0.0 if not v or med(v)==0 else (v[-1]-v[0])/med(v)*100
B=[x for x in [${BW[@]/%/,}] if x>0]; W=[x for x in [${WW[@]/%/,}] if x>0]; A=[x for x in [${AW[@]/%/,}] if x>0]
print("===== 方法1 MERGED (paired same-window, medians) =====")
if B and W and A:
    bm,wm,am=med(B),med(W),med(A)
    print(f"BASE wall median = {bm}  (spread {spread(B):.1f}%)")
    print(f"CAPW wall median = {wm}  (spread {spread(W):.1f}%)  -> wt-pack WALL-delta = {bm-wm} ({(bm-wm)/bm*100:+.1f}% of wall)")
    print(f"CAPA wall median = {am}  (spread {spread(A):.1f}%)  -> Tdiag-act WALL-delta = {bm-am} ({(bm-am)/bm*100:+.1f}% of wall)")
    print(f"  注: wall-delta 是 4-producer 并行后的 wall 影响, 非 Σ-work; per-inst 比较看方法2。")
    print(f"  BASE wins={B}")
    print(f"  CAPW wins={W}")
    print(f"  CAPA wins={A}")
PY
echo "" | tee -a "$OUT"
echo "落盘: $OUT"
