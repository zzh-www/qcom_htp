#!/usr/bin/env bash
# M8 PER-CHUNK split pipeline test: sweep CK (heads/chain) and measure the architectural root-cause fix.
# Emits ceil(H/CK) independent Op1->Op2 chains so QNN can overlap chain-i Op2(HMX) with chain-(i+1) Op1(HVX)
# and keep each chunk's handoff VTCM-resident.  CK=H reproduces the M6/M7 batched baseline.
#
# Per CK reports (raw device numbers):
#   1. Spill/Fill: sum of @Spill+@Fill+Concat+flat_from_vtcm cyc (does it vanish as CK drops?)
#   2. Cross-tid overlap: HVX tids (512-515) vs HMX tid (256) wall-overlap %, total span vs sum
#   3. Total wall cyc/head (span / H) vs the 70,201 baseline
#   4. full-T relerr vs np.linalg.inv (gate ~7e-2 at C=256)
#
# Env: H (default 16), CB (256), CKS (default "16 4 2 1"), EXTRA_DEFS (passed to BOTH builds).
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../../.. && pwd)"
DIAGDIR="$(cd ../../solve_diag_op && pwd)"
MERGEDIR="$(cd .. && pwd)"
ARCH=v75; DEVICE="${DEVICE:-oneplus}"
DPKG="GdnSolveDiagPackage"; DPROV="${DPKG}InterfaceProvider"
MPKG="GdnMergeHmxPackage";  MPROV="${MPKG}InterfaceProvider"
source "$ROOT/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT/.venv/bin/python"
H="${H:-16}"; CB="${CB:-256}"; EXTRA_DEFS="${EXTRA_DEFS:-}"; CKS="${CKS:-16 4 2 1}"

# Build ops ONCE (shape-parametric in CB only; CK lives in the graph, not the op).
EXTRA_DEFS="${EXTRA_DEFS} -DGDN_BR_C=${CB}" bash "$DIAGDIR/build.sh"  >_build_diag.log  2>&1 || { echo BUILDFAIL_DIAG;  tail -20 _build_diag.log;  exit 1; }
EXTRA_DEFS="${EXTRA_DEFS} -DGDN_BR_C=${CB}" bash "$MERGEDIR/build.sh" >_build_merge.log 2>&1 || { echo BUILDFAIL_MERGE; tail -20 _build_merge.log; exit 1; }
DX86="$DIAGDIR/build/x86_64-linux-clang/lib${DPKG}.so"; DHTP="$DIAGDIR/build/hexagon-$ARCH/lib${DPKG}_htp.so"; DCPU="$DIAGDIR/build/aarch64/lib${DPKG}_cpu.so"
MX86="$MERGEDIR/build/x86_64-linux-clang/lib${MPKG}.so"; MHTP="$MERGEDIR/build/hexagon-$ARCH/lib${MPKG}_htp.so"; MCPU="$MERGEDIR/build/aarch64/lib${MPKG}_cpu.so"
CCPL_DIR="$MERGEDIR/converter/build_combined"; mkdir -p "$CCPL_DIR"
CCPL="$CCPL_DIR/libConverterCombined.so"
clang++ -std=c++17 -O2 -shared -fPIC -I "$QNN_SDK_ROOT/include/QNN" -o "$CCPL" \
   "$DIAGDIR/converter/ConverterOpPackage.cpp" "$MERGEDIR/converter/ConverterOpPackage.cpp" || { echo CCPLFAIL; exit 1; }

cat > _htp.json <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":57,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > _cfg.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/_htp.json"}}
EOF

run_one() {  # $1 = CK
  local CK="$1"
  GDN_CK="$CK" "$PY" "$ROOT/scripts/gdn_split_probe.py" . "$H" "$CB" || { echo PROBEFAIL; return 1; }
  qairt-converter -i split.onnx --target_backend HTP \
     --source_model_input_layout A NONTRIVIAL --desired_input_layout A NONTRIVIAL \
     --source_model_output_layout T NONTRIVIAL --desired_output_layout T NONTRIVIAL \
     --op_package_config "$DIAGDIR/${DPKG}.xml" "$MERGEDIR/${MPKG}.xml" \
     --converter_op_package_lib "$CCPL,$CCPL" \
     --quantization_overrides ovr_split.json -o split.dlc >_c.log 2>&1 || { echo "CVTFAIL CK=$CK"; tail -20 _c.log; return 1; }
  rm -rf ctx_s
  qnn-context-binary-generator --dlc_path split.dlc --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
     --op_packages "$DX86:$DPROV,$MX86:$MPROV" --config_file _cfg.json --profiling_level detailed --profiling_option optrace \
     --binary_file split_ctx --output_dir ctx_s >_x.log 2>&1 || { echo "CTXFAIL CK=$CK"; tail -20 _x.log; return 1; }
  for s in *schematic.bin ctx_s/*schematic.bin; do [ -f "$s" ] && mv -f "$s" ctx_s/ 2>/dev/null || true; done

  local W; W="$(ssh "$DEVICE" 'echo $HOME/qnn_run')/split"
  ssh "$DEVICE" "mkdir -p $W"
  ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./htp.json\"}}' > $W/cfg.json"
  ssh "$DEVICE" "cat > $W/htp.json" < _htp.json
  ssh "$DEVICE" "cat > $W/split_ctx.bin" < ctx_s/split_ctx.bin
  ssh "$DEVICE" "cat > $W/lib${DPKG}_htp.so" < "$DHTP"; ssh "$DEVICE" "cat > $W/lib${DPKG}_cpu.so" < "$DCPU"
  ssh "$DEVICE" "cat > $W/lib${MPKG}_htp.so" < "$MHTP"; ssh "$DEVICE" "cat > $W/lib${MPKG}_cpu.so" < "$MCPU"
  ssh "$DEVICE" "cat > $W/A.raw" < A.raw
  ssh "$DEVICE" "printf 'A:=A.raw\n' > $W/list.txt"
  ssh "$DEVICE" "cd $W && rm -rf out && LD_LIBRARY_PATH=..:.:/vendor/lib64 ADSP_LIBRARY_PATH='..;.;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' ../qnn-net-run \
     --backend ../libQnnHtp.so --retrieve_context split_ctx.bin --config_file cfg.json \
     --op_packages ./lib${DPKG}_cpu.so:$DPROV:CPU,./lib${DPKG}_htp.so:$DPROV:HTP,./lib${MPKG}_cpu.so:$MPROV:CPU,./lib${MPKG}_htp.so:$MPROV:HTP \
     --input_list list.txt --output_dir out --profiling_level detailed --profiling_option optrace --perf_profile burst" >_r.log 2>&1 || true
  grep -q 'Finished Executing Graphs' _r.log || { echo "RUNFAIL CK=$CK"; tail -25 _r.log; return 1; }
  rm -rf out_s; mkdir -p out_s; ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C out_s --strip-components=1 2>/dev/null

  local wall; wall=$(qnn-profile-viewer --input_log out_s/qnn-profiling-data_0.log 2>/dev/null | grep -i 'QNN accelerator (execute) time' | grep -io '[0-9]* us' | head -1)
  local T; T=$(ls out_s/Result_0/T.raw 2>/dev/null || ls out_s/*/T.raw 2>/dev/null | head -1)
  echo ""
  echo "================ CK=$CK  (H=$H CB=$CB)  WALL=$wall ================"

  "$PY" - "$H" "$T" "$CB" <<'PY'
import sys, numpy as np
H=int(sys.argv[1]); Tf=sys.argv[2]; C=int(sys.argv[3]); BL=64; NB=C//BL
t=np.fromfile(Tf,dtype=np.float32)
if t.size < H*C*C:
    print("  OUTPUT TRUNCATED size",t.size,"expected",H*C*C); sys.exit()
t=t[:H*C*C].reshape(H,C,C); r=np.fromfile('T_full_ref.raw',dtype=np.float32)[:H*C*C].reshape(H,C,C)
def rel(a,b):
    d=np.linalg.norm(a-b); n=np.linalg.norm(b); return d/(n+1e-12)
hs=list(range(1,H)) if H>1 else [0]
whole=[rel(t[h],r[h]) for h in hs]
print(f"  relerr vs np.linalg.inv (heads {hs[0]}..{hs[-1]}): mean {np.mean(whole):.3e} max {np.max(whole):.3e}  gate~7e-2: {'PASS' if np.mean(whole)<8e-2 else 'CHECK'}")
PY

  "$PY" "$ROOT/scripts/decode_qnn_optrace.py" out_s --profile-log out_s/qnn-profiling-data_0.log --schematic "$(ls ctx_s/*schematic.bin|head -1)" >/dev/null 2>&1
  "$PY" - out_s "$H" "$CB" <<'PY'
import json,sys
from collections import defaultdict
d=sys.argv[1]; H=int(sys.argv[2]); C=int(sys.argv[3])
try:
    ct=json.load(open(f"{d}/optrace/chrometrace.json"))
except Exception as e:
    print("  (no chrometrace:",e,")"); sys.exit()
ev=ct['traceEvents'] if isinstance(ct,dict) else ct
# 1. Spill/Fill/glue work-volume (sum of dur for the boundary-glue node families).
glue_keys=['@Spill','@Fill','Concat','flat_from_vtcm']
glue=defaultdict(lambda:[0,0])
op1=defaultdict(list); op2=defaultdict(list)
# 2. cross-tid intervals: HVX tids 512-515, HMX tid 256.
hvx=[]; hmx=[]
allspan=[1e18,-1e18]
for e in ev:
    if e.get('ph')!='X': continue
    n=str(e.get('name','')); dur=e.get('dur',0); ts=e.get('ts',0); tid=e.get('tid')
    if dur<=0: continue
    allspan[0]=min(allspan[0],ts); allspan[1]=max(allspan[1],ts+dur)
    for k in glue_keys:
        if k in n: glue[k][0]+=dur; glue[k][1]+=1
    if 'GdnSolveDiag' in n: op1[n].append(dur)
    if 'GdnMergeHmx'  in n: op2[n].append(dur)
    if dur>200:
        if tid in (512,513,514,515): hvx.append((ts,ts+dur))
        elif tid==256: hmx.append((ts,ts+dur))
def union(iv):
    iv=sorted(iv); out=0; cs=ce=None
    for s,e in iv:
        if cs is None: cs,ce=s,e
        elif s<=ce: ce=max(ce,e)
        else: out+=ce-cs; cs,ce=s,e
    if cs is not None: out+=ce-cs
    return out
def inter(a,b):
    a=sorted(a); b=sorted(b); i=j=0; ov=0
    while i<len(a) and j<len(b):
        lo=max(a[i][0],b[j][0]); hi=min(a[i][1],b[j][1])
        if hi>lo: ov+=hi-lo
        if a[i][1]<b[j][1]: i+=1
        else: j+=1
    return ov
gtot=sum(v[0] for v in glue.values())
print("  [1] Spill/Fill/glue work-volume (sum dur):")
for k in glue_keys:
    print(f"        {k:16s} sum={glue[k][0]:>12,}  n={glue[k][1]}")
print(f"        ---- TOTAL glue = {gtot:,} cyc  ({gtot/H:,.0f}/head)")
hvxu=union(hvx); hmxu=union(hmx); ov=inter(hvx,hmx)
span=allspan[1]-allspan[0] if allspan[1]>allspan[0] else 0
print("  [2] cross-tid HVX(512-515) vs HMX(256) overlap:")
print(f"        HVX busy-union={hvxu:,}  HMX busy-union={hmxu:,}  overlap={ov:,}")
if hmxu: print(f"        overlap%% of HMX = {100.0*ov/hmxu:.1f}%   (pipelined if high)")
print(f"        sum(HVX+HMX)={hvxu+hmxu:,}  total-span={span:,}  -> {'PIPELINED (span<sum)' if span<0.9*(hvxu+hmxu) else 'SERIAL (span~=sum)'}")
o1tot=sum(sum(v) for v in op1.values()); o2tot=sum(sum(v) for v in op2.values())
print(f"  [3] node counts: Op1(GdnSolveDiag)={len(op1)} nodes sum={o1tot:,}  Op2(GdnMergeHmx)={len(op2)} nodes sum={o2tot:,}")
print(f"        TOTAL-SPAN cyc/head = {span/H:,.0f}   vs baseline 70,201  ({span/H/70201:.2f}x)")
PY
}

for ck in $CKS; do run_one "$ck"; done
echo ""
echo "DONE.  optrace for the last CK at: $(pwd)/out_s/optrace/chrometrace.json"
