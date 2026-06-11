#!/usr/bin/env bash
# M6 two-op SPLIT device validation: A -> GdnSolveDiag(HVX,multithreaded) -> T1 ; GdnMergeHmx(HMX) -> T.
# Loads BOTH op packages.  Reports full-T relerr vs np.linalg.inv + the optrace node list (to check
# whether QNN inserts ForceFormat/convert between Op1 and Op2) + per-op cyc + HVX/HMX tid overlap.
#
# Env: H (heads, default 16), CB (256), EXTRA_DEFS (passed to BOTH builds).
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
H="${H:-16}"; CB="${CB:-256}"; EXTRA_DEFS="${EXTRA_DEFS:-}"

EXTRA_DEFS="${EXTRA_DEFS} -DGDN_BR_C=${CB}" bash "$DIAGDIR/build.sh"  >_build_diag.log  2>&1 || { echo BUILDFAIL_DIAG;  tail -20 _build_diag.log;  exit 1; }
EXTRA_DEFS="${EXTRA_DEFS} -DGDN_BR_C=${CB}" bash "$MERGEDIR/build.sh" >_build_merge.log 2>&1 || { echo BUILDFAIL_MERGE; tail -20 _build_merge.log; exit 1; }

DX86="$DIAGDIR/build/x86_64-linux-clang/lib${DPKG}.so"; DHTP="$DIAGDIR/build/hexagon-$ARCH/lib${DPKG}_htp.so"; DCPU="$DIAGDIR/build/aarch64/lib${DPKG}_cpu.so"
MX86="$MERGEDIR/build/x86_64-linux-clang/lib${MPKG}.so"; MHTP="$MERGEDIR/build/hexagon-$ARCH/lib${MPKG}_htp.so"; MCPU="$MERGEDIR/build/aarch64/lib${MPKG}_cpu.so"
# ONE combined converter lib (both ops' inference symbols); qairt-converter takes a single lib.
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

"$PY" "$ROOT/scripts/gdn_split_probe.py" . "$H" "$CB" || { echo PROBEFAIL; exit 1; }

qairt-converter -i split.onnx --target_backend HTP \
   --source_model_input_layout A NONTRIVIAL --desired_input_layout A NONTRIVIAL \
   --source_model_output_layout T NONTRIVIAL --desired_output_layout T NONTRIVIAL \
   --op_package_config "$DIAGDIR/${DPKG}.xml" "$MERGEDIR/${MPKG}.xml" \
   --converter_op_package_lib "$CCPL,$CCPL" \
   --quantization_overrides ovr_split.json -o split.dlc >_c.log 2>&1 || { echo CVTFAIL; tail -15 _c.log; exit 1; }
rm -rf ctx_s
qnn-context-binary-generator --dlc_path split.dlc --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
   --op_packages "$DX86:$DPROV,$MX86:$MPROV" --config_file _cfg.json --profiling_level detailed --profiling_option optrace \
   --binary_file split_ctx --output_dir ctx_s >_x.log 2>&1 || { echo CTXFAIL; tail -15 _x.log; exit 1; }
for s in *schematic.bin ctx_s/*schematic.bin; do [ -f "$s" ] && mv -f "$s" ctx_s/ 2>/dev/null || true; done

W="$(ssh "$DEVICE" 'echo $HOME/qnn_run')/split"
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
grep -q 'Finished Executing Graphs' _r.log || { echo "RUNFAIL"; tail -20 _r.log; exit 1; }
rm -rf out_s; mkdir -p out_s; ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C out_s --strip-components=1 2>/dev/null

wall=$(qnn-profile-viewer --input_log out_s/qnn-profiling-data_0.log 2>/dev/null | grep -i 'QNN accelerator (execute) time' | grep -io '[0-9]* us' | head -1)
T=$(ls out_s/Result_0/T.raw 2>/dev/null || ls out_s/*/T.raw 2>/dev/null | head -1)
echo "  >>> H=$H CB=$CB WALL=$wall  T=$T  EXTRA_DEFS='${EXTRA_DEFS}'"

"$PY" - "$H" "$T" "$CB" <<'PY'
import sys, numpy as np
H=int(sys.argv[1]); Tf=sys.argv[2]; C=int(sys.argv[3]); BL=64; NB=C//BL
t=np.fromfile(Tf,dtype=np.float32)
if t.size < H*C*C:
    print("  OUTPUT TRUNCATED size",t.size,"expected",H*C*C); sys.exit()
t=t[:H*C*C].reshape(H,C,C)
r=np.fromfile('T_full_ref.raw',dtype=np.float32)[:H*C*C].reshape(H,C,C)
def rel(a,b):
    d=np.linalg.norm(a-b); n=np.linalg.norm(b); return d/(n+1e-12)
hs = list(range(1,H)) if H>1 else [0]
whole=[rel(t[h],r[h]) for h in hs]
def blkrel(diag):
    out=[]
    for h in hs:
        vals=[rel(t[h,i*BL:(i+1)*BL,j*BL:(j+1)*BL], r[h,i*BL:(i+1)*BL,j*BL:(j+1)*BL])
              for i in range(NB) for j in range(NB) if (i==j)==diag and j<=i]
        if vals: out.append(np.mean(vals))
    return out
dg=blkrel(True); off=blkrel(False)
print(f"  full-T relerr vs np.linalg.inv (heads {hs[0]}..{hs[-1]}): mean {np.mean(whole):.3e} max {np.max(whole):.3e}")
print(f"    diag-block relerr mean {np.mean(dg):.3e}   off-diag relerr mean {np.mean(off) if off else 0:.3e}")
for n,h in enumerate(hs[:4]): print(f"    head {h}: whole {whole[n]:.3e}")
print(f"  PASS gate ~2.4e-2 (u8i8 BR ceiling): {'PASS' if np.mean(whole)<3.0e-2 else 'CHECK'}")
PY

# decode optrace -> chrometrace, then per-op cyc + per-stage PROBE_CYCLES
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
byname=defaultdict(list); tids=defaultdict(set)
for e in ev:
    n=str(e.get('name','')); dd=e.get('dur',0)
    if dd>200:
        byname[n].append(dd); tids[n].add(e.get('tid'))
def report(key,div):
    # Op1 is central-tiled 8 heads/op (busiest tile = max_dur, per-head=max/8).
    # Op2 is NOT tiled (one node over all H heads): per-head = max_dur/H.
    tot=0
    for n,ds in byname.items():
        if key in n:
            ds=sorted(ds)
            print(f"  {n}: n_ops={len(ds)} max_dur={ds[-1]:,} sum_dur={sum(ds):,} tids={sorted(tids[n])} per-head={ds[-1]/div:,.0f}")
            tot=max(tot,ds[-1]/div)
    return tot
o1=report('GdnSolveDiag',8); o2=report('GdnMergeHmx',H)
print(f"  >>> SPLIT per-head: Op1={o1:,.0f}  Op2={o2:,.0f}  serial-total={o1+o2:,.0f}  vs baseline 70,201")
PY
