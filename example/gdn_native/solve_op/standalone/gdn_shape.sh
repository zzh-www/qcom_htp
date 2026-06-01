#!/usr/bin/env bash
# Probe which matrix size C is most HVX-efficient for the GdnSolve op: same 32 heads, C = 64/32/16.
# Reports wall, HVX util, T relerr, per-instance cycle spread (straggler), and wall normalized per C^2 elem.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../../.. && pwd)"; OPDIR="$(cd .. && pwd)"
PKG="GdnSolvePackage"; PROV="${PKG}InterfaceProvider"; ARCH=v75; DEVICE="${DEVICE:-oneplus}"
source "$ROOT/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT/.venv/bin/python"
bash "$OPDIR/build.sh" >/dev/null 2>&1
X86="$OPDIR/build/x86_64-linux-clang/lib${PKG}.so"; HTP="$OPDIR/build/hexagon-$ARCH/lib${PKG}_htp.so"
CPU="$OPDIR/build/aarch64/lib${PKG}_cpu.so"; CPL="$OPDIR/converter/build/libConverterOpPackage.so"
cat > _htp.json <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":57,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > _cfg.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/_htp.json"}}
EOF
[ -f A_ref.raw ] || cp A.raw A_ref.raw

for C in ${CS:-64 32 16}; do
  "$PY" "$ROOT/scripts/gdn_shape_probe.py" . "$C" A_ref.raw >/dev/null
  qairt-converter -i solve.onnx --target_backend HTP \
     --source_model_input_layout A NONTRIVIAL --desired_input_layout A NONTRIVIAL \
     --source_model_output_layout T NONTRIVIAL --desired_output_layout T NONTRIVIAL \
     --op_package_config "$OPDIR/${PKG}.xml" --converter_op_package_lib "$CPL" \
     --quantization_overrides ovr_solve.json -o solve.dlc >_c.log 2>&1 || { echo "C=$C CVTFAIL"; tail -5 _c.log; continue; }
  rm -rf ctx_s
  qnn-context-binary-generator --dlc_path solve.dlc --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
     --op_packages "$X86:$PROV" --config_file _cfg.json --profiling_level detailed --profiling_option optrace \
     --binary_file solve_ctx --output_dir ctx_s >_x.log 2>&1 || { echo "C=$C CTXFAIL"; tail -5 _x.log; continue; }
  for s in *schematic.bin ctx_s/*schematic.bin; do [ -f "$s" ] && mv -f "$s" ctx_s/ 2>/dev/null || true; done
  W="$(ssh "$DEVICE" 'echo $HOME/qnn_run')/shape"
  ssh "$DEVICE" "mkdir -p $W"
  ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./htp.json\"}}' > $W/cfg.json"
  ssh "$DEVICE" "cat > $W/htp.json" < _htp.json
  ssh "$DEVICE" "cat > $W/solve_ctx.bin" < ctx_s/solve_ctx.bin
  ssh "$DEVICE" "cat > $W/lib${PKG}_htp.so" < "$HTP"; ssh "$DEVICE" "cat > $W/lib${PKG}_cpu.so" < "$CPU"
  ssh "$DEVICE" "cat > $W/A.raw" < A.raw
  ssh "$DEVICE" "printf 'A:=A.raw\n' > $W/list.txt"
  ssh "$DEVICE" "cd $W && rm -rf out && LD_LIBRARY_PATH=..:.:/vendor/lib64 ADSP_LIBRARY_PATH='..;.;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' ../qnn-net-run \
     --backend ../libQnnHtp.so --retrieve_context solve_ctx.bin --config_file cfg.json \
     --op_packages ./lib${PKG}_cpu.so:$PROV:CPU,./lib${PKG}_htp.so:$PROV:HTP \
     --input_list list.txt --output_dir out --profiling_level detailed --profiling_option optrace --perf_profile burst" >_r.log 2>&1 || true
  rm -rf out_s; mkdir -p out_s; ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C out_s --strip-components=1 2>/dev/null
  wall=$(qnn-profile-viewer --input_log out_s/qnn-profiling-data_0.log 2>/dev/null | grep -i 'QNN accelerator (execute) time' | grep -io '[0-9]* us' | head -1)
  T=$(ls out_s/Result_0/T.raw 2>/dev/null || ls out_s/*/T.raw 2>/dev/null|head -1)
  rel=$("$PY" -c "import numpy as np;t=np.fromfile('$T',dtype=np.float32);r=np.fromfile('T_ref.raw',dtype=np.float32);n=min(t.size,r.size);print(f'{np.linalg.norm(t[:n]-r[:n])/(np.linalg.norm(r[:n])+1e-12):.2e}')" 2>/dev/null)
  "$PY" "$ROOT/scripts/decode_qnn_optrace.py" out_s --profile-log out_s/qnn-profiling-data_0.log --schematic "$(ls ctx_s/*schematic.bin|head -1)" >/dev/null 2>&1
  "$PY" - "$C" "$rel" out_s <<'PY'
import json,sys
C=int(sys.argv[1]); rel=sys.argv[2]; d=sys.argv[3]
s=json.load(open(f"{d}/optrace/chrometrace_qnn_htp_analysis_summary.json"))
def find(o,k):
    if isinstance(o,dict):
        if k in o:return o[k]
        for v in o.values():
            r=find(v,k)
            if r is not None:return r
    elif isinstance(o,list):
        for v in o:
            r=find(v,k)
            if r is not None:return r
insts=find(s,"htp_op_instances"); insts=insts["data"] if isinstance(insts,dict) else insts
gs=sorted(it.get('cycles',0) for it in insts if "GdnSolve" in it.get('qnn_op','')+it.get('htp_op','') and it.get('cycles',0)>500)
import statistics
# WARM = exclude cold-start outliers (tiles > 3x median are warmup); each tile = 8 heads
med=statistics.median(gs); warm=[c for c in gs if c < 3*med]
wmean=statistics.mean(warm); per_head=wmean/8; per_elem=per_head/(C*C)
print(f"  C={C:<3} relerr={rel:<9} tiles={len(gs)} cold(excl)={len(gs)-len(warm)}  "
      f"WARM/tile(8h)={wmean:>8,.0f}  cyc/head={per_head:>7,.0f}  cyc/elem={per_elem:>5.1f}  "
      f"[total_incl_cold={sum(gs):,}]")
PY
done
