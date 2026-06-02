#!/usr/bin/env bash
# Device validation for GdnSolveBR (C=128 block-recursive inverse, HVX diagonals + HMX merge).
# Builds A->GdnSolveBR->T DLC, ctxgen, runs on `ssh oneplus`, reports T relerr vs np.linalg.inv +
# per-block relerr (T11/T22 diag vs T21 off-diag) + steady GdnSolveBR compute cyc.
#
# Env:
#   H=16            number of heads (default 16)
#   EXTRA_DEFS=...  rebuild the op with a debug mode (-DGDN_BR_SKIP_KERNEL / -DGDN_BR_DIAG_ONLY /
#                   -DGDN_BR_DUMP_M / -DGDN_BR_PROBE_CYCLES); passed straight to build.sh
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../../.. && pwd)"; OPDIR="$(cd .. && pwd)"
PKG="GdnSolveBRPackage"; PROV="${PKG}InterfaceProvider"; ARCH=v75; DEVICE="${DEVICE:-oneplus}"
source "$ROOT/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT/.venv/bin/python"
H="${H:-16}"
EXTRA_DEFS="${EXTRA_DEFS:-}"

EXTRA_DEFS="${EXTRA_DEFS}" bash "$OPDIR/build.sh" >_build.log 2>&1 || { echo BUILDFAIL; tail -20 _build.log; exit 1; }
X86="$OPDIR/build/x86_64-linux-clang/lib${PKG}.so"; HTP="$OPDIR/build/hexagon-$ARCH/lib${PKG}_htp.so"
CPU="$OPDIR/build/aarch64/lib${PKG}_cpu.so"; CPL="$OPDIR/converter/build/libConverterOpPackage.so"
cat > _htp.json <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":57,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > _cfg.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/_htp.json"}}
EOF

"$PY" "$ROOT/scripts/gdn_solve_br_probe.py" . "$H" || { echo PROBEFAIL; exit 1; }

qairt-converter -i solve_br.onnx --target_backend HTP \
   --source_model_input_layout A NONTRIVIAL --desired_input_layout A NONTRIVIAL \
   --source_model_output_layout T NONTRIVIAL --desired_output_layout T NONTRIVIAL \
   --op_package_config "$OPDIR/${PKG}.xml" --converter_op_package_lib "$CPL" \
   --quantization_overrides ovr_solve_br.json -o solve_br.dlc >_c.log 2>&1 || { echo CVTFAIL; tail -8 _c.log; exit 1; }
rm -rf ctx_s
qnn-context-binary-generator --dlc_path solve_br.dlc --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
   --op_packages "$X86:$PROV" --config_file _cfg.json --profiling_level detailed --profiling_option optrace \
   --binary_file solve_br_ctx --output_dir ctx_s >_x.log 2>&1 || { echo CTXFAIL; tail -8 _x.log; exit 1; }
for s in *schematic.bin ctx_s/*schematic.bin; do [ -f "$s" ] && mv -f "$s" ctx_s/ 2>/dev/null || true; done

W="$(ssh "$DEVICE" 'echo $HOME/qnn_run')/solve_br"
ssh "$DEVICE" "mkdir -p $W"
ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./htp.json\"}}' > $W/cfg.json"
ssh "$DEVICE" "cat > $W/htp.json" < _htp.json
ssh "$DEVICE" "cat > $W/solve_br_ctx.bin" < ctx_s/solve_br_ctx.bin
ssh "$DEVICE" "cat > $W/lib${PKG}_htp.so" < "$HTP"; ssh "$DEVICE" "cat > $W/lib${PKG}_cpu.so" < "$CPU"
ssh "$DEVICE" "cat > $W/A.raw" < A.raw
ssh "$DEVICE" "printf 'A:=A.raw\n' > $W/list.txt"
ssh "$DEVICE" "cd $W && rm -rf out && LD_LIBRARY_PATH=..:.:/vendor/lib64 ADSP_LIBRARY_PATH='..;.;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' ../qnn-net-run \
   --backend ../libQnnHtp.so --retrieve_context solve_br_ctx.bin --config_file cfg.json \
   --op_packages ./lib${PKG}_cpu.so:$PROV:CPU,./lib${PKG}_htp.so:$PROV:HTP \
   --input_list list.txt --output_dir out --profiling_level detailed --profiling_option optrace --perf_profile burst" >_r.log 2>&1 || true
grep -q 'Finished Executing Graphs' _r.log || { echo "RUNFAIL"; tail -12 _r.log; exit 1; }
rm -rf out_s; mkdir -p out_s; ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C out_s --strip-components=1 2>/dev/null

wall=$(qnn-profile-viewer --input_log out_s/qnn-profiling-data_0.log 2>/dev/null | grep -i 'QNN accelerator (execute) time' | grep -io '[0-9]* us' | head -1)
T=$(ls out_s/Result_0/T.raw 2>/dev/null || ls out_s/*/T.raw 2>/dev/null | head -1)
echo "  >>> H=$H WALL=$wall  T=$T  EXTRA_DEFS='${EXTRA_DEFS}'"

"$PY" - "$H" "$T" <<'PY'
import sys, numpy as np
H=int(sys.argv[1]); Tf=sys.argv[2]; C=128; BL=64
t=np.fromfile(Tf,dtype=np.float32)
if t.size < H*C*C:
    print("  OUTPUT TRUNCATED size",t.size,"expected",H*C*C); sys.exit()
t=t[:H*C*C].reshape(H,C,C)
r=np.fromfile('T_ref.raw',dtype=np.float32)[:H*C*C].reshape(H,C,C)
def rel(a,b):
    d=np.linalg.norm(a-b); n=np.linalg.norm(b)
    return d/(n+1e-12)
whole=[rel(t[h],r[h]) for h in range(H)]
t11=[rel(t[h,:BL,:BL],r[h,:BL,:BL]) for h in range(H)]
t22=[rel(t[h,BL:,BL:],r[h,BL:,BL:]) for h in range(H)]
t21=[rel(t[h,BL:,:BL],r[h,BL:,:BL]) for h in range(H)]
print(f"  per-head T relerr vs np.linalg.inv: mean {np.mean(whole):.3e} max {np.max(whole):.3e}")
print(f"    block T11 diag relerr mean {np.mean(t11):.3e}  T22 diag mean {np.mean(t22):.3e}  T21 off-diag mean {np.mean(t21):.3e}")
for h in range(min(H,4)):
    print(f"    head {h}: whole {whole[h]:.3e}  T11 {t11[h]:.3e}  T22 {t22[h]:.3e}  T21 {t21[h]:.3e}")
print(f"  PASS gate ~7e-3 (host u8i8 BR ceiling): {'PASS' if np.mean(whole)<1.2e-2 else 'CHECK'}")
PY

# steady GdnSolveBR compute cyc (warm tile, cold excluded; 8 heads/tile)
"$PY" "$ROOT/scripts/decode_qnn_optrace.py" out_s --profile-log out_s/qnn-profiling-data_0.log --schematic "$(ls ctx_s/*schematic.bin|head -1)" >/dev/null 2>&1
"$PY" - out_s <<'PY'
import json,sys,statistics
d=sys.argv[1]
try:
    s=json.load(open(f"{d}/optrace/chrometrace_qnn_htp_analysis_summary.json"))
except Exception as e:
    print("  (no optrace summary:",e,")"); sys.exit()
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
gs=sorted(it.get('cycles',0) for it in (insts or []) if "GdnSolveBR" in it.get('qnn_op','')+it.get('htp_op','') and it.get('cycles',0)>500)
if not gs: print("  (no GdnSolveBR op instances in optrace)"); sys.exit()
# single-instance op (multithreaded=false): this optrace 'cycles' is the thread-aggregate over ALL
# heads -- use the PROBE_CYCLES breakdown for real per-stage cycles, not this number.
print(f"  GdnSolveBR instances={len(gs)} aggregate_cyc(all heads)={sum(gs):,}  (use EXTRA_DEFS=-DGDN_BR_PROBE_CYCLES for per-stage cyc)")
PY
