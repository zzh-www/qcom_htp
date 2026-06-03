#!/usr/bin/env bash
# gdn_solve_chain.sh — chain8-style steady per-op cycle test for the QNN GdnSolve op.
# Builds a graph of N chained GdnSolve nodes, runs with optrace, and reports per-NODE cycles so the
# cold first op is separated from the steady op[1..N-1] (aligned PCYCLE). Mirrors the qnn_hmx chain
# methodology (example/qnn_hmx_matmul_u8i8/.../run_native_chain.sh).
#   CHAIN=8 C=256 H=32 bash gdn_solve_chain.sh
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"; OPDIR="$ROOT/example/gdn_native/solve_op"
PKG="GdnSolvePackage"; PROV="${PKG}InterfaceProvider"; ARCH=v75; DEVICE="${DEVICE:-oneplus}"
CHAIN="${CHAIN:-8}"; C="${C:-256}"; H="${H:-32}"
source "$ROOT/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT/.venv/bin/python"
WD="$(pwd)/_chain"; mkdir -p "$WD"; cd "$WD"

echo "[1/5] build op + gen chain onnx (N=$CHAIN, C=$C, H=$H)"
bash "$OPDIR/build.sh" >_b.log 2>&1 || { echo BUILDFAIL; tail -8 _b.log; exit 1; }
X86="$OPDIR/build/x86_64-linux-clang/lib${PKG}.so"; HTP="$OPDIR/build/hexagon-$ARCH/lib${PKG}_htp.so"
CPU="$OPDIR/build/aarch64/lib${PKG}_cpu.so"; CPL="$OPDIR/converter/build/libConverterOpPackage.so"
[ -f "$ROOT/example/gdn_native/solve_op/standalone/A.raw" ] && REFA="$ROOT/example/gdn_native/solve_op/standalone/A.raw" || REFA="$ROOT/example/gdn_native/baremetal/A_h32.raw"
"$PY" "$ROOT/scripts/gdn_solve_chain_probe.py" . "$C" "$CHAIN" "$REFA" "$H"

cat > _htp.json <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":57,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > _cfg.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/_htp.json"}}
EOF

echo "[2/5] convert + ctxgen (optrace)"
qairt-converter -i solve_chain.onnx --target_backend HTP \
   --source_model_input_layout A NONTRIVIAL --desired_input_layout A NONTRIVIAL \
   --source_model_output_layout T NONTRIVIAL --desired_output_layout T NONTRIVIAL \
   --op_package_config "$OPDIR/${PKG}.xml" --converter_op_package_lib "$CPL" \
   --quantization_overrides ovr_chain.json -o solve_chain.dlc >_c.log 2>&1 || { echo CVTFAIL; tail -8 _c.log; exit 1; }
rm -rf ctx_s
qnn-context-binary-generator --dlc_path solve_chain.dlc --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
   --op_packages "$X86:$PROV" --config_file _cfg.json --profiling_level detailed --profiling_option optrace \
   --binary_file chain_ctx --output_dir ctx_s >_x.log 2>&1 || { echo CTXFAIL; tail -8 _x.log; exit 1; }
for s in *schematic.bin ctx_s/*schematic.bin; do [ -f "$s" ] && mv -f "$s" ctx_s/ 2>/dev/null || true; done

echo "[3/5] deploy + run"
W="$(ssh "$DEVICE" 'echo $HOME/qnn_run')/chain"; ssh "$DEVICE" "mkdir -p $W"
ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./htp.json\"}}' > $W/cfg.json"
ssh "$DEVICE" "cat > $W/htp.json" < _htp.json
ssh "$DEVICE" "cat > $W/chain_ctx.bin" < ctx_s/chain_ctx.bin
ssh "$DEVICE" "cat > $W/lib${PKG}_htp.so" < "$HTP"; ssh "$DEVICE" "cat > $W/lib${PKG}_cpu.so" < "$CPU"
ssh "$DEVICE" "cat > $W/A.raw" < A.raw; ssh "$DEVICE" "printf 'A:=A.raw\n' > $W/list.txt"
ssh "$DEVICE" "cd $W && rm -rf out && LD_LIBRARY_PATH=..:.:/vendor/lib64 ADSP_LIBRARY_PATH='..;.;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' ../qnn-net-run \
   --backend ../libQnnHtp.so --retrieve_context chain_ctx.bin --config_file cfg.json \
   --op_packages ./lib${PKG}_cpu.so:$PROV:CPU,./lib${PKG}_htp.so:$PROV:HTP \
   --input_list list.txt --output_dir out --profiling_level detailed --profiling_option optrace --perf_profile burst" >_r.log 2>&1 || true
grep -q 'Finished Executing Graphs' _r.log || { echo RUNFAIL; tail -10 _r.log; exit 1; }
rm -rf out_s; mkdir -p out_s; ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C out_s --strip-components=1 2>/dev/null
wall=$(qnn-profile-viewer --input_log out_s/qnn-profiling-data_0.log 2>/dev/null | grep -i 'QNN accelerator (execute) time' | grep -io '[0-9]* us' | head -1)

echo "[4/5] decode optrace"
"$PY" "$ROOT/scripts/decode_qnn_optrace.py" out_s --profile-log out_s/qnn-profiling-data_0.log --schematic "$(ls ctx_s/*schematic.bin|head -1)" >/dev/null 2>&1

echo "[5/5] per-node steady cycles (chain N=$CHAIN, C=$C, H=$H, graph wall=$wall)"
"$PY" - "$CHAIN" "$H" out_s <<'PY'
import json,sys,statistics
N=int(sys.argv[1]); H=int(sys.argv[2]); d=sys.argv[3]
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
# per-node total cycles = sum of that node's tile instances
node={}
for it in insts:
    nm=it.get('qnn_op','')+it.get('htp_op','')
    cyc=it.get('cycles',0)
    for i in range(N):
        if f"GdnSolve_{i}" in nm and cyc>200:
            node.setdefault(i,[]).append(cyc); break
tot={i:sum(v) for i,v in sorted(node.items())}
print("  node   total_cyc   cyc/head   tiles")
for i in sorted(tot):
    print(f"   {i:<4}  {tot[i]:>9,}  {tot[i]//H:>8,}  {len(node[i])}")
steady=[tot[i] for i in tot if i>0]
if steady:
    med=statistics.median(steady); cold=tot.get(0,med)
    print(f"  => STEADY per-op (median of node[1..{N-1}]) = {med:,.0f} cyc = {med//H:,} cyc/head"
          f"   (cold node0 = {cold:,}, {cold/med:.2f}x)")
PY
