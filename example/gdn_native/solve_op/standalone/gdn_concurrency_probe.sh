#!/usr/bin/env bash
# HVX∥HMX concurrency probe for a CUSTOM (QHPI/plugin) op, at O3 + N HVX threads (the parallelism config).
# Builds GdnSolve(HVX) and a native MatMul(HMX) as a dependent chain and as INDEPENDENT chains (no data dep),
# runs on device with optrace, decodes, and measures the TRUE HVX∥HMX overlap from op timestamps (tid 256 vs
# 512-515) via scripts/gdn_overlap_from_trace.py.  Result: custom ops get ~0% overlap because is_plugin_op
# excludes them from compiler supertiling.  POSITIVE CONTROL (all-native, ~17-22% overlap):
#   example/qnn_hmx_matmul_u8i8/standard_flow/native_baseline/run_native_chain.sh (SIZE=256 CHAIN=8).
# Full write-up: docs/qnn_htp_scheduling_and_custom_op_limits.md.  Env: C= B= OPT=(O level) HVXT=(hvx threads)
# VTCM_MB=  RUN_COMBINED=1 (also run the dependent chain).  Device = ssh ${DEVICE:-oneplus}.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../../.. && pwd)"; OPDIR="$(cd .. && pwd)"
PKG="GdnSolvePackage"; PROV="${PKG}InterfaceProvider"; ARCH=v75; DEVICE="${DEVICE:-oneplus}"
source "$ROOT/scripts/env.sh" >/dev/null 2>&1
source "$ROOT/scripts/dssh.sh"
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT/.venv/bin/python"
C="${C:-64}"; B="${B:-64}"
OPT="${OPT:-3}"; HVXT="${HVXT:-4}"; VTCM_MB="${VTCM_MB:-8}"   # concurrency knobs: O3 + N HVX threads + VTCM
bash "$OPDIR/build.sh" >/dev/null 2>&1
X86="$OPDIR/build/x86_64-linux-clang/lib${PKG}.so"; HTP="$OPDIR/build/hexagon-$ARCH/lib${PKG}_htp.so"
CPU="$OPDIR/build/aarch64/lib${PKG}_cpu.so"; CPL="$OPDIR/converter/build/libConverterOpPackage.so"
[ -f A_ref.raw ] || cp A.raw A_ref.raw
cat > _htp.json <<EOF
{"graphs":[{"graph_names":["solve","mm","combined","comb_plugin","indep_native","indep_plugin"],"vtcm_mb":$VTCM_MB,"O":$OPT,"hvx_threads":$HVXT}],
 "devices":[{"dsp_arch":"$ARCH","soc_id":57,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > _cfg.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/_htp.json"}}
EOF
"$PY" "$ROOT/scripts/gdn_overlap_probe.py" . "$C" "$B" A_ref.raw >/dev/null

run() {  # <name> <inputs "n:=f.raw ...">  -> prints "name cyc=<compute cycles> vtcm=<bytes> dram=<bytes>"
  local nm=$1 inputs=$2 d=ov_$1
  qairt-converter -i "$nm.onnx" --target_backend HTP \
     $(for t in A T V P; do echo --source_model_input_layout $t NONTRIVIAL --desired_input_layout $t NONTRIVIAL \
       --source_model_output_layout $t NONTRIVIAL --desired_output_layout $t NONTRIVIAL; done) \
     --op_package_config "$OPDIR/${PKG}.xml" --converter_op_package_lib "$CPL" \
     --quantization_overrides "$nm.ovr.json" -o "$nm.dlc" >_c_$nm.log 2>&1 || { echo "$nm CVTFAIL"; tail -4 _c_$nm.log; return; }
  rm -rf "$d"
  qnn-context-binary-generator --dlc_path "$nm.dlc" --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
     --op_packages "$X86:$PROV" --config_file _cfg.json --profiling_level detailed --profiling_option optrace \
     --binary_file "${nm}_ctx" --output_dir "$d" >_x_$nm.log 2>&1 || { echo "$nm CTXFAIL"; tail -6 _x_$nm.log; return; }
  for s in *schematic.bin "$d"/*schematic.bin; do [ -f "$s" ] && mv -f "$s" "$d"/ 2>/dev/null || true; done
  local W; W="$(ssh "$DEVICE" 'echo $HOME/qnn_run')/ovl"
  ssh "$DEVICE" "mkdir -p $W"
  ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./htp.json\"}}' > $W/cfg.json"
  ssh "$DEVICE" "cat > $W/htp.json" < _htp.json
  ssh "$DEVICE" "cat > $W/${nm}_ctx.bin" < "$d/${nm}_ctx.bin"
  ssh "$DEVICE" "cat > $W/lib${PKG}_htp.so" < "$HTP"; ssh "$DEVICE" "cat > $W/lib${PKG}_cpu.so" < "$CPU"
  for f in A T_in V X; do [ -f "$f.raw" ] && ssh "$DEVICE" "cat > $W/$f.raw" < "$f.raw"; done
  ssh "$DEVICE" "printf '%s\n' '$inputs' > $W/list.txt"
  ssh "$DEVICE" "cd $W && rm -rf out && LD_LIBRARY_PATH=..:.:/vendor/lib64 ADSP_LIBRARY_PATH='..;.;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' ../qnn-net-run \
     --backend ../libQnnHtp.so --retrieve_context ${nm}_ctx.bin --config_file cfg.json \
     --op_packages ./lib${PKG}_cpu.so:$PROV:CPU,./lib${PKG}_htp.so:$PROV:HTP \
     --input_list list.txt --output_dir out --profiling_level detailed --profiling_option optrace --perf_profile burst" >_r_$nm.log 2>&1 || true
  grep -q 'Finished Executing Graphs' _r_$nm.log || { echo "$nm RUNFAIL"; tail -8 _r_$nm.log; return; }
  rm -rf "out_$nm"; mkdir -p "out_$nm"; ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C "out_$nm" --strip-components=1 2>/dev/null
  local cyc; cyc=$(qnn-profile-viewer --input_log "out_$nm/qnn-profiling-data_0.log" 2>/dev/null | grep -iE 'Accelerator \(execute\) time \(cycles\)' | grep -oE '[0-9]+ cycles' | head -1 | grep -oE '[0-9]+')
  # decode the device profile -> chrometrace, then measure TRUE HVX∥HMX overlap from op timestamps
  local schem; schem="$(find "$d" -iname '*schematic*.bin' 2>/dev/null | head -1)"
  "$PY" "$ROOT/scripts/decode_qnn_optrace.py" "out_$nm" \
        --profile-log "out_$nm/qnn-profiling-data_0.log" --schematic "$schem" >/dev/null 2>&1
  echo "  $nm: Accelerator(execute) = ${cyc:-?} cyc"
  "$PY" "$ROOT/scripts/gdn_overlap_from_trace.py" "out_$nm" --label "  $nm" --producer GdnSolve 2>/dev/null \
        || echo "  $nm: (overlap analyze failed — see out_$nm/optrace)"
}

echo "=== HVX∥HMX overlap probe (CUSTOM op) C=$C B=$B  O=$OPT hvx_threads=$HVXT ==="
echo "    expect ~0% (is_plugin_op excludes custom ops from supertiling); native control = run_native_chain.sh"
run solve        "A:=A.raw"                      # HVX-only reference (GdnSolve)
run indep_native "A:=A.raw X:=X.raw Y:=V.raw"    # GdnSolve(HVX) || native MatMul(HMX), NO data dep
[ "${RUN_COMBINED:-0}" = 1 ] && run combined "A:=A.raw V:=V.raw"   # dependent GdnSolve->MatMul (opt-in)
