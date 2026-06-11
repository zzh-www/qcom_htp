#!/usr/bin/env bash
# Prefill-throughput scaling: HVX GdnSolve op vs HMX squaring-matmul, as batch B (head-solves ~ chunks*heads)
# grows.  At small B the HVX op wins (low dispatch); the question is whether at prefill scale HVX saturates
# and the HMX squaring wins.  Reports wall + HMX/HVX util per (B, method).
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

util() {  # decode optrace -> "wall_us HMXutil HVXmaxutil"
  local d=$1 sch=$2
  "$PY" "$ROOT/scripts/decode_qnn_optrace.py" "$d" --profile-log "$d/qnn-profiling-data_0.log" --schematic "$sch" >/dev/null 2>&1
  "$PY" - "$d" <<'PY'
import json,sys
d=sys.argv[1]
try:
  s=json.load(open(f"{d}/optrace/chrometrace_qnn_htp_analysis_summary.json"))
except: print("? ? ?"); raise SystemExit
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
res=find(s,"htp_resources")["data"]
hmx=max((r["utilization"] for r in res if r["type"]=="HMX"),default=0)
hvx=max((r["utilization"] for r in res if r["type"]=="HVX"),default=0)
print(f"{hmx:.1f} {hvx:.1f}")
PY
}

run_one() {  # method dlc_extra_convert  -> echoes "wall hmx hvx"
  local name=$1 onnx=$2 ovr=$3; shift 3
  local cvt=("$@")
  qairt-converter -i "$onnx" --target_backend HTP \
     --source_model_input_layout A NONTRIVIAL --desired_input_layout A NONTRIVIAL \
     --source_model_output_layout T NONTRIVIAL --desired_output_layout T NONTRIVIAL \
     --quantization_overrides "$ovr" "${cvt[@]}" -o "${name}.dlc" >"_c_$name.log" 2>&1 || { echo "CVTFAIL"; return; }
  rm -rf "ctx_$name"
  local pkg=(); [ "$name" = solve ] && pkg=(--op_packages "$X86:$PROV")
  qnn-context-binary-generator --dlc_path "${name}.dlc" --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
     "${pkg[@]}" --config_file _cfg.json --profiling_level detailed --profiling_option optrace \
     --binary_file "${name}_ctx" --output_dir "ctx_$name" >"_x_$name.log" 2>&1 || { echo "CTXFAIL"; return; }
  for s in *schematic.bin ctx_$name/*schematic.bin; do [ -f "$s" ] && mv -f "$s" "ctx_$name/" 2>/dev/null || true; done
  local W; W="$(ssh "$DEVICE" 'echo $HOME/qnn_run')/scal_$name"
  ssh "$DEVICE" "mkdir -p $W"
  ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./htp.json\"}}' > $W/cfg.json"
  ssh "$DEVICE" "cat > $W/htp.json" < _htp.json
  ssh "$DEVICE" "cat > $W/${name}_ctx.bin" < "ctx_$name/${name}_ctx.bin"
  ssh "$DEVICE" "cat > $W/A.raw" < A.raw
  ssh "$DEVICE" "printf 'A:=A.raw\n' > $W/list.txt"
  local op_args="" adsp=".."
  if [ "$name" = solve ]; then
     ssh "$DEVICE" "cat > $W/lib${PKG}_htp.so" < "$HTP"; ssh "$DEVICE" "cat > $W/lib${PKG}_cpu.so" < "$CPU"
     op_args="--op_packages ./lib${PKG}_cpu.so:$PROV:CPU,./lib${PKG}_htp.so:$PROV:HTP"; adsp="..;.;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp"
  fi
  ssh "$DEVICE" "cd $W && rm -rf out && LD_LIBRARY_PATH=..:.:/vendor/lib64 ADSP_LIBRARY_PATH='$adsp' ../qnn-net-run \
     --backend ../libQnnHtp.so --retrieve_context ${name}_ctx.bin --config_file cfg.json $op_args \
     --input_list list.txt --output_dir out --profiling_level detailed --profiling_option optrace --perf_profile burst" >"_r_$name.log" 2>&1 || true
  ssh "$DEVICE" "grep -q 'Finished Executing Graphs' $W/../scal_$name/_*.log 2>/dev/null" || true
  rm -rf "out_$name"; mkdir -p "out_$name"; ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C "out_$name" --strip-components=1 2>/dev/null
  local wall; wall=$(qnn-profile-viewer --input_log "out_$name/qnn-profiling-data_0.log" 2>/dev/null | grep -i 'QNN accelerator (execute) time' | grep -io '[0-9]* us' | head -1)
  local u; u=$(util "out_$name" "$(ls ctx_$name/*schematic.bin|head -1)")
  echo "${wall:-?} $u"
}

[ -f A_ref.raw ] || cp A.raw A_ref.raw                  # snapshot the real 32-head A (probe overwrites A.raw)
printf '%-6s | %-22s | %-22s\n' "B" "HVX-op  wall/HMX%/HVX%" "squaring wall/HMX%/HVX%"
for B in ${BS:-32 128 512 1024}; do
  "$PY" "$ROOT/scripts/gdn_scaling_probe.py" . "$B" A_ref.raw >/dev/null 2>&1
  s=$(run_one solve solve.onnx ovr_solve.json --op_package_config "$OPDIR/${PKG}.xml" --converter_op_package_lib "$CPL")
  q=$(run_one sq    sq.onnx    ovr_sq.json)
  printf '%-6s | %-22s | %-22s\n' "$B" "$s" "$q"
done
