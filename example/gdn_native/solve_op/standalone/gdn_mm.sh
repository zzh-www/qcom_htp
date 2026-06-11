#!/usr/bin/env bash
# Measure compute-cycle cost of one u8xi8 batched matmul [1,H,C,C]@[1,H,C,C] (the "iterative multiply"
# unit of a matmul-based solve), to compare vs the forward-substitution kernel.  Reports cyc/head and
# what unit it maps to (HMX/HVX).
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../../.. && pwd)"; ARCH=v75; DEVICE="${DEVICE:-oneplus}"
source "$ROOT/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT/.venv/bin/python"
HTP="$(pwd)/_htp.json"; CFG="$(pwd)/_cfg.json"
cat > "$HTP" <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":57,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > "$CFG" <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$HTP"}}
EOF
H="${H:-32}"
for C in ${CS:-32}; do
  "$PY" "$ROOT/scripts/gdn_mm_probe.py" mm_$C "$C" "$H" >/dev/null
  cd mm_$C
  LAY=(); for n in A B; do LAY+=(--source_model_input_layout "$n" NONTRIVIAL --desired_input_layout "$n" NONTRIVIAL); done
  LAY+=(--source_model_output_layout Cout NONTRIVIAL --desired_output_layout Cout NONTRIVIAL)
  qairt-converter -i mm.onnx --target_backend HTP "${LAY[@]}" --quantization_overrides ovr.json -o mm.dlc >_c.log 2>&1 || { echo "C=$C CVTFAIL"; tail -5 _c.log; cd ..; continue; }
  rm -rf ctx
  qnn-context-binary-generator --dlc_path mm.dlc --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
     --config_file "$CFG" --save_backend_op_mapping --profiling_level detailed --profiling_option optrace \
     --binary_file mm_ctx --output_dir ctx >_x.log 2>&1 || { echo "C=$C CTXFAIL"; tail -5 _x.log; cd ..; continue; }
  for s in *schematic.bin ctx/*schematic.bin; do [ -f "$s" ] && mv -f "$s" ctx/ 2>/dev/null || true; done
  W="$(ssh "$DEVICE" 'echo $HOME/qnn_run')/mm"
  ssh "$DEVICE" "mkdir -p $W"
  ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./htp.json\"}}' > $W/cfg.json"
  ssh "$DEVICE" "cat > $W/htp.json" < "$HTP"
  ssh "$DEVICE" "cat > $W/mm_ctx.bin" < ctx/mm_ctx.bin
  ssh "$DEVICE" "cat > $W/A.raw" < A.raw; ssh "$DEVICE" "cat > $W/B.raw" < B.raw
  ssh "$DEVICE" "printf 'A:=A.raw B:=B.raw\n' > $W/list.txt"
  ssh "$DEVICE" "cd $W && rm -rf out && LD_LIBRARY_PATH=..:/vendor/lib64 ADSP_LIBRARY_PATH=.. ../qnn-net-run \
     --backend ../libQnnHtp.so --retrieve_context mm_ctx.bin --config_file cfg.json \
     --input_list list.txt --output_dir out --profiling_level detailed --profiling_option optrace --perf_profile burst" >_r.log 2>&1 || true
  rm -rf out; mkdir -p out; ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C out --strip-components=1 2>/dev/null
  "$PY" "$ROOT/scripts/decode_qnn_optrace.py" out --profile-log out/qnn-profiling-data_0.log --schematic "$(ls ctx/*schematic.bin|head -1)" >/dev/null 2>&1
  "$PY" - "$C" "$H" out <<'PY'
import json,sys,collections
C=int(sys.argv[1]); H=int(sys.argv[2]); d=sys.argv[3]
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
# which unit + cycles, per htp_op
by=collections.Counter(); cyc=collections.defaultdict(int); hmx=False
for it in insts:
    nm=it.get("htp_op","?"); by[nm]+=1; cyc[nm]+=it.get("cycles",0)
    if it.get("hmx"): hmx=True
# the matmul compute = ConvLayer (HMX) or mul_op (HVX); warm-split the matmul instances
mmcyc=sorted(it.get("cycles",0) for it in insts if ("ConvLayer" in it.get("htp_op","") or "mul" in it.get("htp_op","")) and it.get("cycles",0)>500)
import statistics
warm=[c for c in mmcyc if c < 3*statistics.median(mmcyc)] if mmcyc else []
res=find(s,"htp_resources")["data"]; util={r['type']:r['utilization'] for r in res}
print(f"  matmul C={C} u8xi8 -> unit={'HMX' if hmx else 'HVX'}  HMXutil={util.get('HMX',0):.0f}% "
      f"ops={dict(by.most_common(5))}")
print(f"    matmul-instances warm: n={len(warm)} mean={statistics.mean(warm):,.0f}/tile  "
      f"all-op total cyc={sum(cyc.values()):,}")
PY
  cd ..
done
