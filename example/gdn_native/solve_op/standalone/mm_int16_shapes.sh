#!/usr/bin/env bash
# Run int16 (u16 act x i16 wt) QNN MatMul at specific shapes on v75 HTP and report
# which HTP op the MatMul lowers to (from the lowered HTP graph + optrace), plus the
# unit (HMX/HVX) and wall. Cases: 1x1x64x64 (single 64x64 matmul, M=64) and
# 1x8x8x64 (A=[1,8,8,64] @ B=[1,1,64,64] broadcast -> [1,8,8,64], 8 batches x M=8, same MAC count).
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../../.. && pwd)"; ARCH=v75; DEVICE="${DEVICE:-oneplus}"
source "$ROOT/scripts/env.sh" >/dev/null 2>&1
source "$ROOT/scripts/dssh.sh" 2>/dev/null || true
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

# case name | A shape | B shape
CASES=(
  "mm_1x1x64x64|1,1,64,64|1,1,64,64"
  "mm_1x8x8x64|1,8,8,64|1,1,64,64"
)

for spec in "${CASES[@]}"; do
  IFS='|' read -r NAME ASH BSH <<< "$spec"
  echo "=== $NAME : A[$ASH] @ B[$BSH] (u16xi16) ==="
  "$PY" "$ROOT/scripts/mm_int16_probe.py" "$NAME" "$ASH" "$BSH"
  cd "$NAME"
  LAY=(); for n in A B; do LAY+=(--source_model_input_layout "$n" NONTRIVIAL --desired_input_layout "$n" NONTRIVIAL); done
  LAY+=(--source_model_output_layout Cout NONTRIVIAL --desired_output_layout Cout NONTRIVIAL)
  qairt-converter -i mm.onnx --target_backend HTP "${LAY[@]}" --quantization_overrides ovr.json -o mm.dlc >_c.log 2>&1 \
    || { echo "  CVTFAIL"; tail -8 _c.log; cd ..; continue; }
  rm -rf ctx
  qnn-context-binary-generator --dlc_path mm.dlc --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
     --config_file "$CFG" --save_backend_op_mapping --profiling_level detailed --profiling_option optrace \
     --binary_file mm_ctx --output_dir ctx >_x.log 2>&1 \
    || { echo "  CTXFAIL"; tail -8 _x.log; cd ..; continue; }
  for s in *schematic.bin ctx/*schematic.bin; do [ -f "$s" ] && mv -f "$s" ctx/ 2>/dev/null || true; done

  # --- lowered HTP op types (host-side, from ctxgen optrace) ---
  echo "  lowered HTP op_types:"
  "$PY" - <<PY
import json,glob
f=glob.glob("ctx/*graph_before*.json")+glob.glob("ctx/*htp*graph*.json")
import os
cand=[c for c in f if os.path.getsize(c)>0]
if cand:
    d=json.load(open(cand[0]))
    print("   ", d.get("op_types"))
    for nid,n in d.get("graph",{}).get("nodes",{}).items():
        print(f"      node {n.get('grouping','?'):40s} type={n.get('type')}")
else:
    print("    (no graph_before json; will read from device optrace)")
PY

  # --- device run + optrace ---
  W="$(ssh "$DEVICE" 'echo $HOME/qnn_run')/mmshape"
  ssh "$DEVICE" "mkdir -p $W"
  ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./htp.json\"}}' > $W/cfg.json"
  ssh "$DEVICE" "cat > $W/htp.json" < "$HTP"
  ssh "$DEVICE" "cat > $W/mm_ctx.bin" < ctx/mm_ctx.bin
  ssh "$DEVICE" "cat > $W/A.raw" < A.raw; ssh "$DEVICE" "cat > $W/B.raw" < B.raw
  ssh "$DEVICE" "printf 'A:=A.raw B:=B.raw\n' > $W/list.txt"
  ssh "$DEVICE" "cd $W && rm -rf out && LD_LIBRARY_PATH=..:/vendor/lib64 ADSP_LIBRARY_PATH=.. ../qnn-net-run \
     --backend ../libQnnHtp.so --retrieve_context mm_ctx.bin --config_file cfg.json \
     --input_list list.txt --output_dir out --profiling_level detailed --profiling_option optrace --perf_profile burst" >_r.log 2>&1 || true
  if ! grep -q 'Finished Executing Graphs' _r.log; then echo "  RUNFAIL"; tail -12 _r.log; cd ..; continue; fi
  rm -rf out; mkdir -p out; ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C out --strip-components=1 2>/dev/null
  "$PY" "$ROOT/scripts/decode_qnn_optrace.py" out --profile-log out/qnn-profiling-data_0.log --schematic "$(ls ctx/*schematic.bin|head -1)" >/dev/null 2>&1

  qnn-profile-viewer --input_log out/qnn-profiling-data_0.log 2>/dev/null | grep -iE 'accelerator \(execute\) time' | grep -iv cycles | sed 's/^ */    wall /' | sort -u
  "$PY" - out <<'PY'
import json,sys,collections,glob,os
d=sys.argv[1]
f=f"{d}/optrace/chrometrace_qnn_htp_analysis_summary.json"
if not os.path.exists(f):
    print("    (no analysis summary)"); raise SystemExit
s=json.load(open(f))
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
by=collections.Counter(); cyc=collections.defaultdict(int); hmx=False
for it in (insts or []):
    nm=it.get("htp_op","?"); by[nm]+=1; cyc[nm]+=it.get("cycles",0)
    if it.get("hmx"): hmx=True
res=find(s,"htp_resources")
util={}
if res:
    res=res["data"] if isinstance(res,dict) else res
    util={r.get('type'):r.get('utilization',0) for r in res}
print(f"    HTP ops: {dict(by.most_common(8))}")
print(f"    unit={'HMX-used' if hmx else 'no-HMX'}  HMXutil={util.get('HMX',0)}%  HVXutil={util.get('HVX',0)}%  total_cyc={sum(cyc.values()):,}")
PY
  cd ..
done
