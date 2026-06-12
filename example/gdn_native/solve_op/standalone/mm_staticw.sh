#!/usr/bin/env bash
# int16 64^3 MatMul with a STATIC weight (B=initializer) — measures how much of QNN's per-call
# wrapper folds to prepare-time when the weight is reused. Compares to dynamic-weight single 64^3.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../../.. && pwd)"; ARCH=v75; DEVICE="${DEVICE:-oneplus}"
source "$ROOT/scripts/env.sh" >/dev/null 2>&1
source "$ROOT/scripts/dssh.sh"
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT/.venv/bin/python"
HTP="$(pwd)/_htp.json"; CFG="$(pwd)/_cfg.json"
NAME=mm_staticw_64
BATCH="${BATCH:-32}"
"$PY" "$ROOT/scripts/mm_int16_staticw_probe.py" "$NAME" "$BATCH"
cd "$NAME"
qairt-converter -i mm.onnx --target_backend HTP \
  --source_model_input_layout A NONTRIVIAL --desired_input_layout A NONTRIVIAL \
  --source_model_output_layout Cout NONTRIVIAL --desired_output_layout Cout NONTRIVIAL \
  --quantization_overrides ovr.json -o mm.dlc >_c.log 2>&1 || { echo CVTFAIL; tail -8 _c.log; exit 1; }
rm -rf ctx
qnn-context-binary-generator --dlc_path mm.dlc --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
   --config_file "$CFG" --save_backend_op_mapping --profiling_level detailed --profiling_option optrace \
   --binary_file mm_ctx --output_dir ctx >_x.log 2>&1 || { echo CTXFAIL; tail -8 _x.log; exit 1; }
for s in *schematic.bin ctx/*schematic.bin; do [ -f "$s" ] && mv -f "$s" ctx/ 2>/dev/null || true; done
W="$(dssh 'echo $HOME/qnn_run')/mmstatic"; dssh "mkdir -p $W"
dssh "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./htp.json\"}}' > $W/cfg.json"
dssh "cat > $W/htp.json" < "$HTP"
dssh_put ctx/mm_ctx.bin "$W/mm_ctx.bin"; dssh_put A.raw "$W/A.raw"
dssh "printf 'A:=A.raw\n' > $W/list.txt"
dssh "cd $W && rm -rf out && LD_LIBRARY_PATH=..:/vendor/lib64 ADSP_LIBRARY_PATH=.. ../qnn-net-run \
   --backend ../libQnnHtp.so --retrieve_context mm_ctx.bin --config_file cfg.json \
   --input_list list.txt --output_dir out --profiling_level detailed --profiling_option optrace --perf_profile burst" >_r.log 2>&1 || true
grep -q 'Finished Executing Graphs' _r.log || { echo RUNFAIL; tail -12 _r.log; exit 1; }
rm -rf out; mkdir -p out; dssh "cd $W && tar cf - out" | tar xf - -C out --strip-components=1 2>/dev/null
"$PY" "$ROOT/scripts/decode_qnn_optrace.py" out --profile-log out/qnn-profiling-data_0.log --schematic "$(ls ctx/*schematic.bin|head -1)" >/dev/null 2>&1
echo "=== STATIC-WEIGHT int16 (BATCH=$BATCH reuse one weight) — per-op + per-64^3 ==="
BATCH="$BATCH" "$PY" - <<'PYEOF'
import json,os,statistics
from collections import Counter,defaultdict
B=int(os.environ.get("BATCH","32"))
s=json.load(open("out/optrace/chrometrace_qnn_htp_analysis_summary.json"))
insts=[]
def walk(o):
    if isinstance(o,dict):
        if "htp_op_instances" in o:
            v=o["htp_op_instances"]; v=v.get("data",v) if isinstance(v,dict) else v
            if isinstance(v,list): insts.extend(v)
        for vv in o.values(): walk(vv)
    elif isinstance(o,list):
        for vv in o: walk(vv)
walk(s)
cnt=Counter(); dom=defaultdict(int)
for it in insts:
    n=it.get("htp_op","?"); cnt[n]+=1; dom[n]+=it.get("num_dominant_path_cycles",0)
for n,_ in sorted(dom.items(),key=lambda kv:-dom[kv[0]]):
    print(f"   {n[:50]:50s} n={cnt[n]:3d} dom={dom[n]:7d}  per64={dom[n]/B:7.0f}")
tot=sum(dom.values())
print(f"   TOTAL sum-dom = {tot}   per-64^3 = {tot/B:.0f}   (cmp dynamic warm 3867/64^3, bare kernel 264)")
conv=sorted(it.get("num_dominant_path_cycles",0) for it in insts if it.get('htp_op')=='q::ConvLayer_s1.opt')
if conv: print(f"   ConvLayer instances={len(conv)} dom min={conv[0]} med={statistics.median(conv):.0f} max={conv[-1]}")
allsc=sorted(it.get("start_cycle",0) for it in insts if it.get("start_cycle"))
print(f"   whole-graph span = {allsc[-1]-allsc[0]}  per-64^3 = {(allsc[-1]-allsc[0])/B:.0f}")
PYEOF
