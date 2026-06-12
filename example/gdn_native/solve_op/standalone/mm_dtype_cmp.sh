#!/usr/bin/env bash
# Compare native ConvLayer dominant-path for u8i8 vs u16i16 at 64^3 (batched 32) — settles whether the
# 264 int16 dominant-path is the FULL byte-pass cost (expect int16 ≈ 4-6x u8i8) or undercounts.
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
cat > "$HTP" <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":57,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > "$CFG" <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$HTP"}}
EOF
for DT in u8i8 u16i16; do
  NAME="mm_cmp_$DT"
  "$PY" "$ROOT/scripts/mm_int16_probe.py" "$NAME" "1,32,64,64" "1,32,64,64" "$DT"
  ( cd "$NAME"
    LAY=(); for n in A B; do LAY+=(--source_model_input_layout "$n" NONTRIVIAL --desired_input_layout "$n" NONTRIVIAL); done
    LAY+=(--source_model_output_layout Cout NONTRIVIAL --desired_output_layout Cout NONTRIVIAL)
    qairt-converter -i mm.onnx --target_backend HTP "${LAY[@]}" --quantization_overrides ovr.json -o mm.dlc >_c.log 2>&1 || { echo "  $DT CVTFAIL"; tail -5 _c.log; exit 1; }
    rm -rf ctx
    qnn-context-binary-generator --dlc_path mm.dlc --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
       --config_file "$CFG" --save_backend_op_mapping --profiling_level detailed --profiling_option optrace \
       --binary_file mm_ctx --output_dir ctx >_x.log 2>&1 || { echo "  $DT CTXFAIL"; tail -5 _x.log; exit 1; }
    for s in *schematic.bin ctx/*schematic.bin; do [ -f "$s" ] && mv -f "$s" ctx/ 2>/dev/null || true; done
    W="$(dssh 'echo $HOME/qnn_run')/mmcmp"; dssh "mkdir -p $W"
    dssh "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./htp.json\"}}' > $W/cfg.json"
    dssh "cat > $W/htp.json" < "$HTP"
    dssh_put ctx/mm_ctx.bin "$W/mm_ctx.bin"; dssh_put A.raw "$W/A.raw"; dssh_put B.raw "$W/B.raw"
    dssh "printf 'A:=A.raw B:=B.raw\n' > $W/list.txt"
    dssh "cd $W && rm -rf out && LD_LIBRARY_PATH=..:/vendor/lib64 ADSP_LIBRARY_PATH=.. ../qnn-net-run \
       --backend ../libQnnHtp.so --retrieve_context mm_ctx.bin --config_file cfg.json \
       --input_list list.txt --output_dir out --profiling_level detailed --profiling_option optrace --perf_profile burst" >_r.log 2>&1 || true
    grep -q 'Finished Executing Graphs' _r.log || { echo "  $DT RUNFAIL"; tail -10 _r.log; exit 1; }
    rm -rf out; mkdir -p out; dssh "cd $W && tar cf - out" | tar xf - -C out --strip-components=1 2>/dev/null
    "$PY" "$ROOT/scripts/decode_qnn_optrace.py" out --profile-log out/qnn-profiling-data_0.log --schematic "$(ls ctx/*schematic.bin|head -1)" >/dev/null 2>&1 ) || continue
  DT="$DT" "$PY" - "$NAME" <<'PYEOF'
import json,sys,os,statistics
name=sys.argv[1]; dt=os.environ["DT"]
s=json.load(open(f"{name}/out/optrace/chrometrace_qnn_htp_analysis_summary.json"))
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
conv=sorted(it.get("num_dominant_path_cycles",0) for it in insts if it.get("htp_op")=="q::ConvLayer_s1.opt")
if conv:
    print(f">>> {dt}: native ConvLayer_s1.opt dominant-path 64^3  median={statistics.median(conv):.0f}  min={conv[0]} (n={len(conv)})")
PYEOF
done