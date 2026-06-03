#!/usr/bin/env bash
# Enumerate MatMul (act×act, K=128) precision configs. For each: show the encoding (QNN-calibrated
# OR override), whether ctxgen succeeds, and the on-device numerical relerr vs fp32 A@B.
set -uo pipefail
cd "$(dirname "$0")"
ROOT_DIR="$(cd ../../.. && pwd)"
DEVICE="${DEVICE:-oneplus}"; ARCH="${ARCH:-v75}"; SOC_ID="${SOC_ID:-57}"
source "$ROOT_DIR/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT_DIR/.venv/bin/python"; MP="$ROOT_DIR/scripts/mm_probe.py"
W=mm_enum; rm -rf "$W"; mkdir -p "$W"
"$PY" "$MP" --emit "$W" --kind actact >/dev/null
cd "$W"
LAY=(--source_model_input_layout A NONTRIVIAL --desired_input_layout A NONTRIVIAL
     --source_model_input_layout B NONTRIVIAL --desired_input_layout B NONTRIVIAL
     --source_model_output_layout Y NONTRIVIAL --desired_output_layout Y NONTRIVIAL)
cat > _htp.json <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":$SOC_ID,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > _cfg.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/_htp.json"}}
EOF
DEV="$(ssh "$DEVICE" 'echo $HOME/qnn_run')"; RW="$DEV/mm_test"
ssh "$DEVICE" "mkdir -p $RW"
ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./htp.json\"}}' > $RW/cfg.json"
ssh "$DEVICE" "cat > $RW/htp.json" < _htp.json
ssh "$DEVICE" "cat > $RW/A.raw" < A.raw; ssh "$DEVICE" "cat > $RW/B.raw" < B.raw
ssh "$DEVICE" "cat > $RW/test_list.txt" < test_list.txt
sed 's#:=#:=#g' calib_list.txt > calib_abs.txt

dlc_enc () {  # summarize in0/in1/out datatype+offset from a dlc
  qairt-dlc-to-json -i "$1" -o _e.json >/dev/null 2>&1 || { echo "(dlc-to-json failed)"; return; }
  "$PY" - <<'PY'
import json
d=json.load(open("_e.json")); t=d["graph"]["tensors"]
DT={776:"SFX8",790:"SFX16",818:"SFX32",1032:"UFX8",1046:"UFX16",562:"FP32"}
for nm,n in d["graph"]["nodes"].items():
    if n["type"]=="MatMul":
        for lbl,k in [("in0",n["input_names"][0]),("in1",n["input_names"][1]),("out",n["output_names"][0])]:
            x=t.get(k,{}); so=x.get("quant_params",{}).get("scale_offset",{})
            print(f"    {lbl}={DT.get(x.get('data_type'),x.get('data_type'))} off={so.get('offset')} sym={so.get('is_symmetric')}",end="")
        print()
PY
}
device_relerr () {  # ctxgen the dlc in $1, run, print relerr; echoes ctxgen status
  rm -rf ctx
  if ! qnn-context-binary-generator --dlc_path "$1" --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" --config_file _cfg.json --binary_file mm_ctx --output_dir ctx > _ctx.log 2>&1; then
    echo "    ctxgen: FAIL -> $(grep -iE 'incorrect|expected|error|valid' _ctx.log | head -1 | sed 's/^ *//')"; return
  fi
  ssh "$DEVICE" "cat > $RW/mm_ctx.bin" < ctx/mm_ctx.bin
  ssh "$DEVICE" "cd $RW && rm -rf out && LD_LIBRARY_PATH=..:/vendor/lib64 ADSP_LIBRARY_PATH=.. ../qnn-net-run --backend ../libQnnHtp.so --retrieve_context mm_ctx.bin --config_file cfg.json --input_list test_list.txt --output_dir out --perf_profile burst" > _run.log 2>&1 || true
  rm -rf out; ssh "$DEVICE" "cd $RW && tar cf - out" | tar xf - -C . 2>/dev/null
  "$PY" - <<'PY'
import numpy as np, glob
ref=np.fromfile("Y_ref.raw","<f4"); f=glob.glob("out/**/Y*.raw",recursive=True)
if not f: print("    ctxgen: OK   device: NO OUTPUT"); raise SystemExit
raw=np.fromfile(f[0],np.uint8); dev=(raw.view('<f2') if raw.size==ref.size*2 else raw.view('<f4')).astype(np.float32)
print(f"    ctxgen: OK   device relerr={np.linalg.norm(dev-ref)/np.linalg.norm(ref):.4f}")
PY
}

echo "########## A) QNN's OWN calibration ##########"
qairt-converter -i mm.onnx --target_backend HTP "${LAY[@]}" -o mm_f.dlc > /dev/null 2>&1
for BW in "16 16" "8 8"; do
  set -- $BW; AB=$1; WB=$2
  qairt-quantizer --input_dlc mm_f.dlc --input_list calib_abs.txt --act_bitwidth $AB --weights_bitwidth $WB --bias_bitwidth 32 --output_dlc mm_q.dlc > /dev/null 2>&1
  echo "[QNN calib act$AB/weight$WB]"; dlc_enc mm_q.dlc; device_relerr mm_q.dlc
done

echo "########## B) override-specified legal configs (in1 int8-sym) ##########"
for MODE in int16sym uint16mid uint16asym; do
  IN0_MODE=$MODE "$PY" "$MP" --write-v2 . --i8 B >/dev/null 2>&1
  qairt-converter -i mm.onnx --target_backend HTP "${LAY[@]}" --quantization_overrides v2_ovr.json -o mm_o.dlc > /dev/null 2>&1
  echo "[in0=$MODE × in1=int8-sym]"; dlc_enc mm_o.dlc; device_relerr mm_o.dlc
done