#!/usr/bin/env bash
# GDN chunk kernel on HTP, FULLY QUANTIZED via the v2.0.0-schema flow (per qnn-native-op-flow skill).
# Mixed precision int16×int8 on every MatMul, params computed in torch (NO qairt calibration):
#   - v2.0.0 override (output_dtype) pins each tensor's datatype; computed by gdn_v2_override.py
#   - 16-bit activation port = int16 SYMMETRIC (uint16 asymmetric breaks the device accumulator)
#   - l2norm internals skipped (let native L2Norm fuse); exp inputs tightly calibrated
#   - deep MatMul in[1] = int8; solve (K<=32) both operands int8 (SFX8×SFX8)
# Result (p00, self-calibrated): oc relerr 1.35e-2 (PASS), S_out 2.5e-2, real v75 HTP.
#
# Usage: ./run_gdn_v2.sh                      # L0 p00, self-calibrated
#        GDN_LAYER=0 TEST_PROMPT=p00 CALIB_PROMPT=p00 ./run_gdn_v2.sh
set -euo pipefail
cd "$(dirname "$0")"
ROOT_DIR="$(cd ../.. && pwd)"
GDN_LAYER="${GDN_LAYER:-0}"; TEST_PROMPT="${TEST_PROMPT:-p00}"; TEST_CHUNK="${TEST_CHUNK:-0}"
CALIB_PROMPT="${CALIB_PROMPT-$TEST_PROMPT}"      # self-calibration by default; set to "" for multi-sample
DEVICE="${DEVICE:-oneplus}"; ARCH="${ARCH:-v75}"; SOC_ID="${SOC_ID:-57}"
GOLDEN="${GOLDEN:-$ROOT_DIR/tests/gdn/golden}"
WORK="quant_v2_L${GDN_LAYER}"
source "$ROOT_DIR/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT_DIR/.venv/bin/python"; K="$ROOT_DIR/scripts/gdn_onnx_kernel.py"
INPUTS=(qc kc vc gc betac S_in cumsum_U sel0 sel1 sel2 sel3 vscale inv_vscale)
mkdir -p "$WORK"; cd "$WORK"
# identity vscale throughout (per-head pre-scale unneeded; the swapped-operand graph + int16 acts suffice)
export GDN_NO_VSCALE=1

echo "[1/6] export quant-path ONNX"
"$PY" "$K" --export-q gdn_q.onnx >/dev/null 2>&1

echo "[2/6] torch-compute v2.0.0 override (int16-sym acts, int8 matmul ports, skip L2Norm internals)"
export GDN_I16_SYM=1
if [ -n "$CALIB_PROMPT" ]; then export GDN_CALIB_PROMPT="$CALIB_PROMPT"; CALN=2; else unset GDN_CALIB_PROMPT; CALN=12; fi
"$PY" "$ROOT_DIR/scripts/gdn_v2_override.py" gdn_q.onnx v2_ovr.json \
  --golden "$GOLDEN" --layer "$GDN_LAYER" --calib "$CALN"

echo "[3/6] convert with v2 override (no qairt calibration)"
LAY=(); for n in "${INPUTS[@]}"; do LAY+=(--source_model_input_layout "$n" NONTRIVIAL --desired_input_layout "$n" NONTRIVIAL); done
for n in oc S_out; do LAY+=(--source_model_output_layout "$n" NONTRIVIAL --desired_output_layout "$n" NONTRIVIAL); done
qairt-converter -i gdn_q.onnx --target_backend HTP "${LAY[@]}" --quantization_overrides v2_ovr.json -o gdn_v2.dlc > _convert.log 2>&1

echo "[4/6] ctxgen + verify HTP-layer MatMul datatypes (no UFX16×UFX16 should remain)"
cat > _htp.json <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":$SOC_ID,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > _cfg.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/_htp.json"}}
EOF
rm -rf ctx
qnn-context-binary-generator --dlc_path gdn_v2.dlc --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --config_file _cfg.json --save_backend_op_mapping --binary_file gdn_v2_ctx --output_dir ctx > _ctxgen.log 2>&1
GB=$(ls ctx/*_bottom_mapping_graph_before.json 2>/dev/null | head -1)
"$PY" - "$GB" <<'PY'
import json,sys,collections
d=json.load(open(sys.argv[1])); t=d["graph"]["tensors"]; DT={776:"SFX8",790:"SFX16",1032:"UFX8",1046:"UFX16"}
c=collections.Counter((DT.get(t.get(n["input_names"][0],{}).get("data_type")),DT.get(t.get(n["input_names"][1],{}).get("data_type"))) for n in d["graph"]["nodes"].values() if n["type"]=="MatMul")
bad=sum(v for k,v in c.items() if k==("UFX16","UFX16"))
print("    MatMul datatype combos:",dict(c)); print(f"    residual UFX16×UFX16: {bad}")
nrm=sum(1 for n in d["graph"]["nodes"].values() if n["type"]=="L2Norm"); print(f"    fused L2Norm ops: {nrm}")
PY

echo "[5/6] emit real golden IO + run on device ($DEVICE)"
GDN_LAYER="$GDN_LAYER" "$PY" - "$ROOT_DIR" "$GOLDEN" "$TEST_PROMPT" "$GDN_LAYER" "$TEST_CHUNK" <<'PY' >/dev/null
import os,sys; sys.path.insert(0,os.path.join(sys.argv[1],"scripts"))
import gdn_onnx_kernel as g
g.emit_golden_io(".", golden_dir=sys.argv[2], prompt=sys.argv[3], layer=int(sys.argv[4]), chunk=int(sys.argv[5]), storage="f4", consts=True)
PY
DEV="$(ssh "$DEVICE" 'echo $HOME/qnn_run')"; W="$DEV/gdn_v2"
ssh "$DEVICE" "mkdir -p $W"
ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./gdn_htp_ext.json\"}}' > $W/gdn_config.json"
ssh "$DEVICE" "cat > $W/gdn_htp_ext.json" < _htp.json
ssh "$DEVICE" "cat > $W/gdn_v2_ctx.bin" < ctx/gdn_v2_ctx.bin
for n in "${INPUTS[@]}"; do ssh "$DEVICE" "cat > $W/$n.raw" < "$n.raw"; done
ssh "$DEVICE" "cat > $W/input_list.txt" < input_list.txt
ssh "$DEVICE" "cd $W && rm -rf out && LD_LIBRARY_PATH=..:/vendor/lib64 ADSP_LIBRARY_PATH=.. ../qnn-net-run \
    --backend ../libQnnHtp.so --retrieve_context gdn_v2_ctx.bin --config_file gdn_config.json \
    --input_list input_list.txt --output_dir out --perf_profile burst" > _run.log 2>&1 || true
grep -q 'Finished Executing Graphs' _run.log || { echo "  [FAIL] device run"; tail -12 _run.log; exit 1; }
rm -rf device_out; mkdir -p device_out; ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C device_out --strip-components=1

echo "[6/6] compare quantized HTP output to fp64 reference (L${GDN_LAYER} $TEST_PROMPT)"
GDN_NATIVE_TOL="${GDN_NATIVE_TOL:-1.5e-2}" "$PY" "$K" --compare . --result device_out
