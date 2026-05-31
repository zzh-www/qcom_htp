#!/usr/bin/env bash
# Probe HTP's quantized exp / l2norm(rsqrt) in isolation on real GDN ranges.
# Builds a minimal int16 graph (exp(g), l2norm(x)), runs it on HTP, and reports whether the
# device output matches exp-of-quantized-input (sim-faithful) or diverges (HTP LUT error).
set -euo pipefail
cd "$(dirname "$0")"
ROOT_DIR="$(cd ../.. && pwd)"
ARCH="${ARCH:-v75}"; SOC_ID="${SOC_ID:-57}"; DEVICE="${DEVICE:-oneplus}"
GOLDEN="${GOLDEN:-$ROOT_DIR/tests/gdn/golden}"; LAYER="${GDN_LAYER:-0}"; PROMPT="${TEST_PROMPT:-p00}"
source "$ROOT_DIR/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT_DIR/.venv/bin/python"; K="$ROOT_DIR/scripts/gdn_probe_ops.py"
W="probe_htp"; mkdir -p "$W"; cd "$W"

echo "[1] export probe + emit real IO + calib"
"$PY" "$K" --export probe.onnx --golden "$GOLDEN" --layer "$LAYER" --prompt "$PROMPT" >/dev/null 2>&1
"$PY" "$K" --emit-io . --golden "$GOLDEN" --layer "$LAYER" --prompt "$PROMPT"
LAY=(--source_model_input_layout g NONTRIVIAL --desired_input_layout g NONTRIVIAL
     --source_model_input_layout x NONTRIVIAL --desired_input_layout x NONTRIVIAL
     --source_model_output_layout exp_g NONTRIVIAL --desired_output_layout exp_g NONTRIVIAL
     --source_model_output_layout l2_x NONTRIVIAL --desired_output_layout l2_x NONTRIVIAL)

echo "[2] two-pass symmetric int16 quantize"
qairt-converter -i probe.onnx --target_backend HTP "${LAY[@]}" -o p_float.dlc > _c1.log 2>&1
qairt-quantizer --input_dlc p_float.dlc --input_list calib_list.txt --act_bitwidth 16 \
    --weights_bitwidth 16 --bias_bitwidth 32 --dump_encoding_json --output_dlc p_cal.dlc > _q1.log 2>&1
"$PY" - "$ROOT_DIR" <<'PY' >/dev/null
import os,sys; sys.path.insert(0, os.path.join(sys.argv[1],"scripts"))
import gdn_onnx_kernel as g; g.symmetric_overrides_from_dump("p_cal_encoding.json","sym.json")
PY
qairt-converter -i probe.onnx --target_backend HTP --quantization_overrides sym.json "${LAY[@]}" -o p_enc.dlc > _c2.log 2>&1
qairt-quantizer --input_dlc p_enc.dlc --input_list calib_list.txt --act_bitwidth 16 \
    --weights_bitwidth 16 --bias_bitwidth 32 --output_dlc p_quant.dlc > _q2.log 2>&1
# read the symmetric input scales (g, x) from the override we generated (= the applied scales)
read -r GSC XSC < <("$PY" - <<'PY'
import json
d=json.load(open("sym.json"))["activation_encodings"]
def sc(n):
    e=d[n]; e=e[0] if isinstance(e,list) else e; return e["scale"]
print(sc("g"), sc("x"))
PY
)
echo "    input scales: g=$GSC  x=$XSC"

echo "[3] ctxgen + device run"
cat > _htp.json <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":$SOC_ID,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > _cfg.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/_htp.json"}}
EOF
rm -rf ctx dev_out
qnn-context-binary-generator --dlc_path p_quant.dlc --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --config_file _cfg.json --binary_file p_ctx --output_dir ctx > _ctx.log 2>&1
[ -f ctx/p_ctx.bin ] || { echo "ctxgen FAIL"; tail -4 _ctx.log; exit 1; }
DEV="$(ssh "$DEVICE" 'echo $HOME/qnn_run')"; R="$DEV/gdn_probe"
ssh "$DEVICE" "mkdir -p $R"
ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./_htp.json\"}}' > $R/_cfg.json"
ssh "$DEVICE" "cat > $R/_htp.json" < _htp.json
ssh "$DEVICE" "cat > $R/p_ctx.bin" < ctx/p_ctx.bin
for n in g x; do ssh "$DEVICE" "cat > $R/$n.raw" < "$n.raw"; done
ssh "$DEVICE" "cat > $R/input_list.txt" < input_list.txt
ssh "$DEVICE" "cd $R && rm -rf out && LD_LIBRARY_PATH=..:/vendor/lib64 ADSP_LIBRARY_PATH=.. ../qnn-net-run \
    --backend ../libQnnHtp.so --retrieve_context p_ctx.bin --config_file _cfg.json \
    --input_list input_list.txt --output_dir out" > _run.log 2>&1 || true
grep -q 'Finished Executing Graphs' _run.log || { echo "device run FAIL"; tail -10 _run.log; exit 1; }
mkdir -p dev_out; ssh "$DEVICE" "cd $R && tar cf - out" | tar xf - -C dev_out --strip-components=1

echo "[4] verdict (htp vs exact vs exp-of-quantized-input)"
"$PY" "$K" --compare . --result dev_out --gscale "$GSC" --xscale "$XSC"
