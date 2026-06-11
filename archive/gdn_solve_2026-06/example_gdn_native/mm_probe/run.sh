#!/usr/bin/env bash
# Minimal single-MatMul HTP datatype probe.
#   KIND=weight|actact   ovr=<override.json>   ACT_BW=16  W_BW=8
# Goal: see what datatype the HTP backend actually compiles the MatMul to (UFX16 x SFX8 == win).
set -euo pipefail
cd "$(dirname "$0")"
KIND="${KIND:-weight}"; ACT_BW="${ACT_BW:-16}"; W_BW="${W_BW:-8}"; OVR="${OVR:-}"
ARCH="${ARCH:-v75}"; SOC_ID="${SOC_ID:-57}"
ROOT_DIR="$(cd ../../.. && pwd)"
source "$ROOT_DIR/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT_DIR/.venv/bin/python"; MP="$ROOT_DIR/scripts/mm_probe.py"
WORK="work_${KIND}${OVR:+_ovr}"
rm -rf "$WORK"; mkdir -p "$WORK"
"$PY" "$MP" --emit "$WORK" --kind "$KIND"
cd "$WORK"
if [ "$KIND" = weight ]; then INPUTS=(A); else INPUTS=(A B); fi
LAY=(); for n in "${INPUTS[@]}"; do LAY+=(--source_model_input_layout "$n" NONTRIVIAL --desired_input_layout "$n" NONTRIVIAL); done
LAY+=(--source_model_output_layout Y NONTRIVIAL --desired_output_layout Y NONTRIVIAL)

echo "[1] convert"
OVRFLAG=(); [ -n "$OVR" ] && OVRFLAG=(--quantization_overrides "../$OVR")
qairt-converter -i mm.onnx --target_backend HTP "${LAY[@]}" "${OVRFLAG[@]}" -o mm_float.dlc > _c.log 2>&1
echo "[2] quantize (act_bw=$ACT_BW w_bw=$W_BW${OVR:+, ovr=$OVR})"
qairt-quantizer --input_dlc mm_float.dlc --input_list calib_list.txt \
    --act_bitwidth "$ACT_BW" --weights_bitwidth "$W_BW" --bias_bitwidth 32 \
    --output_dlc mm_quant.dlc > _q.log 2>&1
echo "[3] ctxgen (+backend op mapping)"
cat > _htp.json <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":$SOC_ID,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > _cfg.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/_htp.json"}}
EOF
rm -rf ctx
qnn-context-binary-generator --dlc_path mm_quant.dlc \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --config_file _cfg.json --save_backend_op_mapping \
    --binary_file mm_ctx --output_dir ctx > _ctx.log 2>&1 || { echo "  ctxgen FAIL"; tail -8 _ctx.log; exit 1; }
echo "[4] HTP-layer MatMul datatype:"
GB=$(ls ctx/*_bottom_mapping_graph_before.json 2>/dev/null | head -1)
"$PY" "$MP" --check "$GB"
echo "    DLC-layer (for contrast):"
qairt-dlc-to-json -i mm_quant.dlc -o _qjson.json >/dev/null 2>&1
"$PY" - <<PY
import json
d=json.load(open("_qjson.json")); t=d["graph"]["tensors"]
DT={8:"i8",16:"i16",1032:"UFX8",1046:"UFX16",776:"SFX8",790:"SFX16"}
for nm,n in d["graph"]["nodes"].items():
    if n.get("type")=="MatMul":
        for k in n["input_names"]:
            so=t.get(k,{}).get("quant_params",{}).get("scale_offset",{})
            print(f"      {k}: bw={so.get('bitwidth')} off={so.get('offset')} sym={so.get('is_symmetric')}")
PY
