#!/usr/bin/env bash
# Single-MatMul probe, FULL-OVERRIDE / NO-CALIBRATION flow (per user correction):
#   pass-1  = calibrate ONCE only to harvest per-tensor min/max
#   rewrite = write a COMPLETE override for every tensor (A=int16 asym, B=int8 sym, Y=int16 asym)
#   pass-2  = converter eats the complete override -> fully-quantized DLC, NO quantizer calibration
#   then read the converter-output DLC AND the HTP op layer.
# Goal: see if B truly becomes int8 once the flow is clean (no global --act_bitwidth pollution).
set -euo pipefail
cd "$(dirname "$0")"
KIND="${KIND:-actact}"; ARCH="${ARCH:-v75}"; SOC_ID="${SOC_ID:-57}"
ROOT_DIR="$(cd ../../.. && pwd)"
source "$ROOT_DIR/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT_DIR/.venv/bin/python"; MP="$ROOT_DIR/scripts/mm_probe.py"
I8_TENSOR="${I8_TENSOR:-B}"                       # which tensor to force int8-symmetric
WORK="full_${KIND}"; rm -rf "$WORK"; mkdir -p "$WORK"
"$PY" "$MP" --emit "$WORK" --kind "$KIND"
cd "$WORK"
if [ "$KIND" = weight ]; then INPUTS=(A); else INPUTS=(A B); fi
LAY=(); for n in "${INPUTS[@]}"; do LAY+=(--source_model_input_layout "$n" NONTRIVIAL --desired_input_layout "$n" NONTRIVIAL); done
LAY+=(--source_model_output_layout Y NONTRIVIAL --desired_output_layout Y NONTRIVIAL)

echo "[pass-1] calibrate once -> dump per-tensor ranges"
qairt-converter -i mm.onnx --target_backend HTP "${LAY[@]}" -o mm_p1.dlc > _c1.log 2>&1
qairt-quantizer --input_dlc mm_p1.dlc --input_list calib_list.txt \
    --act_bitwidth 16 --weights_bitwidth 8 --bias_bitwidth 32 \
    --dump_encoding_json --output_dlc mm_cal.dlc > _q1.log 2>&1

echo "[rewrite] complete override: every tensor; $I8_TENSOR -> int8-sym, rest int16-asym"
I8="$I8_TENSOR" "$PY" - <<'PY'
import json, os
d = json.load(open("mm_cal_encoding.json"))
i8 = os.environ["I8"]
def enc16(e):                                   # int16 ASYMMETRIC: keep calibrated min/max/offset
    e = e[0] if isinstance(e, list) else e
    mn, mx = float(e.get("min", 0.0)), float(e.get("max", 0.0))
    rng = (mx - mn) or 1.0; scale = rng / 65535.0
    off = int(round(-mn / scale)); off = max(0, min(65535, off))
    return [{"bitwidth": 16, "dtype": "int", "is_symmetric": "False",
             "min": mn, "max": mx, "scale": scale, "offset": -off}]
def enc8(e):                                    # int8 SYMMETRIC
    e = e[0] if isinstance(e, list) else e
    a = max(abs(float(e.get("min", 0.0))), abs(float(e.get("max", 0.0)))) or 1.0
    return [{"bitwidth": 8, "dtype": "int", "is_symmetric": "True",
             "min": -a, "max": a, "scale": a / 127.0, "offset": -128}]
out = {"activation_encodings": {}, "param_encodings": {}}
for grp in ("activation_encodings", "param_encodings"):
    for name, e in d.get(grp, {}).items():
        if name.endswith("_converted_unsigned_symmetric"):
            continue
        out[grp][name] = enc8(e) if name == i8 else enc16(e)
json.dump(out, open("full_ovr.json", "w"), indent=1)
print("  tensors:", {g: list(out[g].keys()) for g in out})
PY

echo "[pass-2] converter with COMPLETE override, NO quantizer calibration"
qairt-converter -i mm.onnx --target_backend HTP "${LAY[@]}" \
    --quantization_overrides full_ovr.json -o mm_enc.dlc > _c2.log 2>&1

echo "[check-DLC] converter output (mm_enc.dlc), pre-ctxgen:"
qairt-dlc-to-json -i mm_enc.dlc -o _enc.json >/dev/null 2>&1
"$PY" - <<'PY'
import json
d=json.load(open("_enc.json")); t=d["graph"]["tensors"]
DT={776:"SFX8",790:"SFX16",1032:"UFX8",1046:"UFX16",818:"SFX32",562:"FP32"}
for nm,n in d["graph"]["nodes"].items():
    if n["type"]=="MatMul":
        for k in n["input_names"]+n["output_names"]:
            x=t.get(k,{}); so=x.get("quant_params",{}).get("scale_offset",{})
            print(f"   {k}: dt={DT.get(x.get('data_type'),x.get('data_type'))} bw={so.get('bitwidth')} off={so.get('offset')} sym={so.get('is_symmetric')}")
PY

echo "[ctxgen] from mm_enc.dlc directly (no quantizer step)"
cat > _htp.json <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":$SOC_ID,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > _cfg.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/_htp.json"}}
EOF
rm -rf ctx
if qnn-context-binary-generator --dlc_path mm_enc.dlc \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --config_file _cfg.json --save_backend_op_mapping \
    --binary_file mm_ctx --output_dir ctx > _ctx.log 2>&1; then
  echo "[check-HTP] HTP op-layer datatype:"
  GB=$(ls ctx/*_bottom_mapping_graph_before.json 2>/dev/null | head -1)
  "$PY" "$MP" --check "$GB"
else
  echo "  ctxgen FAIL:"; grep -iE 'error|fail|valid' _ctx.log | head -6
fi
