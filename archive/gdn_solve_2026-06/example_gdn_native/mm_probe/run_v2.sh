#!/usr/bin/env bash
# Single-MatMul probe, v2.0.0-schema flow (torch-computed params, NO qairt calibration):
#   emit onnx -> torch-compute scale/zp + write v2 override (B=int8-sym, A/Y=uint16-asym)
#   -> converter eats v2 override (auto-triggers quantize-v2) -> DLC -> ctxgen -> HTP op layer
# Goal: confirm v2 output_dtype makes the dynamic activation B compile as SFX8 at the HTP layer.
set -euo pipefail
cd "$(dirname "$0")"
KIND="${KIND:-actact}"; I8="${I8:-B}"; ARCH="${ARCH:-v75}"; SOC_ID="${SOC_ID:-57}"
ROOT_DIR="$(cd ../../.. && pwd)"
source "$ROOT_DIR/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT_DIR/.venv/bin/python"; MP="$ROOT_DIR/scripts/mm_probe.py"
WORK="v2_${KIND}"; rm -rf "$WORK"; mkdir -p "$WORK"
"$PY" "$MP" --emit "$WORK" --kind "$KIND"
echo "[torch] compute quant params + write v2.0.0 override (i8=$I8)"
"$PY" "$MP" --write-v2 "$WORK" --i8 "$I8"
cd "$WORK"
if [ "$KIND" = weight ]; then INPUTS=(A); else INPUTS=(A B); fi
LAY=(); for n in "${INPUTS[@]}"; do LAY+=(--source_model_input_layout "$n" NONTRIVIAL --desired_input_layout "$n" NONTRIVIAL); done
LAY+=(--source_model_output_layout Y NONTRIVIAL --desired_output_layout Y NONTRIVIAL)

echo "[convert] qairt-converter with v2 override (no calibration)"
qairt-converter -i mm.onnx --target_backend HTP "${LAY[@]}" \
    --quantization_overrides v2_ovr.json -o mm_enc.dlc > _c.log 2>&1 || { echo "CONVERT FAIL"; tail -15 _c.log; exit 1; }

echo "[check-DLC] mm_enc.dlc datatypes:"
qairt-dlc-to-json -i mm_enc.dlc -o _enc.json >/dev/null 2>&1 || true
"$PY" - <<'PY'
import json
try: d=json.load(open("_enc.json"))
except Exception as e: print("  (dlc-to-json failed:",e,")"); raise SystemExit
t=d["graph"]["tensors"]
DT={776:"SFX8",790:"SFX16",1032:"UFX8",1046:"UFX16",818:"SFX32",562:"FP32"}
for nm,n in d["graph"]["nodes"].items():
    if n["type"]=="MatMul":
        for k in n["input_names"]+n["output_names"]:
            x=t.get(k,{}); so=x.get("quant_params",{}).get("scale_offset",{})
            print(f"   {k}: dt={DT.get(x.get('data_type'),x.get('data_type'))} bw={so.get('bitwidth')} off={so.get('offset')} sym={so.get('is_symmetric')}")
PY

echo "[ctxgen] -> HTP op layer"
cat > _htp.json <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":$SOC_ID,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > _cfg.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/_htp.json"}}
EOF
rm -rf ctx
if qnn-context-binary-generator --dlc_path mm_enc.dlc \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --config_file _cfg.json --save_backend_op_mapping --binary_file mm_ctx --output_dir ctx > _ctx.log 2>&1; then
  echo "[check-HTP] HTP op-layer MatMul datatype:"
  GB=$(ls ctx/*_bottom_mapping_graph_before.json 2>/dev/null | head -1)
  "$PY" "$MP" --check "$GB"
else
  echo "  ctxgen FAIL:"; grep -iE 'error|fail|valid' _ctx.log | head -8
fi
