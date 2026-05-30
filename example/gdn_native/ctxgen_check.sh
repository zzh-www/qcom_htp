#!/usr/bin/env bash
# Host-only: export -> convert -> quantize(symmetric int16) -> ctxgen. No device.
# Reports whether the fully-quantized graph COMPOSES for HTP (the validation gate).
set -euo pipefail
cd "$(dirname "$0")"
ROOT_DIR="$(cd ../.. && pwd)"
ACT_BW="${ACT_BW:-16}"; W_BW="${W_BW:-16}"; GDN_LAYER="${GDN_LAYER:-0}"
ARCH="${ARCH:-v75}"; SOC_ID="${SOC_ID:-57}"; GOLDEN="${GOLDEN:-$ROOT_DIR/tests/gdn/golden}"
source "$ROOT_DIR/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT_DIR/.venv/bin/python"; K="$ROOT_DIR/scripts/gdn_onnx_kernel.py"
INPUTS=(qc kc vc gc betac S_in cumsum_U sel0 sel1 sel2 sel3)
W="ctxcheck_w${W_BW}a${ACT_BW}"; mkdir -p "$W"; cd "$W"

"$PY" "$K" --export-q gdn_chunk.onnx >/dev/null 2>&1
GDN_LAYER="$GDN_LAYER" "$PY" - "$ROOT_DIR" <<PY >/dev/null 2>&1
import os,sys; sys.path.insert(0, os.path.join(sys.argv[1],"scripts"))
import gdn_onnx_kernel as g; g.emit_calib("calib", golden_dir="$GOLDEN", layer=$GDN_LAYER, consts=True)
PY
"$PY" - "$ROOT_DIR" <<'PY' >/dev/null 2>&1
import os,sys; sys.path.insert(0, os.path.join(sys.argv[1],"scripts"))
import gdn_onnx_kernel as g; g.write_quant_overrides("quant_overrides.json")
PY
LAY=(); for n in "${INPUTS[@]}"; do LAY+=(--source_model_input_layout "$n" NONTRIVIAL --desired_input_layout "$n" NONTRIVIAL); done
for n in oc S_out; do LAY+=(--source_model_output_layout "$n" NONTRIVIAL --desired_output_layout "$n" NONTRIVIAL); done
sed 's#:=#:=calib/#g' calib/calib_list.txt > calib_list_abs.txt
# Pass 1: convert + calibrate (asymmetric) just to DUMP per-tensor min/max ranges.
qairt-converter -i gdn_chunk.onnx --target_backend HTP "${LAY[@]}" -o gdn_float.dlc > _convert.log 2>&1 \
  || { echo "CONVERT FAIL"; grep -iE "error|keyerror|not.*support" _convert.log | tail -5; exit 1; }
qairt-quantizer --input_dlc gdn_float.dlc --input_list calib_list_abs.txt \
    --act_bitwidth "$ACT_BW" --weights_bitwidth "$W_BW" --bias_bitwidth 32 \
    --dump_encoding_json --output_dlc gdn_cal.dlc > _calib.log 2>&1 || { echo "CALIB FAIL"; tail -5 _calib.log; exit 1; }
# Build force-SYMMETRIC overrides (offset -32768) — HTP int16 MatMul needs symmetric operands;
# qairt gives offset 0 to non-negative tensors otherwise.
"$PY" - "$ROOT_DIR" <<'PY'
import os,sys; sys.path.insert(0, os.path.join(sys.argv[1],"scripts"))
import gdn_onnx_kernel as g; g.symmetric_overrides_from_dump("gdn_cal_encoding.json","sym_overrides.json")
PY
# Pass 2: re-convert WITH the symmetric overrides, then finalize (encoding-driven, all-int).
qairt-converter -i gdn_chunk.onnx --target_backend HTP --quantization_overrides sym_overrides.json \
    "${LAY[@]}" -o gdn_enc.dlc > _convert2.log 2>&1 || { echo "CONVERT2 FAIL"; tail -8 _convert2.log; exit 1; }
qairt-quantizer --input_dlc gdn_enc.dlc --input_list calib_list_abs.txt \
    --act_bitwidth "$ACT_BW" --weights_bitwidth "$W_BW" --bias_bitwidth 32 \
    --output_dlc gdn_quant.dlc > _quantize.log 2>&1 || { echo "QUANTIZE FAIL"; tail -5 _quantize.log; exit 1; }
cat > _htp_ext.json <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":$SOC_ID,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > _config_host.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/_htp_ext.json"}}
EOF
rm -rf ctx
qnn-context-binary-generator --dlc_path gdn_quant.dlc \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --config_file _config_host.json --binary_file gdn_quant_ctx --output_dir ctx > _ctxgen.log 2>&1 || true
if ls ctx/*.bin >/dev/null 2>&1; then
    echo "CTXGEN OK -> $W/ctx/$(ls ctx | grep bin)"
else
    echo "CTXGEN FAIL:"
    grep -iE "Failed to validate op|incorrect Value|expected equal|Please refer" _ctxgen.log | head -3
fi
