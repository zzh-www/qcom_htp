#!/usr/bin/env bash
# Fully-quantized (all-integer, NO fp16) QNN-native GDN chunk kernel on HTP.
#
# Every tensor is int16: the 8 GEMMs, Exp, Sqrt/rsqrt l2norm, and the (I-A)^-1 solve all run as
# QNN quantized HTP kernels (no --enable_float_fallback). Validated on real Qwen3.5-4B chunks.
#
# HTP int16 MatMul needs (a) 32-aligned dims (the solve uses 16-logical / 32-physical padded
# blocks), (b) activation×activation MatMul not FullyConnected (the cumsum / block-selector
# constants are fed as runtime INPUTS, not weights), and (c) SYMMETRIC operands (offset -32768).
# qairt assigns offset 0 to non-negative tensors, so we calibrate once, dump ranges, rewrite
# every encoding symmetric, then re-convert encoding-driven.
#
# Usage:   ./run_gdn_native_quant.sh                          # int16, L00, real golden chunk
#          GDN_LAYER=0 TEST_PROMPT=p29 TEST_CHUNK=1 ./run_gdn_native_quant.sh
set -euo pipefail
cd "$(dirname "$0")"
ROOT_DIR="$(cd ../.. && pwd)"
ACT_BW="${ACT_BW:-16}"; W_BW="${W_BW:-16}"
GDN_LAYER="${GDN_LAYER:-0}"; TEST_PROMPT="${TEST_PROMPT:-p00}"; TEST_CHUNK="${TEST_CHUNK:-0}"
DEVICE="${DEVICE:-oneplus}"; ARCH="${ARCH:-v75}"; SOC_ID="${SOC_ID:-57}"
GOLDEN="${GOLDEN:-$ROOT_DIR/tests/gdn/golden}"
WORK="quant_w${W_BW}a${ACT_BW}_L${GDN_LAYER}"

source "$ROOT_DIR/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT_DIR/.venv/bin/python"; K="$ROOT_DIR/scripts/gdn_onnx_kernel.py"
INPUTS=(qc kc vc gc betac S_in cumsum_U sel0 sel1 sel2 sel3 vscale inv_vscale)
mkdir -p "$WORK"; cd "$WORK"

echo "[1/7] export quantized-path ONNX (const inputs) + calib set + real test chunk"
"$PY" "$K" --export-q gdn_chunk.onnx >/dev/null 2>&1
GDN_LAYER="$GDN_LAYER" "$PY" - "$ROOT_DIR" <<PY >/dev/null
import os,sys; sys.path.insert(0, os.path.join(sys.argv[1],"scripts"))
import gdn_onnx_kernel as g
g.emit_calib("calib", golden_dir="$GOLDEN", layer=$GDN_LAYER, consts=True)
g.emit_golden_io(".", golden_dir="$GOLDEN", prompt="$TEST_PROMPT", layer=$GDN_LAYER, chunk=$TEST_CHUNK, storage="f4", consts=True)
PY

LAY=(); for n in "${INPUTS[@]}"; do LAY+=(--source_model_input_layout "$n" NONTRIVIAL --desired_input_layout "$n" NONTRIVIAL); done
for n in oc S_out; do LAY+=(--source_model_output_layout "$n" NONTRIVIAL --desired_output_layout "$n" NONTRIVIAL); done
sed 's#:=#:=calib/#g' calib/calib_list.txt > calib_list_abs.txt

echo "[2/7] pass-1 convert + calibrate -> dump per-tensor ranges"
qairt-converter -i gdn_chunk.onnx --target_backend HTP "${LAY[@]}" -o gdn_float.dlc > _convert.log 2>&1
qairt-quantizer --input_dlc gdn_float.dlc --input_list calib_list_abs.txt \
    --act_bitwidth "$ACT_BW" --weights_bitwidth "$W_BW" --bias_bitwidth 32 \
    --dump_encoding_json --output_dlc gdn_cal.dlc > _calib.log 2>&1

echo "[3/7] force every encoding SYMMETRIC (HTP int16 MatMul operand requirement)"
"$PY" - "$ROOT_DIR" <<'PY' >/dev/null
import os,sys; sys.path.insert(0, os.path.join(sys.argv[1],"scripts"))
import gdn_onnx_kernel as g; g.symmetric_overrides_from_dump("gdn_cal_encoding.json","sym_overrides.json")
PY

echo "[4/7] pass-2 re-convert with symmetric overrides + finalize (all-int)"
qairt-converter -i gdn_chunk.onnx --target_backend HTP --quantization_overrides sym_overrides.json \
    "${LAY[@]}" -o gdn_enc.dlc > _convert2.log 2>&1
qairt-quantizer --input_dlc gdn_enc.dlc --input_list calib_list_abs.txt \
    --act_bitwidth "$ACT_BW" --weights_bitwidth "$W_BW" --bias_bitwidth 32 \
    --output_dlc gdn_quant.dlc > _quantize.log 2>&1

echo "[5/7] qnn-context-binary-generator (HTP $ARCH)"
cat > _htp_ext.json <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":$SOC_ID,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > _config_host.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/_htp_ext.json"}}
EOF
rm -rf ctx device_out
qnn-context-binary-generator --dlc_path gdn_quant.dlc \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --config_file _config_host.json --profiling_level detailed --profiling_option optrace \
    --save_backend_op_mapping --binary_file gdn_quant_ctx --output_dir ctx > _ctxgen.log 2>&1
for s in *schematic.bin ctx/*schematic.bin; do [ -f "$s" ] && mv -f "$s" ctx/ 2>/dev/null || true; done
[ -f ctx/gdn_quant_ctx.bin ] || { echo "    [FAIL] ctxgen"; tail -6 _ctxgen.log; exit 1; }

echo "[6/7] run on device ($DEVICE) from context binary (fp32 I/O boundary, int internal)"
DEV="$(ssh "$DEVICE" 'echo $HOME/qnn_run')"; W="$DEV/gdn_quant"
ssh "$DEVICE" "mkdir -p $W"
ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./gdn_htp_ext.json\"}}' > $W/gdn_config.json"
ssh "$DEVICE" "cat > $W/gdn_htp_ext.json" < _htp_ext.json
ssh "$DEVICE" "cat > $W/gdn_quant_ctx.bin" < ctx/gdn_quant_ctx.bin
for n in "${INPUTS[@]}"; do ssh "$DEVICE" "cat > $W/$n.raw" < "$n.raw"; done
ssh "$DEVICE" "cat > $W/input_list.txt" < input_list.txt
ssh "$DEVICE" "cd $W && rm -rf out && \
    LD_LIBRARY_PATH=..:/vendor/lib64 ADSP_LIBRARY_PATH=.. ../qnn-net-run \
    --backend ../libQnnHtp.so --retrieve_context gdn_quant_ctx.bin \
    --config_file gdn_config.json --input_list input_list.txt --output_dir out \
    --profiling_level detailed --profiling_option optrace --perf_profile burst" > _run.log 2>&1 || true
grep -q 'Finished Executing Graphs' _run.log || { echo "    [FAIL] device run"; tail -15 _run.log; exit 1; }
mkdir -p device_out
ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C device_out --strip-components=1

echo "[7/7] compare quantized HTP output to fp64 reference (L${GDN_LAYER} $TEST_PROMPT chunk $TEST_CHUNK)"
GDN_NATIVE_TOL="${GDN_NATIVE_TOL:-2e-2}" "$PY" "$K" --compare . --result device_out
