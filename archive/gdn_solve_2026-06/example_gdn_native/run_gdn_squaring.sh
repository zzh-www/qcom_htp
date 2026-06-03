#!/usr/bin/env bash
# HMX prototype: replace the GDN solve subgraph with a SQUARING chain of NATIVE MatMuls (no custom op),
# so QNN maps the dense 64x64 matmuls to HMX and overlaps the HVX rest.  Goal metric: HMX utilization up
# + wall, with oc ~1.2e-2.  Same v2.0.0 override flow as run_gdn_v2.sh; override is computed AFTER surgery
# (on the squaring graph) so the new sq_* tensors get encodings.
set -uo pipefail
cd "$(dirname "$0")"
ROOT_DIR="$(cd ../.. && pwd)"
GDN_LAYER="${GDN_LAYER:-0}"; TEST_PROMPT="${TEST_PROMPT:-p00}"; TEST_CHUNK="${TEST_CHUNK:-0}"
CALIB_PROMPT="${CALIB_PROMPT-$TEST_PROMPT}"
DEVICE="${DEVICE:-oneplus}"; ARCH="${ARCH:-v75}"; SOC_ID="${SOC_ID:-57}"
GOLDEN="${GOLDEN:-$ROOT_DIR/tests/gdn/golden}"
WORK="quant_sq_L${GDN_LAYER}"
source "$ROOT_DIR/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT_DIR/.venv/bin/python"; K="$ROOT_DIR/scripts/gdn_onnx_kernel.py"
mkdir -p "$WORK"; cd "$WORK"
export GDN_NO_VSCALE=1

echo "[1/6] export quant-path ONNX"
"$PY" "$K" --export-q gdn_q.onnx >/dev/null 2>&1

echo "[2/6] graph surgery: replace solve with squaring matmul chain"
"$PY" "$ROOT_DIR/scripts/gdn_insert_squaring.py" gdn_q.onnx gdn_q_sq.onnx --steps "${STEPS:-6}"
mapfile -t INPUTS < <("$PY" -c "import onnx;print('\n'.join(i.name for i in onnx.load('gdn_q_sq.onnx').graph.input))")
echo "    graph inputs: ${INPUTS[*]}"

echo "[3/6] v2 override computed ON the squaring graph (so sq_* tensors get encodings)"
export GDN_I16_SYM=1
if [ -n "$CALIB_PROMPT" ]; then export GDN_CALIB_PROMPT="$CALIB_PROMPT"; CALN=2; else unset GDN_CALIB_PROMPT; CALN=12; fi
"$PY" "$ROOT_DIR/scripts/gdn_v2_override.py" gdn_q_sq.onnx v2_ovr.json --golden "$GOLDEN" --layer "$GDN_LAYER" --calib "$CALN" >/dev/null

echo "[4/6] convert + ctxgen (native, optrace for HMX/HVX util)"
LAY=(); for n in "${INPUTS[@]}"; do LAY+=(--source_model_input_layout "$n" NONTRIVIAL --desired_input_layout "$n" NONTRIVIAL); done
for n in oc S_out; do LAY+=(--source_model_output_layout "$n" NONTRIVIAL --desired_output_layout "$n" NONTRIVIAL); done
qairt-converter -i gdn_q_sq.onnx --target_backend HTP "${LAY[@]}" --quantization_overrides v2_ovr.json -o gdn_sq.dlc > _convert.log 2>&1 \
    || { echo "  [FAIL] convert"; tail -20 _convert.log; exit 1; }
cat > _htp.json <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":$SOC_ID,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > _cfg.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/_htp.json"}}
EOF
rm -rf ctx
qnn-context-binary-generator --dlc_path gdn_sq.dlc --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --config_file _cfg.json --save_backend_op_mapping --profiling_level detailed --profiling_option optrace \
    --binary_file gdn_sq_ctx --output_dir ctx > _ctxgen.log 2>&1 \
    || { echo "  [FAIL] ctxgen"; tail -25 _ctxgen.log; exit 1; }
for s in *schematic.bin ctx/*schematic.bin; do [ -f "$s" ] && mv -f "$s" ctx/ 2>/dev/null || true; done

echo "[5/6] emit golden IO + device run ($DEVICE) with optrace"
GDN_LAYER="$GDN_LAYER" "$PY" - "$ROOT_DIR" "$GOLDEN" "$TEST_PROMPT" "$GDN_LAYER" "$TEST_CHUNK" <<'PY' >/dev/null
import os,sys; sys.path.insert(0,os.path.join(sys.argv[1],"scripts"))
import gdn_onnx_kernel as g
g.emit_golden_io(".", golden_dir=sys.argv[2], prompt=sys.argv[3], layer=int(sys.argv[4]), chunk=int(sys.argv[5]), storage="f4", consts=True)
PY
echo "$(for n in "${INPUTS[@]}"; do printf '%s:=%s.raw ' "$n" "$n"; done)" > input_list.txt
DEV="$(ssh "$DEVICE" 'echo $HOME/qnn_run')"; W="$DEV/gdn_sq"
ssh "$DEVICE" "mkdir -p $W"
ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./gdn_htp_ext.json\"}}' > $W/gdn_config.json"
ssh "$DEVICE" "cat > $W/gdn_htp_ext.json" < _htp.json
ssh "$DEVICE" "cat > $W/gdn_sq_ctx.bin" < ctx/gdn_sq_ctx.bin
for n in "${INPUTS[@]}"; do ssh "$DEVICE" "cat > $W/$n.raw" < "$n.raw"; done
ssh "$DEVICE" "cat > $W/input_list.txt" < input_list.txt
ssh "$DEVICE" "cd $W && rm -rf out && LD_LIBRARY_PATH=..:/vendor/lib64 ADSP_LIBRARY_PATH=.. ../qnn-net-run \
    --backend ../libQnnHtp.so --retrieve_context gdn_sq_ctx.bin --config_file gdn_config.json \
    --input_list input_list.txt --output_dir out --profiling_level detailed --profiling_option optrace \
    --perf_profile burst" > _run.log 2>&1 || true
grep -q 'Finished Executing Graphs' _run.log || { echo "  [FAIL] device run"; tail -20 _run.log; exit 1; }
rm -rf device_out; mkdir -p device_out; ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C device_out --strip-components=1

echo "[6/6] compare oc + wall"
GDN_NATIVE_TOL="${GDN_NATIVE_TOL:-1.5e-2}" "$PY" "$K" --compare . --result device_out || true
qnn-profile-viewer --input_log device_out/qnn-profiling-data_0.log 2>/dev/null | grep -iE 'accelerator \(execute\) time' | grep -iv cycles | sed 's/^ */    /' | sort -u
