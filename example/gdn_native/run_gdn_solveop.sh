#!/usr/bin/env bash
# GDN on HTP with the triangular solve done by the GdnSolve HVX/QHPI custom op (int16 forward
# substitution) instead of the int8-matmul Neumann chain.  Same v2.0.0 override flow as run_gdn_v2.sh;
# the only change is graph surgery (scripts/gdn_insert_solve_op.py) swapping the 363-node solve subgraph
# for one GdnSolve node, plus the op-package wiring (--op_package_config / --op_packages).
#   override is computed on the FULL gdn_q.onnx (ORT can't run the custom op), then applied to the
#   surgically-reduced graph (A/T/downstream encodings are identical; extra keys are harmless).
# Result target: U/W reach the int8-in[1] ceiling (T value near-exact); oc improvement + optrace perf.
set -uo pipefail
cd "$(dirname "$0")"
ROOT_DIR="$(cd ../.. && pwd)"
GDN_LAYER="${GDN_LAYER:-0}"; TEST_PROMPT="${TEST_PROMPT:-p00}"; TEST_CHUNK="${TEST_CHUNK:-0}"
CALIB_PROMPT="${CALIB_PROMPT-$TEST_PROMPT}"
DEVICE="${DEVICE:-oneplus}"; ARCH="${ARCH:-v75}"; SOC_ID="${SOC_ID:-57}"
GOLDEN="${GOLDEN:-$ROOT_DIR/tests/gdn/golden}"
WORK="quant_v2_L${GDN_LAYER}"
OPDIR="$ROOT_DIR/example/gdn_native/solve_op"
PKG="GdnSolvePackage"; PROV="${PKG}InterfaceProvider"
source "$ROOT_DIR/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source "$ROOT_DIR/scripts/dssh.sh"
PY="$ROOT_DIR/.venv/bin/python"; K="$ROOT_DIR/scripts/gdn_onnx_kernel.py"
mkdir -p "$WORK"; cd "$WORK"
export GDN_NO_VSCALE=1

echo "[0/7] build GdnSolve op package"
bash "$OPDIR/build.sh" > _opbuild.log 2>&1 || { echo "  [FAIL] op build"; tail -15 _opbuild.log; exit 1; }
X86_PKG="$OPDIR/build/x86_64-linux-clang/lib${PKG}.so"
HTP_PKG="$OPDIR/build/hexagon-${ARCH}/lib${PKG}_htp.so"
CPU_PKG="$OPDIR/build/aarch64/lib${PKG}_cpu.so"
CPL="$OPDIR/converter/build/libConverterOpPackage.so"

echo "[1/7] export quant-path ONNX + override (on full graph, ORT-runnable)"
"$PY" "$K" --export-q gdn_q.onnx >/dev/null 2>&1
export GDN_I16_SYM=1
if [ -n "$CALIB_PROMPT" ]; then export GDN_CALIB_PROMPT="$CALIB_PROMPT"; CALN=2; else unset GDN_CALIB_PROMPT; CALN=12; fi
"$PY" "$ROOT_DIR/scripts/gdn_v2_override.py" gdn_q.onnx v2_ovr.json --golden "$GOLDEN" --layer "$GDN_LAYER" --calib "$CALN" >/dev/null

echo "[2/7] graph surgery: replace solve subgraph with GdnSolve node"
"$PY" "$ROOT_DIR/scripts/gdn_insert_solve_op.py" gdn_q.onnx gdn_q_solveop.onnx --override v2_ovr.json
# kept graph inputs (sel0..3 dropped by surgery)
mapfile -t INPUTS < <("$PY" -c "import onnx;print('\n'.join(i.name for i in onnx.load('gdn_q_solveop.onnx').graph.input))")
echo "    graph inputs: ${INPUTS[*]}"

echo "[3/7] convert with op package + v2 override"
LAY=(); for n in "${INPUTS[@]}"; do LAY+=(--source_model_input_layout "$n" NONTRIVIAL --desired_input_layout "$n" NONTRIVIAL); done
for n in oc S_out; do LAY+=(--source_model_output_layout "$n" NONTRIVIAL --desired_output_layout "$n" NONTRIVIAL); done
qairt-converter -i gdn_q_solveop.onnx --target_backend HTP \
    --op_package_config "$OPDIR/${PKG}.xml" --converter_op_package_lib "$CPL" \
    "${LAY[@]}" --quantization_overrides v2_ovr.json -o gdn_solveop.dlc > _convert.log 2>&1 \
    || { echo "  [FAIL] convert"; tail -20 _convert.log; exit 1; }

echo "[4/7] ctxgen with op package"
cat > _htp.json <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":$SOC_ID,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > _cfg.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/_htp.json"}}
EOF
rm -rf ctx
qnn-context-binary-generator --dlc_path gdn_solveop.dlc --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --op_packages "$X86_PKG:$PROV" --config_file _cfg.json --save_backend_op_mapping \
    --binary_file gdn_solveop_ctx --output_dir ctx > _ctxgen.log 2>&1 \
    || { echo "  [FAIL] ctxgen"; tail -20 _ctxgen.log; exit 1; }

echo "[5/7] emit golden IO + run on device ($DEVICE) with op package"
GDN_LAYER="$GDN_LAYER" "$PY" - "$ROOT_DIR" "$GOLDEN" "$TEST_PROMPT" "$GDN_LAYER" "$TEST_CHUNK" <<'PY' >/dev/null
import os,sys; sys.path.insert(0,os.path.join(sys.argv[1],"scripts"))
import gdn_onnx_kernel as g
g.emit_golden_io(".", golden_dir=sys.argv[2], prompt=sys.argv[3], layer=int(sys.argv[4]), chunk=int(sys.argv[5]), storage="f4", consts=True)
PY
# input_list with only the kept graph inputs, in graph order
echo "$(for n in "${INPUTS[@]}"; do printf '%s:=%s.raw ' "$n" "$n"; done)" > input_list.txt
DEV="$(ssh "$DEVICE" 'echo $HOME/qnn_run')"; W="$DEV/gdn_solveop"
ssh "$DEVICE" "mkdir -p $W"
ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./gdn_htp_ext.json\"}}' > $W/gdn_config.json"
ssh "$DEVICE" "cat > $W/gdn_htp_ext.json" < _htp.json
ssh "$DEVICE" "cat > $W/gdn_solveop_ctx.bin" < ctx/gdn_solveop_ctx.bin
ssh "$DEVICE" "cat > $W/lib${PKG}_htp.so" < "$HTP_PKG"
ssh "$DEVICE" "cat > $W/lib${PKG}_cpu.so" < "$CPU_PKG"
for n in "${INPUTS[@]}"; do ssh "$DEVICE" "cat > $W/$n.raw" < "$n.raw"; done
ssh "$DEVICE" "cat > $W/input_list.txt" < input_list.txt
# ADSP_LIBRARY_PATH uses ';' separators on Hexagon (NOT ':'); must include '.' so the DSP finds the op package skel
ssh "$DEVICE" "cd $W && rm -rf out && LD_LIBRARY_PATH=..:.:/vendor/lib64 ADSP_LIBRARY_PATH='..;.;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/dsp/cdsp' ../qnn-net-run \
    --backend ../libQnnHtp.so --retrieve_context gdn_solveop_ctx.bin --config_file gdn_config.json \
    --op_packages ./lib${PKG}_cpu.so:$PROV:CPU,./lib${PKG}_htp.so:$PROV:HTP \
    --input_list input_list.txt --output_dir out --profiling_level detailed --profiling_option optrace \
    --perf_profile burst" > _run.log 2>&1 || true
grep -q 'Finished Executing Graphs' _run.log || { echo "  [FAIL] device run"; tail -20 _run.log; exit 1; }
rm -rf device_out; mkdir -p device_out; ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C device_out --strip-components=1

echo "[6/7] compare quantized HTP output to fp64 reference (L${GDN_LAYER} $TEST_PROMPT)"
GDN_NATIVE_TOL="${GDN_NATIVE_TOL:-1.5e-2}" "$PY" "$K" --compare . --result device_out

echo "[7/7] device run OK; optrace under $W/out (profiling). Parse with qnn-profile-viewer for wall us."
