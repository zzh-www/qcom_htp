#!/usr/bin/env bash
# QNN-native GDN chunk kernel: ONNX -> DLC -> qnn-net-run, output aligned to the fp64 ref.
#
# This is the FLOAT native reference for the GDN HTP port (`先在qnn native上实现gdn kernel
# 对齐输出`). The chunk kernel (incl. the sequential (I-A)^-1 solve, rewritten as an exact
# Neumann product) is one static graph; we convert it and run it through the real QNN runtime
# and check the output matches scripts/gdn_ref_kernel.gdn_chunk.
#
# Usage:
#   ./run_gdn_native.sh                 # CPU host backend (default), fp32, fast alignment proof
#   BACKEND=htp ./run_gdn_native.sh     # HTP device backend, fp16, context-binary + native I/O
#
# Env: BACKEND=cpu|htp  DEVICE=oneplus  DEVICE_DIR=/data/local/tmp/gdn_native
set -euo pipefail
cd "$(dirname "$0")"
ROOT_DIR="$(cd ../.. && pwd)"
BACKEND="${BACKEND:-cpu}"
DEVICE="${DEVICE:-oneplus}"
DEVICE_DIR="${DEVICE_DIR:-\$HOME/qnn_run}"   # Termux home; QNN runtime libs already live here
ARCH="${ARCH:-v75}"; SOC_ID="${SOC_ID:-57}" # SM8650 (oneplus)

# ---- host QNN env ----
source "$ROOT_DIR/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source "$ROOT_DIR/scripts/dssh.sh"
PY="$ROOT_DIR/.venv/bin/python"
INPUTS=(qc kc vc gc betac S_in)
OUTPUTS=(oc S_out)

echo "[1/5] export ONNX + emit IO + fp64 reference outputs"
"$PY" "$ROOT_DIR/scripts/gdn_onnx_kernel.py" --export gdn_chunk.onnx >/dev/null
# HTP native I/O reads the fp16 public tensors; CPU run uses fp32.
[ "$BACKEND" = htp ] && GDN_IO_STORAGE=f2 || GDN_IO_STORAGE=f4
# GDN_IO=golden -> validate on real Qwen3.5-4B activations (GDN_LAYER / GDN_CHUNK select which);
# otherwise a random case (zero+nonzero state already exercised by S_in!=0).
GDN_IO_STORAGE="$GDN_IO_STORAGE" GDN_IO="${GDN_IO:-random}" \
GDN_LAYER="${GDN_LAYER:-0}" GDN_CHUNK="${GDN_CHUNK:-1}" GOLDEN="${GOLDEN:-$ROOT_DIR/tests/gdn/golden}" \
"$PY" - "$ROOT_DIR" <<'PY' >/dev/null
import os, sys; sys.path.insert(0, os.path.join(sys.argv[1], "scripts"))
import gdn_onnx_kernel as g
st = os.environ["GDN_IO_STORAGE"]
if os.environ["GDN_IO"] == "golden":
    g.emit_golden_io(".", golden_dir=os.environ["GOLDEN"], prompt=os.environ.get("GDN_PROMPT") or None,
                     layer=int(os.environ["GDN_LAYER"]), chunk=int(os.environ["GDN_CHUNK"]), storage=st)
else:
    g.emit_io(".", storage=st)
PY

echo "[2/5] qairt-converter ONNX -> DLC (layout-preserving on every public I/O)"
LAY=()
for n in "${INPUTS[@]}";  do LAY+=(--source_model_input_layout  "$n" NONTRIVIAL --desired_input_layout  "$n" NONTRIVIAL); done
for n in "${OUTPUTS[@]}"; do LAY+=(--source_model_output_layout "$n" NONTRIVIAL --desired_output_layout "$n" NONTRIVIAL); done
CONV=(-i gdn_chunk.onnx --target_backend HTP "${LAY[@]}" -o gdn_chunk.dlc)
[ "$BACKEND" = htp ] && CONV+=(--float_bitwidth 16)
qairt-converter "${CONV[@]}" > _convert.log 2>&1

if [ "$BACKEND" = cpu ]; then
    echo "[3/5] qnn-net-run CPU backend (host, fp32) -> output/"
    rm -rf output
    qnn-net-run --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnCpu.so" \
        --dlc_path gdn_chunk.dlc --input_list input_list.txt --output_dir output \
        > _run_cpu.log 2>&1
    echo "[4/5] compare to fp64 reference"
    "$PY" "$ROOT_DIR/scripts/gdn_onnx_kernel.py" --compare .
    echo "[5/5] done (CPU/fp32 native alignment)"
    exit 0
fi

# ---- HTP device path: context binary + native fp16 I/O ----
echo "[3/5] qnn-context-binary-generator (HTP $ARCH) -> ctx/"
rm -rf ctx device_out
cat > _htp_ext.json <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":$SOC_ID,"pd_session":"unsigned",
  "cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > _config_host.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so",
  "config_file_path":"$(pwd)/_htp_ext.json"}}
EOF
( cd "$(pwd)" && qnn-context-binary-generator \
    --dlc_path gdn_chunk.dlc \
    --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
    --config_file _config_host.json \
    --profiling_level detailed --profiling_option optrace \
    --save_backend_op_mapping \
    --binary_file gdn_chunk_ctx --output_dir ctx ) > _ctxgen.log 2>&1
# ctxgen drops the schematic in CWD; keep it beside the ctx bin.
for s in *schematic.bin ctx/*schematic.bin; do [ -f "$s" ] && mv -f "$s" ctx/ 2>/dev/null || true; done

echo "[4/5] push + run on device ($DEVICE:$DEVICE_DIR) from context binary, native fp16 I/O"
DEV="$(ssh "$DEVICE" "echo $DEVICE_DIR")"   # expand $HOME on device
# device already holds libQnnHtp*, libQnnSystem, qnn-net-run, libQnnHtpNetRunExtensions.so
ssh "$DEVICE" "cat > $DEV/gdn_htp_ext.json" < _htp_ext.json
ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"./libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./gdn_htp_ext.json\"}}' > $DEV/gdn_config.json"
ssh "$DEVICE" "cat > $DEV/gdn_chunk_ctx.bin" < ctx/gdn_chunk_ctx.bin
for n in "${INPUTS[@]}"; do ssh "$DEVICE" "cat > $DEV/gdn_$n.raw" < "$n.raw"; done
ssh "$DEVICE" "printf '%s\n' '$(for n in "${INPUTS[@]}"; do printf '%s:=gdn_%s.raw ' "$n" "$n"; done)' > $DEV/gdn_input_list.txt"
ssh "$DEVICE" "cd $DEV && rm -rf gdn_out && \
    LD_LIBRARY_PATH=.:/vendor/lib64 ADSP_LIBRARY_PATH=. ./qnn-net-run \
    --backend ./libQnnHtp.so --retrieve_context gdn_chunk_ctx.bin \
    --config_file gdn_config.json \
    --input_list gdn_input_list.txt --output_dir gdn_out \
    --use_native_input_files --use_native_output_files \
    --profiling_level detailed --profiling_option optrace --perf_profile burst" > _run_htp.log 2>&1 || true
if ! grep -q 'Finished Executing Graphs' _run_htp.log 2>/dev/null; then
    echo "    [FAIL] device qnn-net-run (see _run_htp.log)"; tail -15 _run_htp.log; exit 1
fi
mkdir -p device_out
ssh "$DEVICE" "cd $DEV && tar cf - gdn_out" | tar xf - -C device_out --strip-components=1
ssh "$DEVICE" "cat $DEV/gdn_out/qnn-profiling-data_0.log 2>/dev/null" > device_out/qnn-profiling-data_0.log 2>/dev/null || true

echo "[5/5] compare device HTP output to fp64 reference"
GDN_NATIVE_TOL="${GDN_NATIVE_TOL:-2e-2}" "$PY" "$ROOT_DIR/scripts/gdn_onnx_kernel.py" --compare . --result device_out
