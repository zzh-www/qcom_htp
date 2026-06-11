#!/usr/bin/env bash
# Single act×act MatMul on device, NUMERICAL check, comparing in[0] encodings:
#   int16sym   : SFX16, zero_point 0           (what GDN uses now)
#   uint16mid  : UFX16, zero_point pinned 32768 (offset -32768, is_symmetric False)  <- the hypothesis
#   uint16asym : UFX16, zero_point fit to range (known-bad, control)
# in[1] is always int8 symmetric. Compares device Y to fp32 A@B.
set -euo pipefail
cd "$(dirname "$0")"
ROOT_DIR="$(cd ../../.. && pwd)"
DEVICE="${DEVICE:-oneplus}"; ARCH="${ARCH:-v75}"; SOC_ID="${SOC_ID:-57}"
source "$ROOT_DIR/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT_DIR/.venv/bin/python"; MP="$ROOT_DIR/scripts/mm_probe.py"
W=mm_dev; rm -rf "$W"; mkdir -p "$W"
"$PY" "$MP" --emit "$W" --kind actact >/dev/null
cd "$W"
LAY=(--source_model_input_layout A NONTRIVIAL --desired_input_layout A NONTRIVIAL
     --source_model_input_layout B NONTRIVIAL --desired_input_layout B NONTRIVIAL
     --source_model_output_layout Y NONTRIVIAL --desired_output_layout Y NONTRIVIAL)
cat > _htp.json <<EOF
{"devices":[{"dsp_arch":"$ARCH","soc_id":$SOC_ID,"pd_session":"unsigned","cores":[{"core_id":0,"perf_profile":"burst","rpc_control_latency":100}]}]}
EOF
cat > _cfg.json <<EOF
{"backend_extensions":{"shared_library_path":"$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"$(pwd)/_htp.json"}}
EOF
DEV="$(ssh "$DEVICE" 'echo $HOME/qnn_run')"; RW="$DEV/mm_test"
ssh "$DEVICE" "mkdir -p $RW"
ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./htp.json\"}}' > $RW/cfg.json"
ssh "$DEVICE" "cat > $RW/htp.json" < _htp.json
ssh "$DEVICE" "cat > $RW/A.raw" < A.raw
ssh "$DEVICE" "cat > $RW/B.raw" < B.raw
ssh "$DEVICE" "cat > $RW/test_list.txt" < test_list.txt

for MODE in int16sym uint16mid uint16asym; do
  IN0_MODE=$MODE "$PY" "$MP" --write-v2 . --i8 B >/dev/null
  A_DT=$("$PY" -c "import json;print([e for e in json.load(open('v2_ovr.json'))['encodings'] if e['name']=='A'][0])")
  qairt-converter -i mm.onnx --target_backend HTP "${LAY[@]}" --quantization_overrides v2_ovr.json -o mm.dlc > _c.log 2>&1 || { echo "$MODE: CONVERT FAIL"; continue; }
  rm -rf ctx
  if ! qnn-context-binary-generator --dlc_path mm.dlc --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" --config_file _cfg.json --binary_file mm_ctx --output_dir ctx > _ctx.log 2>&1; then
    echo "$MODE: CTXGEN FAIL -> $(grep -iE 'error|-32768|valid' _ctx.log | head -1)"; continue
  fi
  ssh "$DEVICE" "cat > $RW/mm_ctx.bin" < ctx/mm_ctx.bin
  ssh "$DEVICE" "cd $RW && rm -rf out && LD_LIBRARY_PATH=..:/vendor/lib64 ADSP_LIBRARY_PATH=.. ../qnn-net-run --backend ../libQnnHtp.so --retrieve_context mm_ctx.bin --config_file cfg.json --input_list test_list.txt --output_dir out --perf_profile burst" > _run.log 2>&1 || true
  rm -rf out; ssh "$DEVICE" "cd $RW && tar cf - out" | tar xf - -C . 2>/dev/null || true
  "$PY" - "$MODE" "$A_DT" <<'PY'
import numpy as np, glob, sys
ref=np.fromfile("Y_ref.raw","<f4")
f=glob.glob("out/**/Y*.raw",recursive=True)
if not f: print(f"  {sys.argv[1]:11s}: NO OUTPUT"); raise SystemExit
raw=np.fromfile(f[0],np.uint8); dev=(raw.view('<f2') if raw.size==ref.size*2 else raw.view('<f4')).astype(np.float32)
e=np.linalg.norm(dev-ref)/np.linalg.norm(ref); a,b=np.polyfit(ref,dev,1)
print(f"  {sys.argv[1]:11s}: Y relerr={e:.4f}  fit dev~{a:.3f}*ref  | A enc: {sys.argv[2]}")
PY
done