#!/usr/bin/env bash
# DEVICE per-stage error: extract a subgraph (inputs → each stage tensor), run it on the real HTP,
# compare to the fp64 reference (/tmp/ref_<stage>.raw). Shows where on-device error actually grows.
# Run scripts/gdn_stage_map.py first (writes /tmp/gdn_stage_map.json + /tmp/ref_*.raw).
set -uo pipefail
cd "$(dirname "$0")/quant_v2_L0"
ROOT_DIR="$(cd ../../.. && pwd)"
DEVICE="${DEVICE:-oneplus}"; ARCH="${ARCH:-v75}"; SOC_ID="${SOC_ID:-57}"
source "$ROOT_DIR/scripts/env.sh" >/dev/null 2>&1
export PATH="$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH"
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PY="$ROOT_DIR/.venv/bin/python"
ALLIN=(qc kc vc gc betac S_in cumsum_U sel0 sel1 sel2 sel3 vscale inv_vscale)
STAGES="${STAGES:-01_g 02_A 02_T 03_U 03_W 04_P 05_oc 06_Sout}"
DEV="$(ssh "$DEVICE" 'echo $HOME/qnn_run')"; W="$DEV/gdn_stage"
ssh "$DEVICE" "mkdir -p $W"
ssh "$DEVICE" "printf '%s' '{\"backend_extensions\":{\"shared_library_path\":\"../libQnnHtpNetRunExtensions.so\",\"config_file_path\":\"./htp.json\"}}' > $W/cfg.json"
ssh "$DEVICE" "cat > $W/htp.json" < _htp.json
for n in "${ALLIN[@]}"; do ssh "$DEVICE" "cat > $W/$n.raw" < "$n.raw"; done

printf "%-9s %-26s %-12s\n" "stage" "onnx tensor" "device relerr"
for S in $STAGES; do
  T=$("$PY" -c "import json;print(json.load(open('/tmp/gdn_stage_map.json'))['$S'])")
  # 1) extract subgraph inputs->T ; report which inputs it actually keeps
  KEEP=$("$PY" - "$T" <<'PY'
import onnx, onnx.utils, sys
allin=["qc","kc","vc","gc","betac","S_in","cumsum_U","sel0","sel1","sel2","sel3","vscale","inv_vscale"]
try:
    onnx.utils.extract_model("gdn_q.onnx","sub.onnx", allin, [sys.argv[1]])
except Exception:
    # retry: only inputs reachable
    m=onnx.load("gdn_q.onnx"); onnx.utils.extract_model("gdn_q.onnx","sub.onnx", allin, [sys.argv[1]], check_model=False)
keep=[i.name for i in onnx.load("sub.onnx").graph.input]
print(" ".join(keep))
PY
) || { printf "%-9s %-26s %s\n" "$S" "$T" "EXTRACT-FAIL"; continue; }
  # 2) override on the subgraph (int16-sym, self-calib p00)
  GDN_I16_SYM=1 GDN_NO_VSCALE=1 GDN_CALIB_PROMPT=p00 "$PY" "$ROOT_DIR/scripts/gdn_v2_override.py" sub.onnx sub_ovr.json --golden "$ROOT_DIR/tests/gdn/golden" --layer 0 --calib 2 >/dev/null 2>&1
  # 3) convert + ctxgen
  LAY=(); for n in $KEEP; do LAY+=(--source_model_input_layout "$n" NONTRIVIAL --desired_input_layout "$n" NONTRIVIAL); done
  LAY+=(--source_model_output_layout "$T" NONTRIVIAL --desired_output_layout "$T" NONTRIVIAL)
  qairt-converter -i sub.onnx --target_backend HTP "${LAY[@]}" --quantization_overrides sub_ovr.json -o sub.dlc > _sc.log 2>&1 || { printf "%-9s %-26s %s\n" "$S" "$T" "CONVERT-FAIL"; continue; }
  rm -rf sctx; qnn-context-binary-generator --dlc_path sub.dlc --backend "$QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" --config_file _cfg.json --binary_file sub_ctx --output_dir sctx > _sx.log 2>&1 || { printf "%-9s %-26s %s\n" "$S" "$T" "CTXGEN-FAIL: $(grep -ioE 'incorrect.*|expected.*|error[^,]*' _sx.log|head -1)"; continue; }
  # 4) input_list for the kept inputs ; run on device
  echo "$(for n in $KEEP; do printf '%s:=%s.raw ' "$n" "$n"; done)" > sub_list.txt
  ssh "$DEVICE" "cat > $W/sub_ctx.bin" < sctx/sub_ctx.bin
  ssh "$DEVICE" "cat > $W/sub_list.txt" < sub_list.txt
  ssh "$DEVICE" "cd $W && rm -rf out && LD_LIBRARY_PATH=..:/vendor/lib64 ADSP_LIBRARY_PATH=.. ../qnn-net-run --backend ../libQnnHtp.so --retrieve_context sub_ctx.bin --config_file cfg.json --input_list sub_list.txt --output_dir out --perf_profile burst" > _sr.log 2>&1 || true
  rm -rf sout; ssh "$DEVICE" "cd $W && tar cf - out" | tar xf - -C . 2>/dev/null && mv out sout 2>/dev/null
  # 5) compare device output to fp64 ref
  "$PY" - "$S" "$T" <<'PY'
import numpy as np, glob, sys
S,T=sys.argv[1],sys.argv[2]
ref=np.fromfile(f"/tmp/ref_{S}.raw","<f4")
f=glob.glob("sout/**/*.raw",recursive=True)
if not f: print(f"  {S:<7s} {T:<26s} NO-OUTPUT"); raise SystemExit
raw=np.fromfile(f[0],np.uint8); dev=(raw.view('<f2') if raw.size==ref.size*2 else raw.view('<f4')).astype(np.float32)
e=np.linalg.norm(dev.ravel()-ref)/(np.linalg.norm(ref)+1e-12)
print(f"  {S:<7s} {T:<26s} {e:.4e}")
PY
done