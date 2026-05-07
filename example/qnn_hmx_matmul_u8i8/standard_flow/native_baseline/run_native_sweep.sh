#!/usr/bin/env bash
# Phase-A native sweep: convert, ctxgen, run on device, extract chrometrace + profile.
# Usage:  SIZES="256 512 1024" ./run_native_sweep.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$ROOT_DIR/scripts/env.sh" >/dev/null
source "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH=$QNN_SDK_ROOT/lib/python
export PATH=$ANDROID_NDK_ROOT:$PATH
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}

SIZES="${SIZES:-256}"
DEVICE="${DEVICE:-oneplus}"
SHAPES_MKN="${SHAPES_MKN:-}"  # space-separated triplets like "256x512x256 256x256x512"

iter_cmds=()
for S in $SIZES; do iter_cmds+=("--size $S|s${S}_w8a8"); done
for T in $SHAPES_MKN; do
  M=$(echo "$T" | cut -d'x' -f1); K=$(echo "$T" | cut -d'x' -f2); N=$(echo "$T" | cut -d'x' -f3)
  iter_cmds+=("--M $M --K $K --N $N|m${M}k${K}n${N}_w8a8")
done

for cmd in "${iter_cmds[@]}"; do
  GENARGS="${cmd%|*}"
  OUT_NAME="${cmd#*|}"
  OUT_DIR="$SCRIPT_DIR/$OUT_NAME"
  echo "=== native sweep: $GENARGS -> $OUT_DIR ==="

  python "$SCRIPT_DIR/gen_matmul_onnx_size.py" $GENARGS --out "$OUT_NAME"

  cd "$OUT_DIR"
  cp -f "$SCRIPT_DIR/htp_config.json" "$SCRIPT_DIR/htp_backend_ext.json" .

  cp -f "$SCRIPT_DIR/baseline_s512_w8a8_existing/quant_overrides.json" .

  $QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter \
      -i model.onnx \
      --quantization_overrides quant_overrides.json \
      --source_model_input_layout A NONTRIVIAL \
      --desired_input_layout A NONTRIVIAL \
      --source_model_output_layout Y NONTRIVIAL \
      --desired_output_layout Y NONTRIVIAL \
      -o model.dlc 2>&1 | tee _convert.log | tail -5

  rm -rf ctx
  $QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator \
      --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
      --dlc_path model.dlc \
      --binary_file matmul_native_ctx \
      --output_dir ctx \
      --config_file htp_config.json \
      --profiling_level detailed --profiling_option optrace \
      --save_backend_op_mapping 2>&1 | tee _ctxgen.log | tail -3
  # schematic.bin lands in CWD next to ctx — keep it for chrometrace decode
  ls schematic.bin 2>/dev/null || ls model_schematic.bin 2>/dev/null || true

  # Push to device and run
  DEV_DIR="qnn_run/native_${OUT_NAME}"
  ssh "$DEVICE" "mkdir -p $DEV_DIR/runtime_inputs_u8"
  ssh "$DEVICE" "cat > $DEV_DIR/matmul_native_ctx.bin"     < ctx/matmul_native_ctx.bin
  ssh "$DEVICE" "cat > $DEV_DIR/runtime_inputs_u8/a.raw"   < runtime_inputs_u8/a.raw
  ssh "$DEVICE" "cat > $DEV_DIR/htp_config.json"           < "$SCRIPT_DIR/htp_config.json"
  ssh "$DEVICE" "cat > $DEV_DIR/htp_backend_ext.json"      < "$SCRIPT_DIR/htp_backend_ext.json"
  ssh "$DEVICE" "cat > $DEV_DIR/input_list.txt"            < runtime_input_list.txt

  ssh "$DEVICE" "cd $DEV_DIR && rm -rf out && \
      LD_LIBRARY_PATH=../:/vendor/lib64 ADSP_LIBRARY_PATH=../ \
      ../qnn-net-run \
        --backend ../libQnnHtp.so \
        --retrieve_context matmul_native_ctx.bin \
        --input_list input_list.txt \
        --profiling_level detailed --profiling_option optrace \
        --output_dir out \
        --config_file htp_config.json \
        --use_native_input_files \
        --use_native_output_files \
        --num_inferences 3 \
        --perf_profile burst 2>&1 | tail -5" > _run.log

  mkdir -p device_out
  ssh "$DEVICE" "cat $DEV_DIR/out/qnn-profiling-data_0.log" > device_out/qnn-profiling-data_0.log 2>/dev/null || true
  ssh "$DEVICE" "cat $DEV_DIR/out/qnn-profiling-data_2.log" > device_out/qnn-profiling-data_2.log 2>/dev/null || true

  if [ "${DECODE_OPTRACE:-1}" = "1" ]; then
    echo "  --- decode optrace artifacts ---"
    SCHEMATIC="model_schematic.bin"
    [ -f schematic.bin ] && SCHEMATIC="schematic.bin"
    PROFILE_LOG="device_out/qnn-profiling-data_2.log"
    [ -s "$PROFILE_LOG" ] || PROFILE_LOG="device_out/qnn-profiling-data_0.log"
    python "$ROOT_DIR/scripts/decode_qnn_optrace.py" "$OUT_DIR" \
        --profile-log "$OUT_DIR/$PROFILE_LOG" \
        --schematic "$OUT_DIR/$SCHEMATIC" || {
      [ "${STRICT_OPTRACE:-0}" = "1" ] && exit 1
      echo "  [warn] optrace decode failed; raw log kept in $OUT_DIR/device_out" >&2
    }
  fi

  # Plain profile-viewer for wall µs / cycles
  LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang \
  $QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-profile-viewer \
      --reader $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpProfilingReader.so \
      --input_log device_out/qnn-profiling-data_2.log > device_out/profile.txt 2>&1 || true

  echo "  --- ConvLayer event count (chrometrace) ---"
  python3 -c "
import json,os
for cand in ['optrace/chrometrace.json','device_out/chrometrace.json','chrometrace.json']:
    if os.path.exists(cand):
        d=json.load(open(cand))
        evs = d if isinstance(d,list) else d.get('traceEvents',[])
        from collections import Counter
        c=Counter(e['name'] for e in evs if isinstance(e,dict) and e.get('ph')=='X' and 'name' in e)
        for k,v in c.most_common():
            if 'Conv' in k or 'Crouton' in k or 'Slice' in k or 'Concat' in k or 'Dma' in k or 'Sync' in k:
                print(f'    {v:3d} {k}')
        break
" || true
  echo "  --- profile.txt key lines ---"
  grep -E 'Accelerator \(execute\) time|Total Inference Time|HVX threads' device_out/profile.txt | head -6 || true
  cd "$SCRIPT_DIR"
done
echo "=== native sweep done ==="
