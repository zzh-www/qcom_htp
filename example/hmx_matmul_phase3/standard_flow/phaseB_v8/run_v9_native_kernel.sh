#!/usr/bin/env bash
# run_v9_native_kernel.sh — Step 5.2/5.3 V9_USE_NATIVE_KERNEL test.
# Calls dlsym'd libQnnHtpV75Skel.so :: hmx_convbbb1x1_stride1 per (mt, nt)
# tile and verifies output bit-exact against numpy reference at 256³.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$ROOT_DIR/scripts/env.sh" >/dev/null
source "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH=$QNN_SDK_ROOT/lib/python
export PATH=$ANDROID_NDK_ROOT:$PATH
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}

DEVICE="${DEVICE:-oneplus}"
M="${M:-256}"; K="${K:-256}"; N="${N:-256}"
MODE="${MODE:-default}"   # default = saturating random; layout_test = m & 0xF
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/phase1_validation/v9_native_kernel}"
export OUT_DIR M K N MODE
mkdir -p "$OUT_DIR"

CPL="$SCRIPT_DIR/gen_out/HmxMatMulPhase3Package_Converter_Op_Package/ConverterOpPackage/libConverterOpPackage.so"
X86_PKG="$(cd "$SCRIPT_DIR/../../build/x86_64-linux-clang" && pwd)/libQnnHmxMatMulPhase3.so"
PKG_HTP="$SCRIPT_DIR/../../build/hexagon-v75/libQnnHmxMatMulPhase3_htp.so"
PKG_CPU="$SCRIPT_DIR/../../build/aarch64/libQnnHmxMatMulPhase3_cpu.so"

cd "$SCRIPT_DIR"

echo "=== [1/5] gen_v8c8_test.py --mode $MODE ${M}×${K}×${N} ==="
python gen_v8c8_test.py --M "$M" --K "$K" --N "$N" --mode "$MODE" -o "$OUT_DIR/v8c8.onnx" 2>&1 | tail -3

echo "=== [2/5] qairt-converter ==="
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter \
    -i "$OUT_DIR/v8c8.onnx" \
    --op_package_config MatMulV8Package.xml \
    --converter_op_package_lib "$CPL" \
    --quantization_overrides quant_overrides.json \
    --source_model_input_layout act_raw NONTRIVIAL \
    --source_model_output_layout out NONTRIVIAL \
    --desired_input_layout act_raw NONTRIVIAL \
    --desired_output_layout out NONTRIVIAL \
    -o "$OUT_DIR/v8c8.dlc" 2>&1 | tee "$OUT_DIR/convert.log" | tail -2

echo "=== [3/5] qnn-context-binary-generator (with optrace) ==="
rm -rf "$OUT_DIR/ctx"
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator \
    --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
    --dlc_path "$OUT_DIR/v8c8.dlc" \
    --op_packages "$X86_PKG:HmxMatMulPhase3InterfaceProvider" \
    --binary_file v8c8_ctx --output_dir "$OUT_DIR/ctx" \
    --config_file htp_config.json \
    --profiling_level detailed --profiling_option optrace \
    --save_backend_op_mapping 2>&1 | tee "$OUT_DIR/ctxgen.log" | tail -3
# Capture schematic.bin (lands in CWD).
[ -f v8c8_schematic.bin ] && mv v8c8_schematic.bin "$OUT_DIR/ctx/"

echo "=== [4/5] push and run on device ==="
ssh "$DEVICE" "mkdir -p qnn_run/phaseB_c8/runtime_inputs_u8"
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/v8c8_ctx.bin" < "$OUT_DIR/ctx/v8c8_ctx.bin"
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/runtime_inputs_u8/act_v8c8.raw" < runtime_inputs_u8/act_v8c8.raw
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/htp_config.json" < htp_config.json
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/htp_backend_ext.json" < htp_backend_ext.json
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/libQnnHmxMatMulPhase3_htp.so" < "$PKG_HTP"
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/libQnnHmxMatMulPhase3_cpu.so" < "$PKG_CPU"
ssh "$DEVICE" "cat > qnn_run/libQnnHmxMatMulPhase3_htp.so" < "$PKG_HTP"
ssh "$DEVICE" "cat > qnn_run/libQnnHmxMatMulPhase3_cpu.so" < "$PKG_CPU"
echo 'act_raw:=runtime_inputs_u8/act_v8c8.raw' > "$OUT_DIR/input_list.txt"
ssh "$DEVICE" "cat > qnn_run/phaseB_c8/input_list.txt" < "$OUT_DIR/input_list.txt"
ssh "$DEVICE" 'logcat -c 2>/dev/null || true'

ssh "$DEVICE" "cd qnn_run/phaseB_c8 && rm -rf out && \
    LD_LIBRARY_PATH=../:.:/vendor/lib64 ADSP_LIBRARY_PATH=../ \
    ../qnn-net-run \
      --backend ../libQnnHtp.so \
      --retrieve_context v8c8_ctx.bin \
      --op_packages ../libQnnHmxMatMulPhase3_cpu.so:HmxMatMulPhase3InterfaceProvider:CPU,../libQnnHmxMatMulPhase3_htp.so:HmxMatMulPhase3InterfaceProvider:HTP \
      --input_list input_list.txt \
      --output_dir out \
      --config_file htp_config.json \
      --use_native_input_files \
      --num_inferences 3 \
      --profiling_level detailed --profiling_option optrace \
      --perf_profile burst 2>&1" > "$OUT_DIR/run.log" 2>&1 || true
tail -10 "$OUT_DIR/run.log"

# Pull profile log + decode chrometrace (Step 5.4 perf metric).
ssh "$DEVICE" 'cat qnn_run/phaseB_c8/out/qnn-profiling-data_0.log 2>/dev/null' \
    > "$OUT_DIR/profile_dev.log" 2>/dev/null || true
if [ -s "$OUT_DIR/profile_dev.log" ] && [ -f "$OUT_DIR/ctx/v8c8_schematic.bin" ]; then
    PV=$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-profile-viewer
    LDIR=$QNN_SDK_ROOT/lib/x86_64-linux-clang
    echo '{"htp_json": true, "runtrace": true, "memory_info": true}' > /tmp/optrace_config.json
    LD_LIBRARY_PATH=$LDIR $PV \
        --config /tmp/optrace_config.json \
        --reader $LDIR/libQnnHtpOptraceProfilingReader.so \
        --input_log "$OUT_DIR/profile_dev.log" \
        --schematic "$OUT_DIR/ctx/v8c8_schematic.bin" \
        --output "$OUT_DIR/chrometrace.json" 2>&1 | tail -3
fi

echo "  --- DSP-side logcat (filtered for crashes) ---"
ssh "$DEVICE" 'logcat -d -t 500 2>&1' \
    | grep -iE 'qnn|adsprpc|skel|fastrpc|crash|fault|err|fatal' \
    | grep -vE 'avc:|F2FS' \
    | tail -20 \
    | tee "$OUT_DIR/logcat.txt" || true

mkdir -p "$OUT_DIR/device_out"
ssh "$DEVICE" 'cat qnn_run/phaseB_c8/out/Result_0/out.raw 2>/dev/null' > "$OUT_DIR/device_out/out.raw" 2>/dev/null || true

echo "=== [5/5] compare to reference ==="
if [ ! -s "$OUT_DIR/device_out/out.raw" ]; then
    echo "  output empty — kernel likely crashed (see logcat above)"
    exit 1
fi
python3 - <<'PY'
import numpy as np, os
M, N = int(os.environ["M"]), int(os.environ["N"])
b = np.fromfile(os.environ["OUT_DIR"] + "/device_out/out.raw", dtype=np.float32)
dev = np.round(b).astype(int).clip(0,255).astype(np.uint8).reshape(M, N)
ref = np.load(os.environ["OUT_DIR"] + "/v8c8.onnx.out_ref_u8.npy")

match = (dev == ref).sum()
total = M * N
print(f"Bit-exact: {match}/{total} ({100*match/total:.2f}%)")

# Show first few cells
print(f"  ref[:4, :8] = {ref[:4, :8].tolist()}")
print(f"  dev[:4, :8] = {dev[:4, :8].tolist()}")

# Quantify
diff = (dev != ref)
if diff.any():
    print(f"  rows with any diff: {diff.any(axis=1).sum()}/{M}")
    print(f"  total diff cells: {diff.sum()}/{total}")
    abs_diff = np.abs(dev.astype(int) - ref.astype(int))
    print(f"  abs_diff max={abs_diff.max()}, mean={abs_diff.mean():.2f}")
PY
# BbbKMajor single-event dur extraction (PRIMARY perf metric per Step 5 plan).
if [ -s "$OUT_DIR/chrometrace.json" ]; then
    echo
    echo "=== chrometrace BbbKMajor cyc/event (warm avg, instances 1+) ==="
    python3 - <<'PY'
import json, os
d = json.load(open(os.environ["OUT_DIR"] + "/chrometrace.json"))
xs = [e for e in d.get('traceEvents',[]) if e.get('ph') == 'X']
durs = []
for e in xs:
    name = e.get('name', '')
    if 'BbbKMajor' not in name: continue
    durs.append((e.get('ts'), e.get('dur', 0), name))
durs.sort()
print(f"  {len(durs)} BbbKMajor events:")
for ts, dur, n in durs[:10]:
    print(f"    ts={ts}  dur={dur}  name={n}")
if len(durs) > 1:
    warm = [d for _, d, _ in durs[1:]]
    print(f"  warm avg (skip first): {sum(warm)/len(warm):.0f} cyc")
    print(f"  TARGET: ~1500 cyc (native ConvLayer_s1.opt level)")
PY
fi
echo "=== done ==="
