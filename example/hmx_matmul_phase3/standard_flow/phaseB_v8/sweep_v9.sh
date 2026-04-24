#!/usr/bin/env bash
# Sweep V9 adaptive across multiple shapes; capture detailed profile for each.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$ROOT_DIR/scripts/env.sh" >/dev/null
source "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH=$QNN_SDK_ROOT/lib/python PATH=$ANDROID_NDK_ROOT:$PATH
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}

SHAPES="${SHAPES:-512 1024 2048 4096}"
CPL="$SCRIPT_DIR/gen_out/HmxMatMulPhase3Package_Converter_Op_Package/ConverterOpPackage/libConverterOpPackage.so"
X86_PKG="$(cd "$SCRIPT_DIR/../../build/x86_64-linux-clang" && pwd)/libQnnHmxMatMulPhase3.so"
SWEEP_DIR="$SCRIPT_DIR/sweep_v9"
mkdir -p "$SWEEP_DIR"

cd "$SCRIPT_DIR"

for S in $SHAPES; do
    echo "=== V9 sweep: ${S}³ ==="
    OUT_DIR="$SWEEP_DIR/s$S"; mkdir -p "$OUT_DIR"

    # Generate
    python gen_v8_graph.py --M $S --K $S --N $S 2>&1 | tee "$OUT_DIR/gen.log" | grep -E "rounds|nodes"

    # Convert
    $QNN_SDK_ROOT/bin/x86_64-linux-clang/qairt-converter \
        -i v8_model.onnx --op_package_config MatMulV8Package.xml --converter_op_package_lib "$CPL" \
        --quantization_overrides quant_overrides.json \
        --source_model_input_layout act_raw NONTRIVIAL --source_model_output_layout out NONTRIVIAL \
        --desired_input_layout act_raw NONTRIVIAL --desired_output_layout out NONTRIVIAL \
        -o "$OUT_DIR/v8_model.dlc" 2>&1 > "$OUT_DIR/convert.log" || { echo "  [FAIL convert]"; continue; }

    # Context binary
    rm -rf "$OUT_DIR/ctx"
    $QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator \
        --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
        --dlc_path "$OUT_DIR/v8_model.dlc" \
        --op_packages "$X86_PKG:HmxMatMulPhase3InterfaceProvider" \
        --binary_file v8_ctx --output_dir "$OUT_DIR/ctx" \
        --config_file htp_config.json 2>&1 > "$OUT_DIR/ctxgen.log" || { echo "  [FAIL ctxgen]"; grep ERROR "$OUT_DIR/ctxgen.log" | head -3; continue; }
    # Also record spill/fill bytes from the ctxgen summary
    grep -E "spill_bytes|fill_bytes" "$OUT_DIR/ctxgen.log" | tee -a "$OUT_DIR/ctxgen.summary"

    # Push to device and run
    ssh oneplus "cat > qnn_run/phaseB/v8_ctx.bin" < "$OUT_DIR/ctx/v8_ctx.bin"
    ssh oneplus "cat > qnn_run/phaseB/runtime_inputs_u8/act.raw" < runtime_inputs_u8/act.raw

    ssh oneplus "cd ~/qnn_run/phaseB && rm -rf out && \
        LD_LIBRARY_PATH=../:.:/vendor/lib64 ADSP_LIBRARY_PATH=../ \
        ../qnn-net-run --backend ../libQnnHtp.so --retrieve_context v8_ctx.bin \
          --op_packages ../libQnnHmxMatMulPhase3_cpu.so:HmxMatMulPhase3InterfaceProvider:CPU,../libQnnHmxMatMulPhase3_htp.so:HmxMatMulPhase3InterfaceProvider:HTP \
          --input_list input_list.txt --profiling_level detailed --config_file htp_config.json \
          --output_dir out --use_native_input_files --num_inferences 3 2>&1 | tail -3" \
        > "$OUT_DIR/run.log" 2>&1

    ssh oneplus "cat qnn_run/phaseB/out/qnn-profiling-data_2.log 2>/dev/null || cat qnn_run/phaseB/out/qnn-profiling-data_0.log" > "$OUT_DIR/profile.log" 2>/dev/null

    # Extract key numbers
    LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang \
    $QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-profile-viewer \
        --reader $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpProfilingReader.so \
        --input_log "$OUT_DIR/profile.log" > "$OUT_DIR/profile.txt" 2>&1
    grep -E "Accelerator \(execute\) time \(cycles\)|HVX threads used|QNN accelerator \(execute\) time " "$OUT_DIR/profile.txt" | tail -6 > "$OUT_DIR/summary.txt"
    cat "$OUT_DIR/summary.txt"
    echo ""
done
echo "=== V9 sweep done. Results in $SWEEP_DIR/s*/ ==="
