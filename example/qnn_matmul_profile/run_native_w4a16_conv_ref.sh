#!/usr/bin/env bash
#
# Generate, convert, export context, run, and decode the QNN-native W4A16
# Conv1x1 reference with native u16 input/output files.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env.sh" >/dev/null

DEVICE="${DEVICE:-oneplus}"
CONNECT="${CONNECT:-ssh}"
ARCH="${ARCH:-v75}"
SHAPE="${SHAPE:-256,256,256}"
OUT_DIR="${OUT_DIR:-}"
NUM_INFERENCES="${NUM_INFERENCES:-3}"
SKIP_DEVICE="${SKIP_DEVICE:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device|-d) DEVICE="$2"; shift 2 ;;
        --connect|-c) CONNECT="$2"; shift 2 ;;
        --arch|-a) ARCH="$2"; shift 2 ;;
        --shape) SHAPE="$2"; shift 2 ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        --skip-device) SKIP_DEVICE=1; shift ;;
        --help|-h)
            sed -n '3,28p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

case "$CONNECT" in
    ssh|adb) ;;
    *) echo "invalid --connect $CONNECT (need ssh|adb)" >&2; exit 2 ;;
esac
case "$ARCH" in
    v66|v68|v69|v73|v75|v79|v81) ;;
    *) echo "unknown --arch $ARCH" >&2; exit 2 ;;
esac

IFS=',' read -r SHAPE_M SHAPE_K SHAPE_N <<<"$SHAPE"
if [ -z "${SHAPE_M:-}" ] || [ -z "${SHAPE_K:-}" ] || [ -z "${SHAPE_N:-}" ]; then
    echo "bad --shape '$SHAPE' (expected M,K,N)" >&2
    exit 2
fi
SCRIPT_DIR="$ROOT_DIR/example/qnn_matmul_profile"
if [ -z "$OUT_DIR" ]; then
    OUT_DIR="$SCRIPT_DIR/output_native_w4a16_conv_${SHAPE_M}"
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

if [ -z "${DEVICE_DIR:-}" ]; then
    case "$CONNECT" in
        ssh) DEVICE_DIR="~/qnn_run" ;;
        adb) DEVICE_DIR="/data/local/tmp/qnn_run" ;;
    esac
fi

declare -A ARCH_SOCID=(
    [v66]=18
    [v68]=30
    [v69]=36
    [v73]=43
    [v75]=57
    [v79]=69
    [v81]=87
)
SOC_ID="${SOC_ID:-${ARCH_SOCID[$ARCH]:-0}}"
QNN="$QNN_SDK_ROOT"
ARCH_SKEL_LIB="libQnnHtp${ARCH^^}Skel.so"
ARCH_STUB_LIB="libQnnHtp${ARCH^^}Stub.so"
ARCH_CORE_LIB="libQnnHtp${ARCH^^}.so"
SKEL_DIR="$QNN/lib/hexagon-$ARCH/unsigned"

LIBCXX_DIR="$ROOT_DIR/.libcxx_shim/usr/lib/x86_64-linux-gnu"
if [ -f "$LIBCXX_DIR/libc++.so.1" ]; then
    export LD_LIBRARY_PATH="$LIBCXX_DIR:$QNN/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="$QNN/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="$QNN/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$QNN/bin/x86_64-linux-clang:$PATH"

remote_exec() {
    case "$CONNECT" in
        ssh) ssh "$DEVICE" "$1" ;;
        adb) adb -s "$DEVICE" shell "$1" ;;
    esac
}
remote_push() {
    case "$CONNECT" in
        ssh) ssh "$DEVICE" "cat > $DEVICE_DIR/$2" < "$1" ;;
        adb) adb -s "$DEVICE" push "$1" "$DEVICE_DIR/$2" >/dev/null ;;
    esac
}
remote_push_path() {
    local dst_dir
    dst_dir="$(dirname "$2")"
    case "$CONNECT" in
        ssh) ssh "$DEVICE" "mkdir -p $DEVICE_DIR/$dst_dir && cat > $DEVICE_DIR/$2" < "$1" ;;
        adb)
            adb -s "$DEVICE" shell "mkdir -p $DEVICE_DIR/$dst_dir" >/dev/null
            adb -s "$DEVICE" push "$1" "$DEVICE_DIR/$2" >/dev/null
            ;;
    esac
}
remote_pull_optional() {
    case "$CONNECT" in
        ssh) ssh "$DEVICE" "cat $DEVICE_DIR/$1 2>/dev/null" > "$2" 2>/dev/null ;;
        adb) adb -s "$DEVICE" pull "$DEVICE_DIR/$1" "$2" >/dev/null 2>&1 ;;
    esac
}
remote_pull_first() {
    local dst="$1"
    shift
    local src
    for src in "$@"; do
        if remote_pull_optional "$src" "$dst" && [ -s "$dst" ]; then
            return 0
        fi
    done
    rm -f "$dst"
    return 1
}

build_configs() {
    cat > "$OUT_DIR/_config.json" <<'EOF'
{
  "backend_extensions": {
    "shared_library_path": "./libQnnHtpNetRunExtensions.so",
    "config_file_path": "./_htp_ext.json"
  }
}
EOF
    cat > "$OUT_DIR/_config_host.json" <<EOF
{
  "backend_extensions": {
    "shared_library_path": "$QNN/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so",
    "config_file_path": "$OUT_DIR/_htp_ext.json"
  }
}
EOF
    cat > "$OUT_DIR/_htp_ext.json" <<EOF
{
  "devices": [
    {
      "dsp_arch": "$ARCH",
      "soc_id": $SOC_ID,
      "pd_session": "unsigned",
      "cores": [
        {"core_id": 0, "perf_profile": "burst", "rpc_control_latency": 100}
      ]
    }
  ]
}
EOF
}

ensure_device_libs() {
    remote_exec "mkdir -p $DEVICE_DIR"
    local needed="qnn-net-run libQnnHtp.so libQnnSystem.so $ARCH_SKEL_LIB $ARCH_STUB_LIB $ARCH_CORE_LIB"
    local probe
    probe=$(remote_exec "cd $DEVICE_DIR && for f in $needed; do [ -f \"\$f\" ] || echo missing=\"\$f\"; done")
    if echo "$probe" | grep -q missing=; then
        cat >&2 <<EOF
ERROR: $DEVICE_DIR is missing required QNN runtime files on device $DEVICE:
$(echo "$probe" | sed 's/^/  /')

One-time push from host:
  $QNN/bin/aarch64-android/qnn-net-run
  $QNN/lib/aarch64-android/libQnnHtp.so
  $QNN/lib/aarch64-android/libQnnSystem.so
  $QNN/lib/aarch64-android/$ARCH_STUB_LIB
  $SKEL_DIR/$ARCH_SKEL_LIB
  $SKEL_DIR/$ARCH_CORE_LIB
EOF
        exit 2
    fi
    remote_push "$QNN/lib/aarch64-android/libQnnModelDlc.so" "libQnnModelDlc.so"
    remote_push "$QNN/lib/aarch64-android/libQnnHtpNetRunExtensions.so" "libQnnHtpNetRunExtensions.so"
}

echo "=== native W4A16 Conv ref: device=$DEVICE via $CONNECT arch=$ARCH shape=${SHAPE_M}x${SHAPE_K}x${SHAPE_N} ==="

python "$SCRIPT_DIR/gen_native_w4a16_conv.py" "$OUT_DIR" \
    --m "$SHAPE_M" --k "$SHAPE_K" --n "$SHAPE_N"
build_configs

echo "=== qairt-converter ==="
qairt-converter \
    -i "$OUT_DIR/conv.onnx" \
    --target_backend HTP \
    --enable_framework_trace \
    --quantization_overrides "$OUT_DIR/quant_overrides.json" \
    --source_model_input_layout A NONTRIVIAL \
    --desired_input_layout A NONTRIVIAL \
    --source_model_output_layout Y NONTRIVIAL \
    --desired_output_layout Y NONTRIVIAL \
    -o "$OUT_DIR/conv.dlc" \
    > "$OUT_DIR/convert.log" 2>&1

echo "=== qnn-context-binary-generator ==="
rm -rf "$OUT_DIR/ctx"
( cd "$OUT_DIR" && qnn-context-binary-generator \
    --dlc_path "$OUT_DIR/conv.dlc" \
    --backend "$QNN/lib/x86_64-linux-clang/libQnnHtp.so" \
    --config_file "$OUT_DIR/_config_host.json" \
    --profiling_level detailed --profiling_option optrace \
    --save_backend_op_mapping \
    --binary_file conv_ctx \
    --output_dir ctx \
) > "$OUT_DIR/ctxgen.log" 2>&1
mkdir -p "$OUT_DIR/ctx"
for sch in "$OUT_DIR"/*schematic.bin "$OUT_DIR"/schematic.bin; do
    [ -f "$sch" ] || continue
    mv "$sch" "$OUT_DIR/ctx/$(basename "$sch")"
done

if [ "$SKIP_DEVICE" = "1" ]; then
    echo "=== done: host artifacts in $OUT_DIR ==="
    exit 0
fi

ensure_device_libs
echo "=== qnn-net-run --retrieve_context with native I/O ==="
remote_push "$OUT_DIR/ctx/conv_ctx.bin" "conv_ctx.bin"
remote_push "$OUT_DIR/_config.json" "_config.json"
remote_push "$OUT_DIR/_htp_ext.json" "_htp_ext.json"
remote_push_path "$OUT_DIR/runtime_inputs_native/A.raw" "runtime_inputs_native/A.raw"
remote_push "$OUT_DIR/runtime_input_list.txt" "input_list.txt"

remote_exec "cd $DEVICE_DIR && rm -rf out && \
LD_LIBRARY_PATH=.:/vendor/lib64 ADSP_LIBRARY_PATH=. \
./qnn-net-run \
  --backend ./libQnnHtp.so \
  --retrieve_context conv_ctx.bin \
  --input_list input_list.txt \
  --config_file _config.json \
  --profiling_level detailed --profiling_option optrace \
  --use_native_input_files --use_native_output_files \
  --num_inferences $NUM_INFERENCES \
  --output_dir out \
  --perf_profile burst 2>&1 | tail -10" > "$OUT_DIR/run.log" 2>&1 || true

if ! grep -q 'Finished Executing Graphs' "$OUT_DIR/run.log"; then
    echo "ERROR: qnn-net-run failed; see $OUT_DIR/run.log" >&2
    exit 1
fi

mkdir -p "$OUT_DIR/device_out"
remote_pull_optional "out/qnn-profiling-data_0.log" "$OUT_DIR/device_out/qnn-profiling-data_0.log" || true
cp -f "$OUT_DIR/device_out/qnn-profiling-data_0.log" "$OUT_DIR/profile.log"
remote_pull_first "$OUT_DIR/device_out/Y.raw" \
    "out/Result_0/Y_native.raw" \
    "out/Result_0/Y.raw" \
    "out/Y_native.raw" \
    "out/Y.raw" \
    >/dev/null || true

echo "=== decode optrace artifacts ==="
python "$ROOT_DIR/scripts/decode_qnn_optrace.py" "$OUT_DIR" \
    > "$OUT_DIR/_decode_optrace.log" 2>&1 || {
    echo "  [warn] optrace decode failed; see $OUT_DIR/_decode_optrace.log" >&2
}

CHECK_ARGS=("$OUT_DIR" --require-native-io --require-layout-flags --reject-float-io)
[ "${STRICT_ARTIFACT_STANDARD:-1}" = "0" ] && CHECK_ARGS+=(--warn-only)
python "$ROOT_DIR/scripts/check_qnn_artifact_standard.py" "${CHECK_ARGS[@]}"

echo "=== done: artifacts in $OUT_DIR ==="
