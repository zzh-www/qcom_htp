#!/usr/bin/env bash
#
# profile_all.sh — end-to-end QNN MatMul cycle profiling on a real device.
#
# For each dtype config (fp16, w16a16, w8a16, w8a8, w4a16, w4a8, w4a4):
#   1) gen ONNX + quant_overrides + fp32/native input data (Python helper)
#   2) qairt-converter                   -> .dlc, preserving public I/O layout
#   3) qnn-context-binary-generator      -> ctx/*.bin + schematic
#   4) push context binary + native input to device, run qnn-net-run with
#      --retrieve_context, native I/O, and --profiling_option optrace
#   5) pull profiling-data.log + native output
#   6) decode standard optrace artifacts under <config>/optrace/
# After all configs: parse all chrometrace.json -> summary table.
#
# Usage:
#   profile_all.sh [--device <id>] [--connect ssh|adb] [--arch v73|v75|v79|v81]
#
# Flags (env fallbacks in parens):
#   --device    DEVICE    device id / ssh host alias / adb serial (DEVICE, default oneplus)
#   --connect   CONNECT   transport: ssh | adb                    (CONNECT, default ssh)
#   --arch      ARCH      Hexagon HTP arch: v66/v68/v69/v73/v75/v79/v81 (ARCH, default v75)
#   --out-dir   DIR       artifact root                           (OUT_DIR, default ./output)
#   --configs   LIST      space-separated config list             (CONFIGS, default all)
#   --shape     M,K,N     matmul dimensions A[1,M,K] @ W[1,K,N]   (SHAPE, default 32,32,32)
#
# Environment:
#   NUM_INFERENCES=20     inferences per qnn-net-run invocation
#   NATIVE_IO=1           use qnn-net-run native input/output files
#   FLAT_OUT=0            write one config directly to OUT_DIR instead of
#                         OUT_DIR/<config>; requires exactly one config
#   DEVICE_DIR=...        remote working dir (default: adb => /data/local/tmp/qnn_run,
#                                                     ssh => ~/qnn_run)
#   SOC_ID=...            override HTP soc_id if auto-pick is wrong for the chip

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env.sh" >/dev/null
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/qairt_quant_flow.sh"

# ---- defaults ---------------------------------------------------------------
DEVICE="${DEVICE:-oneplus}"
CONNECT="${CONNECT:-ssh}"
ARCH="${ARCH:-v75}"
OUT_DIR="${OUT_DIR:-$PWD/output}"
CONFIGS="${CONFIGS:-fp16 w16a16 w8a16 w8a8 w4a16 w4a8 w4a4}"
SHAPE="${SHAPE:-32,32,32}"   # M,K,N
FLAT_OUT="${FLAT_OUT:-0}"

# ---- parse flags (flags override env) ---------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device|-d)    DEVICE="$2";    shift 2 ;;
        --connect|-c)   CONNECT="$2";   shift 2 ;;
        --arch|-a)      ARCH="$2";      shift 2 ;;
        --out-dir)      OUT_DIR="$2";   shift 2 ;;
        --configs)      CONFIGS="$2";   shift 2 ;;
        --shape)        SHAPE="$2";     shift 2 ;;
        --help|-h)
            sed -n '3,30p' "$0"; exit 0 ;;
        *)
            echo "unknown flag: $1" >&2; exit 2 ;;
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
# Parse M,K,N from --shape.
IFS=',' read -r SHAPE_M SHAPE_K SHAPE_N <<<"$SHAPE"
if [ -z "${SHAPE_M:-}" ] || [ -z "${SHAPE_K:-}" ] || [ -z "${SHAPE_N:-}" ]; then
    echo "bad --shape '$SHAPE' (expected M,K,N)" >&2; exit 2
fi

# Per-connect default DEVICE_DIR. ssh assumes Termux-style home; adb uses the
# only reliably writable path for a non-root user PD.
if [ -z "${DEVICE_DIR:-}" ]; then
    case "$CONNECT" in
        ssh) DEVICE_DIR="~/qnn_run" ;;
        adb) DEVICE_DIR="/data/local/tmp/qnn_run" ;;
    esac
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
if [ "$FLAT_OUT" = "1" ]; then
    config_count=$(set -- $CONFIGS; echo "$#")
    if [ "$config_count" != "1" ]; then
        echo "FLAT_OUT=1 requires exactly one config; got: $CONFIGS" >&2
        exit 2
    fi
fi

# Per-arch QNN SocModel enum (from tools/qnn-sdk/include/QNN/QnnTypes.h).
# This is NOT the hardware soc revision id (/sys/devices/soc0/soc_id);
# it's the QNN-internal enum the HTP backend validates against.
declare -A ARCH_SOCID=(
    [v66]=18    # SM8150 (closest available)
    [v68]=30    # SM8350
    [v69]=36    # SM8450
    [v73]=43    # SM8550
    [v75]=57    # SM8650
    [v79]=69    # SM8750
    [v81]=87    # SM8850
)
SOC_ID="${SOC_ID:-${ARCH_SOCID[$ARCH]:-0}}"

SCRIPT_DIR="$ROOT_DIR/example/qnn_matmul_profile"
QNN="$QNN_SDK_ROOT"
ARCH_SKEL_LIB="libQnnHtp${ARCH^^}Skel.so"    # e.g. libQnnHtpV75Skel.so
ARCH_STUB_LIB="libQnnHtp${ARCH^^}Stub.so"
ARCH_CORE_LIB="libQnnHtp${ARCH^^}.so"
SKEL_DIR="$QNN/lib/hexagon-$ARCH/unsigned"

# ---- libc++ shim for host qairt-converter (one-time) ------------------------
LIBCXX_DIR="$ROOT_DIR/.libcxx_shim/usr/lib/x86_64-linux-gnu"
if [ ! -f "$LIBCXX_DIR/libc++.so.1" ]; then
    echo "[host] extracting libc++ shim (one-time) ..."
    mkdir -p "$ROOT_DIR/.libcxx_shim"
    ( cd "$ROOT_DIR/.libcxx_shim" && {
        apt-get download libc++1-14 libc++abi1-14 libunwind8 2>&1 | tail -1 || true
        for deb in libc++1-14_*.deb libc++abi1-14_*.deb libunwind8_*.deb; do
            [ -f "$deb" ] && dpkg-deb -x "$deb" . && rm -f "$deb"
        done
    })
    ln -sf libunwind.so.8 "$LIBCXX_DIR/libunwind.so.1"
fi
export LD_LIBRARY_PATH="$LIBCXX_DIR:$QNN/lib/x86_64-linux-clang${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$QNN/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$QNN/bin/x86_64-linux-clang:$PATH"

# ---- transport abstraction (ssh | adb) --------------------------------------
remote_exec() {
    # remote_exec <shell-command-string>
    case "$CONNECT" in
        ssh) ssh "$DEVICE" "$1" ;;
        adb) adb -s "$DEVICE" shell "$1" ;;
    esac
}
remote_push() {
    # remote_push <local_file> <dst_basename>    (dst lives in $DEVICE_DIR)
    case "$CONNECT" in
        ssh) ssh "$DEVICE" "cat > $DEVICE_DIR/$2" < "$1" ;;
        adb) adb -s "$DEVICE" push "$1" "$DEVICE_DIR/$2" >/dev/null ;;
    esac
}
remote_push_path() {
    # remote_push_path <local_file> <dst_relative_path>  (dst lives under $DEVICE_DIR)
    local dst_dir
    dst_dir="$(dirname "$2")"
    case "$CONNECT" in
        ssh)
            ssh "$DEVICE" "mkdir -p $DEVICE_DIR/$dst_dir && cat > $DEVICE_DIR/$2" < "$1"
            ;;
        adb)
            adb -s "$DEVICE" shell "mkdir -p $DEVICE_DIR/$dst_dir" >/dev/null
            adb -s "$DEVICE" push "$1" "$DEVICE_DIR/$2" >/dev/null
            ;;
    esac
}
remote_pull() {
    # remote_pull <src_basename> <local_path>
    case "$CONNECT" in
        ssh) ssh "$DEVICE" "cat $DEVICE_DIR/$1" > "$2" ;;
        adb) adb -s "$DEVICE" pull "$DEVICE_DIR/$1" "$2" >/dev/null ;;
    esac
}
remote_pull_optional() {
    # remote_pull_optional <src_relative_path> <local_path>
    case "$CONNECT" in
        ssh) ssh "$DEVICE" "cat $DEVICE_DIR/$1 2>/dev/null" > "$2" 2>/dev/null ;;
        adb) adb -s "$DEVICE" pull "$DEVICE_DIR/$1" "$2" >/dev/null 2>&1 ;;
    esac
}
remote_pull_first() {
    # remote_pull_first <local_path> <src_relative_path>...
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

# ---- device config files (per-run; written to $OUT_DIR, pushed each config) -
build_device_configs() {
    # Two copies of the main config: one for qnn-net-run on-device (relative
    # paths, resolved against $DEVICE_DIR), one for host ctxgen (absolute
    # paths, so it works when ctxgen runs with a different CWD).
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
    cat > "$OUT_DIR/_optrace_config.json" <<'EOF'
{"enable_input_output_flow_events": false, "enable_sequencer_flow_events": false,
 "htp_json": true, "runtrace": true, "memory_info": true}
EOF
}

# ---- device prerequisites check + per-run lib push --------------------------
ensure_device_libs() {
    remote_exec "mkdir -p $DEVICE_DIR"

    # Sanity: the persistent runtime files must already be on device.
    local needed="qnn-net-run libQnnHtp.so libQnnSystem.so $ARCH_SKEL_LIB $ARCH_STUB_LIB $ARCH_CORE_LIB"
    local probe
    probe=$(remote_exec "cd $DEVICE_DIR && for f in $needed; do [ -f \"\$f\" ] || echo missing=\"\$f\"; done")
    if echo "$probe" | grep -q missing=; then
        cat >&2 <<EOF
ERROR: $DEVICE_DIR is missing required QNN runtime files on device $DEVICE:
$(echo "$probe" | sed 's/^/  /')

One-time push from host (edit paths if needed):
  $QNN/bin/aarch64-android/qnn-net-run
  $QNN/lib/aarch64-android/libQnnHtp.so
  $QNN/lib/aarch64-android/libQnnSystem.so
  $QNN/lib/aarch64-android/$ARCH_STUB_LIB
  $SKEL_DIR/$ARCH_SKEL_LIB
  $SKEL_DIR/$ARCH_CORE_LIB
EOF
        exit 2
    fi

    # Push the small per-run libs (cheap, avoids version skew).
    remote_push "$QNN/lib/aarch64-android/libQnnModelDlc.so"            "libQnnModelDlc.so"
    remote_push "$QNN/lib/aarch64-android/libQnnHtpNetRunExtensions.so" "libQnnHtpNetRunExtensions.so"
}

# ---- per-config pipeline ----------------------------------------------------
convert_and_ctx() {
    local name="$1" cfg_dir="$2"
    local onnx="$cfg_dir/matmul.onnx"
    local dlc="$cfg_dir/matmul.dlc"
    local encoded_dlc="$cfg_dir/matmul_encoded.dlc"

    echo "  [$name] convert ONNX -> DLC"
    local -a conv_args=(
        -i "$onnx"
        --target_backend HTP
        --enable_framework_trace
        --source_model_input_layout A NONTRIVIAL
        --desired_input_layout A NONTRIVIAL
        --source_model_output_layout Y NONTRIVIAL
        --desired_output_layout Y NONTRIVIAL
    )
    if [ "$name" = "fp16" ]; then
        conv_args+=(--float_bitwidth 16)
        conv_args+=(-o "$dlc")
        if ! qairt-converter "${conv_args[@]}" > "$cfg_dir/_convert.log" 2>&1; then
            echo "    [FAIL] qairt-converter (see $cfg_dir/_convert.log)"
            return 1
        fi
    else
        local act_bits weight_bits
        case "$name" in
            w16a16) act_bits=16; weight_bits=16 ;;
            w8a16) act_bits=16; weight_bits=8 ;;
            w8a8) act_bits=8; weight_bits=8 ;;
            w4a16) act_bits=16; weight_bits=4 ;;
            w4a8) act_bits=8; weight_bits=4 ;;
            w4a4) act_bits=4; weight_bits=4 ;;
            *) echo "    [FAIL] unknown quantized config $name" >&2; return 1 ;;
        esac
        conv_args+=(--quantization_overrides "$cfg_dir/quant_overrides.json")
        conv_args+=(-o "$encoded_dlc")
        if ! qairt-converter "${conv_args[@]}" > "$cfg_dir/_convert.log" 2>&1; then
            echo "    [FAIL] qairt-converter (see $cfg_dir/_convert.log)"
            return 1
        fi
        local pack_4bit=0
        case "$name" in
            w4*) pack_4bit=1 ;;
        esac
        if ! qairt_quantize_encoded_dlc \
                "$encoded_dlc" "$dlc" \
                "$act_bits" "$weight_bits" 32 "$pack_4bit" \
                "$cfg_dir/_quantize.log"; then
            echo "    [FAIL] qairt-quantizer (see $cfg_dir/_quantize.log)"
            return 1
        fi
    fi

    echo "  [$name] generate context binary + schematic"
    # ctxgen drops matmul_schematic.bin in CWD (ignoring --output_dir), so
    # run it from inside cfg_dir to keep the artifact beside the DLC.
    local abs_dlc="$cfg_dir/matmul.dlc"
    # Without --config_file, ctxgen defaults HTP arch to v68 and rejects
    # kernels (e.g. int16×int16 MatMul) that require v73+. Pass the same
    # backend_extensions config we use at runtime.
    if ! ( cd "$cfg_dir" && qnn-context-binary-generator \
            --dlc_path "$abs_dlc" \
            --backend "$QNN/lib/x86_64-linux-clang/libQnnHtp.so" \
            --config_file "$OUT_DIR/_config_host.json" \
            --profiling_level detailed --profiling_option optrace \
            --save_backend_op_mapping \
            --binary_file "matmul_${name}_ctx" \
            --output_dir "ctx" \
        ) > "$cfg_dir/_ctxgen.log" 2>&1; then
        echo "    [FAIL] ctxgen failed (see $cfg_dir/_ctxgen.log)"
        return 1
    fi
    mkdir -p "$cfg_dir/ctx"
    local sch
    for sch in "$cfg_dir"/*schematic.bin "$cfg_dir"/schematic.bin; do
        [ -f "$sch" ] || continue
        mv "$sch" "$cfg_dir/ctx/$(basename "$sch")"
    done
    if [ ! -f "$cfg_dir/ctx/matmul_${name}_ctx.bin" ]; then
        echo "    [FAIL] missing ctx/matmul_${name}_ctx.bin after ctxgen"
        return 1
    fi
    return 0
}

run_on_device() {
    local name="$1" cfg_dir="$2"
    echo "  [$name] push + run on $DEVICE ($CONNECT)"

    remote_push "$cfg_dir/ctx/matmul_${name}_ctx.bin" "matmul_${name}_ctx.bin"
    remote_push "$OUT_DIR/_config.json"    "_config.json"
    remote_push "$OUT_DIR/_htp_ext.json"   "_htp_ext.json"

    local input_flags=""
    if [ "${NATIVE_IO:-1}" = "1" ]; then
        remote_push_path "$cfg_dir/runtime_inputs_native/A.raw" "runtime_inputs_native/A.raw"
        remote_push "$cfg_dir/runtime_input_list.txt" "input_list.txt"
        input_flags="--use_native_input_files --use_native_output_files"
    else
        remote_push "$cfg_dir/input_A.raw"    "input_A.raw"
        remote_push "$cfg_dir/input_list.txt" "input_list.txt"
    fi

    local cmd="cd $DEVICE_DIR && rm -rf out && \
LD_LIBRARY_PATH=.:/vendor/lib64 ADSP_LIBRARY_PATH=. \
./qnn-net-run \
    --backend ./libQnnHtp.so \
    --retrieve_context matmul_${name}_ctx.bin \
    --input_list input_list.txt \
    --config_file _config.json \
    --profiling_level detailed --profiling_option optrace \
    --perf_profile burst \
    $input_flags \
    --num_inferences ${NUM_INFERENCES:-20} \
    --output_dir out 2>&1 | tail -6"
    remote_exec "$cmd" > "$cfg_dir/_run.log" 2>&1 || true

    if ! grep -q 'Finished Executing Graphs' "$cfg_dir/_run.log"; then
        echo "    [FAIL] qnn-net-run (see $cfg_dir/_run.log)"
        return 1
    fi
    mkdir -p "$cfg_dir/device_out"
    remote_pull "out/qnn-profiling-data_0.log" "$cfg_dir/device_out/qnn-profiling-data_0.log"
    cp -f "$cfg_dir/device_out/qnn-profiling-data_0.log" "$cfg_dir/profile.log"
    remote_pull_first "$cfg_dir/device_out/Y.raw" \
        "out/Result_0/Y_native.raw" \
        "out/Result_0/Y.raw" \
        "out/Y_native.raw" \
        "out/Y.raw" \
        >/dev/null || true
}

postprocess() {
    local name="$1" cfg_dir="$2"
    echo "  [$name] decode standard optrace artifacts"
    python "$ROOT_DIR/scripts/decode_qnn_optrace.py" "$cfg_dir" \
        > "$cfg_dir/_decode_optrace.log" 2>&1 || {
        echo "    [warn] optrace decode failed; see $cfg_dir/_decode_optrace.log"
        return 1
    }
    local -a check_args=("$cfg_dir" --require-native-io --require-layout-flags)
    [ "$name" != "fp16" ] && check_args+=(--reject-float-io)
    [ "${STRICT_ARTIFACT_STANDARD:-1}" = "0" ] && check_args+=(--warn-only)
    python "$ROOT_DIR/scripts/check_qnn_artifact_standard.py" "${check_args[@]}"
}

# ---- main -------------------------------------------------------------------
echo "=== target: device=$DEVICE via $CONNECT  arch=$ARCH  soc_id=$SOC_ID  shape=${SHAPE_M}x${SHAPE_K}x${SHAPE_N} ==="

build_device_configs
ensure_device_libs

for name in $CONFIGS; do
    if [ "$FLAT_OUT" = "1" ]; then
        cfg_dir="$OUT_DIR"
    else
        cfg_dir="$OUT_DIR/$name"
    fi
    mkdir -p "$cfg_dir"

    echo "=== $name ==="
    python "$SCRIPT_DIR/gen_onnx.py" "$name" "$cfg_dir" \
        --m "$SHAPE_M" --k "$SHAPE_K" --n "$SHAPE_N" \
        || { echo "    [FAIL] gen_onnx"; continue; }
    convert_and_ctx "$name" "$cfg_dir" || continue
    run_on_device   "$name" "$cfg_dir" || continue
    postprocess     "$name" "$cfg_dir" || continue
done

if [ "$FLAT_OUT" != "1" ]; then
    echo
    echo "=== Summary (device=$DEVICE, arch=$ARCH, shape=${SHAPE_M}x${SHAPE_K}x${SHAPE_N}) ==="
    # Primary summary — QHAS with the profiler-reader UNK fixup applied.
    python "$SCRIPT_DIR/parse_qhas.py" "$OUT_DIR" || true
    echo
    echo "--- matmul event breakdown (first-inference chrometrace) ---"
    python "$SCRIPT_DIR/parse_chrometrace.py" "$OUT_DIR" || true
fi
