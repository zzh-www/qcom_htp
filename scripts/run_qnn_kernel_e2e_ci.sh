#!/usr/bin/env bash
#
# Run or validate QNN HMX MatMul kernel CI groups.
#
# Default mode rebuilds the selected package, runs the device E2E flow, then
# validates the produced artifact against the retained native reference.  Set
# ARTIFACT_ONLY=1 to validate the retained standard artifacts without rerunning
# the device flow.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Harden device ssh: multiplex all `ssh "$DEVICE"` calls in the run scripts over one ControlMaster
# (termux sshd hangs ~10% of fresh per-connection ssh; a long multi-ssh gate would otherwise hang).
# shellcheck source=scripts/ci_ssh_mux.sh
. "$ROOT_DIR/scripts/ci_ssh_mux.sh"

usage() {
    cat <<'EOF'
Usage:
  scripts/run_qnn_kernel_e2e_ci.sh all|correctness|performance|lpbq|u8i8|u8i8_native_match|w4a8_per_channel|w4a8_per_channel_native_match|w4a8_lpbq|w4a8_lpbq_native_match|w8a16|w8a16_per_channel_native_match|w4a16_per_channel|w4a16_per_channel_native_match|w4a16_per_channel_chain1|w4a16_lpbq|w4a16_lpbq_native_match

Environment:
  DEVICE=oneplus                  SSH target for qnn-net-run device execution.
  ARTIFACT_ONLY=1                 Validate retained artifacts only.
  BUILD_PACKAGES=0                Skip package rebuild in device-run mode.
  KERNEL_E2E_OUT_ROOT=<dir>       Output root for regenerated custom artifacts.
  U8I8_MATCH_M=256                M/K/N size for the u8i8 custom/native exactness gate.
  U8I8_MATCH_CHAIN=1              Chain length for the u8i8 custom/native exactness gate.
  U8I8_MATCH_CASE=normal_random   Float-derived u8i8 Python/QNN case for the custom/native gate.
  W4A8_MATCH_M=256                M size for the w4a8 per-channel custom/native gate.
  W4A8_MATCH_K=256                K size for the w4a8 per-channel custom/native gate.
  W4A8_MATCH_N=256                N size for the w4a8 per-channel custom/native gate.
  W4A8_MATCH_CASES="normal_random zp_neutral positive_boundary negative_boundary single_k_impulse bias_only scale_only"
                                  Float-derived w4a8 per-channel Python/QNN cases for the custom/native gate.
                                  Also used by w4a8_lpbq_native_match.
  W8A16_MATCH_M=256               M size for the w8a16 per-channel custom/native gate.
  W8A16_MATCH_K=256               K size for the w8a16 per-channel custom/native gate.
  W8A16_MATCH_N=256               N size for the w8a16 per-channel custom/native gate.
  W8A16_MATCH_CASES="normal_random zp_neutral positive_boundary negative_boundary single_k_impulse bias_only scale_only"
                                  Float-derived w8a16 Python/QNN cases for the custom/native gate.
  W8A16_SIDECAR_SOURCE=generated   W8A16 custom sidecar source: generated, hybrid, or native_final.
  W4A16_MATCH_M=256               M size for the w4a16 per-channel custom/native gate.
  W4A16_MATCH_K=256               K size for the w4a16 per-channel custom/native gate.
  W4A16_MATCH_N=256               N size for the w4a16 per-channel custom/native gate.
  W4A16_MATCH_CASES="normal_random zp_neutral positive_boundary negative_boundary single_k_impulse bias_only scale_only"
                                  Float-derived w4a16 per-channel Python/QNN cases for the custom/native gate.
                                  Also used by w4a16_lpbq_native_match.
  W4A16_CHAIN1_CASES="default zp k_impulse0 k_impulse1 k_impulse7"
                                  Case list for the W4A16 CHAIN=1 precision suite.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ $# -ne 1 ]; then
    usage
    exit 2
fi

KERNEL="$1"
if [ "$KERNEL" = "all" ]; then
    "$0" correctness
    "$0" performance
    exit 0
fi
if [ "$KERNEL" = "correctness" ]; then
    items=(
        u8i8_native_match
        w4a8_per_channel_native_match
        w4a8_lpbq_native_match
        w8a16_per_channel_native_match
        w4a16_per_channel_native_match
        w4a16_lpbq_native_match
        w4a16_per_channel_chain1
    )
    for item in "${items[@]}"; do
        "$0" "$item"
    done
    exit 0
fi
if [ "$KERNEL" = "performance" ]; then
    items=(u8i8 w4a8_per_channel w8a16 w4a16_per_channel)
    for item in "${items[@]}"; do
        "$0" "$item"
    done
    exit 0
fi
if [ "$KERNEL" = "lpbq" ]; then
    items=(w4a8_lpbq_native_match w4a16_lpbq_native_match w4a8_lpbq w4a16_lpbq)
    for item in "${items[@]}"; do
        "$0" "$item"
    done
    exit 0
fi

PROFILE_DIR="$ROOT_DIR/example/qnn_matmul_profile"
ARTIFACT_ONLY="${ARTIFACT_ONLY:-0}"
BUILD_PACKAGES="${BUILD_PACKAGES:-1}"
OUT_ROOT="${KERNEL_E2E_OUT_ROOT:-$PROFILE_DIR}"
RUN_NATIVE_W4A16_CHAIN=""
RUN_MATCHED_U8I8_NATIVE=""
RUN_MATCHED_W4A8_NATIVE=""
RUN_MATCHED_W8A16_NATIVE=""
RUN_MATCHED_W4A16_NATIVE=""
MATCH_W4_ENCODING="symmetric"
REF_FLAG=()

case "$KERNEL" in
    w4a8)
        echo "ERROR: target 'w4a8' is ambiguous; use w4a8_per_channel or w4a8_lpbq" >&2
        exit 2
        ;;
    w4a16)
        echo "ERROR: target 'w4a16' is ambiguous; use w4a16_per_channel or w4a16_lpbq" >&2
        exit 2
        ;;
    u8i8)
        EXAMPLE_DIR="$ROOT_DIR/example/qnn_hmx_matmul_u8i8"
        RUN_SCRIPT="$EXAMPLE_DIR/standard_flow/custom_u8i8/run_u8i8_chain.sh"
        RETAINED_CUSTOM="$PROFILE_DIR/output_u8i8_aligned_e2e_256"
        NATIVE_DIR="$PROFILE_DIR/output_u8i8_native_ref_e2e_256"
        CI_CUSTOM="$OUT_ROOT/output_u8i8_ci_e2e_256"
        HTP_TYPE="QnnHmxMatMulU8I8Package::HmxU8I8ToU8MatMul"
        QNN_PREFIX="hmx_u8i8_chain"
        EXPECTED_QNN_OPS=8
        DTYPE="uint8"
        BUILD_ENV=()
        RUN_ENV=(M=256 K=256 N=256 CHAIN=8 MODE=chain)
        TRANSPOSE_FLAG=()
        LPBQ_FLAG=()
        ;;
    u8i8_native_match)
        EXAMPLE_DIR="$ROOT_DIR/example/qnn_hmx_matmul_u8i8"
        RUN_SCRIPT="$EXAMPLE_DIR/standard_flow/custom_u8i8/run_u8i8_chain.sh"
        RETAINED_CUSTOM="$PROFILE_DIR/output_u8i8_aligned_e2e_256"
        NATIVE_DIR="$PROFILE_DIR/output_u8i8_native_ref_e2e_256"
        CI_CUSTOM="$OUT_ROOT/output_u8i8_native_match_custom_ci"
        CI_NATIVE="$OUT_ROOT/output_u8i8_native_match_native_ci"
        RUN_MATCHED_U8I8_NATIVE=1
        HTP_TYPE="QnnHmxMatMulU8I8Package::HmxU8I8ToU8MatMul"
        QNN_PREFIX="hmx_u8i8_chain"
        EXPECTED_QNN_OPS="${U8I8_MATCH_CHAIN:-1}"
        DTYPE="uint8"
        BUILD_ENV=()
        RUN_ENV=()
        TRANSPOSE_FLAG=()
        LPBQ_FLAG=()
        ;;
    w4a8_per_channel)
        EXAMPLE_DIR="$ROOT_DIR/example/qnn_hmx_matmul_w4a8"
        RUN_SCRIPT="$EXAMPLE_DIR/standard_flow/custom_w4a8/run_w4a8_chain.sh"
        RETAINED_CUSTOM="$PROFILE_DIR/output_w4a8_aligned_e2e_256"
        NATIVE_DIR="$PROFILE_DIR/output_w4a8_native_ref_e2e_256"
        CI_CUSTOM="$OUT_ROOT/output_w4a8_ci_e2e_256"
        HTP_TYPE="QnnHmxMatMulW4A8Package::HmxU8I4ToU8MatMul"
        QNN_PREFIX="hmx_w4a8_chain"
        EXPECTED_QNN_OPS=8
        DTYPE="uint8"
        BUILD_ENV=()
        RUN_ENV=(M=256 K=256 N=256 CHAIN=8 MODE=chain)
        TRANSPOSE_FLAG=()
        LPBQ_FLAG=()
        ;;
    w4a8_per_channel_native_match)
        EXAMPLE_DIR="$ROOT_DIR/example/qnn_hmx_matmul_w4a8"
        RUN_SCRIPT="$EXAMPLE_DIR/standard_flow/custom_w4a8/run_w4a8_chain.sh"
        RETAINED_CUSTOM="$PROFILE_DIR/output_w4a8_aligned_e2e_256"
        NATIVE_DIR="$PROFILE_DIR/output_w4a8_native_ref_e2e_256"
        CI_CUSTOM="$OUT_ROOT/output_w4a8_per_channel_native_match_custom_ci"
        CI_NATIVE="$OUT_ROOT/output_w4a8_per_channel_native_match_native_ci"
        RUN_MATCHED_W4A8_NATIVE=1
        HTP_TYPE="QnnHmxMatMulW4A8Package::HmxU8I4ToU8MatMul"
        QNN_PREFIX="hmx_w4a8_chain"
        EXPECTED_QNN_OPS=1
        DTYPE="uint8"
        BUILD_ENV=()
        RUN_ENV=()
        TRANSPOSE_FLAG=()
        LPBQ_FLAG=()
        ;;
    w4a8_lpbq)
        EXAMPLE_DIR="$ROOT_DIR/example/qnn_hmx_matmul_w4a8"
        RUN_SCRIPT="$EXAMPLE_DIR/standard_flow/custom_w4a8/run_w4a8_chain.sh"
        RETAINED_CUSTOM="$PROFILE_DIR/output_w4a8_lpbq_ci_e2e_256"
        NATIVE_DIR="$PROFILE_DIR/output_w4a8_native_ref_e2e_256"
        CI_CUSTOM="$OUT_ROOT/output_w4a8_lpbq_ci_e2e_256"
        HTP_TYPE="QnnHmxMatMulW4A8Package::HmxU8I4ToU8MatMul"
        QNN_PREFIX="hmx_w4a8_chain"
        EXPECTED_QNN_OPS=8
        DTYPE="uint8"
        BUILD_ENV=(LPBQ_ONLY=1)
        RUN_ENV=(M=256 K=256 N=256 CHAIN=8 MODE=chain W4_ENCODING=lpbq)
        TRANSPOSE_FLAG=()
        LPBQ_FLAG=(--expect-lpbq)
        ;;
    w4a8_lpbq_native_match)
        EXAMPLE_DIR="$ROOT_DIR/example/qnn_hmx_matmul_w4a8"
        RUN_SCRIPT="$EXAMPLE_DIR/standard_flow/custom_w4a8/run_w4a8_chain.sh"
        RETAINED_CUSTOM="$PROFILE_DIR/output_w4a8_lpbq_e2e_256"
        NATIVE_DIR="$PROFILE_DIR/output_w4a8_native_ref_e2e_256"
        CI_CUSTOM="$OUT_ROOT/output_w4a8_lpbq_native_match_custom_ci"
        CI_NATIVE="$OUT_ROOT/output_w4a8_lpbq_native_match_native_ci"
        RUN_MATCHED_W4A8_NATIVE=1
        MATCH_W4_ENCODING="lpbq"
        HTP_TYPE="QnnHmxMatMulW4A8Package::HmxU8I4ToU8MatMul"
        QNN_PREFIX="hmx_w4a8_chain"
        EXPECTED_QNN_OPS=1
        DTYPE="uint8"
        BUILD_ENV=(LPBQ_ONLY=1)
        RUN_ENV=()
        TRANSPOSE_FLAG=()
        LPBQ_FLAG=(--expect-lpbq)
        ;;
    w8a16)
        EXAMPLE_DIR="$ROOT_DIR/example/qnn_hmx_matmul_w8a16"
        RUN_SCRIPT="$EXAMPLE_DIR/standard_flow/custom_w8a16/run_w8a16_chain.sh"
        RETAINED_CUSTOM="$PROFILE_DIR/output_w8a16_aligned_e2e_256"
        NATIVE_DIR="$PROFILE_DIR/output_w8a16_native_ref_e2e_256"
        CI_CUSTOM="$OUT_ROOT/output_w8a16_ci_e2e_256"
        HTP_TYPE="QnnHmxMatMulW8A16Package::HmxU16I8ToU16MatMul"
        QNN_PREFIX="hmx_w8a16_chain"
        EXPECTED_QNN_OPS=8
        DTYPE="uint16"
        BUILD_ENV=()
        RUN_ENV=(M=256 K=256 N=256 CHAIN=8 MODE=chain_qdq OP_INPUT_LAYOUT=tiled VERIFY_NATIVE_RAW="$NATIVE_DIR/device_out/Y.raw")
        TRANSPOSE_FLAG=()
        LPBQ_FLAG=()
        ;;
    w8a16_per_channel_native_match)
        EXAMPLE_DIR="$ROOT_DIR/example/qnn_hmx_matmul_w8a16"
        RUN_SCRIPT="$EXAMPLE_DIR/standard_flow/custom_w8a16/run_w8a16_chain.sh"
        RETAINED_CUSTOM="$PROFILE_DIR/output_w8a16_aligned_e2e_256"
        NATIVE_DIR="$PROFILE_DIR/output_w8a16_native_ref_e2e_256"
        CI_CUSTOM="$OUT_ROOT/output_w8a16_per_channel_native_match_custom_ci"
        CI_NATIVE="$OUT_ROOT/output_w8a16_per_channel_native_match_native_ci"
        RUN_MATCHED_W8A16_NATIVE=1
        HTP_TYPE="QnnHmxMatMulW8A16Package::HmxU16I8ToU16MatMul"
        QNN_PREFIX="hmx_w8a16_chain"
        EXPECTED_QNN_OPS=1
        DTYPE="uint16"
        BUILD_ENV=()
        RUN_ENV=()
        TRANSPOSE_FLAG=()
        LPBQ_FLAG=()
        ;;
    w4a16_per_channel)
        EXAMPLE_DIR="$ROOT_DIR/example/qnn_hmx_matmul_w4a16"
        RUN_SCRIPT="$EXAMPLE_DIR/standard_flow/custom_w4a16/run_w4a16_chain.sh"
        RETAINED_CUSTOM="$PROFILE_DIR/output_w4a16_aligned_e2e_256"
        NATIVE_DIR="$PROFILE_DIR/output_w4a16_native_ref_e2e_256"
        CI_CUSTOM="$OUT_ROOT/output_w4a16_ci_e2e_256"
        HTP_TYPE="QnnHmxMatMulW4A16Package::HmxU16I4ToU16MatMul"
        QNN_PREFIX="hmx_w4a16_chain"
        EXPECTED_QNN_OPS=8
        DTYPE="uint16"
        BUILD_ENV=(EXTRA_DEFS="-UHMX_W4A16_SKIP_KERNEL -DHMX_W4A16_ALLOW_UNVALIDATED_KERNEL")
        RUN_ENV=(M=256 K=256 N=256 CHAIN=8 MODE=chain_qdq OP_INPUT_LAYOUT=tiled VERIFY_NATIVE_RAW="$NATIVE_DIR/device_out/Y.raw" VERIFY_NATIVE_TRANSPOSE=1)
        TRANSPOSE_FLAG=(--native-transpose-2d)
        LPBQ_FLAG=()
        ;;
    w4a16_per_channel_native_match)
        EXAMPLE_DIR="$ROOT_DIR/example/qnn_hmx_matmul_w4a16"
        RUN_SCRIPT="$EXAMPLE_DIR/standard_flow/custom_w4a16/run_w4a16_chain.sh"
        RETAINED_CUSTOM="$PROFILE_DIR/output_w4a16_aligned_e2e_256"
        NATIVE_DIR="$PROFILE_DIR/output_w4a16_native_ref_e2e_256"
        CI_CUSTOM="$OUT_ROOT/output_w4a16_per_channel_native_match_custom_ci"
        CI_NATIVE="$OUT_ROOT/output_w4a16_per_channel_native_match_native_ci"
        RUN_MATCHED_W4A16_NATIVE=1
        HTP_TYPE="QnnHmxMatMulW4A16Package::HmxU16I4ToU16MatMul"
        QNN_PREFIX="hmx_w4a16_chain"
        EXPECTED_QNN_OPS=1
        DTYPE="uint16"
        BUILD_ENV=(EXTRA_DEFS="-UHMX_W4A16_SKIP_KERNEL -DHMX_W4A16_ALLOW_UNVALIDATED_KERNEL")
        RUN_ENV=()
        TRANSPOSE_FLAG=(--native-transpose-2d)
        LPBQ_FLAG=()
        ;;
    w4a16_per_channel_chain1)
        EXAMPLE_DIR="$ROOT_DIR/example/qnn_hmx_matmul_w4a16"
        RUN_SCRIPT="$EXAMPLE_DIR/standard_flow/custom_w4a16/run_w4a16_chain.sh"
        RETAINED_CUSTOM="$PROFILE_DIR/output_w4a16_chain1_e2e_256"
        NATIVE_DIR="$PROFILE_DIR/output_w4a16_native_chain1_e2e_256"
        CI_CUSTOM="$OUT_ROOT/output_w4a16_chain1_ci_e2e_256"
        CI_NATIVE="$OUT_ROOT/output_w4a16_native_chain1_ci_e2e_256"
        RUN_NATIVE_W4A16_CHAIN=1
        HTP_TYPE="QnnHmxMatMulW4A16Package::HmxU16I4ToU16MatMul"
        QNN_PREFIX="hmx_w4a16_chain"
        EXPECTED_QNN_OPS=1
        DTYPE="uint16"
        BUILD_ENV=(EXTRA_DEFS="-UHMX_W4A16_SKIP_KERNEL -DHMX_W4A16_ALLOW_UNVALIDATED_KERNEL")
        RUN_ENV=(M=256 K=256 N=256 CHAIN=1 MODE=chain_qdq OP_INPUT_LAYOUT=tiled VERIFY_NATIVE_TRANSPOSE=1 VERIFY_ABS_TOL=0)
        TRANSPOSE_FLAG=(--native-transpose-2d)
        LPBQ_FLAG=()
        REF_FLAG=(--expect-ref-exact)
        ;;
    w4a16_lpbq)
        EXAMPLE_DIR="$ROOT_DIR/example/qnn_hmx_matmul_w4a16"
        RUN_SCRIPT="$EXAMPLE_DIR/standard_flow/custom_w4a16/run_w4a16_chain.sh"
        RETAINED_CUSTOM="$PROFILE_DIR/output_w4a16_lpbq_ci_e2e_256"
        NATIVE_DIR="$PROFILE_DIR/output_w4a16_native_ref_e2e_256"
        CI_CUSTOM="$OUT_ROOT/output_w4a16_lpbq_ci_e2e_256"
        HTP_TYPE="QnnHmxMatMulW4A16Package::HmxU16I4ToU16MatMul"
        QNN_PREFIX="hmx_w4a16_chain"
        EXPECTED_QNN_OPS=8
        DTYPE="uint16"
        BUILD_ENV=(LPBQ_ONLY=1)
        RUN_ENV=(M=256 K=256 N=256 CHAIN=8 MODE=chain_qdq OP_INPUT_LAYOUT=tiled W4_ENCODING=lpbq VERIFY_NATIVE_RAW="$NATIVE_DIR/device_out/Y.raw" VERIFY_NATIVE_TRANSPOSE=1)
        TRANSPOSE_FLAG=(--native-transpose-2d)
        LPBQ_FLAG=(--expect-lpbq)
        ;;
    w4a16_lpbq_native_match)
        EXAMPLE_DIR="$ROOT_DIR/example/qnn_hmx_matmul_w4a16"
        RUN_SCRIPT="$EXAMPLE_DIR/standard_flow/custom_w4a16/run_w4a16_chain.sh"
        RETAINED_CUSTOM="$PROFILE_DIR/output_w4a16_lpbq_e2e_256"
        NATIVE_DIR="$PROFILE_DIR/output_w4a16_native_ref_e2e_256"
        CI_CUSTOM="$OUT_ROOT/output_w4a16_lpbq_native_match_custom_ci"
        CI_NATIVE="$OUT_ROOT/output_w4a16_lpbq_native_match_native_ci"
        RUN_MATCHED_W4A16_NATIVE=1
        MATCH_W4_ENCODING="lpbq"
        HTP_TYPE="QnnHmxMatMulW4A16Package::HmxU16I4ToU16MatMul"
        QNN_PREFIX="hmx_w4a16_chain"
        EXPECTED_QNN_OPS=1
        DTYPE="uint16"
        BUILD_ENV=(LPBQ_ONLY=1 EXTRA_DEFS="-UHMX_W4A16_SKIP_KERNEL -DHMX_W4A16_ALLOW_UNVALIDATED_KERNEL")
        RUN_ENV=()
        TRANSPOSE_FLAG=(--native-transpose-2d)
        LPBQ_FLAG=(--expect-lpbq)
        ;;
    *)
        echo "ERROR: unknown kernel '$KERNEL'" >&2
        usage >&2
        exit 2
        ;;
esac

CUSTOM_DIR="$RETAINED_CUSTOM"
if [ "$ARTIFACT_ONLY" = "1" ] && [ -n "$RUN_NATIVE_W4A16_CHAIN" ]; then
    echo "=== skip $KERNEL in ARTIFACT_ONLY mode: CHAIN=1 native oracle is generated on demand ==="
    exit 0
fi
if [ "$ARTIFACT_ONLY" = "1" ] && [ -n "$RUN_MATCHED_U8I8_NATIVE" ]; then
    echo "=== skip $KERNEL in ARTIFACT_ONLY mode: matched native oracle is generated on demand ==="
    exit 0
fi
if [ "$ARTIFACT_ONLY" = "1" ] && [ -n "$RUN_MATCHED_W4A8_NATIVE" ]; then
    echo "=== skip $KERNEL in ARTIFACT_ONLY mode: matched native oracle is generated on demand ==="
    exit 0
fi
if [ "$ARTIFACT_ONLY" = "1" ] && [ -n "$RUN_MATCHED_W8A16_NATIVE" ]; then
    echo "=== skip $KERNEL in ARTIFACT_ONLY mode: matched native oracle is generated on demand ==="
    exit 0
fi
if [ "$ARTIFACT_ONLY" = "1" ] && [ -n "$RUN_MATCHED_W4A16_NATIVE" ]; then
    echo "=== skip $KERNEL in ARTIFACT_ONLY mode: matched native oracle is generated on demand ==="
    exit 0
fi

if [ "$ARTIFACT_ONLY" != "1" ]; then
    CUSTOM_DIR="$CI_CUSTOM"
    if [ "$BUILD_PACKAGES" = "1" ]; then
        echo "=== build $KERNEL package ==="
        env "${BUILD_ENV[@]}" bash "$EXAMPLE_DIR/build.sh"
        env "${BUILD_ENV[@]}" bash "$EXAMPLE_DIR/build_x86.sh"
    fi

    if [ -n "$RUN_MATCHED_U8I8_NATIVE" ]; then
        match_size="${U8I8_MATCH_M:-256}"
        match_chain="${U8I8_MATCH_CHAIN:-1}"
        match_case="${U8I8_MATCH_CASE:-normal_random}"
        echo "=== run $KERNEL Python-case custom/native exactness on ${DEVICE:-oneplus} ==="
        OUT_ROOT="$OUT_ROOT/output_u8i8_native_match_${match_case}_ci" \
            CASE_NAME="$match_case" \
            M="$match_size" K="$match_size" N="$match_size" CHAIN="$match_chain" \
            BUILD_PACKAGES=0 DEVICE="${DEVICE:-oneplus}" \
            bash "$ROOT_DIR/scripts/run_u8i8_python_case_custom_native_match.sh"
        echo "PASS: $KERNEL Python-case custom/native exactness"
        exit 0
    fi

    if [ -n "$RUN_MATCHED_W4A8_NATIVE" ]; then
        match_m="${W4A8_MATCH_M:-256}"
        match_k="${W4A8_MATCH_K:-256}"
        match_n="${W4A8_MATCH_N:-256}"
        match_root="output_w4a8_per_channel_native_match_ci"
        if [ "$MATCH_W4_ENCODING" = "lpbq" ]; then
            match_root="output_w4a8_lpbq_native_match_ci"
        fi
        echo "=== run $KERNEL Python-case custom/native exactness on ${DEVICE:-oneplus} ==="
        OUT_ROOT="$OUT_ROOT/$match_root" \
            M="$match_m" K="$match_k" N="$match_n" CHAIN=1 \
            W4_ENCODING="$MATCH_W4_ENCODING" BUILD_PACKAGES=0 DEVICE="${DEVICE:-oneplus}" \
            bash "$ROOT_DIR/scripts/run_w4a8_python_case_custom_native_match.sh"
        echo "PASS: $KERNEL Python-case custom/native exactness"
        exit 0
    fi

    if [ -n "$RUN_MATCHED_W8A16_NATIVE" ]; then
        match_m="${W8A16_MATCH_M:-256}"
        match_k="${W8A16_MATCH_K:-256}"
        match_n="${W8A16_MATCH_N:-256}"
        echo "=== run $KERNEL Python-case custom/native exactness on ${DEVICE:-oneplus} ==="
        OUT_ROOT="$OUT_ROOT/output_w8a16_per_channel_native_match_ci" \
            M="$match_m" K="$match_k" N="$match_n" CHAIN=1 \
            BUILD_PACKAGES=0 DEVICE="${DEVICE:-oneplus}" \
            bash "$ROOT_DIR/scripts/run_w8a16_python_case_custom_native_match.sh"
        echo "PASS: $KERNEL Python-case custom/native exactness"
        exit 0
    fi

    if [ -n "$RUN_MATCHED_W4A16_NATIVE" ]; then
        match_m="${W4A16_MATCH_M:-256}"
        match_k="${W4A16_MATCH_K:-256}"
        match_n="${W4A16_MATCH_N:-256}"
        match_root="output_w4a16_per_channel_native_match_ci"
        if [ "$MATCH_W4_ENCODING" = "lpbq" ]; then
            match_root="output_w4a16_lpbq_native_match_ci"
        fi
        echo "=== run $KERNEL Python-case custom/native exactness on ${DEVICE:-oneplus} ==="
        OUT_ROOT="$OUT_ROOT/$match_root" \
            M="$match_m" K="$match_k" N="$match_n" CHAIN=1 \
            W4_ENCODING="$MATCH_W4_ENCODING" BUILD_PACKAGES=0 DEVICE="${DEVICE:-oneplus}" \
            bash "$ROOT_DIR/scripts/run_w4a16_python_case_custom_native_match.sh"
        echo "PASS: $KERNEL Python-case custom/native exactness"
        exit 0
    fi

    if [ -n "$RUN_NATIVE_W4A16_CHAIN" ]; then
        echo "=== run $KERNEL precision case suite on ${DEVICE:-oneplus} ==="
        for case_name in ${W4A16_CHAIN1_CASES:-default zp k_impulse0 k_impulse1 k_impulse7}; do
            activation_mode="default"
            activation_k="0"
            suffix="$case_name"
            gen_extra_args="--activation-mode default"
            case "$case_name" in
                default)
                    ;;
                zp)
                    activation_mode="zp"
                    gen_extra_args="--activation-mode zp"
                    ;;
                k_impulse*)
                    activation_mode="k_impulse"
                    activation_k="${case_name#k_impulse}"
                    suffix="k_impulse_${activation_k}"
                    gen_extra_args="--activation-mode k_impulse --activation-k ${activation_k}"
                    ;;
                k*)
                    activation_mode="k_impulse"
                    activation_k="${case_name#k}"
                    suffix="k_impulse_${activation_k}"
                    gen_extra_args="--activation-mode k_impulse --activation-k ${activation_k}"
                    ;;
                *)
                    echo "ERROR: unknown W4A16 CHAIN=1 case '$case_name'" >&2
                    exit 2
                    ;;
            esac
            case_native="$OUT_ROOT/output_w4a16_native_chain1_${suffix}_ci_e2e_256"
            case_custom="$OUT_ROOT/output_w4a16_chain1_${suffix}_ci_e2e_256"
            echo "=== run $KERNEL case=$case_name native oracle ==="
            CHAIN="$RUN_NATIVE_W4A16_CHAIN" OUT_DIR="$case_native" DEVICE="${DEVICE:-oneplus}" \
                ACTIVATION_MODE="$activation_mode" ACTIVATION_K="$activation_k" \
                bash "$ROOT_DIR/example/qnn_matmul_profile/run_native_w4a16_conv_ref.sh"
            echo "=== run $KERNEL case=$case_name custom E2E ==="
            mkdir -p "$case_custom"
            env "${RUN_ENV[@]}" \
                GEN_EXTRA_ARGS="$gen_extra_args" \
                VERIFY_NATIVE_RAW="$case_native/device_out/Y.raw" \
                OUT_DIR="$case_custom" DEVICE="${DEVICE:-oneplus}" \
                bash "$RUN_SCRIPT"
            echo "=== validate $KERNEL case=$case_name E2E artifact ==="
            uv run python "$ROOT_DIR/scripts/validate_qnn_kernel_e2e.py" \
                --kernel "$KERNEL:$case_name" \
                --custom-dir "$case_custom" \
                --native-dir "$case_native" \
                --expected-htp-type "$HTP_TYPE" \
                --qnn-prefix "$QNN_PREFIX" \
                --expected-qnn-ops "$EXPECTED_QNN_OPS" \
                --dtype "$DTYPE" \
                "${TRANSPOSE_FLAG[@]}" \
                "${REF_FLAG[@]}"
        done
        echo "PASS: $KERNEL precision case suite"
        exit 0
    fi

    echo "=== run $KERNEL canonical E2E on ${DEVICE:-oneplus} ==="
    mkdir -p "$CUSTOM_DIR"
    env "${RUN_ENV[@]}" OUT_DIR="$CUSTOM_DIR" DEVICE="${DEVICE:-oneplus}" bash "$RUN_SCRIPT"
fi

echo "=== validate $KERNEL E2E artifact ==="
uv run python "$ROOT_DIR/scripts/validate_qnn_kernel_e2e.py" \
    --kernel "$KERNEL" \
    --custom-dir "$CUSTOM_DIR" \
    --native-dir "$NATIVE_DIR" \
    --expected-htp-type "$HTP_TYPE" \
    --qnn-prefix "$QNN_PREFIX" \
    --expected-qnn-ops "$EXPECTED_QNN_OPS" \
    --dtype "$DTYPE" \
    "${TRANSPOSE_FLAG[@]}" \
    "${LPBQ_FLAG[@]}" \
    "${REF_FLAG[@]}"
