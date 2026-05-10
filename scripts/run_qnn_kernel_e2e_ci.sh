#!/usr/bin/env bash
#
# Run or validate the canonical 256^3 E2E gate for QNN HMX MatMul kernels.
#
# Default mode rebuilds the selected package, runs the device E2E flow, then
# validates the produced artifact against the retained native reference.  Set
# ARTIFACT_ONLY=1 to validate the retained standard artifacts without rerunning
# the device flow.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  scripts/run_qnn_kernel_e2e_ci.sh all|u8i8|w4a8|w4a8_lpbq|w8a16|w4a16|w4a16_lpbq|w16a16

Environment:
  DEVICE=oneplus                  SSH target for qnn-net-run device execution.
  ARTIFACT_ONLY=1                 Validate retained artifacts only.
  BUILD_PACKAGES=0                Skip package rebuild in device-run mode.
  KERNEL_E2E_OUT_ROOT=<dir>       Output root for regenerated custom artifacts.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ $# -ne 1 ]; then
    usage
    exit 2
fi

KERNEL="$1"
if [ "$KERNEL" = "all" ]; then
    for item in u8i8 w4a8 w4a8_lpbq w8a16 w4a16 w4a16_lpbq w16a16; do
        "$0" "$item"
    done
    exit 0
fi

PROFILE_DIR="$ROOT_DIR/example/qnn_matmul_profile"
ARTIFACT_ONLY="${ARTIFACT_ONLY:-0}"
BUILD_PACKAGES="${BUILD_PACKAGES:-1}"
OUT_ROOT="${KERNEL_E2E_OUT_ROOT:-$PROFILE_DIR}"

case "$KERNEL" in
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
        W16_FLAG=()
        LPBQ_FLAG=()
        ;;
    w4a8)
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
        W16_FLAG=()
        LPBQ_FLAG=()
        ;;
    w4a8_lpbq)
        EXAMPLE_DIR="$ROOT_DIR/example/qnn_hmx_matmul_w4a8"
        RUN_SCRIPT="$EXAMPLE_DIR/standard_flow/custom_w4a8/run_w4a8_chain.sh"
        RETAINED_CUSTOM="$PROFILE_DIR/output_w4a8_lpbq_e2e_256"
        NATIVE_DIR="$PROFILE_DIR/output_w4a8_native_ref_e2e_256"
        CI_CUSTOM="$OUT_ROOT/output_w4a8_lpbq_ci_e2e_256"
        HTP_TYPE="QnnHmxMatMulW4A8Package::HmxU8I4ToU8MatMul"
        QNN_PREFIX="hmx_w4a8_chain"
        EXPECTED_QNN_OPS=8
        DTYPE="uint8"
        BUILD_ENV=(LPBQ_ONLY=1)
        RUN_ENV=(M=256 K=256 N=256 CHAIN=8 MODE=chain W4_ENCODING=lpbq)
        TRANSPOSE_FLAG=()
        W16_FLAG=()
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
        W16_FLAG=()
        LPBQ_FLAG=()
        ;;
    w4a16)
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
        W16_FLAG=()
        LPBQ_FLAG=()
        ;;
    w4a16_lpbq)
        EXAMPLE_DIR="$ROOT_DIR/example/qnn_hmx_matmul_w4a16"
        RUN_SCRIPT="$EXAMPLE_DIR/standard_flow/custom_w4a16/run_w4a16_chain.sh"
        RETAINED_CUSTOM="$PROFILE_DIR/output_w4a16_lpbq_e2e_256"
        NATIVE_DIR="$PROFILE_DIR/output_w4a16_native_ref_e2e_256"
        CI_CUSTOM="$OUT_ROOT/output_w4a16_lpbq_ci_e2e_256"
        HTP_TYPE="QnnHmxMatMulW4A16Package::HmxU16I4ToU16MatMul"
        QNN_PREFIX="hmx_w4a16_chain"
        EXPECTED_QNN_OPS=8
        DTYPE="uint16"
        BUILD_ENV=(LPBQ_ONLY=1)
        RUN_ENV=(M=256 K=256 N=256 CHAIN=8 MODE=chain_qdq OP_INPUT_LAYOUT=tiled W4_ENCODING=lpbq VERIFY_NATIVE_RAW="$NATIVE_DIR/device_out/Y.raw" VERIFY_NATIVE_TRANSPOSE=1)
        TRANSPOSE_FLAG=(--native-transpose-2d)
        W16_FLAG=()
        LPBQ_FLAG=(--expect-lpbq)
        ;;
    w16a16)
        EXAMPLE_DIR="$ROOT_DIR/example/qnn_hmx_matmul_w16a16"
        RUN_SCRIPT="$EXAMPLE_DIR/standard_flow/custom_w16a16/run_w16a16_chain.sh"
        RETAINED_CUSTOM="$PROFILE_DIR/output_w16a16_accepted_256"
        NATIVE_DIR="$PROFILE_DIR/output_w16a16_native_ref_e2e_256"
        CI_CUSTOM="$OUT_ROOT/output_w16a16_ci_accepted_256"
        HTP_TYPE="QnnHmxMatMulW16A16Package::HmxU16I16ToU16MatMul"
        QNN_PREFIX="hmx_w16a16_chain"
        EXPECTED_QNN_OPS=1
        DTYPE="uint16"
        BUILD_ENV=(W16A16_KERNEL_PROFILE=accepted)
        RUN_ENV=(M=256 K=256 N=256 CHAIN=1 MODE=chain_qdq W16A16_KERNEL_PROFILE=accepted W16A16_NATIVE_ORACLE_DIR="$NATIVE_DIR" VERIFY_NATIVE_RAW="$NATIVE_DIR/device_out/Y.raw")
        TRANSPOSE_FLAG=()
        W16_FLAG=(--w16-accepted)
        LPBQ_FLAG=()
        ;;
    *)
        echo "ERROR: unknown kernel '$KERNEL'" >&2
        usage >&2
        exit 2
        ;;
esac

CUSTOM_DIR="$RETAINED_CUSTOM"
if [ "$ARTIFACT_ONLY" != "1" ]; then
    CUSTOM_DIR="$CI_CUSTOM"
    if [ "$BUILD_PACKAGES" = "1" ]; then
        echo "=== build $KERNEL package ==="
        env "${BUILD_ENV[@]}" bash "$EXAMPLE_DIR/build.sh"
        env "${BUILD_ENV[@]}" bash "$EXAMPLE_DIR/build_x86.sh"
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
    "${W16_FLAG[@]}" \
    "${LPBQ_FLAG[@]}"
