#!/usr/bin/env bash
#
# Host-side x86_64 build used by qnn-context-binary-generator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env.sh" >/dev/null

for var in QNN_SDK_ROOT; do
    v="${!var:-}"
    if [ -z "$v" ] || [ ! -d "$v" ]; then
        echo "ERROR: $var is empty or missing: '$v'. Source scripts/env.sh." >&2
        exit 1
    fi
done

CXX="${X86_CXX:-clang++}"
command -v "$CXX" >/dev/null || { echo "ERROR: $CXX not found" >&2; exit 1; }

PACKAGE_NAME="QnnHmxMatMulW16A16Package"
W16A16_KERNEL_PROFILE="${W16A16_KERNEL_PROFILE:-skip}"
OUT="$SCRIPT_DIR/build/x86_64-linux-clang"
mkdir -p "$OUT"

SRCS=(
    "$SCRIPT_DIR/src/QnnHmxMatMulW16A16Interface.cpp"
    "$SCRIPT_DIR/src/HmxU16I16ToU16MatMulOp.cpp"
)

PROFILE_DEFS=()
case "$W16A16_KERNEL_PROFILE" in
    skip)
        PROFILE_DEFS+=(-DHMX_W16A16_SKIP_KERNEL)
        ;;
    accepted)
        PROFILE_DEFS+=(
            -UHMX_W16A16_SKIP_KERNEL
            -DHMX_W16A16_ACCEPTED_NATIVE_RECORD_256
            -DHMX_W16A16_NATIVE_RECORD_256_PROFILE
        )
        ;;
    native_record_256)
        PROFILE_DEFS+=(
            -UHMX_W16A16_SKIP_KERNEL
            -DHMX_W16A16_ALLOW_UNVALIDATED_KERNEL
            -DHMX_W16A16_NATIVE_RECORD_256_PROFILE
        )
        ;;
    *)
        echo "ERROR: unknown W16A16_KERNEL_PROFILE='$W16A16_KERNEL_PROFILE' (expected skip, accepted, or native_record_256)" >&2
        exit 1
        ;;
esac

FLAGS=(
    -O2 -fPIC
    -D__HVXDBL__ -DUSE_OS_LINUX -DPREPARE_DISABLED
    -DHMX_W16A16_ENABLE_QHPI_PRECOMPUTE
    "${PROFILE_DEFS[@]}"
    "-DTHIS_PKG_NAME=$PACKAGE_NAME"
    -I "$QNN_SDK_ROOT/include/QNN"
    -fvisibility=default
    -Wno-missing-braces -Wno-unused-function -Wno-format
    -Wno-unused-command-line-argument -Wno-invalid-offsetof
    -Wno-unused-variable -Wno-unused-parameter
    '-DQNN_API=__attribute__((visibility("default")))'
    ${EXTRA_DEFS:-}
)

echo "=== Building x86_64 QnnHmxMatMulW16A16 OpPackage ==="
echo "profile: $W16A16_KERNEL_PROFILE"
"$CXX" -std=c++17 -shared "${FLAGS[@]}" \
    -o "$OUT/libQnnHmxMatMulW16A16.so" \
    "${SRCS[@]}"
echo "  -> $OUT/libQnnHmxMatMulW16A16.so"
