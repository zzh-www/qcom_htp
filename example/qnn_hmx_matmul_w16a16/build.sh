#!/usr/bin/env bash
#
# Build the QnnHmxMatMulW16A16 op package.
#
# Produces:
#   build/hexagon-v75/libQnnHmxMatMulW16A16_htp.so
#   build/aarch64/libQnnHmxMatMulW16A16_cpu.so

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env.sh" >/dev/null

for var in HEXAGON_SDK_ROOT HEXAGON_TOOLS_ROOT QNN_SDK_ROOT ANDROID_NDK_ROOT; do
    v="${!var:-}"
    if [ -z "$v" ] || [ ! -d "$v" ]; then
        echo "ERROR: $var is empty or missing: '$v'. Source scripts/env.sh." >&2
        exit 1
    fi
done

HEX_CXX="$HEXAGON_TOOLS_ROOT/bin/hexagon-clang++"
NDK_BIN="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin"
ARM_CXX="$NDK_BIN/clang++"

PACKAGE_NAME="QnnHmxMatMulW16A16Package"
ARCH="${ARCH:-v75}"

OUT_HTP="$SCRIPT_DIR/build/hexagon-$ARCH"
OUT_ARM="$SCRIPT_DIR/build/aarch64"
mkdir -p "$OUT_HTP" "$OUT_ARM"

SRCS=(
    "$SCRIPT_DIR/src/QnnHmxMatMulW16A16Interface.cpp"
    "$SCRIPT_DIR/src/HmxU16I16ToU16MatMulOp.cpp"
)

echo "=== Building QnnHmxMatMulW16A16Package (arch=$ARCH) ==="

HEX_FLAGS=(
    -O2 -fPIC
    -mhvx -mhvx-length=128B -mhmx "-m$ARCH"
    -DUSE_OS_QURT -DPREPARE_DISABLED
    -DHMX_W16A16_ENABLE_QHPI_PRECOMPUTE
    -DHMX_W16A16_SKIP_KERNEL
    "-DTHIS_PKG_NAME=$PACKAGE_NAME"
    -I "$QNN_SDK_ROOT/include/QNN"
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$ARCH/include/qurt"
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$ARCH/include/posix"
    -I "$HEXAGON_SDK_ROOT/incs"
    -I "$HEXAGON_SDK_ROOT/incs/stddef"
    -Wall -Wno-missing-braces -Wno-unused-function -Wno-format
    -Wno-unused-command-line-argument -Wno-unused-variable
    -Wno-unused-parameter -fvisibility=default
    '-DQNN_API=__attribute__((visibility("default")))'
    '-D__QAIC_HEADER_EXPORT=__attribute__((visibility("default")))'
    ${EXTRA_DEFS:-}
)

echo "--- hexagon-$ARCH ---"
"$HEX_CXX" -std=c++17 -shared -stdlib=libc++ "${HEX_FLAGS[@]}" \
    -o "$OUT_HTP/libQnnHmxMatMulW16A16_htp.so" \
    "${SRCS[@]}"
echo "  -> $OUT_HTP/libQnnHmxMatMulW16A16_htp.so"

ARM_FLAGS=(
    --target=aarch64-none-linux-android21
    --sysroot="$NDK_BIN/../sysroot"
    -stdlib=libc++
    -D__HVXDBL__ -DUSE_OS_LINUX -DANDROID -DPREPARE_DISABLED
    "-DTHIS_PKG_NAME=$PACKAGE_NAME"
    -I "$QNN_SDK_ROOT/include/QNN"
    -fomit-frame-pointer -fvisibility=default
    -Wno-missing-braces -Wno-unused-function -Wno-format
    -Wno-invalid-offsetof -Wno-unused-variable -Wno-unused-parameter
    -O2 -fPIC
    '-DQNN_API=__attribute__((visibility("default")))'
    '-D__QAIC_HEADER_EXPORT=__attribute__((visibility("default")))'
    ${EXTRA_DEFS:-}
)

echo "--- aarch64-android ---"
"$ARM_CXX" -std=c++17 -static-libstdc++ -shared "${ARM_FLAGS[@]}" \
    -o "$OUT_ARM/libQnnHmxMatMulW16A16_cpu.so" \
    "${SRCS[@]}" \
    -L "$QNN_SDK_ROOT/lib/aarch64-android" -lQnnHtp -lQnnHtpPrepare
echo "  -> $OUT_ARM/libQnnHmxMatMulW16A16_cpu.so"

echo "=== Build complete ==="
echo "Next: bash build_x86.sh"
