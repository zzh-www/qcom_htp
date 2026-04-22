#!/usr/bin/env bash
#
# build.sh — build the HmxW16A16MatMul QNN OpPackage.
#
# Produces:
#   build/hexagon-v75/libQnnHmxW16A16MatMul_htp.so   — DSP-side kernel + QHPI
#   build/aarch64/libQnnHmxW16A16MatMul_cpu.so       — ARM-side registration
#                                                    (CPU fallback; used by
#                                                     qairt-converter to
#                                                     validate the graph)
#
# Depends on env.sh for HEXAGON_SDK_ROOT, HEXAGON_TOOLS_ROOT, QNN_SDK_ROOT,
# and ANDROID_NDK_ROOT. Source scripts/env.sh before running.

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
if [ ! -x "$HEX_CXX" ]; then
    echo "ERROR: $HEX_CXX not found" >&2
    exit 1
fi

NDK_BIN="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin"
ARM_CXX="$NDK_BIN/clang++"
if [ ! -x "$ARM_CXX" ]; then
    echo "ERROR: $ARM_CXX not found (check ANDROID_NDK_ROOT)" >&2
    exit 1
fi

PACKAGE_NAME="HmxW16A16MatMulPackage"
ARCH="${ARCH:-v75}"
# Set SCALAR_ONLY=1 to disable HMX and build the scalar-only reference kernel
# (useful to isolate QNN integration issues from HMX bugs).
# SCALAR_ONLY=1 forces the scalar reference kernel (no HMX). Default is
# HMX-on. Useful for correctness bisection if an HMX change regresses.
EXTRA_DEFS=""
if [ "${SCALAR_ONLY:-0}" = "1" ]; then
    EXTRA_DEFS="-DSCALAR_ONLY"
    echo "  (SCALAR_ONLY=1 — HMX path disabled)"
fi

OUT_HTP="$SCRIPT_DIR/build/hexagon-$ARCH"
OUT_ARM="$SCRIPT_DIR/build/aarch64"
mkdir -p "$OUT_HTP" "$OUT_ARM"

echo "=== Building HmxW16A16MatMul OpPackage (arch=$ARCH) ==="

# ----- DSP side (hexagon-$ARCH) ----------------------------------------------
echo "--- DSP package (hexagon-$ARCH) ---"
"$HEX_CXX" -std=c++17 -O2 -fPIC -shared \
    -mhvx -mhvx-length=128B -mhmx "-m$ARCH" \
    -DUSE_OS_QURT -DPREPARE_DISABLED \
    "-DTHIS_PKG_NAME=$PACKAGE_NAME" $EXTRA_DEFS \
    -I "$QNN_SDK_ROOT/include/QNN" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$ARCH/include/qurt" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$ARCH/include/posix" \
    -I "$HEXAGON_SDK_ROOT/incs" \
    -I "$HEXAGON_SDK_ROOT/incs/stddef" \
    -Wall -Wno-missing-braces -Wno-unused-function -Wno-format \
    -Wno-unused-command-line-argument -fvisibility=default -stdlib=libc++ \
    '-DQNN_API=__attribute__((visibility("default")))' \
    '-D__QAIC_HEADER_EXPORT=__attribute__((visibility("default")))' \
    -o "$OUT_HTP/libQnnHmxW16A16MatMul_htp.so" \
    "$SCRIPT_DIR/src/HmxW16A16MatMulInterface.cpp" \
    "$SCRIPT_DIR/src/HmxW16A16MatMulOp.cpp" \
    "$SCRIPT_DIR/kernel/hmx_int16x16_matmul.c"

echo "  -> $OUT_HTP/libQnnHmxW16A16MatMul_htp.so"

# ----- ARM side (aarch64-android, for QNN prepare-time validation) -----------
echo "--- ARM package (aarch64-android) ---"
"$ARM_CXX" -std=c++17 -O2 -fPIC -shared \
    --target=aarch64-none-linux-android21 \
    --sysroot="$NDK_BIN/../sysroot" \
    -stdlib=libc++ -static-libstdc++ \
    -D__HVXDBL__ -DUSE_OS_LINUX -DANDROID -DPREPARE_DISABLED \
    "-DTHIS_PKG_NAME=$PACKAGE_NAME" $EXTRA_DEFS \
    -I "$QNN_SDK_ROOT/include/QNN" \
    -fomit-frame-pointer -fvisibility=default \
    -Wno-missing-braces -Wno-unused-function -Wno-format \
    -Wno-invalid-offsetof -Wno-unused-variable -Wno-unused-parameter \
    '-DQNN_API=__attribute__((visibility("default")))' \
    '-D__QAIC_HEADER_EXPORT=__attribute__((visibility("default")))' \
    -o "$OUT_ARM/libQnnHmxW16A16MatMul_cpu.so" \
    "$SCRIPT_DIR/src/HmxW16A16MatMulInterface.cpp" \
    "$SCRIPT_DIR/src/HmxW16A16MatMulOp.cpp" \
    -L "$QNN_SDK_ROOT/lib/aarch64-android" -lQnnHtp -lQnnHtpPrepare

echo "  -> $OUT_ARM/libQnnHmxW16A16MatMul_cpu.so"

# ----- ARM side host test harness --------------------------------------------
echo "--- Host test (aarch64-android) ---"
"$ARM_CXX" -std=c++17 -O2 \
    --target=aarch64-none-linux-android21 \
    --sysroot="$NDK_BIN/../sysroot" \
    -stdlib=libc++ -static-libstdc++ \
    -I "$QNN_SDK_ROOT/include/QNN" \
    -o "$OUT_ARM/run_w16a16_matmul" \
    "$SCRIPT_DIR/src/run_w16a16_matmul.cpp" \
    -ldl

echo "  -> $OUT_ARM/run_w16a16_matmul"

echo ""
echo "=== Build complete ==="
