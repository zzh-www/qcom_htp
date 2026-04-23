#!/usr/bin/env bash
#
# build.sh — build Phase 3 probe OpPackage.
#
# Produces:
#   build/hexagon-v75/libQnnHmxMatMulPhase3_htp.so
#   build/aarch64/libQnnHmxMatMulPhase3_cpu.so
#   build/aarch64/run_phase3_probe

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

PACKAGE_NAME="HmxMatMulPhase3Package"
ARCH="${ARCH:-v75}"

OUT_HTP="$SCRIPT_DIR/build/hexagon-$ARCH"
OUT_ARM="$SCRIPT_DIR/build/aarch64"
mkdir -p "$OUT_HTP" "$OUT_ARM"

echo "=== Building Phase 3 OpPackage (arch=$ARCH) ==="

# ----- DSP side -----
echo "--- DSP package (hexagon-$ARCH) ---"
"$HEX_CXX" -std=c++17 -O2 -fPIC -shared \
    -mhvx -mhvx-length=128B -mhmx "-m$ARCH" \
    -DUSE_OS_QURT -DPREPARE_DISABLED \
    "-DTHIS_PKG_NAME=$PACKAGE_NAME" \
    -I "$QNN_SDK_ROOT/include/QNN" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$ARCH/include/qurt" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$ARCH/include/posix" \
    -I "$HEXAGON_SDK_ROOT/incs" \
    -I "$HEXAGON_SDK_ROOT/incs/stddef" \
    -Wall -Wno-missing-braces -Wno-unused-function -Wno-format \
    -Wno-unused-command-line-argument -fvisibility=default -stdlib=libc++ \
    '-DQNN_API=__attribute__((visibility("default")))' \
    '-D__QAIC_HEADER_EXPORT=__attribute__((visibility("default")))' \
    -o "$OUT_HTP/libQnnHmxMatMulPhase3_htp.so" \
    "$SCRIPT_DIR/src/HmxMatMulPhase3Interface.cpp" \
    "$SCRIPT_DIR/src/HmxMatMulPhase3Op.cpp"

echo "  -> $OUT_HTP/libQnnHmxMatMulPhase3_htp.so"

# ----- ARM side (CPU fallback .so) -----
echo "--- ARM package (aarch64-android) ---"
"$ARM_CXX" -std=c++17 -O2 -fPIC -shared \
    --target=aarch64-none-linux-android21 \
    --sysroot="$NDK_BIN/../sysroot" \
    -stdlib=libc++ -static-libstdc++ \
    -D__HVXDBL__ -DUSE_OS_LINUX -DANDROID -DPREPARE_DISABLED -DSCALAR_ONLY \
    "-DTHIS_PKG_NAME=$PACKAGE_NAME" \
    -I "$QNN_SDK_ROOT/include/QNN" \
    -fomit-frame-pointer -fvisibility=default \
    -Wno-missing-braces -Wno-unused-function -Wno-format \
    -Wno-invalid-offsetof -Wno-unused-variable -Wno-unused-parameter \
    '-DQNN_API=__attribute__((visibility("default")))' \
    '-D__QAIC_HEADER_EXPORT=__attribute__((visibility("default")))' \
    -o "$OUT_ARM/libQnnHmxMatMulPhase3_cpu.so" \
    "$SCRIPT_DIR/src/HmxMatMulPhase3Interface.cpp" \
    "$SCRIPT_DIR/src/HmxMatMulPhase3Op.cpp" \
    -L "$QNN_SDK_ROOT/lib/aarch64-android" -lQnnHtp -lQnnHtpPrepare

echo "  -> $OUT_ARM/libQnnHmxMatMulPhase3_cpu.so"

# ----- Host probe binary -----
echo "--- Host probe (aarch64-android) ---"
"$ARM_CXX" -std=c++17 -O2 \
    --target=aarch64-none-linux-android21 \
    --sysroot="$NDK_BIN/../sysroot" \
    -stdlib=libc++ -static-libstdc++ \
    -I "$QNN_SDK_ROOT/include/QNN" \
    -o "$OUT_ARM/run_phase3_probe" \
    "$SCRIPT_DIR/src/run_phase3_probe.cpp" \
    -ldl

echo "  -> $OUT_ARM/run_phase3_probe"

echo ""
echo "=== Build complete ==="
