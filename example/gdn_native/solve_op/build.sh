#!/usr/bin/env bash
#
# Build the GdnSolvePackage QNN custom op (GdnSolve: T=(I-A)^-1, KDA int16 forward-substitution).
# Produces:
#   build/hexagon-v75/libGdnSolvePackage_htp.so   (device HTP kernel)
#   build/aarch64/libGdnSolvePackage_cpu.so       (device CPU registration)
#   build/x86_64-linux-clang/libGdnSolvePackage.so (ctxgen-side registration)
#   converter/build/libConverterOpPackage.so      (qairt-converter shape/dtype inference)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env.sh" >/dev/null

for var in HEXAGON_SDK_ROOT HEXAGON_TOOLS_ROOT QNN_SDK_ROOT ANDROID_NDK_ROOT; do
    v="${!var:-}"
    if [ -z "$v" ] || [ ! -d "$v" ]; then
        echo "ERROR: $var empty/missing: '$v'. Source scripts/env.sh." >&2; exit 1
    fi
done

HEX_CXX="$HEXAGON_TOOLS_ROOT/bin/hexagon-clang++"
NDK_BIN="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin"
ARM_CXX="$NDK_BIN/clang++"
PACKAGE_NAME="GdnSolvePackage"
ARCH="${ARCH:-v75}"
SRCS=("$SCRIPT_DIR/src/GdnSolveInterface.cpp" "$SCRIPT_DIR/src/GdnSolveOp.cpp")

OUT_HTP="$SCRIPT_DIR/build/hexagon-$ARCH"; OUT_ARM="$SCRIPT_DIR/build/aarch64"
OUT_X86="$SCRIPT_DIR/build/x86_64-linux-clang"; OUT_CVT="$SCRIPT_DIR/converter/build"
mkdir -p "$OUT_HTP" "$OUT_ARM" "$OUT_X86" "$OUT_CVT"

COMMON=(
    -std=c++17 -O2 -fPIC -shared -fvisibility=default
    "-DTHIS_PKG_NAME=$PACKAGE_NAME"
    -I "$QNN_SDK_ROOT/include/QNN"
    -Wall -Wno-missing-braces -Wno-unused-function -Wno-format
    -Wno-unused-command-line-argument -Wno-unused-variable -Wno-unused-parameter
    '-DQNN_API=__attribute__((visibility("default")))'
    '-D__QAIC_HEADER_EXPORT=__attribute__((visibility("default")))'
)

echo "--- hexagon-$ARCH (HTP kernel) ---"
"$HEX_CXX" "${COMMON[@]}" -stdlib=libc++ \
    -mhvx -mhvx-length=128B "-m$ARCH" -DUSE_OS_QURT -DPREPARE_DISABLED \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$ARCH/include/qurt" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$ARCH/include/posix" \
    -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
    -o "$OUT_HTP/libGdnSolvePackage_htp.so" "${SRCS[@]}"
echo "  -> $OUT_HTP/libGdnSolvePackage_htp.so"

echo "--- aarch64-android (CPU registration) ---"
"$ARM_CXX" "${COMMON[@]}" --target=aarch64-none-linux-android21 --sysroot="$NDK_BIN/../sysroot" \
    -stdlib=libc++ -static-libstdc++ -D__HVXDBL__ -DUSE_OS_LINUX -DANDROID -DPREPARE_DISABLED \
    -fomit-frame-pointer -Wno-invalid-offsetof \
    -o "$OUT_ARM/libGdnSolvePackage_cpu.so" "${SRCS[@]}" \
    -L "$QNN_SDK_ROOT/lib/aarch64-android" -lQnnHtp -lQnnHtpPrepare
echo "  -> $OUT_ARM/libGdnSolvePackage_cpu.so"

echo "--- x86_64 (ctxgen registration) ---"
"${X86_CXX:-clang++}" "${COMMON[@]}" -D__HVXDBL__ -DUSE_OS_LINUX -DPREPARE_DISABLED -Wno-invalid-offsetof \
    -o "$OUT_X86/libGdnSolvePackage.so" "${SRCS[@]}"
echo "  -> $OUT_X86/libGdnSolvePackage.so"

echo "--- converter op package (shape/dtype inference) ---"
"${X86_CXX:-clang++}" -std=c++17 -O2 -shared -fPIC -I "$QNN_SDK_ROOT/include/QNN" \
    -o "$OUT_CVT/libConverterOpPackage.so" "$SCRIPT_DIR/converter/ConverterOpPackage.cpp"
echo "  -> $OUT_CVT/libConverterOpPackage.so"

echo "=== GdnSolvePackage build complete ==="
