#!/usr/bin/env bash
#
# Build the GdnSolveBRPackage QNN custom op (GdnSolveBR: T=(I-A)^-1, C=128 block-recursive,
# HVX diagonals + HMX u8i8 merge driven from inside the op).
# Produces:
#   build/hexagon-v75/libGdnSolveBRPackage_htp.so   (device HTP kernel; HVX+HMX)
#   build/aarch64/libGdnSolveBRPackage_cpu.so       (device CPU registration / x86-style fallback)
#   build/x86_64-linux-clang/libGdnSolveBRPackage.so (ctxgen-side registration)
#   converter/build/libConverterOpPackage.so        (qairt-converter shape/dtype inference)
#
# Pass debug modes via EXTRA_DEFS, e.g.:
#   EXTRA_DEFS="-DGDN_BR_SKIP_KERNEL" bash build.sh
#   EXTRA_DEFS="-DGDN_BR_DIAG_ONLY"   bash build.sh
#   EXTRA_DEFS="-DGDN_BR_DUMP_M"      bash build.sh
#   EXTRA_DEFS="-DGDN_BR_PROBE_CYCLES" bash build.sh
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
PACKAGE_NAME="GdnSolveBRPackage"
ARCH="${ARCH:-v75}"
EXTRA_DEFS="${EXTRA_DEFS:-}"
SRCS=("$SCRIPT_DIR/src/GdnSolveBRInterface.cpp" "$SCRIPT_DIR/src/GdnSolveBROp.cpp")

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

echo "--- hexagon-$ARCH (HTP kernel, HVX+HMX) ---"
"$HEX_CXX" "${COMMON[@]}" -stdlib=libc++ \
    -mhvx -mhvx-length=128B -mhmx "-m$ARCH" -DUSE_OS_QURT -DPREPARE_DISABLED $EXTRA_DEFS \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$ARCH/include/qurt" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$ARCH/include/posix" \
    -I "$HEXAGON_SDK_ROOT/incs" -I "$HEXAGON_SDK_ROOT/incs/stddef" \
    -o "$OUT_HTP/libGdnSolveBRPackage_htp.so" "${SRCS[@]}"
echo "  -> $OUT_HTP/libGdnSolveBRPackage_htp.so"

echo "--- aarch64-android (CPU registration) ---"
"$ARM_CXX" "${COMMON[@]}" --target=aarch64-none-linux-android21 --sysroot="$NDK_BIN/../sysroot" \
    -stdlib=libc++ -static-libstdc++ -D__HVXDBL__ -DUSE_OS_LINUX -DANDROID -DPREPARE_DISABLED $EXTRA_DEFS \
    -fomit-frame-pointer -Wno-invalid-offsetof \
    -o "$OUT_ARM/libGdnSolveBRPackage_cpu.so" "${SRCS[@]}" \
    -L "$QNN_SDK_ROOT/lib/aarch64-android" -lQnnHtp -lQnnHtpPrepare
echo "  -> $OUT_ARM/libGdnSolveBRPackage_cpu.so"

echo "--- x86_64 (ctxgen registration) ---"
"${X86_CXX:-clang++}" "${COMMON[@]}" -D__HVXDBL__ -DUSE_OS_LINUX -DPREPARE_DISABLED $EXTRA_DEFS -Wno-invalid-offsetof \
    -o "$OUT_X86/libGdnSolveBRPackage.so" "${SRCS[@]}"
echo "  -> $OUT_X86/libGdnSolveBRPackage.so"

echo "--- converter op package (shape/dtype inference) ---"
"${X86_CXX:-clang++}" -std=c++17 -O2 -shared -fPIC -I "$QNN_SDK_ROOT/include/QNN" \
    -o "$OUT_CVT/libConverterOpPackage.so" "$SCRIPT_DIR/converter/ConverterOpPackage.cpp"
echo "  -> $OUT_CVT/libConverterOpPackage.so"

echo "=== GdnSolveBRPackage build complete ==="
