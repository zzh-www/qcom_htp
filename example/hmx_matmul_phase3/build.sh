#!/usr/bin/env bash
#
# build.sh — build HmxMatMulPhase3 OpPackage (V8 path only).
#
# Produces:
#   build/hexagon-v75/libQnnHmxMatMulPhase3_htp.so   (HTP kernel runtime)
#   build/aarch64/libQnnHmxMatMulPhase3_cpu.so        (ARM prepare on device)
#   build/aarch64/run_matmul_v8_graph                 (host bit-exact harness)
#
# For x86_64 (needed by qnn-context-binary-generator host prepare):
#   bash build_x86.sh  → build/x86_64-linux-clang/libQnnHmxMatMulPhase3.so
#
# See docs/qnn_custom_op_sop.md for the full standard flow.

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

echo "=== Building Phase3 OpPackage V8 path (arch=$ARCH) ==="

KERNEL_SRCS=(pack_act_rm_hvx pack_wt_v3_hvx tcm_dram_copy_hvx untile_to_rowmajor_hvx)
V8_SRCS=(HmxMatMulPhase3Interface.cpp HmxMatMulV8Op.cpp)

# ---- Hexagon (runtime HTP) ----
echo "--- hexagon-$ARCH ---"
HEX_FLAGS=(
    -O2 -fPIC
    -mhvx -mhvx-length=128B -mhmx "-m$ARCH"
    -DUSE_OS_QURT -DPREPARE_DISABLED
    "-DTHIS_PKG_NAME=$PACKAGE_NAME"
    -I "$QNN_SDK_ROOT/include/QNN"
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$ARCH/include/qurt"
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$ARCH/include/posix"
    -I "$HEXAGON_SDK_ROOT/incs"
    -I "$HEXAGON_SDK_ROOT/incs/stddef"
    -Wall -Wno-missing-braces -Wno-unused-function -Wno-format
    -Wno-unused-command-line-argument -fvisibility=default
    '-DQNN_API=__attribute__((visibility("default")))'
    '-D__QAIC_HEADER_EXPORT=__attribute__((visibility("default")))'
)
KERNEL_OBJS=()
for src in "${KERNEL_SRCS[@]}"; do
    obj="$OUT_HTP/${src}.o"
    "$HEX_CXX" -std=c++17 -stdlib=libc++ "${HEX_FLAGS[@]}" \
        -c "$SCRIPT_DIR/kernel/${src}.c" -o "$obj"
    KERNEL_OBJS+=("$obj")
done
"$HEX_CXX" -std=c++17 -shared -stdlib=libc++ "${HEX_FLAGS[@]}" \
    -o "$OUT_HTP/libQnnHmxMatMulPhase3_htp.so" \
    "${V8_SRCS[@]/#/$SCRIPT_DIR/src/}" \
    "${KERNEL_OBJS[@]}"
echo "  -> $OUT_HTP/libQnnHmxMatMulPhase3_htp.so"

# ---- ARM (on-device CPU/prepare) ----
echo "--- aarch64-android ---"
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
)
ARM_OBJS=()
for src in "${KERNEL_SRCS[@]}"; do
    obj="$OUT_ARM/${src}.o"
    "$ARM_CXX" -std=c++17 "${ARM_FLAGS[@]}" -x c++ \
        -c "$SCRIPT_DIR/kernel/${src}.c" -o "$obj"
    ARM_OBJS+=("$obj")
done
"$ARM_CXX" -std=c++17 -static-libstdc++ -shared "${ARM_FLAGS[@]}" \
    -o "$OUT_ARM/libQnnHmxMatMulPhase3_cpu.so" \
    "${V8_SRCS[@]/#/$SCRIPT_DIR/src/}" \
    "${ARM_OBJS[@]}" \
    -L "$QNN_SDK_ROOT/lib/aarch64-android" -lQnnHtp -lQnnHtpPrepare
echo "  -> $OUT_ARM/libQnnHmxMatMulPhase3_cpu.so"

# ---- Host harness (V8 bit-exact ground truth) ----
echo "--- run_matmul_v8_graph (aarch64-android) ---"
"$ARM_CXX" -std=c++17 -O2 \
    --target=aarch64-none-linux-android21 \
    --sysroot="$NDK_BIN/../sysroot" \
    -stdlib=libc++ -static-libstdc++ \
    -I "$QNN_SDK_ROOT/include/QNN" \
    -o "$OUT_ARM/run_matmul_v8_graph" \
    "$SCRIPT_DIR/src/run_matmul_v8_graph.cpp" \
    -ldl
echo "  -> $OUT_ARM/run_matmul_v8_graph"

echo ""
echo "=== Build complete ==="
echo "Next: bash build_x86.sh  (for ctxgen host-side op package)"
