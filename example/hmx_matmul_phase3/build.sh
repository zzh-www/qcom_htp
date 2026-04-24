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

HEX_CFLAGS_COMMON=(
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

# Compile kernel files. Agent B's HVX ops use QHPI which has nullptr
# (C++ header), so compile as C++. hmx_core_v2 is plain C, compile with -x c.
KERNEL_OBJS=()
for src in pack_act_hvx pack_wt_hvx pack_act_u8_hvx pack_act_rm_hvx pack_wt_v3_hvx combine_hi_lo_hvx int4_expand_hvx; do
    obj="$OUT_HTP/${src}.o"
    EXTRA=()
    if [ -n "${PACK_SCALAR_FORCE:-}" ]; then
        if [ "$src" = "pack_act_u8_hvx" ]; then EXTRA+=(-DPACK_ACT_U8_SCALAR_FORCE); fi
        if [ "$src" = "pack_wt_v3_hvx" ];  then EXTRA+=(-DPACK_WT_V3_SCALAR_FORCE);  fi
    fi
    if [ -n "${PACK_ACT_SCALAR:-}" ] && [ "$src" = "pack_act_u8_hvx" ]; then
        EXTRA+=(-DPACK_ACT_U8_SCALAR_FORCE)
    fi
    if [ -n "${PACK_WT_SCALAR:-}" ] && [ "$src" = "pack_wt_v3_hvx" ]; then
        EXTRA+=(-DPACK_WT_V3_SCALAR_FORCE)
    fi
    "$HEX_CXX" -std=c++17 -stdlib=libc++ "${HEX_CFLAGS_COMMON[@]}" "${EXTRA[@]}" ${V6_FLAGS:-} \
        -c "$SCRIPT_DIR/kernel/${src}.c" -o "$obj"
    KERNEL_OBJS+=("$obj")
done
# hmx_core_v2: pure C, no QHPI includes.
"$HEX_CXX" -x c -std=c11 "${HEX_CFLAGS_COMMON[@]}" ${V6_FLAGS:-} \
    -c "$SCRIPT_DIR/kernel/hmx_core_v2.c" -o "$OUT_HTP/hmx_core_v2.o"
KERNEL_OBJS+=("$OUT_HTP/hmx_core_v2.o")

# Compile + link C++ interface + op.
"$HEX_CXX" -std=c++17 -shared -stdlib=libc++ \
    "${HEX_CFLAGS_COMMON[@]}" ${V6_FLAGS:-} \
    -o "$OUT_HTP/libQnnHmxMatMulPhase3_htp.so" \
    "$SCRIPT_DIR/src/HmxMatMulPhase3Interface.cpp" \
    "$SCRIPT_DIR/src/HmxMatMulPhase3Op.cpp" \
    "$SCRIPT_DIR/src/HmxMatMulV2Op.cpp" \
    "$SCRIPT_DIR/src/HmxMatMulV3Op.cpp" \
    "$SCRIPT_DIR/src/HmxMatMulV4Op.cpp" \
    "$SCRIPT_DIR/src/HmxMatMulV6Op.cpp" \
    "$SCRIPT_DIR/src/HmxMatMulV7Op.cpp" \
    "$SCRIPT_DIR/src/HmxMatMulV8Op.cpp" \
    "$SCRIPT_DIR/src/RequantHvxOp.cpp" \
    "${KERNEL_OBJS[@]}"

# --- Need to add V4 to ARM package too ---

echo "  -> $OUT_HTP/libQnnHmxMatMulPhase3_htp.so"

# ----- ARM side (CPU fallback .so) — separate C/C++ compilation -----
echo "--- ARM package (aarch64-android) ---"

ARM_CXX_FLAGS=(
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
# HVX op files: compile as C++ (they use QHPI / nullptr).
for src in pack_act_hvx pack_wt_hvx pack_act_u8_hvx pack_act_rm_hvx pack_wt_v3_hvx combine_hi_lo_hvx int4_expand_hvx; do
    obj="$OUT_ARM/${src}.o"
    "$ARM_CXX" -std=c++17 "${ARM_CXX_FLAGS[@]}" -x c++ \
        -c "$SCRIPT_DIR/kernel/${src}.c" -o "$obj"
    ARM_OBJS+=("$obj")
done
# hmx_core_v2: plain C. Use NDK's clang (C compiler, not clang++).
NDK_CC="$NDK_BIN/clang"
"$NDK_CC" -std=c11 --target=aarch64-none-linux-android21 \
    --sysroot="$NDK_BIN/../sysroot" \
    -O2 -fPIC -fvisibility=default \
    -c "$SCRIPT_DIR/kernel/hmx_core_v2.c" -o "$OUT_ARM/hmx_core_v2.o"
ARM_OBJS+=("$OUT_ARM/hmx_core_v2.o")

# C++ source files.
"$ARM_CXX" -std=c++17 -static-libstdc++ -shared "${ARM_CXX_FLAGS[@]}" \
    -o "$OUT_ARM/libQnnHmxMatMulPhase3_cpu.so" \
    "$SCRIPT_DIR/src/HmxMatMulPhase3Interface.cpp" \
    "$SCRIPT_DIR/src/HmxMatMulPhase3Op.cpp" \
    "$SCRIPT_DIR/src/HmxMatMulV2Op.cpp" \
    "$SCRIPT_DIR/src/HmxMatMulV3Op.cpp" \
    "$SCRIPT_DIR/src/HmxMatMulV4Op.cpp" \
    "$SCRIPT_DIR/src/HmxMatMulV6Op.cpp" \
    "$SCRIPT_DIR/src/HmxMatMulV7Op.cpp" \
    "$SCRIPT_DIR/src/HmxMatMulV8Op.cpp" \
    "$SCRIPT_DIR/src/RequantHvxOp.cpp" \
    "${ARM_OBJS[@]}" \
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

# ----- Host V2 matmul runner -----
echo "--- Host run_matmul_v2 (aarch64-android) ---"
"$ARM_CXX" -std=c++17 -O2 \
    --target=aarch64-none-linux-android21 \
    --sysroot="$NDK_BIN/../sysroot" \
    -stdlib=libc++ -static-libstdc++ \
    -I "$QNN_SDK_ROOT/include/QNN" \
    -o "$OUT_ARM/run_matmul_v2" \
    "$SCRIPT_DIR/src/run_matmul_v2.cpp" \
    -ldl

echo "  -> $OUT_ARM/run_matmul_v2"

# ----- Host V3 matmul runner (pure HMX op, host pre-packs) -----
echo "--- Host run_matmul_v3 (aarch64-android) ---"
"$ARM_CXX" -std=c++17 -O2 \
    --target=aarch64-none-linux-android21 \
    --sysroot="$NDK_BIN/../sysroot" \
    -stdlib=libc++ -static-libstdc++ \
    -I "$QNN_SDK_ROOT/include/QNN" \
    -o "$OUT_ARM/run_matmul_v3" \
    "$SCRIPT_DIR/src/run_matmul_v3.cpp" \
    -ldl

echo "  -> $OUT_ARM/run_matmul_v3"

# ----- Host run_matmul_v4 (:cm row-major act) -----
echo "--- Host run_matmul_v4 (aarch64-android) ---"
"$ARM_CXX" -std=c++17 -O2 \
    --target=aarch64-none-linux-android21 \
    --sysroot="$NDK_BIN/../sysroot" \
    -stdlib=libc++ -static-libstdc++ \
    -I "$QNN_SDK_ROOT/include/QNN" \
    -o "$OUT_ARM/run_matmul_v4" \
    "$SCRIPT_DIR/src/run_matmul_v4.cpp" \
    -ldl

echo "  -> $OUT_ARM/run_matmul_v4"

# ----- Host run_matmul_v6 (quantized u8×i8→i8) -----
echo "--- Host run_matmul_v6 (aarch64-android) ---"
"$ARM_CXX" -std=c++17 -O2 \
    --target=aarch64-none-linux-android21 \
    --sysroot="$NDK_BIN/../sysroot" \
    -stdlib=libc++ -static-libstdc++ \
    -I "$QNN_SDK_ROOT/include/QNN" \
    -o "$OUT_ARM/run_matmul_v6" \
    "$SCRIPT_DIR/src/run_matmul_v6.cpp" \
    -ldl

echo "  -> $OUT_ARM/run_matmul_v6"

# ----- Host run_matmul_v6_graph (V6 quantized path W) -----
echo "--- Host run_matmul_v6_graph (aarch64-android) ---"
"$ARM_CXX" -std=c++17 -O2 \
    --target=aarch64-none-linux-android21 \
    --sysroot="$NDK_BIN/../sysroot" \
    -stdlib=libc++ -static-libstdc++ \
    -I "$QNN_SDK_ROOT/include/QNN" \
    -o "$OUT_ARM/run_matmul_v6_graph" \
    "$SCRIPT_DIR/src/run_matmul_v6_graph.cpp" \
    -ldl
echo "  -> $OUT_ARM/run_matmul_v6_graph"

# ----- Host run_matmul_v7_graph (HMX-only V7 + separate HVX requant) -----
echo "--- Host run_matmul_v7_graph (aarch64-android) ---"
"$ARM_CXX" -std=c++17 -O2 \
    --target=aarch64-none-linux-android21 \
    --sysroot="$NDK_BIN/../sysroot" \
    -stdlib=libc++ -static-libstdc++ \
    -I "$QNN_SDK_ROOT/include/QNN" \
    -o "$OUT_ARM/run_matmul_v7_graph" \
    "$SCRIPT_DIR/src/run_matmul_v7_graph.cpp" \
    -ldl
echo "  -> $OUT_ARM/run_matmul_v7_graph"

# ----- Host run_matmul_v8_graph (Phase 3D.4 HMX-only replica of ConvLayer_s1.opt) -----
echo "--- Host run_matmul_v8_graph (aarch64-android) ---"
"$ARM_CXX" -std=c++17 -O2 \
    --target=aarch64-none-linux-android21 \
    --sysroot="$NDK_BIN/../sysroot" \
    -stdlib=libc++ -static-libstdc++ \
    -I "$QNN_SDK_ROOT/include/QNN" \
    -o "$OUT_ARM/run_matmul_v8_graph" \
    "$SCRIPT_DIR/src/run_matmul_v8_graph.cpp" \
    -ldl
echo "  -> $OUT_ARM/run_matmul_v8_graph"

# ----- Host run_matmul_graph (Path W: PackActU8 + PackWtV3 + MatMulV3) -----
echo "--- Host run_matmul_graph (aarch64-android) ---"
"$ARM_CXX" -std=c++17 -O2 \
    --target=aarch64-none-linux-android21 \
    --sysroot="$NDK_BIN/../sysroot" \
    -stdlib=libc++ -static-libstdc++ \
    -I "$QNN_SDK_ROOT/include/QNN" \
    -o "$OUT_ARM/run_matmul_graph" \
    "$SCRIPT_DIR/src/run_matmul_graph.cpp" \
    -ldl

echo "  -> $OUT_ARM/run_matmul_graph"

echo ""
echo "=== Build complete ==="
