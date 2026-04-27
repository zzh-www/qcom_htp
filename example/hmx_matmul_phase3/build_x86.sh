#!/usr/bin/env bash
# build_x86.sh — x86_64 variant of libQnnHmxMatMulPhase3.so (V8 path).
#
# Needed by qnn-context-binary-generator's HTP prepare step on host.
# Kernel .c files fall back to scalar when !defined(__hexagon__) — ctxgen
# never runs them, it only reads metadata via the OpPackage interface.
#
# Output: build/x86_64-linux-clang/libQnnHmxMatMulPhase3.so

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env.sh" >/dev/null

for var in HEXAGON_SDK_ROOT QNN_SDK_ROOT; do
    v="${!var:-}"
    if [ -z "$v" ] || [ ! -d "$v" ]; then
        echo "ERROR: $var is empty or missing: '$v'"; exit 1
    fi
done

CXX="${X86_CXX:-clang++}"
command -v "$CXX" >/dev/null || { echo "ERROR: $CXX not found"; exit 1; }

# libnative provides x86 stubs for Hexagon intrinsics. Kernel .c files need
# hexagon_types.h / hvx_hexagon_protos.h to compile even though no hardware
# will run them on host.
HEX_TOOLS_DIR="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools"
HEX_TOOLS_VERSION="$(ls "$HEX_TOOLS_DIR" | sort -V | tail -1)"
X86_LIBNATIVE="$HEX_TOOLS_DIR/$HEX_TOOLS_VERSION/Tools/libnative"
[ -d "$X86_LIBNATIVE" ] || { echo "ERROR: libnative not found at $X86_LIBNATIVE"; exit 1; }

OUT="$SCRIPT_DIR/build/x86_64-linux-clang"
mkdir -p "$OUT"

echo "=== Building x86_64 variant of HmxMatMulPhase3 OpPackage ==="
echo "  CXX:         $CXX"
echo "  libnative:   $X86_LIBNATIVE"

FLAGS=(
    -O2 -fPIC
    -D__HVXDBL__ -DUSE_OS_LINUX
    -DTHIS_PKG_NAME=HmxMatMulPhase3Package
    -I "$QNN_SDK_ROOT/include/QNN"
    -I "$X86_LIBNATIVE/include"
    -I "$HEXAGON_SDK_ROOT/incs"
    -I "$HEXAGON_SDK_ROOT/incs/stddef"
    -fvisibility=default
    -Wno-missing-braces -Wno-unused-function -Wno-format
    -Wno-unused-command-line-argument -Wno-invalid-offsetof
    -Wno-unused-variable -Wno-unused-parameter -Wno-unused-but-set-variable
    '-DQNN_API=__attribute__((visibility("default")))'
    ${EXTRA_DEFS:-}
)

KERNEL_OBJS=()
for src in pack_act_rm_hvx pack_wt_v3_hvx tcm_dram_copy_hvx untile_to_rowmajor_hvx crouton_pack_spike_hvx pack_act_crouton_skel; do
    obj="$OUT/${src}.o"
    "$CXX" -std=c++17 "${FLAGS[@]}" -x c++ \
        -c "$SCRIPT_DIR/kernel/${src}.c" -o "$obj"
    KERNEL_OBJS+=("$obj")
done

# NB: do NOT link against libQnnHtp / libHtpPrepare — they pull in
# libSnpeHtp.so which has no x86 build.  libnative + libc are enough for
# ctxgen's metadata-only load.
"$CXX" -std=c++17 -shared "${FLAGS[@]}" \
    -o "$OUT/libQnnHmxMatMulPhase3.so" \
    "$SCRIPT_DIR/src/HmxMatMulPhase3Interface.cpp" \
    "$SCRIPT_DIR/src/HmxMatMulV8Op.cpp" \
    "$SCRIPT_DIR/src/HmxMatMulV9SkelOp.cpp" \
    "${KERNEL_OBJS[@]}" \
    -Wl,--whole-archive -L "$X86_LIBNATIVE/lib" -lnative -Wl,--no-whole-archive \
    -lpthread

echo "  -> $OUT/libQnnHmxMatMulPhase3.so"
