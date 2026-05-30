#!/usr/bin/env bash
#
# Host-side x86_64 build used by qnn-context-binary-generator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env.sh" >/dev/null

for var in QNN_SDK_ROOT HEXAGON_TOOLS_ROOT; do
    v="${!var:-}"
    if [ -z "$v" ] || [ ! -d "$v" ]; then
        echo "ERROR: $var is empty or missing: '$v'. Source scripts/env.sh." >&2
        exit 1
    fi
done

CXX="${X86_CXX:-clang++}"
command -v "$CXX" >/dev/null || { echo "ERROR: $CXX not found" >&2; exit 1; }

PACKAGE_NAME="QnnHmxMatMulW4A8Package"
OUT="$SCRIPT_DIR/build/x86_64-linux-clang"
mkdir -p "$OUT"

SRCS=(
    "$SCRIPT_DIR/src/QnnHmxMatMulW4A8Interface.cpp"
    "$SCRIPT_DIR/src/HmxU8I4ToU8MatMulOp.cpp"
    "$ROOT_DIR/example/qnn_hmx_matmul_common/HmxW4LpbqExpandToI8Op.cpp"
    "$ROOT_DIR/example/qnn_hmx_matmul_u8i8/src/HmxU8I8ToU8MatMulOp.cpp"
)

FLAGS=(
    -O2 -fPIC -nostdlib++
    -nostdinc++ -I "$HEXAGON_TOOLS_ROOT/target/hexagon/include/c++/v1"
    -D__HVXDBL__ -DUSE_OS_LINUX -DPREPARE_DISABLED
    -DHMX_W4A8_ENABLE_QHPI_PRECOMPUTE
    -DHMX_U8I8_ENABLE_QHPI_PRECOMPUTE
    "-DTHIS_PKG_NAME=$PACKAGE_NAME"
    -I "$QNN_SDK_ROOT/include/QNN"
    -fvisibility=default
    -Wno-missing-braces -Wno-unused-function -Wno-format
    -Wno-unused-command-line-argument -Wno-invalid-offsetof
    -Wno-unused-variable -Wno-unused-parameter
    '-DQNN_API=__attribute__((visibility("default")))'
    ${EXTRA_DEFS:-}
)
if [ "${LPBQ_ONLY:-0}" = "1" ]; then
    FLAGS+=(-DHMX_W4A8_LPBQ_ONLY)
fi
if [ "${LPBQ_SCALAR:-0}" = "1" ]; then
    FLAGS+=(-UHMX_W4A8_ENABLE_QHPI_PRECOMPUTE -DHMX_W4A8_LPBQ_SCALAR_ONLY)
fi

echo "=== Building x86_64 QnnHmxMatMulW4A8 OpPackage ==="
"$CXX" -std=c++17 -shared "${FLAGS[@]}" \
    -o "$OUT/libQnnHmxMatMulW4A8.so" \
    "${SRCS[@]}" \
    /usr/lib/x86_64-linux-gnu/libc++.so.1 /usr/lib/x86_64-linux-gnu/libc++abi.so.1 \
    -L "$QNN_SDK_ROOT/lib/x86_64-linux-clang" -lQnnHtp
echo "  -> $OUT/libQnnHmxMatMulW4A8.so"
