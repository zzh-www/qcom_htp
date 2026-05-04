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

PACKAGE_NAME="QnnHmxMatMulU8I8Package"
OUT="$SCRIPT_DIR/build/x86_64-linux-clang"
mkdir -p "$OUT"

SRCS=(
    "$SCRIPT_DIR/src/QnnHmxMatMulU8I8Interface.cpp"
    "$SCRIPT_DIR/src/HmxU8I8ToU8MatMulOp.cpp"
)

FLAGS=(
    -O2 -fPIC
    -D__HVXDBL__ -DUSE_OS_LINUX -DPREPARE_DISABLED
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

echo "=== Building x86_64 QnnHmxMatMulU8I8 OpPackage ==="
"$CXX" -std=c++17 -shared "${FLAGS[@]}" \
    -o "$OUT/libQnnHmxMatMulU8I8.so" \
    "${SRCS[@]}"
echo "  -> $OUT/libQnnHmxMatMulU8I8.so"
