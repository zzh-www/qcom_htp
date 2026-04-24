#!/usr/bin/env bash
#
# build.sh — 编译 HMX 编程指南 demo 成 hexagon-sim 可执行 ELF。
# 用的是同 example/hmx_matmul_int16/build.sh 完全一致的构建方式。
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env.sh"

H2_INSTALL="$ROOT_DIR/tools/h2-install"
if [ -L "$H2_INSTALL" ]; then
    H2_ROOT="$(dirname "$(readlink -f "$H2_INSTALL")")"
else
    H2_ROOT="$(cd "$H2_INSTALL/.." && pwd)"
fi
H2_KERNEL_INC="$H2_ROOT/kernel/include"

compile_demo() {
    local src="$1"
    local out="${src%.c}"
    hexagon-clang -O2 -mv75 \
        -mhvx -mhvx-length=128B \
        -mhmx \
        -DARCHV=75 \
        -I "$H2_INSTALL/include" \
        -I "$H2_KERNEL_INC" \
        -moslib=h2 \
        -Wl,-L,"$H2_INSTALL/lib" \
        -Wl,--section-start=.start=0x02000000 \
        -o "$out" \
        "$src" \
        -lhexagon
    echo "  -> $out"
}

for src in "$SCRIPT_DIR"/demo*.c; do
    [ -f "$src" ] || continue
    name=$(basename "$src" .c)
    echo "=== Compile: $name ==="
    compile_demo "$src"
done
