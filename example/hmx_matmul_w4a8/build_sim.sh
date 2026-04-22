#!/usr/bin/env bash
#
# build_sim.sh — compile the standalone hexagon-sim test for the w4a8
# kernel. Mirrors example/hmx_matmul_int16/build.sh's pattern.
#
# Output: test_w4a8_matmul_sim (Hexagon ELF, runs on hexagon-sim + H2).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env.sh" >/dev/null

H2_INSTALL="$ROOT_DIR/tools/h2-install"
if [ -L "$H2_INSTALL" ]; then
    H2_ROOT="$(dirname "$(readlink -f "$H2_INSTALL")")"
else
    H2_ROOT="$(cd "$H2_INSTALL/.." && pwd)"
fi
H2_KERNEL_INC="$H2_ROOT/kernel/include"

OUT="$SCRIPT_DIR/test_w4a8_matmul_sim"
echo "=== Compile: test_w4a8_matmul_sim ==="
hexagon-clang -O2 -mv75 \
    -mhvx -mhvx-length=128B -mhmx \
    -DARCHV=75 \
    -I "$H2_INSTALL/include" \
    -I "$H2_KERNEL_INC" \
    -I "$SCRIPT_DIR" \
    -moslib=h2 \
    -Wl,-L,"$H2_INSTALL/lib" \
    -Wl,--section-start=.start=0x02000000 \
    -o "$OUT" \
    "$SCRIPT_DIR/test_w4a8_matmul_sim.c" \
    "$SCRIPT_DIR/kernel/hmx_int4xint8_matmul.c" \
    -lhexagon
echo "  -> $OUT"
