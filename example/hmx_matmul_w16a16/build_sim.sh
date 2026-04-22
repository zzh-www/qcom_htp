#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$ROOT_DIR/scripts/env.sh" >/dev/null
H2_INSTALL="$ROOT_DIR/tools/h2-install"
if [ -L "$H2_INSTALL" ]; then
    H2_ROOT="$(dirname "$(readlink -f "$H2_INSTALL")")"
else
    H2_ROOT="$(cd "$H2_INSTALL/.." && pwd)"
fi
H2_KERNEL_INC="$H2_ROOT/kernel/include"
OUT="$SCRIPT_DIR/test_w16a16_matmul_sim"
echo "=== Compile: test_w16a16_matmul_sim ==="
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
    "$SCRIPT_DIR/test_w16a16_matmul_sim.c" \
    "$SCRIPT_DIR/kernel/hmx_int16x16_matmul.c" \
    -lhexagon
echo "  -> $OUT"
