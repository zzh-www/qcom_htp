#!/usr/bin/env bash
# build_sim.sh — standalone hexagon-sim test for the w4a16 HMX kernel.
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
OUT="$SCRIPT_DIR/test_w4a16_matmul_sim"
echo "=== Compile: test_w4a16_matmul_sim ==="
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
    "$SCRIPT_DIR/test_w4a16_matmul_sim.c" \
    "$SCRIPT_DIR/kernel/hmx_int4_matmul.c" \
    -lhexagon
echo "  -> $OUT"
