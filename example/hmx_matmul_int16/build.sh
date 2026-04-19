#!/usr/bin/env bash
#
# build.sh — compile the int16-matmul HMX test into a Hexagon ELF
# that runs under hexagon-sim + H2 booter (same recipe as ch01 test).
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

OUT="$SCRIPT_DIR/test_int16_matmul"

compile_bin() {
    local out="$1"; shift
    hexagon-clang -O2 -mv75 \
        -mhvx -mhvx-length=128B \
        -mhmx \
        -DARCHV=75 \
        -I "$H2_INSTALL/include" \
        -I "$H2_KERNEL_INC" \
        -I "$SCRIPT_DIR" \
        -moslib=h2 \
        -Wl,-L,"$H2_INSTALL/lib" \
        -Wl,--section-start=.start=0x02000000 \
        -o "$out" \
        "$@" \
        -lhexagon
    echo "  -> $out"
}

echo "=== Compile: test_int16_matmul ==="
compile_bin "$OUT" \
    "$SCRIPT_DIR/test_int16_matmul.c" \
    "$SCRIPT_DIR/int16_matmul_ref.c" \
    "$SCRIPT_DIR/int16_matmul_hmx.c"

# Probes: build if source exists.
for probe in probe_hmx_acc probe_dual_scale probe_bias_slots probe_2x2 probe_f16_profile; do
    src="$SCRIPT_DIR/$probe.c"
    if [ -f "$src" ]; then
        echo "=== Compile: $probe ==="
        compile_bin "$SCRIPT_DIR/$probe" "$src"
    fi
done

# probe_cycle_bench needs the int16 kernel + ref objects linked in.
if [ -f "$SCRIPT_DIR/probe_cycle_bench.c" ]; then
    echo "=== Compile: probe_cycle_bench ==="
    compile_bin "$SCRIPT_DIR/probe_cycle_bench" \
        "$SCRIPT_DIR/probe_cycle_bench.c" \
        "$SCRIPT_DIR/int16_matmul_hmx.c" \
        "$SCRIPT_DIR/int16_matmul_ref.c"
fi
