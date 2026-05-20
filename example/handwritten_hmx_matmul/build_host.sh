#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CXX="${CXX:-clang++}"
command -v "$CXX" >/dev/null || { echo "ERROR: $CXX not found" >&2; exit 1; }

OUT="$SCRIPT_DIR/build/host"
mkdir -p "$OUT"

"$CXX" -std=c++17 -O2 -Wall -Wextra -pedantic \
  -I "$SCRIPT_DIR/include" \
  "$SCRIPT_DIR/src/handwritten_hmx_matmul.cpp" \
  "$SCRIPT_DIR/tools/owned_smoke.cpp" \
  -o "$OUT/owned_smoke"

echo "built $OUT/owned_smoke"
