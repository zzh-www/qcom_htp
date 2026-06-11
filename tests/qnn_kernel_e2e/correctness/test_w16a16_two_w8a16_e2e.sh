#!/usr/bin/env bash
# w16a16 has no single-pass kernel: it is byte-exactly two 8-bit-weight passes (2x w8a16).
# This gate proves the int16->(int8 hi, uint8 lo) byte decomposition and matmul identity are
# exact, and that the high pass *256 overflows int32 (=> the two passes must drain separately).
# Principle + usage: docs/w16a16_is_two_w8a16.md
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"

for shape in "64 64 64" "256 256 256" "128 256 96"; do
  set -- $shape
  python3 "$ROOT_DIR/scripts/verify_w16a16_two_w8a16.py" --M "$1" --K "$2" --N "$3"
done
