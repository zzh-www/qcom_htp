#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/env.sh" >/dev/null

NDK_BIN="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin"
CXX="$NDK_BIN/clang++"
if [ ! -x "$CXX" ]; then
  echo "ERROR: missing Android clang++ at $CXX" >&2
  exit 1
fi

OUT="$SCRIPT_DIR/build/android-aarch64"
mkdir -p "$OUT"

"$CXX" -std=c++17 -O2 -Wall -Wextra \
  --target=aarch64-none-linux-android21 \
  --sysroot="$NDK_BIN/../sysroot" \
  -static-libstdc++ \
  -I "$SCRIPT_DIR/include" \
  "$SCRIPT_DIR/src/handwritten_hmx_matmul.cpp" \
  "$SCRIPT_DIR/tools/owned_smoke.cpp" \
  -o "$OUT/owned_smoke"

echo "built $OUT/owned_smoke"
