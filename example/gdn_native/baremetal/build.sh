#!/usr/bin/env bash
# Build the bare-metal FastRPC HMX-threading probe: DSP skel (hexagon) + host driver (aarch64-android).
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
source "$ROOT/scripts/env.sh" >/dev/null 2>&1
ARCH="${ARCH:-v75}"
SDK="$HEXAGON_SDK_ROOT"
HEXCC="$HEXAGON_TOOLS_ROOT/bin/hexagon-clang"
NDK_BIN="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin"
ARMCC="$NDK_BIN/clang"
mkdir -p build

INCS=(-Iinc -I"$SDK/incs" -I"$SDK/incs/stddef" -I"$SDK/ipc/fastrpc/incs")

echo "--- DSP skel (hexagon-$ARCH) ---"
"$HEXCC" -O2 -fPIC -shared -m$ARCH -mhvx -mhvx-length=128B -mhmx -DUSE_OS_QURT \
    "${INCS[@]}" \
    -I"$SDK/rtos/qurt/compute$ARCH/include/qurt" -I"$SDK/rtos/qurt/compute$ARCH/include/posix" \
    -Wall -Wno-unused-parameter \
    src/gdnbm_imp.c inc/gdnbm_skel.c -o build/libgdnbm_skel.so 2>&1 | head -30
[ -f build/libgdnbm_skel.so ] && echo "  -> build/libgdnbm_skel.so" || { echo "SKEL BUILD FAILED"; exit 1; }

echo "--- host driver (aarch64-android) ---"
"$ARMCC" --target=aarch64-none-linux-android21 --sysroot="$NDK_BIN/../sysroot" -O2 \
    "${INCS[@]}" -DUSE_OS_LINUX \
    src/gdnbm_test.c inc/gdnbm_stub.c \
    -L"$SDK/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc \
    -o build/gdnbm 2>&1 | head -30
[ -f build/gdnbm ] && echo "  -> build/gdnbm" || { echo "HOST BUILD FAILED"; exit 1; }
echo "BUILD OK"
