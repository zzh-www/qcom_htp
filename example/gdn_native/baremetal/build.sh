#!/usr/bin/env bash
# Build the bare-metal FastRPC GDN solve: DSP skel (hexagon C++) + host driver (aarch64-android).
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../../.. && pwd)"
source "$ROOT/scripts/env.sh" >/dev/null 2>&1
ARCH="${ARCH:-v75}"
SDK="$HEXAGON_SDK_ROOT"
HEXCXX="$HEXAGON_TOOLS_ROOT/bin/hexagon-clang++"
HEXCC="$HEXAGON_TOOLS_ROOT/bin/hexagon-clang"
NDK_BIN="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin"
ARMCC="$NDK_BIN/clang"
mkdir -p build
rm -f build/libgdnbm_skel.so build/gdnbm build/*.o

INCS=(-Iinc -I"$SDK/incs" -I"$SDK/incs/stddef" -I"$SDK/ipc/fastrpc/incs")
HEXFLAGS=(-O2 -fPIC -m$ARCH -mhvx -mhvx-length=128B -mhmx -DUSE_OS_QURT -DPREPARE_DISABLED "${INCS[@]}"
    -I"$SDK/rtos/qurt/compute$ARCH/include/qurt" -I"$SDK/rtos/qurt/compute$ARCH/include/posix")

echo "--- DSP skel (gdnbm_skel.c as C) ---"
"$HEXCC" "${HEXFLAGS[@]}" -c inc/gdnbm_skel.c -o build/gdnbm_skel.o 2>&1 | head -20 || { echo "SKEL C FAIL"; exit 1; }
echo "--- DSP imp (gdnbm_imp.cpp as C++, includes the device solve) ---"
"$HEXCXX" -std=c++17 "${HEXFLAGS[@]}" \
    -Wno-unused-function -Wno-unused-variable -Wno-unused-parameter -Wno-missing-braces \
    -c src/gdnbm_imp.cpp -o build/gdnbm_imp.o 2>&1 | head -40 || { echo "IMP C++ FAIL"; exit 1; }
echo "--- link skel .so ---"
"$HEXCXX" -shared -m$ARCH build/gdnbm_skel.o build/gdnbm_imp.o -o build/libgdnbm_skel.so 2>&1 | head -20
[ -f build/libgdnbm_skel.so ] && echo "  -> build/libgdnbm_skel.so" || { echo "SKEL LINK FAILED"; exit 1; }

echo "--- host driver (aarch64-android) ---"
"$ARMCC" --target=aarch64-none-linux-android21 --sysroot="$NDK_BIN/../sysroot" -O2 \
    "${INCS[@]}" -DUSE_OS_LINUX \
    src/gdnbm_test.c inc/gdnbm_stub.c \
    -L"$SDK/ipc/fastrpc/remote/ship/android_aarch64" -lcdsprpc \
    -o build/gdnbm 2>&1 | head -30
[ -f build/gdnbm ] && echo "  -> build/gdnbm" || { echo "HOST BUILD FAILED"; exit 1; }
echo "BUILD OK"
