#!/usr/bin/env bash
#
# install.sh — install Hexagon SDK, QAIRT (QNN) SDK, and Android NDK
# into ./tools. Idempotent: re-runs skip phases already complete.
#
# Usage:
#   bash scripts/install.sh
#
# After install:
#   source scripts/env.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS_DIR="$ROOT_DIR/tools"
DOWNLOADS_DIR="$ROOT_DIR/downloads"

mkdir -p "$TOOLS_DIR" "$DOWNLOADS_DIR"

# ============================================================
# Versions and URLs
# ============================================================
HEXAGON_VERSION="6.6.0.0"
HEXAGON_ZIP="Hexagon_SDK_Linux_${HEXAGON_VERSION}.zip"
HEXAGON_URL="https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/sdks/Hexagon_SDK/Linux/Debian/${HEXAGON_VERSION}/Hexagon_SDK_Linux.zip"

QNN_VERSION="2.46.0.260424"
QNN_ZIP="v${QNN_VERSION}.zip"
QNN_URL="https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${QNN_VERSION}/${QNN_ZIP}"

NDK_VERSION="r27c"
NDK_ZIP="android-ndk-${NDK_VERSION}-linux.zip"
NDK_URL="https://dl.google.com/android/repository/${NDK_ZIP}"

# ============================================================
# Helpers
# ============================================================
download() {
    local url="$1" dest="$2"
    if [ -f "$dest" ]; then
        echo "  cached: $(basename "$dest")"
    else
        echo "  downloading $(basename "$dest")"
        wget -q --show-progress "$url" -O "$dest.part"
        mv "$dest.part" "$dest"
    fi
}

# ============================================================
# 1. Hexagon SDK
# ============================================================
echo "=== [1/3] Hexagon SDK ${HEXAGON_VERSION} ==="
if [ -d "$TOOLS_DIR/hexagon-sdk" ]; then
    echo "  already installed: $TOOLS_DIR/hexagon-sdk"
else
    download "$HEXAGON_URL" "$DOWNLOADS_DIR/$HEXAGON_ZIP"
    TMP_SDK="$TOOLS_DIR/_hex_extract"
    rm -rf "$TMP_SDK"
    mkdir -p "$TMP_SDK"
    echo "  extracting..."
    unzip -q "$DOWNLOADS_DIR/$HEXAGON_ZIP" -d "$TMP_SDK"

    # Probe the extracted layout. Expect either Hexagon_SDK/<version>/ or
    # just <version>/ at the top. Resolve the deepest dir that contains
    # the 'tools' subdir.
    SDK_INNER=""
    if [ -d "$TMP_SDK/Hexagon_SDK/$HEXAGON_VERSION" ]; then
        SDK_INNER="$TMP_SDK/Hexagon_SDK/$HEXAGON_VERSION"
    elif [ -d "$TMP_SDK/$HEXAGON_VERSION" ]; then
        SDK_INNER="$TMP_SDK/$HEXAGON_VERSION"
    else
        SDK_INNER="$(find "$TMP_SDK" -maxdepth 4 -type d -name tools -printf '%h\n' | head -1)"
    fi

    if [ -z "$SDK_INNER" ] || [ ! -d "$SDK_INNER/tools" ]; then
        echo "ERROR: could not locate Hexagon SDK root in extracted archive"
        echo "       contents of $TMP_SDK:"
        find "$TMP_SDK" -maxdepth 3 -type d | sed 's/^/         /'
        exit 1
    fi

    mv "$SDK_INNER" "$TOOLS_DIR/hexagon-sdk"
    rm -rf "$TMP_SDK"
fi

HEX_CLANG="$(ls "$TOOLS_DIR"/hexagon-sdk/tools/HEXAGON_Tools/*/Tools/bin/hexagon-clang 2>/dev/null | head -1 || true)"
if [ -z "$HEX_CLANG" ] || [ ! -x "$HEX_CLANG" ]; then
    echo "ERROR: hexagon-clang not found under $TOOLS_DIR/hexagon-sdk/tools/HEXAGON_Tools/*/Tools/bin/"
    exit 1
fi
echo "  hexagon-clang: $HEX_CLANG"

# ============================================================
# 2. QAIRT / QNN SDK
# ============================================================
echo ""
echo "=== [2/3] QAIRT (QNN) SDK ${QNN_VERSION} ==="
if [ -d "$TOOLS_DIR/qnn-sdk" ]; then
    echo "  already installed: $TOOLS_DIR/qnn-sdk"
else
    download "$QNN_URL" "$DOWNLOADS_DIR/$QNN_ZIP"
    TMP_QNN="$TOOLS_DIR/_qnn_extract"
    rm -rf "$TMP_QNN"
    mkdir -p "$TMP_QNN"
    echo "  extracting..."
    unzip -q "$DOWNLOADS_DIR/$QNN_ZIP" -d "$TMP_QNN"

    QNN_INNER=""
    if [ -d "$TMP_QNN/qairt/$QNN_VERSION" ]; then
        QNN_INNER="$TMP_QNN/qairt/$QNN_VERSION"
    elif [ -d "$TMP_QNN/$QNN_VERSION" ]; then
        QNN_INNER="$TMP_QNN/$QNN_VERSION"
    else
        QNN_INNER="$(find "$TMP_QNN" -maxdepth 3 -type d -name 'include' -printf '%h\n' | head -1)"
    fi

    if [ -z "$QNN_INNER" ] || [ ! -d "$QNN_INNER/include/QNN" ]; then
        echo "ERROR: could not locate QAIRT SDK root in extracted archive"
        echo "       contents of $TMP_QNN:"
        find "$TMP_QNN" -maxdepth 3 -type d | sed 's/^/         /'
        exit 1
    fi

    mv "$QNN_INNER" "$TOOLS_DIR/qnn-sdk"
    rm -rf "$TMP_QNN"
fi

for f in \
    "$TOOLS_DIR/qnn-sdk/include/QNN/QnnInterface.h" \
    "$TOOLS_DIR/qnn-sdk/lib/aarch64-android/libQnnHtp.so"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: expected QAIRT file missing: $f"
        exit 1
    fi
done
echo "  QNN headers: $TOOLS_DIR/qnn-sdk/include/QNN/"

# ============================================================
# 3. Android NDK
# ============================================================
echo ""
echo "=== [3/3] Android NDK ${NDK_VERSION} ==="
if [ -d "$TOOLS_DIR/android-ndk" ]; then
    echo "  already installed: $TOOLS_DIR/android-ndk"
else
    download "$NDK_URL" "$DOWNLOADS_DIR/$NDK_ZIP"
    TMP_NDK="$TOOLS_DIR/_ndk_extract"
    rm -rf "$TMP_NDK"
    mkdir -p "$TMP_NDK"
    echo "  extracting..."
    unzip -q "$DOWNLOADS_DIR/$NDK_ZIP" -d "$TMP_NDK"

    NDK_INNER="$TMP_NDK/android-ndk-${NDK_VERSION}"
    if [ ! -d "$NDK_INNER" ]; then
        NDK_INNER="$(find "$TMP_NDK" -maxdepth 2 -type d -name 'android-ndk-*' | head -1)"
    fi
    if [ -z "$NDK_INNER" ] || [ ! -d "$NDK_INNER/toolchains" ]; then
        echo "ERROR: could not locate NDK root in extracted archive"
        exit 1
    fi
    mv "$NDK_INNER" "$TOOLS_DIR/android-ndk"
    rm -rf "$TMP_NDK"
fi

NDK_CLANG="$TOOLS_DIR/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang"
if [ ! -x "$NDK_CLANG" ]; then
    echo "ERROR: NDK clang not found: $NDK_CLANG"
    exit 1
fi
echo "  aarch64 clang: $NDK_CLANG"

# ============================================================
# Compatibility symlink for existing tutorial scripts
# ============================================================
TUT_DIR="$ROOT_DIR/docs/hexagon-tutorial"
if [ -d "$TUT_DIR" ] && [ ! -e "$TUT_DIR/tools" ]; then
    ln -s ../../tools "$TUT_DIR/tools"
    echo ""
    echo "  symlink: $TUT_DIR/tools -> ../../tools"
fi

echo ""
echo "=== Done ==="
echo "  Hexagon SDK: $TOOLS_DIR/hexagon-sdk"
echo "  QNN SDK:     $TOOLS_DIR/qnn-sdk"
echo "  Android NDK: $TOOLS_DIR/android-ndk"
echo ""
echo "Next: source scripts/env.sh"
