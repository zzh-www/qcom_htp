# env.sh — source this to set up Hexagon / QNN / Android NDK env vars.
#
# Usage:
#   source scripts/env.sh
#
# Intended to be sourced, not executed. Does not `set -e` / `exit` so an
# incomplete install just produces a warning.

_qcom_env_root="$( cd "$( dirname "${BASH_SOURCE[0]:-$0}" )/.." && pwd )"

export HEXAGON_SDK_ROOT="$_qcom_env_root/tools/hexagon-sdk"
export QNN_SDK_ROOT="$_qcom_env_root/tools/qnn-sdk"
export ANDROID_NDK_ROOT="$_qcom_env_root/tools/android-ndk"
export ANDROID_NDK_HOME="$ANDROID_NDK_ROOT"

# HEXAGON_Tools subversion differs between SDK releases; glob for it.
_hex_tools_dir=$(ls -d "$HEXAGON_SDK_ROOT"/tools/HEXAGON_Tools/*/Tools 2>/dev/null | head -1)
export HEXAGON_TOOLS_ROOT="${_hex_tools_dir:-$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/unknown/Tools}"

_ndk_bin="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin"

# Prepend toolchain dirs to PATH (avoid duplicate entries on re-source).
case ":$PATH:" in
    *":$HEXAGON_TOOLS_ROOT/bin:"*) ;;
    *) PATH="$HEXAGON_TOOLS_ROOT/bin:$PATH" ;;
esac
case ":$PATH:" in
    *":$_ndk_bin:"*) ;;
    *) PATH="$_ndk_bin:$PATH" ;;
esac
export PATH

# QNN host libs (for qnn-net-run and friends on x86_64 Linux).
_qnn_host_lib="$QNN_SDK_ROOT/lib/x86_64-linux-clang"
case ":${LD_LIBRARY_PATH:-}:" in
    *":$_qnn_host_lib:"*) ;;
    *) export LD_LIBRARY_PATH="$_qnn_host_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

# Activate project-local Python venv (managed by uv). Put .venv/bin first
# on PATH so `python`, `pip`, etc. pick up the project environment instead
# of the system interpreter.
_venv="$_qcom_env_root/.venv"
if [ -x "$_venv/bin/python" ]; then
    case ":$PATH:" in
        *":$_venv/bin:"*) ;;
        *) export PATH="$_venv/bin:$PATH" ;;
    esac
    export VIRTUAL_ENV="$_venv"
fi

# Sanity-check roots; warn but do not exit.
_qcom_env_warn=0
for _d in "$HEXAGON_SDK_ROOT" "$QNN_SDK_ROOT" "$ANDROID_NDK_ROOT"; do
    if [ ! -d "$_d" ]; then
        echo "warning: missing $_d (run ./install.sh)" >&2
        _qcom_env_warn=1
    fi
done
if [ ! -x "$HEXAGON_TOOLS_ROOT/bin/hexagon-clang" ]; then
    echo "warning: hexagon-clang not found under $HEXAGON_TOOLS_ROOT/bin" >&2
    _qcom_env_warn=1
fi

if [ "$_qcom_env_warn" = 0 ]; then
    echo "qcom env: HEXAGON=$HEXAGON_SDK_ROOT  QNN=$QNN_SDK_ROOT  NDK=$ANDROID_NDK_ROOT"
fi

unset _qcom_env_root _hex_tools_dir _ndk_bin _qnn_host_lib _qcom_env_warn _d _venv
