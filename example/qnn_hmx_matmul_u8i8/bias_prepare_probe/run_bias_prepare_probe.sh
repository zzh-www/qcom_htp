#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

source scripts/env.sh >/dev/null
export PYTHONPATH="$QNN_SDK_ROOT/lib/python${PYTHONPATH:+:$PYTHONPATH}"

uv run python example/qnn_hmx_matmul_u8i8/bias_prepare_probe/run_bias_prepare_probe.py "$@"
