#!/usr/bin/env bash
#
# Run the full kernel CI preflight, then push through the fast proof-check hook.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

"$ROOT_DIR/scripts/run_kernel_ci_preflight.sh"

if [ "$#" -eq 0 ]; then
    set -- origin "$(git branch --show-current)"
fi

git push "$@"
