#!/usr/bin/env bash
#
# Install this repo's versioned git hooks for the current checkout.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

chmod +x .githooks/pre-push
git config core.hooksPath .githooks

echo "installed git hooks: core.hooksPath=$(git config --get core.hooksPath)"
