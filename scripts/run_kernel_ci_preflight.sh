#!/usr/bin/env bash
#
# Run the full device-backed kernel CI before opening a git push connection.
# On success, record a proof tied to the current HEAD and tree for pre-push.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ "${ARTIFACT_ONLY:-0}" = "1" ]; then
    cat >&2 <<'EOF'
ERROR: kernel CI preflight must run the full device E2E gate.
       ARTIFACT_ONLY=1 is only allowed for manual local smoke tests.
EOF
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is required to record the kernel CI preflight proof." >&2
    exit 1
fi

HEAD_SHA="$(git rev-parse HEAD)"
TREE_SHA="$(git rev-parse HEAD^{tree})"
PROOF_FILE="$(git rev-parse --git-path qcom_htp_kernel_ci_pass.json)"
GATE_CMD="tests/qnn_kernel_e2e/run_all.sh"

echo "=== qcom_htp kernel CI preflight ==="
echo "    head: $HEAD_SHA"
echo "    tree: $TREE_SHA"
echo "    gate: $GATE_CMD"

"$ROOT_DIR/$GATE_CMD"

END_HEAD_SHA="$(git rev-parse HEAD)"
END_TREE_SHA="$(git rev-parse HEAD^{tree})"

if [ "$END_HEAD_SHA" != "$HEAD_SHA" ] || [ "$END_TREE_SHA" != "$TREE_SHA" ]; then
    cat >&2 <<EOF
ERROR: HEAD or tree changed while kernel CI preflight was running.
       start HEAD: $HEAD_SHA
       end HEAD:   $END_HEAD_SHA
       start tree: $TREE_SHA
       end tree:   $END_TREE_SHA
       Re-run scripts/run_kernel_ci_preflight.sh for the final commit.
EOF
    exit 1
fi

PASSED_AT_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
mkdir -p "$(dirname "$PROOF_FILE")"

python3 - "$PROOF_FILE" "$HEAD_SHA" "$TREE_SHA" "$GATE_CMD" "$PASSED_AT_UTC" "${DEVICE:-}" <<'PY'
import json
import sys

proof_path, head_sha, tree_sha, gate_cmd, passed_at_utc, device = sys.argv[1:7]

proof = {
    "schema": "qcom_htp_kernel_ci_preflight.v1",
    "head": head_sha,
    "tree": tree_sha,
    "command": gate_cmd,
    "device": device,
    "passed_at_utc": passed_at_utc,
}

with open(proof_path, "w", encoding="utf-8") as f:
    json.dump(proof, f, indent=2, sort_keys=True)
    f.write("\n")
PY

echo "=== qcom_htp kernel CI preflight proof recorded: $PROOF_FILE ==="
