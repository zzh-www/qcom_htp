#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

uv run python - <<'PY'
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "scripts"))

from scripts.validate_handwritten_hmx_matmul import validate_completion_checklist


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def checklist_payload(root: Path, *, route: str, include_audit: bool) -> dict:
    evidence = [str(root / "roadmap_audit.json")] if include_audit else []
    return {
        "schema": "handwritten_hmx_matmul_completion_checklist.v1",
        "route": route,
        "artifact_root": str(root),
        "criteria": [
            {
                "id": 1,
                "requirement": "roadmap audit refresh",
                "status": "open",
                "satisfied": False,
                "evidence": evidence,
                "blockers": ["open blocker"],
            }
        ],
        "summary": {
            "roadmap_complete": False,
            "pass": 0,
            "open": 1,
            "fail": 0,
            "blockers": [{"id": 1, "requirement": "roadmap audit refresh", "blockers": ["open blocker"]}],
        },
    }


with tempfile.TemporaryDirectory(prefix="completion_checklist_audit_refresh_") as tmp:
    root = Path(tmp)
    write_json(root / "roadmap_audit.json", {"schema": "handwritten_hmx_matmul_roadmap_audit.v1"})

    write_json(root / "completion_checklist.json", checklist_payload(root, route="old_route", include_audit=True))
    errors: list[str] = []
    validate_completion_checklist(root, errors)
    assert any("direct body route" in error for error in errors), errors

    write_json(
        root / "completion_checklist.json",
        checklist_payload(root, route="direct_body_custom_baseline", include_audit=False),
    )
    errors = []
    validate_completion_checklist(root, errors)
    assert "completion checklist must include roadmap_audit.json evidence" in errors, errors

    write_json(
        root / "completion_checklist.json",
        checklist_payload(root, route="direct_body_custom_baseline", include_audit=True),
    )
    errors = []
    validate_completion_checklist(root, errors)
    assert errors == [], errors
PY
