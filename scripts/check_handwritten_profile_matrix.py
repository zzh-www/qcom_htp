#!/usr/bin/env python3
"""Validate the handwritten HMX MatMul required-profile matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from check_qnn_artifact_standard import check_artifact


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "u8i8",
    "w4a8",
    "w8a16",
    "w4a16",
    "w4a16_chain1",
)
DEFERRED: tuple[str, ...] = ()
ACCEPTED_PROFILES = ("u8i8", "w4a8", "w8a16")
ACCEPTANCE_SCOPES = {
    "u8i8": "m4_canonical_e2e",
    "w4a8": "m4_canonical_e2e",
    "w8a16": "m4_canonical_e2e",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing profile matrix: {path}"]
    data = load_json(path)
    if data.get("schema") != "handwritten_hmx_matmul_profile_matrix.v1":
        errors.append(f"unexpected profile-matrix schema: {data.get('schema')!r}")
    required = data.get("required_final_gate_profiles")
    if required != list(REQUIRED):
        errors.append(f"required profile order changed: {required!r}")
    profiles = data.get("profiles", {})
    for name in REQUIRED:
        record = profiles.get(name)
        if not isinstance(record, dict):
            errors.append(f"missing required profile: {name}")
            continue
        if record.get("required_by_final_gate") is not True:
            errors.append(f"{name}: required_by_final_gate must be true")
        accepted = record.get("accepted") is True
        if accepted and name not in ACCEPTED_PROFILES:
            errors.append(f"{name}: profile must not be accepted without owned exactness/perf evidence")
        if accepted:
            expected_scope = ACCEPTANCE_SCOPES.get(name)
            if record.get("acceptance_scope") != expected_scope:
                errors.append(f"{name}: accepted profile must record acceptance_scope={expected_scope}")
            evidence = record.get("acceptance_evidence")
            if not isinstance(evidence, list) or not evidence:
                errors.append(f"{name}: accepted profile missing acceptance_evidence")
            if record.get("blocker") is not None:
                errors.append(f"{name}: accepted profile must not keep a blocker")
        elif not record.get("blocker"):
            errors.append(f"{name}: missing blocker")
        shape = record.get("shape_mkn")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(not isinstance(v, int) or v <= 0 for v in shape)
        ):
            errors.append(f"{name}: missing positive shape_mkn")
        qnn_reference = record.get("qnn_reference")
        if not isinstance(qnn_reference, str) or not (ROOT / qnn_reference).exists():
            errors.append(f"{name}: qnn_reference is missing: {qnn_reference!r}")
        else:
            standard_errors, _ = check_artifact(
                ROOT / qnn_reference,
                require_native_io=True,
                require_layout_flags=True,
                reject_float_io=True,
            )
            if standard_errors:
                errors.append(f"{name}: qnn_reference does not pass artifact standard: {standard_errors}")
        if name.endswith("_chain1") and record.get("chain") != 1:
            errors.append(f"{name}: chain1 profile must record chain=1")
    for name in DEFERRED:
        record = profiles.get(name)
        if not isinstance(record, dict):
            errors.append(f"missing deferred diagnostic profile: {name}")
            continue
        if record.get("required_by_final_gate") is not False:
            errors.append(f"{name}: deferred profile must not be required_by_final_gate")
        if record.get("accepted") is True:
            errors.append(f"{name}: deferred profile must not be marked accepted")
        if not record.get("blocker"):
            errors.append(f"{name}: missing deferred-scope note")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile-matrix",
        type=Path,
        default=ROOT / "example" / "handwritten_hmx_matmul" / "profile_matrix.json",
    )
    args = parser.parse_args()
    errors = validate(args.profile_matrix)
    if errors:
        print("handwritten profile matrix: FAILED", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"handwritten profile matrix: ok ({args.profile_matrix})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
