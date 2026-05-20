#!/usr/bin/env python3
"""Validate the handwritten HMX MatMul shape-matrix scaffold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("u8i8", "w4a8", "w8a16", "w4a16")
ACCEPTED_CANONICAL = ("u8i8", "w4a8", "w8a16")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing shape matrix: {path}"]
    data = load_json(path)
    if data.get("schema") != "handwritten_hmx_matmul_shape_matrix.v1":
        errors.append(f"unexpected shape-matrix schema: {data.get('schema')!r}")
    policy = data.get("policy", {})
    if policy.get("static_weights_only") is not True:
        errors.append("shape policy must keep static_weights_only=true")
    if int(policy.get("minimum_tile_multiple", 0)) != 32:
        errors.append("shape policy must record minimum_tile_multiple=32")
    families = data.get("families", {})
    for family in FAMILIES:
        record = families.get(family)
        if not isinstance(record, dict):
            errors.append(f"shape matrix missing family: {family}")
            continue
        for bucket in ("accepted", "diagnostic", "rejected"):
            entries = record.get(bucket)
            if not isinstance(entries, list):
                errors.append(f"{family}: {bucket} must be a list")
                continue
            if bucket != "accepted" and not entries:
                errors.append(f"{family}: {bucket} must have at least one entry")
            for idx, entry in enumerate(entries):
                shape = entry.get("shape_mkn")
                if (
                    not isinstance(shape, list)
                    or len(shape) != 3
                    or any(not isinstance(v, int) or v <= 0 for v in shape)
                ):
                    errors.append(f"{family}: {bucket}[{idx}] missing positive shape_mkn")
                if not entry.get("reason"):
                    errors.append(f"{family}: {bucket}[{idx}] missing reason")
                oracle = entry.get("oracle")
                if oracle:
                    oracle_path = ROOT / oracle
                    if not oracle_path.exists():
                        errors.append(f"{family}: {bucket}[{idx}] oracle path is missing: {oracle}")
                if bucket == "accepted":
                    if family not in ACCEPTED_CANONICAL:
                        errors.append(f"{family}: accepted shapes are not allowed until exactness/perf promotion exists")
                    if entry.get("acceptance_scope") != "m4_canonical_e2e":
                        errors.append(f"{family}: accepted[{idx}] must record acceptance_scope=m4_canonical_e2e")
                    evidence = entry.get("evidence")
                    if not isinstance(evidence, list) or not evidence:
                        errors.append(f"{family}: accepted[{idx}] missing promotion evidence list")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape-matrix",
        type=Path,
        default=ROOT / "example" / "handwritten_hmx_matmul" / "shape_matrix.json",
    )
    args = parser.parse_args()
    errors = validate(args.shape_matrix)
    if errors:
        print("handwritten shape matrix: FAILED", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"handwritten shape matrix: ok ({args.shape_matrix})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
