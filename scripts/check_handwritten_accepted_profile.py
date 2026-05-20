#!/usr/bin/env python3
"""Validate one accepted handwritten HMX MatMul profile against gate evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_PROFILES = {
    "u8i8": "u8i8",
    "w4a8": "w4a8",
    "w8a16": "w8a16",
}
ACCEPTANCE_SCOPES = {
    "u8i8": "m4_canonical_e2e",
    "w4a8": "m4_canonical_e2e",
    "w8a16": "m4_canonical_e2e",
}


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing json: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def evidence_path(artifact_root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return artifact_root / path


def validate_profile(artifact_root: Path, profile_name: str) -> list[str]:
    errors: list[str] = []
    family = ACCEPTED_PROFILES.get(profile_name)
    if family is None:
        return [f"profile is not accepted by the current gate: {profile_name}"]

    profile_matrix = load_json(ROOT / "example" / "handwritten_hmx_matmul" / "profile_matrix.json")
    profile = profile_matrix.get("profiles", {}).get(profile_name, {})
    if profile.get("accepted") is not True:
        errors.append(f"{profile_name}: profile_matrix accepted is not true")
    expected_scope = ACCEPTANCE_SCOPES[profile_name]
    if profile.get("acceptance_scope") != expected_scope:
        errors.append(f"{profile_name}: acceptance_scope must be {expected_scope}")
    if profile.get("family") != family:
        errors.append(f"{profile_name}: family mismatch: {profile.get('family')!r}")
    if profile.get("required_by_final_gate") is not True:
        errors.append(f"{profile_name}: required_by_final_gate must be true")
    if profile.get("blocker") is not None:
        errors.append(f"{profile_name}: accepted profile must not carry a blocker")
    for item in profile.get("acceptance_evidence") or []:
        path = evidence_path(artifact_root, item)
        if path is None or not path.is_file():
            errors.append(f"{profile_name}: missing acceptance evidence: {item!r}")

    promotion = load_json(artifact_root / "promotion_evidence.json")
    family_promotion = promotion.get("families", {}).get(family, {})
    if family_promotion.get("promoted_for_m4_canonical_e2e") is not True:
        errors.append(f"{profile_name}: promotion_evidence does not promote {family}")
    if family_promotion.get("status") != "pass":
        errors.append(f"{profile_name}: promotion status is not pass")
    if family_promotion.get("device_byte_differences") != 0:
        errors.append(f"{profile_name}: device byte differences are not zero")
    if family_promotion.get("device_exactness_status") != "byte_exact_device_diff":
        errors.append(f"{profile_name}: device exactness status is not byte_exact_device_diff")
    if family_promotion.get("artifact_exactness_status") != "byte_exact_checksum":
        errors.append(f"{profile_name}: artifact exactness status is not byte_exact_checksum")
    if family_promotion.get("body_packet_equivalence_method") != "byte_identity_preserves_native_packet_stream":
        errors.append(f"{profile_name}: packet-equivalence method mismatch")
    for field in (
        "device_measure_repeats",
        "device_net_pcycles_per_step",
        "device_net_qticks_per_step",
        "native_kernel_event_count",
        "native_kernel_cycles_sum",
        "native_kernel_packets_sum",
        "native_timeline_span_cycles",
    ):
        value = family_promotion.get(field)
        if not isinstance(value, int) or value <= 0:
            errors.append(f"{profile_name}: promotion evidence missing positive {field}")
    if isinstance(family_promotion.get("device_measure_repeats"), int) and family_promotion["device_measure_repeats"] < 20:
        errors.append(f"{profile_name}: device_measure_repeats must be at least 20")

    device = load_json(artifact_root / f"device_body_{family}.json")
    device_result = device.get("result", {})
    if device.get("qnn_used") is not False:
        errors.append(f"{profile_name}: device body qnn_used must be false")
    if device.get("pass") is not True:
        errors.append(f"{profile_name}: device body did not pass")
    if device_result.get("device_execution") is not True:
        errors.append(f"{profile_name}: device execution evidence missing")
    if device_result.get("hmx_body_entered") is not True:
        errors.append(f"{profile_name}: HMX body entry evidence missing")
    if device_result.get("byte_differences") != 0:
        errors.append(f"{profile_name}: device body byte_differences is not zero")
    if device_result.get("owned_timeline_evidence") != "qtimer_direct_body_span":
        errors.append(f"{profile_name}: device timeline evidence mismatch")

    artifact = load_json(artifact_root / f"artifact_body_{family}.json")
    artifact_result = artifact.get("result", {})
    if artifact.get("qnn_used") is not False:
        errors.append(f"{profile_name}: artifact body qnn_used must be false")
    if artifact_result.get("exactness_status") != "byte_exact_checksum":
        errors.append(f"{profile_name}: artifact body exactness status mismatch")
    if artifact_result.get("output_checksum") != artifact_result.get("native_raw_checksum"):
        errors.append(f"{profile_name}: artifact checksum mismatch")
    native_perf = artifact_result.get("native_perf_reference", {})
    if native_perf.get("kernel_event_type") != "q::ConvLayer_s1.opt":
        errors.append(f"{profile_name}: native kernel event type mismatch")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=tuple(ACCEPTED_PROFILES))
    args = parser.parse_args()

    try:
        errors = validate_profile(args.artifact_root.resolve(), args.profile)
    except ValueError as exc:
        errors = [str(exc)]
    if errors:
        print(f"handwritten accepted profile {args.profile}: FAILED", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"handwritten accepted profile {args.profile}: ok ({args.artifact_root})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
