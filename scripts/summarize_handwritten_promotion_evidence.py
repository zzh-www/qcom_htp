#!/usr/bin/env python3
"""Summarize canonical promotion evidence for handwritten HMX MatMul.

The promotion record is intentionally narrower than final roadmap completion:
it only records whether a canonical family has enough no-QNN owned-body
evidence to close the M4 canonical E2E item.  Shape/profile acceptance and LPBQ
coverage remain outside the current tutorial/direct-HMX gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_FAMILIES = ("u8i8", "w4a8", "w8a16", "w4a16")
PROMOTABLE_FAMILIES = ("u8i8", "w4a8", "w8a16", "w4a16")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def body_identity_record(body_check: dict, family: str) -> dict:
    for record in body_check.get("results", []):
        if isinstance(record, dict) and record.get("family") == family:
            return record
    return {}


def positive_int(value: object) -> bool:
    return isinstance(value, int) and value > 0


def family_artifact_path(artifact_root: Path, family: str) -> Path:
    if family == "w4a16":
        return artifact_root / "artifact_body_w4a16_chain8_custom_baseline.json"
    return artifact_root / f"artifact_body_{family}.json"


def family_device_path(artifact_root: Path, family: str) -> Path:
    if family == "w4a16":
        return artifact_root / "device_body_w4a16_chain8_custom_baseline.json"
    return artifact_root / f"device_body_{family}.json"


def w4a16_bridge_path(artifact_root: Path) -> Path:
    return artifact_root / "w4a16_chain8_custom_baseline_native_bridge.json"


def summarize_family(artifact_root: Path, body_check: dict, family: str) -> dict:
    artifact_path = family_artifact_path(artifact_root, family)
    device_path = family_device_path(artifact_root, family)
    evidence = [rel(artifact_path), rel(device_path), rel(artifact_root / "body_check.json")]
    if family == "w4a16":
        evidence.append(rel(w4a16_bridge_path(artifact_root)))
    blockers: list[str] = []

    artifact = load_json(artifact_path) if artifact_path.is_file() else {}
    device = load_json(device_path) if device_path.is_file() else {}
    artifact_result = artifact.get("result", {})
    device_result = device.get("result", {})
    identity = body_identity_record(body_check, family)
    native_perf = artifact_result.get("native_perf_reference", {})
    bridge = load_json(w4a16_bridge_path(artifact_root)) if family == "w4a16" and w4a16_bridge_path(artifact_root).is_file() else {}
    bridge_layout = bridge.get("custom_vs_native_public_layout", {})

    if artifact.get("schema") != "handwritten_hmx_artifact_body_sim.v1":
        blockers.append("artifact_body_schema")
    if artifact.get("qnn_used") is not False:
        blockers.append("artifact_body_qnn_used")
    if artifact_result.get("exactness_status") != "byte_exact_checksum":
        blockers.append("artifact_body_not_byte_exact")
    if artifact_result.get("output_checksum") != artifact_result.get("native_raw_checksum"):
        blockers.append("artifact_checksum_mismatch")
    if native_perf.get("kernel_event_type") != "q::ConvLayer_s1.opt":
        blockers.append("native_kernel_event_type")
    for field in (
        "kernel_event_count",
        "kernel_cycles_sum",
        "kernel_packets_sum",
        "qnn_aggregate_cycles_sum",
        "timeline_span_cycles",
    ):
        if not positive_int(native_perf.get(field)):
            blockers.append(f"native_perf_{field}")

    if device.get("schema") != "handwritten_hmx_matmul_device_body.v1":
        blockers.append("device_body_schema")
    if device.get("qnn_used") is not False:
        blockers.append("device_body_qnn_used")
    if device.get("pass") is not True:
        blockers.append("device_body_not_pass")
    if device_result.get("device_execution") is not True:
        blockers.append("device_execution_missing")
    if device_result.get("hmx_body_entered") is not True:
        blockers.append("hmx_body_entry_missing")
    if device_result.get("exactness_status") != "byte_exact_device_diff":
        blockers.append("device_body_not_byte_exact")
    if device_result.get("byte_differences") != 0:
        blockers.append("device_byte_differences_nonzero")
    if device_result.get("output_checksum") != device_result.get("native_raw_checksum"):
        blockers.append("device_checksum_mismatch")
    if device_result.get("owned_timeline_evidence") != "qtimer_direct_body_span":
        blockers.append("device_timeline_evidence")
    for field in (
        "measure_repeats",
        "net_pcycles_per_step",
        "net_qticks_per_step",
        "runtime_us",
    ):
        if not positive_int(device_result.get(field)):
            blockers.append(f"device_perf_{field}")
    if positive_int(device_result.get("measure_repeats")) and device_result["measure_repeats"] < 20:
        blockers.append("device_measure_repeats_lt_20")

    if identity.get("byte_identity_pass") is not True:
        blockers.append("body_byte_identity")
    if identity.get("packet_equivalence_pass") is not True:
        blockers.append("body_packet_equivalence")
    if identity.get("packet_equivalence_method") != "byte_identity_preserves_native_packet_stream":
        blockers.append("body_packet_equivalence_method")

    if family == "w4a16":
        if bridge.get("accepted_bridge") is not True:
            blockers.append("w4a16_bridge_not_accepted")
        if bridge_layout.get("required_transform") != "native_transpose_2d":
            blockers.append("w4a16_bridge_transform")
        if bridge_layout.get("exact_after_transform") is not True:
            blockers.append("w4a16_bridge_not_exact_after_transform")

    promoted = family in PROMOTABLE_FAMILIES and not blockers
    return {
        "family": family,
        "promoted_for_m4_canonical_e2e": promoted,
        "status": "pass" if promoted else "open",
        "blockers": blockers,
        "evidence": evidence,
        "artifact_exactness_status": artifact_result.get("exactness_status"),
        "device_exactness_status": device_result.get("exactness_status"),
        "device_byte_differences": device_result.get("byte_differences"),
        "device_net_pcycles_per_step": device_result.get("net_pcycles_per_step"),
        "device_net_qticks_per_step": device_result.get("net_qticks_per_step"),
        "device_measure_repeats": device_result.get("measure_repeats"),
        "native_kernel_event_count": native_perf.get("kernel_event_count"),
        "native_kernel_cycles_sum": native_perf.get("kernel_cycles_sum"),
        "native_kernel_packets_sum": native_perf.get("kernel_packets_sum"),
        "native_timeline_span_cycles": native_perf.get("timeline_span_cycles"),
        "body_packet_equivalence_method": identity.get("packet_equivalence_method"),
        "w4a16_bridge_required_transform": bridge_layout.get("required_transform"),
        "w4a16_bridge_exact_after_transform": bridge_layout.get("exact_after_transform"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    artifact_root = args.artifact_root.resolve()
    body_check_path = artifact_root / "body_check.json"
    body_check = load_json(body_check_path) if body_check_path.is_file() else {}
    families = {
        family: summarize_family(artifact_root, body_check, family)
        for family in CANONICAL_FAMILIES
    }
    promoted = [
        family
        for family, record in families.items()
        if record.get("promoted_for_m4_canonical_e2e") is True
    ]
    payload = {
        "schema": "handwritten_hmx_matmul_promotion_evidence.v1",
        "artifact_root": rel(artifact_root),
        "qnn_used": False,
        "scope": {
            "promotable_families": list(PROMOTABLE_FAMILIES),
            "w16a16_policy": "retained_smoke_only_not_active_promotion_scope",
            "remaining_gates": ["shape_matrix"],
        },
        "families": families,
        "summary": {
            "promoted_families": promoted,
            "unpromoted_families": [
                family
                for family in CANONICAL_FAMILIES
                if family not in promoted
            ],
            "m4_promoted_count": len(promoted),
            "m4_promotable_count": len(PROMOTABLE_FAMILIES),
        },
    }
    out = args.json_out or artifact_root / "promotion_evidence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "handwritten promotion evidence: "
        f"promoted={promoted} out={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
