#!/usr/bin/env python3
"""Build a current-route completion checklist for handwritten HMX MatMul."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ACTIVE_FAMILIES = ("u8i8", "w4a8", "w8a16", "w4a16")
PROMOTABLE_FAMILIES = ("u8i8", "w4a8", "w8a16", "w4a16")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def optional_json(path: Path) -> dict:
    return load_json(path) if path.is_file() else {}


def criterion(
    criterion_id: int,
    requirement: str,
    status: str,
    evidence: list[str],
    blockers: list[str],
) -> dict:
    return {
        "id": criterion_id,
        "requirement": requirement,
        "status": status,
        "satisfied": status == "pass",
        "evidence": evidence,
        "blockers": blockers,
    }


def tutorial_gate(root: Path) -> tuple[bool, dict]:
    data = optional_json(root / "w4a16_qnn_kernel_tutorial" / "device_result.json")
    matches = [
        data.get(key, {}).get("match") is True
        for key in (
            "prepared_state_compare",
            "call_abi_compare",
            "vtcm_offset_compare",
            "step_trace_compare",
            "hnh_path_compare",
        )
    ]
    result = data.get("result", {}) if isinstance(data.get("result"), dict) else {}
    ok = bool(data) and data.get("qnn_runtime_used") is False and all(matches)
    return ok, {
        "present": bool(data),
        "qnn_runtime_used": data.get("qnn_runtime_used"),
        "gate_matches": matches,
        "hnh_path": result.get("hnh_path"),
        "entered_and_returned": result.get("entered_and_returned"),
        "byte_differences": result.get("byte_differences"),
        "output_checksum": result.get("output_checksum"),
        "native_raw_checksum": result.get("native_raw_checksum"),
    }


def device_body_exact(root: Path, family: str) -> tuple[bool, dict]:
    data = optional_json(device_body_path(root, family))
    result = data.get("result", {}) if isinstance(data.get("result"), dict) else {}
    exact = (
        data.get("qnn_used") is False
        and data.get("pass") is True
        and result.get("device_execution") is True
        and result.get("hmx_body_entered") is True
        and result.get("exactness_status") == "byte_exact_device_diff"
        and result.get("byte_differences") == 0
        and result.get("output_checksum") == result.get("native_raw_checksum")
    )
    return exact, {
        "present": bool(data),
        "qnn_used": data.get("qnn_used"),
        "pass": data.get("pass"),
        "hmx_body_entered": result.get("hmx_body_entered"),
        "exactness_status": result.get("exactness_status"),
        "byte_differences": result.get("byte_differences"),
        "output_checksum": result.get("output_checksum"),
        "native_raw_checksum": result.get("native_raw_checksum"),
        "net_pcycles_per_step": result.get("net_pcycles_per_step"),
    }


def device_body_path(root: Path, family: str) -> Path:
    if family == "w4a16":
        return root / "device_body_w4a16_chain8_custom_baseline.json"
    return root / f"device_body_{family}.json"


def w4a16_bridge_exact(root: Path) -> tuple[bool, dict]:
    data = optional_json(root / "w4a16_chain8_custom_baseline_native_bridge.json")
    layout = data.get("custom_vs_native_public_layout", {})
    exact = (
        data.get("accepted_bridge") is True
        and layout.get("required_transform") == "native_transpose_2d"
        and layout.get("exact_after_transform") is True
        and layout.get("native_transpose_2d", {}).get("byte_differences") == 0
    )
    return exact, {
        "present": bool(data),
        "accepted_bridge": data.get("accepted_bridge"),
        "required_transform": layout.get("required_transform"),
        "exact_after_transform": layout.get("exact_after_transform"),
        "native_transpose_2d": layout.get("native_transpose_2d"),
    }


def summarize(root: Path) -> dict:
    audit = optional_json(root / "roadmap_audit.json")
    promotion = optional_json(root / "promotion_evidence.json")
    promoted = (
        promotion.get("summary", {}).get("promoted_families", [])
        if isinstance(promotion.get("summary"), dict)
        else []
    )
    tutorial_ok, tutorial = tutorial_gate(root)
    body_exact = {}
    body_records = {}
    for family in ACTIVE_FAMILIES:
        exact, record = device_body_exact(root, family)
        body_exact[family] = exact
        body_records[family] = record
    bridge_exact, bridge_record = w4a16_bridge_exact(root)

    criteria = [
        criterion(
            1,
            "current W4A16 route is tutorial/direct-HMX and does not use QNN runtime",
            "pass" if tutorial_ok else "open",
            [str(root / "w4a16_qnn_kernel_tutorial" / "device_result.json")],
            [] if tutorial_ok else [f"tutorial_gate={tutorial}"],
        ),
        criterion(
            2,
            "retained exact families keep QNN-free direct-device body evidence",
            "pass" if all(body_exact[family] for family in PROMOTABLE_FAMILIES) else "open",
            [str(device_body_path(root, family)) for family in PROMOTABLE_FAMILIES],
            [
                f"{family}={body_records[family]}"
                for family in PROMOTABLE_FAMILIES
                if not body_exact[family]
            ],
        ),
        criterion(
            3,
            "W4A16 chain8 custom-baseline direct-HMX output is byte-exact and bridges to native via native_transpose_2d",
            "pass" if body_exact["w4a16"] and bridge_exact else "open",
            [
                str(root / "device_body_w4a16_chain8_custom_baseline.json"),
                str(root / "w4a16_chain8_custom_baseline_native_bridge.json"),
            ],
            []
            if body_exact["w4a16"] and bridge_exact
            else [f"w4a16_chain8={body_records['w4a16']}", f"bridge={bridge_record}"],
        ),
        criterion(
            4,
            "roadmap audit has been regenerated for this artifact root",
            "pass" if audit.get("schema") == "handwritten_hmx_matmul_roadmap_audit.v1" else "open",
            [str(root / "roadmap_audit.json")],
            [] if audit.get("schema") == "handwritten_hmx_matmul_roadmap_audit.v1" else ["missing_or_stale_roadmap_audit"],
        ),
    ]
    blockers = [
        {
            "id": item["id"],
            "requirement": item["requirement"],
            "blockers": item["blockers"],
        }
        for item in criteria
        if item["status"] != "pass"
    ]
    return {
        "schema": "handwritten_hmx_matmul_completion_checklist.v1",
        "route": "tutorial_direct_hmx_wrapper",
        "artifact_root": str(root),
        "scope": {
            "active_families": list(ACTIVE_FAMILIES),
            "promotable_retained_families": list(PROMOTABLE_FAMILIES),
            "w16a16_policy": "retained_reference_only_not_active",
            "old_qnn_blackbox_route": "closed_for_current_goal",
        },
        "criteria": criteria,
        "summary": {
            "roadmap_complete": not blockers,
            "pass": sum(1 for item in criteria if item["status"] == "pass"),
            "open": sum(1 for item in criteria if item["status"] == "open"),
            "fail": sum(1 for item in criteria if item["status"] == "fail"),
            "promoted_families": promoted,
            "blockers": blockers,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()

    payload = summarize(args.artifact_root.resolve())
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = payload["summary"]
    print(
        "handwritten completion checklist: ok "
        f"({args.json_out}, pass={summary['pass']}, open={summary['open']}, fail={summary['fail']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
