#!/usr/bin/env python3
"""Audit the current handwritten HMX MatMul roadmap evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def exact_device_body(data: dict | None) -> bool:
    result = (data or {}).get("result", {})
    return (
        (data or {}).get("qnn_used") is False
        and (data or {}).get("pass") is True
        and result.get("device_execution") is True
        and result.get("hmx_body_entered") is True
        and result.get("exactness_status") == "byte_exact_device_diff"
        and result.get("byte_differences") == 0
        and result.get("output_checksum") == result.get("native_raw_checksum")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--require-device", action="store_true")
    args = parser.parse_args()

    root = args.artifact_root.resolve()
    tutorial = load_json(root / "w4a16_qnn_kernel_tutorial" / "device_result.json")
    w4a16_chain8 = load_json(root / "device_body_w4a16_chain8_custom_baseline.json")
    w4a16_bridge = load_json(root / "w4a16_chain8_custom_baseline_native_bridge.json")
    result = (tutorial or {}).get("result", {})
    gates = [
        (tutorial or {}).get(key, {}).get("match") is True
        for key in (
            "prepared_state_compare",
            "call_abi_compare",
            "vtcm_offset_compare",
            "step_trace_compare",
            "hnh_path_compare",
        )
    ]
    tutorial_exact = (
        result.get("byte_differences") == 0
        and result.get("exactness_status") == "byte_exact_device_diff"
    )
    chain8_exact = exact_device_body(w4a16_chain8)
    bridge_exact = (
        (w4a16_bridge or {}).get("accepted_bridge") is True
        and (
            (w4a16_bridge or {})
            .get("custom_vs_native_public_layout", {})
            .get("exact_after_transform")
            is True
        )
    )
    entries = [
        {
            "id": "m0_oracles",
            "status": "pass" if (Path("example/handwritten_hmx_matmul/oracles.json")).is_file() else "fail",
        },
        {
            "id": "m1_owned_runtime_boundary",
            "status": "pass" if (Path("example/handwritten_hmx_matmul").is_dir()) else "fail",
        },
        {
            "id": "w4a16_tutorial_direct_hmx_route",
            "status": "pass" if tutorial is not None and all(gates) else "open",
            "device_required": args.require_device,
            "gate_names": [
                "prepared_state_compare",
                "call_abi_compare",
                "vtcm_offset_compare",
                "step_trace_compare",
                "hnh_path_compare",
            ],
            "gate_matches": gates,
        },
        {
            "id": "w4a16_chain8_custom_baseline_exactness",
            "status": "pass" if chain8_exact else "open",
            "evidence": str(root / "device_body_w4a16_chain8_custom_baseline.json"),
            "byte_differences": (w4a16_chain8 or {}).get("result", {}).get("byte_differences"),
            "checksum": (w4a16_chain8 or {}).get("result", {}).get("output_checksum"),
        },
        {
            "id": "w4a16_native_transpose_bridge",
            "status": "pass" if bridge_exact else "open",
            "evidence": str(root / "w4a16_chain8_custom_baseline_native_bridge.json"),
            "required_transform": (
                (w4a16_bridge or {})
                .get("custom_vs_native_public_layout", {})
                .get("required_transform")
            ),
            "exact_after_transform": (
                (w4a16_bridge or {})
                .get("custom_vs_native_public_layout", {})
                .get("exact_after_transform")
            ),
        },
    ]
    payload = {
        "schema": "handwritten_hmx_matmul_roadmap_audit.v1",
        "route": "tutorial_direct_hmx_wrapper",
        "summary": {
            "pass": sum(1 for item in entries if item["status"] == "pass"),
            "open": sum(1 for item in entries if item["status"] == "open"),
            "fail": sum(1 for item in entries if item["status"] == "fail"),
            "w4a16_complete": chain8_exact and bridge_exact,
            "legacy_tutorial_exact": tutorial_exact,
        },
        "entries": entries,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.json_out}")
    return 0 if not any(item["status"] == "fail" for item in entries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
