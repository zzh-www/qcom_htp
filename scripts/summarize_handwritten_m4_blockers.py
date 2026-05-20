#!/usr/bin/env python3
"""Summarize current M4 completion state for handwritten HMX MatMul."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def result(data: dict[str, Any] | None) -> dict[str, Any]:
    value = (data or {}).get("result", {})
    return value if isinstance(value, dict) else {}


def exact_device_body(data: dict[str, Any] | None) -> bool:
    r = result(data)
    return (
        (data or {}).get("qnn_used") is False
        and (data or {}).get("pass") is True
        and r.get("device_execution") is True
        and r.get("hmx_body_entered") is True
        and r.get("exactness_status") == "byte_exact_device_diff"
        and r.get("byte_differences") == 0
        and r.get("output_checksum") == r.get("native_raw_checksum")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()

    root = args.artifact_root.resolve()
    tutorial = load_json(root / "w4a16_qnn_kernel_tutorial" / "device_result.json")
    chain8 = load_json(root / "device_body_w4a16_chain8_custom_baseline.json")
    bridge = load_json(root / "w4a16_chain8_custom_baseline_native_bridge.json")

    tutorial_result = result(tutorial)
    chain8_result = result(chain8)
    bridge_layout = (bridge or {}).get("custom_vs_native_public_layout", {})
    bridge_exact = (
        (bridge or {}).get("accepted_bridge") is True
        and bridge_layout.get("required_transform") == "native_transpose_2d"
        and bridge_layout.get("exact_after_transform") is True
        and bridge_layout.get("native_transpose_2d", {}).get("byte_differences") == 0
    )
    chain8_exact = exact_device_body(chain8)

    tutorial_gates = {
        key: (tutorial or {}).get(key, {}).get("match")
        for key in (
            "prepared_state_compare",
            "call_abi_compare",
            "vtcm_offset_compare",
            "step_trace_compare",
            "hnh_path_compare",
        )
    }
    accepted = chain8_exact and bridge_exact
    payload = {
        "schema": "handwritten_hmx_matmul_m4_blockers.v1",
        "route": "tutorial_direct_hmx_wrapper",
        "qnn_runtime_used": False,
        "w4a16": {
            "accepted": accepted,
            "blockers": []
            if accepted
            else ["w4a16_chain8_custom_baseline_or_native_transpose_bridge_not_exact"],
            "tutorial_route_gate_result": str(
                root / "w4a16_qnn_kernel_tutorial" / "device_result.json"
            ),
            "tutorial_route_gate_matches": tutorial_gates,
            "legacy_tutorial_exactness_status": tutorial_result.get("exactness_status"),
            "legacy_tutorial_byte_differences": tutorial_result.get("byte_differences"),
            "chain8_custom_baseline_device_result": str(
                root / "device_body_w4a16_chain8_custom_baseline.json"
            ),
            "chain8_custom_baseline_exact": chain8_exact,
            "chain8_custom_baseline_checksum": chain8_result.get("output_checksum"),
            "chain8_custom_baseline_byte_differences": chain8_result.get(
                "byte_differences"
            ),
            "native_transpose_bridge": str(
                root / "w4a16_chain8_custom_baseline_native_bridge.json"
            ),
            "native_transpose_bridge_exact": bridge_exact,
            "native_transpose_bridge_stats": bridge_layout.get("native_transpose_2d"),
        },
        "closed_routes": [
            "old_w4a16_qnn_blackbox_route",
            "k64_k256_native_boundary_retarget",
            "legacy_w4a16_residual_cvt_exploration",
        ],
        "current_gate": "tests/handwritten_hmx_matmul/run_all.sh",
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
