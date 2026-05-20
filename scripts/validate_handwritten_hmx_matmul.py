#!/usr/bin/env python3
"""Validate the current handwritten HMX MatMul tutorial/direct-HMX route."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("u8i8", "w4a8", "w8a16", "w4a16")


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def executable(path: Path) -> bool:
    return path.is_file() and bool(path.stat().st_mode & 0o111)


def validate_static_files(errors: list[str]) -> None:
    required = [
        "example/handwritten_hmx_matmul/oracles.json",
        "example/handwritten_hmx_matmul/shape_matrix.json",
        "example/handwritten_hmx_matmul/profile_matrix.json",
        "example/handwritten_hmx_matmul/tutorial_w4a16_qnn_kernel/README.md",
        "scripts/build_w4a16_qnn_kernel_tutorial.py",
        "scripts/run_w4a16_qnn_kernel_tutorial_device.py",
        "scripts/run_handwritten_artifact_body_device.py",
        "scripts/run_handwritten_artifact_body_sim.py",
        "scripts/check_w4a16_tutorial_chain1_sources.py",
        "scripts/prepare_w4a16_small_shape_direct_hmx_artifact.py",
        "scripts/summarize_w4a16_custom_baseline_native_bridge.py",
    ]
    for item in required:
        path = ROOT / item
        if not path.is_file():
            errors.append(f"missing required route file: {item}")

    tests = [
        "tests/handwritten_hmx_matmul/run_all.sh",
        "tests/handwritten_hmx_matmul/test_w4a16_qnn_kernel_tutorial_wrapper.sh",
        "tests/handwritten_hmx_matmul/test_w4a16_run_all_matrix_wiring.sh",
    ]
    for item in tests:
        path = ROOT / item
        if not executable(path):
            errors.append(f"missing or non-executable route test: {item}")


def validate_no_old_route_wiring(errors: list[str]) -> None:
    run_all = ROOT / "tests/handwritten_hmx_matmul/run_all.sh"
    if not run_all.is_file():
        return
    text = run_all.read_text(encoding="utf-8")
    for marker in (
        "test_w4a16_qnn_kernel_tutorial_wrapper.sh",
        "prepare_w4a16_small_shape_direct_hmx_artifact.py",
        "device_body_w4a16_chain8_custom_baseline.json",
        "w4a16_chain8_custom_baseline_native_bridge.json",
        "--reference-raw-override",
        "--native-transpose-2d",
    ):
        if marker not in text:
            errors.append(f"run_all.sh missing direct-HMX marker: {marker}")
    for marker in (
        "run_w4a16_descdump_payload_matrix.py",
        "host_ctxgen_compile_exec",
        "context_record_state_before_bias_control_load",
        "wrapper_payload_window_probe",
        "selector_mutation_plan",
    ):
        if marker in text:
            errors.append(f"run_all.sh still wires old W4A16 route: {marker}")
    tutorial_build = ROOT / "example/handwritten_hmx_matmul/tutorial_w4a16_qnn_kernel/build.sh"
    if tutorial_build.is_file():
        build_text = tutorial_build.read_text(encoding="utf-8")
        for marker in ("DESCRIPTOR_CARRIER", "--descriptor-carrier"):
            if marker in build_text:
                errors.append(f"tutorial build still exposes old descriptor carrier route: {marker}")


def validate_body_json(path: Path, schema: str, errors: list[str]) -> None:
    try:
        data = load_json(path)
    except ValueError as exc:
        errors.append(str(exc))
        return
    if data.get("schema") != schema:
        errors.append(f"unexpected schema in {path}: {data.get('schema')!r}")
    if data.get("qnn_used") is not False:
        errors.append(f"{path} must record qnn_used=false")


def validate_artifacts(root: Path, require_device: bool, errors: list[str]) -> None:
    validate_body_json(root / "body_check.json", "handwritten_hmx_body_check.v1", errors)
    validate_body_json(root / "body_entry_sim.json", "handwritten_hmx_body_entry_sim.v1", errors)
    for family in FAMILIES:
        path = root / family / "owned_run.json"
        if not path.is_file():
            errors.append(f"missing owned smoke artifact for {family}: {path}")
    for family in FAMILIES:
        path = root / f"artifact_body_{family}.json"
        if not path.is_file():
            errors.append(f"missing artifact-body sim result for {family}: {path}")
            continue
        try:
            data = load_json(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if data.get("qnn_used") is not False:
            errors.append(f"artifact body must be QNN-free for {family}")
    if not require_device:
        return
    for family in ("u8i8", "w4a8", "w8a16"):
        path = root / f"device_body_{family}.json"
        if not path.is_file():
            errors.append(f"missing device body evidence for {family}: {path}")
    path = root / "device_body_w4a16_chain8_custom_baseline.json"
    if not path.is_file():
        errors.append(f"missing W4A16 chain8 custom-baseline device evidence: {path}")


def validate_w4a16_chain8_custom_baseline(root: Path, require_device: bool, errors: list[str]) -> None:
    bridge_path = root / "w4a16_chain8_custom_baseline_native_bridge.json"
    if not require_device and not bridge_path.is_file():
        return
    try:
        bridge = load_json(bridge_path)
    except ValueError as exc:
        errors.append(str(exc))
        return
    layout = bridge.get("custom_vs_native_public_layout", {})
    if bridge.get("accepted_bridge") is not True:
        errors.append("W4A16 custom/native bridge is not accepted")
    if layout.get("required_transform") != "native_transpose_2d":
        errors.append("W4A16 custom/native bridge must use native_transpose_2d")
    if layout.get("exact_after_transform") is not True:
        errors.append("W4A16 custom/native bridge is not exact after native_transpose_2d")
    if layout.get("native_transpose_2d", {}).get("byte_differences") != 0:
        errors.append("W4A16 custom/native bridge has transpose byte differences")
    if not require_device:
        return
    device_path = root / "device_body_w4a16_chain8_custom_baseline.json"
    try:
        device = load_json(device_path)
    except ValueError as exc:
        errors.append(str(exc))
        return
    result = device.get("result", {})
    if device.get("qnn_used") is not False:
        errors.append("W4A16 chain8 custom-baseline device body must record qnn_used=false")
    if device.get("pass") is not True:
        errors.append("W4A16 chain8 custom-baseline device body did not pass")
    if result.get("exactness_status") != "byte_exact_device_diff":
        errors.append("W4A16 chain8 custom-baseline device body is not byte-exact")
    if result.get("byte_differences") != 0:
        errors.append("W4A16 chain8 custom-baseline byte differences are nonzero")
    if result.get("output_checksum") != result.get("native_raw_checksum"):
        errors.append("W4A16 chain8 custom-baseline checksum mismatch")


def validate_tutorial_device(root: Path, require_device: bool, errors: list[str]) -> None:
    if not require_device:
        return
    path = root / "w4a16_qnn_kernel_tutorial" / "device_result.json"
    try:
        data = load_json(path)
    except ValueError as exc:
        errors.append(str(exc))
        return
    if data.get("qnn_runtime_used") is not False:
        errors.append("tutorial device result must record qnn_runtime_used=false")
    for key in (
        "prepared_state_compare",
        "call_abi_compare",
        "vtcm_offset_compare",
        "step_trace_compare",
        "hnh_path_compare",
    ):
        if data.get(key, {}).get("match") is not True:
            errors.append(f"tutorial gate mismatch: {key}")
    result = data.get("result", {})
    if result.get("entered_and_returned") is not True:
        errors.append("tutorial direct HMX body did not enter and return")


def validate_completion_checklist(root: Path, errors: list[str]) -> None:
    audit_path = root / "roadmap_audit.json"
    checklist_path = root / "completion_checklist.json"
    if not audit_path.is_file():
        return
    try:
        audit = load_json(audit_path)
    except ValueError as exc:
        errors.append(str(exc))
        return
    if audit.get("schema") != "handwritten_hmx_matmul_roadmap_audit.v1":
        return
    try:
        checklist = load_json(checklist_path)
    except ValueError as exc:
        errors.append(str(exc))
        return
    if checklist.get("schema") != "handwritten_hmx_matmul_completion_checklist.v1":
        errors.append(f"unexpected schema in {checklist_path}: {checklist.get('schema')!r}")
        return
    if checklist.get("route") != "tutorial_direct_hmx_wrapper":
        errors.append(f"completion checklist must use tutorial route: {checklist.get('route')!r}")
    evidence = []
    for item in checklist.get("criteria", []):
        if isinstance(item, dict):
            evidence.extend(str(path) for path in item.get("evidence", []))
    if str(root / "roadmap_audit.json") not in evidence:
        errors.append("completion checklist must include roadmap_audit.json evidence")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--body-check-json", type=Path)
    parser.add_argument("--body-entry-json", type=Path)
    parser.add_argument("--require-device", action="store_true")
    args = parser.parse_args()

    root = args.artifact_root.resolve()
    errors: list[str] = []
    validate_static_files(errors)
    validate_no_old_route_wiring(errors)
    validate_artifacts(root, args.require_device, errors)
    validate_tutorial_device(root, args.require_device, errors)
    validate_w4a16_chain8_custom_baseline(root, args.require_device, errors)
    validate_completion_checklist(root, errors)

    if errors:
        print("handwritten HMX MatMul gate: FAILED", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    mode = "device" if args.require_device else "artifact"
    print(f"handwritten HMX MatMul gate: ok ({mode} tutorial/direct-HMX route, {root})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
