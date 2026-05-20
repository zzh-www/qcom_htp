#!/usr/bin/env python3
"""Run the W4A16 tutorial/direct-HMX wrapper on device and write evidence JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_handwritten_artifact_body_device import parse_device_log, rel, run_on_device


DEFAULT_BUILD_DIR = (
    Path(__file__).resolve().parents[1]
    / "example"
    / "handwritten_hmx_matmul"
    / "tutorial_w4a16_qnn_kernel"
    / "build"
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compare_prepared_checksums(manifest: dict, result: dict) -> dict:
    expected = manifest.get("expected_prepared_state_checksums")
    actual = result.get("prepared_state_checksums")
    if not isinstance(expected, dict) or not expected:
        return {
            "checked": False,
            "match": False,
            "reason": "missing_expected_prepared_state_checksums",
            "mismatches": [],
        }
    if not isinstance(actual, dict) or not actual:
        return {
            "checked": True,
            "match": False,
            "reason": "missing_device_prepared_state_checksums",
            "mismatches": sorted(expected),
        }
    mismatches = [
        key
        for key, value in sorted(expected.items())
        if actual.get(key) != value
    ]
    return {
        "checked": True,
        "match": not mismatches,
        "reason": None if not mismatches else "checksum_mismatch",
        "mismatches": mismatches,
        "expected": expected,
        "actual": actual,
    }


def compare_call_abi_scalars(manifest: dict, result: dict) -> dict:
    expected = manifest.get("expected_call_abi_scalars")
    actual = result.get("call_abi_scalars")
    if not isinstance(expected, dict) or not expected:
        return {
            "checked": False,
            "match": False,
            "reason": "missing_expected_call_abi_scalars",
            "mismatches": [],
        }
    if not isinstance(actual, dict) or not actual:
        return {
            "checked": True,
            "match": False,
            "reason": "missing_device_call_abi_scalars",
            "mismatches": sorted(expected),
        }
    mismatches = [
        key
        for key, value in sorted(expected.items())
        if actual.get(key) != value
    ]
    return {
        "checked": True,
        "match": not mismatches,
        "reason": None if not mismatches else "call_abi_scalar_mismatch",
        "mismatches": mismatches,
        "expected": expected,
        "actual": actual,
    }


def compare_vtcm_offsets(manifest: dict, result: dict) -> dict:
    expected = manifest.get("expected_vtcm_offsets")
    layout = result.get("pointer_layout")
    actual = layout.get("vtcm_delta_bytes") if isinstance(layout, dict) else None
    if not isinstance(expected, dict) or not expected:
        return {
            "checked": False,
            "match": False,
            "reason": "missing_expected_vtcm_offsets",
            "mismatches": [],
        }
    if not isinstance(actual, dict) or not actual:
        return {
            "checked": True,
            "match": False,
            "reason": "missing_device_pointer_layout",
            "mismatches": sorted(expected),
        }
    mismatches = [
        key
        for key, value in sorted(expected.items())
        if actual.get(key) != value
    ]
    return {
        "checked": True,
        "match": not mismatches,
        "reason": None if not mismatches else "vtcm_offset_mismatch",
        "mismatches": mismatches,
        "expected": expected,
        "actual": {key: actual.get(key) for key in sorted(expected)},
        "absolute_hex": layout.get("absolute_hex"),
    }


def compare_step_trace(manifest: dict, result: dict) -> dict:
    expected_steps = manifest.get("expected_chain_steps")
    step_trace = result.get("step_trace")
    if not manifest.get("step_trace_enabled"):
        return {
            "checked": False,
            "match": False,
            "reason": "step_trace_not_enabled",
            "mismatches": [],
        }
    if not isinstance(step_trace, list) or not step_trace:
        return {
            "checked": True,
            "match": False,
            "reason": "missing_device_step_trace",
            "mismatches": ["step_trace"],
        }
    mismatches = []
    if isinstance(expected_steps, int) and len(step_trace) != expected_steps:
        mismatches.append("step_count")
    expected_step0 = manifest.get("expected_step0_native_raw")
    if isinstance(expected_step0, dict):
        if not step_trace:
            mismatches.append("step0")
        else:
            first = step_trace[0]
            if first.get("step") != 0:
                mismatches.append("step0_index")
            if first.get("native_checksum") != expected_step0.get("checksum"):
                mismatches.append("step0_native_checksum")
            if "native_exact" not in first:
                mismatches.append("step0_native_compare_missing")
    return {
        "checked": True,
        "match": not mismatches,
        "reason": None if not mismatches else "step_trace_mismatch",
        "mismatches": mismatches,
        "expected_chain_steps": expected_steps,
        "observed_chain_steps": len(step_trace),
        "expected_step0_native_raw": expected_step0,
        "observed_step0": step_trace[0] if step_trace else None,
    }


def compare_hnh_path(manifest: dict, result: dict) -> dict:
    expected = manifest.get("expected_hnh_path")
    actual = result.get("hnh_path")
    if not isinstance(expected, dict) or not expected:
        return {
            "checked": False,
            "match": False,
            "reason": "missing_expected_hnh_path",
            "mismatches": [],
        }
    if not isinstance(actual, dict) or not actual:
        return {
            "checked": True,
            "match": False,
            "reason": "missing_device_hnh_path",
            "mismatches": sorted(expected),
        }
    mismatches = [
        key
        for key, value in sorted(expected.items())
        if actual.get(key) != value
    ]
    return {
        "checked": True,
        "match": not mismatches,
        "reason": None if not mismatches else "hnh_path_mismatch",
        "mismatches": mismatches,
        "expected": expected,
        "actual": {key: actual.get(key) for key in sorted(expected)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--device", default="oneplus")
    parser.add_argument("--remote-dir", default="w4a16_qnn_kernel_tutorial")
    parser.add_argument("--timeout-s", type=int, default=90)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    binary = build_dir / "libw4a16_qnn_kernel_tutorial.so"
    manifest_path = build_dir / "manifest.json"
    if not binary.is_file():
        parser.error(f"missing {binary}; run build.sh first")
    if not manifest_path.is_file():
        parser.error(f"missing {manifest_path}; run build.sh first")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    returncode, log = run_on_device(
        args.device,
        args.remote_dir,
        binary,
        args.timeout_s,
    )
    result = parse_device_log(log, "w4a16")
    prepared_compare = compare_prepared_checksums(manifest, result)
    call_abi_compare = compare_call_abi_scalars(manifest, result)
    vtcm_offset_compare = compare_vtcm_offsets(manifest, result)
    step_trace_compare = compare_step_trace(manifest, result)
    hnh_path_compare = compare_hnh_path(manifest, result)
    payload = {
        "schema": "w4a16_qnn_kernel_tutorial_device_run.v1",
        "device": args.device,
        "remote_dir": args.remote_dir,
        "binary": rel(binary),
        "build_manifest": rel(manifest_path),
        "qnn_runtime_used": False,
        "entry_style": "run_main_on_hexagon_hap_vtcm_hmx_lock",
        "kernel_body": manifest.get("kernel_body"),
        "kernel_entry": manifest.get("kernel_entry"),
        "prepared_state_carrier": manifest.get("prepared_state_carrier"),
        "pre_clear_acc": manifest.get("pre_clear_acc"),
        "native_wrapper_prefetch": manifest.get("native_wrapper_prefetch"),
        "preload_hmx_identity_bias": manifest.get("preload_hmx_identity_bias"),
        "output_seed_mode": manifest.get("output_seed_mode"),
        "activation_raw_override": manifest.get("activation_raw_override"),
        "packed_weight_byte_offset": manifest.get("packed_weight_byte_offset"),
        "packed_weight_raw_override": manifest.get("packed_weight_raw_override"),
        "folded_bias_byte_offset": manifest.get("folded_bias_byte_offset"),
        "folded_bias_raw_override": manifest.get("folded_bias_raw_override"),
        "extra_word_overrides": manifest.get("extra_word_overrides"),
        "mask_word_overrides": manifest.get("mask_word_overrides"),
        "measure_repeats": manifest.get("measure_repeats"),
        "chain_steps": manifest.get("chain_steps"),
        "final_native_oracle": manifest.get("final_native_oracle"),
        "run_main_returncode": returncode,
        "prepared_state_compare": prepared_compare,
        "call_abi_compare": call_abi_compare,
        "vtcm_offset_compare": vtcm_offset_compare,
        "step_trace_compare": step_trace_compare,
        "hnh_path_compare": hnh_path_compare,
        "result": result,
    }
    json_out = args.json_out or build_dir / "device_result.json"
    write_json(json_out.resolve(), payload)
    print(f"w4a16 tutorial wrapper device result: {json_out.resolve()}")
    print(
        "  entered={entered} prepared={prepared} prepared_match={prepared_match} callabi_match={callabi_match} vtcm_match={vtcm_match} step_trace_match={step_trace_match} hnh_path_match={hnh_path_match} exactness={exactness} diffs={diffs} checksum={checksum} native={native}".format(
            entered=int(result.get("entered_and_returned") is True),
            prepared=int(result.get("prepared_state_device_visible") is True),
            prepared_match=int(prepared_compare.get("match") is True),
            callabi_match=int(call_abi_compare.get("match") is True),
            vtcm_match=int(vtcm_offset_compare.get("match") is True),
            step_trace_match=int(step_trace_compare.get("match") is True),
            hnh_path_match=int(hnh_path_compare.get("match") is True),
            exactness=result.get("exactness_status"),
            diffs=result.get("byte_differences"),
            checksum=result.get("output_checksum"),
            native=result.get("native_raw_checksum"),
        )
    )
    return (
        0
        if (
            result.get("entered_and_returned") is True
            and prepared_compare.get("match") is True
            and call_abi_compare.get("match") is True
            and vtcm_offset_compare.get("match") is True
            and step_trace_compare.get("match") is True
            and hnh_path_compare.get("match") is True
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
