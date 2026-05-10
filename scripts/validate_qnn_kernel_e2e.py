#!/usr/bin/env python3
"""Validate a canonical QNN HMX MatMul kernel E2E artifact pair.

This is the cheap CI gate after a device run has produced artifacts.  It checks
the repo-standard artifact shape, compares custom output against the native
reference bytes, and verifies that optrace recorded the expected custom kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from check_qnn_artifact_standard import check_artifact


def _read_bytes(path: Path) -> bytes:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty file: {path}")
    return path.read_bytes()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty json: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _check_standard(path: Path, label: str) -> list[str]:
    errors, warnings = check_artifact(
        path,
        require_native_io=True,
        require_layout_flags=True,
        reject_float_io=True,
    )
    for warning in warnings:
        print(f"  [warn] {label}: {warning}")
    return [f"{label}: {error}" for error in errors]


def _validate_optrace(
    custom_dir: Path,
    expected_htp_type: str,
    qnn_prefix: str,
    expected_qnn_ops: int,
) -> list[str]:
    errors: list[str] = []
    summary = _load_json(custom_dir / "optrace" / "summary.json")
    by_type = summary.get("by_htp_type_cycles", {})
    cycles = int(by_type.get(expected_htp_type, 0) or 0)
    if cycles <= 0:
        errors.append(f"custom optrace missing positive cycles for {expected_htp_type}")

    events = summary.get("events", [])
    custom_events = [event for event in events if event.get("htp_type") == expected_htp_type]
    if not custom_events:
        errors.append(f"custom optrace has no event for {expected_htp_type}")
    if any("uses_hmx" not in event.get("flags", []) for event in custom_events if "flags" in event):
        errors.append(f"custom optrace event for {expected_htp_type} is missing uses_hmx flag")

    qnn_names = {
        name
        for name in summary.get("by_qnn_op_cycles", {})
        if isinstance(name, str) and name.startswith(qnn_prefix)
    }
    if len(qnn_names) != expected_qnn_ops:
        errors.append(
            f"expected {expected_qnn_ops} qnn ops with prefix {qnn_prefix}, got {len(qnn_names)}"
        )

    totals = summary.get("totals", {})
    if int(totals.get("timeline_span_cycles", 0) or 0) <= 0:
        errors.append("custom optrace has no positive timeline span")
    return errors


def _validate_w16_profile(custom_dir: Path) -> list[str]:
    profile = _load_json(custom_dir / "w16a16_run_profile.json")
    expected = {
        "kernel_profile": "accepted",
        "acceptance_scope": "canonical_256_native_oracle",
        "boundary_policy": "single_custom_op_internal_split_n128",
    }
    errors = []
    for key, value in expected.items():
        if profile.get(key) != value:
            errors.append(f"w16a16 profile {key}={profile.get(key)!r}, expected {value!r}")

    compare_path = custom_dir / "analysis" / "w16a16_custom_compare.json"
    if compare_path.exists():
        compare = _load_json(compare_path)
        gate = compare.get("alignment_gate", {})
        checks = gate.get("checks", {})
        stable_checks = (
            "native_raw_exact",
            "default_acceptance_profile",
            "same_or_justified_boundary",
            "native_packet_budget",
            "diagnostic_profile_only",
        )
        for name in stable_checks:
            check = checks.get(name, {})
            if check.get("pass") is not True:
                errors.append(f"w16a16 analysis {name} did not pass")
        cycle_check = checks.get("native_cycle_budget", {})
        if cycle_check.get("pass") is not True:
            print(
                "  [warn] w16a16 native_cycle_budget did not pass: "
                f"{cycle_check.get('evidence', 'no evidence')}"
            )
    return errors


def _validate_lpbq_profile(custom_dir: Path) -> list[str]:
    errors: list[str] = []
    overrides = _load_json(custom_dir / "quant_overrides.json")
    if overrides.get("version") != "1.0.0":
        errors.append(f"LPBQ quant_overrides version={overrides.get('version')!r}, expected '1.0.0'")

    params = overrides.get("param_encodings", [])
    if not isinstance(params, list):
        errors.append("LPBQ quant_overrides param_encodings is not a v1 list")
        return errors

    weight = next((enc for enc in params if isinstance(enc, dict) and enc.get("name") == "weight"), None)
    if weight is None:
        errors.append("LPBQ quant_overrides missing weight encoding")
        return errors

    expected = {
        "enc_type": "LPBQ",
        "dtype": "INT",
        "bw": 8,
        "compressed_bw": 4,
        "block_size": 32,
        "is_sym": True,
    }
    for key, value in expected.items():
        if weight.get(key) != value:
            errors.append(f"LPBQ weight {key}={weight.get(key)!r}, expected {value!r}")

    scales = weight.get("scale")
    per_block = weight.get("per_block_int_scale")
    if not isinstance(scales, list) or not scales:
        errors.append("LPBQ weight scale list is missing or empty")
    if not isinstance(per_block, list) or not per_block:
        errors.append("LPBQ weight per_block_int_scale list is missing or empty")
    elif isinstance(scales, list) and scales and len(per_block) != len(scales):
        errors.append(
            "LPBQ weight per_block_int_scale channel count does not match scale count: "
            f"{len(per_block)} != {len(scales)}"
        )
    return errors


def _dtype(name: str) -> np.dtype:
    if name == "uint8":
        return np.dtype("<u1")
    if name == "uint16":
        return np.dtype("<u2")
    raise ValueError(f"unsupported dtype: {name}")


def _compare_output(
    custom_path: Path,
    native_path: Path,
    dtype_name: str,
    native_transpose_2d: bool,
) -> str | None:
    custom = _read_bytes(custom_path)
    native = _read_bytes(native_path)
    if not native_transpose_2d:
        if custom == native:
            print(f"  output exact: {len(custom)} bytes, sha256={_sha256(custom)}")
            return None
        return (
            "custom/native output mismatch: "
            f"custom={len(custom)}B sha256={_sha256(custom)} "
            f"native={len(native)}B sha256={_sha256(native)}"
        )

    dtype = _dtype(dtype_name)
    if len(custom) != len(native):
        return f"custom/native output size mismatch: custom={len(custom)}B native={len(native)}B"
    items = len(custom) // dtype.itemsize
    side = int(items**0.5)
    if side * side != items:
        return f"cannot transpose non-square native output with {items} elements"
    custom_arr = np.frombuffer(custom, dtype=dtype).reshape(side, side)
    native_arr = np.frombuffer(native, dtype=dtype).reshape(side, side).T
    if np.array_equal(custom_arr, native_arr):
        print(f"  output exact after native transpose: {items} elements, sha256={_sha256(custom)}")
        return None
    diff = np.abs(custom_arr.astype(np.int64) - native_arr.astype(np.int64))
    return (
        "custom/native output mismatch after native transpose: "
        f"custom={len(custom)}B sha256={_sha256(custom)} "
        f"native={len(native)}B sha256={_sha256(native)} "
        f"maxdiff={int(diff.max())}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--custom-dir", required=True, type=Path)
    parser.add_argument("--native-dir", required=True, type=Path)
    parser.add_argument("--custom-raw", default="device_out/out.raw")
    parser.add_argument("--native-raw", default="device_out/Y.raw")
    parser.add_argument("--expected-htp-type", required=True)
    parser.add_argument("--qnn-prefix", required=True)
    parser.add_argument("--expected-qnn-ops", required=True, type=int)
    parser.add_argument("--dtype", choices=("uint8", "uint16"), default="uint8")
    parser.add_argument("--native-transpose-2d", action="store_true")
    parser.add_argument("--w16-accepted", action="store_true")
    parser.add_argument("--expect-lpbq", action="store_true")
    args = parser.parse_args()

    custom_dir = args.custom_dir.resolve()
    native_dir = args.native_dir.resolve()
    errors: list[str] = []

    errors.extend(_check_standard(custom_dir, "custom"))
    errors.extend(_check_standard(native_dir, "native"))

    try:
        output_error = _compare_output(
            custom_dir / args.custom_raw,
            native_dir / args.native_raw,
            args.dtype,
            args.native_transpose_2d,
        )
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if output_error:
            errors.append(output_error)

    errors.extend(
        _validate_optrace(
            custom_dir,
            args.expected_htp_type,
            args.qnn_prefix,
            args.expected_qnn_ops,
        )
    )
    if args.w16_accepted:
        errors.extend(_validate_w16_profile(custom_dir))
    if args.expect_lpbq:
        errors.extend(_validate_lpbq_profile(custom_dir))

    if errors:
        print(f"FAIL: {args.kernel} E2E validation")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"PASS: {args.kernel} E2E validation ({custom_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
