#!/usr/bin/env python3
"""Summarize the W4A16 direct-body custom-baseline to native-layout bridge.

The native/public bridge intentionally uses the same 2D transpose rule as the
existing W4A16 custom/native Python tooling:

- scripts/analyze_w4a16_native_run.py --native-transpose
- scripts/validate_qnn_kernel_e2e.py --native-transpose-2d
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_w4a16_native_run import _load_quantized_raw


DEFAULT_DIRECT = Path("/tmp/w4a16_256_custombaseline_deep_probe.json")
DEFAULT_CUSTOM = Path("example/qnn_matmul_profile/output_w4a16_aligned_e2e_256")
DEFAULT_NATIVE = Path("example/qnn_matmul_profile/output_w4a16_native_ref_e2e_256")
DEFAULT_OUT = Path("/tmp/w4a16_custom_baseline_native_bridge_summary.json")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pair_stats(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    if lhs.shape != rhs.shape:
        return {
            "shape_match": False,
            "lhs_shape": list(lhs.shape),
            "rhs_shape": list(rhs.shape),
        }
    diff = lhs.astype(np.int32) - rhs.astype(np.int32)
    exact = int((diff == 0).sum())
    return {
        "shape_match": True,
        "exact": exact,
        "total": int(lhs.size),
        "max_absdiff": int(np.max(np.abs(diff))) if lhs.size else 0,
        "byte_differences": int(
            (lhs.reshape(-1).view(np.uint8) != rhs.reshape(-1).view(np.uint8)).sum()
        ),
    }


def read_u16_matrix(path: Path, shape: tuple[int, int], *, transpose: bool = False) -> np.ndarray:
    return _load_quantized_raw(
        path,
        np.dtype("<u2"),
        shape,
        scale=1.0,
        offset=0.0,
        transpose_2d=transpose,
    )


def summarize(direct: Path, custom_dir: Path, native_dir: Path) -> dict[str, Any]:
    direct_json = load_json(direct)
    result = direct_json.get("result", {})

    custom_io = load_json(custom_dir / "native_io.json")
    shape_mkn = custom_io.get("shape_mkn")
    if not shape_mkn:
        raise ValueError(f"{custom_dir}/native_io.json missing shape_mkn")
    m, _k, n = [int(v) for v in shape_mkn]
    shape = (m, n)

    custom_raw = custom_dir / "device_out" / "out.raw"
    native_raw = native_dir / "device_out" / "Y.raw"
    custom = read_u16_matrix(custom_raw, shape)
    native_direct = read_u16_matrix(native_raw, shape)
    native_transposed = read_u16_matrix(native_raw, shape, transpose=True)

    custom_vs_native_direct = pair_stats(custom, native_direct)
    custom_vs_native_transposed = pair_stats(custom, native_transposed)
    direct_vs_custom_exact = (
        result.get("exactness_status") == "byte_exact_device_diff"
        and int(result.get("byte_differences", -1)) == 0
        and result.get("native_raw_checksum") == result.get("output_checksum")
    )
    native_transpose_exact = (
        custom_vs_native_transposed.get("exact") == custom_vs_native_transposed.get("total")
        and custom_vs_native_transposed.get("max_absdiff") == 0
    )
    return {
        "schema": "w4a16_custom_baseline_native_bridge.v1",
        "direct_result": str(direct),
        "custom_artifact": str(custom_dir),
        "native_artifact": str(native_dir),
        "shape_mkn": shape_mkn,
        "chain": int(custom_io.get("chain", 0)),
        "existing_python_alignment_rule": {
            "required_transform": "native_transpose_2d",
            "implementation_sources": [
                "scripts/analyze_w4a16_native_run.py --native-transpose",
                "scripts/validate_qnn_kernel_e2e.py --native-transpose-2d",
            ],
        },
        "direct_vs_custom": {
            "exact": direct_vs_custom_exact,
            "exactness_status": result.get("exactness_status"),
            "byte_differences": result.get("byte_differences"),
            "checksum": result.get("output_checksum"),
            "custom_baseline_checksum": result.get("native_raw_checksum"),
            "kernel_entry": result.get("kernel_entry"),
            "public_output_layout": result.get("public_output_layout"),
            "hnh_path": result.get("hnh_path"),
        },
        "custom_vs_native_public_layout": {
            "required_transform": "native_transpose_2d",
            "direct": custom_vs_native_direct,
            "native_transpose_2d": custom_vs_native_transposed,
            "exact_after_transform": native_transpose_exact,
        },
        "accepted_bridge": bool(direct_vs_custom_exact and native_transpose_exact),
        "next_attempt": (
            "promote_w4a16_direct_body_custom_exact_with_native_transpose_2d_acceptance"
            if direct_vs_custom_exact and native_transpose_exact
            else "debug_w4a16_custom_baseline_native_bridge"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--custom-dir", type=Path, default=DEFAULT_CUSTOM)
    parser.add_argument("--native-dir", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    report = summarize(args.direct, args.custom_dir, args.native_dir)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"accepted_bridge={report['accepted_bridge']} "
        f"next={report['next_attempt']} json={args.json_out}"
    )
    return 0 if report["accepted_bridge"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
