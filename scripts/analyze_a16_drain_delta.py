#!/usr/bin/env python3
"""Analyze A16 HMX drain deltas against native and analytic outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_FAMILIES = ("w8a16", "w4a16")


def _infer_family(out_dir: Path, requested: str | None = None) -> str:
    if requested:
        return requested
    name = out_dir.name.lower()
    for family in SUPPORTED_FAMILIES:
        if family in name:
            return family
    for path in out_dir.glob("*.out_ref_u16.npy"):
        for family in SUPPORTED_FAMILIES:
            if family in path.name.lower():
                return family
    raise ValueError(f"cannot infer family from {out_dir}; pass --family")


def _load_shape(out_dir: Path) -> tuple[int, int, int]:
    native_io = out_dir / "native_io.json"
    if native_io.exists():
        data = json.loads(native_io.read_text(encoding="utf-8"))
        shape = data.get("shape_mkn")
        if isinstance(shape, list) and len(shape) == 3:
            return int(shape[0]), int(shape[1]), int(shape[2])
    ref = np.load(sorted(out_dir.glob("*.out_ref_u16.npy"))[0])
    return int(ref.shape[0]), int(ref.shape[0]), int(ref.shape[1])


def _first(out_dir: Path, pattern: str) -> Path:
    matches = sorted(out_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"{out_dir} has no {pattern}")
    return matches[0]


def _default_native_out(out_dir: Path, family: str) -> Path | None:
    candidates = [
        out_dir.parent / f"output_{family}_native_ref_e2e_256" / "device_out" / "Y.raw",
        out_dir.parent / out_dir.name.replace("_aligned_", "_native_ref_") / "device_out" / "Y.raw",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_u16_raw(path: Path, shape: tuple[int, int], transpose: bool = False) -> np.ndarray:
    data = np.fromfile(path, dtype="<u2")
    if data.size != shape[0] * shape[1]:
        raise ValueError(f"{path} has {data.size} u16 values, expected {shape[0] * shape[1]}")
    arr = data.reshape(tuple(reversed(shape)) if transpose else shape)
    return arr.T.copy() if transpose else arr


def _load_native_best_layout(path: Path | None, shape: tuple[int, int], custom: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    if path is None or not path.exists():
        return None, None
    direct = _load_u16_raw(path, shape, transpose=False)
    transposed = _load_u16_raw(path, shape, transpose=True)
    direct_exact = int((direct == custom).sum())
    transposed_exact = int((transposed == custom).sum())
    if transposed_exact > direct_exact:
        return transposed, "transpose"
    return direct, "direct"


def _pair_stats(lhs: np.ndarray, rhs: np.ndarray, tolerances: tuple[int, ...]) -> dict[str, Any]:
    diff = lhs.astype(np.int64) - rhs.astype(np.int64)
    absdiff = np.abs(diff)
    stats: dict[str, Any] = {
        "exact": int((diff == 0).sum()),
        "total": int(diff.size),
        "max_absdiff": int(absdiff.max()),
        "mean_absdiff": float(absdiff.mean()),
        "signed_min": int(diff.min()),
        "signed_max": int(diff.max()),
    }
    for tol in tolerances:
        stats[f"abs_le_{tol}"] = int((absdiff <= tol).sum())
    return stats


def _group_counts(mask: np.ndarray, block: int, axis: int) -> list[int]:
    size = mask.shape[axis]
    groups = size // block
    out = []
    for i in range(groups):
        if axis == 0:
            out.append(int(mask[i * block : (i + 1) * block, :].sum()))
        else:
            out.append(int(mask[:, i * block : (i + 1) * block].sum()))
    return out


def _delta_distribution(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    diff = lhs.astype(np.int64) - rhs.astype(np.int64)
    absdiff = np.abs(diff)
    values, counts = np.unique(diff, return_counts=True)
    top = np.argsort(counts)[-12:][::-1]
    nonzero = diff != 0
    return {
        "top_signed_deltas": [
            {"delta": int(values[i]), "count": int(counts[i])}
            for i in top
        ],
        "nonzero_count": int(nonzero.sum()),
        "nonzero_by_row32": _group_counts(nonzero, 32, axis=0),
        "nonzero_by_n32": _group_counts(nonzero, 32, axis=1),
        "absdiff_by_bucket": {
            "0": int((absdiff == 0).sum()),
            "1": int((absdiff == 1).sum()),
            "2_3": int(((absdiff >= 2) & (absdiff <= 3)).sum()),
            "4_15": int(((absdiff >= 4) & (absdiff <= 15)).sum()),
            "16_255": int(((absdiff >= 16) & (absdiff <= 255)).sum()),
            "256_plus": int((absdiff >= 256).sum()),
        },
    }


def _saturation_cross(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    lhs_zero = lhs == 0
    lhs_sat = lhs == 65535
    rhs_zero = rhs == 0
    rhs_sat = rhs == 65535
    return {
        "lhs_zero_rhs_not_zero": int((lhs_zero & ~rhs_zero).sum()),
        "lhs_sat_rhs_not_sat": int((lhs_sat & ~rhs_sat).sum()),
        "rhs_zero_lhs_not_zero": int((rhs_zero & ~lhs_zero).sum()),
        "rhs_sat_lhs_not_sat": int((rhs_sat & ~lhs_sat).sum()),
        "both_zero": int((lhs_zero & rhs_zero).sum()),
        "both_sat": int((lhs_sat & rhs_sat).sum()),
    }


def analyze(out_dir: Path, family: str, native_out: Path | None) -> dict[str, Any]:
    m, _k, n = _load_shape(out_dir)
    shape = (m, n)
    analytic_path = _first(out_dir, "*.out_ref_u16.npy")
    custom_path = out_dir / "device_out" / "out.raw"
    if native_out is None:
        native_out = _default_native_out(out_dir, family)

    analytic = np.load(analytic_path).astype(np.uint16).reshape(shape)
    custom = _load_u16_raw(custom_path, shape)
    native, native_layout = _load_native_best_layout(native_out, shape, custom)

    report: dict[str, Any] = {
        "family": family,
        "out_dir": str(out_dir),
        "shape_mn": list(shape),
        "inputs": {
            "custom_output": str(custom_path),
            "analytic_output": str(analytic_path),
            "native_output": str(native_out) if native_out is not None and native_out.exists() else None,
            "native_layout": native_layout,
        },
        "custom_vs_analytic": _pair_stats(custom, analytic, (1, 2, 3, 7, 15, 255)),
        "custom_minus_analytic": _delta_distribution(custom, analytic),
        "custom_analytic_saturation_cross": _saturation_cross(custom, analytic),
    }
    if native is not None:
        report["custom_vs_native"] = _pair_stats(custom, native, (1, 2, 3, 7, 15, 255))
        report["native_vs_analytic"] = _pair_stats(native, analytic, (1, 2, 3, 7, 15, 255))
        report["native_minus_analytic"] = _delta_distribution(native, analytic)
        report["native_analytic_saturation_cross"] = _saturation_cross(native, analytic)
    return report


def print_report(report: dict[str, Any]) -> None:
    print(f"=== {report['family'].upper()} A16 drain delta ===")
    print(f"out_dir: {report['out_dir']}")
    print(f"shape_mn: {report['shape_mn']}")
    for name in ("custom_vs_native", "custom_vs_analytic", "native_vs_analytic"):
        stats = report.get(name)
        if not stats:
            continue
        print(
            f"{name}: exact={stats['exact']}/{stats['total']} "
            f"max={stats['max_absdiff']} mean={stats['mean_absdiff']:.3f} "
            f"abs<=3={stats['abs_le_3']}"
        )
    dist = report.get("native_minus_analytic") or report["custom_minus_analytic"]
    print("top signed deltas:")
    for item in dist["top_signed_deltas"][:8]:
        print(f"  {item['delta']}: {item['count']}")
    print(f"nonzero_by_n32: {dist['nonzero_by_n32']}")
    print(f"nonzero_by_row32: {dist['nonzero_by_row32']}")
    print(f"absdiff buckets: {dist['absdiff_by_bucket']}")
    sat = report.get("native_analytic_saturation_cross") or report["custom_analytic_saturation_cross"]
    print(f"saturation cross: {sat}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--family", choices=SUPPORTED_FAMILIES, default=None)
    parser.add_argument("--native-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    family = _infer_family(out_dir, args.family)
    native_out = args.native_out.resolve() if args.native_out is not None else None
    report = analyze(out_dir, family, native_out)
    print_report(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
