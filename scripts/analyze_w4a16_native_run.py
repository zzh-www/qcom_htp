#!/usr/bin/env python3
"""Analyze a W4A16 custom run against the native QNN oracle and optrace.

The standard runner writes raw device output and decoded optrace artifacts under
one OUT_DIR.  This helper turns those files into a durable comparison report so
failed probes are still useful and comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_output_encoding(out_dir: Path) -> tuple[float, float]:
    scale = 1.0
    offset = 0.0
    overrides = out_dir / "quant_overrides.json"
    if not overrides.exists():
        return scale, offset
    with overrides.open("r", encoding="utf-8") as f:
        encodings = json.load(f).get("activation_encodings", {})
    if "out" in encodings and encodings["out"]:
        enc = encodings["out"][0]
        scale = float(enc.get("scale", scale))
        offset = float(enc.get("offset", offset))
    return scale, offset


def _find_reference(out_dir: Path) -> Path | None:
    refs = sorted(out_dir.glob("*.out_ref_u*.npy"))
    return refs[0] if refs else None


def _load_quantized_raw(
    path: Path,
    dtype: np.dtype,
    shape: tuple[int, ...],
    scale: float,
    offset: float,
    transpose_2d: bool = False,
) -> np.ndarray:
    q_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    f_bytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
    read_shape = tuple(reversed(shape)) if transpose_2d and len(shape) == 2 else shape
    raw_size = path.stat().st_size
    if raw_size == q_bytes:
        arr = np.fromfile(path, dtype=dtype).reshape(read_shape)
    elif raw_size == f_bytes:
        if scale == 0.0:
            raise ValueError("cannot quantize float output with zero scale")
        max_value = np.iinfo(dtype).max
        arr_f = np.fromfile(path, dtype=np.float32).reshape(read_shape)
        arr = np.clip(np.rint(arr_f / scale - offset), 0, max_value).astype(dtype)
    else:
        raise ValueError(f"{path} has unexpected byte count {raw_size}")
    if transpose_2d and len(shape) == 2:
        arr = arr.T.copy()
    return arr


def _pair_stats(lhs: np.ndarray, rhs: np.ndarray, abs_tols: tuple[int, ...]) -> dict[str, Any]:
    lhs_i = lhs.astype(np.int64)
    rhs_i = rhs.astype(np.int64)
    diff = np.abs(lhs_i - rhs_i)
    stats: dict[str, Any] = {
        "exact": int((lhs == rhs).sum()),
        "total": int(lhs.size),
        "maxdiff": int(diff.max()) if diff.size else 0,
        "mean_absdiff": float(diff.mean()) if diff.size else 0.0,
    }
    for tol in abs_tols:
        stats[f"abs_le_{tol}"] = int((diff <= tol).sum())
    return stats


def _distribution(arr: np.ndarray) -> dict[str, Any]:
    info = np.iinfo(arr.dtype)
    unique, counts = np.unique(arr, return_counts=True)
    top_idx = np.argsort(counts)[-8:][::-1]
    return {
        "zeros": int((arr == 0).sum()),
        "saturated": int((arr == info.max).sum()),
        "unique": int(unique.size),
        "top_values": [
            {"value": int(unique[i]), "count": int(counts[i])}
            for i in top_idx
        ],
    }


def _spatial_stats(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, Any]:
    if lhs.ndim != 2 or rhs.ndim != 2:
        return {}
    exact = lhs == rhs
    row_exact = exact.sum(axis=1).astype(int)
    col_exact = exact.sum(axis=0).astype(int)
    rows, cols = lhs.shape
    row4 = rows // 4
    n32 = cols // 32
    row4_exact = [
        int(exact[g * 4 : (g + 1) * 4, :].sum())
        for g in range(row4)
    ]
    n32_exact = [
        int(exact[:, g * 32 : (g + 1) * 32].sum())
        for g in range(n32)
    ]
    tile_row4_n32 = [
        [
            int(exact[rg * 4 : (rg + 1) * 4, ng * 32 : (ng + 1) * 32].sum())
            for ng in range(n32)
        ]
        for rg in range(row4)
    ]
    best_rows = np.argsort(row_exact)[-8:][::-1]
    best_row4 = np.argsort(np.asarray(row4_exact))[-8:][::-1] if row4_exact else []
    return {
        "row_exact_counts": row_exact.tolist(),
        "col_exact_counts": col_exact.tolist(),
        "row4_group_exact_counts": row4_exact,
        "n32_group_exact_counts": n32_exact,
        "tile_row4_n32_exact_counts": tile_row4_n32,
        "best_rows": [
            {"row": int(i), "exact": int(row_exact[i])}
            for i in best_rows
        ],
        "best_row4_groups": [
            {"row4_group": int(i), "exact": int(row4_exact[int(i)])}
            for i in best_row4
        ],
    }


def _load_optrace_summary(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / "optrace" / "summary.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _summarize_optrace(summary: dict[str, Any], main_substr: str) -> dict[str, Any]:
    events = summary.get("events", [])
    matched = [
        event for event in events
        if main_substr in str(event.get("htp_type", "")) or main_substr in str(event.get("qnn_op", ""))
    ]
    top_htp = list(summary.get("by_htp_type_cycles", {}).items())[:12]
    return {
        "totals": summary.get("totals", {}),
        "main_event_count": len(matched),
        "main_event_cycles_sum": int(sum(int(e.get("dur", 0)) for e in matched)),
        "main_events": matched,
        "top_htp_type_cycles": [
            {"name": name, "cycles": int(cycles)} for name, cycles in top_htp
        ],
    }


def _native_perf_summary(native_out_dir: Path | None) -> dict[str, Any] | None:
    if native_out_dir is None:
        return None
    summary = _load_optrace_summary(native_out_dir)
    if summary is None:
        return None
    by_htp = summary.get("by_htp_type_cycles", {})
    by_qnn = summary.get("by_qnn_op_cycles", {})
    return {
        "out_dir": str(native_out_dir),
        "kernel_convlayer_s1_cycles": int(by_htp.get("q::ConvLayer_s1.opt", 0)),
        "weights_to_vtcm_cycles": int(by_htp.get("q::ConvLayer.opt.weights_to_vtcm", 0)),
        "bias_to_vtcm_cycles": int(by_htp.get("q::ConvLayer.opt.bias_to_vtcm", 0)),
        "conv1x1_qnn_op_cycles": int(by_qnn.get("conv1x1", 0)),
        "totals": summary.get("totals", {}),
    }


def _default_native_out_dir(native_raw: Path | None) -> Path | None:
    if native_raw is None:
        return None
    if native_raw.parent.name == "device_out":
        return native_raw.parent.parent
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--native-raw", type=Path)
    parser.add_argument("--native-transpose", action="store_true")
    parser.add_argument("--native-out-dir", type=Path)
    parser.add_argument("--main-op-substr", default="HmxU16I4ToU16MatMul")
    parser.add_argument("--abs-tols", default="1,3")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_raw = out_dir / "device_out" / "out.raw"
    ref_path = _find_reference(out_dir)
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "out_dir": str(out_dir),
        "output_raw": str(out_raw),
        "reference": str(ref_path) if ref_path else None,
    }
    try:
        if not out_raw.exists():
            raise FileNotFoundError(out_raw)
        if ref_path is None:
            raise FileNotFoundError("no *.out_ref_u*.npy reference found")
        ref = np.load(ref_path)
        if ref.ndim > 2:
            ref = ref.reshape(ref.shape[-2], ref.shape[-1])
        scale, offset = _load_output_encoding(out_dir)
        abs_tols = tuple(int(v) for v in args.abs_tols.split(",") if v.strip())
        out_q = _load_quantized_raw(out_raw, ref.dtype, ref.shape, scale, offset)
        report["output_distribution"] = _distribution(out_q)
        report["analytic"] = {
            "stats": _pair_stats(out_q, ref, abs_tols),
            "spatial": _spatial_stats(out_q, ref),
        }
        if args.native_raw:
            native_raw = args.native_raw.resolve()
            native_q = _load_quantized_raw(
                native_raw,
                ref.dtype,
                ref.shape,
                scale,
                offset,
                transpose_2d=args.native_transpose,
            )
            report["native_raw"] = str(native_raw)
            report["native_transpose"] = bool(args.native_transpose)
            report["native_distribution"] = _distribution(native_q)
            report["native"] = {
                "stats": _pair_stats(out_q, native_q, abs_tols),
                "spatial": _spatial_stats(out_q, native_q),
            }
    except Exception as exc:  # Keep optrace summaries useful even on bad output.
        report["error"] = str(exc)

    custom_summary = _load_optrace_summary(out_dir)
    if custom_summary is not None:
        report["custom_optrace"] = _summarize_optrace(custom_summary, args.main_op_substr)
    native_out_dir = args.native_out_dir.resolve() if args.native_out_dir else _default_native_out_dir(args.native_raw)
    native_perf = _native_perf_summary(native_out_dir)
    if native_perf is not None:
        report["native_optrace"] = native_perf

    json_path = analysis_dir / "w4a16_native_compare.json"
    txt_path = analysis_dir / "w4a16_native_compare.txt"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [f"W4A16 analysis: {out_dir}"]
    native_stats = report.get("native", {}).get("stats")
    if native_stats:
        lines.append(
            "native-exact: "
            f"{native_stats['exact']}/{native_stats['total']} "
            f"maxdiff={native_stats['maxdiff']}"
        )
    analytic_stats = report.get("analytic", {}).get("stats")
    if analytic_stats:
        lines.append(
            "analytic-exact: "
            f"{analytic_stats['exact']}/{analytic_stats['total']} "
            f"maxdiff={analytic_stats['maxdiff']}"
        )
    custom_perf = report.get("custom_optrace", {})
    if custom_perf:
        totals = custom_perf.get("totals", {})
        lines.append(
            "custom-optrace: "
            f"main={custom_perf.get('main_event_cycles_sum', 0)} "
            f"timeline={totals.get('timeline_span_cycles', 0)} "
            f"sum_pid0={totals.get('sum_pid0_event_cycles', 0)}"
        )
    native_perf = report.get("native_optrace", {})
    if native_perf:
        lines.append(
            "native-optrace: "
            f"kernel={native_perf.get('kernel_convlayer_s1_cycles', 0)} "
            f"conv1x1_qnn={native_perf.get('conv1x1_qnn_op_cycles', 0)} "
            f"timeline={native_perf.get('totals', {}).get('timeline_span_cycles', 0)}"
        )
    if "error" in report:
        lines.append(f"error: {report['error']}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  analysis: {txt_path}")
    for line in lines[1:]:
        print(f"    {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
