#!/usr/bin/env python3
"""Dump observable u8i8 bias-prepare stage inputs and outputs.

This script is intentionally diagnostic.  It aligns all known artifacts by
output channel so the HTP prepare path can be checked stage by stage:

    source float B -> quantized DLC B -> dequantized float B
      -> find_bias_scale candidate -> final bias_to_vtcm effective int32

The final sidecar value is extracted from a generated QNN Native context binary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_u8i8_native_bias_record import (
    ACT_ZP,
    counts,
    expected_record,
    extract_dlc_bias,
    find_native_record,
    load_case,
    parse_record,
)
from scripts.qnn_htp_bias_prepare import qnn_htp_u8i8_prepare_bias_stages


def write_raw_npy(out_dir: Path, name: str, array: np.ndarray) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}.npy", array)
    array.reshape(-1).tofile(out_dir / f"{name}.raw")
    return {
        "npy": f"{name}.npy",
        "raw": f"{name}.raw",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()) if array.dtype.kind == "f" else int(array.min()),
        "max": float(array.max()) if array.dtype.kind == "f" else int(array.max()),
        "first8": [
            float(v) if array.dtype.kind == "f" else int(v)
            for v in array.reshape(-1)[:8]
        ],
    }


def round_away_from_zero(values: np.ndarray) -> np.ndarray:
    vals = values.astype(np.float32)
    return np.trunc(vals + np.copysign(np.float32(0.5), vals)).astype(np.int64)


def best_aligned_window(candidate: np.ndarray, target: np.ndarray) -> dict[str, int | None]:
    if candidate.size < target.size:
        return {"match": 0, "offset": None}
    best_match = -1
    best_offset = 0
    for offset in range(candidate.size - target.size + 1):
        match = int((candidate[offset : offset + target.size] == target).sum())
        if match > best_match:
            best_match = match
            best_offset = offset
    return {"match": best_match, "offset": best_offset}


def scan_gdb_int32_consts(
    gdb_dir: Path,
    dlc_bias_q: np.ndarray,
    native_sidecar_bias_q: np.ndarray,
) -> dict[str, Any]:
    files = sorted(gdb_dir.glob("gen_const_i32*.raw")) + sorted(
        gdb_dir.glob("gen_const_int32_common_*.raw")
    )
    notable = []
    best_dlc = {"file": None, "match": 0, "offset": None}
    best_native = {"file": None, "match": 0, "offset": None}
    for path in files:
        values = np.fromfile(path, dtype=np.int32)
        if values.size == 0:
            continue
        dlc_hit = best_aligned_window(values, dlc_bias_q)
        native_hit = best_aligned_window(values, native_sidecar_bias_q)
        if dlc_hit["match"] > best_dlc["match"]:
            best_dlc = {"file": path.name, **dlc_hit}
        if native_hit["match"] > best_native["match"]:
            best_native = {"file": path.name, **native_hit}
        if (
            values.size in (25, 100, 256, 512)
            or dlc_hit["match"] >= dlc_bias_q.size // 2
            or native_hit["match"] >= native_sidecar_bias_q.size // 2
        ):
            notable.append(
                {
                    "file": path.name,
                    "int32_count": int(values.size),
                    "min": int(values.min()),
                    "max": int(values.max()),
                    "best_dlc_bias_match": dlc_hit,
                    "best_native_sidecar_bias_match": native_hit,
                    "first8": [int(v) for v in values[:8]],
                }
            )
    return {
        "files_scanned": len(files),
        "target_int32_count": int(dlc_bias_q.size),
        "best_dlc_bias_match": best_dlc,
        "best_native_sidecar_bias_match": best_native,
        "notable_files": notable,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--native-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--gdb-dump-dir",
        type=Path,
        help="Optional output from scripts/gdb_dump_u8i8_bias_prepare.py for direct comparison.",
    )
    parser.add_argument(
        "--qnn-sdk-root",
        type=Path,
        default=Path(os.environ.get("QNN_SDK_ROOT", "tools/qnn-sdk")),
    )
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    native_dir = args.native_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    qnn_py = args.qnn_sdk_root.resolve() / "lib/python"
    if str(qnn_py) not in sys.path:
        sys.path.insert(0, str(qnn_py))

    meta, expected, _, _, exp_eff = expected_record(case_dir)
    _, weight_q_nk, generated_bias_q, weight_scale, bias_scale, shape = load_case(case_dir)
    m, k, n = [int(v) for v in shape]
    if meta["family"] != "u8i8":
        raise ValueError(f"expected u8i8, got {meta['family']!r}")

    source_bias_float = np.load(case_dir / meta["files"]["bias_float"]["npy"]).astype(np.float32)
    quant_path_bias_float = np.load(case_dir / meta["files"]["bias_quantized_path_float"]["npy"]).astype(np.float32)
    dlc_bias_q = extract_dlc_bias(native_dir / "case.dlc").astype(np.int32)

    offset, native_record, candidates = find_native_record(native_dir / "ctx/case_native_ctx.bin", expected)
    exp_scale, exp_control, _ = parse_record(expected)
    nat_scale, nat_control, nat_eff = parse_record(native_record)
    folded_i32 = (-ACT_ZP) * weight_q_nk.T.astype(np.int32).sum(axis=0).astype(np.int32)
    native_sidecar_bias_q = (nat_eff - folded_i32).astype(np.int32)

    # Per-channel bias_scale is the conventional accumulator bias scale, but the
    # recovered dequantize_bias function scans the scale tensor with fmaxf and
    # uses act_scale * max(weight_scale) for the float const it emits.
    act_scale = np.float32(meta["qparams"]["activation"]["scale"])
    global_bias_scale = (act_scale * np.max(weight_scale.astype(np.float32))).astype(np.float32)
    dequant_from_dlc_per_channel_f32 = (
        dlc_bias_q.astype(np.float32) * bias_scale.astype(np.float32)
    ).astype(np.float32)
    dequant_from_dlc_global_max_f32 = (dlc_bias_q.astype(np.float32) * global_bias_scale).astype(np.float32)
    dequant_from_generated_per_channel_f32 = (
        generated_bias_q.astype(np.float32) * bias_scale.astype(np.float32)
    ).astype(np.float32)
    prepare_stages = qnn_htp_u8i8_prepare_bias_stages(dlc_bias_q, act_scale, weight_scale)
    max_abs_dequant = np.max(np.abs(prepare_stages.dequantized_bias_f32)).astype(np.float32)
    find_bias_scale_f32 = prepare_stages.find_bias_scale_f32
    requant_bias_pre_nearby_f32 = prepare_stages.requant_bias_pre_nearby_f32
    requant_bias_expanded_i32 = prepare_stages.requant_bias_expanded_i32
    bias_scale_shuff_restore_f32 = prepare_stages.bias_scale_shuff_restore_f32
    bias_scale_shuff_trunc_i32 = prepare_stages.final_bias_i32

    arrays: dict[str, dict[str, Any]] = {}
    for name, array in {
        "source_bias_float": source_bias_float,
        "quant_path_bias_float": quant_path_bias_float,
        "generated_bias_q_int32": generated_bias_q,
        "dlc_bias_q_int32": dlc_bias_q,
        "bias_scale_float32": bias_scale.astype(np.float32),
        "weight_scale_float32": weight_scale.astype(np.float32),
        "dequantize_bias_out_per_channel_assumption_float32": dequant_from_dlc_per_channel_f32,
        "dequantize_bias_out_global_max_scale_float32": dequant_from_dlc_global_max_f32,
        "dequantize_bias_out_from_generated_per_channel_float32": dequant_from_generated_per_channel_f32,
        "find_bias_scale_out_float32": np.array([find_bias_scale_f32], dtype=np.float32),
        "requant_bias_pre_nearby_float32": requant_bias_pre_nearby_f32,
        "requant_bias_expanded_i32": requant_bias_expanded_i32,
        "bias_scale_shuff_restore_float32": bias_scale_shuff_restore_f32,
        "bias_scale_shuff_trunc_i32": bias_scale_shuff_trunc_i32,
        "folded_activation_zp_i32": folded_i32,
        "expected_effective_i32": exp_eff,
        "native_sidecar_effective_i32": nat_eff,
        "native_sidecar_bias_q_int32": native_sidecar_bias_q,
        "native_sidecar_scale_u16": nat_scale,
        "native_sidecar_control_u16": nat_control,
        "generated_sidecar_scale_u16": exp_scale,
        "generated_sidecar_control_u16": exp_control,
    }.items():
        arrays[name] = write_raw_npy(out_dir, name, np.asarray(array))

    nonzero = np.flatnonzero(native_sidecar_bias_q != dlc_bias_q)
    csv_path = out_dir / "channel_dump.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "channel",
                "source_bias_float",
                "quant_path_bias_float",
                "bias_scale",
                "generated_bias_q",
                "dlc_bias_q",
                "dequant_per_channel_assumption_f32",
                "dequant_global_max_scale_f32",
                "native_sidecar_bias_q",
                "native_minus_dlc",
                "native_effective_i32",
                "expected_effective_i32",
                "effective_delta",
                "requant_bias_pre_nearby_f32",
                "requant_bias_expanded_i32",
                "bias_scale_shuff_restore_f32",
                "bias_scale_shuff_trunc_i32",
            ],
        )
        writer.writeheader()
        interesting = list(nonzero[:128])
        if not interesting:
            interesting = list(range(min(n, 32)))
        for idx in interesting:
            writer.writerow(
                {
                    "channel": int(idx),
                    "source_bias_float": float(source_bias_float[idx]),
                    "quant_path_bias_float": float(quant_path_bias_float[idx]),
                    "bias_scale": float(bias_scale[idx]),
                    "generated_bias_q": int(generated_bias_q[idx]),
                    "dlc_bias_q": int(dlc_bias_q[idx]),
                    "dequant_per_channel_assumption_f32": float(dequant_from_dlc_per_channel_f32[idx]),
                    "dequant_global_max_scale_f32": float(dequant_from_dlc_global_max_f32[idx]),
                    "native_sidecar_bias_q": int(native_sidecar_bias_q[idx]),
                    "native_minus_dlc": int(native_sidecar_bias_q[idx] - dlc_bias_q[idx]),
                    "native_effective_i32": int(nat_eff[idx]),
                    "expected_effective_i32": int(exp_eff[idx]),
                    "effective_delta": int(nat_eff[idx] - exp_eff[idx]),
                    "requant_bias_pre_nearby_f32": float(requant_bias_pre_nearby_f32[idx]),
                    "requant_bias_expanded_i32": int(requant_bias_expanded_i32[idx]),
                    "bias_scale_shuff_restore_f32": float(bias_scale_shuff_restore_f32[idx]),
                    "bias_scale_shuff_trunc_i32": int(bias_scale_shuff_trunc_i32[idx]),
                }
            )

    gdb_compare = None
    if args.gdb_dump_dir:
        gdb_dir = args.gdb_dump_dir.resolve()
        gdb_dequant = np.fromfile(gdb_dir / "dequantize_bias_out_float32.raw", dtype=np.float32)
        if gdb_dequant.shape != dequant_from_dlc_global_max_f32.shape:
            raise ValueError(
                f"{gdb_dir}: gdb dequant shape {gdb_dequant.shape} "
                f"!= expected {dequant_from_dlc_global_max_f32.shape}"
            )
        gdb_compare = {
            "gdb_dump_dir": str(gdb_dir),
            "dequant_global_max_exact_words": int(
                (gdb_dequant.view(np.uint32) == dequant_from_dlc_global_max_f32.view(np.uint32)).sum()
            ),
            "dequant_total_words": int(gdb_dequant.size),
            "dequant_global_max_absmax": float(
                np.max(np.abs(gdb_dequant - dequant_from_dlc_global_max_f32))
            ),
            "dequant_per_channel_absmax": float(
                np.max(np.abs(gdb_dequant - dequant_from_dlc_per_channel_f32))
            ),
            "int32_const_probe": scan_gdb_int32_consts(gdb_dir, dlc_bias_q, native_sidecar_bias_q),
        }

    summary = {
        "schema": "qcom_htp.u8i8_bias_prepare_stage_dump.v1",
        "case": f"{meta['family']}/{meta['case']}",
        "shape_mkn": [m, k, n],
        "case_dir": str(case_dir),
        "native_dir": str(native_dir),
        "native_record_offset": int(offset),
        "top_candidates": [
            {
                "score": int(score),
                "offset": int(off),
                "scale_match": int(scale_match),
                "control_match": int(control_match),
                "effective_close": int(eff_close),
            }
            for score, off, scale_match, control_match, eff_close in candidates
        ],
        "stage_interpretation": {
            "source_bias_float": "ONNX Conv B initializer",
            "dlc_bias_q_int32": "quantized DLC Conv B tensor",
            "dequantize_bias_out_per_channel_assumption_float32": (
                "DLC B dequantized with per-channel bias_scale; this is not the recovered HTP rule"
            ),
            "dequantize_bias_out_global_max_scale_float32": (
                "DLC B dequantized with act_scale * max(weight_scale); matches gdb dequantize_bias output"
            ),
            "find_bias_scale_out_float32": "max(abs(dequantized_bias)) * 16 / 2^32 candidate",
            "requant_bias_expanded_i32": "nearbyintf(dequantized_bias / find_bias_scale)",
            "bias_scale_shuff_trunc_i32": (
                "trunc(float32(float32(requant_bias_expanded_i32 * find_bias_scale) / global_bias_scale))"
            ),
            "native_sidecar_bias_q_int32": "final context sidecar effective_i32 minus folded activation zp term",
        },
        "find_bias_scale_out_float32": float(find_bias_scale_f32),
        "max_abs_dequantize_bias_out": float(max_abs_dequant),
        "act_scale": float(act_scale),
        "max_weight_scale": float(np.max(weight_scale.astype(np.float32))),
        "global_bias_scale_used_by_dequantize_bias": float(global_bias_scale),
        "dlc_B_vs_generated_bias_q": counts(dlc_bias_q - generated_bias_q),
        "source_bias_vs_dequant_global_max_absmax": float(
            np.max(np.abs(source_bias_float - dequant_from_dlc_global_max_f32))
        ),
        "quant_path_bias_vs_dequant_global_max_absmax": float(
            np.max(np.abs(quant_path_bias_float - dequant_from_dlc_global_max_f32))
        ),
        "per_channel_dequant_vs_global_max_dequant_absmax": float(
            np.max(np.abs(dequant_from_dlc_per_channel_f32 - dequant_from_dlc_global_max_f32))
        ),
        "gdb_compare": gdb_compare,
        "native_sidecar_bias_q_vs_DLC_B": counts(native_sidecar_bias_q - dlc_bias_q),
        "bias_scale_shuff_trunc_vs_native_sidecar_bias_q": counts(
            bias_scale_shuff_trunc_i32 - native_sidecar_bias_q
        ),
        "native_effective_vs_expected_effective": counts(nat_eff - exp_eff),
        "scale_match": f"{int((nat_scale == exp_scale).sum())}/{exp_scale.size}",
        "control_match": f"{int((nat_control == exp_control).sum())}/{exp_control.size}",
        "nonzero_native_minus_dlc_count": int(nonzero.size),
        "nonzero_native_minus_dlc_preview": [
            {
                "channel": int(i),
                "source_bias_float": float(source_bias_float[i]),
                "dlc_bias_q": int(dlc_bias_q[i]),
                "dequant_per_channel_assumption_f32": float(dequant_from_dlc_per_channel_f32[i]),
                "dequant_global_max_scale_f32": float(dequant_from_dlc_global_max_f32[i]),
                "native_sidecar_bias_q": int(native_sidecar_bias_q[i]),
                "native_minus_dlc": int(native_sidecar_bias_q[i] - dlc_bias_q[i]),
                "bias_scale": float(bias_scale[i]),
            }
            for i in nonzero[:32]
        ],
        "arrays": arrays,
        "channel_csv": csv_path.name,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {out_dir / 'summary.json'}")
    print(f"wrote {csv_path}")
    print(f"DLC B vs generated bias_q: {summary['dlc_B_vs_generated_bias_q']}")
    print(
        "source bias vs dequant(global max) absmax: "
        f"{summary['source_bias_vs_dequant_global_max_absmax']:.9g}"
    )
    if gdb_compare is not None:
        print(
            "gdb dequant vs global-max model: "
            f"{gdb_compare['dequant_global_max_exact_words']}/"
            f"{gdb_compare['dequant_total_words']} words, "
            f"maxabs={gdb_compare['dequant_global_max_absmax']:.9g}"
        )
        int32_probe = gdb_compare["int32_const_probe"]
        print(
            "gdb int32 const best matches: "
            f"DLC={int32_probe['best_dlc_bias_match']}, "
            f"native={int32_probe['best_native_sidecar_bias_match']}"
        )
    print(f"find_bias_scale_out_float32: {summary['find_bias_scale_out_float32']:.9g}")
    print(f"native sidecar bias_q vs DLC B: {summary['native_sidecar_bias_q_vs_DLC_B']}")
    print(
        "bias_scale_shuff_trunc vs native sidecar bias_q: "
        f"{summary['bias_scale_shuff_trunc_vs_native_sidecar_bias_q']}"
    )
    print(f"native effective vs expected: {summary['native_effective_vs_expected_effective']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
