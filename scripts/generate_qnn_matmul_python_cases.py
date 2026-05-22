#!/usr/bin/env python3
"""Generate float-derived Python-reference QNN HMX MatMul cases.

Each case starts from torch-style float tensors:

    output_float = x_float @ weight_float.T + bias_float

Then it derives quantization parameters from the observed min/max range.
Activation/output use AIMET/QNN affine schemas with family-fixed offsets.
Weight schemas are generated as separate variants: per-output-channel and
per-K-group are not collapsed into one mixed schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Family:
    name: str
    act_bits: int
    weight_bits: int
    out_bits: int
    weight_schema: str = "per_output_channel"
    kernel_family: str | None = None

    @property
    def act_zp(self) -> int:
        return 1 << (self.act_bits - 1)

    @property
    def out_zp(self) -> int:
        return 32768 if self.out_bits == 16 else 0

    @property
    def act_qmax(self) -> int:
        return (1 << self.act_bits) - 1

    @property
    def out_qmax(self) -> int:
        return (1 << self.out_bits) - 1

    @property
    def weight_qmax(self) -> int:
        return (1 << (self.weight_bits - 1)) - 1

    @property
    def weight_group_size(self) -> int | None:
        return 64 if self.weight_schema == "per_group" else None

    @property
    def kernel_name(self) -> str:
        return self.kernel_family or self.name

    @property
    def act_dtype(self) -> np.dtype:
        return np.dtype("<u2") if self.act_bits == 16 else np.dtype("uint8")

    @property
    def out_dtype(self) -> np.dtype:
        return np.dtype("<u2") if self.out_bits == 16 else np.dtype("uint8")


FAMILIES = {
    "u8i8": Family("u8i8", act_bits=8, weight_bits=8, out_bits=8),
    "w4a8_per_channel": Family(
        "w4a8_per_channel",
        act_bits=8,
        weight_bits=4,
        out_bits=8,
        weight_schema="per_output_channel",
        kernel_family="w4a8",
    ),
    "w4a8_per_group": Family(
        "w4a8_per_group",
        act_bits=8,
        weight_bits=4,
        out_bits=8,
        weight_schema="per_group",
        kernel_family="w4a8",
    ),
    "w8a16": Family("w8a16", act_bits=16, weight_bits=8, out_bits=16),
    "w4a16_per_channel": Family(
        "w4a16_per_channel",
        act_bits=16,
        weight_bits=4,
        out_bits=16,
        weight_schema="per_output_channel",
        kernel_family="w4a16",
    ),
    "w4a16_per_group": Family(
        "w4a16_per_group",
        act_bits=16,
        weight_bits=4,
        out_bits=16,
        weight_schema="per_group",
        kernel_family="w4a16",
    ),
}


CASES = (
    "normal_random",
    "zp_neutral",
    "positive_boundary",
    "negative_boundary",
    "single_k_impulse",
    "bias_only",
    "scale_only",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(arr: np.ndarray) -> dict[str, Any]:
    values = np.unique(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()) if arr.dtype.kind == "f" else int(arr.min()),
        "max": float(arr.max()) if arr.dtype.kind == "f" else int(arr.max()),
        "unique_count": int(values.size),
        "first8": [
            float(v) if arr.dtype.kind == "f" else int(v)
            for v in arr.reshape(-1)[:8]
        ],
    }


def write_array(path: Path, arr: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), arr)
    arr.reshape(-1).tofile(path.with_suffix(".raw"))
    return {
        "npy": path.with_suffix(".npy").name,
        "raw": path.with_suffix(".raw").name,
        "raw_sha256": sha256_file(path.with_suffix(".raw")),
        **stats(arr),
    }


def affine_scale_from_minmax(min_v: float, max_v: float, qmin: int, qmax: int, zp: int) -> float:
    """Derive scale for fixed-zp affine quantization."""
    eps = 1.0e-12
    candidates = [eps]
    if min_v < 0 and zp > qmin:
        candidates.append((-min_v) / float(zp - qmin))
    if max_v > 0 and qmax > zp:
        candidates.append(max_v / float(qmax - zp))
    return max(candidates)


def symmetric_scale_from_minmax(min_v: float, max_v: float, qmax: int) -> float:
    return max(abs(min_v), abs(max_v), 1.0e-12) / float(qmax)


def quantize_affine_aimet_qnn(
    values: np.ndarray,
    scale: float,
    qnn_offset: int,
    qmin: int,
    qmax: int,
    dtype: np.dtype,
) -> np.ndarray:
    """AIMET affine quantize for a QNN encoding offset.

    AIMET uses round(x / scale) - offset and dequantizes as
    (q + offset) * scale.  QNN encodings store offset=-zero_point for the
    unsigned affine tensors used by these native Conv1x1 MatMul cases.
    """
    q = np.rint(values.astype(np.float64) / scale - qnn_offset)
    return np.clip(q, qmin, qmax).astype(dtype)


def dequantize_affine_aimet_qnn(values_q: np.ndarray, scale: float, qnn_offset: int) -> np.ndarray:
    return (values_q.astype(np.float64) + qnn_offset) * scale


def seed_for(family: str, case: str) -> int:
    seed = 0x5EED
    for ch in f"{family}:{case}":
        seed = ((seed * 131) + ord(ch)) & 0xFFFFFFFF
    return seed


def rng_for(family: str, case: str) -> np.random.Generator:
    return np.random.default_rng(seed_for(family, case))


def make_float_case(family: Family, case: str, m: int, k: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = rng_for(family.name, case)
    weight_width = 1.0 if family.weight_bits == 8 else 0.75
    act_width = 1.0 if family.act_bits == 16 else 0.85

    if case == "normal_random":
        x = rng.uniform(-act_width, act_width, size=(m, k)).astype(np.float32)
        w = rng.uniform(-weight_width, weight_width, size=(n, k)).astype(np.float32)
        bias = rng.uniform(-0.05, 0.05, size=n).astype(np.float32)
        return x, w, bias

    if case == "zp_neutral":
        x = np.zeros((m, k), dtype=np.float32)
        w = rng.uniform(-weight_width, weight_width, size=(n, k)).astype(np.float32)
        bias = np.zeros(n, dtype=np.float32)
        return x, w, bias

    if case == "positive_boundary":
        x = rng.uniform(0.75 * act_width, act_width, size=(m, k)).astype(np.float32)
        w = rng.uniform(0.75 * weight_width, weight_width, size=(n, k)).astype(np.float32)
        bias = rng.uniform(0.05, 0.15, size=n).astype(np.float32)
        return x, w, bias

    if case == "negative_boundary":
        x = rng.uniform(0.75 * act_width, act_width, size=(m, k)).astype(np.float32)
        w = rng.uniform(-weight_width, -0.75 * weight_width, size=(n, k)).astype(np.float32)
        bias = rng.uniform(-0.15, -0.05, size=n).astype(np.float32)
        return x, w, bias

    if case == "single_k_impulse":
        x = np.zeros((m, k), dtype=np.float32)
        x[:, 1 % k] = 0.25 * act_width
        w = np.zeros((n, k), dtype=np.float32)
        w[:, 1 % k] = rng.uniform(-weight_width, weight_width, size=n).astype(np.float32)
        bias = np.zeros(n, dtype=np.float32)
        return x, w, bias

    if case == "bias_only":
        x = rng.uniform(-0.1 * act_width, 0.1 * act_width, size=(m, k)).astype(np.float32)
        w = rng.uniform(-0.1 * weight_width, 0.1 * weight_width, size=(n, k)).astype(np.float32)
        bias = rng.uniform(-0.75, 0.75, size=n).astype(np.float32)
        return x, w, bias

    if case == "scale_only":
        x = rng.uniform(-3.0 * act_width, 3.0 * act_width, size=(m, k)).astype(np.float32)
        w = rng.uniform(-2.5 * weight_width, 2.5 * weight_width, size=(n, k)).astype(np.float32)
        bias = np.zeros(n, dtype=np.float32)
        return x, w, bias

    raise ValueError(f"unknown case: {case}")


def quantize_weight(
    family: Family,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Quantize weight[N,K] with the family weight schema."""
    n, k = w.shape
    qmax = family.weight_qmax
    q_dtype = np.int8 if qmax <= 127 else np.int16

    if family.weight_schema == "per_output_channel":
        scales = np.empty((n,), dtype=np.float32)
        q = np.empty((n, k), dtype=q_dtype)
        deq = np.empty((n, k), dtype=np.float32)
        for out_ch in range(n):
            row = w[out_ch, :]
            scale = symmetric_scale_from_minmax(float(row.min()), float(row.max()), qmax)
            scales[out_ch] = scale
            q_row = np.clip(np.rint(row.astype(np.float64) / scale), -qmax, qmax).astype(q_dtype)
            q[out_ch, :] = q_row
            deq[out_ch, :] = q_row.astype(np.float32) * scale
        qparams = {
            "schema": "signed_symmetric_per_output_channel",
            "axis": "N",
            "bitwidth": family.weight_bits,
            "scale_shape": list(scales.shape),
            "zero_point": 0,
            "signed_range": [-qmax, qmax],
            "observed_min": float(w.min()),
            "observed_max": float(w.max()),
        }
        return q, scales, deq, qparams

    group_size = family.weight_group_size
    if group_size is None:
        raise ValueError(f"{family.name}: missing W4 group size")
    if k <= group_size or k % group_size:
        raise ValueError(f"{family.name}: W4 K={k} must be > {group_size} and divisible by {group_size}")
    groups = k // group_size
    scales = np.empty((n, groups), dtype=np.float32)
    q = np.empty((n, k), dtype=q_dtype)
    deq = np.empty((n, k), dtype=np.float32)
    for out_ch in range(n):
        for group in range(groups):
            start = group * group_size
            stop = start + group_size
            block = w[out_ch, start:stop]
            scale = symmetric_scale_from_minmax(float(block.min()), float(block.max()), qmax)
            scales[out_ch, group] = scale
            q_block = np.clip(np.rint(block.astype(np.float64) / scale), -qmax, qmax).astype(q_dtype)
            q[out_ch, start:stop] = q_block
            deq[out_ch, start:stop] = q_block.astype(np.float32) * scale
    qparams = {
        "schema": "signed_symmetric_per_group",
        "axis": "K_group",
        "scale_storage_axes": ["output_row", "k_group"],
        "group_size": group_size,
        "bitwidth": family.weight_bits,
        "scale_shape": list(scales.shape),
        "zero_point": 0,
        "signed_range": [-qmax, qmax],
        "observed_min": float(w.min()),
        "observed_max": float(w.max()),
    }
    return q, scales, deq, qparams


def quantize_case(family: Family, x: np.ndarray, w: np.ndarray, bias: np.ndarray) -> dict[str, Any]:
    # Float torch oracle.
    output_float = x.astype(np.float64) @ w.astype(np.float64).T + bias.astype(np.float64)

    act_scale = affine_scale_from_minmax(float(x.min()), float(x.max()), 0, family.act_qmax, family.act_zp)
    output_scale = affine_scale_from_minmax(
        float(output_float.min()),
        float(output_float.max()),
        0,
        family.out_qmax,
        family.out_zp,
    )

    act_offset = -family.act_zp
    out_offset = -family.out_zp
    x_q = quantize_affine_aimet_qnn(x, act_scale, act_offset, 0, family.act_qmax, family.act_dtype)
    w_q, weight_scales, w_deq, weight_qparams = quantize_weight(family, w)
    if weight_scales.ndim == 1:
        bias_scale = act_scale * weight_scales.astype(np.float64)
        bias_q = np.rint(bias.astype(np.float64) / bias_scale)
        bias_q = np.clip(bias_q, -(1 << 31), (1 << 31) - 1).astype(np.int32)
        bias_deq = bias_q.astype(np.float64) * bias_scale
        bias_qparams = {
            "schema": "float_source_quantized_to_int32",
            "bitwidth": 32,
            "scale_shape": list(bias_scale.shape),
            "scale": bias_scale.astype(np.float64).tolist(),
            "zero_point": 0,
            "observed_min": float(bias.min()),
            "observed_max": float(bias.max()),
        }
    else:
        # Per-group bias handling depends on the exact lowered kernel contract.
        # Keep it explicit rather than mixing this with per-channel bias.
        bias_scale = None
        bias_q = np.zeros_like(bias, dtype=np.int32)
        bias_deq = bias.astype(np.float64)
        bias_qparams = {
            "schema": "float32_per_group_bias_not_modeled",
            "observed_min": float(bias.min()),
            "observed_max": float(bias.max()),
        }

    x_deq = dequantize_affine_aimet_qnn(x_q, act_scale, act_offset)
    quant_real = x_deq @ w_deq.astype(np.float64).T + bias_deq
    output_q = quantize_affine_aimet_qnn(
        quant_real,
        output_scale,
        out_offset,
        0,
        family.out_qmax,
        family.out_dtype,
    )

    return {
        "output_float": output_float.astype(np.float32),
        "output_quantized_path_float": quant_real.astype(np.float32),
        "activation_q": x_q,
        "weight_q_nk": w_q,
        "weight_q_kn": w_q.T.copy(),
        "weight_scale": weight_scales,
        "bias_q_int32": bias_q,
        "bias_quantized_path_float": bias_deq.astype(np.float32),
        "output_q": output_q,
        "qparams": {
            "activation": {
                "schema": "aimet_qnn_affine_fixed_offset",
                "bitwidth": family.act_bits,
                "scale": act_scale,
                "zero_point": family.act_zp,
                "qnn_offset": act_offset,
                "quantize_formula": "round(x / scale) - qnn_offset, then clamp",
                "dequantize_formula": "(q + qnn_offset) * scale",
                "observed_min": float(x.min()),
                "observed_max": float(x.max()),
            },
            "weight": weight_qparams,
            "bias": bias_qparams,
            "output": {
                "schema": "aimet_qnn_affine_fixed_offset",
                "bitwidth": family.out_bits,
                "scale": output_scale,
                "zero_point": family.out_zp,
                "qnn_offset": out_offset,
                "quantize_formula": "round(x / scale) - qnn_offset, then clamp",
                "dequantize_formula": "(q + qnn_offset) * scale",
                "observed_min": float(output_float.min()),
                "observed_max": float(output_float.max()),
            },
        },
    }


def output_counts(out: np.ndarray, family: Family) -> dict[str, int]:
    return {
        "zero": int((out == 0).sum()),
        "max": int((out == family.out_qmax).sum()),
        "zp": int((out == family.out_zp).sum()),
        "total": int(out.size),
    }


def generate_case(root: Path, family: Family, case: str, m: int, k: int, n: int) -> dict[str, Any]:
    out_dir = root / family.name / case
    out_dir.mkdir(parents=True, exist_ok=True)

    x, w, bias = make_float_case(family, case, m, k, n)
    q = quantize_case(family, x, w, bias)

    files = {
        "activation_float": write_array(out_dir / "activation_float", x.astype(np.float32)),
        "weight_float_nk": write_array(out_dir / "weight_float_nk", w.astype(np.float32)),
        "bias_float": write_array(out_dir / "bias_float", bias.astype(np.float32)),
        "bias_q_int32": write_array(out_dir / "bias_q_int32", q["bias_q_int32"]),
        "bias_quantized_path_float": write_array(
            out_dir / "bias_quantized_path_float",
            q["bias_quantized_path_float"],
        ),
        "output_float": write_array(out_dir / "output_float", q["output_float"]),
        "output_quantized_path_float": write_array(
            out_dir / "output_quantized_path_float",
            q["output_quantized_path_float"],
        ),
        "activation_q": write_array(out_dir / "activation_q", q["activation_q"]),
        "weight_scale": write_array(out_dir / "weight_scale", q["weight_scale"]),
        "weight_q_nk": write_array(out_dir / "weight_q_nk", q["weight_q_nk"]),
        "weight_q_kn": write_array(out_dir / "weight_q_kn", q["weight_q_kn"]),
        "output_ref_q": write_array(out_dir / "output_ref_q", q["output_q"]),
    }

    payload = {
        "schema": "qnn_hmx_matmul_python_case.v2",
        "family": family.name,
        "kernel_family": family.kernel_name,
        "weight_schema_variant": family.weight_schema,
        "case": case,
        "seed": seed_for(family.name, case),
        "shape_mkn": [m, k, n],
        "torch_formula": "output_float = activation_float @ weight_float_nk.T + bias_float",
        "qparams": q["qparams"],
        "reference_formula": (
            "output_ref_q = quantize_affine_aimet_qnn(((activation_q+qnn_offset_A)*act_scale) @ "
            "dequantize_weight(weight_q_nk, weight_scale).T + "
            "dequantize_bias(bias_q_int32, act_scale*weight_scale), output_scale, qnn_offset_Out)"
        ),
        "quantization_reference": (
            "AIMET torch_builtins affine quantize/dequantize semantics: "
            "quantize=round(x/scale)-offset; dequantize=(q+offset)*scale. "
            "QNN encoding offset is -zero_point for these affine tensors."
        ),
        "w4a16_note": (
            "Small W4+A16 deltas are accepted only for QNN Native HTP vs "
            "Python/AIMET-style oracle comparison. Same-hardware comparisons "
            "such as custom op vs native op and handwritten vs custom op remain exact-output gates."
        ),
        "files": files,
        "output_counts": output_counts(q["output_q"], family),
    }
    (out_dir / "case.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "family": family.name,
        "kernel_family": family.kernel_name,
        "weight_schema_variant": family.weight_schema,
        "case": case,
        "path": str(out_dir),
        "output_counts": payload["output_counts"],
        "qparams": payload["qparams"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("tests/qnn_kernel_e2e/correctness/python_cases"))
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--families", nargs="+", default=sorted(FAMILIES))
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    if args.m % 32 or args.k % 32 or args.n % 32:
        raise SystemExit("M/K/N must be multiples of 32")

    selected_families = []
    for family_name in args.families:
        if family_name not in FAMILIES:
            raise SystemExit(f"unknown family: {family_name}")
        selected_families.append(FAMILIES[family_name])
    if any(f.weight_bits == 4 for f in selected_families) and (args.k <= 64 or args.k % 64):
        raise SystemExit("W4 channel dimension K must be > 64 and divisible by 64")

    out_root = args.out_root.resolve()
    if out_root.exists() and not args.no_clean:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    entries = []
    for family in selected_families:
        for case in args.cases:
            if case not in CASES:
                raise SystemExit(f"unknown case: {case}")
            entries.append(generate_case(out_root, family, case, args.m, args.k, args.n))

    manifest = {
        "schema": "qnn_hmx_matmul_python_case_manifest.v2",
        "shape_mkn": [args.m, args.k, args.n],
        "families": args.families,
        "cases": args.cases,
        "zero_point_policy": {
            "u8i8/w4a8": "activation fixed zp=128; output fixed zp=0",
            "w8a16/w4a16": "activation/output fixed native A16 zp=32768",
            "weights": "signed symmetric zp=0; W4 cases include separate per-output-channel and per-group variants",
        },
        "entries": entries,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {len(entries)} python correctness cases to {out_root}")
    for entry in entries:
        counts = entry["output_counts"]
        qparams = entry["qparams"]
        print(
            f"  {entry['family']}:{entry['case']} "
            f"sA={qparams['activation']['scale']:.6g} "
            f"sW_shape={qparams['weight']['scale_shape']} "
            f"sO={qparams['output']['scale']:.6g} "
            f"zero={counts['zero']} max={counts['max']} zp={counts['zp']} total={counts['total']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
