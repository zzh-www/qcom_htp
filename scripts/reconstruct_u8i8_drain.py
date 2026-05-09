#!/usr/bin/env python3
"""Reconstruct the U8I8 accumulator-to-drain path from retained artifacts.

This is a mathematical reverse view of the current HmxU8I8ToU8MatMul flow.  It
does not read HMX accumulator banks directly; those are not memory-visible.  It
reconstructs the values that must feed the U8 drain from the known activation,
logical weight, and folded bias/control contract:

  raw_acc[m,n]   = sum_k act_u8[m,k] * weight_i8[k,n]
  effective[n]   = -ACT_ZP * sum_k weight_i8[k,n] + bias_q[n]
  drain_in[m,n]  = raw_acc[m,n] + effective[n]
  output_u8[m,n] = clamp(trunc(drain_in[m,n] * scale / 512)
                         + (baseline_u16 >> 7), 0, 255)

For chain graphs, the output of one reconstructed drain becomes the U8
activation input of the next chain stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


ACT_ZP = 128


def _first(patterns: list[str], out_dir: Path) -> Path:
    for pattern in patterns:
        matches = sorted(out_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"none of these files exist under {out_dir}: {patterns}")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(arr).tobytes())


def _load_shape(out_dir: Path) -> tuple[int, int, int]:
    native_io = out_dir / "native_io.json"
    if native_io.exists():
        data = json.loads(native_io.read_text(encoding="utf-8"))
        shape = data.get("shape_mkn")
        if isinstance(shape, list) and len(shape) == 3:
            return int(shape[0]), int(shape[1]), int(shape[2])
    ref_path = _first(["*.out_ref_u8.npy"], out_dir)
    ref = np.load(ref_path)
    if ref.ndim == 2 and ref.shape[0] == ref.shape[1]:
        return int(ref.shape[0]), int(ref.shape[0]), int(ref.shape[1])
    raise ValueError(f"cannot infer M,K,N from {out_dir}")


def _detect_chain(out_dir: Path) -> int:
    summary = out_dir / "optrace" / "summary.json"
    if summary.exists():
        data = json.loads(summary.read_text(encoding="utf-8"))
        names = list(data.get("by_qnn_op_cycles", {}).keys())
        ids = []
        for name in names:
            m = re.search(r"hmx_u8i8_chain(\d+)", name)
            if m:
                ids.append(int(m.group(1)))
        if ids:
            return max(ids) + 1
    return 1


def _sample_points(m: int, n: int) -> list[tuple[int, int]]:
    candidates = [
        (0, 0),
        (0, 1),
        (1, 0),
        (min(31, m - 1), min(31, n - 1)),
        (min(32, m - 1), 0),
        (m - 1, n - 1),
    ]
    out: list[tuple[int, int]] = []
    seen = set()
    for p in candidates:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _terms_for(cur: np.ndarray, weight: np.ndarray, m: int, n: int, limit: int) -> list[dict[str, int]]:
    out = []
    for k in range(min(limit, weight.shape[0])):
        a = int(cur[m, k])
        w = int(weight[k, n])
        out.append({"k": k, "act_u8": a, "weight_i8": w, "product": a * w})
    return out


def _probe_config(out_dir: Path) -> dict[str, Any]:
    native_io = out_dir / "native_io.json"
    if not native_io.exists():
        return {}
    data = json.loads(native_io.read_text(encoding="utf-8"))
    probe = data.get("u8i8_probe")
    return probe if isinstance(probe, dict) else {}


def _drain_to_u8(drain_in: np.ndarray, scale_f16: float, baseline_u16: int) -> np.ndarray:
    scaled = np.trunc(drain_in.astype(np.float64) * float(scale_f16) / 512.0).astype(np.int64)
    shifted = scaled + (int(baseline_u16) >> 7)
    return np.clip(shifted, 0, 255).astype(np.uint8)


def reconstruct(out_dir: Path, chain: int, sample_terms: int) -> dict[str, Any]:
    m, k, n = _load_shape(out_dir)
    probe = _probe_config(out_dir)
    scale_f16 = float(probe.get("bias_scale_f16", 512.0))
    baseline_u16 = int(probe.get("bias_baseline_u16", 0))
    act_path = out_dir / "runtime_inputs_u8" / "act_u8i8.raw"
    if not act_path.exists():
        act_path = _first(["runtime_inputs*/act*.raw", "act*.raw"], out_dir)
    weight_path = _first(["*.wRaw_KN.npy"], out_dir)
    bias_q_path = _first(["*.bias_q_int32.npy"], out_dir)
    effective_path = _first(["*.effective_int32.npy"], out_dir)
    ref_path = _first(["*.out_ref_u8.npy"], out_dir)
    device_path = out_dir / "device_out" / "out.raw"

    act = np.fromfile(act_path, dtype=np.uint8).reshape(m, k)
    weight = np.load(weight_path).astype(np.int32)
    bias_q = np.load(bias_q_path).astype(np.int32)
    effective = np.load(effective_path).astype(np.int32)
    ref = np.load(ref_path).astype(np.uint8).reshape(m, n)
    device = None
    if device_path.exists():
        device = np.fromfile(device_path, dtype=np.uint8).reshape(m, n)

    if weight.shape != (k, n):
        raise ValueError(f"weight shape {weight.shape} does not match K,N {(k, n)}")
    if bias_q.shape != (n,) or effective.shape != (n,):
        raise ValueError("bias_q/effective arrays must be length N")

    sum_w = weight.sum(axis=0).astype(np.int32)
    expected_effective = (-ACT_ZP) * sum_w + bias_q
    effective_matches = bool(np.array_equal(effective, expected_effective))

    cur = act
    stages: list[dict[str, Any]] = []
    points = _sample_points(m, n)
    final_out = None
    for stage in range(chain):
        raw_acc = cur.astype(np.int32) @ weight
        drain_in = raw_acc + effective.reshape(1, n)
        scaled = np.trunc(drain_in.astype(np.float64) * scale_f16 / 512.0).astype(np.int64)
        shifted = scaled + (baseline_u16 >> 7)
        out = np.clip(shifted, 0, 255).astype(np.uint8)
        final_out = out

        samples = []
        for row, col in points:
            logical_acc = int((cur[row, :].astype(np.int32) - ACT_ZP) @ weight[:, col] + bias_q[col])
            samples.append(
                {
                    "m": row,
                    "n": col,
                    "raw_acc_sum_act_times_weight": int(raw_acc[row, col]),
                    "sum_weight": int(sum_w[col]),
                    "bias_q": int(bias_q[col]),
                    "effective_bias_record_i32": int(effective[col]),
                    "drain_input_raw_acc_plus_effective": int(drain_in[row, col]),
                    "scale_f16": scale_f16,
                    "scaled_drain_input": int(scaled[row, col]),
                    "baseline_u16": baseline_u16,
                    "baseline_shift": int(baseline_u16 >> 7),
                    "logical_acc_minus_zp_plus_bias": logical_acc,
                    "output_u8_after_clamp": int(out[row, col]),
                    "first_terms": _terms_for(cur, weight, row, col, sample_terms),
                }
            )

        stages.append(
            {
                "stage": stage,
                "input_sha256": _sha256_array(cur),
                "raw_acc_min": int(raw_acc.min()),
                "raw_acc_max": int(raw_acc.max()),
                "drain_input_min": int(drain_in.min()),
                "drain_input_max": int(drain_in.max()),
                "scaled_min": int(scaled.min()),
                "scaled_max": int(scaled.max()),
                "baseline_shift": int(baseline_u16 >> 7),
                "clamped_low_count": int((shifted < 0).sum()),
                "clamped_high_count": int((shifted > 255).sum()),
                "output_sha256": _sha256_array(out),
                "samples": samples,
            }
        )
        cur = out

    assert final_out is not None
    ref_matches = bool(np.array_equal(final_out, ref))
    device_matches = bool(device is not None and np.array_equal(final_out, device))
    return {
        "artifact": str(out_dir),
        "shape_mkn": [m, k, n],
        "chain": chain,
        "act_zp": ACT_ZP,
        "inputs": {
            "activation_raw": str(act_path),
            "weight_raw_kn": str(weight_path),
            "bias_q": str(bias_q_path),
            "effective_i32": str(effective_path),
            "reference": str(ref_path),
            "device_output": str(device_path) if device_path.exists() else None,
        },
        "bias_contract": {
            "formula": "effective[n] = -ACT_ZP * sum_k(weight[k,n]) + bias_q[n]",
            "effective_matches_saved_file": effective_matches,
            "sum_weight_min": int(sum_w.min()),
            "sum_weight_max": int(sum_w.max()),
            "effective_min": int(effective.min()),
            "effective_max": int(effective.max()),
        },
        "drain_contract": {
            "formula": "out_u8 = clamp(trunc((raw_acc + effective) * scale_f16 / 512) + (baseline_u16 >> 7), 0, 255)",
            "scale_f16": scale_f16,
            "baseline_u16": baseline_u16,
            "baseline_shift": int(baseline_u16 >> 7),
            "probe_config": probe,
        },
        "final": {
            "reconstructed_matches_ref": ref_matches,
            "reconstructed_matches_device": device_matches,
            "reconstructed_sha256": _sha256_array(final_out),
            "ref_sha256": _sha256_array(ref),
            "device_sha256": _sha256_array(device) if device is not None else None,
        },
        "stages": stages,
    }


def print_report(report: dict[str, Any]) -> None:
    print("=== U8I8 accumulator -> drain reconstruction ===")
    print(f"artifact: {report['artifact']}")
    print(f"shape: M,K,N={report['shape_mkn']}  chain={report['chain']}  ACT_ZP={report['act_zp']}")
    bc = report["bias_contract"]
    print("\n[bias/control contract]")
    print(f"  {bc['formula']}")
    print(f"  effective matches saved file: {bc['effective_matches_saved_file']}")
    print(f"  sum_weight range: {bc['sum_weight_min']} .. {bc['sum_weight_max']}")
    print(f"  effective range: {bc['effective_min']} .. {bc['effective_max']}")
    dc = report["drain_contract"]
    print("\n[drain contract]")
    print(f"  {dc['formula']}")
    print(f"  scale_f16={dc['scale_f16']} baseline_u16={dc['baseline_u16']} baseline_shift={dc['baseline_shift']}")
    print("\n[stage summary]")
    for st in report["stages"]:
        print(
            f"  stage {st['stage']}: raw_acc {st['raw_acc_min']}..{st['raw_acc_max']}  "
            f"drain_in {st['drain_input_min']}..{st['drain_input_max']}  "
            f"scaled {st['scaled_min']}..{st['scaled_max']}  "
            f"clamp_low={st['clamped_low_count']} clamp_high={st['clamped_high_count']}  "
            f"out_sha256={st['output_sha256'][:12]}"
        )
    final = report["final"]
    print("\n[final check]")
    print(f"  reconstructed == ref:    {final['reconstructed_matches_ref']}")
    print(f"  reconstructed == device: {final['reconstructed_matches_device']}")
    print(f"  reconstructed sha256:    {final['reconstructed_sha256']}")
    print(f"  ref sha256:              {final['ref_sha256']}")
    print(f"  device sha256:           {final['device_sha256']}")

    first_stage = report["stages"][0]
    print("\n[samples from stage 0]")
    for sample in first_stage["samples"]:
        print(
            f"  m{sample['m']},n{sample['n']}: "
            f"raw_acc={sample['raw_acc_sum_act_times_weight']}  "
            f"effective={sample['effective_bias_record_i32']}  "
            f"drain_in={sample['drain_input_raw_acc_plus_effective']}  "
            f"scaled={sample['scaled_drain_input']}  "
            f"baseline_shift={sample['baseline_shift']}  "
            f"out={sample['output_u8_after_clamp']}"
        )
        terms = ", ".join(
            f"k{t['k']}:{t['act_u8']}*{t['weight_i8']}={t['product']}"
            for t in sample["first_terms"]
        )
        print(f"    first terms: {terms}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "out_dir",
        type=Path,
        help="u8i8 artifact directory, e.g. example/qnn_matmul_profile/output_u8i8_aligned_e2e_256",
    )
    parser.add_argument("--chain", type=int, default=None, help="override chain length; default detects optrace or uses 1")
    parser.add_argument("--sample-terms", type=int, default=8, help="number of K terms to show for sample dot products")
    parser.add_argument("--json-out", type=Path, default=None, help="optional path to write full JSON reconstruction")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    chain = args.chain if args.chain is not None else _detect_chain(out_dir)
    report = reconstruct(out_dir, chain=chain, sample_terms=args.sample_terms)
    print_report(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
