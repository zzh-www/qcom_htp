#!/usr/bin/env python3
"""Analyze an A16-output native Conv1x1 bias/control sidecar record.

This helper is the A16 counterpart of ``analyze_u8i8_native_bias_record.py``.
It compares a generated per-channel Python/QNN case against a native context
binary and, optionally, a custom-op ONNX initializer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.qnn_htp_bias_prepare import qnn_htp_perchannel_a16_sidecar_bias_q
from scripts.qnn_htp_u8_drain import (
    qnn_htp_w8a16_drain_control_words,
    qnn_htp_w8a16_drain_scale,
)

ACT_ZP = 128


def load_case(case_dir: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    if meta["family"] not in {"w8a16", "w4a16_per_channel"}:
        raise ValueError(f"expected w8a16 or w4a16_per_channel case, got {meta['family']!r}")
    if meta.get("weight_schema_variant") != "per_output_channel":
        raise ValueError(
            "A16 native bias record analysis currently supports per-output-channel cases only"
        )
    files = meta["files"]
    weight_q_nk = np.load(case_dir / files["weight_q_nk"]["npy"]).astype(np.int8)
    bias_q = np.load(case_dir / files["bias_q_int32"]["npy"]).astype(np.int32)
    weight_scale = np.load(case_dir / files["weight_scale"]["npy"]).astype(np.float32)
    shape = np.array(meta["shape_mkn"], dtype=np.int64)
    return meta, weight_q_nk, bias_q, weight_scale, shape


def const_words_for_family(family: str) -> list[int]:
    if family == "w8a16":
        return [0x4440, 0x8040, 0x0008, 0x4000]
    if family == "w4a16_per_channel":
        return [0x5524, 0x8040, 0x0092, 0x4000]
    raise ValueError(f"unsupported family: {family}")


def expected_record(case_dir: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    meta, weight_q_nk, bias_q, weight_scale, shape = load_case(case_dir)
    _, k, n = [int(v) for v in shape]
    if k % 32 or n % 32:
        raise ValueError(f"K and N must be multiples of 32, got K={k} N={n}")

    act_scale_f64 = float(meta["qparams"]["activation"]["scale"])
    output_scale_f64 = float(meta["qparams"]["output"]["scale"])
    act_scale = np.float32(act_scale_f64)
    sidecar_bias = qnn_htp_perchannel_a16_sidecar_bias_q(
        bias_q,
        act_scale,
        weight_scale,
    )
    weight_q_kn = weight_q_nk.T.astype(np.int32)
    folded = (-ACT_ZP) * weight_q_kn.sum(axis=0).astype(np.int32)
    effective_i32 = (folded + sidecar_bias).astype(np.int32)

    record = np.zeros((n // 32, 512), dtype=np.uint8)
    const = np.array(const_words_for_family(meta["family"]), dtype=np.uint16)
    if meta["family"] in {"w8a16", "w4a16_per_channel"}:
        exact = qnn_htp_w8a16_drain_scale(
            act_scale_f64,
            weight_scale,
            output_scale_f64,
        )
        word0_u16, word1_u16, word2_u16, word3_u16 = qnn_htp_w8a16_drain_control_words(
            exact
        )
    else:
        word0_u16 = np.full(n, const[0], dtype=np.uint16)
        word1_u16 = np.full(n, const[1], dtype=np.uint16)
        word2_u16 = np.full(n, const[2], dtype=np.uint16)
        word3_u16 = np.full(n, const[3], dtype=np.uint16)
    for nt in range(n // 32):
        for parity in (0, 1):
            half_base = parity * 256
            for lane, c in enumerate(range(parity, 32, 2)):
                col = nt * 32 + c
                lane_base = half_base + 8 * lane
                record[nt, lane_base : lane_base + 8] = np.array(
                    [word0_u16[col], word1_u16[col], word2_u16[col], word3_u16[col]],
                    dtype=np.uint16,
                ).view(np.uint8)
                record[nt, half_base + 128 + 8 * lane : half_base + 132 + 8 * lane] = (
                    np.array([effective_i32[col]], dtype=np.int32).view(np.uint8)
                )
    return meta, record, effective_i32


def parse_record(record: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tiles = record.reshape(-1, 512)
    control = []
    effective = []
    for nt in range(tiles.shape[0]):
        for c in range(32):
            parity = c & 1
            lane = c // 2
            half_base = parity * 256
            lane_base = half_base + 8 * lane
            control.append(tiles[nt, lane_base : lane_base + 8].copy())
            effective.append(tiles[nt, half_base + 128 + 8 * lane : half_base + 132 + 8 * lane].view(np.int32)[0])
    control_arr = np.stack(control).reshape(-1, 8)
    effective_arr = np.array(effective, dtype=np.int32)
    return control_arr, effective_arr, tiles


def extract_custom_record(onnx_path: Path) -> np.ndarray:
    model = onnx.load(str(onnx_path))
    for init in model.graph.initializer:
        if init.name == "bias":
            return numpy_helper.to_array(init).view(np.uint8).reshape(-1, 512).copy()
    raise ValueError(f"{onnx_path}: missing bias initializer")


def find_native_record(ctx_bin: Path, expected: np.ndarray) -> tuple[int, np.ndarray, list[tuple[int, int, int, int]]]:
    buf = ctx_bin.read_bytes()
    length = int(expected.size)
    exp_control, exp_eff, _ = parse_record(expected)
    candidates: list[tuple[int, int, int, int]] = []
    for off in range(0, len(buf) - length + 1, 32):
        record = np.frombuffer(buf[off : off + length], dtype=np.uint8).reshape(expected.shape)
        control, eff, _ = parse_record(record)
        control_bytes = int((control == exp_control).sum())
        eff_match = int((eff == exp_eff).sum())
        byte_match = int((record == expected).sum())
        score = 8 * control_bytes + 64 * eff_match + byte_match
        candidates.append((score, off, control_bytes, eff_match))
    candidates.sort(reverse=True)
    best = candidates[0]
    record = np.frombuffer(buf[best[1] : best[1] + length], dtype=np.uint8).reshape(expected.shape).copy()
    return best[1], record, candidates[:5]


def counts(values: np.ndarray) -> dict[str, int]:
    return {str(int(v)): int((values == v).sum()) for v in np.unique(values)}


def control_word_mismatch_counts(lhs: np.ndarray, rhs: np.ndarray) -> list[int]:
    lhs_words = lhs.view(np.uint16).reshape(-1, 4)
    rhs_words = rhs.view(np.uint16).reshape(-1, 4)
    return [int((lhs_words[:, idx] != rhs_words[:, idx]).sum()) for idx in range(4)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--native-context-bin", required=True)
    parser.add_argument("--custom-onnx")
    parser.add_argument("--out-raw")
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    meta, expected, exp_eff = expected_record(case_dir)
    offset, native, candidates = find_native_record(Path(args.native_context_bin), expected)
    exp_control, _, _ = parse_record(expected)
    nat_control, nat_eff, _ = parse_record(native)

    print(f"case: {meta['family']}/{meta['case']} shape={meta['shape_mkn']}")
    print(f"native record offset: {offset}")
    print(f"top candidates: {candidates}")
    print(f"native vs generated bytes: {int((native == expected).sum())}/{expected.size}")
    print(f"control bytes match: {int((nat_control == exp_control).sum())}/{exp_control.size}")
    print(f"control word mismatches: {control_word_mismatch_counts(nat_control, exp_control)}")
    print(f"effective match: {int((nat_eff == exp_eff).sum())}/{exp_eff.size}")
    print(f"effective delta counts: {counts(nat_eff - exp_eff)}")
    nonzero = np.where(nat_eff != exp_eff)[0]
    if nonzero.size:
        preview = [(int(i), int(nat_eff[i] - exp_eff[i])) for i in nonzero[:32]]
        print(f"effective nonzero preview: {preview}")

    if args.custom_onnx:
        custom = extract_custom_record(Path(args.custom_onnx))
        custom_control, custom_eff, _ = parse_record(custom)
        print(f"native vs custom bytes: {int((native == custom).sum())}/{native.size}")
        print(
            "native vs custom control bytes: "
            f"{int((nat_control == custom_control).sum())}/{nat_control.size}"
        )
        print(
            "native vs custom control word mismatches: "
            f"{control_word_mismatch_counts(nat_control, custom_control)}"
        )
        print(
            "native vs custom effective fields: "
            f"{int((nat_eff == custom_eff).sum())}/{nat_eff.size}"
        )

    if args.out_raw:
        out = Path(args.out_raw)
        out.parent.mkdir(parents=True, exist_ok=True)
        native.reshape(-1).tofile(out)
        print(f"wrote native record: {out}")


if __name__ == "__main__":
    main()
