#!/usr/bin/env python3
"""Analyze an A8-output native Conv1x1 bias/control sidecar record.

This is a calibration helper for the custom u8i8 and w4a8-per-channel HMX
MatMul paths.  It does not build QNN artifacts; it compares a generated case
against an existing native context binary and, optionally, an existing custom
ONNX initializer.
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

from scripts.qnn_htp_bias_prepare import (
    qnn_htp_perchannel_a8_sidecar_bias_q,
    qnn_htp_perchannel_w4a8_sidecar_effective_i32,
)
from scripts.qnn_htp_u8_drain import qnn_htp_u8_drain_scale_control

ACT_ZP = 128


def native_drain_scale_control(exact_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return qnn_htp_u8_drain_scale_control(exact_scale)


def load_case(case_dir: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    if meta["family"] not in {"u8i8", "w4a8_per_channel"}:
        raise ValueError(f"expected u8i8 or w4a8_per_channel case, got {meta['family']!r}")
    files = meta["files"]
    weight_q_nk = np.load(case_dir / files["weight_q_nk"]["npy"]).astype(np.int8)
    bias_q = np.load(case_dir / files["bias_q_int32"]["npy"]).astype(np.int32)
    weight_scale = np.load(case_dir / files["weight_scale"]["npy"]).astype(np.float32)
    bias_scale = np.array(meta["qparams"]["bias"]["scale"], dtype=np.float32).reshape(-1)
    return meta, weight_q_nk, bias_q, weight_scale, bias_scale, np.array(meta["shape_mkn"], dtype=np.int64)


def expected_record(case_dir: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    meta, weight_q_nk, bias_q, weight_scale, bias_scale, shape = load_case(case_dir)
    _, _, n = [int(v) for v in shape]
    if n % 32:
        raise ValueError(f"N must be a multiple of 32, got {n}")

    act_scale = np.float32(meta["qparams"]["activation"]["scale"])
    output_scale = np.float32(meta["qparams"]["output"]["scale"])
    exact_scale = (
        np.float32(512.0) * act_scale * weight_scale.astype(np.float32) / output_scale
    ).astype(np.float32)
    if meta["family"] == "w4a8_per_channel":
        exact_scale = (exact_scale / np.float32(16.0)).astype(np.float32)
    scale_u16, control_u16 = native_drain_scale_control(exact_scale)

    weight_q_kn = weight_q_nk.T.astype(np.int32)
    if meta["family"] == "w4a8_per_channel":
        effective_i32 = qnn_htp_perchannel_w4a8_sidecar_effective_i32(
            bias_q,
            act_scale,
            weight_scale,
            weight_q_kn,
            ACT_ZP,
        )
    else:
        prepared_bias_q = qnn_htp_perchannel_a8_sidecar_bias_q(bias_q, act_scale, weight_scale)
        effective_i32 = (-ACT_ZP) * weight_q_kn.sum(axis=0).astype(np.int32) + prepared_bias_q

    record = np.zeros((n // 32, 256), dtype=np.uint8)
    for nt in range(n // 32):
        for c in range(32):
            col = nt * 32 + c
            record[nt, 4 * c : 4 * c + 2] = np.array([scale_u16[col]], np.uint16).view(np.uint8)
            record[nt, 4 * c + 2 : 4 * c + 4] = np.array([control_u16[col]], np.uint16).view(np.uint8)
            record[nt, 128 + 4 * c : 128 + 4 * c + 4] = np.array(
                [effective_i32[col]], np.int32
            ).view(np.uint8)
    return meta, record, scale_u16, control_u16, effective_i32


def parse_record(record: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tiles = record.reshape(-1, 256)
    scale_u16: list[np.uint16] = []
    control_u16: list[np.uint16] = []
    effective_i32: list[np.int32] = []
    for nt in range(tiles.shape[0]):
        for c in range(32):
            scale_u16.append(tiles[nt, 4 * c : 4 * c + 2].view(np.uint16)[0])
            control_u16.append(tiles[nt, 4 * c + 2 : 4 * c + 4].view(np.uint16)[0])
            effective_i32.append(tiles[nt, 128 + 4 * c : 128 + 4 * c + 4].view(np.int32)[0])
    return (
        np.array(scale_u16, dtype=np.uint16),
        np.array(control_u16, dtype=np.uint16),
        np.array(effective_i32, dtype=np.int32),
    )


def extract_custom_record(onnx_path: Path) -> np.ndarray:
    model = onnx.load(str(onnx_path))
    for init in model.graph.initializer:
        if init.name == "bias":
            return numpy_helper.to_array(init).view(np.uint8).reshape(-1, 256).copy()
    raise ValueError(f"{onnx_path}: missing bias initializer")


def extract_dlc_bias(dlc_path: Path) -> np.ndarray:
    """Read the quantized Conv bias tensor from a QAIRT DLC."""
    try:
        from qti.aisw.dlc_utils import snpe_dlc_utils
    except ImportError as exc:
        raise RuntimeError(
            "reading DLC bias requires QNN SDK PYTHONPATH, for example: "
            "source scripts/env.sh && export PYTHONPATH=\"$QNN_SDK_ROOT/lib/python\""
        ) from exc

    reader = snpe_dlc_utils.modeltools.IrDlcReader()
    reader.open(str(dlc_path))
    try:
        for graph_name in reader.get_ir_graph_names():
            graph = reader.get_ir_graph(graph_name)
            for op in graph.get_ops():
                if getattr(op, "name", None) != "conv1x1":
                    continue
                for tensor in op.inputs():
                    if tensor.name() == "B":
                        return np.array(tensor.get_data(), dtype=np.int32).reshape(-1)
    finally:
        reader.close()
    raise ValueError(f"{dlc_path}: missing conv1x1 static B tensor")


def find_native_record(ctx_bin: Path, expected: np.ndarray) -> tuple[int, np.ndarray, list[tuple[int, int, int, int, int]]]:
    buf = ctx_bin.read_bytes()
    length = int(expected.size)
    exp_scale, exp_control, exp_eff = parse_record(expected)
    candidates: list[tuple[int, int, int, int, int]] = []
    for off in range(0, len(buf) - length + 1, 32):
        record = np.frombuffer(buf[off : off + length], dtype=np.uint8).reshape(expected.shape)
        scale, control, eff = parse_record(record)
        scale_match = int((scale == exp_scale).sum())
        control_match = int((control == exp_control).sum())
        eff_close = int((np.abs(eff - exp_eff) <= 1).sum())
        byte_match = int((record == expected).sum())
        score = 8 * (scale_match + control_match) + eff_close + byte_match
        candidates.append((score, off, scale_match, control_match, eff_close))
    candidates.sort(reverse=True)
    best = candidates[0]
    record = np.frombuffer(buf[best[1] : best[1] + length], dtype=np.uint8).reshape(expected.shape).copy()
    return best[1], record, candidates[:5]


def counts(values: np.ndarray) -> dict[str, int]:
    return {str(int(v)): int((values == v).sum()) for v in np.unique(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--native-context-bin", required=True)
    parser.add_argument(
        "--native-dlc",
        help="Optional quantized native DLC. When provided, compare its static Conv B tensor too.",
    )
    parser.add_argument("--custom-onnx")
    parser.add_argument("--out-raw")
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    meta, expected, _, _, _ = expected_record(case_dir)
    offset, native, candidates = find_native_record(Path(args.native_context_bin), expected)
    exp_scale, exp_control, exp_eff = parse_record(expected)
    nat_scale, nat_control, nat_eff = parse_record(native)
    _, weight_q_nk, bias_q, _, _, _ = load_case(case_dir)
    folded_i32 = (-ACT_ZP) * weight_q_nk.T.astype(np.int32).sum(axis=0).astype(np.int32)
    nat_bias_q = (nat_eff - folded_i32).astype(np.int32)

    print(f"case: {meta['family']}/{meta['case']} shape={meta['shape_mkn']}")
    print(f"native record offset: {offset}")
    print(f"top candidates: {candidates}")
    print(f"native vs generated bytes: {int((native == expected).sum())}/{expected.size}")
    print(f"scale match: {int((nat_scale == exp_scale).sum())}/{exp_scale.size}")
    print(f"control match: {int((nat_control == exp_control).sum())}/{exp_control.size}")
    print(f"effective delta counts: {counts(nat_eff - exp_eff)}")
    nonzero = np.where(nat_eff != exp_eff)[0]
    if nonzero.size:
        preview = [(int(i), int(nat_eff[i] - exp_eff[i])) for i in nonzero[:32]]
        print(f"effective nonzero preview: {preview}")

    if args.native_dlc:
        dlc_bias = extract_dlc_bias(Path(args.native_dlc))
        if dlc_bias.shape != bias_q.shape:
            raise ValueError(
                f"{args.native_dlc}: DLC B shape {dlc_bias.shape} != case bias shape {bias_q.shape}"
            )
        print(f"DLC B vs generated bias_q: {counts(dlc_bias - bias_q)}")
        print(f"native sidecar bias_q vs DLC B: {counts(nat_bias_q - dlc_bias)}")
        nonzero_bias = np.where(nat_bias_q != dlc_bias)[0]
        if nonzero_bias.size:
            preview = [
                (
                    int(i),
                    int(bias_q[i]),
                    int(dlc_bias[i]),
                    int(nat_bias_q[i]),
                    int(nat_bias_q[i] - dlc_bias[i]),
                )
                for i in nonzero_bias[:32]
            ]
            print(
                "native/DLC bias nonzero preview: "
                "[(idx, generated, dlc, native_sidecar, delta), ...] "
                f"{preview}"
            )

    if args.custom_onnx:
        custom = extract_custom_record(Path(args.custom_onnx))
        print(f"native vs custom bytes: {int((native == custom).sum())}/{native.size}")
        print(f"native vs custom scale/control bytes: {int((native[:, :128] == custom[:, :128]).sum())}/{native[:, :128].size}")
        print(f"native vs custom effective bytes: {int((native[:, 128:] == custom[:, 128:]).sum())}/{native[:, 128:].size}")

    if args.out_raw:
        out = Path(args.out_raw)
        out.parent.mkdir(parents=True, exist_ok=True)
        native.reshape(-1).tofile(out)
        print(f"wrote native record: {out}")


if __name__ == "__main__":
    main()
