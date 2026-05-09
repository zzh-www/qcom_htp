#!/usr/bin/env python3
"""Compare W16A16 prepared sidecars against public weight candidates.

This is a diagnostic, not a packer.  It records whether the current imported
QInt8 sidecars can be explained by simple projections of either the custom
logical W16 matrix or the QNN-native public float W tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stats(candidate: np.ndarray, native: np.ndarray) -> dict[str, Any]:
    cand = candidate.astype(np.int8, copy=False).reshape(-1)
    nat = native[: cand.size]
    cand_hist = np.bincount(cand.view(np.uint8), minlength=256)
    nat_hist = np.bincount(nat.view(np.uint8), minlength=256)
    return {
        "bytes": int(cand.size),
        "sha256": _sha256(cand.tobytes()),
        "exact_prefix": int((cand == nat).sum()),
        "exact_prefix_total": int(cand.size),
        "sorted_equal_prefix": bool(np.array_equal(np.sort(cand), np.sort(nat))),
        "hist_absdiff_sum": int(np.abs(cand_hist - nat_hist).sum()),
        "first32_hex": cand.tobytes()[:32].hex(),
    }


def _prefix_equal_len(lhs: np.ndarray, rhs: np.ndarray) -> int:
    n = min(lhs.size, rhs.size)
    if n == 0:
        return 0
    diff = np.nonzero(lhs[:n] != rhs[:n])[0]
    return int(diff[0]) if diff.size else int(n)


def _extract_mod_lanes(native: np.ndarray, mods: list[int], group: int = 16) -> np.ndarray:
    lanes: list[np.ndarray] = []
    for base in range(0, native.size, group):
        for mod in mods:
            idx = base + mod
            if idx < native.size:
                lanes.append(native[idx:idx + 1])
    if not lanes:
        return np.zeros(0, dtype=np.int8)
    return np.concatenate(lanes).astype(np.int8, copy=False)


def _lane_report(native: np.ndarray, candidates: dict[str, np.ndarray]) -> dict[str, Any]:
    lane_sets = {
        "lane_0_3": [0, 1, 2, 3],
        "lane_4_7": [4, 5, 6, 7],
        "lane_8_11": [8, 9, 10, 11],
        "lane_12_15": [12, 13, 14, 15],
        "lane_0_3_8_11": [0, 1, 2, 3, 8, 9, 10, 11],
        "lane_4_7_12_15": [4, 5, 6, 7, 12, 13, 14, 15],
    }
    report: dict[str, Any] = {}
    for lane_name, mods in lane_sets.items():
        extracted = _extract_mod_lanes(native, mods)
        best = []
        for cand_name, candidate in candidates.items():
            cand = candidate.reshape(-1).astype(np.int8, copy=False)
            n = min(extracted.size, cand.size)
            if n == 0:
                continue
            chunk = 16
            cand_chunks = {
                cand[idx:idx + chunk].tobytes()
                for idx in range(0, n - chunk + 1, chunk)
            }
            chunk_hits = 0
            chunk_total = 0
            for idx in range(0, n - chunk + 1, chunk):
                chunk_total += 1
                if extracted[idx:idx + chunk].tobytes() in cand_chunks:
                    chunk_hits += 1
            best.append({
                "candidate": cand_name,
                "exact": int((extracted[:n] == cand[:n]).sum()),
                "total": int(n),
                "prefix_equal": _prefix_equal_len(extracted, cand),
                "chunk16_hits_anywhere": chunk_hits,
                "chunk16_total": chunk_total,
                "candidate_first32_hex": cand.tobytes()[:32].hex(),
            })
        best.sort(key=lambda item: (item["exact"], item["prefix_equal"]), reverse=True)
        report[lane_name] = {
            "mods": mods,
            "bytes": int(extracted.size),
            "first32_hex": extracted.tobytes()[:32].hex(),
            "best_candidates": best[:6],
        }
    return report


def _pack_w8_kmajor_split128(w: np.ndarray) -> np.ndarray:
    k, n = w.shape
    if k % 32 or n % 128:
        raise ValueError("K must be multiple of 32 and N must be multiple of 128")
    out = np.zeros(k * n, dtype=np.int8)
    pos = 0
    for n_base in range(0, n, 128):
        for kt in range(k // 32):
            for nt in range(4):
                tile = np.zeros(1024, dtype=np.int8)
                k_base = kt * 32
                col_base = n_base + nt * 32
                for r in range(32):
                    for c in range(32):
                        dst = (r // 4) * 128 + c * 4 + (r % 4)
                        tile[dst] = w[k_base + r, col_base + c]
                out[pos:pos + 1024] = tile
                pos += 1024
    return out


def _load_native_float_w(path: Path) -> np.ndarray | None:
    if not path:
        return None
    model = onnx.load(str(path))
    for init in model.graph.initializer:
        if init.name == "W":
            arr = numpy_helper.to_array(init)
            return arr.reshape(arr.shape[-2], arr.shape[-1]).astype(np.float32)
    raise ValueError(f"{path}: no initializer named W")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    native = np.fromfile(args.sidecar_raw, dtype=np.int8)
    report: dict[str, Any] = {
        "sidecar_raw": str(args.sidecar_raw),
        "sidecar_bytes": int(native.size),
        "sidecar_sha256": _sha256(native.tobytes()),
        "candidates": {},
    }
    candidate_arrays: dict[str, np.ndarray] = {}

    if args.custom_w_raw:
        w = np.load(args.custom_w_raw).astype(np.int32)
        simple = {
            "custom_clip_flat": np.clip(w, -128, 127).astype(np.int8),
            "custom_hi8_flat": (w >> 8).astype(np.int8),
            "custom_lo8_flat": w.astype(np.int8),
            "custom_clip_kmajor_split128": _pack_w8_kmajor_split128(
                np.clip(w, -128, 127).astype(np.int8)
            ),
        }
        for name, arr in simple.items():
            candidate_arrays[name] = arr.reshape(-1).astype(np.int8, copy=False)
            report["candidates"][name] = _stats(arr, native)

    if args.native_onnx:
        w_float = _load_native_float_w(args.native_onnx)
        for scale in args.native_float_scales:
            q = np.clip(np.rint(w_float * scale), -128, 127).astype(np.int8)
            flat_name = f"native_float_round_x{scale:g}_flat"
            packed_name = f"native_float_round_x{scale:g}_kmajor_split128"
            packed = _pack_w8_kmajor_split128(q)
            candidate_arrays[flat_name] = q.reshape(-1).astype(np.int8, copy=False)
            candidate_arrays[packed_name] = packed.reshape(-1).astype(np.int8, copy=False)
            report["candidates"][flat_name] = _stats(q, native)
            report["candidates"][packed_name] = _stats(packed, native)

    report["lane_candidates"] = _lane_report(native, candidate_arrays)

    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-raw", type=Path, required=True)
    parser.add_argument("--custom-w-raw", type=Path)
    parser.add_argument("--native-onnx", type=Path)
    parser.add_argument(
        "--native-float-scales",
        type=float,
        nargs="*",
        default=[64.0, 127.0, 128.0, 255.0, 256.0],
    )
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args()

    report = analyze(args)
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
