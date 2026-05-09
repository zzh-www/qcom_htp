#!/usr/bin/env python3
"""Generate W16A16 native prepared sidecars from a public ONNX W.

The native W16A16 ConvLayer_s1 path stores each quantized int16 weight as
low/high byte lanes in 16-byte groups.  The traversal below was recovered from
the 256^3 native oracle:

  split N128 -> N32 tile -> q16 low/high half -> K32 tile -> row group -> lane

For each eight q16 values, native storage is:

  low[0:4], rounded_high[0:4], low[4:8], rounded_high[4:8]

where q16 = round(float_W * 32767) in float64 and rounded_high is
(q16 + 128) >> 8.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def _load_w(path: Path, name: str) -> np.ndarray:
    model = onnx.load(str(path))
    for init in model.graph.initializer:
        if init.name == name:
            arr = numpy_helper.to_array(init).astype(np.float64)
            return arr.reshape(arr.shape[-2], arr.shape[-1])
    raise ValueError(f"{path}: initializer {name!r} not found")


def generate_sidecar(w: np.ndarray) -> bytes:
    if w.shape[0] % 32 or w.shape[1] % 128:
        raise ValueError(f"W shape must be K multiple of 32 and N multiple of 128, got {w.shape}")
    q16 = np.clip(np.rint(w * 32767.0), -32768, 32767).astype(np.int32)
    out = bytearray()
    k_total, n_total = q16.shape
    for n_base in range(0, n_total, 128):
        for nt in range(4):
            for half in range(2):
                for kt in range(k_total // 32):
                    for grp in range(8):
                        vals: list[int] = []
                        for lane in range(4):
                            ch = grp * 8 + half * 4 + lane
                            for off in range(ch * 16, (ch + 1) * 16):
                                rgrp = off // 128
                                rem = off % 128
                                col = rem // 4
                                rmod = rem % 4
                                row = rgrp * 4 + rmod
                                vals.append(int(q16[kt * 32 + row, n_base + nt * 32 + col]))
                        for idx in range(0, len(vals), 8):
                            block = vals[idx:idx + 8]
                            out.extend((value & 0xff) for value in block[:4])
                            out.extend((((value + 128) >> 8) & 0xff) for value in block[:4])
                            out.extend((value & 0xff) for value in block[4:8])
                            out.extend((((value + 128) >> 8) & 0xff) for value in block[4:8])
    return bytes(out)


def generate_bias_sidecar(w: np.ndarray) -> bytes:
    if w.shape[0] % 32 or w.shape[1] % 128:
        raise ValueError(f"W shape must be K multiple of 32 and N multiple of 128, got {w.shape}")
    q16 = np.clip(np.rint(w * 32767.0), -32768, 32767).astype(np.int64)
    control = np.array([0x00404420, 0x40000000] * 16, dtype="<i4")
    records = np.zeros((w.shape[1] // 128, 512), dtype="<i4")
    for split, n_base in enumerate(range(0, w.shape[1], 128)):
        vals = ((-q16[:, n_base:n_base + 128].sum(axis=0)) // 2).astype(np.int32)
        for group in range(8):
            records[split, group * 64:group * 64 + 32] = control
            value_base = group * 64 + 32
            for idx, value in enumerate(vals[group * 16:(group + 1) * 16]):
                records[split, value_base + idx * 2] = value
                records[split, value_base + idx * 2 + 1] = 0
    return records.tobytes()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--initializer", default="W")
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--bias-out", type=Path)
    args = parser.parse_args()

    w = _load_w(args.onnx, args.initializer)
    data = generate_sidecar(w)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(f"wrote {args.out}")
    print(f"bytes={len(data)} sha256={hashlib.sha256(data).hexdigest()}")
    print(f"first32={data[:32].hex()}")
    if args.bias_out:
        bias = generate_bias_sidecar(w)
        args.bias_out.parent.mkdir(parents=True, exist_ok=True)
        args.bias_out.write_bytes(bias)
        print(f"wrote {args.bias_out}")
        print(f"bias_bytes={len(bias)} bias_sha256={hashlib.sha256(bias).hexdigest()}")
        print(f"bias_first32={bias[:32].hex()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
