#!/usr/bin/env python3
"""Generate a QNN-native A8 MatMul+Add chain matched to a custom artifact.

The input, logical weight matrix, folded bias, chain length, and output shape
come from an existing custom u8i8/w4a8 run directory.  This is the precision
and performance oracle path for A8 custom kernels: native references must not
use random same-shape tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


FAMILIES = {
    "u8i8": {"weight_bits": 8, "onnx_prefix": "u8i8"},
    "w4a8": {"weight_bits": 4, "onnx_prefix": "w4a8"},
}


def _u8_encoding() -> dict:
    return {
        "bitwidth": 8,
        "dtype": "int",
        "is_symmetric": "False",
        "scale": 1.0,
        "offset": 0,
        "min": 0.0,
        "max": 255.0,
    }


def _signed_weight_encoding(bits: int) -> dict:
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    return {
        "bitwidth": bits,
        "dtype": "int",
        "is_symmetric": "True",
        "scale": 1.0,
        "offset": 0,
        "min": float(qmin),
        "max": float(qmax),
    }


def _load_custom_arrays(custom_dir: Path, family: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = FAMILIES[family]["onnx_prefix"]
    a_path = custom_dir / "runtime_inputs_u8" / f"act_{family}.raw"
    w_path = custom_dir / f"{prefix}.onnx.wRaw_KN.npy"
    b_path = custom_dir / f"{prefix}.onnx.effective_int32.npy"
    missing = [str(p) for p in (a_path, w_path, b_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing custom source artifact(s): " + ", ".join(missing))
    w = np.load(w_path).astype(np.float32)
    b = np.load(b_path).astype(np.float32)
    a = np.fromfile(a_path, dtype=np.uint8)
    k, n = w.shape
    if b.shape != (n,):
        raise ValueError(f"bias shape mismatch: got {b.shape}, expected {(n,)}")
    if a.size % k:
        raise ValueError(f"activation size {a.size} is not divisible by K={k}")
    m = a.size // k
    return a.reshape(1, m, k), w, b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES.keys(), required=True)
    parser.add_argument("--custom-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chain", type=int, default=8)
    args = parser.parse_args()

    custom_dir = args.custom_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    a_u8, w_float, bias_float = _load_custom_arrays(custom_dir, args.family)
    _, m, k = a_u8.shape
    wk, n = w_float.shape
    if wk != k:
        raise ValueError(f"W K mismatch: input K={k}, W shape={w_float.shape}")
    chain = max(1, int(args.chain))

    input_info = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, m, k])
    output_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, m, n])
    initializers = [
        numpy_helper.from_array(w_float.astype(np.float32), name="W"),
        numpy_helper.from_array(bias_float.astype(np.float32), name="B"),
    ]
    nodes = []
    prev = "A"
    activation_encodings = {"A": [_u8_encoding()]}
    for i in range(chain):
        mm_out = f"MM_{i}"
        add_out = "Y" if i == chain - 1 else f"Y_{i}"
        nodes.append(helper.make_node("MatMul", [prev, "W"], [mm_out], name=f"MatMul_{i}"))
        nodes.append(helper.make_node("Add", [mm_out, "B"], [add_out], name=f"AddBias_{i}"))
        activation_encodings[add_out] = [_u8_encoding()]
        prev = add_out

    graph = helper.make_graph(nodes, f"matched_native_{args.family}", [input_info], [output_info], initializers)
    model = helper.make_model(
        graph,
        producer_name=f"matched_native_{args.family}_{m}x{k}x{n}",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, out_dir / "matmul.onnx")

    overrides = {
        "activation_encodings": activation_encodings,
        "param_encodings": {"W": [_signed_weight_encoding(FAMILIES[args.family]["weight_bits"])]},
    }
    (out_dir / "quant_overrides.json").write_text(json.dumps(overrides, indent=2), encoding="utf-8")

    runtime_dir = out_dir / "runtime_inputs_native"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    a_u8.tofile(runtime_dir / "A.raw")
    (out_dir / "runtime_input_list.txt").write_text("A:=runtime_inputs_native/A.raw\n", encoding="utf-8")
    # Keep a legacy float input only for host-side inspection; device execution
    # uses runtime_inputs_native/A.raw.
    a_u8.astype(np.float32).tofile(out_dir / "input_A.raw")
    (out_dir / "input_list.txt").write_text("A:=input_A.raw\n", encoding="utf-8")

    native_io = {
        "input_name": "A",
        "output_name": "Y",
        "legacy_fp32_input": "input_A.raw",
        "native_input": "runtime_inputs_native/A.raw",
        "runtime_input_list": "runtime_input_list.txt",
        "native_input_storage": "uint8",
        "native_input_bytes": int(a_u8.size),
        "expected_native_output_storage": "uint8",
        "expected_native_output_bytes": int(m * n),
        "shape_mkn": [m, k, n],
        "chain": chain,
        "matched_custom_dir": str(custom_dir),
        "matched_weight": f"{FAMILIES[args.family]['onnx_prefix']}.onnx.wRaw_KN.npy",
        "matched_effective_bias": f"{FAMILIES[args.family]['onnx_prefix']}.onnx.effective_int32.npy",
    }
    (out_dir / "native_io.json").write_text(json.dumps(native_io, indent=2), encoding="utf-8")

    np.save(out_dir / "wRaw_KN.npy", w_float.astype(np.int32))
    np.save(out_dir / "effective_int32.npy", bias_float.astype(np.int32))
    print(f"matched native {args.family}: {m}x{k}x{n}, chain={chain} -> {out_dir}")


if __name__ == "__main__":
    main()
