#!/usr/bin/env python3
"""Generate a QNN-native W8A16 MatMul chain matched to a custom artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _a16_encoding() -> dict:
    return {
        "bitwidth": 16,
        "dtype": "int",
        "is_symmetric": "True",
        "scale": 1.0 / 32767.0,
        "offset": -32768,
        "min": -1.0,
        "max": 1.0,
    }


def _w8_encoding() -> dict:
    return {
        "bitwidth": 8,
        "dtype": "int",
        "is_symmetric": "True",
        "scale": 1.0 / 127.0,
        "offset": -128,
        "min": -1.0,
        "max": 1.0,
    }


def _load_custom(custom_dir: Path) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    native_io = json.loads((custom_dir / "native_io.json").read_text(encoding="utf-8"))
    m, k, n = native_io["shape_mkn"]
    chain = int(native_io.get("chain", 1))
    a_path = custom_dir / "runtime_inputs_u8" / "act_w8a16.raw"
    w_path = custom_dir / "w8a16.onnx.wRaw_KN.npy"
    a = np.fromfile(a_path, dtype=np.uint16)
    if a.size != m * k:
        raise ValueError(f"activation size mismatch: got {a.size}, expected {m * k}")
    w = np.load(w_path).astype(np.int8)
    if w.shape != (k, n):
        raise ValueError(f"weight shape mismatch: got {w.shape}, expected {(k, n)}")
    return a.reshape(1, m, k), w, m, k, n, chain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chain", type=int, default=0, help="override chain length; 0 reads custom native_io.json")
    args = parser.parse_args()

    custom_dir = args.custom_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    a_u16, w_i8, m, k, n, custom_chain = _load_custom(custom_dir)
    chain = int(args.chain) if args.chain else custom_chain
    if chain < 1:
        raise ValueError("chain must be >= 1")
    if chain != 1 and k != n:
        raise ValueError("W8A16 native chain requires K == N between MatMul nodes")

    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, m, k])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, m, n])
    w_init = numpy_helper.from_array((w_i8.astype(np.float32) / 127.0).reshape(1, k, n), name="W")

    nodes = []
    value_infos = []
    prev = "A"
    for i in range(chain):
        out_name = "Y" if i == chain - 1 else f"Y_{i}"
        nodes.append(helper.make_node(
            "MatMul",
            [prev, "W"],
            [out_name],
            name=f"matmul_{i}" if chain > 1 else "matmul_1",
        ))
        if i < chain - 1:
            value_infos.append(helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [1, m, n]))
        prev = out_name

    graph = helper.make_graph(nodes, "matched_native_w8a16", [a], [y], [w_init], value_info=value_infos)
    model = helper.make_model(
        graph,
        producer_name=f"matched_native_w8a16_{m}x{k}x{n}",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, out_dir / "matmul.onnx")

    activation_encodings = {"A": [_a16_encoding()], "Y": [_a16_encoding()]}
    for i in range(chain - 1):
        activation_encodings[f"Y_{i}"] = [_a16_encoding()]
    overrides = {
        "activation_encodings": activation_encodings,
        "param_encodings": {"W": [_w8_encoding()]},
    }
    (out_dir / "quant_overrides.json").write_text(json.dumps(overrides, indent=2), encoding="utf-8")

    runtime_dir = out_dir / "runtime_inputs_native"
    runtime_dir.mkdir(exist_ok=True)
    a_u16.astype("<u2", copy=False).tofile(runtime_dir / "A.raw")
    (out_dir / "runtime_input_list.txt").write_text("A:=runtime_inputs_native/A.raw\n", encoding="utf-8")
    native_io = {
        "input_name": "A",
        "output_name": "Y",
        "native_input": "runtime_inputs_native/A.raw",
        "runtime_input_list": "runtime_input_list.txt",
        "native_input_storage": "uint16_le",
        "native_input_bytes": int(a_u16.size * 2),
        "expected_native_output_storage": "uint16_le",
        "expected_native_output_bytes": int(m * n * 2),
        "shape_mkn": [m, k, n],
        "shape": [1, m, k],
        "output_shape": [1, m, n],
        "chain": chain,
        "source_custom_dir": str(custom_dir),
        "activation_encoding": _a16_encoding(),
        "weight_encoding": _w8_encoding(),
    }
    (out_dir / "native_io.json").write_text(json.dumps(native_io, indent=2), encoding="utf-8")
    np.save(out_dir / "wRaw_KN.npy", w_i8)
    print(f"matched native w8a16: {m}x{k}x{n}, chain={chain} -> {out_dir}")


if __name__ == "__main__":
    main()
