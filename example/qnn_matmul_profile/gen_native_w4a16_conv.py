#!/usr/bin/env python3
"""Generate the QNN-native W4A16 Conv1x1 reference graph.

The graph mirrors QNN's native quantized-model entry convention: the ONNX public
input/output are float tensors, and quantization overrides force QNN HTP to
lower into the native W4 ConvLayer_s1 path.  This float ONNX surface is not the
comparison runtime contract.  Standard reference runs must use the emitted u16
native input with qnn-net-run --use_native_input_files and pull the u16 native
output with --use_native_output_files.
"""

import argparse
import json
import os

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


def _w4_encoding() -> dict:
    return {
        "bitwidth": 4,
        "dtype": "int",
        "is_symmetric": "True",
        "scale": 1.0 / 7.0,
        "offset": -8,
        "min": -8.0 / 7.0,
        "max": 1.0,
    }


def _make_activation_u16(m: int, k: int, idx: int = 0) -> np.ndarray:
    seed = (idx + 1) * 374761393
    return np.array(
        [((i * 37 + seed) & 0xFFFF) for i in range(m * k)],
        dtype=np.uint16,
    ).reshape(m, k)


def _make_weight_i4(k: int, n: int) -> np.ndarray:
    qmax = 7
    k_idx, n_idx = np.meshgrid(np.arange(k), np.arange(n), indexing="ij")
    return (((k_idx * 31 + n_idx * 13) % (2 * qmax + 1)) - qmax).astype(np.int8)


def _dequant_u16(q: np.ndarray) -> np.ndarray:
    enc = _a16_encoding()
    return ((q.astype(np.int32) + int(enc["offset"])) * enc["scale"]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    native_dir = os.path.join(args.out_dir, "runtime_inputs_native")
    os.makedirs(native_dir, exist_ok=True)

    act_mk = _make_activation_u16(args.m, args.k)
    act_conv_q = act_mk.T.reshape(1, args.k, 1, args.m)
    act_conv_f = _dequant_u16(act_conv_q)
    act_conv_f.astype("<f4").tofile(os.path.join(args.out_dir, "input_A.raw"))
    act_conv_q.astype("<u2").tofile(os.path.join(native_dir, "A.raw"))

    w_kn = _make_weight_i4(args.k, args.n)
    w_conv = (w_kn.T.astype(np.float32) / 7.0).reshape(args.n, args.k, 1, 1)

    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, args.k, 1, args.m])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, args.n, 1, args.m])
    w_init = numpy_helper.from_array(w_conv, name="W")
    node = helper.make_node(
        "Conv",
        ["A", "W"],
        ["Y"],
        name="conv1x1",
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )
    graph = helper.make_graph([node], "native_w4a16_conv", [a], [y], [w_init])
    model = helper.make_model(
        graph,
        producer_name="qnn_native_w4a16_conv_ref",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, os.path.join(args.out_dir, "conv.onnx"))

    overrides = {
        "activation_encodings": {
            "A": [_a16_encoding()],
            "Y": [_a16_encoding()],
        },
        "param_encodings": {"W": [_w4_encoding()]},
    }
    with open(os.path.join(args.out_dir, "quant_overrides.json"), "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2)

    with open(os.path.join(args.out_dir, "input_list.txt"), "w", encoding="utf-8") as f:
        f.write("A:=input_A.raw\n")
    with open(os.path.join(args.out_dir, "runtime_input_list.txt"), "w", encoding="utf-8") as f:
        f.write("A:=runtime_inputs_native/A.raw\n")
    with open(os.path.join(args.out_dir, "native_io.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_name": "A",
                "output_name": "Y",
                "native_input": "runtime_inputs_native/A.raw",
                "runtime_input_list": "runtime_input_list.txt",
                "native_input_storage": "uint16_le",
                "native_input_bytes": int(act_conv_q.size * 2),
                "expected_native_output_storage": "uint16_le",
                "expected_native_output_bytes": int(args.n * args.m * 2),
                "conv_input_shape": [1, args.k, 1, args.m],
                "conv_output_shape": [1, args.n, 1, args.m],
                "logical_matmul_shape_mkn": [args.m, args.k, args.n],
                "activation_encoding": _a16_encoding(),
                "weight_encoding": _w4_encoding(),
            },
            f,
            indent=2,
        )
    np.save(os.path.join(args.out_dir, "actRaw_u16.npy"), act_mk)
    np.save(os.path.join(args.out_dir, "wRaw_KN.npy"), w_kn)
    print(f"  wrote native W4A16 Conv ref: {args.out_dir}")


if __name__ == "__main__":
    main()
