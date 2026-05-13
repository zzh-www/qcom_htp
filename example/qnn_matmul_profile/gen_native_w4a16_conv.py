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


def _make_activation_u16(
    m: int,
    k: int,
    idx: int = 0,
    activation_mode: str = "default",
    activation_k: int = 0,
) -> np.ndarray:
    if activation_mode == "zp":
        return np.full((m, k), 32768, dtype=np.uint16)
    if activation_mode == "k_impulse":
        act = np.full((m, k), 32768, dtype=np.uint16)
        act[:, activation_k % k] = np.uint16(32769)
        return act
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
    parser.add_argument("--chain", type=int, default=1)
    parser.add_argument("--activation-mode", choices=["default", "zp", "k_impulse"], default="default")
    parser.add_argument("--activation-k", type=int, default=0)
    args = parser.parse_args()
    if args.chain < 1:
        raise ValueError("--chain must be >= 1")
    if args.chain != 1 and args.k != args.n:
        raise ValueError("native W4A16 Conv chain requires K == N between Conv nodes")

    os.makedirs(args.out_dir, exist_ok=True)
    native_dir = os.path.join(args.out_dir, "runtime_inputs_native")
    os.makedirs(native_dir, exist_ok=True)

    act_mk = _make_activation_u16(
        args.m,
        args.k,
        activation_mode=args.activation_mode,
        activation_k=args.activation_k,
    )
    act_conv_q = act_mk.T.reshape(1, args.k, 1, args.m)
    act_conv_f = _dequant_u16(act_conv_q)
    act_conv_f.astype("<f4").tofile(os.path.join(args.out_dir, "input_A.raw"))
    act_conv_q.astype("<u2").tofile(os.path.join(native_dir, "A.raw"))

    w_kn = _make_weight_i4(args.k, args.n)
    w_conv = (w_kn.T.astype(np.float32) / 7.0).reshape(args.n, args.k, 1, 1)

    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, args.k, 1, args.m])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, args.n, 1, args.m])
    w_init = numpy_helper.from_array(w_conv, name="W")
    nodes = []
    value_infos = []
    prev = "A"
    for i in range(args.chain):
        out_name = "Y" if i == args.chain - 1 else f"Y_{i}"
        nodes.append(helper.make_node(
            "Conv",
            [prev, "W"],
            [out_name],
            name=f"conv1x1_{i}" if args.chain > 1 else "conv1x1",
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        ))
        if i < args.chain - 1:
            value_infos.append(helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [1, args.n, 1, args.m]))
        prev = out_name
    graph = helper.make_graph(nodes, "native_w4a16_conv", [a], [y], [w_init], value_info=value_infos)
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
    for i in range(args.chain - 1):
        overrides["activation_encodings"][f"Y_{i}"] = [_a16_encoding()]
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
                "chain": args.chain,
                "activation_mode": args.activation_mode,
                "activation_k": args.activation_k,
                "activation_encoding": _a16_encoding(),
                "weight_encoding": _w4_encoding(),
            },
            f,
            indent=2,
        )
    np.save(os.path.join(args.out_dir, "actRaw_u16.npy"), act_mk)
    np.save(os.path.join(args.out_dir, "wRaw_KN.npy"), w_kn)
    print(f"  wrote native W4A16 Conv ref: {args.out_dir} (chain={args.chain})")


if __name__ == "__main__":
    main()
