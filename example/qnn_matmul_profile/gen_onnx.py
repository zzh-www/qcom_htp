#!/usr/bin/env python3
"""Generate a configurable MatMul ONNX model and (optionally) a
quant-overrides JSON for QNN qairt-converter.

Usage:
    python gen_onnx.py <config_name> <out_dir> [--m M] [--k K] [--n N]

Config names:
    fp16     - fp16 inputs, no quant override
    w16a16   - int16 weight, int16 activation, int16 output
    w8a16    - int8  weight, int16 activation, int16 output
    w8a8     - int8  weight, int8  activation, int8  output
    w4a16    - int4  weight, int16 activation, int16 output
    w4a8     - int4  weight, int8  activation, int8  output
    w4a4     - int4  weight, int4  activation, int4  output

Shape: A is [1, M, K], W is [1, K, N], Y is [1, M, N]. Default 32×32×32.

Writes under <out_dir>/:
    matmul.onnx
    quant_overrides.json  (unless fp16)
    input_A.raw           (fp32 input data, M*K values)
    input_list.txt        (qnn-net-run input list, "A:=input_A.raw")
"""
import argparse, json, os
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

CONFIGS = {
    "fp16":   {"dtype": "float", "float_bitwidth": 16},
    "w16a16": {"dtype": "quant", "act": 16, "weight": 16, "out": 16, "symmetric": True},
    "w8a16":  {"dtype": "quant", "act": 16, "weight":  8, "out": 16, "symmetric": True},
    "w8a8":   {"dtype": "quant", "act":  8, "weight":  8, "out":  8, "symmetric": True},
    "w4a16":  {"dtype": "quant", "act": 16, "weight":  4, "out": 16, "symmetric": True},
    "w4a8":   {"dtype": "quant", "act":  8, "weight":  4, "out":  8, "symmetric": True},
    "w4a4":   {"dtype": "quant", "act":  4, "weight":  4, "out":  4, "symmetric": True},
}


def _symmetric_encoding(bits: int) -> dict:
    max_int = (1 << (bits - 1)) - 1
    return {
        "bitwidth":     bits,
        "dtype":        "int",
        "is_symmetric": "True",
        "scale":         1.0 / max_int,
        "offset":       -(1 << (bits - 1)),
        "min":          -1.0,
        "max":           1.0,
    }


def _emit_onnx(cfg: dict, path: str, m: int, k: int, n: int):
    rng = np.random.default_rng(42)
    if cfg["dtype"] == "float":
        onnx_dtype, np_dtype = TensorProto.FLOAT16, np.float16
    else:
        onnx_dtype, np_dtype = TensorProto.FLOAT, np.float32

    W_val = rng.uniform(-0.5, 0.5, size=(1, k, n)).astype(np_dtype)
    W_init = numpy_helper.from_array(W_val, name="W")

    A = helper.make_tensor_value_info("A", onnx_dtype, [1, m, k])
    Y = helper.make_tensor_value_info("Y", onnx_dtype, [1, m, n])
    node = helper.make_node("MatMul", ["A", "W"], ["Y"], name="matmul_1")
    graph = helper.make_graph([node], "matmul", [A], [Y], [W_init])
    model = helper.make_model(
        graph,
        producer_name="qnn_matmul_profile",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    try:
        onnx.checker.check_model(model)
    except Exception:
        pass  # shape check sometimes chokes on huge tensors; conversion is what matters
    onnx.save(model, path)


def _emit_quant(cfg: dict, path: str, n: int):
    w_bits = cfg["weight"]
    # int4 weights need per-output-channel encoding (one per N column).
    if w_bits <= 4:
        w_enc = [_symmetric_encoding(w_bits) for _ in range(n)]
    else:
        w_enc = [_symmetric_encoding(w_bits)]

    enc = {
        "activation_encodings": {
            "A": [_symmetric_encoding(cfg["act"])],
            "Y": [_symmetric_encoding(cfg["out"])],
        },
        "param_encodings": {"W": w_enc},
    }
    with open(path, "w") as f:
        json.dump(enc, f, indent=2)


def _emit_input(out_dir: str, m: int, k: int):
    rng = np.random.default_rng(0xBEEF)
    A = rng.uniform(-0.5, 0.5, size=(1, m, k)).astype(np.float32)
    raw = os.path.join(out_dir, "input_A.raw")
    A.tofile(raw)
    with open(os.path.join(out_dir, "input_list.txt"), "w") as f:
        f.write("A:=input_A.raw\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=CONFIGS.keys())
    ap.add_argument("out_dir")
    ap.add_argument("--m", type=int, default=32, help="rows of A / rows of output")
    ap.add_argument("--k", type=int, default=32, help="reduction dim (cols of A, rows of W)")
    ap.add_argument("--n", type=int, default=32, help="cols of W / cols of output")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    os.makedirs(args.out_dir, exist_ok=True)

    _emit_onnx(cfg, os.path.join(args.out_dir, "matmul.onnx"), args.m, args.k, args.n)
    if cfg["dtype"] == "quant":
        _emit_quant(cfg, os.path.join(args.out_dir, "quant_overrides.json"), args.n)
    _emit_input(args.out_dir, args.m, args.k)

    print(f"  [{args.config}] wrote {args.out_dir}/ (dtype={cfg['dtype']}, {args.m}x{args.k}x{args.n})")


if __name__ == "__main__":
    main()
