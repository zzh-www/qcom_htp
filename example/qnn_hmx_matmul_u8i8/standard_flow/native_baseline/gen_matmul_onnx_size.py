#!/usr/bin/env python3
"""Phase A native: parametric size variant of gen_matmul_onnx.py.

Usage:
    python gen_matmul_onnx_size.py --size 256 --out s256_w8a8

Produces a self-contained directory with model.onnx, calibration files,
runtime u8 input, and the qairt input lists.
"""
import argparse, json, os, sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=None, help="cube shape; sets M=K=N")
ap.add_argument("--M", type=int, default=None)
ap.add_argument("--K", type=int, default=None)
ap.add_argument("--N", type=int, default=None)
ap.add_argument("--out", required=True, help="output subdir relative to this script")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, args.out)
os.makedirs(OUT, exist_ok=True)
if args.size is not None:
    M = K = N = args.size
else:
    assert args.M and args.K and args.N, "must pass --size or --M --K --N"
    M, K, N = args.M, args.K, args.N

np.random.seed(0xB17E)

wRaw_i8 = np.array([((i * 13) % 15) - 7 for i in range(K * N)], dtype=np.int8).reshape(K, N)
w_fp32  = wRaw_i8.astype(np.float32) / 127.0

aRaw_u8 = np.array([(i * 37) & 0xFF for i in range(M * K)], dtype=np.uint8).reshape(1, M, K)
a_fp32  = aRaw_u8.astype(np.float32) / 255.0

input_info  = helper.make_tensor_value_info("A",  TensorProto.FLOAT, [1, M, K])
output_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, M, N])
w_init = numpy_helper.from_array(w_fp32, name="W")
node   = helper.make_node("MatMul", inputs=["A", "W"], outputs=["Y"], name="MatMul_0")
graph  = helper.make_graph([node], "model", [input_info], [output_info], [w_init])
model  = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, os.path.join(OUT, "model.onnx"))

calib = os.path.join(OUT, "inputs_fp32"); os.makedirs(calib, exist_ok=True)
a_fp32.astype("<f4").tofile(os.path.join(calib, "A.raw"))
a2 = (np.random.randint(0, 256, size=(1, M, K)).astype(np.float32) / 255.0)
a2.astype("<f4").tofile(os.path.join(calib, "input_1.raw"))
with open(os.path.join(OUT, "input_list.txt"), "w") as f:
    f.write("A:=inputs_fp32/A.raw\n")
    f.write("A:=inputs_fp32/input_1.raw\n")

u8 = os.path.join(OUT, "runtime_inputs_u8"); os.makedirs(u8, exist_ok=True)
aRaw_u8.tofile(os.path.join(u8, "a.raw"))
with open(os.path.join(OUT, "runtime_input_list.txt"), "w") as f:
    f.write("A:=runtime_inputs_u8/a.raw\n")

with open(os.path.join(OUT, "native_io.json"), "w") as f:
    json.dump(
        {
            "input_name": "A",
            "output_name": "Y",
            "native_input": "runtime_inputs_u8/a.raw",
            "runtime_input_list": "runtime_input_list.txt",
            "native_input_storage": "uint8",
            "native_input_bytes": int(M * K),
            "expected_native_output_storage": "uint8",
            "expected_native_output_bytes": int(M * N),
            "shape": [1, M, K],
            "output_shape": [1, M, N],
        },
        f,
        indent=2,
    )

print(f"  size={M} -> {OUT}/model.onnx + inputs")
