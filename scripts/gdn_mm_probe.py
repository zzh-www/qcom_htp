#!/usr/bin/env python3
"""Generate a standalone batched matmul [1,H,C,C] @ [1,H,C,C] -> [1,H,C,C] at u8xi8 (uint8 act x int8 wt,
the efficient HMX/native quantized primitive) to measure the compute-cycle cost of ONE "iterative
multiply" step of a matmul-based solve, for comparison vs the forward-substitution kernel.

Usage: gdn_mm_probe.py <outdir> <C> <H>
"""
import sys, os, json, numpy as np, onnx
from onnx import helper, TensorProto

outdir, C, H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
os.makedirs(outdir, exist_ok=True)
# deterministic small operands (values irrelevant to cycle count)
ii = (np.arange(H*C*C).reshape(1, H, C, C) % 17 - 8).astype(np.float32) * 0.03
jj = (np.arange(H*C*C).reshape(1, H, C, C) % 13 - 6).astype(np.float32) * 0.05
ii.tofile(os.path.join(outdir, "A.raw")); jj.tofile(os.path.join(outdir, "B.raw"))
io = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, H, C, C])
mm = helper.make_node("MatMul", ["A", "B"], ["Cout"], name="mm0")
m = helper.make_model(helper.make_graph([mm], "mm", [io("A"), io("B")], [io("Cout")]),
                      opset_imports=[helper.make_opsetid("", 17)])
onnx.save(m, os.path.join(outdir, "mm.onnx"))
aA = max(abs(ii).max(), 1e-6); aB = max(abs(jj).max(), 1e-6); aC = aA*aB*C
# u8 x i8: A uint8 (asymmetric), B int8 (symmetric), out uint16
json.dump({"version": "2.0.0", "encodings": [
    {"name": "A", "output_dtype": "uint8",  "y_scale": 2*aA/255.0, "y_zero_point": 128},
    {"name": "B", "output_dtype": "int8",   "y_scale": aB/127.0},
    {"name": "Cout", "output_dtype": "uint16", "y_scale": 2*aC/65535.0, "y_zero_point": 32768}]},
    open(os.path.join(outdir, "ovr.json"), "w"))
print(f"C={C} H={H}: matmul [1,{H},{C},{C}]@[1,{H},{C},{C}] u8xi8 -> mm.onnx")
