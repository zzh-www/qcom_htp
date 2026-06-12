#!/usr/bin/env python3
"""Generate a standalone int16 (u16 act x i16 wt) QNN MatMul case at arbitrary
shapes, to inspect which HTP op QNN lowers an int16 matmul to.

Usage: mm_int16_probe.py <outdir> <A_shape> <B_shape>
  shapes are comma-separated, e.g.  1,1,64,64   or  1,8,8,64
  B may broadcast over leading (batch) dims per ONNX MatMul rules.

Emits: mm.onnx (float source, single MatMul A,B->Cout), A.raw, B.raw (float32),
ovr.json (v2.0.0 output_dtype schema: A=uint16 zp32768, B=int16, Cout=uint16 zp32768).
"""
import sys, os, json
import numpy as np
import onnx
from onnx import helper, TensorProto

outdir = sys.argv[1]
A_shape = [int(x) for x in sys.argv[2].split(",")]
B_shape = [int(x) for x in sys.argv[3].split(",")]
DTYPE = sys.argv[4] if len(sys.argv) > 4 else "u16i16"   # u16i16 | u8i8
os.makedirs(outdir, exist_ok=True)

rng = np.random.default_rng(0)
A = (rng.integers(-8, 8, size=int(np.prod(A_shape))).reshape(A_shape).astype(np.float32)) * 0.03
B = (rng.integers(-6, 7, size=int(np.prod(B_shape))).reshape(B_shape).astype(np.float32)) * 0.05
A.tofile(os.path.join(outdir, "A.raw"))
B.tofile(os.path.join(outdir, "B.raw"))

Cf = np.matmul(A.astype(np.float64), B.astype(np.float64))  # numpy broadcasts batch dims
C_shape = list(Cf.shape)
K = A_shape[-1]

io = lambda n, s: helper.make_tensor_value_info(n, TensorProto.FLOAT, s)
mm = helper.make_node("MatMul", ["A", "B"], ["Cout"], name="mm0")
m = helper.make_model(
    helper.make_graph([mm], "mm",
                      [io("A", A_shape), io("B", B_shape)],
                      [io("Cout", C_shape)]),
    opset_imports=[helper.make_opsetid("", 17)])
onnx.save(m, os.path.join(outdir, "mm.onnx"))

aA = max(abs(A).max(), 1e-6)
aB = max(abs(B).max(), 1e-6)
aC = max(abs(Cf).max(), 1e-6)
if DTYPE == "u8i8":
    enc = [
        {"name": "A",    "output_dtype": "uint8", "y_scale": 2 * aA / 255.0, "y_zero_point": 128},
        {"name": "B",    "output_dtype": "int8",  "y_scale": aB / 127.0},
        {"name": "Cout", "output_dtype": "uint16","y_scale": 2 * aC / 65535.0, "y_zero_point": 32768}]
else:  # u16i16: A uint16 (asym, zp 32768), B int16 (sym), Cout uint16
    enc = [
        {"name": "A",    "output_dtype": "uint16", "y_scale": 2 * aA / 65535.0, "y_zero_point": 32768},
        {"name": "B",    "output_dtype": "int16",  "y_scale": aB / 32767.0},
        {"name": "Cout", "output_dtype": "uint16", "y_scale": 2 * aC / 65535.0, "y_zero_point": 32768}]
json.dump({"version": "2.0.0", "encodings": enc}, open(os.path.join(outdir, "ovr.json"), "w"))
print(f"MatMul A{A_shape} @ B{B_shape} -> Cout{C_shape}  K={K}  {DTYPE} -> {outdir}/mm.onnx")
