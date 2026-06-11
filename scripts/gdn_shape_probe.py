#!/usr/bin/env python3
"""Generate the standalone A->GdnSolve->T graph at a given matrix size C (16/32/64), 32 heads, to probe
which C is most HVX-efficient (C=64 -> 2 full HVX vectors; C=32 -> 1 full vector; C=16 -> half a vector,
50% lane waste).  Emits A.raw [1,32,C,C], T_ref.raw, solve.onnx, ovr_solve.json into <outdir>.

Usage: gdn_shape_probe.py <outdir> <C> <ref_A.raw>
"""
import sys, os, json, numpy as np, onnx
from onnx import helper, TensorProto

outdir, C, refA = sys.argv[1], int(sys.argv[2]), sys.argv[3]
H = 32
os.makedirs(outdir, exist_ok=True)
a0 = np.fromfile(refA, dtype=np.float32).reshape(-1, 64, 64)
reps = (C + 63) // 64
def mk(i):                                                  # tile the 64x64 ref up/down to C x C, strictly-lower
    big = np.tile(a0[i % a0.shape[0]], (reps, reps))[:C, :C]
    return np.tril(big * (0.7 if C > 64 else 1.0), -1)       # damp for C>64 so |T| stays bounded
A = np.stack([mk(i) for i in range(H)]).astype(np.float32)   # [H,C,C] strictly-lower
A.reshape(1, H, C, C).tofile(os.path.join(outdir, "A.raw"))
Tref = np.stack([np.linalg.inv(np.eye(C) - A[h]) for h in range(H)]).astype(np.float32)
Tref.reshape(1, H, C, C).tofile(os.path.join(outdir, "T_ref.raw"))
sA = max(abs(A).max() / 32767.0, 1e-12); sT = 2.0 / 32767.0

io = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, H, C, C])
gs = helper.make_node("GdnSolve", ["A"], ["T"], name="GdnSolve_0", domain="gdn")
m = helper.make_model(helper.make_graph([gs], "solve", [io("A")], [io("T")]),
                      opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("gdn", 1)])
onnx.save(m, os.path.join(outdir, "solve.onnx"))
json.dump({"version": "2.0.0", "encodings": [
    {"name": "A", "output_dtype": "int16", "y_scale": sA},
    {"name": "T", "output_dtype": "int16", "y_scale": sT}]},
    open(os.path.join(outdir, "ovr_solve.json"), "w"))
print(f"C={C}: A[1,{H},{C},{C}] absmax {abs(A).max():.3f}  ({'2 vecs' if C>32 else '1 vec' if C==32 else 'half vec'})")
