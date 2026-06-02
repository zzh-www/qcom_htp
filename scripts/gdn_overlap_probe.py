#!/usr/bin/env python3
"""Overlap probe for #2: does QNN run the HVX GdnSolve and an HMX MatMul that consumes its output
CONCURRENTLY (pipelined across the batch), or serially?  Emits a combined graph
    A[1,B,C,C] -> GdnSolve -> T -> MatMul(T, V) -> P
plus the two reference graphs (solve-only, matmul-only) at the SAME B,C so their walls are comparable.
If wall(combined) ~ max(solve, matmul) -> the units overlap; if ~ solve+matmul -> serial.

Everything is sized to fit VTCM (small B,C) so the HVX->HMX handoff of T can stay on-chip (the
condition under which #2's pipeline wins). T is uint16 (the GdnSolve op's QUInt16 form, no Cast);
V is int8 so T(u16)xV(i8) maps to the HMX w8a16 primitive.

Usage: gdn_overlap_probe.py <outdir> <C> <B> <ref_A.raw>
"""
import sys, os, json, numpy as np, onnx
from onnx import helper, TensorProto

outdir, C, B, refA = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.makedirs(outdir, exist_ok=True)
a0 = np.fromfile(refA, dtype=np.float32).reshape(-1, 64, 64)
reps = (C + 63) // 64
def mk(i):
    big = np.tile(a0[i % a0.shape[0]], (reps, reps))[:C, :C]
    return np.tril(big * (0.7 if C > 64 else 1.0), -1)
A = np.stack([mk(i) for i in range(B)]).astype(np.float32).reshape(1, B, C, C)
A.tofile(os.path.join(outdir, "A.raw"))
T_in = np.stack([np.linalg.inv(np.eye(C) - A[0, i]) for i in range(B)]).astype(np.float32).reshape(1, B, C, C)
T_in.tofile(os.path.join(outdir, "T_in.raw"))                # direct T for the matmul-only ref
V = ((np.arange(B*C*C).reshape(1, B, C, C) % 11 - 5) * 0.07).astype(np.float32)
V.tofile(os.path.join(outdir, "V.raw"))

io = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, B, C, C])
sA = max(abs(A).max() / 32767.0, 1e-12); sT = 2.0 / 32767.0
sV = max(abs(V).max() / 127.0, 1e-12); sP = sT * sV * C
encA = {"name": "A", "output_dtype": "uint16", "y_scale": sA, "y_zero_point": 32768}
encT = {"name": "T", "output_dtype": "uint16", "y_scale": sT, "y_zero_point": 32768}
encV = {"name": "V", "output_dtype": "int8",   "y_scale": sV}
encP = {"name": "P", "output_dtype": "uint16", "y_scale": 2*sP/65535.0, "y_zero_point": 32768}
ov = lambda *e: {"version": "2.0.0", "encodings": list(e)}

def save(name, graph, enc):
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("gdn", 1)])
    onnx.save(m, os.path.join(outdir, name + ".onnx"))
    json.dump(ov(*enc), open(os.path.join(outdir, name + ".ovr.json"), "w"))

gs  = helper.make_node("GdnSolve", ["A"], ["T"], name="GdnSolve_0", domain="gdn")
mm  = helper.make_node("MatMul", ["T", "V"], ["P"], name="mm0")
save("solve",    helper.make_graph([gs],     "solve",    [io("A")],          [io("T")]),          [encA, encT])
save("mm",       helper.make_graph([mm],     "mm",       [io("T"), io("V")], [io("P")]),          [dict(encT, name="T"), encV, encP])
save("combined", helper.make_graph([gs, mm], "combined", [io("A"), io("V")], [io("P")]),          [encA, encT, encV, encP])

# --- INDEPENDENT-chains probe: GdnSolve(A)->T  ||  MatMul(X,Y)->Z, NO data dep between them.
# Tests whether QNN's background-HMX worker / runlist parallelism overlaps an independent plugin-HVX
# op with an independent HMX matmul (the only inter-op concurrency path left for plugin ops, since
# supertile fusion excludes plugin ops). X is a runtime input (no const-fold); Y=V (int8) so X(u16)xY(i8)->HMX.
X = ((np.arange(B*C*C).reshape(1, B, C, C) % 7 - 3) * 0.05 + 0.4).astype(np.float32)
X.tofile(os.path.join(outdir, "X.raw"))
sX = max(abs(X).max() / 127.0, 1e-12)   # int8 act: avoids the uint16-from-DDR pathological matmul path
encX = {"name": "X", "output_dtype": "int8", "y_scale": sX}
encZ = {"name": "Z", "output_dtype": "uint16", "y_scale": 2*sX*sV*C/65535.0, "y_zero_point": 32768}
gs2 = helper.make_node("GdnSolve", ["A"], ["T"], name="GdnSolve_i", domain="gdn")
mm2 = helper.make_node("MatMul", ["X", "Y"], ["Z"], name="mm_i")
save("indep_native", helper.make_graph([gs2, mm2], "indep_native",
     [io("A"), io("X"), io("Y")], [io("T"), io("Z")]),
     [encA, encT, encX, dict(encV, name="Y"), encZ])
print(f"C={C} B={B}: solve/mm/combined/indep_native  (T_in.raw for mm-only ref)")
