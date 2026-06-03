#!/usr/bin/env python3
"""Generate a CHAIN of N GdnSolve ops in series (chain8-style), to read steady per-op cycles.

A -> GdnSolve_0 -> t0 -> GdnSolve_1 -> t1 -> ... -> GdnSolve_{N-1} -> T   (all [1,H,C,C]).
Chaining solve(solve(...)) is NOT semantically meaningful (each op just runs its kernel on the prior
output) — the point is PERF: N back-to-back instances let the decoder separate op[0] (cold) from
op[1..N-1] (steady) per-op cycles, exactly like example/qnn_hmx_matmul_u8i8/.../gen_matmul_onnx_chain.py.

Emits into <outdir>: A.raw [1,H,C,C], solve_chain.onnx, ovr_chain.json (v2 uint16 encodings for every
tensor in the chain).

Usage: gdn_solve_chain_probe.py <outdir> <C> <N> <ref_A.raw> [H]
"""
import sys, os, json, numpy as np, onnx
from onnx import helper, TensorProto

outdir, C, N, refA = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
H = int(sys.argv[5]) if len(sys.argv) > 5 else 32
os.makedirs(outdir, exist_ok=True)
a0 = np.fromfile(refA, dtype=np.float32).reshape(-1, 64, 64)
reps = (C + 63) // 64
def mk(i):
    big = np.tile(a0[i % a0.shape[0]], (reps, reps))[:C, :C]
    return np.tril(big * (0.7 if C > 64 else 1.0), -1)
A = np.stack([mk(i) for i in range(H)]).astype(np.float32)
A.reshape(1, H, C, C).tofile(os.path.join(outdir, "A.raw"))
sA = max(abs(A).max() / 32767.0, 1e-12); sT = 2.0 / 32767.0

io = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, H, C, C])
nodes, names = [], []
prev = "A"
for i in range(N):
    out = "T" if i == N - 1 else f"t{i}"
    nodes.append(helper.make_node("GdnSolve", [prev], [out], name=f"GdnSolve_{i}", domain="gdn"))
    names.append(out); prev = out
m = helper.make_model(helper.make_graph(nodes, "solve_chain", [io("A")], [io("T")]),
                      opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("gdn", 1)])
onnx.save(m, os.path.join(outdir, "solve_chain.onnx"))

enc = [{"name": "A", "output_dtype": "uint16", "y_scale": sA, "y_zero_point": 32768}]
for n in names:   # every intermediate + final T uses the T scale (uint16 midpoint)
    enc.append({"name": n, "output_dtype": "uint16", "y_scale": sT, "y_zero_point": 32768})
json.dump({"version": "2.0.0", "encodings": enc}, open(os.path.join(outdir, "ovr_chain.json"), "w"))
print(f"chain N={N} C={C} H={H}: A[1,{H},{C},{C}] absmax {abs(A).max():.3f}  nodes={[n.name for n in nodes]}")
