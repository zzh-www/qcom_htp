#!/usr/bin/env python3
"""Generate the standalone A->T solve graphs at a given batch B (= head-solves ~= chunks*heads), for the
HVX-op vs HMX-squaring scaling comparison (prefill-throughput question: HVX saturates at scale, HMX idle).

Emits into <outdir>:  A.raw (B*64*64 fp32, real strictly-lower tiled),  solve.onnx + ovr_solve.json
(GdnSolve custom op),  sq.onnx + ovr_sq.json (squaring native matmuls).

Usage: gdn_scaling_probe.py <outdir> <B> <ref_A.raw>
"""
import sys, json, numpy as np, onnx
from onnx import helper, TensorProto

C, BL = 64, 16
outdir, B, refA = sys.argv[1], int(sys.argv[2]), sys.argv[3]
import os; os.makedirs(outdir, exist_ok=True)

# --- A[1,B,64,64]: tile the real 32-head strictly-lower A to B blocks ---
a0 = np.fromfile(refA, dtype=np.float32).reshape(-1, C, C)
A = np.stack([a0[i % a0.shape[0]] for i in range(B)]).astype(np.float32)
for k in range(B): A[k] = np.tril(A[k], -1)                     # enforce strictly-lower
A.reshape(1, B, C, C).tofile(os.path.join(outdir, "A.raw"))
sA = max(abs(A).max() / 32767.0, 1e-12)
sT = 2.0 / 32767.0

def io(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

# --- solve.onnx : A -> GdnSolve -> T ---
gs = helper.make_node("GdnSolve", ["A"], ["T"], name="GdnSolve_0", domain="gdn")
g = helper.make_graph([gs], "solve", [io("A", [1, B, C, C])], [io("T", [1, B, C, C])])
ms = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("gdn", 1)])
onnx.save(ms, os.path.join(outdir, "solve.onnx"))
json.dump({"version": "2.0.0", "encodings": [
    {"name": "A", "output_dtype": "int16", "y_scale": sA},
    {"name": "T", "output_dtype": "int16", "y_scale": sT}]},
    open(os.path.join(outdir, "ovr_solve.json"), "w"))

# --- sq.onnx : A -> squaring native matmuls -> T ; compute per-tensor scales by numpy squaring ---
def q16(M): s = max(abs(M).max()/32767.0, 1e-12); return np.round(M/s).clip(-32767,32767)*s, s
eye = np.eye(C)
P = A.copy(); T = eye + A
rng = {"A": sA, "sq_T0": max(abs(T).max()/32767.0,1e-12)}
nodes = [helper.make_node("Add", ["A", "sq_I"], ["sq_T0"], name="sq_T0")]
Tn, Pn = "sq_T0", "A"
for k in range(1, 6):
    Pc, _ = q16(P); PP = Pc @ Pc; Pq, sP = q16(PP); P = Pq
    Tc, _ = q16(T); TP = Tc @ P; T2 = T + TP; Tq, sTk = q16(T2); T = Tq
    Pk, TPn = f"sq_P{k}", f"sq_TP{k}"; Tk = "T" if k == 5 else f"sq_T{k}"
    nodes += [helper.make_node("MatMul", [Pn, Pn], [Pk], name="sq_"+Pk),
              helper.make_node("MatMul", [Tn, Pk], [TPn], name="sq_"+TPn),
              helper.make_node("Add", [Tn, TPn], [Tk], name="sq_"+Tk)]
    rng[Pk] = sP; rng[TPn] = max(abs(TP).max()/32767.0,1e-12); rng[Tk] = sTk
    Tn, Pn = Tk, Pk
I_init = helper.make_tensor("sq_I", TensorProto.FLOAT, [C, C], eye.astype(np.float32).flatten())
g2 = helper.make_graph(nodes, "sq", [io("A", [1, B, C, C])], [io("T", [1, B, C, C])], [I_init])
m2 = helper.make_model(g2, opset_imports=[helper.make_opsetid("", 17)])
onnx.save(m2, os.path.join(outdir, "sq.onnx"))
# every MatMul in[1] -> int8 (overflow-safe over 64-contraction); rest int16
i8 = {n.input[1] for n in nodes if n.op_type == "MatMul"}
encs = []
for nm, s in rng.items():
    if nm in i8: encs.append({"name": nm, "output_dtype": "int8", "y_scale": max(abs(s*32767/127),1e-12)})
    else:        encs.append({"name": nm, "output_dtype": "int16", "y_scale": s})
for n in nodes:
    for t in list(n.input)+list(n.output):
        if t and t != "sq_I" and t not in {e["name"] for e in encs}:
            encs.append({"name": t, "output_dtype": "int16", "y_scale": sT})
json.dump({"version": "2.0.0", "encodings": encs}, open(os.path.join(outdir, "ovr_sq.json"), "w"))
print(f"B={B}: A absmax {abs(A).max():.3f}  solve.onnx + sq.onnx ({len(nodes)} nodes) written to {outdir}")
