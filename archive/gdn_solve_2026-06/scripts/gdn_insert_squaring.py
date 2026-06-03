#!/usr/bin/env python3
"""Graph surgery (HMX prototype): replace the GDN triangular-solve subgraph (A -> T) in gdn_q.onnx with
a SQUARING chain of NATIVE MatMuls — T=(I-A)^-1 = prod_k (I + A^(2^k)), A strictly-lower 64x64 (nilpotent,
A^64=0 -> 6 factors exact).  All dense 64x64 matmuls -> QNN maps them to HMX and overlaps the HVX rest
(QNN co-schedules HMX/HVX ~81%).  ~16 nodes vs the old 363, so no dispatch blow-up.

Recurrence:  T0 = I + A ;  P0 = A ;  for k in 1..steps-1:  Pk = P(k-1)@P(k-1);  Tk = T(k-1) + T(k-1)@Pk
De-risked in scripts/gdn_solve_squaring_probe.py: int16 requant between matmuls -> U/W identical to the
forward-subst op (both at the int8 downstream ceiling).

Usage: gdn_insert_squaring.py in.onnx out.onnx [--A /Mul_9_output_0] [--T /Add_32_output_0] [--steps 6]
"""
import argparse, sys
import numpy as np, onnx
from onnx import helper, TensorProto

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--A", default="/Mul_9_output_0")
ap.add_argument("--T", default="/Add_32_output_0")
ap.add_argument("--C", type=int, default=64)
ap.add_argument("--steps", type=int, default=6)           # log2(64)=6 factors -> exact
a = ap.parse_args()

m = onnx.load(a.inp); g = m.graph
prod = {o: n for n in g.node for o in n.output}

def ancestors(start):
    seen, stack = set(), [start]
    while stack:
        t = stack.pop(); n = prod.get(t)
        if n is None or id(n) in seen: continue
        seen.add(id(n))
        for i in n.input:
            if i: stack.append(i)
    return seen

solve_ids = ancestors(a.T) - ancestors(a.A)
keep = [n for n in g.node if id(n) not in solve_ids]
n_removed = len(g.node) - len(keep)

# identity constant [C,C], broadcast over the leading (batch/head) dims in Add
eye = helper.make_tensor("sq_I", TensorProto.FLOAT, [a.C, a.C], np.eye(a.C, dtype=np.float32).flatten())
g.initializer.append(eye)

nn = []
def mm(x, y, o):  nn.append(helper.make_node("MatMul", [x, y], [o], name="sq_"+o.strip("/")))
def add(x, y, o): nn.append(helper.make_node("Add", [x, y], [o], name="sq_"+o.strip("/")))

T_prev = "sq_T0"; add(a.A, "sq_I", T_prev)                # T0 = A + I
P_prev = a.A                                              # P0 = A
for k in range(1, a.steps):
    Pk = f"sq_P{k}"; mm(P_prev, P_prev, Pk)               # Pk = P(k-1) @ P(k-1) = A^(2^k)
    TP = f"sq_TP{k}"; mm(T_prev, Pk, TP)                  # T(k-1) @ Pk
    Tk = a.T if k == a.steps - 1 else f"sq_T{k}"
    add(T_prev, TP, Tk)                                   # Tk = T(k-1) + T(k-1)@Pk
    T_prev, P_prev = Tk, Pk

# rebuild node list: keep-nodes in order, splice the squaring right after A is produced (topo-safe)
del g.node[:]
inserted = False
for n in keep:
    g.node.append(n)
    if not inserted and a.A in n.output:
        g.node.extend(nn); inserted = True
if not inserted:
    for x in reversed(nn): g.node.insert(0, x)

# drop graph inputs orphaned by removing the old solve (e.g. sel0..3)
used = {i for n in g.node for i in n.input if i}
kept = [vi for vi in g.input if vi.name in used]
del g.input[:]; g.input.extend(kept)

onnx.save(m, a.out)
print(f"removed {n_removed} solve nodes; inserted squaring chain "
      f"({a.steps-1} squarings, {2*(a.steps-1)} matmuls) {a.A} -> {a.T}")
