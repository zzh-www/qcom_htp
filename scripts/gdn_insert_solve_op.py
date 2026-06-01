#!/usr/bin/env python3
"""Graph surgery: replace the GDN triangular-solve subgraph (A -> T, the int8-matmul block forward
substitution) in gdn_q.onnx with a single custom node  GdnSolve(A) -> T  (domain "gdn"), backed by
the HTP op package example/gdn_native/solve_op.  Keeps the float solve_T_blocked path untouched (this
only rewrites the exported quant graph for the device).

Boundary tensors (from scripts/gdn_stage_map.py): A = /Mul_9_output_0, T = /Add_32_output_0.

Usage: gdn_insert_solve_op.py in.onnx out.onnx [--A /Mul_9_output_0] [--T /Add_32_output_0]
"""
import sys, argparse
import onnx
from onnx import helper

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--A", default="/Mul_9_output_0")
ap.add_argument("--T", default="/Add_32_output_0")
ap.add_argument("--domain", default="gdn")
ap.add_argument("--split", type=int, default=int(__import__("os").environ.get("GDN_SOLVE_SPLIT", "1")),
                help="N: emit N independent GdnSolve nodes over head chunks (graph-level parallelism) + Concat")
ap.add_argument("--override", default=None, help="v2 override json to patch with per-chunk encodings")
a = ap.parse_args()

m = onnx.load(a.inp)
g = m.graph
prod = {o: n for n in g.node for o in n.output}            # tensor -> producing node

def ancestors(start):
    """set of node ids (python id) that are backward-reachable producers of `start` tensor."""
    seen, stack = set(), [start]
    while stack:
        t = stack.pop()
        n = prod.get(t)
        if n is None or id(n) in seen:
            continue
        seen.add(id(n))
        for i in n.input:
            if i:
                stack.append(i)
    return seen

anc_T = ancestors(a.T)
anc_A = ancestors(a.A)                                      # everything up to & including A's producer
solve_ids = anc_T - anc_A                                   # nodes strictly between A and T (incl. T producer)
solve_nodes = [n for n in g.node if id(n) in solve_ids]

# sanity: every solve-node output must be consumed only inside the solve set, except T itself
solve_outs = {o for n in solve_nodes for o in n.output if o}
consumers = {}
for n in g.node:
    for i in n.input:
        consumers.setdefault(i, []).append(n)
leaks = []
for o in solve_outs:
    if o == a.T:
        continue
    for c in consumers.get(o, []):
        if id(c) not in solve_ids:
            leaks.append((o, c.name))
if leaks:
    print("ERROR: solve-internal tensors consumed outside the solve (cannot cleanly replace):")
    for o, c in leaks[:10]:
        print(f"   {o} -> {c}")
    sys.exit(1)

# build the replacement node(s): one GdnSolve, or N independent GdnSolve over head chunks + Concat
from onnx import TensorProto
keep = [n for n in g.node if id(n) not in solve_ids]
N = max(1, a.split)
new_nodes = []
if N == 1:
    new_nodes.append(helper.make_node("GdnSolve", [a.A], [a.T], name="GdnSolve_0", domain=a.domain))
else:
    # infer A's head-dim size (dim 1) via shape inference
    try:
        mi = onnx.shape_inference.infer_shapes(m)
        sh = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
              for vi in list(mi.graph.value_info) + list(mi.graph.input)}
        H = sh[a.A][1]
    except Exception:
        H = 32
    assert H % N == 0, f"heads {H} not divisible by split {N}"
    ch = H // N
    t_chunks = []
    for i in range(N):
        st = helper.make_tensor(f"gs_st{i}", TensorProto.INT64, [1], [i*ch])
        en = helper.make_tensor(f"gs_en{i}", TensorProto.INT64, [1], [(i+1)*ch])
        ax = helper.make_tensor(f"gs_ax{i}", TensorProto.INT64, [1], [1])     # slice head dim
        g.initializer.extend([st, en, ax])
        ai, ti = f"gs_A{i}", f"gs_T{i}"
        new_nodes.append(helper.make_node("Slice", [a.A, st.name, en.name, ax.name], [ai], name=f"gs_slice{i}"))
        new_nodes.append(helper.make_node("GdnSolve", [ai], [ti], name=f"GdnSolve_{i}", domain=a.domain))
        t_chunks.append(ti)
    new_nodes.append(helper.make_node("Concat", t_chunks, [a.T], axis=1, name="gs_concat"))

# place new node(s) right after A's producer (topo-safe: input A already produced)
del g.node[:]
inserted = False
for n in keep:
    g.node.append(n)
    if not inserted and a.A in n.output:
        g.node.extend(new_nodes); inserted = True
if not inserted:                                           # A is a graph input -> prepend
    for nn in reversed(new_nodes): g.node.insert(0, nn)

# patch the override: each per-chunk A_i / T_i tensor needs A's / T's int16 encoding
if a.override and N > 1:
    import json
    ov = json.load(open(a.override)); enc = {e["name"]: e for e in ov["encodings"]}
    eA, eT = enc.get(a.A), enc.get(a.T)
    add = []
    for i in range(N):
        if eA: add.append({**eA, "name": f"gs_A{i}"})
        if eT: add.append({**eT, "name": f"gs_T{i}"})
    ov["encodings"].extend(add); json.dump(ov, open(a.override, "w"))
    print(f"patched {a.override}: +{len(add)} per-chunk encodings")

# drop graph inputs that the surgery orphaned (e.g. sel0..3 used only by the old solve)
used = {i for n in g.node for i in n.input if i}
kept_inputs = [vi for vi in g.input if vi.name in used]
removed_inputs = [vi.name for vi in g.input if vi.name not in used]
del g.input[:]; g.input.extend(kept_inputs)

# register the custom domain opset
if all(op.domain != a.domain for op in m.opset_import):
    m.opset_import.append(helper.make_opsetid(a.domain, 1))

onnx.save(m, a.out)
print(f"removed {len(solve_nodes)} solve nodes; inserted GdnSolve({a.A}) -> {a.T} (domain '{a.domain}')")
print(f"dropped orphaned graph inputs: {removed_inputs}")
print(f"wrote {a.out}  (nodes {len(g.node)})")
