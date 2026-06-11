#!/usr/bin/env python3
"""Map GDN pipeline-stage intermediates to their ONNX tensor names, by running the quant-path ONNX
through ORT (float ≈ fp64 truth) and matching each stage's fp64 value to the ORT tensor it equals.
Output: a stage→tensor-name table, used by the device per-stage error probe."""
import os, sys, json, glob
import numpy as np, torch, onnx, onnxruntime as ort
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("GDN_NO_VSCALE", "1")
import gdn_onnx_kernel as G
from gdn_onnx_kernel import (INPUT_NAMES, CONST_INPUT_NAMES, const_inputs, per_head_vscale,
                             _golden_chunk_args, _masks, solve_T_blocked, l2norm_lastdim, CHUNK)

ONNX = sys.argv[1] if len(sys.argv) > 1 else "example/gdn_native/quant_v2_L0/gdn_q.onnx"
GOLDEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "gdn", "golden")
a = _golden_chunk_args(sorted(glob.glob(os.path.join(GOLDEN, "p00*_L00.npz")))[0], 0)
qc, kc, vc, gc, betac, S_in = a
vs, ivs = per_head_vscale(vc, S_in)

# --- recompute every stage intermediate in fp64 (same ops as gdn_chunk_onnx) ---
tl, sl, cu, ey = _masks(CHUNK, "cpu", torch.float64)
cs = {k: torch.from_numpy(v).double() for k, v in const_inputs().items()}
qn = l2norm_lastdim(qc) * (1.0 / (qc.shape[-1] ** 0.5))
kn = l2norm_lastdim(kc)
k_beta = kn * betac.unsqueeze(-1); v_beta = vc * betac.unsqueeze(-1)
g = torch.matmul(gc.unsqueeze(-2), cu.reshape(1, 1, CHUNK, CHUNK)).squeeze(-2)
eg = torch.exp(g).unsqueeze(-1)
diff = g.unsqueeze(-1) - g.unsqueeze(-2)
decay = torch.exp(diff * tl) * tl
A = (-torch.matmul(k_beta, kn.transpose(-1, -2)) * decay) * sl
sel = [cs[f"sel{i}"].reshape(1, 1, 32, CHUNK) for i in range(4)]
T = solve_T_blocked(A, bl=16, bp=32, sel=sel)
U = torch.matmul(v_beta.transpose(-1, -2), T.transpose(-1, -2)).transpose(-1, -2)
W = torch.matmul(T, k_beta * eg)
P = (torch.matmul(qn, kn.transpose(-1, -2)) * decay) * tl
St = S_in.transpose(-1, -2)
v_new = U - torch.matmul(St, W.transpose(-1, -2)).transpose(-1, -2)
attn = torch.matmul(St, (qn * eg).transpose(-1, -2)).transpose(-1, -2)
oc = attn + torch.matmul(v_new.transpose(-1, -2), P.transpose(-1, -2)).transpose(-1, -2)
g_last = g[..., -1:]
dec_k = torch.exp(g_last - g).unsqueeze(-1)
S_out = S_in * torch.exp(g_last).unsqueeze(-1) + torch.matmul(v_new.transpose(-1, -2), kn * dec_k).transpose(-1, -2)
STAGES = {"01_g": g, "01_eg": eg.squeeze(-1), "02_A": A, "02_T": T, "03_U": U, "03_W": W,
          "04_P": P, "04_vnew": v_new, "05_attn": attn, "05_oc": oc, "06_Sout": S_out}

# --- ORT all intermediates ---
m = onnx.load(ONNX); have = {o.name for o in m.graph.output}
for n in m.graph.node:
    for o in n.output:
        if o and o not in have:
            vi = onnx.ValueInfoProto(); vi.name = o; m.graph.output.append(vi); have.add(o)
sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
feed = {n: t.float().numpy().astype(np.float32) for n, t in zip(INPUT_NAMES, a)}
feed.update({k: v.astype(np.float32) for k, v in const_inputs().items()})
feed["vscale"] = vs.numpy().astype(np.float32); feed["inv_vscale"] = ivs.numpy().astype(np.float32)
outs = [o.name for o in sess.get_outputs()]
vals = {n: v for n, v in zip(outs, sess.run(outs, feed))}

# --- match each stage to the ORT tensor whose value equals it ---
mapping = {}
for label, ten in STAGES.items():
    ref = ten.float().numpy().ravel()
    best, berr = None, 1e9
    for n, v in vals.items():
        v = np.asarray(v)
        if v.size != ref.size:
            continue
        e = np.linalg.norm(v.ravel() - ref) / (np.linalg.norm(ref) + 1e-12)
        if e < berr:
            berr, best = e, n
    mapping[label] = best
    ten.float().numpy().astype("<f4").tofile(f"/tmp/ref_{label}.raw")   # fp64≈true reference
    print(f"  {label:9s} -> {best:28s} (match relerr {berr:.1e}, shape {tuple(ten.shape)})")
json.dump(mapping, open("/tmp/gdn_stage_map.json", "w"))
print("wrote /tmp/gdn_stage_map.json + /tmp/ref_*.raw")
