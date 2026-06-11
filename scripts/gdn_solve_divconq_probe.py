#!/usr/bin/env python3
"""Recursive divide-and-conquer triangular inverse for the GDN solve (log-depth, mathematically exact).

M=I-A lower-tri unit-diag. Partition M=[[L11,0],[L21,L22]] ->
  M^-1 = [[L11^-1, 0], [-L22^-1 L21 L11^-1, L22^-1]].
Recurse to a `base` block (exact small inverse), combine with 2 matmuls per merge.
Depth = log2(C/base); total work O(C^3); NEVER forms A^k -> no power-explosion (unlike Neumann/squaring),
so int16 survives. This is the "log-n provable" method (vs the diverging Taylor/Neumann/Newton routes in
gdn_solve_taylor_probe.py / gdn_solve_hybrid_probe.py).

Reports T relerr, end-to-end oc relerr, matmul count, critical-path depth, per base x precision.
Usage: gdn_solve_divconq_probe.py [--C 256]
"""
import sys, os, glob, argparse
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
ap = argparse.ArgumentParser(); ap.add_argument("--C", type=int, default=256); a = ap.parse_args(); C = a.C
import gdn_onnx_kernel as gok; gok.CHUNK = C
from gdn_onnx_kernel import _golden_chunk_args, l2norm_lastdim, _masks
from gdn_solve_int16_model import GOLDEN

def _l2(x, d=-1): return x / (x.norm(dim=d, keepdim=True) + 1e-12)
def relerr(x, y): return float(np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-12))
def gdn_oc(qc, kc, vc, gc, betac, S_in, T):
    dt = torch.float64; qc, kc, vc, gc, betac, S_in = [t.to(dt) for t in (qc, kc, vc, gc, betac, S_in)]
    Cc, Dk = qc.shape[-2], qc.shape[-1]; z = torch.zeros((), dtype=dt)
    qc = _l2(qc) * (1.0 / (Dk ** 0.5)); kc = _l2(kc); v_beta = vc * betac.unsqueeze(-1); k_beta = kc * betac.unsqueeze(-1)
    gg = torch.cumsum(gc, dim=-1); dff = gg.unsqueeze(-1) - gg.unsqueeze(-2)
    trl = torch.tril(torch.ones(Cc, Cc, dtype=torch.bool)); dec = torch.exp(torch.where(trl, dff, z)) * trl.to(dt)
    attn = T.to(dt); U = attn @ v_beta; W = attn @ (k_beta * torch.exp(gg).unsqueeze(-1))
    P = (qc @ kc.transpose(-1, -2)) * dec; P = P.masked_fill(torch.triu(torch.ones(Cc, Cc, dtype=torch.bool), 1), 0.0)
    v_new = U - W @ S_in
    return (qc * torch.exp(gg).unsqueeze(-1)) @ S_in + P @ v_new

def qsym(x, nb):
    if nb is None: return x
    lim = (1 << (nb - 1)) - 1; m = np.abs(x).max()
    return x.copy() if m == 0 else np.round(x / (m / lim)).clip(-lim, lim) * (m / lim)

CNT = [0, 0]   # [matmuls, max-depth]
def tri_inv(M, nb, base, depth=0):
    CNT[1] = max(CNT[1], depth); n = M.shape[0]
    if n <= base: return np.linalg.inv(M)         # small diagonal base = exact (forward-subst on device)
    h = n // 2
    I11 = tri_inv(M[:h, :h], nb, base, depth + 1)
    I22 = tri_inv(M[h:, h:], nb, base, depth + 1)
    t = qsym(qsym(M[h:, :h], nb) @ qsym(I11, nb), nb); CNT[0] += 1
    X21 = qsym(-(qsym(I22, nb) @ t), nb); CNT[0] += 1
    out = np.zeros((n, n)); out[:h, :h] = I11; out[h:, h:] = I22; out[h:, :h] = X21
    return out

npz = next(f for f in sorted(glob.glob(os.path.join(GOLDEN, "*.npz"))) if np.load(f)["query"].shape[1] >= C)
print(f"golden = {os.path.basename(npz)}  C={C}")
qc, kc, vc, gc, betac, S_in = _golden_chunk_args(npz, 0)
tl, sl, cu, ey = _masks(C, "cpu", torch.float64); kn = l2norm_lastdim(kc); k_beta = kn * betac.unsqueeze(-1)
g = torch.matmul(gc.unsqueeze(-2), cu.reshape(1, 1, C, C)).squeeze(-2)
diff = g.unsqueeze(-1) - g.unsqueeze(-2); decay = torch.exp(diff * tl) * tl
A = ((-torch.matmul(k_beta, kn.transpose(-1, -2)) * decay) * sl).double().numpy()[0]; H = A.shape[0]
Tex = np.stack([np.linalg.inv(np.eye(C) - A[h]) for h in range(H)])
oce = gdn_oc(qc, kc, vc, gc, betac, S_in, torch.from_numpy(Tex[None])).numpy()
print(f"{'base':>5} {'prec':>6} {'T relerr':>11} {'oc relerr':>11} {'#matmul':>8} {'depth':>6}")
print("-" * 52)
for base in (8, 16, 32):
    for nb in (None, 16, 8):
        CNT[0] = CNT[1] = 0
        Tb = np.stack([tri_inv(np.eye(C) - A[h], nb, base) for h in range(H)])
        oc = gdn_oc(qc, kc, vc, gc, betac, S_in, torch.from_numpy(Tb[None])).numpy()
        tag = {None: "fp32", 16: "int16", 8: "int8"}[nb]
        print(f"{base:>5} {tag:>6} {relerr(Tb, Tex):>11.3e} {relerr(oc, oce):>11.3e} {CNT[0]:>8} {CNT[1]:>6}")
