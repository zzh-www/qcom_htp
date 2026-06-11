#!/usr/bin/env python3
"""Hybrid solve: 64x64 diagonal blocks via iterative-matmul (horner Neumann), C>64 via block-triangular
recursion. Tests the user's proposal — cap the iterative solve at 64 (where int16 survives, partial-sum
peak ~254) and recurse for larger C. Compares T relerr / oc relerr / #matmul vs shipped fwd-subst.

Usage: gdn_solve_hybrid_probe.py [--C 256] [--bl 64]
"""
import sys, os, glob, argparse
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
ap = argparse.ArgumentParser(); ap.add_argument("--C", type=int, default=256); ap.add_argument("--bl", type=int, default=64)
a = ap.parse_args(); C = a.C; BL = a.bl

import gdn_onnx_kernel as gok; gok.CHUNK = C
from gdn_onnx_kernel import _golden_chunk_args, l2norm_lastdim, _masks
from gdn_solve_int16_model import GOLDEN

def relerr(x, y): return float(np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-12))
def _l2(x, d=-1): return x / (x.norm(dim=d, keepdim=True) + 1e-12)

def gdn_oc(qc, kc, vc, gc, betac, S_in, T):
    dt = torch.float64
    qc, kc, vc, gc, betac, S_in = [t.to(dt) for t in (qc, kc, vc, gc, betac, S_in)]
    Cc, Dk = qc.shape[-2], qc.shape[-1]; z = torch.zeros((), dtype=dt)
    qc = _l2(qc) * (1.0 / (Dk ** 0.5)); kc = _l2(kc)
    v_beta = vc * betac.unsqueeze(-1); k_beta = kc * betac.unsqueeze(-1)
    gg = torch.cumsum(gc, dim=-1); dff = gg.unsqueeze(-1) - gg.unsqueeze(-2)
    trl = torch.tril(torch.ones(Cc, Cc, dtype=torch.bool)); dec = torch.exp(torch.where(trl, dff, z)) * trl.to(dt)
    attn = T.to(dt); U = attn @ v_beta; W = attn @ (k_beta * torch.exp(gg).unsqueeze(-1))
    P = (qc @ kc.transpose(-1, -2)) * dec
    P = P.masked_fill(torch.triu(torch.ones(Cc, Cc, dtype=torch.bool), 1), 0.0)
    v_new = U - W @ S_in
    return (qc * torch.exp(gg).unsqueeze(-1)) @ S_in + P @ v_new

npz = next(f for f in sorted(glob.glob(os.path.join(GOLDEN, "*.npz"))) if np.load(f)["query"].shape[1] >= C)
print(f"golden = {os.path.basename(npz)}  C={C}  BL={BL}  NB={C//BL}")
qc, kc, vc, gc, betac, S_in = _golden_chunk_args(npz, 0)
tl, sl, cu, ey = _masks(C, "cpu", torch.float64)
kn = l2norm_lastdim(kc); k_beta = kn * betac.unsqueeze(-1)
g = torch.matmul(gc.unsqueeze(-2), cu.reshape(1, 1, C, C)).squeeze(-2)
diff = g.unsqueeze(-1) - g.unsqueeze(-2); decay = torch.exp(diff * tl) * tl
A = ((-torch.matmul(k_beta, kn.transpose(-1, -2)) * decay) * sl).double().numpy()[0]
H = A.shape[0]
Texact = np.stack([np.linalg.inv(np.eye(C) - A[h]) for h in range(H)])

MM = [0]   # matmul counter (per head, counted once on head 0)
def qsym(x, nb):
    if nb is None: return x
    lim = (1 << (nb - 1)) - 1; m = np.abs(x).max()
    return x.copy() if m == 0 else np.round(x / (m / lim)).clip(-lim, lim) * (m / lim)

def horner_block(Aii, nb, count):
    """64x64 (I-Aii)^-1 via T<-I+Aii@T, BL-1 matmuls; nb=requant precision each step."""
    n = Aii.shape[0]; I = np.eye(n); T = I.copy(); Aq = qsym(Aii, nb)
    for _ in range(n - 1):
        T = qsym(I + Aq @ qsym(T, nb), nb)
        if count: MM[0] += 1
    return T

def block_recursive(Ah, nb_diag, nb_merge, count=False):
    NB = C // BL; I = np.eye(BL); Tblk = {}
    for i in range(NB):
        Tblk[(i, i)] = horner_block(Ah[i*BL:(i+1)*BL, i*BL:(i+1)*BL], nb_diag, count)
    for d in range(1, NB):
        for j in range(NB - d):
            i = j + d; S = np.zeros((BL, BL))
            for k in range(j, i):
                S += qsym(Ah[i*BL:(i+1)*BL, k*BL:(k+1)*BL], nb_merge) @ qsym(Tblk[(k, j)], nb_merge)
                if count: MM[0] += 1
            Tblk[(i, j)] = qsym(Tblk[(i, i)], nb_merge) @ qsym(S, nb_merge)
            if count: MM[0] += 1
    T = np.zeros((C, C))
    for (i, j), b in Tblk.items(): T[i*BL:(i+1)*BL, j*BL:(j+1)*BL] = b
    return T

def ev(name, fn, nmm):
    Tb = np.stack([fn(A[h]) for h in range(H)])
    oc = gdn_oc(qc, kc, vc, gc, betac, S_in, torch.from_numpy(Tb[None])).numpy()
    oce = gdn_oc(qc, kc, vc, gc, betac, S_in, torch.from_numpy(Texact[None])).numpy()
    print(f"{name:<34} {str(nmm):>8}   {relerr(Tb, Texact):>11.3e}   {relerr(oc, oce):>11.3e}")

# count matmuls for the hybrid (one head)
MM[0] = 0; block_recursive(A[0], 16, 16, count=True); nmm_hybrid = MM[0]
print(f"\n{'method':<34} {'#mm/head':>8}   {'T relerr':>11}   {'oc relerr':>11}")
print("-" * 72)
print(f"{'shipped fwd-subst (1 fused op)':<34} {'~C^3/3':>8}   {'3.6e-5(ref)':>11}   {0.0:>11.3e}")
ev("hybrid: 64x64 horner(fp) + merge", lambda Ah: block_recursive(Ah, None, None), nmm_hybrid)
ev("hybrid: 64x64 horner(i16)+merge(i16)", lambda Ah: block_recursive(Ah, 16, 16), nmm_hybrid)
ev("hybrid: 64x64 horner(i16)+merge(i8)", lambda Ah: block_recursive(Ah, 16, 8), nmm_hybrid)
ev("hybrid: 64x64 horner(i8) +merge(i8)", lambda Ah: block_recursive(Ah, 8, 8), nmm_hybrid)
print(f"\n(#mm = 64x64x64 matmuls per head; fwd-subst exploits triangular ~C^3/3 as ONE op)")
print(f" hybrid breakdown: {C//BL} diag blocks x {BL-1} horner + merges = {nmm_hybrid} matmuls/head")
