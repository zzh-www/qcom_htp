#!/usr/bin/env python3
"""Evaluate the TAYLOR / NEUMANN-series iterative-matmul route for the GDN triangular solve,
vs the shipped block forward-substitution, on REAL golden A.

Math: A is strictly-lower CxC -> nilpotent (A^C=0), so  (I-A)^-1 = sum_{k=0..C-1} A^k  EXACTLY.
Two iterative-matmul realisations:
  horner   : T_{n+1} = I + A @ T_n   (C-1 matmuls; exact at n=C-1; truncatable)
  squaring : (I-A)^-1 = prod_k (I + A^{2^k})  (~2*log2(C) matmuls; doubling)
Both reduce the inverse to a chain of CxC matmuls -> the question is precision under int8/int16
requant between matmuls, and the matmul COUNT (timing).

Reports: T relerr vs fp64 inverse, end-to-end oc relerr, and #CxC-matmuls (timing proxy).
Usage: gdn_solve_taylor_probe.py [--C 64]
"""
import sys, os, glob, argparse
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
ap = argparse.ArgumentParser(); ap.add_argument("--C", type=int, default=64); a = ap.parse_args()
C = a.C

import gdn_onnx_kernel as gok; gok.CHUNK = C
import gdn_ref_kernel as grk; grk.CHUNK = C
from gdn_onnx_kernel import _golden_chunk_args, l2norm_lastdim, _masks
from gdn_solve_int16_model import GOLDEN

def relerr(x, y):
    return float(np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-12))

def _l2(x, dim=-1):
    return x / (x.norm(dim=dim, keepdim=True) + 1e-12)

def gdn_oc(qc, kc, vc, gc, betac, S_in, T):
    """fp64 GDN chunk forward using attn=T."""
    dt = torch.float64
    qc, kc, vc = qc.to(dt), kc.to(dt), vc.to(dt)
    gc, betac, S_in = gc.to(dt), betac.to(dt), S_in.to(dt)
    Cc, Dk = qc.shape[-2], qc.shape[-1]; z = torch.zeros((), dtype=dt)
    qc = _l2(qc) * (1.0 / (Dk ** 0.5)); kc = _l2(kc)
    v_beta = vc * betac.unsqueeze(-1); k_beta = kc * betac.unsqueeze(-1)
    gg = torch.cumsum(gc, dim=-1)
    dff = gg.unsqueeze(-1) - gg.unsqueeze(-2)
    trl = torch.tril(torch.ones(Cc, Cc, dtype=torch.bool)); dec = torch.exp(torch.where(trl, dff, z)) * trl.to(dt)
    attn = T.to(dt)
    U = attn @ v_beta; W = attn @ (k_beta * torch.exp(gg).unsqueeze(-1))
    P = (qc @ kc.transpose(-1, -2)) * dec
    P = P.masked_fill(torch.triu(torch.ones(Cc, Cc, dtype=torch.bool), 1), 0.0)
    v_new = U - W @ S_in
    return (qc * torch.exp(gg).unsqueeze(-1)) @ S_in + P @ v_new

# ---- real chunk + strictly-lower A ----
npz = next(f for f in sorted(glob.glob(os.path.join(GOLDEN, "*.npz")))
           if np.load(f)["query"].shape[1] >= C)
print(f"golden = {os.path.basename(npz)}  C={C}")
qc, kc, vc, gc, betac, S_in = _golden_chunk_args(npz, 0)
tl, sl, cu, ey = _masks(C, "cpu", torch.float64)
kn = l2norm_lastdim(kc); k_beta = kn * betac.unsqueeze(-1)
g = torch.matmul(gc.unsqueeze(-2), cu.reshape(1, 1, C, C)).squeeze(-2)
diff = g.unsqueeze(-1) - g.unsqueeze(-2); decay = torch.exp(diff * tl) * tl
A = ((-torch.matmul(k_beta, kn.transpose(-1, -2)) * decay) * sl).double().numpy()[0]  # [H,C,C]
H = A.shape[0]
Texact = np.stack([np.linalg.inv(np.eye(C) - A[h]) for h in range(H)])

def qsym(x, nbits):
    """symmetric per-matrix quant->dequant at exact-max scale (best case for nbits)."""
    lim = (1 << (nbits - 1)) - 1
    m = np.abs(x).max()
    if m == 0: return x.copy()
    s = m / lim
    return np.round(x / s).clip(-lim, lim) * s

def Q(x, nbits):
    return x if nbits is None else qsym(x, nbits)

# ---- iterative-matmul realisations (per head). nbits=None -> fp; else requant each matmul result ----
def horner(Ah, nbits, K=None):
    """T = I + A@(I + A@(...)); K matmuls (default C-1 = exact)."""
    K = (C - 1) if K is None else K
    I = np.eye(C); T = I.copy()
    Aq = Q(Ah, nbits)
    for _ in range(K):
        T = Q(I + Aq @ Q(T, nbits), nbits)
    return T

def squaring(Ah, nbits):
    """(I-A)^-1 = prod_k (I + A^{2^k}); stop when A^{2^k}==0 (k up to log2(C))."""
    I = np.eye(C)
    Ak = Q(Ah, nbits)              # A^{2^0}
    P = Q(I + Ak, nbits)
    k = 1
    while (1 << k) < C:
        Ak = Q(Ak @ Ak, nbits)     # A^{2^k}
        P = Q(P @ Q(I + Ak, nbits), nbits)
        k += 1
    return P

def n_matmul(method):
    if method == "horner_full": return C - 1
    if method == "squaring":
        k = 0
        while (1 << k) < C: k += 1     # factors 0..k-1 ; squarings k-1 ; products k-1
        return (k - 1) + (k - 1)       # squarings + product-multiplies
    return None

def eval_T(name, Tfn, nmm):
    Tb = np.stack([Tfn(A[h]) for h in range(H)])
    tre = relerr(Tb, Texact)
    oc = gdn_oc(qc, kc, vc, gc, betac, S_in, torch.from_numpy(Tb[None])).numpy()
    oce = gdn_oc(qc, kc, vc, gc, betac, S_in, torch.from_numpy(Texact[None])).numpy()
    ocre = relerr(oc, oce)
    print(f"{name:<28} {str(nmm):>6}   {tre:>11.3e}   {ocre:>11.3e}")

print(f"\n||A||2 mean {np.mean([np.linalg.norm(A[h],2) for h in range(H)]):.2f} "
      f"max {np.max([np.linalg.norm(A[h],2) for h in range(H)]):.2f}  "
      f"(A nilpotent: A^{C}=0, but A^8 absmax {np.abs(np.linalg.matrix_power(A[2],8)).max():.0f})\n")
print(f"{'method':<28} {'#mm':>6}   {'T relerr':>11}   {'oc relerr':>11}")
print("-" * 64)
# forward-subst reference = exact inverse (the shipped op realises this near-exact, T 3.6e-5)
print(f"{'fwd-subst (shipped, fp)':<28} {'~':>6}   {'(ref:3.6e-5)':>11}   {0.0:>11.3e}")
print("- Taylor/Neumann iterative-matmul ".ljust(64, "-"))
for nb, tag in [(None, "fp64"), (16, "int16"), (8, "int8")]:
    eval_T(f"horner-full ({tag})", lambda Ah, nb=nb: horner(Ah, nb), n_matmul("horner_full"))
for nb, tag in [(None, "fp64"), (16, "int16"), (8, "int8")]:
    eval_T(f"squaring ({tag})", lambda Ah, nb=nb: squaring(Ah, nb), n_matmul("squaring"))
print("- truncated Horner (fp64) — does early-stop work? ".ljust(64, "-"))
for K in [4, 8, 16, 24, 32, C - 1]:
    eval_T(f"horner K={K} (fp64)", lambda Ah, K=K: horner(Ah, None, K=K), K)
