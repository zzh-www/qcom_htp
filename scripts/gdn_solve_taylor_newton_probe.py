#!/usr/bin/env python3
"""Taylor-seed + Newton-Schulz refinement for the GDN 64x64 base-block triangular inverse.

X0 = order-p Taylor (Horner) -> residual R0 = A^{p+1}. Newton-Schulz X<-X(2I-MX) SQUARES the residual:
R_k = A^{(p+1)*2^k}. For 64x64 (A^64=0), p=3 + 4 Newton steps -> R=A^64=0 EXACT in ~11 matmuls.
All-matmul (parallel/HMX-HVX-friendly) unlike the serial forward-subst recurrence.

Quadratic convergence collapses the order: standalone Taylor needs 63 terms, standalone Newton (bad init)
needs 20+ steps; the COMBINATION needs ~11 matmuls. Caveat: ||A||>1 nilpotent overshoots mid-iteration
(R = A^4..A^32 grows to ~1e11 before A^64=0 zeroes it) -> int16 overflows on the high-||A|| tail (~18%
of real blocks); fp32 (e.g. HVX qf32, which the shipped op already uses) converges 100%.

Reports convergence rate / worst-case / iterate-peak across real 64x64 blocks, fp32 vs int16.
Usage: gdn_solve_taylor_newton_probe.py [--samples 60]
"""
import sys, os, glob, argparse
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
ap = argparse.ArgumentParser(); ap.add_argument("--samples", type=int, default=60); a = ap.parse_args()
import gdn_onnx_kernel as gok; gok.CHUNK = 64
from gdn_onnx_kernel import _golden_chunk_args, l2norm_lastdim, _masks
C = 64; tl, sl, cu, ey = _masks(C, "cpu", torch.float64)

def buildA(npz):
    qc, kc, vc, gc, betac, S_in = _golden_chunk_args(npz, 0)
    kn = l2norm_lastdim(kc); k_beta = kn * betac.unsqueeze(-1)
    g = torch.matmul(gc.unsqueeze(-2), cu.reshape(1, 1, C, C)).squeeze(-2)
    diff = g.unsqueeze(-1) - g.unsqueeze(-2); decay = torch.exp(diff * tl) * tl
    return ((-torch.matmul(k_beta, kn.transpose(-1, -2)) * decay) * sl).double().numpy()[0]

def relerr(x, y): return np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-12)
def qsym(x, nb):
    if nb is None: return x
    lim = (1 << (nb - 1)) - 1; m = np.abs(x).max()
    return x.copy() if m == 0 else np.round(x / (m / lim)).clip(-lim, lim) * (m / lim)

def taylor_newton(Aii, p, K, nb):
    n = Aii.shape[0]; I = np.eye(n); A = qsym(Aii, nb)
    X = I.copy()
    for _ in range(p): X = qsym(I + A @ qsym(X, nb), nb)        # order-p Taylor seed (Horner)
    M = I - A; pk = np.abs(X).max()
    for _ in range(K):                                         # Newton-Schulz refine
        MX = qsym(M @ qsym(X, nb), nb); X = qsym(qsym(X, nb) @ qsym(2 * I - MX, nb), nb)
        pk = max(pk, np.abs(X).max())
    return X, pk

files = [f for f in sorted(glob.glob(os.path.join(ROOT, "tests/gdn/golden/*.npz"))) if np.load(f)["query"].shape[1] >= C]
samp = files[::max(1, len(files) // a.samples)]
blocks = [A[h] for f in samp for A in [buildA(f)] for h in range(A.shape[0])]
specs = np.array([np.linalg.norm(b, 2) for b in blocks])
print(f"{len(blocks)} real 64x64 A-blocks  ||A||2: median {np.median(specs):.2f} p90 {np.percentile(specs,90):.2f} max {specs.max():.2f}")
print(f"{'config':>20} {'#mm':>4} | {'fp32 conv%':>10} {'fp32 max':>10} | {'int16 conv%':>11} {'int16 max':>10} {'int16 peak':>11}")
for p, K in [(3, 2), (3, 3), (3, 4), (4, 3), (4, 4)]:
    out = {}
    for nb in (None, 16):
        res = np.array([relerr(*[taylor_newton(b, p, K, nb)[0:1][0], np.linalg.inv(np.eye(C) - b)]) for b in blocks])
        pk = max(taylor_newton(b, p, K, nb)[1] for b in blocks) if nb is not None else 0
        out[nb] = (100 * np.mean(res < 1e-2), res.max(), pk)
    print(f"  Taylor{p}+Newton x{K:<5} {p+2*K:>4} | {out[None][0]:>9.0f}% {out[None][1]:>10.1e} | {out[16][0]:>10.0f}% {out[16][1]:>10.1e} {out[16][2]:>11.1e}")
