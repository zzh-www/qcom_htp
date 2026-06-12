#!/usr/bin/env python3
"""Lever 4 (pure-HMX §6): diagonal-equilibration preconditioning to drop the diagonal Newton dtype.

The all-matmul diagonal inverse (Taylor+Newton) needs w16a16 ONLY because the intermediate iterate A^k
transiently blows up (peak ~1e13) on the high-||A|| tail and overflows int16. That blow-up is NON-NORMAL
TRANSIENT, not spectral (A strictly lower-triangular nilpotent -> rho=0). A diagonal similarity
  Atil = D^-1 A D,  D = diag(s^0,...,s^{n-1})   =>   Atil[i,j] = A[i,j] * s^(j-i)
shrinks every sub-diagonal entry (i>j => s^(j-i)<1 for s>1) so ||Atil||<1 and Atil^k stays BOUNDED.
inv(L) = D inv(Ltil) D^-1 is EXACT in real arithmetic (the s-factors telescope along every path), so
un-conditioning recovers all Neumann orders. Question this probe answers, on REAL blocks:
  - can a scalar s bound the iterate so the cheap dtype (int8 / int16) stops overflowing?
  - after un-conditioning, does the cheap-dtype inverse still converge (relerr<1e-2)?
  - i.e. does preconditioning let the diagonal Newton run at u8i8/w8a16 instead of w16a16?

Trade-off the probe surfaces: large s bounds Atil^k but inflates the DYNAMIC RANGE of inv(Ltil)
(entries decay as 1/s^(i-j)); a single-scale int quantizer then loses the far-sub-diagonal entries,
which un-conditioning (x s^(i-j)) re-amplifies into error. So there is an optimal s -- we grid-search it
per block (best-case feasibility; a runtime rule comes later if it pans out).

Usage: uv run python scripts/gdn_solve_precond_probe.py [--samples 60]
"""
import sys, os, glob, argparse
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
ap = argparse.ArgumentParser(); ap.add_argument("--samples", type=int, default=60); a = ap.parse_args()
import gdn_onnx_kernel as gok; gok.CHUNK = 64
from gdn_onnx_kernel import _golden_chunk_args, l2norm_lastdim, _masks
C = 64; tl, sl, cu, ey = _masks(C, "cpu", torch.float64)
IDX = np.arange(C); POW = IDX[None, :] - IDX[:, None]   # POW[i,j] = j - i

def buildA(npz):
    qc, kc, vc, gc, betac, S_in = _golden_chunk_args(npz, 0)
    kn = l2norm_lastdim(kc); k_beta = kn * betac.unsqueeze(-1)
    g = torch.matmul(gc.unsqueeze(-2), cu.reshape(1, 1, C, C)).squeeze(-2)
    diff = g.unsqueeze(-1) - g.unsqueeze(-2); decay = torch.exp(diff * tl) * tl
    return ((-torch.matmul(k_beta, kn.transpose(-1, -2)) * decay) * sl).double().numpy()[0]

def relerr(x, y): return np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-12)
def qfix(x, scale, nb):
    """FIXED-scale symmetric quant -> models the hardware int register: values beyond +-lim*scale CLIP
    (the real A^k overflow), unlike a per-call max-rescale which would hide it."""
    if nb is None or scale == 0: return x
    lim = (1 << (nb - 1)) - 1
    return np.round(x / scale).clip(-lim, lim) * scale

def taylor_newton(A, p, K, nb, sA, sX):
    """All-matmul inverse of (I-A): order-p Taylor + K Newton-Schulz, FIXED-scale quant (sA operands, sX iterate).
    Iterate stored int16/int8 at fixed sX -> transient overshoot CLIPS at lim*sX. Returns (X, peak abs iterate)."""
    n = A.shape[0]; I = np.eye(n); Aq = qfix(A, sA, nb); X = I.copy()
    for _ in range(p): X = qfix(I + Aq @ qfix(X, sX, nb), sX, nb)
    M = I - Aq; pk = np.abs(X).max()
    for _ in range(K):
        MX = qfix(M @ qfix(X, sX, nb), sX, nb); X = qfix(qfix(X, sX, nb) @ qfix(2 * I - MX, sX, nb), sX, nb)
        pk = max(pk, np.abs(X).max())
    return X, pk

def scales(A, true_inv, nb):
    """Oracle-best FIXED scales: operand sA from |A|max, iterate sX from |true inv|max (full-range, no clip
    of the *answer* -> only transient overshoot clips). This is the most favorable fixed scale."""
    lim = (1 << (nb - 1)) - 1
    return (np.abs(A).max() / lim or 1.0), (np.abs(true_inv).max() / lim or 1.0)

def precond(A, s):
    return A * (s ** POW)                 # Atil[i,j] = A[i,j] * s^(j-i)
def uncond(Xtil, s):
    return Xtil * (s ** (-POW))           # X[i,j] = Xtil[i,j] * s^(i-j)

S_GRID = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]

def solve_precond(A, p, K, nb, true_inv):
    """Best-over-s preconditioned cheap-dtype inverse. Returns (best_relerr, best_s, peak_at_best, ||Atil||)."""
    best = (np.inf, 1.0, 0.0, 0.0)
    for s in S_GRID:
        At = precond(A, s); inv_til = precond(true_inv, s)   # inv(Ltil) = D^-1 inv(L) D = precond(inv,s)
        sA, sX = scales(At, inv_til, nb)
        Xt, pk = taylor_newton(At, p, K, nb, sA, sX)
        err = relerr(uncond(Xt, s), true_inv)
        if err < best[0]: best = (err, s, pk, np.linalg.norm(At, 2))
    return best

files = [f for f in sorted(glob.glob(os.path.join(ROOT, "tests/gdn/golden/*.npz"))) if np.load(f)["query"].shape[1] >= C]
samp = files[::max(1, len(files) // a.samples)]
blocks = [A[h] for f in samp for A in [buildA(f)] for h in range(A.shape[0])]
true_inv = [np.linalg.inv(np.eye(C) - b) for b in blocks]
specs = np.array([np.linalg.norm(b, 2) for b in blocks])
tail = specs >= 4.0
print(f"{len(blocks)} real 64x64 A-blocks  ||A||2: median {np.median(specs):.2f} p90 {np.percentile(specs,90):.2f} "
      f"max {specs.max():.2f}  | high-||A|| tail(>=4): {tail.sum()} ({100*tail.mean():.0f}%)")
print("\nbaseline (NO precond)  vs  best-scalar-s precond     [conv% = frac relerr<1e-2;  tail = same on >=4 blocks]")
print(f"{'config':>16} {'dtype':>6} | {'conv%':>6} {'tail%':>6} {'maxRelErr':>10} {'iterPeak':>10}"
      f" | {'PC conv%':>8} {'PC tail%':>8} {'PC maxErr':>10} {'PC peak':>9} {'med ||Atil||':>12} {'med s*':>7}")
for p, K in [(3, 3), (3, 4), (4, 4)]:
    for nb in (16, 8):
        bsc = [scales(b, true_inv[i], nb) for i, b in enumerate(blocks)]
        base = np.array([relerr(taylor_newton(b, p, K, nb, *bsc[i])[0], true_inv[i]) for i, b in enumerate(blocks)])
        basepk = max(taylor_newton(b, p, K, nb, *bsc[i])[1] for i, b in enumerate(blocks))
        pc = [solve_precond(b, p, K, nb, true_inv[i]) for i, b in enumerate(blocks)]
        pcerr = np.array([x[0] for x in pc]); pcpk = max(x[2] for x in pc)
        atiln = np.median([x[3] for x in pc]); meds = np.median([x[1] for x in pc])
        print(f"  T{p}+N{K:<2}({p+2*K:>2}mm) {('int'+str(nb)):>6} | "
              f"{100*np.mean(base<1e-2):>5.0f}% {100*np.mean(base[tail]<1e-2):>5.0f}% {base.max():>10.1e} {basepk:>10.1e}"
              f" | {100*np.mean(pcerr<1e-2):>7.0f}% {100*np.mean(pcerr[tail]<1e-2):>7.0f}% {pcerr.max():>10.1e}"
              f" {pcpk:>9.1e} {atiln:>12.2f} {meds:>7.1f}")
print("\nread: if 'PC tail%' jumps to ~100 at int8/int16 where baseline tail% is low, preconditioning")
print("enables the cheaper diagonal dtype (lever 4 -> ~0.4-0.9M floor). 'PC peak' << iterPeak confirms the bound.")
