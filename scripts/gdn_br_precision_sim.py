#!/usr/bin/env python3
"""Host sim: does the block-recursive solve's off-diagonal/oc accuracy come back if we (a) use int16
merges instead of 8-bit, and/or (b) subtract the identity from diagonal operands before quantizing
(user's hypothesis: the diagonal '1's inflate the operand scale and crush small entries)?

Mirrors the device BR algorithm per-block (symmetric per-block quant, integer matmul, dequant, accumulate)
on the real p29_L00 C=256 chunk, then runs the fp64 GDN forward to get end-to-end oc relerr.

Usage: gdn_br_precision_sim.py [--C 256]
"""
import sys, os, glob, argparse
import numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
ap = argparse.ArgumentParser(); ap.add_argument("--C", type=int, default=256); a = ap.parse_args()
C = a.C; BL = 64; NB = C // BL

import gdn_onnx_kernel as gok; gok.CHUNK = C
import gdn_ref_kernel as grk; grk.CHUNK = C
from gdn_onnx_kernel import _golden_chunk_args, l2norm_lastdim, _masks
from gdn_solve_int16_model import GOLDEN

def relerr(x, y):
    return float(np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-12))

def _l2(x, dim=-1):
    return x / (x.norm(dim=dim, keepdim=True) + 1e-12)

def gdn_oc(qc, kc, vc, gc, betac, S_in, T=None):
    """fp64 GDN chunk forward; if T given, use it as attn=(I-A)^-1 instead of solving."""
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

# ---- build the real chunk + its strictly-lower A (same as the probe) ----
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


ROUND = np.round   # set to np.floor to mimic the v73deep kernel's FLOOR drain

def qsym(x, nbits, rnd=None):
    """symmetric per-array quant->dequant at exact max scale (best-case for nbits)."""
    lim = (1 << (nbits - 1)) - 1
    m = np.abs(x).max()
    if m == 0:
        return x.copy()
    s = m / lim
    r = rnd if rnd is not None else np.round
    return r(x / s).clip(-lim, lim) * s


def oq(x, out_rnd):
    """mimic the device per-merge OUTPUT int8 quant (depack); out_rnd None=keep float (idealized)."""
    return x if out_rnd is None else qsym(x, 8, rnd=out_rnd)

def solve_br(Ah, nbits, sub_diag, out_rnd=None):
    """block-recursive (I-Ah)^-1 with nbits merge OPERANDS; out_rnd!=None also quantizes each merge
    OUTPUT to int8 (round=np.round, floor=np.floor) to mimic the device depack/FLOOR drain."""
    I = np.eye(BL)
    Tblk = {}
    for i in range(NB):
        Tblk[(i, i)] = np.linalg.inv(I - Ah[i*BL:(i+1)*BL, i*BL:(i+1)*BL])
    for d in range(1, NB):
        for j in range(NB - d):
            i = j + d
            S = np.zeros((BL, BL))
            for k in range(j, i):
                Aik = Ah[i*BL:(i+1)*BL, k*BL:(k+1)*BL]
                Tkj = Tblk[(k, j)]
                if sub_diag and k == j:
                    term = qsym(Aik, nbits) @ qsym(Tkj - I, nbits) + qsym(Aik, nbits) @ I
                else:
                    term = qsym(Aik, nbits) @ qsym(Tkj, nbits)
                S += oq(term, out_rnd)            # device: each inner term depacked to int8
            Tii = Tblk[(i, i)]
            if sub_diag:
                Tblk[(i, j)] = oq(qsym(S, nbits) + qsym(Tii - I, nbits) @ qsym(S, nbits), out_rnd)
            else:
                Tblk[(i, j)] = oq(qsym(Tii, nbits) @ qsym(S, nbits), out_rnd)
    T = np.zeros((C, C))
    for (i, j), b in Tblk.items():
        T[i*BL:(i+1)*BL, j*BL:(j+1)*BL] = b
    return T


def oc_for_T(Tfull, Texact):  # Tfull [H,C,C] -> oc relerr vs exact-solve oc
    oc = gdn_oc(qc, kc, vc, gc, betac, S_in, T=torch.from_numpy(Tfull[None].astype(np.float64))).numpy()
    oc_ex = gdn_oc(qc, kc, vc, gc, betac, S_in, T=torch.from_numpy(Texact[None].astype(np.float64))).numpy()
    return relerr(oc, oc_ex)


# configs: (operand_bits, sub_diag, per-merge output quant) — the last mimics the device depack
configs = [(8, False, None), (8, False, np.round), (8, False, np.floor),
           (16, False, None), (16, False, np.floor), (16, True, np.floor)]
print(f"{'op-bits':>7} {'sub_diag':>9} {'out-quant':>10} {'oc relerr':>12}")
Texact = np.stack([np.linalg.inv(np.eye(C) - A[h]) for h in range(H)])
for nb, sd, orr in configs:
    Tb = np.stack([solve_br(A[h], nb, sd, out_rnd=orr) for h in range(H)])
    tag = {None: "float", np.round: "int8 round", np.floor: "int8 FLOOR"}[orr]
    print(f"{nb:>7} {str(sd):>9} {tag:>10} {oc_for_T(Tb, Texact):>12.3e}")
