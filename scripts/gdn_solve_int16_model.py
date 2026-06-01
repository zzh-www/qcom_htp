#!/usr/bin/env python3
"""Bit-faithful host model of the GDN triangular-solve T=(I-A)^-1 done the KDA way: per-16x16-block
forward substitution + block-triangular merge, INT16 storage with INT32 accumulation, requant only
when a result tensor is written (NOT at every matmul like the int8-HMX graph).  This is the de-risk
+ golden reference for the planned HVX custom solve op (op scope: A[B,H,64,64] -> T[B,H,64,64];
int16 in, int16 out, int32 internal; downstream casts T to int8 for the U/W consume matmuls).

Result (p00 L00 chunk0): int16 solve T relerr vs fp64 = 3.6e-5 (vs current int8-matmul solve 4.4e-3),
int32 accumulator peak 5.4e8 < 2^31 (no overflow), consume U/W hit the int8-in[1] ceiling (~6e-3).
The current int8 `solve_T_blocked` in gdn_onnx_kernel.py stays the FLOAT reference; this proves an
int16-internal solve reaches the U/W ceiling that the int8-matmul Neumann chain cannot.

Run:  GDN_NO_VSCALE=1 .venv/bin/python scripts/gdn_solve_int16_model.py
"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GDN_NO_VSCALE", "1")
from gdn_onnx_kernel import _golden_chunk_args, l2norm_lastdim, _masks, CHUNK

GOLDEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "gdn", "golden")
BL = 16                                                          # logical block (KDA sub-chunk BC)
ST = 2.0 / 32767.0                                              # T output int16 scale (|T| < 2)


def solve_int16(Ah, stats):
    """Ah: [C,C] fp strictly-lower (one head).  Returns T [C,C] fp (dequantized int16 codes)."""
    C = Ah.shape[0]; nb = C // BL
    sA = (np.abs(Ah).max() / 32767.0) or 1e-12
    Aq = np.round(Ah / sA).clip(-32767, 32767).astype(np.int64)  # int16 codes of A
    # --- diagonal block inverses: forward substitution, int32 accum, requant rows to int16 ---
    Ti = np.zeros((nb, BL, BL), np.int64)
    for b in range(nb):
        d = Aq[b*BL:(b+1)*BL, b*BL:(b+1)*BL]                     # strictly-lower int16 codes
        Tb = np.zeros((BL, BL), np.int64)
        for i in range(BL):                                     # T[i,:] = e_i + A[i,:i] @ T[:i,:]
            acc = np.zeros(BL, np.int64)
            if i > 0:
                acc = (d[i, :i][:, None] * Tb[:i, :]).sum(0)    # int16*int16 -> int32 accum (scale sA*sT)
                stats[0] = max(stats[0], int(np.abs(acc).max()))
            ei = np.zeros(BL); ei[i] = 1.0
            Tb[i, :] = np.round(ei / ST + acc * sA).clip(-32767, 32767).astype(np.int64)
        Ti[b] = Tb
    # --- block-triangular merge: T_ij = T_ii @ sum_{k=j..i-1} A_ik T_kj  (int16 codes, int32 accum) ---
    Tblk = {(b, b): Ti[b].astype(np.float64) * ST for b in range(nb)}
    for i in range(nb):
        for j in range(i-1, -1, -1):
            acc = np.zeros((BL, BL), np.float64)
            for k in range(j, i):
                Aik = Aq[i*BL:(i+1)*BL, k*BL:(k+1)*BL].astype(np.float64) * sA
                acc += Aik @ Tblk[(k, j)]
            Tblk[(i, j)] = (Ti[i].astype(np.float64) * ST) @ acc
    T = np.zeros((C, C))
    for i in range(nb):
        for j in range(i+1):
            T[i*BL:(i+1)*BL, j*BL:(j+1)*BL] = Tblk[(i, j)]
    return T


def main():
    a = _golden_chunk_args(os.path.join(GOLDEN, "p00_L00.npz"), 0)
    qc, kc, vc, gc, betac, S_in = a
    tl, sl, cu, ey = _masks(CHUNK, "cpu", torch.float64)
    kn = l2norm_lastdim(kc); k_beta = kn * betac.unsqueeze(-1); v_beta = vc * betac.unsqueeze(-1)
    g = torch.matmul(gc.unsqueeze(-2), cu.reshape(1, 1, CHUNK, CHUNK)).squeeze(-2)
    eg = torch.exp(g).unsqueeze(-1)
    diff = g.unsqueeze(-1) - g.unsqueeze(-2); decay = torch.exp(diff * tl) * tl
    A = ((-torch.matmul(k_beta, kn.transpose(-1, -2)) * decay) * sl).double().numpy()[0]   # [H,C,C]
    vb = v_beta.double().numpy()[0]; kbe = (k_beta * eg).double().numpy()[0]
    H, C, _ = A.shape

    def q8(x):
        s = (np.abs(x).max() / 127.0) or 1e-12
        return np.round(x / s).clip(-127, 127) * s

    stats = [0]; relT, relU, relW = [], [], []
    for h in range(H):
        Tref = np.linalg.inv(np.eye(C) - A[h])
        Tint = solve_int16(A[h], stats)
        relT.append(np.linalg.norm(Tint - Tref) / np.linalg.norm(Tref))
        T8 = q8(Tint)
        relU.append(np.linalg.norm(T8 @ vb[h] - Tref @ vb[h]) / np.linalg.norm(Tref @ vb[h]))
        relW.append(np.linalg.norm(T8 @ kbe[h] - Tref @ kbe[h]) / np.linalg.norm(Tref @ kbe[h]))
    print(f"int16 forward-subst solve : T relerr vs fp64  mean {np.mean(relT):.3e}  max {np.max(relT):.3e}"
          f"   (int8-matmul solve device T = 4.4e-3)")
    print(f"int32 accumulator peak    : {stats[0]:,}  (limit 2^31 = {2**31:,})  "
          f"{'OK' if stats[0] < 2**31 else 'OVERFLOW'}")
    print(f"consume T int16->int8 in1 : U {np.mean(relU):.3e}  W {np.mean(relW):.3e}   "
          f"(device ceiling U 6.3e-3 / W 1.09e-2 ; current int8-solve U 1.12e-2 / W 1.63e-2)")


if __name__ == "__main__":
    main()
