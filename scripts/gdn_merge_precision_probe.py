#!/usr/bin/env python3
"""Does the #2 block-recursive MERGE need w16a16, or do w8a16 / w8a8 suffice?

#2 builds T=(I-A)^-1 block-recursively: diagonal BL x BL blocks inverted on HVX (int16, untouched here),
off-diagonal blocks merged by HMX matmuls.  HMX is native u8 x i8, so the merge operand precision sets
the HMX cost: w8a8=1x, w8a16=2x, w16a16=3-4x the 8-bit matmuls (reference_hmx_arch).  The cheaper the
merge, the smaller BL can go (rebalancing the HVX||HMX pipeline) -> lower ceiling.  Open question: does
int8 on a merge operand keep T accurate?  Both merge operands are DYNAMIC inverse intermediates (a T
block and an A block), unlike the end-to-end GDN where the int8 side is a static weight.

2x2 block recursion on each head's 64x64 A (BL=32):
  L = I - A (strictly-lower A) = [[L11,0],[L21,L22]];  T = L^-1 = [[T11,0],[T21,T22]]
  T11=L11^-1, T22=L22^-1 (diagonal, kept int16);  T21 = -T22 @ (L21 @ T11)  (the 2 MERGE matmuls)
We hold the diagonals at int16 and vary ONLY the merge-matmul operand precision.

Run:  GDN_NO_VSCALE=1 .venv/bin/python scripts/gdn_merge_precision_probe.py

CAVEAT: the current p00_L00 golden chunk has only ~13 REAL tokens (padded to 64), so its 32-apart
off-diagonal merge blocks are exactly zero -> all precisions tie (degenerate). To actually separate
w16a16/w8a16/w8a8 you need a C=128/256 prefill golden whose real sequence fills the off-diagonal
blocks. This script is the ready harness for that data; it does not settle the question on p00_L00.
"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("GDN_NO_VSCALE", "1")
from gdn_solve_int16_model import solve_int16, GOLDEN
from gdn_onnx_kernel import _golden_chunk_args, l2norm_lastdim, _masks, CHUNK


def q(M, bits):
    """Per-tensor symmetric quant->dequant to `bits` (16 or 8)."""
    lim = (1 << (bits - 1)) - 1
    s = (np.abs(M).max() / lim) or 1e-12
    return np.round(M / s).clip(-lim, lim).astype(np.int64), s


def matmul_q(Aop, Bop, a_bits, b_bits):
    """int matmul with operands quantized to a_bits/b_bits, int32-accumulated, dequantized."""
    Aq, sa = q(Aop, a_bits); Bq, sb = q(Bop, b_bits)
    acc = Aq @ Bq
    return acc.astype(np.float64) * (sa * sb), int(np.abs(acc).max())


def block_recursive_T(L, BL, a_bits, b_bits):
    """2x2 block inverse; diagonal blocks exact-int16, merge matmuls at (a_bits,b_bits). returns T, peak."""
    C = L.shape[0]; n = BL
    L11, L21, L22 = L[:n, :n], L[n:, :n], L[n:, n:]
    A11 = np.eye(n) - L11; A22 = np.eye(n) - L22       # strictly-lower diag blocks
    T11 = solve_int16(A11, [0]).astype(np.float64)      # HVX diagonal solve (int16) -- unchanged
    T22 = solve_int16(A22, [0]).astype(np.float64)
    M, p1 = matmul_q(L21, T11, b_bits, a_bits)          # MERGE matmul 1
    T21m, p2 = matmul_q(T22, M, a_bits, b_bits)         # MERGE matmul 2
    T21 = -T21m
    T = np.zeros((C, C)); T[:n, :n] = T11; T[n:, n:] = T22; T[n:, :n] = T21
    return T, max(p1, p2)


def main():
    a = _golden_chunk_args(os.path.join(GOLDEN, "p00_L00.npz"), 0)
    qc, kc, vc, gc, betac, S_in = a
    tl, sl, cu, ey = _masks(CHUNK, "cpu", torch.float64)
    kn = l2norm_lastdim(kc); k_beta = kn * betac.unsqueeze(-1); v_beta = vc * betac.unsqueeze(-1)
    g = torch.matmul(gc.unsqueeze(-2), cu.reshape(1, 1, CHUNK, CHUNK)).squeeze(-2)
    eg = torch.exp(g).unsqueeze(-1)
    diff = g.unsqueeze(-1) - g.unsqueeze(-2); decay = torch.exp(diff * tl) * tl
    A = ((-torch.matmul(k_beta, kn.transpose(-1, -2)) * decay) * sl).double().numpy()[0]
    vb = v_beta.double().numpy()[0]; kbe = (k_beta * eg).double().numpy()[0]
    H, C, _ = A.shape; BL = C // 2

    def q8(x):
        s = (np.abs(x).max() / 127.0) or 1e-12
        return np.round(x / s).clip(-127, 127) * s

    def consume(T8, Tref, rhs):                          # downstream U/W use T at int8 (the in[1] ceiling)
        return np.linalg.norm(T8 @ rhs - Tref @ rhs) / np.linalg.norm(Tref @ rhs)

    variants = {"w16a16": (16, 16), "w8a16": (8, 16), "w8a8": (8, 8)}
    rows = {k: [[], [], []] for k in variants}
    rows["fwd-subst-i16"] = [[], [], []]
    peaks = {k: 0 for k in variants}
    for h in range(H):
        Tref = np.linalg.inv(np.eye(C) - A[h])
        L = np.eye(C) - A[h]
        for nm, (ab, bb) in variants.items():
            T, pk = block_recursive_T(L, BL, ab, bb); peaks[nm] = max(peaks[nm], pk)
            T8 = q8(T)
            rows[nm][0].append(np.linalg.norm(T - Tref) / np.linalg.norm(Tref))
            rows[nm][1].append(consume(T8, Tref, vb[h]))
            rows[nm][2].append(consume(T8, Tref, kbe[h]))
        Tf = solve_int16(A[h], [0]).astype(np.float64); Tf8 = q8(Tf)   # reference: full int16 fwd-subst
        rows["fwd-subst-i16"][0].append(np.linalg.norm(Tf - Tref) / np.linalg.norm(Tref))
        rows["fwd-subst-i16"][1].append(consume(Tf8, Tref, vb[h]))
        rows["fwd-subst-i16"][2].append(consume(Tf8, Tref, kbe[h]))

    print(f"{'merge variant':16} {'T relerr':>12} {'U (int8)':>10} {'W (int8)':>10} {'acc peak':>14}")
    for nm in ("fwd-subst-i16", "w16a16", "w8a16", "w8a8"):
        t, u, w = rows[nm]
        pk = f"{peaks.get(nm, 0):,}" if nm in peaks else "-"
        print(f"{nm:16} {np.mean(t):>12.3e} {np.mean(u):>10.3e} {np.mean(w):>10.3e} {pk:>14}")
    print(f"\nint32 limit 2^31 = {2**31:,}")
    print("downstream ceiling (T exact->int8): U 6.3e-3 / W 1.09e-2 ; old full-int8-matmul chain: U 1.1e-2 / W 1.6e-2")
    print("=> a merge variant is viable if its U/W ~= fwd-subst-i16 (no extra error past the int8-T ceiling).")


if __name__ == "__main__":
    main()
