#!/usr/bin/env python3
"""M2 design probe: block-recursive C=128 inverse with HMX-quant merge — does u8i8 hold, or need w8a16?

The #2 block-recursive inverse builds T=(I-A)^-1 for C=128 as a 2x2 block recursion (BL=64):
  L = I - A (strictly-lower A) = [[L11,0],[L21,L22]];  T = [[T11,0],[T21,T22]]
  T11 = L11^-1, T22 = L22^-1            -- diagonal 64x64 solves, HVX int16 forward-subst (kept int16)
  T21 = T22 @ (A21 @ T11)               -- the 2 MERGE matmuls (64^3), run on HMX
Diagonals stay int16; we vary ONLY the merge-matmul operand precision:
  w16a16 (both int16, reference)  |  w8a16 (int16 act + int8 wt, 2x HMX)  |  u8i8 (both int8, 1x HMX, cheapest)
This is the accuracy gate that the C=64 p00 probe could not settle (that golden had only ~13 real tokens,
so the 64-apart off-diagonal merge block was zero). Here we use a REAL >=128-token GDN chunk so the merge
is non-degenerate. Decision feeds M2/M5: default u8i8, fall back to w8a16 if it exceeds the int8-T ceiling.

Run:  GDN_NO_VSCALE=1 .venv/bin/python scripts/gdn_blockrec_c128_probe.py
"""
import os, sys, glob, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("GDN_NO_VSCALE", "1")
import gdn_onnx_kernel as gok
gok.CHUNK = 128                                            # build A at the prefill chunk size
from gdn_onnx_kernel import _golden_chunk_args, l2norm_lastdim, _masks
from gdn_solve_int16_model import solve_int16, GOLDEN

C = 128


def pick_golden(min_tokens=128):
    for f in sorted(glob.glob(os.path.join(GOLDEN, "*.npz"))):
        if np.load(f)["query"].shape[1] >= min_tokens:
            return f
    raise SystemExit("no golden chunk with >=128 real tokens")


def build_A(npz):
    """Strictly-lower [H,C,C] A for chunk 0 of a real GDN layer (same formula as gdn_merge_precision_probe)."""
    qc, kc, vc, gc, betac, S_in = _golden_chunk_args(npz, 0)
    tl, sl, cu, ey = _masks(C, "cpu", torch.float64)
    kn = l2norm_lastdim(kc); k_beta = kn * betac.unsqueeze(-1); v_beta = vc * betac.unsqueeze(-1)
    g = torch.matmul(gc.unsqueeze(-2), cu.reshape(1, 1, C, C)).squeeze(-2)
    eg = torch.exp(g).unsqueeze(-1)
    diff = g.unsqueeze(-1) - g.unsqueeze(-2); decay = torch.exp(diff * tl) * tl
    A = ((-torch.matmul(k_beta, kn.transpose(-1, -2)) * decay) * sl).double().numpy()[0]
    vb = v_beta.double().numpy()[0]; kbe = (k_beta * eg).double().numpy()[0]
    return A, vb, kbe


def q(M, bits):
    lim = (1 << (bits - 1)) - 1
    s = (np.abs(M).max() / lim) or 1e-12
    return np.round(M / s).clip(-lim, lim).astype(np.int64), s


def matmul_q(Aop, Bop, a_bits, b_bits):
    Aq, sa = q(Aop, a_bits); Bq, sb = q(Bop, b_bits)
    acc = Aq @ Bq
    return acc.astype(np.float64) * (sa * sb), int(np.abs(acc).max())


def block_recursive_T(A_h, BL, a_bits, b_bits):
    """2x2 block inverse; diagonal blocks via int16 forward-subst, merge matmuls at (a_bits,b_bits)."""
    n = BL
    A11, A21, A22 = A_h[:n, :n], A_h[n:, :n], A_h[n:, n:]
    T11 = solve_int16(A11, [0]).astype(np.float64)         # HVX int16 diagonal solve
    T22 = solve_int16(A22, [0]).astype(np.float64)
    M, p1 = matmul_q(A21, T11, b_bits, a_bits)             # MERGE 1: A21 @ T11
    T21m, p2 = matmul_q(T22, M, a_bits, b_bits)            # MERGE 2: T22 @ (A21 @ T11)
    T = np.zeros((C, C)); T[:n, :n] = T11; T[n:, n:] = T22; T[n:, :n] = T21m
    return T, max(p1, p2)


def main():
    npz = pick_golden(C)
    A, vb, kbe = build_A(npz)
    H = A.shape[0]; BL = C // 2
    print(f"golden={os.path.basename(npz)}  C={C}  H={H}  BL={BL}")
    # coupling sanity: is the off-diagonal merge block non-trivial here?
    r = np.mean([np.linalg.norm(np.linalg.inv(np.eye(C) - A[h])[BL:, :BL]) /
                 np.linalg.norm(np.linalg.inv(np.eye(C) - A[h])) for h in range(H)])
    print(f"mean ||T21||/||T|| over heads = {r:.3e}  ({'NON-degenerate, merge matters' if r > 1e-3 else 'still degenerate'})")

    def q8(x):
        s = (np.abs(x).max() / 127.0) or 1e-12
        return np.round(x / s).clip(-127, 127) * s

    def consume(T8, Tref, rhs):
        return np.linalg.norm(T8 @ rhs - Tref @ rhs) / np.linalg.norm(Tref @ rhs)

    variants = {"w16a16": (16, 16), "w8a16": (8, 16), "u8i8(=w8a8)": (8, 8)}
    rows = {k: [[], [], []] for k in variants}; rows["fwd-subst-i16"] = [[], [], []]
    peaks = {k: 0 for k in variants}
    for h in range(H):
        Tref = np.linalg.inv(np.eye(C) - A[h])
        for nm, (ab, bb) in variants.items():
            T, pk = block_recursive_T(A[h], BL, ab, bb); peaks[nm] = max(peaks[nm], pk)
            T8 = q8(T)
            rows[nm][0].append(np.linalg.norm(T - Tref) / np.linalg.norm(Tref))
            rows[nm][1].append(consume(T8, Tref, vb[h])); rows[nm][2].append(consume(T8, Tref, kbe[h]))
        Tf = solve_int16(A[h], [0]).astype(np.float64); Tf8 = q8(Tf)
        rows["fwd-subst-i16"][0].append(np.linalg.norm(Tf - Tref) / np.linalg.norm(Tref))
        rows["fwd-subst-i16"][1].append(consume(Tf8, Tref, vb[h])); rows["fwd-subst-i16"][2].append(consume(Tf8, Tref, kbe[h]))

    print(f"\n{'merge variant':16} {'T relerr':>12} {'U (int8)':>10} {'W (int8)':>10} {'acc peak':>14}")
    for nm in ("fwd-subst-i16", "w16a16", "w8a16", "u8i8(=w8a8)"):
        t, u, w = rows[nm]; pk = f"{peaks.get(nm,0):,}" if nm in peaks else "-"
        print(f"{nm:16} {np.mean(t):>12.3e} {np.mean(u):>10.3e} {np.mean(w):>10.3e} {pk:>14}")
    print(f"\nint32 limit 2^31 = {2**31:,}")
    print("ceiling (T exact->int8): U~6.3e-3 / W~1.09e-2.  Viable if a variant's U/W ~= fwd-subst-i16.")


if __name__ == "__main__":
    main()
