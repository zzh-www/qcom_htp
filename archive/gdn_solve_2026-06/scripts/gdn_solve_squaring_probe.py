#!/usr/bin/env python3
"""De-risk probe: is the HMX-SHAPED solve form (dense-matmul squaring, NOT serial forward-subst) accurate
ENOUGH for the int8 downstream U/W consume?  Compares two int16 solves of T=(I-A)^-1 on real GDN A:
  (a) forward-substitution (the current HVX op) — accurate but serial + thin matmuls (HMX-hostile);
  (b) squaring/Neumann  T = prod_k (I + A^(2^k)) — ~6 DENSE 64x64 matmuls (HMX-shaped, K=64), int16
      requant between every matmul (the operand re-narrowing that degraded the old int8 chain).
A is strictly-lower 64x64 -> nilpotent (A^64=0) -> 6 factors are exact in fp.  The question is whether
int16-requant-between-dense-matmuls keeps T below the int8-in[1] downstream ceiling (~6e-3 U / 1.1e-2 W).
If yes, the squaring form can run on HMX (overlapping HVX) without hurting oc.

Run:  GDN_NO_VSCALE=1 .venv/bin/python scripts/gdn_solve_squaring_probe.py
"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("GDN_NO_VSCALE", "1")
from gdn_solve_int16_model import solve_int16, GOLDEN
from gdn_onnx_kernel import _golden_chunk_args, l2norm_lastdim, _masks, CHUNK


def _q16(M):
    s = (np.abs(M).max() / 32767.0) or 1e-12
    return np.round(M / s).clip(-32767, 32767).astype(np.int64), s


def solve_squaring_int16(Ah, nsteps=6):
    """T=(I-A)^-1 = prod_{k}(I+A^(2^k)).  Dense int16xint16->int32 matmuls, requant operands to int16
    between steps (mimics what an HMX matmul chain would carry)."""
    C = Ah.shape[0]; I = np.eye(C)
    P = Ah.copy()                                  # A^(2^0)
    T = I + Ah
    peak = 0
    for k in range(1, nsteps):
        Pc, sP = _q16(P)                           # P @ P  (int32 accum, dequant, requant to int16)
        acc = Pc @ Pc; peak = max(peak, int(np.abs(acc).max()))
        Pc2, sP2 = _q16(acc.astype(np.float64) * (sP * sP)); P = Pc2.astype(np.float64) * sP2
        if np.abs(P).max() < 1e-15:
            break
        Tc, sT = _q16(T); Pc3, sP3 = _q16(P)       # T = T + T@P  (int32 accum, dequant)
        accT = Tc @ Pc3; peak = max(peak, int(np.abs(accT).max()))
        T = T + accT.astype(np.float64) * (sT * sP3)
        Tc2, sT2 = _q16(T); T = Tc2.astype(np.float64) * sT2   # requant T (operand for next step)
    return T, peak


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
    H, C, _ = A.shape

    def q8(x):
        s = (np.abs(x).max() / 127.0) or 1e-12
        return np.round(x / s).clip(-127, 127) * s

    def consume(T8, Tref, rhs):
        return np.linalg.norm(T8 @ rhs - Tref @ rhs) / np.linalg.norm(Tref @ rhs)

    rows = {"fwd-subst": [[], [], []], "squaring": [[], [], []]}
    peak = 0
    for h in range(H):
        Tref = np.linalg.inv(np.eye(C) - A[h])
        Tf = solve_int16(A[h], [0])
        Ts, pk = solve_squaring_int16(A[h]); peak = max(peak, pk)
        for nm, T in (("fwd-subst", Tf), ("squaring", Ts)):
            T8 = q8(T)
            rows[nm][0].append(np.linalg.norm(T - Tref) / np.linalg.norm(Tref))
            rows[nm][1].append(consume(T8, Tref, vb[h]))
            rows[nm][2].append(consume(T8, Tref, kbe[h]))
    print(f"{'method':12} {'T relerr':>12} {'U (int8)':>10} {'W (int8)':>10}")
    for nm, (t, u, w) in rows.items():
        print(f"{nm:12} {np.mean(t):>12.3e} {np.mean(u):>10.3e} {np.mean(w):>10.3e}")
    print(f"\nsquaring int32 accumulator peak: {peak:,}  (limit 2^31 = {2**31:,})  "
          f"{'OK' if peak < 2**31 else 'OVERFLOW'}")
    print("downstream ceiling (T exact->int8): U 6.3e-3 / W 1.09e-2 ; old int8-matmul chain: U 1.1e-2 / W 1.6e-2")
    print("=> if squaring U/W ~= fwd-subst U/W, the HMX-shaped form is accuracy-viable.")


if __name__ == "__main__":
    main()
