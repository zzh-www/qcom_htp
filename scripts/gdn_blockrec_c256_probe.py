#!/usr/bin/env python3
"""M4 design probe: block-recursive C=256 inverse (nb=4, BL=64) with u8i8-quant merges.

Validates the nb=4 block-recursive formula + the per-merge int8 scale chain BEFORE touching the
device op, against np.linalg.inv on REAL >=256-token GDN data.

Lower-tri inverse of L = I - A (strictly-lower A), 4x4 blocks A_ij (i,j in 0..3), BL=64:
  T_ii = inv(I - A_ii)                                   -- 4 HVX int16 diagonal solves
  T_ij = T_ii @ ( sum_{k=j}^{i-1} A_ik @ T_kj )   (i>j)  -- off-diagonal, increasing i-j
The 6 off-diag blocks in dependency order: (1,0),(2,0),(3,0),(2,1),(3,1),(3,2).
Each inner sum term  A_ik @ T_kj  is one 64^3 merge; the  T_ii @ (.)  is one more.

Quant model mirrors the device op gdn_hmx_merge: every 64^3 matmul is u8i8 (act recentred to u8 zp128,
wt int8), scale of the product estimated EXACTLY from max|P_int| (the int matmul the HMX computes).

Run:  GDN_NO_VSCALE=1 .venv/bin/python scripts/gdn_blockrec_c256_probe.py [--heads N]
"""
import os, sys, glob, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("GDN_NO_VSCALE", "1")
import gdn_onnx_kernel as gok
C = 256
gok.CHUNK = C
from gdn_onnx_kernel import _golden_chunk_args, l2norm_lastdim, _masks
from gdn_solve_int16_model import solve_int16, GOLDEN

BL = 64
NB = C // BL  # 4


def pick_golden(min_tokens=C):
    for f in sorted(glob.glob(os.path.join(GOLDEN, "*.npz"))):
        if np.load(f)["query"].shape[1] >= min_tokens:
            return f
    raise SystemExit(f"no golden chunk with >={min_tokens} real tokens")


def build_A(npz):
    qc, kc, vc, gc, betac, S_in = _golden_chunk_args(npz, 0)
    tl, sl, cu, ey = _masks(C, "cpu", torch.float64)
    kn = l2norm_lastdim(kc); k_beta = kn * betac.unsqueeze(-1); v_beta = vc * betac.unsqueeze(-1)
    g = torch.matmul(gc.unsqueeze(-2), cu.reshape(1, 1, C, C)).squeeze(-2)
    eg = torch.exp(g).unsqueeze(-1)
    diff = g.unsqueeze(-1) - g.unsqueeze(-2); decay = torch.exp(diff * tl) * tl
    A = ((-torch.matmul(k_beta, kn.transpose(-1, -2)) * decay) * sl).double().numpy()[0]
    vb = v_beta.double().numpy()[0]; kbe = (k_beta * eg).double().numpy()[0]
    return A, vb, kbe


def u8i8_merge(act_f, wt_f):
    """One 64^3 u8i8 merge mirroring the device op.
    act_f, wt_f are float [64,64]. Returns (product_float[64,64], peak_int).
    act -> symmetric int8 code in [-127,127] then +128 (u8 zp128); wt -> symmetric int8.
    Scale of P estimated exactly from max|P_int| (= the int product the HMX computes)."""
    sa = (np.abs(act_f).max() / 127.0) or 1e-12
    sw = (np.abs(wt_f).max() / 127.0) or 1e-12
    aq = np.round(act_f / sa).clip(-127, 127).astype(np.int64)   # signed int8 code
    wq = np.round(wt_f / sw).clip(-127, 127).astype(np.int64)
    P_int = aq @ wq                                              # exact signed int product
    return P_int.astype(np.float64) * (sa * sw), int(np.abs(P_int).max())


def block_recursive_T(A_h):
    """nb=4 block-recursive lower-tri inverse with u8i8 merges. Returns (T[256,256], peak_int)."""
    n = BL
    Aij = [[A_h[i*n:(i+1)*n, j*n:(j+1)*n] for j in range(NB)] for i in range(NB)]
    T = [[None]*NB for _ in range(NB)]
    peak = 0
    # diagonals: HVX int16 forward-subst (exactly what the device does)
    for i in range(NB):
        T[i][i] = solve_int16(Aij[i][i], [0]).astype(np.float64)
    # off-diagonals in increasing i-j (so all T_kj with k>j, k<i are ready)
    for d in range(1, NB):
        for j in range(NB - d):
            i = j + d
            # inner sum S = sum_{k=j}^{i-1} A_ik @ T_kj   (each term a 64^3 u8i8 merge)
            S = np.zeros((n, n), dtype=np.float64)
            for k in range(j, i):
                term, pk = u8i8_merge(Aij[i][k], T[k][j]); peak = max(peak, pk)
                S += term
            # T_ij = T_ii @ S   (one more 64^3 u8i8 merge)
            Tij, pk = u8i8_merge(T[i][i], S); peak = max(peak, pk)
            T[i][j] = Tij
    # assemble
    out = np.zeros((C, C))
    for i in range(NB):
        for j in range(i+1):
            out[i*n:(i+1)*n, j*n:(j+1)*n] = T[i][j]
    return out, peak


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--heads", type=int, default=0)
    args = ap.parse_args()
    npz = pick_golden(C)
    A, vb, kbe = build_A(npz)
    H = A.shape[0] if args.heads == 0 else min(args.heads, A.shape[0])
    print(f"golden={os.path.basename(npz)}  C={C}  NB={NB}  BL={BL}  H={H}")

    def q8(x):
        s = (np.abs(x).max() / 127.0) or 1e-12
        return np.round(x / s).clip(-127, 127) * s

    def consume(T8, Tref, rhs):
        return np.linalg.norm(T8 @ rhs - Tref @ rhs) / np.linalg.norm(Tref @ rhs)

    whole, blkdiag, blkoff = [], [], []
    u_err, w_err, fwd_w = [], [], []
    peak = 0
    for h in range(H):
        Aref = np.eye(C) - A[h]
        Tref = np.linalg.inv(Aref)
        T, pk = block_recursive_T(A[h]); peak = max(peak, pk)
        whole.append(np.linalg.norm(T - Tref) / np.linalg.norm(Tref))
        # diag blocks vs off-diag blocks
        dm = np.mean([np.linalg.norm(T[i*BL:(i+1)*BL, i*BL:(i+1)*BL] - Tref[i*BL:(i+1)*BL, i*BL:(i+1)*BL])
                      / (np.linalg.norm(Tref[i*BL:(i+1)*BL, i*BL:(i+1)*BL]) + 1e-12) for i in range(NB)])
        om = np.mean([np.linalg.norm(T[i*BL:(i+1)*BL, j*BL:(j+1)*BL] - Tref[i*BL:(i+1)*BL, j*BL:(j+1)*BL])
                      / (np.linalg.norm(Tref[i*BL:(i+1)*BL, j*BL:(j+1)*BL]) + 1e-12)
                      for i in range(NB) for j in range(i)])
        blkdiag.append(dm); blkoff.append(om)
        T8 = q8(T)
        u_err.append(consume(T8, Tref, vb[h])); w_err.append(consume(T8, Tref, kbe[h]))
        Tf = solve_int16(A[h], [0]).astype(np.float64); Tf8 = q8(Tf)
        fwd_w.append(consume(Tf8, Tref, kbe[h]))

    print(f"\nblock-recursive u8i8 nb=4:")
    print(f"  whole-T relerr vs np.linalg.inv: mean {np.mean(whole):.3e}  max {np.max(whole):.3e}")
    print(f"  block diag relerr mean {np.mean(blkdiag):.3e}   off-diag relerr mean {np.mean(blkoff):.3e}")
    print(f"  downstream U(int8) {np.mean(u_err):.3e}   W(int8) {np.mean(w_err):.3e}")
    print(f"  fwd-subst-i16 baseline W(int8) {np.mean(fwd_w):.3e}")
    print(f"  merge int peak {peak:,}  (int32 limit 2^31={2**31:,}, {'SAFE' if peak < 2**31 else 'OVERFLOW'})")


if __name__ == "__main__":
    main()
