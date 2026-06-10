"""max|P| predictability probe — feasibility of killing gdn_merge_packed's multi-pass gain search.

The GDNSolveHVXMixHMX solve runs the 64^3 HMX kernel 2-3x per logical matmul: PASS 1/2 just measure max|P| (the
int32-codes output magnitude) to pick the tight output gain g2=127/max|P|; their outputs are thrown away.
That probing is ~66% of the path's VTCM traffic (Agent/current/gdn_solve.md).

This probe asks: can we PREDICT max|P| from cheap input norms (no extra HMX run), accurately enough that a
single pass at gain=127/est fills the int8 output range (so the gain search collapses to 1 pass)?

We replay the real block-recursive forward-subst on REAL golden A, and at every merge matmul (A_ik@T_kj and
T_ii@S_ij) record actual max|P| (int8 codes domain) vs predictors:
  - LOOSE   : the device's current PASS-1 constant 64*127*127 (worst case)
  - Holder  : min( max_i ||cA[i,:]||_1 * max|cT| ,  max_c ||cT[:,c]||_1 * max|cA| )  -- a true upper bound

Metric = OUTPUT FILL = 127 / (est/actual): the int8 code level the OUTPUT lands at when we scale by 127/est.
Want fill >= ~32 (>=5 bits) so the integer maxabs rounding error is <~3% -> one pass suffices (no PASS 2).
Reproduce: source scripts/env.sh && python scripts/gdn_solve_maxp_probe.py   (C=256 H=32 real golden)
"""
import os, sys, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C = int(os.environ.get("C", "256"))
import gdn_onnx_kernel as gok; gok.CHUNK = C
import gdn_ref_kernel as grk; grk.CHUNK = C
from gdn_onnx_kernel import _golden_chunk_args, l2norm_lastdim, _masks
from gdn_solve_int16_model import GOLDEN

BL = 64; NB = C // BL
LOOSE = 64 * 127 * 127

# ---- real chunk + strictly-lower A (same load as gdn_solve_taylor_probe.py) ----
npz = next(f for f in sorted(glob.glob(os.path.join(GOLDEN, "*.npz")))
           if np.load(f)["query"].shape[1] >= C)
print(f"golden = {os.path.basename(npz)}  C={C} NB={NB}")
qc, kc, vc, gc, betac, S_in = _golden_chunk_args(npz, 0)
tl, sl, cu, ey = _masks(C, "cpu", torch.float64)
kn = l2norm_lastdim(kc); k_beta = kn * betac.unsqueeze(-1)
g = torch.matmul(gc.unsqueeze(-2), cu.reshape(1, 1, C, C)).squeeze(-2)
diff = g.unsqueeze(-1) - g.unsqueeze(-2); decay = torch.exp(diff * tl) * tl
A = ((-torch.matmul(k_beta, kn.transpose(-1, -2)) * decay) * sl).double().numpy()[0]  # [H,C,C]
H = A.shape[0]
Texact = np.stack([np.linalg.inv(np.eye(C) - A[h]) for h in range(H)])


def qsym_i8(x):
    mx = float(np.abs(x).max())
    if mx < 1e-30:
        return np.zeros_like(x, dtype=np.int64), 1.0
    s = mx / 127.0
    return np.round(x / s).clip(-127, 127).astype(np.int64), s


K_LIST = [1.0, 2.0, 3.0, 4.0]                      # correction: assume max|P| ~ Holder/K (K>1 tightens, risks clip)

def _quant_relerr(P, gain):
    deq = np.clip(np.round(P * gain), -127, 127) / gain
    return float(np.linalg.norm(deq - P) / (np.linalg.norm(P) + 1e-9))

def record(cA, cT, kind, recs):
    P = (cA @ cT).astype(np.float64)              # exact int32-codes matmul
    maxP = max(float(np.abs(P).max()), 1.0)
    est_row = float(np.abs(cA).sum(1).max()) * max(float(np.abs(cT).max()), 1.0)
    est_col = float(np.abs(cT).sum(0).max()) * max(float(np.abs(cA).max()), 1.0)
    est = max(min(est_row, est_col), 1.0)         # Holder upper bound (>= maxP, never saturates at K=1)
    re_exact = _quant_relerr(P, 127.0 / maxP)     # PASS3 today (true maxP, after 2 probe runs)
    preds = []
    for K in K_LIST:
        gpred = K * 127.0 / est                   # norm-predicted gain, NO probe run
        sat = float(np.mean(np.abs(P * gpred) > 127.5))   # fraction of entries clipped (the oc killer)
        preds.append((_quant_relerr(P, gpred), sat))
    recs.append((kind, re_exact, preds))


recs = []
for h in range(H):
    Ah, Th = A[h], Texact[h]
    for d in range(1, NB):
        for j in range(NB - d):
            i = j + d
            for k in range(j, i):                 # inner: A_ik @ T_kj
                cA, _ = qsym_i8(Ah[i*BL:(i+1)*BL, k*BL:(k+1)*BL])
                cT, _ = qsym_i8(Th[k*BL:(k+1)*BL, j*BL:(j+1)*BL])
                record(cA, cT, "inner", recs)
            Sij = sum(Ah[i*BL:(i+1)*BL, k*BL:(k+1)*BL] @ Th[k*BL:(k+1)*BL, j*BL:(j+1)*BL]
                      for k in range(j, i))         # final: T_ii @ S_ij
            cA, _ = qsym_i8(Th[i*BL:(i+1)*BL, i*BL:(i+1)*BL])
            cT, _ = qsym_i8(Sij)
            record(cA, cT, "final", recs)

print(f"\n{len(recs)} merge matmuls ({sum(r[0]=='inner' for r in recs)} inner, "
      f"{sum(r[0]=='final' for r in recs)} final) over {H} heads")
print("Output-quant relerr per matmul: PASS3-exact (true maxP, today, after 2 probe runs) vs norm-PREDICTED")
print("(gain = K*127/Holder, ZERO probe runs). sat% = entries clipped (the oc killer). Lower relerr better.\n")
for kind in ("inner", "final", "all"):
    rs = [r for r in recs if kind == "all" or r[0] == kind]
    re_ex = np.array([r[1] for r in rs])
    print(f"--- {kind} (n={len(rs)}) ---   PASS3-exact relerr: p50={np.percentile(re_ex,50):.4f} "
          f"p90={np.percentile(re_ex,90):.4f}")
    print(f"      {'predictor':>16} | {'relerr p50':>10} {'relerr p90':>10} | {'sat% p50':>9} {'sat% max':>9}")
    for ki, K in enumerate(K_LIST):
        rp = np.array([r[2][ki][0] for r in rs]); sat = np.array([r[2][ki][1] for r in rs]) * 100
        print(f"      K={K:<3.0f}127/Holder | {np.percentile(rp,50):10.4f} {np.percentile(rp,90):10.4f} | "
              f"{np.percentile(sat,50):9.2f} {sat.max():9.2f}")
print("\nVERDICT: a predictor whose relerr p90 ~ PASS3-exact AND sat%~0 => that K kills the 2 probe runs safely.")
print("""
FINDING (real golden p29_L00, C=256): norm-prediction as the FINAL output scale is too lossy (K=1 relerr
~0.08-0.11 vs PASS3-exact ~0.02-0.03; block-recursive propagation would push oc up — the code's own warning).
BUT the Holder bound's output fill is ~37 (>=5-bit), enough for PASS-1's integer maxabs to be ACCURATE in
ONE shot. => SAFE optimization (lever #1c): replace PASS-1's LOOSE constant gain (fill ~1, needs PASS-2
refine) with the Holder norm-predicted gain; PASS-1 then measures max|P| accurately, so PASS-2 is dropped
and PASS-3 still uses the MEASURED max|P| (oc unchanged). Saves 1 HMX run + 1 maxabs per matmul (~33% of
the path's VTCM traffic). The Holder norm (max-row-|sum| of act, max|wt|) is ~free: fp_pack_effbias already
computes the wt column-sums for bias; the act row-sums can piggyback on the act-pack pass.""")
