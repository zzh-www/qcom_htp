"""Static-vs-dynamic quantization feasibility for the GDN block-recursive solve (real golden, C=256).

Question (user's target architecture): can we drop DYNAMIC per-block quantization (runtime maxabs) for
STATIC fixed-scale quantization — which lets us preprocess A/T once, make acc a pure int32 add, and run
pure vector int ops — and still hold oc? If int8-static loses too much, does int16-static recover it?

Replays the block-recursive forward-subst with quantized operands at each merge matmul, under:
  scale mode: DYNAMIC (per-block maxabs)  vs  STATIC (one fixed scale per operand-type, from the global range)
  bitwidth:   int8 (qmax 127)            vs  int16 (qmax 32767)
Metric = end-to-end off-diagonal oc = ||T_q - Texact|| / ||Texact|| (lower=better). Diagonal blocks use the
fp64 inverse (we isolate the MERGE quant, which is the 51% glue we want to make static).

Reproduce: source scripts/env.sh && python scripts/gdn_solve_static_quant_probe.py
"""
import os, glob, numpy as np, torch
import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
C = 256; BL = 64; NB = C // BL
import gdn_onnx_kernel as gok; gok.CHUNK = C
import gdn_ref_kernel as grk; grk.CHUNK = C
from gdn_onnx_kernel import _golden_chunk_args, l2norm_lastdim, _masks
from gdn_solve_int16_model import GOLDEN

npz = next(f for f in sorted(glob.glob(os.path.join(GOLDEN, "*.npz"))) if np.load(f)["query"].shape[1] >= C)
qc, kc, vc, gc, betac, S_in = _golden_chunk_args(npz, 0)
tl, sl, cu, ey = _masks(C, "cpu", torch.float64)
kn = l2norm_lastdim(kc); k_beta = kn * betac.unsqueeze(-1)
g = torch.matmul(gc.unsqueeze(-2), cu.reshape(1, 1, C, C)).squeeze(-2)
diff = g.unsqueeze(-1) - g.unsqueeze(-2); decay = torch.exp(diff * tl) * tl
A = ((-torch.matmul(k_beta, kn.transpose(-1, -2)) * decay) * sl).double().numpy()[0]   # [H,C,C]
H = A.shape[0]
Texact = np.stack([np.linalg.inv(np.eye(C) - A[h]) for h in range(H)])
print(f"golden={os.path.basename(npz)} C={C} NB={NB} H={H}")

# global ranges (what a STATIC scale must cover)
amax = float(np.abs(A).max())
tmax = float(np.abs(Texact).max())
# per block-distance range of T (does magnitude grow with distance? -> layered static scale)
print(f"|A|max={amax:.4f}  |T|max={tmax:.4f}")
for d in range(NB):
    blocks = [np.abs(Texact[h, (j+d)*BL:(j+d+1)*BL, j*BL:(j+1)*BL]).max()
              for h in range(H) for j in range(NB-d)]
    print(f"  T block-dist {d}: |T|max range [{min(blocks):.4f}, {max(blocks):.4f}]")

def quant(x, scale, qmax):
    return np.clip(np.round(x / scale), -qmax, qmax) * scale   # fake-quant (dequant back to float)

def solve(h, scale_mode, bits):
    qmax = 127 if bits == 8 else 32767
    Ah = A[h]
    # static scales: one per operand type, from the global range (a single fixed scale for ALL blocks)
    sA_static = amax / qmax
    sT_static = tmax / qmax
    T = np.zeros((C, C))
    for i in range(NB):                       # diagonal blocks: fp64 inverse (isolate merge quant)
        T[i*BL:(i+1)*BL, i*BL:(i+1)*BL] = np.linalg.inv(np.eye(BL) - Ah[i*BL:(i+1)*BL, i*BL:(i+1)*BL])
    for d in range(1, NB):
        for j in range(NB-d):
            i = j + d
            S = np.zeros((BL, BL))
            for k in range(j, i):
                Aik = Ah[i*BL:(i+1)*BL, k*BL:(k+1)*BL]
                Tkj = T[k*BL:(k+1)*BL, j*BL:(j+1)*BL]
                if scale_mode == "dyn":
                    sA = max(np.abs(Aik).max(), 1e-30) / qmax
                    sT = max(np.abs(Tkj).max(), 1e-30) / qmax
                else:
                    sA, sT = sA_static, sT_static
                S += quant(Aik, sA, qmax) @ quant(Tkj, sT, qmax)
            Tii = T[i*BL:(i+1)*BL, i*BL:(i+1)*BL]
            if scale_mode == "dyn":
                sTi = max(np.abs(Tii).max(), 1e-30) / qmax; sS = max(np.abs(S).max(), 1e-30) / qmax
            else:
                sTi, sS = sT_static, tmax / qmax
            T[i*BL:(i+1)*BL, j*BL:(j+1)*BL] = quant(Tii, sTi, qmax) @ quant(S, sS, qmax)
    return T

def oc(Tq, h):                                 # off-diagonal relerr
    Te = Texact[h]; m = np.tril(np.ones((C, C)), -1).astype(bool)
    return float(np.linalg.norm((Tq - Te)[m]) / (np.linalg.norm(Te[m]) + 1e-12))

print("\n=== end-to-end off-diag oc (mean over heads), MERGE quant only ===")
for bits in (8, 16):
    for mode in ("dyn", "static"):
        ocs = [oc(solve(h, mode, bits), h) for h in range(H)]
        print(f"  int{bits:<2}  {mode:6s}  oc = {np.mean(ocs):.5f}  (max over heads {np.max(ocs):.5f})")
