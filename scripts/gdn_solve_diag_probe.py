#!/usr/bin/env python3
"""Generate the standalone A->GdnSolveDiag->T graph (M6 Op1) from REAL GDN data.

Op1 fills ONLY the diagonal 64x64 blocks T_ii of T=(I-A)^-1 (off-diag left at zp).  So the device-T
reference here is the BLOCK-DIAGONAL of np.linalg.inv per head: T_ref[i*BL:(i+1)*BL, i*BL:(i+1)*BL] =
inv(I-A)[that block]; everything else 0.  We also save T_full_ref (the full inverse) for later steps.

Emits A.raw [1,H,C,C], T_diag_ref.raw, T_full_ref.raw, solve_diag.onnx, ovr_solve_diag.json.

Usage: gdn_solve_diag_probe.py <outdir> [H] [C]
"""
import sys, os, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("GDN_NO_VSCALE", "1")

import onnx
from onnx import helper, TensorProto

outdir = sys.argv[1]
H = int(sys.argv[2]) if len(sys.argv) > 2 else 16
C = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("CB", "256"))
BL = 64
NB = C // BL
MIN_TOK = C
os.makedirs(outdir, exist_ok=True)


def build_real_A(min_tokens):
    import torch
    import gdn_onnx_kernel as gok
    gok.CHUNK = C
    from gdn_onnx_kernel import _golden_chunk_args, l2norm_lastdim, _masks
    from gdn_solve_int16_model import GOLDEN
    npz = None
    for f in sorted(glob.glob(os.path.join(GOLDEN, "*.npz"))):
        if np.load(f)["query"].shape[1] >= min_tokens:
            npz = f
            break
    if npz is None:
        raise SystemExit(f"no golden chunk with >={min_tokens} real tokens")
    qc, kc, vc, gc, betac, S_in = _golden_chunk_args(npz, 0)
    tl, sl, cu, ey = _masks(C, "cpu", torch.float64)
    kn = l2norm_lastdim(kc); k_beta = kn * betac.unsqueeze(-1)
    g = torch.matmul(gc.unsqueeze(-2), cu.reshape(1, 1, C, C)).squeeze(-2)
    diff = g.unsqueeze(-1) - g.unsqueeze(-2); decay = torch.exp(diff * tl) * tl
    A = ((-torch.matmul(k_beta, kn.transpose(-1, -2)) * decay) * sl).double().numpy()[0]
    return os.path.basename(npz), A


name, Areal = build_real_A(MIN_TOK)
Hreal = Areal.shape[0]
A = np.stack([Areal[h % Hreal] for h in range(H)]).astype(np.float32)
A.reshape(1, H, C, C).tofile(os.path.join(outdir, "A.raw"))

Tfull = np.stack([np.linalg.inv(np.eye(C) - A[h]) for h in range(H)]).astype(np.float32)
Tfull.reshape(1, H, C, C).tofile(os.path.join(outdir, "T_full_ref.raw"))
# block-diagonal reference (what Op1 writes into T)
Tdiag = np.zeros_like(Tfull)
for h in range(H):
    for i in range(NB):
        Tdiag[h, i*BL:(i+1)*BL, i*BL:(i+1)*BL] = Tfull[h, i*BL:(i+1)*BL, i*BL:(i+1)*BL]
Tdiag.reshape(1, H, C, C).tofile(os.path.join(outdir, "T_diag_ref.raw"))

sA = max(abs(A).max() / 32767.0, 1e-12)
sT = 2.0 / 32767.0

# Op1 = 1 input (A) + 2 outputs (T uint16 block-diag, Hd uint8 handoff [B,H,C,C], TCM_Only via the op sig).
io  = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, H, C, C])
io8 = lambda n: helper.make_tensor_value_info(n, TensorProto.UINT8, [1, H, C, C])
gs = helper.make_node("GdnSolveDiag", ["A"], ["T", "Hd"], name="GdnSolveDiag_0", domain="gdn")
graph = helper.make_graph([gs], "solve_diag", [io("A")], [io("T"), io8("Hd")])
m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("gdn", 1)])
onnx.save(m, os.path.join(outdir, "solve_diag.onnx"))
json.dump({"version": "2.0.0", "encodings": [
    {"name": "A", "output_dtype": "uint16", "y_scale": sA, "y_zero_point": 32768},
    {"name": "T", "output_dtype": "uint16", "y_scale": sT, "y_zero_point": 32768},
    {"name": "Hd", "output_dtype": "uint8", "y_scale": 1.0, "y_zero_point": 0}]},
    open(os.path.join(outdir, "ovr_solve_diag.json"), "w"))
print(f"golden={name} A[1,{H},{C},{C}] absmax {abs(A).max():.4f} sA={sA:.3e} Hreal={Hreal} NB={NB}")
