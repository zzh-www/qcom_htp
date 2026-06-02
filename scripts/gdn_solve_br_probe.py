#!/usr/bin/env python3
"""Generate the standalone A->GdnSolveBR->T graph at C=128 from REAL GDN data, for device validation
of the block-recursive HMX-merge op.  Emits A.raw [1,H,128,128], T_ref.raw (np.linalg.inv), solve_br.onnx,
ovr_solve_br.json into <outdir>.

A is built at CHUNK=128 from a real >=128-token golden (gdn_blockrec_c128_probe style), tiled across H
heads (one real head replicated/cycled to fill H).  T_ref = np.linalg.inv(I - A[h]) per head.

Usage: gdn_solve_br_probe.py <outdir> [H] [min_tokens]
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
MIN_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 128
C = 128
os.makedirs(outdir, exist_ok=True)


def build_real_A(min_tokens):
    """Real strictly-lower [Hreal,C,C] A at CHUNK=128 (same as gdn_blockrec_sim.build_real_A)."""
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
    return os.path.basename(npz), A   # [Hreal, C, C]


name, Areal = build_real_A(MIN_TOK)
Hreal = Areal.shape[0]
# fill H heads by cycling the real heads (gives a realistic mix; all distinct if H<=Hreal)
A = np.stack([Areal[h % Hreal] for h in range(H)]).astype(np.float32)
A.reshape(1, H, C, C).tofile(os.path.join(outdir, "A.raw"))
Tref = np.stack([np.linalg.inv(np.eye(C) - A[h]) for h in range(H)]).astype(np.float32)
Tref.reshape(1, H, C, C).tofile(os.path.join(outdir, "T_ref.raw"))

sA = max(abs(A).max() / 32767.0, 1e-12)
sT = 2.0 / 32767.0

# scratch VTCM workspace input S (uint8 zeros): the op carves act/wt/bias/out/tables out of it.
# Buffers are spaced 64KB apart (see GdnSolveBROp gdn_vtcm_from); 0x60000 covers the layout.
SCRATCH = 0x60000
io = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, H, C, C])
S_const = helper.make_tensor("S", TensorProto.UINT8, [1, 1, 1, SCRATCH], np.zeros(SCRATCH, dtype=np.uint8).tobytes(), raw=True)
gs = helper.make_node("GdnSolveBR", ["A", "S"], ["T"], name="GdnSolveBR_0", domain="gdn")
graph = helper.make_graph([gs], "solve_br", [io("A")], [io("T")], initializer=[S_const])
m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("gdn", 1)])
onnx.save(m, os.path.join(outdir, "solve_br.onnx"))
# uint16-midpoint (zp=32768): matches the QUInt16 op input the HTP backend stages in TCM (no q::Cast).
json.dump({"version": "2.0.0", "encodings": [
    {"name": "A", "output_dtype": "uint16", "y_scale": sA, "y_zero_point": 32768},
    {"name": "T", "output_dtype": "uint16", "y_scale": sT, "y_zero_point": 32768}]},
    open(os.path.join(outdir, "ovr_solve_br.json"), "w"))
print(f"golden={name} A[1,{H},{C},{C}] absmax {abs(A).max():.4f} sA={sA:.3e} Hreal={Hreal}")
