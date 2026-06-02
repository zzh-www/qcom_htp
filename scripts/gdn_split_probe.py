#!/usr/bin/env python3
"""M6 two-op SPLIT graph:  A -> GdnSolveDiag(A,S) -> T1 ;  GdnMergeHmx(T1,S,Hs) -> T.

Op1 (HVX, multithreaded) solves diagonals + writes int8 tiles into the shared handoff scratch S.
Op2 (HMX) reads S (the int8 tiles) + T1 (dependency/quant carrier) and runs the merge chain into T.
S is ONE shared VTCM constant written by Op1, read by Op2; the T1->Op2 edge forces the ordering.
Hs is Op2's own HMX-surface VTCM scratch.

Emits A.raw, T_full_ref.raw (np.linalg.inv), split.onnx, ovr_split.json into <outdir>.
Usage: gdn_split_probe.py <outdir> [H] [C]
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
BL = 64; NB = C // BL
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
            npz = f; break
    if npz is None:
        raise SystemExit(f"no golden chunk with >={min_tokens} real tokens")
    qc, kc, vc, gc, betac, S_in = _golden_chunk_args(npz, 0)
    tl, sl, cu, ey = _masks(C, "cpu", torch.float64)
    kn = l2norm_lastdim(kc); k_beta = kn * betac.unsqueeze(-1)
    g = torch.matmul(gc.unsqueeze(-2), cu.reshape(1, 1, C, C)).squeeze(-2)
    diff = g.unsqueeze(-1) - g.unsqueeze(-2); decay = torch.exp(diff * tl) * tl
    A = ((-torch.matmul(k_beta, kn.transpose(-1, -2)) * decay) * sl).double().numpy()[0]
    return os.path.basename(npz), A


name, Areal = build_real_A(C)
Hreal = Areal.shape[0]
A = np.stack([Areal[h % Hreal] for h in range(H)]).astype(np.float32)
A.reshape(1, H, C, C).tofile(os.path.join(outdir, "A.raw"))
Tfull = np.stack([np.linalg.inv(np.eye(C) - A[h]) for h in range(H)]).astype(np.float32)
Tfull.reshape(1, H, C, C).tofile(os.path.join(outdir, "T_full_ref.raw"))

sA = max(abs(A).max() / 32767.0, 1e-12)
sT = 2.0 / 32767.0

HMX_STRIDE = 0x60000   # Op2 HMX-surface region (single thread -> 1 region)
SCRATCH_HS = HMX_STRIDE

io = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, H, C, C])
# Hs = Op2's HMX-surface VTCM scratch (the ONLY constant now; handoff is a real Op1->Op2 edge Hd).
Hs_const = helper.make_tensor("Hs", TensorProto.UINT8, [1, 1, 1, SCRATCH_HS],
                              np.zeros(SCRATCH_HS, dtype=np.uint8).tobytes(), raw=True)
# Op1: A -> T1 (block-diag uint16), Hd (int8 tile handoff uint8, a real graph edge -> Op2 reads it).
n1 = helper.make_node("GdnSolveDiag", ["A"], ["T1", "Hd"], name="GdnSolveDiag_0", domain="gdn")
# Op2: T1 (dep+quant), Hd (handoff), Hs (HMX scratch) -> T.
n2 = helper.make_node("GdnMergeHmx", ["T1", "Hd", "Hs"], ["T"], name="GdnMergeHmx_0", domain="gdn")
T1_vi = helper.make_tensor_value_info("T1", TensorProto.FLOAT, [1, H, C, C])
Hd_vi = helper.make_tensor_value_info("Hd", TensorProto.UINT8, [1, H, C, C])
graph = helper.make_graph([n1, n2], "split", [io("A")], [io("T")],
                          initializer=[Hs_const], value_info=[T1_vi, Hd_vi])
m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("gdn", 1)])
onnx.save(m, os.path.join(outdir, "split.onnx"))
json.dump({"version": "2.0.0", "encodings": [
    {"name": "A",  "output_dtype": "uint16", "y_scale": sA, "y_zero_point": 32768},
    {"name": "T1", "output_dtype": "uint16", "y_scale": sT, "y_zero_point": 32768},
    {"name": "Hd", "output_dtype": "uint8",  "y_scale": 1.0, "y_zero_point": 0},
    {"name": "T",  "output_dtype": "uint16", "y_scale": sT, "y_zero_point": 32768}]},
    open(os.path.join(outdir, "ovr_split.json"), "w"))
print(f"golden={name} A[1,{H},{C},{C}] absmax {abs(A).max():.4f} sA={sA:.3e} NB={NB} scratchHs=0x{SCRATCH_HS:x}")
