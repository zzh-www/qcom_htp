#!/usr/bin/env python3
"""M8 PER-CHUNK two-op SPLIT graph.  Tests the architectural root-cause fix from M6/M7:

The M6/M7 graph had ONE GdnSolveDiag (Op1, HVX, all H heads) -> ONE GdnMergeHmx (Op2, HMX, all H heads):
 - 0% Op1<->Op2 overlap (Op2 waits for ALL of Op1, single batched edge),
 - ~545K DDR Spill/Fill (all H heads' handoff materialized at once, exceeds VTCM -> spills).
Heads are INDEPENDENT.  This emits the split as ceil(H/CK) INDEPENDENT chains over CK-head slices:
   A[chunk_i] -> Slice -> Ai -> GdnSolveDiag(Ai) -> {T1_i, Hd_i} -> GdnMergeHmx(T1_i,Hd_i,Hs_i) -> T_i
   ... -> Concat(T_0..T_{n-1}) -> T
The chains are mutually independent (heads don't interact), so QNN CAN run chain-i's Op2 (HMX)
concurrently with chain-(i+1)'s Op1 (HVX) -- the 98%-overlap mechanism, now across chains -- and each
chunk's ~CK*40KB handoff stays VTCM-resident (no DDR spill).  CK=H reproduces the M6/M7 batched baseline.

The ops (solve_diag_op, merge_hmx_op) are unchanged: each derives head-count from the tensor shape and
indexes Hd/T/handoff by tile-local head, so it processes whatever CK-head slice the graph hands it.

Emits A.raw, T_full_ref.raw (np.linalg.inv), split.onnx, ovr_split.json into <outdir>.
Usage: gdn_split_probe.py <outdir> [H] [C]    (CK via env GDN_CK, default = H = batched baseline)
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
CK = int(os.environ.get("GDN_CK", str(H)))   # heads per chain; CK==H -> the M6/M7 batched baseline
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

HMX_STRIDE = 0x60000   # per-chain Op2 HMX-surface region (one region per HMX node, single HMX thread)
SCRATCH_HS = HMX_STRIDE

n_chunks = (H + CK - 1) // CK
single = (n_chunks == 1)   # CK>=H -> the M6/M7 batched baseline: feed A straight to Op1, no Slice/Concat
io = lambda n, h: helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, h, C, C])

nodes, inits, vinfos, encs = [], [], [], []
encs.append({"name": "A", "output_dtype": "uint16", "y_scale": sA, "y_zero_point": 32768})

# Per-chain {Slice -> GdnSolveDiag -> GdnMergeHmx}, then Concat the T_i into the full T.
# single-chain (CK>=H): skip Slice/Concat entirely and wire A->Op1->T directly (EXACT M6/M7 graph).
chunk_outs = []
for ci in range(n_chunks):
    h0 = ci * CK
    hc = min(CK, H - h0)                       # heads in this chain (last may be short)
    T1i, Hdi, Hsi = (f"T1_{ci}", f"Hd_{ci}", f"Hs_{ci}")
    Ti = "T" if single else f"T_{ci}"          # single chain writes the graph output T directly

    if single:
        Ai = "A"                               # no Slice: feed the whole A tensor in (matches M6)
    else:
        Ai = f"A{ci}"
        # Slice A[:, h0:h0+hc, :, :] -> Ai   (axis=1, the head dim).
        starts = helper.make_tensor(f"st{ci}", TensorProto.INT64, [1], np.array([h0], np.int64))
        ends   = helper.make_tensor(f"en{ci}", TensorProto.INT64, [1], np.array([h0 + hc], np.int64))
        axes   = helper.make_tensor(f"ax{ci}", TensorProto.INT64, [1], np.array([1], np.int64))
        inits += [starts, ends, axes]
        nodes.append(helper.make_node("Slice", ["A", f"st{ci}", f"en{ci}", f"ax{ci}"], [Ai],
                                      name=f"Slice_{ci}"))
        vinfos.append(io(Ai, hc))
        # the sliced A keeps A's quant (Slice is dtype-preserving); the quantizer needs an encoding.
        encs.append({"name": Ai, "output_dtype": "uint16", "y_scale": sA, "y_zero_point": 32768})

    # Per-chain Op2 HMX-surface scratch (one constant per chain so concurrent chains never collide).
    inits.append(helper.make_tensor(Hsi, TensorProto.UINT8, [1, 1, 1, SCRATCH_HS],
                                    np.zeros(SCRATCH_HS, dtype=np.uint8).tobytes(), raw=True))
    # Op1: Ai -> {T1_i (block-diag uint16), Hd_i (int8 tile handoff, a real graph edge)}.
    nodes.append(helper.make_node("GdnSolveDiag", [Ai], [T1i, Hdi],
                                  name=f"GdnSolveDiag_{ci}", domain="gdn"))
    # Op2: T1_i (dep+quant), Hd_i (handoff), Hs_i (HMX scratch) -> T_i.
    nodes.append(helper.make_node("GdnMergeHmx", [T1i, Hdi, Hsi], [Ti],
                                  name=f"GdnMergeHmx_{ci}", domain="gdn"))
    vinfos += [helper.make_tensor_value_info(T1i, TensorProto.FLOAT, [1, hc, C, C]),
               helper.make_tensor_value_info(Hdi, TensorProto.UINT8, [1, hc, C, C])]
    if not single:
        vinfos.append(io(Ti, hc))
    encs += [{"name": T1i, "output_dtype": "uint16", "y_scale": sT, "y_zero_point": 32768},
             {"name": Hdi, "output_dtype": "uint8",  "y_scale": 1.0, "y_zero_point": 0}]
    if not single:
        encs.append({"name": Ti, "output_dtype": "uint16", "y_scale": sT, "y_zero_point": 32768})
    chunk_outs.append(Ti)

# Concat the per-chain T_i back into the full T (axis=1, head dim).
if not single:
    nodes.append(helper.make_node("Concat", chunk_outs, ["T"], name="Concat_T", axis=1))
encs.append({"name": "T", "output_dtype": "uint16", "y_scale": sT, "y_zero_point": 32768})

graph = helper.make_graph(nodes, "split", [io("A", H)], [io("T", H)],
                          initializer=inits, value_info=vinfos)
m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17),
                                            helper.make_opsetid("gdn", 1)])
onnx.save(m, os.path.join(outdir, "split.onnx"))
json.dump({"version": "2.0.0", "encodings": encs}, open(os.path.join(outdir, "ovr_split.json"), "w"))
print(f"golden={name} A[1,{H},{C},{C}] absmax {abs(A).max():.4f} sA={sA:.3e} NB={NB} "
      f"CK={CK} n_chunks={n_chunks} scratchHs=0x{SCRATCH_HS:x}")
