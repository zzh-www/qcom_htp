#!/usr/bin/env python3
"""End-to-end oc validation for the GdnSolveBR custom op (Task 6, integration check).

The BR op replaces ONLY the intra-chunk triangular solve T=(I-A)^-1.  Rather than rebuild a full C=256
device graph (the GDN export pipeline is hardwired to C=64), this isolates the SOLVE's contribution to
oc error: it runs the real fp64 GDN chunk forward but substitutes the device-computed BR T (read back from
the standalone run's T.raw) for the exact T, and compares oc to the exact-solve oc on the SAME real chunk.

Inputs (from example/gdn_native/solve_br_op/standalone, after `CB=256 H=32 bash gdn_br.sh`):
  T.raw      device BR T  [1,H,C,C] float32   (out_s/Result_0/T.raw)
  T_ref.raw  exact  inv(I-A) [1,H,C,C] float32
The real chunk (q/k/v/g/beta/S_in) is rebuilt from the same golden npz the probe used (C=256, chunk 0).

Usage: gdn_br_oc_check.py <standalone_dir> [--golden p29_L00] [--C 256]
"""
import sys, os, glob, argparse
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

ap = argparse.ArgumentParser()
ap.add_argument("stdir")
ap.add_argument("--golden", default=None, help="golden basename (default: same auto-pick as the probe)")
ap.add_argument("--C", type=int, default=256)
a = ap.parse_args()
C = a.C

import gdn_onnx_kernel as gok
gok.CHUNK = C
import gdn_ref_kernel as grk
grk.CHUNK = C
from gdn_onnx_kernel import _golden_chunk_args
from gdn_solve_int16_model import GOLDEN


def l2norm(x, dim=-1):
    return x / (x.norm(dim=dim, keepdim=True) + 1e-12)


def gdn_oc(qc, kc, vc, gc, betac, S_in, T=None):
    """fp64 GDN chunk forward; if T is given, use it as attn=(I-A)^-1 instead of solving."""
    dtype = torch.float64
    qc, kc, vc = qc.to(dtype), kc.to(dtype), vc.to(dtype)
    gc, betac, S_in = gc.to(dtype), betac.to(dtype), S_in.to(dtype)
    dev = qc.device
    Cc, Dk = qc.shape[-2], qc.shape[-1]
    zero = torch.zeros((), dtype=dtype, device=dev)
    qc = l2norm(qc) * (1.0 / (Dk ** 0.5))
    kc = l2norm(kc)
    v_beta = vc * betac.unsqueeze(-1)
    k_beta = kc * betac.unsqueeze(-1)
    g = torch.cumsum(gc, dim=-1)
    diff = g.unsqueeze(-1) - g.unsqueeze(-2)
    tril = torch.tril(torch.ones(Cc, Cc, dtype=torch.bool, device=dev))
    decay = torch.exp(torch.where(tril, diff, zero)) * tril.to(dtype)
    if T is None:
        attn = -(k_beta @ kc.transpose(-1, -2)) * decay
        attn = attn.masked_fill(torch.triu(torch.ones(Cc, Cc, dtype=torch.bool, device=dev), 0), 0.0)
        for i in range(1, Cc):
            row = attn[..., i, :i].clone()
            attn[..., i, :i] = row + torch.einsum('...j,...jk->...k', row, attn[..., :i, :i])
        attn = attn + torch.eye(Cc, dtype=dtype, device=dev)
    else:
        attn = T.to(dtype)
    U = attn @ v_beta
    W = attn @ (k_beta * torch.exp(g).unsqueeze(-1))
    P = (qc @ kc.transpose(-1, -2)) * decay
    P = P.masked_fill(torch.triu(torch.ones(Cc, Cc, dtype=torch.bool, device=dev), 1), 0.0)
    v_new = U - W @ S_in
    attn_inter = (qc * torch.exp(g).unsqueeze(-1)) @ S_in
    oc = attn_inter + P @ v_new
    return oc


def relerr(x, y):
    return float(np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-12))


# pick the same golden the probe auto-selected (first with >=C real tokens)
npz = None
for f in sorted(glob.glob(os.path.join(GOLDEN, "*.npz"))):
    if (a.golden and a.golden in f) or (not a.golden and np.load(f)["query"].shape[1] >= C):
        npz = f
        break
if npz is None:
    sys.exit(f"no golden chunk with >= {C} tokens")
print(f"golden = {os.path.basename(npz)}  C={C}")

qc, kc, vc, gc, betac, S_in = _golden_chunk_args(npz, 0)   # chunk 0: S_in = 0
H = qc.shape[1]

# device BR T and exact T from the standalone run
Td = np.fromfile(os.path.join(a.stdir, "out_s/Result_0/T.raw"), dtype=np.float32)
Tr = np.fromfile(os.path.join(a.stdir, "T_ref.raw"), dtype=np.float32)
n = H * C * C
Td = torch.from_numpy(Td[:n].reshape(1, H, C, C).astype(np.float64))
Tr = torch.from_numpy(Tr[:n].reshape(1, H, C, C).astype(np.float64))
print(f"  device-BR T vs exact T relerr: {relerr(Td.numpy(), Tr.numpy()):.3e}")

oc_exact = gdn_oc(qc, kc, vc, gc, betac, S_in, T=None).numpy()     # full fp64 (solve included)
oc_refT  = gdn_oc(qc, kc, vc, gc, betac, S_in, T=Tr).numpy()       # exact T injected (sanity: ~0 vs above)
oc_brT   = gdn_oc(qc, kc, vc, gc, betac, S_in, T=Td).numpy()       # device BR T injected

print(f"  sanity (exact-T injected vs internal solve) oc relerr: {relerr(oc_refT, oc_exact):.3e}")
print(f"  >>> oc relerr (device BR-T vs exact solve):            {relerr(oc_brT, oc_exact):.3e}")
# also vs the real model output o for these tokens (chunk 0 = tokens 0..C-1)
try:
    o_gold = torch.from_numpy(gok.bf16_to_f32(np.load(npz)["o"])).double()  # [1,T,H,Dv]
    o_gold = o_gold.transpose(1, 2)[:, :, :C].numpy()                       # [1,H,C,Dv]
    print(f"  oc relerr vs golden o:  exact-solve {relerr(oc_exact, o_gold):.3e}   BR {relerr(oc_brT, o_gold):.3e}")
except Exception as e:
    print(f"  (golden o compare skipped: {e})")
