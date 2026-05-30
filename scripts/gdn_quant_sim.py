#!/usr/bin/env python3
"""Host-side fixed-point SIMULATOR of the fully-quantized GDN chunk (all-integer, no fp16).

Mirrors QNN affine per-tensor quantization on EVERY intermediate tensor so we can iterate the
quantization scheme (bit-widths, per-tensor scales, where the dynamic range breaks) in seconds
instead of a device round-trip. Two passes:
  calibrate -> collect per-(name) min/max over real golden calib chunks
  eval      -> q/dq every op output with those static ranges, run on a held-out test chunk,
               report relerr(o_quant, o_fp64).

Quant model = QNN asymmetric per-tensor (TF-style): scale=(max-min)/(2^bw-1), zp, clamp.
Per-tensor bit-width is configurable per name via BITS (default 16).

Run:  .venv/bin/python scripts/gdn_quant_sim.py            # int16 everywhere, L00
      GDN_QUANT_GEMM_ONLY=1 .venv/bin/python scripts/gdn_quant_sim.py   # only the 8 GEMM I/O
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_onnx_kernel import _masks, _golden_chunk_args, emit_calib, CHUNK, EPS  # noqa
from gdn_ref_kernel import gdn_chunk, relerr  # noqa

GOLDEN = os.environ.get("GDN_GOLDEN", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "gdn", "golden"))
DEF_BW = int(os.environ.get("GDN_BW", "16"))


class Quantizer:
    """Per-name static affine quant. mode='calib' records ranges; 'eval' q/dq with them."""
    def __init__(self, bits=None, ranges=None, enabled=True):
        self.mode = "calib" if ranges is None else "eval"
        self.ranges = {} if ranges is None else ranges
        self.bits = bits or {}
        self.enabled = enabled

    def __call__(self, name, x, bw=None):
        if not self.enabled:
            return x
        if self.mode == "calib":
            mn, mx = float(x.min()), float(x.max())
            r = self.ranges.get(name)
            self.ranges[name] = (min(mn, r[0]), max(mx, r[1])) if r else (mn, mx)
            return x
        mn, mx = self.ranges[name]
        bw = bw or self.bits.get(name, DEF_BW)
        if os.environ.get("GDN_SYM") == "1":                  # symmetric (HTP int16 MatMul rule)
            ma = max(abs(mn), abs(mx)) or 1.0
            qm = (1 << (bw - 1)) - 1
            scale = ma / qm
            return torch.clamp(torch.round(x / scale), -qm - 1, qm) * scale
        qmax = (1 << bw) - 1
        mn = min(mn, 0.0); mx = max(mx, 0.0)                  # include 0
        scale = (mx - mn) / qmax if mx > mn else 1.0
        if scale == 0:
            scale = 1.0
        zp = round(-mn / scale)
        q = torch.clamp(torch.round(x / scale) + zp, 0, qmax)
        return (q - zp) * scale


def chunk_q(qc, kc, vc, gc, betac, S_in, Q, gemm_only=False, solve="neumann"):
    """gdn_chunk with a quantizer Q applied at every op output (or only GEMM I/O).
    solve: 'neumann' (product form, quant-hostile) | 'fwdsub' (bounded forward substitution)
           | 'blocked<b>' (block forward-subst, block size b: bounded AND static)."""
    dtype, dev = qc.dtype, qc.device
    C, Dk = qc.shape[-2], qc.shape[-1]
    tl, sl, cu, ey = _masks(C, dev, dtype)
    A = (lambda n, x: x) if gemm_only else Q                  # 'A' = activation-quant (all ops)
    G = Q                                                     # 'G' = the 8 GEMM operands/outputs

    qc = A("in_q", qc); kc = A("in_k", kc); vc = A("in_v", vc)
    gc = A("in_g", gc); betac = A("in_beta", betac); S_in = A("in_S", S_in)

    qc = A("q_l2", qc * torch.rsqrt((qc * qc).sum(-1, keepdim=True) + EPS))
    kc = A("k_l2", kc * torch.rsqrt((kc * kc).sum(-1, keepdim=True) + EPS))
    qc = A("q_sc", qc * (1.0 / (Dk ** 0.5)))
    betac = betac.unsqueeze(-1)
    v_beta = A("v_beta", vc * betac)
    k_beta = A("k_beta", kc * betac)
    g = A("g_cumsum", torch.matmul(gc.unsqueeze(-2), cu).squeeze(-2))
    eg = A("exp_g", torch.exp(g)).unsqueeze(-1)

    diff = g.unsqueeze(-1) - g.unsqueeze(-2)
    decay = A("decay", torch.exp(A("diff", diff * tl)) * tl)

    # A_kk GEMM: k_beta @ k^T
    kk = G("kk", torch.matmul(G("kk_a", k_beta), G("kk_w", kc).transpose(-1, -2)))
    Amat = A("A_intra", (-kk * decay) * sl)

    if solve == "neumann":
        # T = (I-A)^-1 via Neumann product — each power gets its own range (QUANT-HOSTILE:
        # A^16 transiently reaches ~1e7 on extreme prompts, destroying int16).
        T = ey + Amat
        Ap = Amat
        for i in range(5):
            Ap = A(f"Apow{i}", torch.matmul(Ap, Ap))
            T = A(f"T{i}", torch.matmul(T, ey + Ap))
    elif solve == "fwdsub":
        # exact forward substitution — bounded (T stays O(10)); reference [SEQ] form.
        Amat2 = Amat.clone()
        for i in range(1, C):
            row = Amat2[..., i, :i].clone()
            Amat2[..., i, :i] = A(f"fs{i}", row + torch.einsum('...j,...jk->...k', row, Amat2[..., :i, :i]))
        T = Amat2 + ey
    elif solve.startswith("blocked"):
        b = int(solve[len("blocked"):])                      # block forward substitution
        nb = C // b
        # invert each diagonal b-block with a SMALL Neumann product (bounded for small b)
        def inv_block(blk, bi):                              # (I-blk)^-1, blk strictly-lower b×b
            eyb = torch.eye(b, dtype=dtype, device=dev)
            Tb = eyb + blk; Apb = blk
            for s in range(max(1, (b - 1).bit_length())):
                Apb = A(f"ibAp{bi}_{s}", torch.matmul(Apb, Apb))
                Tb = A(f"ibT{bi}_{s}", torch.matmul(Tb, eyb + Apb))
            return Tb
        Tinv = [inv_block(Amat[..., i*b:(i+1)*b, i*b:(i+1)*b], i) for i in range(nb)]
        Tblk = [[None] * nb for _ in range(nb)]
        for i in range(nb):
            Tblk[i][i] = Tinv[i]
            for j in range(i - 1, -1, -1):                   # T_ij = Tinv_i * (sum_k A_ik T_kj)
                acc = torch.zeros_like(Amat[..., i*b:(i+1)*b, j*b:(j+1)*b])
                for k in range(j, i):
                    acc = acc + torch.matmul(Amat[..., i*b:(i+1)*b, k*b:(k+1)*b], Tblk[k][j])
                Tblk[i][j] = A(f"Tb{i}_{j}", torch.matmul(Tinv[i], acc))
        rows = [torch.cat([Tblk[i][j] if j <= i else torch.zeros_like(Tinv[0])
                           for j in range(nb)], dim=-1) for i in range(nb)]
        T = torch.cat(rows, dim=-2)
    else:
        raise ValueError(solve)

    U = G("U", torch.matmul(G("U_a", T), G("U_w", v_beta)))
    W = G("W", torch.matmul(G("W_a", T), G("W_w", k_beta * eg)))

    P = A("P", (torch.matmul(G("P_a", qc), G("P_w", kc).transpose(-1, -2)) * decay) * tl)
    WS = G("WS", torch.matmul(G("WS_a", W), G("WS_w", S_in)))
    v_new = A("v_new", U - WS)
    qS = G("qS", torch.matmul(G("qS_a", qc * eg), G("qS_w", S_in)))
    Pv = G("Pv", torch.matmul(G("Pv_a", P), G("Pv_w", v_new)))
    oc = A("oc", qS + Pv)

    g_last = g[..., -1:]
    e_glast = torch.exp(g_last).unsqueeze(-1)
    dec_k = A("dec_k", torch.exp(g_last - g)).unsqueeze(-1)
    kv = G("kv", torch.matmul(G("kv_a", (kc * dec_k)).transpose(-1, -2), G("kv_w", v_new)))
    S_out = A("S_out", S_in * e_glast + kv)
    return oc, S_out


def run(layer=0, test_prompt="p00", test_chunk=0, gemm_only=False, max_calib=16, solve="neumann"):
    man = json.load(open(os.path.join(GOLDEN, "manifest.json")))
    calib = [r["file"] for r in man["records"] if r["layer"] == layer and r["split"] == "calib"]
    cases = []
    for f in calib:
        p = os.path.join(GOLDEN, f); T = int(np.load(p)["query"].shape[1])
        for ch in ([0, 1] if T > CHUNK else [0]):
            cases.append(_golden_chunk_args(p, ch))
            if len(cases) >= max_calib:
                break
        if len(cases) >= max_calib:
            break
    Qc = Quantizer()
    for args in cases:
        chunk_q(*[a.clone() for a in args], Qc, gemm_only=gemm_only, solve=solve)
    Qe = Quantizer(ranges=Qc.ranges)

    tp = [r["file"] for r in man["records"] if r["layer"] == layer
          and os.path.basename(r["file"]).startswith(test_prompt)]
    tf = tp[0] if tp else calib[0]
    targs = _golden_chunk_args(os.path.join(GOLDEN, tf), test_chunk)
    oc_ref, S_ref = gdn_chunk(*targs)
    oc_q, S_q = chunk_q(*[a.clone() for a in targs], Qe, gemm_only=gemm_only, solve=solve)
    return relerr(oc_q, oc_ref), relerr(S_q, S_ref), Qc.ranges


if __name__ == "__main__":
    go = os.environ.get("GDN_QUANT_GEMM_ONLY") == "1"
    layer = int(os.environ.get("GDN_LAYER", "0"))
    eo, es, ranges = run(layer=layer, gemm_only=go)
    print(f"L{layer:02d} bw={DEF_BW} gemm_only={go}: oc relerr {eo:.2e}  S_out relerr {es:.2e}")
    if os.environ.get("GDN_DUMP_RANGES"):
        for n, (mn, mx) in sorted(ranges.items(), key=lambda kv: kv[1][1] - kv[1][0], reverse=True):
            print(f"  {n:12s} [{mn:+.3g}, {mx:+.3g}]  span {mx-mn:.3g}")
