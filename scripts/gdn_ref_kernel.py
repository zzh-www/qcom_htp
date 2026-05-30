#!/usr/bin/env python3
"""Self-contained, head-batched reference of the GDN core kernel (HTP port spec), in torch.

OUR implementation of the chunked gated-delta-rule compute (NOT FLA). Reproduces exactly
the math at the kernel boundary captured in the golden set:

  input  (post-conv, as captured): q,k,v [B,T,H,D], g,beta [B,T,H]
  inside kernel: l2norm(q,k) + scale + k_beta/v_beta + g=cumsum + chunked recurrence
  output: o [B,T,H,D]   (+ final state [B,H,Dk,Dv])

The FIXED-length HTP kernel unit is `gdn_chunk` — exactly ONE C=64 token chunk plus the
incoming state (HTP only accepts fixed shapes). `gdn_kernel` is the outer loop over
ceil(T/64) chunks (orchestration, not the kernel).

Each step is tagged with the HTP engine it maps to:
  [HMX] integer 32x32-tile matmul   [HVX] vector elementwise / exp / l2norm   [SEQ] sequential

Everything runs in fp64 (no quantization). Quant/dequant plugs in via the `mm(A,B,site)`
hook at the 8 [HMX] GEMMs; the [SEQ] triangular solve stays fp. Validate against golden:
  .venv/bin/python scripts/gdn_ref_kernel.py --validate
"""
import argparse, glob, os
import numpy as np
import torch

CHUNK = 64                                              # fixed T for the HTP kernel unit
DEF_DEVICE = os.environ.get("GDN_DEVICE", "cpu")
DEF_DTYPE = torch.float64


def l2norm(x, dim=-1, eps=1e-6):                        # [HVX] rsqrt over D
    return x / torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def bf16_to_f32(a):
    """Reconstruct fp32 from stored bf16 bit patterns (uint16); pass fp32 through. numpy."""
    a = np.asarray(a)
    if a.dtype == np.uint16:
        return (a.astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32)


def load_case(npz_path, device=DEF_DEVICE, dtype=DEF_DTYPE):
    """Load one golden capture -> dict of torch tensors (query,key,value,g,beta,o)."""
    z = np.load(npz_path)
    return {k: torch.from_numpy(np.ascontiguousarray(bf16_to_f32(z[k]))).to(device=device, dtype=dtype)
            for k in ("query", "key", "value", "g", "beta", "o")}


def relerr(a, b):
    a = a.to(torch.float64); b = b.to(torch.float64)
    return (torch.linalg.vector_norm(a - b) / (torch.linalg.vector_norm(b) + 1e-12)).item()


@torch.no_grad()
def gdn_chunk(qc, kc, vc, gc, betac, S_in, mm=None, dtype=DEF_DTYPE):
    """The FIXED-length GDN kernel unit (the thing we put on HTP): ONE C=64 chunk + state.
    Shapes (B=batch, H=heads, C=64, Dk=Dv=128):
        qc,kc,vc : [B,H,C,128]   gc,betac : [B,H,C]   S_in : [B,H,Dk,Dv]
        returns  oc : [B,H,C,Dv]   S_out : [B,H,Dk,Dv]
    All 8 GEMMs (mm(A,B,site)) are fixed-shape; the [SEQ] triangular solve is fp."""
    if mm is None:
        mm = lambda A, B, site=None: A @ B
    qc = qc.to(dtype); kc = kc.to(dtype); vc = vc.to(dtype)
    gc = gc.to(dtype); betac = betac.to(dtype)
    dev = qc.device
    C, Dk = qc.shape[-2], qc.shape[-1]
    zero = torch.zeros((), dtype=dtype, device=dev)

    # --- preprocessing (per-token / per-chunk, inside the kernel) ---
    qc = l2norm(qc, dim=-1)                                              # [HVX]
    kc = l2norm(kc, dim=-1)                                              # [HVX]
    qc = qc * (1.0 / (Dk ** 0.5))                                        # [HVX]
    v_beta = vc * betac.unsqueeze(-1)                                    # [HVX]
    k_beta = kc * betac.unsqueeze(-1)                                    # [HVX]
    g = torch.cumsum(gc, dim=-1)                                         # [HVX] prefix-sum over C

    # --- intra-chunk (Stage A) ---
    diff = g.unsqueeze(-1) - g.unsqueeze(-2)
    tril = torch.tril(torch.ones(C, C, dtype=torch.bool, device=dev))
    decay = torch.exp(torch.where(tril, diff, zero)) * tril.to(dtype)    # [HVX] vexp on CxC
    attn = -mm(k_beta, kc.transpose(-1, -2), 'A_kk') * decay             # [HMX] contract Dk
    attn = attn.masked_fill(torch.triu(torch.ones(C, C, dtype=torch.bool, device=dev), 0), 0.0)
    for i in range(1, C):                                                # [SEQ] forward subst, 64 steps
        row = attn[..., i, :i].clone()
        attn[..., i, :i] = row + torch.einsum('...j,...jk->...k', row, attn[..., :i, :i])
    attn = attn + torch.eye(C, dtype=dtype, device=dev)                  # T = (I-A)^-1
    U = mm(attn, v_beta, 'A_Uv')                                         # [HMX] contract C
    W = mm(attn, k_beta * torch.exp(g).unsqueeze(-1), 'A_Wk')           # [HMX] contract C

    # --- inter-chunk (Stage B) with incoming state S_in ---
    P = mm(qc, kc.transpose(-1, -2), 'B_qk') * decay                     # [HMX] contract Dk
    P = P.masked_fill(torch.triu(torch.ones(C, C, dtype=torch.bool, device=dev), 1), 0.0)  # causal
    v_new = U - mm(W, S_in, 'B_WS')                                      # [HMX] contract Dk; [HVX] sub
    attn_inter = mm(qc * torch.exp(g).unsqueeze(-1), S_in, 'B_qS')       # [HMX] contract Dk
    oc = attn_inter + mm(P, v_new, 'B_Pv')                               # [HMX] contract C; [HVX] add
    g_last = g[..., -1]
    dec_k = torch.exp(g_last.unsqueeze(-1) - g).unsqueeze(-1)            # [HVX]
    S_out = S_in * torch.exp(g_last)[..., None, None] \
        + mm((kc * dec_k).transpose(-1, -2), v_new, 'B_kv')              # [HVX] decay; [HMX] contract C
    return oc, S_out


@torch.no_grad()
def gdn_kernel(q, k, v, g, beta, chunk_size=CHUNK, initial_state=None, dtype=DEF_DTYPE, mm=None):
    """Full variable-length prefill = OUTER loop over fixed-T=64 `gdn_chunk` units, carrying
    S. q,k,v: [B,T,H,D]; g,beta: [B,T,H]. Returns o [B,T,H,D], S [B,H,Dk,Dv]. On HTP this
    loop is orchestration; only `gdn_chunk` is the fixed-shape kernel."""
    q, k, v = (x.to(dtype).transpose(1, 2) for x in (q, k, v))          # [B,H,T,D]
    beta = beta.to(dtype).transpose(1, 2)
    g = g.to(dtype).transpose(1, 2)
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    C = chunk_size
    pad = (C - T % C) % C
    if pad:                                                             # zero-pad last chunk to 64
        F = torch.nn.functional
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        beta = F.pad(beta, (0, pad)); g = F.pad(g, (0, pad))
    Tp = T + pad
    dev = k.device
    S = (torch.zeros(B, H, Dk, Dv, dtype=dtype, device=dev) if initial_state is None
         else initial_state.to(dtype).clone())
    o = torch.zeros(B, H, Tp, Dv, dtype=dtype, device=dev)
    for i in range(Tp // C):
        sl = slice(i * C, (i + 1) * C)
        oc, S = gdn_chunk(q[:, :, sl], k[:, :, sl], v[:, :, sl],
                          g[:, :, sl], beta[:, :, sl], S, mm=mm, dtype=dtype)
        o[:, :, sl] = oc
    return o[:, :, :T].transpose(1, 2), S


def validate(golden_dir="downloads/gdn_golden"):
    files = sorted(glob.glob(os.path.join(golden_dir, "p*_L*.npz")))
    if not files:
        print(f"no golden at {golden_dir} (need the fp32-ref golden to check exactness)")
        return
    er, eb = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        get = lambda key: torch.from_numpy(np.ascontiguousarray(bf16_to_f32(z[key]))).to(DEF_DTYPE)
        o, _ = gdn_kernel(get("query"), get("key"), get("value"), get("g"), get("beta"))
        if "o_ref_fp32" in z.files:
            oref = z["o_ref_fp32"]
            if oref.dtype != object and tuple(oref.shape) == tuple(o.shape):
                er.append(relerr(o, torch.from_numpy(np.ascontiguousarray(oref)).to(DEF_DTYPE)))
        eb.append(relerr(o, get("o")))
    if er:
        er = np.array(er)
        print(f"ours(torch) vs FLA fp32 ref: median {np.median(er):.2e} max {er.max():.2e}  "
              f"{'PASS' if er.max() < 1e-5 else 'CHECK'}")
    eb = np.array(eb)
    print(f"ours(torch) vs model bf16 o:  median {np.median(eb):.2e} max {eb.max():.2e}  (~bf16 floor 3e-3)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--golden", default="downloads/gdn_golden")
    args = ap.parse_args()
    if args.validate:
        validate(args.golden)
    else:
        print("nothing to do; pass --validate")


if __name__ == "__main__":
    main()
