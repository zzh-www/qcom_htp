#!/usr/bin/env python3
"""QNN-native GDN chunk kernel: `gdn_chunk` expressed entirely in static, ONNX-exportable
ops so QNN (qairt-converter) can lower it to a native graph.

This is the FLOAT native reference for the GDN HTP port — the goal is an aligned output
through the QNN stack (`先在qnn native上实现gdn kernel对齐输出`), not yet quantized. It must
reproduce `gdn_ref_kernel.gdn_chunk` to fp32 rounding.

Two pieces of the reference are sequential / op-unfriendly; both are rewritten as static
linear algebra so the whole chunk is one feed-forward graph:

  * cumsum(gc) over C  ->  gc @ Uc          (Uc = upper-triangular ones, [C,C])
  * T = (I - A)^-1, A strictly-lower 64x64 (nilpotent, A^64 = 0)
      ->  Neumann PRODUCT  T = prod_{i=0..5} (I + A^(2^i))
      Exact: the exponents reachable by subsets of {1,2,4,8,16,32} are exactly 0..63,
      so the product equals sum_{k=0..63} A^k = (I-A)^-1.  6 squarings, no iteration.

Everything else (l2norm, masks, exp, the 8 GEMMs) is elementwise / matmul. Masks are baked
in as constants (Mul before exp to avoid inf*0 on the masked-out upper triangle).

Boundary (B=1, H=32, C=64, Dk=Dv=128):
    qc,kc,vc [B,H,C,128]  gc,betac [B,H,C]  S_in [B,H,128,128]
      -> oc [B,H,C,128]   S_out [B,H,128,128]

Run:
    .venv/bin/python scripts/gdn_onnx_kernel.py --validate          # torch module vs fp64 ref
    .venv/bin/python scripts/gdn_onnx_kernel.py --export model.onnx # write ONNX + check via ORT
"""
import argparse, glob, json, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_ref_kernel import gdn_chunk, bf16_to_f32, relerr  # noqa: E402

CHUNK = 64
EPS = 1e-6


def _masks(C, device, dtype):
    rows = torch.arange(C, device=device).view(C, 1)
    cols = torch.arange(C, device=device).view(1, C)
    tril_incl = (rows >= cols).to(dtype)      # i>=j : decay / causal P keep
    strict_low = (rows > cols).to(dtype)      # i>j  : A intra-chunk (zero diagonal)
    cumsum_U = (rows <= cols).to(dtype)       # j<=i along cols : g = gc @ cumsum_U
    eye = torch.eye(C, device=device, dtype=dtype)
    return tril_incl, strict_low, cumsum_U, eye


class _L2Norm(torch.autograd.Function):
    """l2norm over the last dim that exports as one ONNX LpNormalization node (-> QNN's fused
    int16 L2Norm op). Doing it as separate sum/rsqrt/mul exposes the per-tensor-quantized
    sum-of-squares, whose huge per-head dynamic range crushes small-norm heads on HTP (~27%)."""
    @staticmethod
    def forward(ctx, x):
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + EPS)

    @staticmethod
    def symbolic(g, x):
        return g.op("LpNormalization", x, axis_i=-1, p_i=2)


def l2norm_lastdim(x):
    return _L2Norm.apply(x)


def sel_array(i, C=CHUNK, bl=16, bp=32):
    """The padded 0/1 selector sel[i] [bp,C]: picks block i's bl rows into a bp-tall tile."""
    s = np.zeros((bp, C), dtype=np.float32)
    s[:bl, i*bl:(i+1)*bl] = np.eye(bl, dtype=np.float32)
    return s


def cumsum_array(C=CHUNK):
    """Upper-triangular ones [C,C]: g = gc @ cumsum_U gives the prefix sum over C."""
    r = np.arange(C); return (r[:, None] <= r[None, :]).astype(np.float32)


def solve_T_blocked(A, bl=16, bp=32, sel=None):
    """T = (I - A)^-1 for strictly-lower A [...,C,C], via BLOCK forward substitution.

    Static (unrolled) and BOUNDED: unlike the Neumann product over the full matrix (whose powers
    transiently reach ~1e7 and destroy int16), every intermediate here stays O(10), so the graph
    quantizes cleanly.  Two block sizes:
      bl = LOGICAL block (=16): small enough that the per-block Neumann inverse stays bounded.
      bp = PHYSICAL block (=32): every block tensor is padded to bp×bp so ALL matmuls are
           32-aligned (HTP's quantized HMX MatMul rejects 16×16; the bl..bp padding rows are
           zero off-diagonal / identity on the diagonal, so they don't affect the result).
        T_ii = (I-A_ii)^-1 ;  T_ij = T_ii * sum_{k=j..i-1} A_ik T_kj   (i>j).
    `sel` = list of nb selector tensors [1,1,bp,C].  Pass them as RUNTIME INPUTS (activations)
    for the quantized HTP graph: a MatMul with a constant operand is lowered to FullyConnected
    (no int16 support), but activation×activation stays MatMul (int16 OK).  When None they are
    baked constants (fine for the float CPU path).  Placement/extraction is MatMul/Add ONLY (no
    Concat/Slice, which HTP rejects for differently-scaled block tensors)."""
    C = A.shape[-1]; nb = C // bl
    dtype, dev = A.dtype, A.device
    c4 = lambda a: torch.from_numpy(a.copy()).to(dtype=dtype, device=dev).reshape(1, 1, *a.shape)
    eyb = c4(np.eye(bp, dtype=np.float32))                     # bp×bp identity (used in Add only)
    if sel is None:
        sel = [c4(sel_array(i, C, bl, bp)) for i in range(nb)]
    selT = [s.transpose(-1, -2) for s in sel]                  # [1,1,C,bp]

    SA = [torch.matmul(sel[i], A) for i in range(nb)]          # [bp,C] = padded rows of block-row i
    Aij = [[torch.matmul(SA[i], selT[j]) for j in range(nb)] for i in range(nb)]  # [bp,bp] blocks

    def inv_block(d):                                          # (I-d)^-1, d strictly-lower (active bl)
        Tb = eyb + d; Ap = d
        for _ in range(max(1, (bl - 1).bit_length())):        # bl=16 -> A^2,A^4,A^8,A^16(=0)
            Ap = torch.matmul(Ap, Ap)
            Tb = torch.matmul(Tb, eyb + Ap)
        return Tb

    Tinv = [inv_block(Aij[i][i]) for i in range(nb)]
    Tblk = [[None] * nb for _ in range(nb)]
    for i in range(nb):
        Tblk[i][i] = Tinv[i]
        for j in range(i - 1, -1, -1):                        # T_ij = Tinv_i * sum_k A_ik T_kj
            acc = None
            for k in range(j, i):
                t = torch.matmul(Aij[i][k], Tblk[k][j])
                acc = t if acc is None else acc + t
            Tblk[i][j] = torch.matmul(Tinv[i], acc)
    # assemble T = sum_{i>=j} sel_i^T @ Tblk[i][j] @ sel_j  (places each block; no Concat)
    T = None
    for i in range(nb):
        for j in range(i + 1):
            term = torch.matmul(torch.matmul(selT[i], Tblk[i][j]), sel[j])
            T = term if T is None else T + term
    return T


def gdn_chunk_onnx(qc, kc, vc, gc, betac, S_in, masks=None, solve_block=16, consts=None):
    """Static, ONNX-friendly reproduction of gdn_ref_kernel.gdn_chunk (same math, no [SEQ]).

    `masks` = (tril_incl, strict_low, cumsum_U, eye) as constant [C,C] tensors. When None we
    build them from arange (fine for torch validation); for ONNX export pass baked constants
    so the graph has no runtime EyeLike/Compare ops (QNN has no translation for those).
    `consts` (quantized HTP path) = {'cumsum_U': tensor, 'sel': [tensors]} fed as RUNTIME inputs
    so cumsum / block-select stay activation×activation MatMul (not FullyConnected)."""
    dtype, dev = qc.dtype, qc.device
    C, Dk = qc.shape[-2], qc.shape[-1]
    tril_incl, strict_low, cumsum_U, eye = masks if masks is not None else _masks(C, dev, dtype)
    if consts is not None:
        cumsum_U = consts["cumsum_U"]

    # --- preprocessing ---
    qc = l2norm_lastdim(qc)                                             # fused L2Norm (QNN op)
    kc = l2norm_lastdim(kc)
    qc = qc * (1.0 / (Dk ** 0.5))
    betac = betac.unsqueeze(-1)
    v_beta = vc * betac
    k_beta = kc * betac
    g = torch.matmul(gc.unsqueeze(-2), cumsum_U).squeeze(-2)            # cumsum via matmul
    eg = torch.exp(g).unsqueeze(-1)                                     # [B,H,C,1]

    # --- intra-chunk decay + strictly-lower A ---
    diff = g.unsqueeze(-1) - g.unsqueeze(-2)                            # [B,H,C,C]
    decay = torch.exp(diff * tril_incl) * tril_incl                    # mask before exp
    A = (-torch.matmul(k_beta, kc.transpose(-1, -2)) * decay) * strict_low

    # --- T = (I-A)^-1 via BOUNDED block forward substitution (quant-friendly: all
    #     intermediates stay O(10), vs the Neumann product whose powers reach ~1e7) ---
    sel = consts["sel"] if consts is not None else None
    T = solve_T_blocked(A, bl=solve_block, bp=2 * solve_block, sel=sel)

    U = torch.matmul(T, v_beta)
    W = torch.matmul(T, k_beta * eg)

    # --- inter-chunk with state ---
    P = (torch.matmul(qc, kc.transpose(-1, -2)) * decay) * tril_incl
    v_new = U - torch.matmul(W, S_in)
    attn_inter = torch.matmul(qc * eg, S_in)
    oc = attn_inter + torch.matmul(P, v_new)

    g_last = g[..., -1:]                                               # [B,H,1] keepdim
    e_glast = torch.exp(g_last).unsqueeze(-1)                          # [B,H,1,1]
    dec_k = torch.exp(g_last - g).unsqueeze(-1)                        # [B,H,C,1]
    S_out = S_in * e_glast + torch.matmul((kc * dec_k).transpose(-1, -2), v_new)
    return oc, S_out


class GDNChunk(nn.Module):
    """Bakes the [C,C] masks as constant buffers so the exported ONNX has no runtime
    mask-construction ops (EyeLike/GreaterOrEqual/ConstantOfShape) that QNN can't lower."""

    def __init__(self, C=CHUNK):
        super().__init__()
        tl, sl, cu, ey = _masks(C, "cpu", torch.float32)
        for n, t in (("tril_incl", tl), ("strict_low", sl), ("cumsum_U", cu), ("eye", ey)):
            self.register_buffer(n, t)

    def forward(self, qc, kc, vc, gc, betac, S_in):
        masks = (self.tril_incl, self.strict_low, self.cumsum_U, self.eye)
        return gdn_chunk_onnx(qc, kc, vc, gc, betac, S_in, masks=masks)


# structural-constant inputs for the quantized HTP graph (fed at runtime as activations so the
# cumsum / block-select matmuls stay activation×activation MatMul, not FullyConnected)
def const_inputs(C=CHUNK, bl=16, bp=32):
    nb = C // bl
    out = {"cumsum_U": cumsum_array(C)}                       # [C,C]
    for i in range(nb):
        out[f"sel{i}"] = sel_array(i, C, bl, bp)              # [bp,C]
    return out


CONST_INPUT_NAMES = list(const_inputs().keys())              # cumsum_U, sel0..sel3


class GDNChunkQ(nn.Module):
    """Quantized-path module: the masks used in ELEMENTWISE ops stay baked, but the matmul
    structural constants (cumsum_U, block selectors) are forward INPUTS so they quantize as
    activations and their matmuls remain int16-MatMul (not FullyConnected)."""

    def __init__(self, C=CHUNK, solve_block=16):
        super().__init__()
        tl, sl, cu, ey = _masks(C, "cpu", torch.float32)
        self.solve_block = solve_block
        for n, t in (("tril_incl", tl), ("strict_low", sl), ("eye", ey)):
            self.register_buffer(n, t)

    def forward(self, qc, kc, vc, gc, betac, S_in, cumsum_U, sel0, sel1, sel2, sel3):
        masks = (self.tril_incl, self.strict_low, None, self.eye)
        r4 = lambda s: s.reshape(1, 1, *s.shape[-2:])          # 4D so matmuls stay MatMul, not FC
        consts = {"cumsum_U": r4(cumsum_U), "sel": [r4(sel0), r4(sel1), r4(sel2), r4(sel3)]}
        return gdn_chunk_onnx(qc, kc, vc, gc, betac, S_in, masks=masks,
                              solve_block=self.solve_block, consts=consts)


def _rand_case(device="cpu", dtype=torch.float32, with_state=True, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g).to(device=device, dtype=dtype)
    B, H, C, D = 1, 32, CHUNK, 128
    qc, kc, vc = r(B, H, C, D), r(B, H, C, D), r(B, H, C, D)
    gc = -torch.nn.functional.softplus(r(B, H, C))                     # g<0 like real decay
    betac = torch.sigmoid(r(B, H, C))
    S_in = (r(B, H, D, D) if with_state else torch.zeros(B, H, D, D, device=device, dtype=dtype))
    return qc, kc, vc, gc, betac, S_in


INPUT_NAMES = ["qc", "kc", "vc", "gc", "betac", "S_in"]


def write_quant_overrides(path, bits=16):
    """quant_overrides.json forcing the non-negative const inputs (0/1 valued) to UNSIGNED
    symmetric int — HTP MatMul requires offset 0 for non-negative operands (signed tensors get
    offset -(2^(b-1))); a global symmetric schema mislabels these as signed and fails ctxgen."""
    qmax = (1 << (bits - 1)) - 1                              # symmetric: [-1,1] so 0/1 fit
    enc = {"bitwidth": bits, "dtype": "int", "is_symmetric": "True",
           "scale": 1.0 / qmax, "offset": -(1 << (bits - 1)), "min": -1.0, "max": 1.0}
    overrides = {"activation_encodings": {n: [dict(enc)] for n in CONST_INPUT_NAMES},
                 "param_encodings": {}}
    with open(path, "w") as f:
        json.dump(overrides, f, indent=2)
    return CONST_INPUT_NAMES


def symmetric_overrides_from_dump(dump_path, out_path, bits=16):
    """Read qairt-quantizer's --dump_encoding_json and rewrite EVERY tensor as symmetric
    (offset=-(2^(b-1)), scale=max_abs/(2^(b-1)-1)).  HTP int16 MatMul requires symmetric
    operands, but qairt assigns offset 0 to non-negative tensors; this forces them symmetric
    while keeping each tensor's calibrated range."""
    d = json.load(open(dump_path))
    qmax = (1 << (bits - 1)) - 1
    def fix(enc):
        e = enc[0] if isinstance(enc, list) else enc
        mx = max(abs(float(e.get("min", 0.0))), abs(float(e.get("max", 0.0)))) or 1.0
        out = {"bitwidth": bits, "dtype": "int", "is_symmetric": "True",
               "scale": mx / qmax, "offset": -(1 << (bits - 1)), "min": -mx, "max": mx}
        return [out]
    ov = {"activation_encodings": {}, "param_encodings": {}}
    for grp in ("activation_encodings", "param_encodings"):
        for name, enc in d.get(grp, {}).items():
            if name.endswith("_converted_unsigned_symmetric"):
                continue
            ov[grp][name] = fix(enc)
    json.dump(ov, open(out_path, "w"), indent=1)
    print(f"wrote {out_path}: {len(ov['activation_encodings'])} act + "
          f"{len(ov['param_encodings'])} param symmetric encodings")
    return ov


def write_consts(out_dir, storage="f4"):
    """Write the fixed structural-constant inputs (cumsum_U, sel0..3) as raw; return name list."""
    os.makedirs(out_dir, exist_ok=True)
    cs = const_inputs()
    for n, a in cs.items():
        a.astype(f"<{storage}").tofile(os.path.join(out_dir, f"{n}.raw"))
    return list(cs.keys())


def emit_io(out_dir, seed=1234, storage="f4"):
    """Write raw inputs (`storage` f4=fp32 / f2=fp16) + input_list.txt + fp64-ref outputs.

    Reference outputs are always stored fp32 (oc_ref.raw / S_out_ref.raw). fp16 storage is for
    HTP native-I/O runs (qnn-net-run --use_native_input_files reads the fp16 public tensor)."""
    os.makedirs(out_dir, exist_ok=True)
    args = _rand_case(dtype=torch.float32, with_state=True, seed=seed)
    for n, a in zip(INPUT_NAMES, args):
        a.numpy().astype(f"<{storage}").tofile(os.path.join(out_dir, f"{n}.raw"))
    with open(os.path.join(out_dir, "input_list.txt"), "w") as f:
        f.write(" ".join(f"{n}:={n}.raw" for n in INPUT_NAMES) + "\n")
    oc_ref, S_ref = gdn_chunk(*[a.double() for a in args])
    oc_ref.float().numpy().astype("<f4").tofile(os.path.join(out_dir, "oc_ref.raw"))
    S_ref.float().numpy().astype("<f4").tofile(os.path.join(out_dir, "S_out_ref.raw"))
    print(f"emitted IO -> {out_dir}/ (inputs {'fp16' if storage=='f2' else 'fp32'}, ref fp32)")
    return args


def emit_golden_io(out_dir, golden_dir="tests/gdn/golden", layer=0, chunk=0, storage="f4",
                   prompt=None, consts=False):
    """Emit a REAL Qwen3.5-4B chunk (post-conv q/k/v/g/beta) + its real incoming state S_in,
    so the QNN run is validated on real activations, not random. `chunk`>0 runs the fp64
    recurrence up to that chunk to produce the true S_in. `prompt` = basename (e.g. p21) or
    None to auto-pick the longest prompt (so chunk>0 is in range)."""
    files = sorted(glob.glob(os.path.join(golden_dir, f"*_L{layer:02d}.npz")))
    if not files:
        raise FileNotFoundError(f"no golden *_L{layer:02d}.npz under {golden_dir}")
    if prompt:
        files = [f for f in files if os.path.basename(f).startswith(prompt)] or files
        z = np.load(files[0])
    else:                                                  # auto: longest sequence
        f = max(files, key=lambda f: int(np.load(f)["query"].shape[1]))
        files = [f]; z = np.load(f)
    get = lambda k: torch.from_numpy(np.ascontiguousarray(bf16_to_f32(z[k]))).double()
    # [B,T,H,D] -> [B,H,T,D]; pad to a whole number of chunks
    q, k, v = (get(x).transpose(1, 2) for x in ("query", "key", "value"))
    gg, bb = (get(x).transpose(1, 2) for x in ("g", "beta"))
    T = q.shape[2]
    nchunks = (T + CHUNK - 1) // CHUNK
    if chunk >= nchunks:
        raise ValueError(f"{os.path.basename(files[0])} has {nchunks} chunks; chunk={chunk} OOR")
    pad = nchunks * CHUNK - T
    if pad:
        F = torch.nn.functional
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        gg, bb = (F.pad(x, (0, pad)) for x in (gg, bb))
    B, H = q.shape[0], q.shape[1]
    S = torch.zeros(B, H, 128, 128, dtype=torch.float64)
    sl0 = lambda i: slice(i * CHUNK, (i + 1) * CHUNK)
    for i in range(chunk):                                  # rebuild real state up to `chunk`
        _, S = gdn_chunk(q[:, :, sl0(i)], k[:, :, sl0(i)], v[:, :, sl0(i)],
                         gg[:, :, sl0(i)], bb[:, :, sl0(i)], S)
    sl = sl0(chunk)
    args = (q[:, :, sl], k[:, :, sl], v[:, :, sl], gg[:, :, sl], bb[:, :, sl], S)
    os.makedirs(out_dir, exist_ok=True)
    for n, a in zip(INPUT_NAMES, args):
        a.float().numpy().astype(f"<{storage}").tofile(os.path.join(out_dir, f"{n}.raw"))
    cnames = write_consts(out_dir, storage) if consts else []
    cref = "".join(f" {c}:={c}.raw" for c in cnames)
    with open(os.path.join(out_dir, "input_list.txt"), "w") as f:
        f.write(" ".join(f"{n}:={n}.raw" for n in INPUT_NAMES) + cref + "\n")
    oc_ref, S_ref = gdn_chunk(*args)
    oc_ref.float().numpy().astype("<f4").tofile(os.path.join(out_dir, "oc_ref.raw"))
    S_ref.float().numpy().astype("<f4").tofile(os.path.join(out_dir, "S_out_ref.raw"))
    print(f"emitted REAL golden IO -> {out_dir}/ "
          f"(L{layer:02d} {os.path.basename(files[0])} chunk {chunk}/{nchunks}, "
          f"inputs {'fp16' if storage=='f2' else 'fp32'})")
    return args


def _golden_chunk_args(npz_path, chunk):
    """Real (q,k,v,g,beta,S_in) fp64 args for `chunk` of one golden capture (rebuilds S)."""
    z = np.load(npz_path)
    get = lambda k: torch.from_numpy(np.ascontiguousarray(bf16_to_f32(z[k]))).double()
    q, k, v = (get(x).transpose(1, 2) for x in ("query", "key", "value"))
    gg, bb = (get(x).transpose(1, 2) for x in ("g", "beta"))
    T = q.shape[2]; nchunks = (T + CHUNK - 1) // CHUNK
    chunk = min(chunk, nchunks - 1)
    pad = nchunks * CHUNK - T
    if pad:
        F = torch.nn.functional
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        gg, bb = (F.pad(x, (0, pad)) for x in (gg, bb))
    S = torch.zeros(q.shape[0], q.shape[1], 128, 128, dtype=torch.float64)
    sl = lambda i: slice(i * CHUNK, (i + 1) * CHUNK)
    for i in range(chunk):
        _, S = gdn_chunk(q[:, :, sl(i)], k[:, :, sl(i)], v[:, :, sl(i)], gg[:, :, sl(i)], bb[:, :, sl(i)], S)
    c = sl(chunk)
    return (q[:, :, c], k[:, :, c], v[:, :, c], gg[:, :, c], bb[:, :, c], S)


def emit_calib(out_dir, golden_dir="tests/gdn/golden", layer=0, max_samples=16, consts=False):
    """Write a multi-sample fp32 calibration set (real golden chunks) + calib_list.txt for
    qairt-quantizer. Covers chunk0 (S_in=0) and chunk1 (real state) across the layer's prompts.
    `consts`=True also references the fixed structural-constant inputs in every list line."""
    man = json.load(open(os.path.join(golden_dir, "manifest.json")))
    calib = [r["file"] for r in man["records"] if r["layer"] == layer and r["split"] == "calib"]
    os.makedirs(out_dir, exist_ok=True)
    cnames = write_consts(out_dir) if consts else []
    cref = "".join(f" {c}:={c}.raw" for c in cnames)
    lines, n = [], 0
    for f in calib:
        p = os.path.join(golden_dir, f)
        T = int(np.load(p)["query"].shape[1])
        for chunk in ([0, 1] if T > CHUNK else [0]):
            if n >= max_samples:
                break
            sub = os.path.join(out_dir, f"s{n:02d}")
            os.makedirs(sub, exist_ok=True)
            for nm, a in zip(INPUT_NAMES, _golden_chunk_args(p, chunk)):
                a.float().numpy().astype("<f4").tofile(os.path.join(sub, f"{nm}.raw"))
            lines.append(" ".join(f"{nm}:=s{n:02d}/{nm}.raw" for nm in INPUT_NAMES) + cref)
            n += 1
    with open(os.path.join(out_dir, "calib_list.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"emitted {n} calib samples -> {out_dir}/calib_list.txt (L{layer:02d})"
          + (f" +{len(cnames)} const inputs" if consts else ""))


def compare_out(out_dir, result_dir=None):
    """Compare qnn-net-run outputs (Result_0/oc.raw, S_out.raw) to the fp64 ref."""
    result_dir = result_dir or os.path.join(out_dir, "output")
    ref_oc = np.fromfile(os.path.join(out_dir, "oc_ref.raw"), "<f4")
    ref_S = np.fromfile(os.path.join(out_dir, "S_out_ref.raw"), "<f4")

    def find(name):
        for cand in (os.path.join(result_dir, f"{name}.raw"),
                     os.path.join(result_dir, "Result_0", f"{name}.raw")):
            if os.path.exists(cand):
                return cand
        hits = glob.glob(os.path.join(result_dir, "**", f"{name}*.raw"), recursive=True)
        return hits[0] if hits else None

    tol = float(os.environ.get("GDN_NATIVE_TOL", "1e-3"))
    ok = True
    for name, ref in (("oc", ref_oc), ("S_out", ref_S)):
        p = find(name)
        if not p:
            print(f"  {name}: MISSING in {result_dir}"); ok = False; continue
        raw = np.fromfile(p, np.uint8)
        # auto-detect output storage by byte count (fp16 = 2 B/elem, fp32 = 4 B/elem)
        got = (raw.view("<f2") if raw.size == ref.size * 2 else raw.view("<f4")).astype(np.float32)
        e = relerr(torch.from_numpy(got.copy()), torch.from_numpy(ref.copy()))
        good = e < tol
        ok &= good
        print(f"  {name}: relerr {e:.2e}  {'PASS' if good else 'FAIL'}  "
              f"(tol {tol:.0e}, {os.path.relpath(p)})")
    return ok


def export_onnx(path, opset=17):
    m = GDNChunk().eval()
    args = _rand_case(dtype=torch.float32)
    names = ["qc", "kc", "vc", "gc", "betac", "S_in"]
    torch.onnx.export(
        m, args, path, opset_version=opset,
        input_names=names, output_names=["oc", "S_out"], dynamo=False,
    )
    print(f"wrote {path}")
    return args


def export_onnx_q(path, opset=17):
    """Export the QUANTIZED-path graph (GDNChunkQ): 11 inputs incl. the structural constants."""
    m = GDNChunkQ().eval()
    args = _rand_case(dtype=torch.float32)
    cs = const_inputs()
    cargs = tuple(torch.from_numpy(cs[n]) for n in CONST_INPUT_NAMES)
    names = INPUT_NAMES + CONST_INPUT_NAMES
    torch.onnx.export(m, args + cargs, path, opset_version=opset,
                      input_names=names, output_names=["oc", "S_out"], dynamo=False)
    print(f"wrote {path} (quantized-path, {len(names)} inputs)")
    return args, cargs


def _ort_check(path, args, cargs=None):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feed = {n: a.numpy() for n, a in zip(INPUT_NAMES, args)}
    if cargs is not None:
        feed.update({n: a.numpy() for n, a in zip(CONST_INPUT_NAMES, cargs)})
    oc_ort, S_ort = sess.run(None, feed)
    oc_ref, S_ref = gdn_chunk(*[a.double() for a in args])
    eo = relerr(torch.from_numpy(oc_ort), oc_ref)
    es = relerr(torch.from_numpy(S_ort), S_ref)
    print(f"ORT vs fp64 ref:  oc relerr {eo:.2e}   S_out relerr {es:.2e}   "
          f"{'PASS' if max(eo, es) < 1e-4 else 'CHECK'}")
    return max(eo, es)


def validate(golden_dir="tests/gdn/golden"):
    # 1) random cases (zero + nonzero state), torch module vs fp64 reference
    for with_state in (False, True):
        args = _rand_case(dtype=torch.float64, with_state=with_state)
        oc, S = gdn_chunk_onnx(*args)
        oc_r, S_r = gdn_chunk(*args)
        tag = "S!=0" if with_state else "S==0"
        print(f"  random {tag}: oc {relerr(oc, oc_r):.2e}  S_out {relerr(S, S_r):.2e}")
    # 2) a real golden chunk (first C tokens of an L00 capture, S_in=0)
    files = sorted(glob.glob(os.path.join(golden_dir, "*_L00.npz")))
    if files:
        z = np.load(files[0])
        get = lambda k: torch.from_numpy(np.ascontiguousarray(bf16_to_f32(z[k]))).double()
        F = torch.nn.functional
        pad1 = lambda x: F.pad(x.transpose(1, 2)[:, :, :CHUNK], (0, 0, 0, max(0, CHUNK - x.shape[1])))
        padg = lambda x: F.pad(x.transpose(1, 2)[:, :, :CHUNK], (0, max(0, CHUNK - x.shape[1])))
        q, k, v = (pad1(get(x)) for x in ("query", "key", "value"))
        gg, bb = (padg(get(x)) for x in ("g", "beta"))
        S0 = torch.zeros(1, 32, 128, 128, dtype=torch.float64)
        oc, _ = gdn_chunk_onnx(q, k, v, gg, bb, S0)
        oc_r, _ = gdn_chunk(q, k, v, gg, bb, S0)
        print(f"  golden L00 chunk0: oc {relerr(oc, oc_r):.2e}  ({os.path.basename(files[0])})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--export", metavar="PATH")
    ap.add_argument("--emit-io", metavar="DIR")
    ap.add_argument("--export-q", metavar="PATH", help="export quantized-path graph (const inputs)")
    ap.add_argument("--compare", metavar="DIR")
    ap.add_argument("--result", metavar="DIR", help="dir holding qnn outputs (default DIR/output)")
    ap.add_argument("--golden", default="tests/gdn/golden")
    args = ap.parse_args()
    if args.validate:
        validate(args.golden)
    if args.export:
        a = export_onnx(args.export)
        _ort_check(args.export, a)
    if args.export_q:
        a, c = export_onnx_q(args.export_q)
        _ort_check(args.export_q, a, c)
    if args.emit_io:
        emit_io(args.emit_io)
    if args.compare:
        ok = compare_out(args.compare, args.result)
        sys.exit(0 if ok else 1)
    if not (args.validate or args.export or args.export_q or args.emit_io or args.compare):
        ap.print_help()


if __name__ == "__main__":
    main()
