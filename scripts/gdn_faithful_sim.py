#!/usr/bin/env python3
"""DEVICE-FAITHFUL fixed-point simulator for the all-integer GDN chunk on HTP.

Runs the exact deployed graph (gdn_onnx_kernel.GDNChunkQ) and quantizes the output of every
compute op via a TorchFunctionMode, modelling the TWO things that make the device differ from a
naive float-accumulate sim:

  1. **L2Norm is the native fused op** — its internal sum-of-squares / rsqrt are NOT quantized
     (l2norm_lastdim uses torch F.normalize, which the QAIRT converter fuses into the QNN L2Norm
     op; verified with `qairt-dlc-info`). Quantizing those internals is a SIM ARTIFACT.
  2. **The matmul accumulator is int32, and it OVERFLOWS.** int16×int16 ≈ 2^30 per product;
     summed over the 128-dim head contraction → ~2^37 ≫ int32 (2^31). `ACC=int32` models the
     saturating int32 accumulator and reproduces the device; `ACC=float` does not.

KEY RESULT (real golden L00, oc relerr p00/p29):
  ACC=float  w16a16 : 0.001 / 0.001   (wrong — no overflow modelled)
  ACC=int32  w16a16 : 0.146 / 0.693   == DEVICE (0.164 / 0.431); int16×int16 OVERFLOWS int32
  ACC=int32  w8a16  : 0.053 / 0.049   one matmul operand int8 -> accumulator fits 2^31
=> "all int16" (int16×int16 matmul) is impossible for the 128-dim GEMMs: the int32 accumulator
   overflows. The weight-port must be int8 (w8a16, the project's locked design); acts stay int16.
   Everything else (exp, the (I-A)^-1 solve, elementwise) is fine at int16.

Run:  ACC=int32 WBITS=16 .venv/bin/python scripts/gdn_faithful_sim.py   # reproduces device 16%
      ACC=int32 WBITS=8  .venv/bin/python scripts/gdn_faithful_sim.py   # w8a16
"""
import os, sys, glob, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.overrides import TorchFunctionMode
import gdn_onnx_kernel as G
from gdn_onnx_kernel import GDNChunkQ, const_inputs, _golden_chunk_args, CONST_INPUT_NAMES
from gdn_ref_kernel import gdn_chunk, relerr

ACC = os.environ.get("ACC", "int32")           # float | int32 (saturating) matmul accumulator
WBITS = int(os.environ.get("WBITS", "16"))      # matmul weight-port (in[1]) bits: 16=w16a16, 8=w8a16
ABITS = int(os.environ.get("ABITS", "16"))      # activation / elementwise bits
GOLDEN = os.environ.get("GDN_GOLDEN", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "gdn", "golden"))
INT32 = 2**31 - 1
EW = {"mul", "add", "sub", "exp", "div", "neg"}
_skip = [False]


def _fused_l2(x):                              # native L2Norm: internals computed high-precision
    _skip[0] = True
    try:
        return torch.nn.functional.normalize(x, p=2.0, dim=-1, eps=G.EPS)
    finally:
        _skip[0] = False


G.l2norm_lastdim = _fused_l2


def _qint(t, bits):                            # symmetric int quantize -> (int tensor, scale)
    mx = float(t.abs().max()) or 1.0
    qm = (1 << (bits - 1)) - 1
    sc = mx / qm
    return torch.clamp(torch.round(t / sc), -qm - 1, qm), sc


def _qsym(t, mx, bits):                         # symmetric quant-dequant to `bits`
    qm = (1 << (bits - 1)) - 1
    sc = mx / qm
    return torch.clamp(torch.round(t / sc), -qm - 1, qm) * sc if sc > 0 else t


class _QMode(TorchFunctionMode):
    def __init__(self, ranges, mode):
        self.r = ranges; self.m = mode; self.i = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        n = getattr(func, "__name__", "")
        if (n == "matmul" and ACC == "int32" and not _skip[0] and len(args) >= 2
                and all(isinstance(x, torch.Tensor) for x in args[:2]) and args[0].is_floating_point()):
            Ai, sa = _qint(args[0], ABITS)         # in[0] activation int{ABITS}
            Bi, sb = _qint(args[1], WBITS)         # in[1] weight     int{WBITS}
            acc = torch.clamp(torch.matmul(Ai, Bi), -INT32, INT32)   # int32 accumulator saturates
            return acc * (sa * sb)
        out = func(*args, **kwargs)
        if n in EW and not _skip[0] and isinstance(out, torch.Tensor) and out.is_floating_point() and out.numel() > 1:
            k = self.i; self.i += 1
            mx = float(out.abs().max())
            if self.m == "calib":
                self.r[k] = max(self.r.get(k, 0.0), mx)
            else:
                out = _qsym(out, self.r.get(k, mx), ABITS)
        return out


def _run(args, ranges, mode):
    md = GDNChunkQ().eval().double()
    cs = const_inputs()
    cargs = [torch.from_numpy(cs[n]).double() for n in CONST_INPUT_NAMES]
    one = torch.ones(1, 32, 1, 1, dtype=torch.float64)         # per-head vscale off (identity)
    with _QMode(ranges, mode):
        return md(*args, *cargs, one, one)


def run(layer=0, calib_n=12, test=("p00", "p29")):
    man = json.load(open(os.path.join(GOLDEN, "manifest.json")))
    calib = [r["file"] for r in man["records"] if r["layer"] == layer and r["split"] == "calib"][:calib_n]
    ranges = {}
    for f in calib:
        _run(list(_golden_chunk_args(os.path.join(GOLDEN, f), 0)), ranges, "calib")
    out = {}
    for P in test:
        a = list(_golden_chunk_args(sorted(glob.glob(os.path.join(GOLDEN, f"{P}*_L{layer:02d}.npz")))[0], 0))
        oc_ref, _ = gdn_chunk(*a)
        oc_q, _ = _run(a, ranges, "eval")
        out[P] = relerr(oc_q, oc_ref)
    return out


if __name__ == "__main__":
    e = run()
    print(f"ACC={ACC} w{WBITS}a{ABITS}: oc relerr " + "  ".join(f"{k} {v:.3f}" for k, v in e.items()))
