#!/usr/bin/env python3
"""Single-layer GDN kernel test.

A single test == ONE GDN layer. For the layer we:
  1. build STATIC, PER-TENSOR quant params from that layer's *calib* prompts under the HTP
     hardware constraint: every GEMM is A @ W^T with
       - A (left)  = per-tensor ASYMMETRIC int16 (activation port: scale + zero-point)
       - W (right) = per-tensor SYMMETRIC  int8  (weight port: scale only)
     The per-GEMM operand->slot map (ASYM_SIDE below) orients each GEMM so the outlier-prone
     operand lands in A (int16) and the bounded one in W (int8); 6/8 GEMMs are computed
     transposed (C^T) to satisfy this. Asymmetric int16 fits the non-zero-centered post-silu
     activations; the zero-point is folded into the HMX drain.
  2. run the kernel on that layer's held-out *test* prompts,
  3. assert the output o is within tolerance of the fp32 reference.

By default we FOCUS on the best-aligned layer (L00); set GDN_LAYER=all to test every layer.

Env: GDN_LAYER=<n>|all  GDN_TOL=1.5e-2
Run:        .venv/bin/python tests/gdn/test_gdn_layer.py
Under pytest: .venv/bin/python -m pytest tests/gdn/test_gdn_layer.py -q
"""
import json, os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from gdn_ref_kernel import gdn_kernel, load_case, relerr  # noqa: E402

GOLDEN = os.environ.get("GDN_GOLDEN", os.path.join(ROOT, "tests", "gdn", "golden"))
TOL = float(os.environ.get("GDN_TOL", "1.5e-2"))
FOCUS_LAYER = os.environ.get("GDN_LAYER", "0")            # best-aligned layer
SITES = ["A_kk", "A_Uv", "A_Wk", "B_qk", "B_WS", "B_qS", "B_Pv", "B_kv"]
I8, I16 = 127, 32767                                       # symmetric int8 / asymmetric int16

# HTP hardware constraint: every GEMM is A @ W^T with A (left) = ASYMMETRIC int16
# (activation port) and W (right) = SYMMETRIC int8 (weight port). Which reference operand
# becomes A vs W is OUR orientation choice (compute C or C^T) — we orient each GEMM so the
# outlier-prone operand lands in A (int16) and the bounded one in W (int8). 'A'/'B' below =
# which arg of the reference mm(X=A_arg, Y=B_arg) is the ASYMMETRIC (int16) operand; the
# other is symmetric int8.  transpose? = whether the HTP kernel computes C^T (i.e. the int16
# operand is the reference's RIGHT operand, so the natural form must be transposed).
ASYM_SIDE = {
    "A_kk": "A",   # A=k_beta(int16)        W=k(int8)            no transpose
    "A_Uv": "B",   # A=v_beta(int16)        W=T(int8)            transpose
    "A_Wk": "B",   # A=k_beta*e^g(int16)    W=T(int8)            transpose
    "B_qk": "A",   # A=q(int16)             W=k(int8)            no transpose
    "B_WS": "B",   # A=S(int16)             W=W_mat(int8)        transpose
    "B_qS": "B",   # A=S(int16)             W=q*e^g(int8)        transpose
    "B_Pv": "B",   # A=v_new(int16)         W=P(int8)            transpose
    "B_kv": "B",   # A=v_new(int16)         W=(k*dec)(int8)      transpose
}


def by_layer():
    man = json.load(open(os.path.join(GOLDEN, "manifest.json")))
    layers = {}
    for r in man["records"]:
        layers.setdefault(r["layer"], {"calib": [], "test": []})[r["split"]].append(r["file"])
    return layers


def calibrate(calib_inputs):
    """Per (site, operand), per-tensor: running amax, min, max, sumsq, count."""
    st = {}
    def rec(A, B, site=None):
        for tag, X in (("A", A), ("B", B)):
            k = (site, tag)
            d = st.setdefault(k, {"amax": 0.0, "mn": float("inf"), "mx": float("-inf"), "sq": 0.0, "n": 0})
            d["amax"] = max(d["amax"], X.abs().max().item())
            d["mn"] = min(d["mn"], X.min().item())
            d["mx"] = max(d["mx"], X.max().item())
            d["sq"] += (X.double() ** 2).sum().item(); d["n"] += X.numel()
        return A @ B
    for c in calib_inputs:
        gdn_kernel(c["query"], c["key"], c["value"], c["g"], c["beta"], mm=rec)
    return st


def build_params(st):
    """Per HTP constraint: ASYM_SIDE operand -> ('asym', scale, zp) int16 (left/A);
    the other -> ('sym', scale, 0) int8 (right/W)."""
    p = {}
    for s in SITES:
        for tag in ("A", "B"):
            d = st[(s, tag)]
            if ASYM_SIDE[s] == tag:                               # int16 asymmetric (left/A)
                qmin, qmax = -I16 - 1, I16
                sc = (d["mx"] - d["mn"]) / (qmax - qmin) + 1e-12
                z = int(round(qmin - d["mn"] / sc))
                p[(s, tag)] = ("asym", sc, max(qmin, min(qmax, z)))
            else:                                                 # int8 symmetric (right/W)
                p[(s, tag)] = ("sym", d["amax"] / I8 + 1e-12, 0)
    return p


def qdq(X, param):
    kind, sc, z = param
    if kind == "sym":
        return sc * torch.clamp(torch.round(X / sc), -I8, I8)
    return sc * (torch.clamp(torch.round(X / sc) + z, -I16 - 1, I16) - z)


def static_mm(params):
    def mm(A, B, site=None):
        return qdq(A, params[(site, "A")]) @ qdq(B, params[(site, "B")])
    return mm


def eval_layer(layer):
    L = by_layer()[layer]
    calib = [load_case(os.path.join(GOLDEN, f)) for f in L["calib"]]
    mm = static_mm(build_params(calibrate(calib)))
    errs = []
    for f in L["test"]:
        c = load_case(os.path.join(GOLDEN, f))
        o_ref, _ = gdn_kernel(c["query"], c["key"], c["value"], c["g"], c["beta"])
        o_q, _ = gdn_kernel(c["query"], c["key"], c["value"], c["g"], c["beta"], mm=mm)
        errs.append((f, relerr(o_q, o_ref)))
    return errs


def _layers():
    avail = sorted(by_layer().keys()) if os.path.isdir(GOLDEN) else []
    if FOCUS_LAYER == "all":
        return avail
    return [int(FOCUS_LAYER)] if avail else []


# ---- pytest: one parametrized test per (focused) layer ----
try:
    import pytest
    @pytest.mark.parametrize("layer", _layers())
    def test_gdn_layer(layer):
        for f, e in eval_layer(layer):
            assert e < TOL, f"L{layer} {f}: relerr {e:.2e} >= tol {TOL:.1e}"
except ImportError:
    pass


def main():
    print(f"single-layer GDN test: per-tensor sym-int8 @ asym-int16  tol={TOL:.1e} "
          f"focus={FOCUS_LAYER}\n  golden={GOLDEN}\n")
    npass = 0; ls = _layers()
    for L in ls:
        errs = eval_layer(L)
        mx = max(e for _, e in errs); ok = mx < TOL; npass += ok
        worst = max(errs, key=lambda x: x[1])
        print(f"  L{L:02d}: {'PASS' if ok else 'FAIL'}  max relerr {mx:.2e}  "
              f"(worst {worst[0]}) over {len(errs)} test prompts")
    print(f"\n{npass}/{len(ls)} pass (per-tensor sym-int8 @ asym-int16, tol {TOL:.1e})")
    sys.exit(0 if npass == len(ls) else 1)


if __name__ == "__main__":
    main()
