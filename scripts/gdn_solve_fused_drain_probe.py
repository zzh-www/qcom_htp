#!/usr/bin/env python3
"""Step-2 待办2 (end-to-end oc) + 待定决策 (w16a16 vs w8a16): does the STATIC-int merge
with FUSED accumulation + the REAL convhbh cvt drain hold oc on real GDN golden data?

Models the hardware faithfully (per 待办1/2 sim findings,
reference_convhbh_cvt_drain_semantics):
  - operands: static SYMMETRIC quant (zp=0).  Operand scales sA,sT are PREPROCESSING
    (arbitrary float, done off-device); only the DRAIN gain is HW-constrained.
  - fused inner accumulation: Scode = Σ_{k=j..i-1} cA_ik @ cT_kj in ONE int32 acc
    (wrap on overflow, to detect int32 insufficiency).  NO intermediate drain.
  - drain = HARDWARE model: int16 = clip(round_half_up(f16(net_i32) * 2^kshift),-32768,32767),
    where 2^kshift is the ONLY requant the HW offers (power-of-2 gain via the bias-const
    exponent), chosen statically to fill the int16 range.  f16(net) emulates the cvt's
    net-accumulator f16 conversion (exact <=2048, ~2^-11 rel above).
  - T blocks are stored as VALUES (code * carried-scale) and re-quantized on each use,
    so the drained scale is just carried, NOT forced to a global scale.

Variants (the 待定决策): w16a16 / w8a16 (two operand-role assignments) / int8.
drain='exact' (skip f16) isolates the f16-drain loss from the power-of-2 requant loss.

Reproduce: source scripts/env.sh && python scripts/gdn_solve_fused_drain_probe.py
"""
from __future__ import annotations
import sys, os, math
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import gdn_solve_static_quant_probe as Q

C, BL, NB, H = Q.C, Q.BL, Q.NB, Q.H
A, Texact = Q.A, Q.Texact
amax, tmax = Q.amax, Q.tmax


def qcode_sym(x, scale, qmax):
    return np.clip(np.round(x / scale), -qmax, qmax).astype(np.int64)


def round_sig_bits(net, mant_bits=11):
    """Round each int to `mant_bits` significant bits — models the cvt's mantissa-limited
    (f16-like, 11-bit) intermediate.  Unlike np.float16 this never overflows to inf for
    large int32 (the HW applies the narrowing scale, it does not naive-f16 the raw acc).
    Empirically matches the sim drain to <=~1 LSB (sim: net30000>>1 -> +1)."""
    net = net.astype(np.int64)
    n = np.abs(net)
    bl = np.where(n > 0, np.floor(np.log2(np.maximum(n, 1))).astype(np.int64) + 1, 0)
    shift = np.maximum(bl - mant_bits, 0)
    half = np.where(shift > 0, 1 << np.maximum(shift - 1, 0), 0)
    rounded = ((n + half) >> shift) << shift
    return np.sign(net) * rounded


def hw_drain(net_i32, model="f16"):
    """net int32 -> (int16 codes, kshift, eff_gain).  kshift = power-of-2 downshift that
    fills the int16 range; code = clip(round_half_up(narrow(net)*2^kshift), -32768, 32767).
    model='f16' applies the 11-sig-bit mantissa narrowing; 'exact' skips it."""
    net_max = int(np.abs(net_i32).max())
    if net_max <= 0:
        return np.zeros_like(net_i32), 0, 1.0
    kshift = int(math.floor(math.log2(32767.0 / net_max)))   # <=0 if net_max>32767
    narrowed = round_sig_bits(net_i32) if model == "f16" else net_i32
    v = narrowed.astype(np.float64)
    code = np.floor(v * (2.0 ** kshift) + 0.5)
    code = np.clip(code, -32768, 32767).astype(np.int64)
    return code, kshift, 2.0 ** kshift


def imatmul_wrap(cA, cT):
    P64 = cA @ cT
    P32 = ((P64 + 2**31) % 2**32 - 2**31)
    ov = int(np.sum(np.abs(P64) > 2**31 - 1))
    return P32, ov


# per-block-distance |T| max (STATIC, precomputable from calibration) -> layered scale.
# Still static (compile-time const per block position), just one scale per distance d.
_Tdist = None
def tdist_max(d):
    global _Tdist
    if _Tdist is None:
        _Tdist = {}
        for dd in range(NB):
            vals = [np.abs(Texact[h, (j+dd)*BL:(j+dd+1)*BL, j*BL:(j+1)*BL]).max()
                    for h in range(H) for j in range(NB-dd)]
            _Tdist[dd] = max(max(vals), 1e-30)
    return _Tdist[d]


def solve_fused(h, qm, drain_model="f16", layered=False):
    qmA, qmT = qm
    Ah = A[h]
    sA = amax / qmA
    sT_global = tmax / qmT
    T = np.zeros((C, C))
    Tdist = np.zeros((NB, NB), dtype=int)   # block-distance of each T block (for layered scale)
    ov = 0
    maxcode = 0
    for i in range(NB):
        T[i*BL:(i+1)*BL, i*BL:(i+1)*BL] = np.linalg.inv(np.eye(BL) - Ah[i*BL:(i+1)*BL, i*BL:(i+1)*BL])
    for d in range(1, NB):
        for j in range(NB-d):
            i = j + d
            # inner fused accumulation (one int32 acc over k, no intermediate drain)
            Scode = np.zeros((BL, BL), dtype=np.int64)
            for k in range(j, i):
                sTk = (tdist_max(k - j) / qmT) if layered else sT_global   # T_kj distance = k-j
                cA = qcode_sym(Ah[i*BL:(i+1)*BL, k*BL:(k+1)*BL], sA, qmA)
                cT = qcode_sym(T[k*BL:(k+1)*BL, j*BL:(j+1)*BL], sTk, qmT)
                # NOTE: layered changes per-term operand scale -> products NOT directly
                # additive in fixed point.  Renormalize each term to a common scale before
                # the int32 acc (still a static shift, precomputable).  Model exactly:
                P32, o = imatmul_wrap(cA, cT); ov += o
                if layered:
                    # rescale this term's value to the global product scale sA*sT_global
                    term_scale = sA * sTk
                    P32 = np.round(P32.astype(np.float64) * (term_scale / (sA * sT_global))).astype(np.int64)
                Scode = ((Scode + P32 + 2**31) % 2**32 - 2**31)
            cS, kS, gS = hw_drain(Scode, drain_model)         # int16 S codes
            sS = sA * sT_global / gS                                  # value(S) = cS * sS
            maxcode = max(maxcode, int(np.abs(cS).max()))
            # outer Tii @ S
            cTii = qcode_sym(T[i*BL:(i+1)*BL, i*BL:(i+1)*BL], sT_global, qmT)
            Tcode, o = imatmul_wrap(cTii, cS); ov += o
            cTij, kT, gT = hw_drain(Tcode, drain_model)
            sTij = sT_global * sS / gT                                # value(T) = cTij * sTij
            maxcode = max(maxcode, int(np.abs(cTij).max()))
            T[i*BL:(i+1)*BL, j*BL:(j+1)*BL] = cTij.astype(np.float64) * sTij
    return T, ov, maxcode


def main():
    print(f"\n=== fused-drain static-int merge, REAL convhbh drain model (C={C}, H={H}) ===")
    print(f"|A|max={amax:.4f} |T|max={tmax:.4f}  (zp=0 symmetric; operand scales=float preproc, "
          f"drain gain=power-of-2)")
    variants = [
        ("w16a16 (act16 x wt16)", (32767, 32767)),
        ("w8a16  (act16 x wt8) ", (32767, 127)),
        ("w8a16' (act8  x wt16)", (127, 32767)),
        ("int8   (act8  x wt8) ", (127, 127)),
    ]
    print(f"\n{'variant':>24} {'layered':>7} {'drain':>6}  {'oc(mean)':>9} {'oc(max)':>9} {'ovf':>5} {'maxcode':>8}")
    for name, qm in variants:
        for layered in (False, True):
            dm = "f16"
            res = [solve_fused(h, qm, dm, layered) for h in range(H)]
            ocs = [Q.oc(T, h) for h, (T, _, _) in enumerate(res)]
            ov = max(o for _, o, _ in res)
            mc = max(c for _, _, c in res)
            tag = "per-d" if layered else "global"
            print(f"{name:>24} {tag:>7} {dm:>6}  {np.mean(ocs):9.5f} {np.max(ocs):9.5f} {ov:5d} {mc:8d}")
    print("\nglobal = single static scale (fusion-compatible).  per-d = per-block-distance static scale")
    print("(BREAKS fusion: forces per-term rescale = the 51% glue we eliminate).  Shows the precision")
    print("ceiling if we gave up fusion.  Goal: cheapest FUSION-COMPATIBLE (global) variant <~1.2e-2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
