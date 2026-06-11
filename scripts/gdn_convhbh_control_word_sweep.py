#!/usr/bin/env python3
"""Step-2 待办1: nail the convhbh cvt-drain control word — confirm "exponent field
= power-of-2 shift k" is arbitrarily settable (incl. DOWNSHIFT) with zero rounding.

Why: the static-int merge drains an int32 HMX accumulator to int16 as
  out = saturate(acc >> k) + zp     (k = static-scale exponent, pure arithmetic shift)
so we must prove the cvt control word's exponent field is a clean settable k, and
that word2=0x0000 truncates (no float rounding).

Method (all hexagon-sim, ZERO SSR).  Use a UNIFORM-P probe so readout needs NO
depack: constant activation (act-128 = a, every byte the same) + r weight rows all
= b  =>  P[m,n] = a*b*r  CONSTANT for every (m,n).  The drained output is then a
single value = clip(zp + shift(P,k,round), 0, 65535); we read the dominant nonzero
u16 and compare to the model.  Layout/M-pass degeneracy is irrelevant (all cells
equal), so this isolates ONLY the cvt gain/shift/rounding.

Sweep word0 = 0x4040 + k*0x400 (each exponent step = x2), word2 in {0x0000,0x0008}.

Reproduce: source scripts/env.sh && python scripts/gdn_convhbh_control_word_sweep.py
"""
from __future__ import annotations
import argparse
import re
import sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "example" / "handwritten_hmx_matmul"))

import gdn_hmx_convhbh_sim as C
import gdn_hmx_matmul_sim as base
from prepare_owned_inputs import pack_w8_kmajor

M = K = N = 64
ZP = 0x8000


def build_bias(w_i8, const_words):
    """convhbh w8a16 bias surface with a CHOSEN const (word0..word3) + eff=-128*sum_w."""
    k, n = w_i8.shape
    eff = (-128 * w_i8.astype(np.int32).sum(axis=0)).astype(np.int32)
    pk = np.zeros((n // 32, 512), np.uint8)
    const = np.array(const_words, np.uint16).view(np.uint8)
    for nt in range(n // 32):
        for par in (0, 1):
            hb = par * 256
            for lane, c in enumerate(range(par, 32, 2)):
                col = nt * 32 + c
                lb = hb + 8 * lane
                pk[nt, lb:lb + 8] = const
                pk[nt, hb + 128 + 8 * lane:hb + 132 + 8 * lane] = (
                    np.array([int(eff[col])], np.int32).view(np.uint8))
    return pk.reshape(-1)


def run_uniform(a, b, r, const_words, keep=False):
    """act-128 = a (uniform), w[:r,:] = b  =>  P = a*b*r everywhere.  Returns (Pconst, o16, raw)."""
    act_u8 = np.full((M, K), 128 + a, np.uint8)
    w = np.zeros((K, N), np.int32)
    w[:r, :] = b
    w_i8 = w.astype(np.int8)
    Pconst = a * b * r
    d = C.descriptor()
    # uniform activation -> layout-independent at the u16 level.  The convhbh single
    # u8 pass reads the HIGH byte only; the low byte MUST be 0 (a nonzero low byte
    # leaks a fractional term).  So each u16 = (128+a)<<8 : bytes [lo=0, hi=128+a].
    act_packed = bytes([0, 128 + a]) * (d["act_surface_bytes"] // 2)
    w_packed = pack_w8_kmajor(w_i8).tobytes()
    bias_bytes = build_bias(w_i8, const_words).tobytes()
    arrays = "\n".join([
        base.c_u8_array("k_activation", act_packed),
        base.c_u8_array("k_packed_weight", w_packed),
        base.c_u8_array("k_folded_bias", bias_bytes),
        base.c_u32_array("k_act_off", d["act_off"]),
        base.c_u32_array("k_out_off", d["out_off"]),
        base.c_u32_array("k_mask_control", C.CONVHBH_MASK),
    ])
    od, ad = d["out_desc"], d["act_desc"]
    src = C.HARNESS % dict(
        ARRAYS=arrays, ACT_BYTES=len(act_packed), WEIGHT_BYTES=len(w_packed),
        BIAS_BYTES=len(bias_bytes), OUT_BYTES=d["out_buf_bytes"], ACT_ENTRIES=len(d["act_off"]),
        OUT_ENTRIES=len(d["out_off"]), E0=C.CONVHBH_EXTRA[0], E1=C.CONVHBH_EXTRA[1], E2=C.CONVHBH_EXTRA[2],
        OTS=od["out_table_stride_dwords"], OYS=od["out_y_stride_words"], NTP=od["n_tiles_pow2"],
        MTM=od["m_total_minus_step"], KTB=od["k_total_bytes"],
        NAP=ad["n_act_pairs"], AYS=ad["act_table_y_stride_words"])
    out = base.build_and_run(src, keep=keep)
    if out is None:
        return Pconst, None, "BUILD FAIL"
    om = re.search(r"\[OUT\]([0-9a-f]+)", out)
    if not om:
        return Pconst, None, out
    o16 = np.frombuffer(bytes.fromhex(om.group(1)), dtype="<u2").astype(np.int64)
    return Pconst, o16, out


def dominant_value(o16):
    """The uniform-P output: the most common value among the cells the kernel actually
    drained.  Exclude pure zeros (untouched slots) and the bare zp baseline."""
    nz = o16[o16 != 0]
    if nz.size == 0:
        return None, 0
    c = Counter(int(v) for v in nz)
    val, cnt = c.most_common(1)[0]
    return val, cnt


def shift_model(P, k, do_round):
    P = int(P)
    if k >= 0:
        v = P << k
    else:
        s = -k
        v = (P + (1 << (s - 1))) >> s if do_round else (P >> s)
    return int(np.clip(ZP + v, 0, 65535))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    print("=== convhbh cvt control-word sweep (uniform-P, 64^3, sim) ===")
    print("model: out = clip(0x8000 + shift(P, k), 0, 65535);  word0 = 0x4040 + k*0x400")
    print(f"{'word0':>7} {'word2':>7} {'k':>3} {'round':>5} {'a':>4} {'b':>4} {'r':>3} "
          f"{'P':>8} {'out_hw':>7} {'model':>7} {'cnt':>5}  verdict")

    # explicit (k, do_round, a, b, r): P = a*b*r everywhere.  a = act-128 (<=127),
    # b = weight value (<=127), r = #weight rows summed.
    cases = [
        # k>=0 (unity/upshift): keep P modest so x4 doesn't clip u16
        (0, False,  16, 1, 2),    # P=32
        (1, False,  16, 1, 2),    # P=32 -> x2
        (2, False,  16, 1, 2),    # P=32 -> x4
        (0, False,  50, 5, 2),    # P=500
        (1, False,  50, 5, 2),    # P=500 -> x2
        # k<0 (downshift): big P (kept so zp+shifted stays in u16), both rounding modes
        (-1, False, 100, 100, 6),  # P=60000 -> >>1 = 30000 (zp+30000=62768 in-range)
        (-1, True,  100, 100, 6),
        (-2, False, 100, 100, 6),  # >>2
        (-2, True,  100, 100, 6),
        (-3, False, 100, 100, 6),  # >>3
        (-4, False, 100, 100, 6),  # >>4
        (-5, False, 100, 100, 6),  # >>5
        (-1, False,  99, 101,  1),  # odd P=9999 -> floor vs round
        (-1, True,   99, 101,  1),
    ]

    npass = 0
    total = 0
    for k, do_round, a, b, r in cases:
        word0 = 0x4040 + k * 0x400
        word2 = 0x0008 if do_round else 0x0000
        if not (0 <= word0 <= 0xFFFF):
            continue
        P, o16, raw = run_uniform(a, b, r, [word0, 0x8040, word2, 0x4000], keep=args.keep)
        total += 1
        if o16 is None:
            print(f"{word0:#7x} {word2:#7x} {k:3d} {str(do_round):>5} {a:4d} {b:4d} {r:3d} "
                  f"{P:8d}   FAULT  ({str(raw)[:40]})")
            continue
        val, cnt = dominant_value(o16)
        model = shift_model(P, k, do_round)
        ok = (val == model)
        npass += ok
        verdict = f"OK shift=2^{k}" + ("(round)" if do_round else "(trunc)") if ok else "MISMATCH"
        print(f"{word0:#7x} {word2:#7x} {k:3d} {str(do_round):>5} {a:4d} {b:4d} {r:3d} "
              f"{P:8d} {str(val):>7} {model:7d} {cnt:5d}  {verdict}")

    print(f"\n{npass}/{total} cases matched out = saturate(P*2^k)+zp.")
    return 0 if npass == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
