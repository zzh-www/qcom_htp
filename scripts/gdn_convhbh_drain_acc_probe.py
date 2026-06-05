#!/usr/bin/env python3
"""Step-2 待办2 (precision premise): does the cvt-drain f16 precision loss depend on
the NET accumulator (post effective-bias) or the RAW accumulator (pre-bias)?

待办1 found: small acc => exact round-half-up; raw_acc≈137000 => +2 LSB.  But that
case had BOTH a large net acc (60000) AND a large raw_acc (136800) AND a large bias
(-76800).  This probe disambiguates by holding the NET acc constant (= what the merge
drains, controllable via output scale) while varying the RAW acc / bias magnitude.

Construction (uniform-P, zero depack): act-128 = a, w[:r,:] = b  =>
  raw_acc = (128+a)*b*r,  sum_w = b*r,  eff = -128*sum_w,  net = a*b*r = P.
To hold net P constant while growing raw_acc: SHRINK a, GROW (b*r).
  e.g. net=1000:  a=100,b=10,r=1 (raw=22800)  vs  a=1,b=100,r=10 (raw=129000).

If drain error tracks RAW acc -> must minimize zero-point bias (use signed zp=0
operands) for fused accumulation.  If it tracks NET acc -> fused accumulation is safe
as long as the final drained int16 is kept small (design already does this via k).

Reproduce: source scripts/env.sh && python scripts/gdn_convhbh_drain_acc_probe.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gdn_convhbh_control_word_sweep as S

ZP = 0x8000


def run(a, b, r, k, do_round=False):
    word0 = 0x4040 + k * 0x400
    word2 = 0x0008 if do_round else 0x0000
    P, o16, raw = S.run_uniform(a, b, r, [word0, 0x8040, word2, 0x4000])
    if o16 is None:
        return None
    val, cnt = S.dominant_value(o16)
    raw_acc = (128 + a) * b * r
    sum_w = b * r
    eff = -128 * sum_w
    s = -k
    model = (P + (1 << (s - 1))) >> s if k < 0 else P << k
    hw = (val - ZP) if val is not None else None
    return dict(a=a, b=b, r=r, P=P, raw_acc=raw_acc, eff=eff, k=k,
                hw=hw, model=int(model), err=(hw - int(model)) if hw is not None else None)


def main():
    print("=== fused-accumulation drain precision: NET vs RAW accumulator ===")
    print("hold NET acc P constant, vary RAW acc via zero-point bias (shrink a, grow b*r)")
    print(f"{'a':>4} {'b':>4} {'r':>3} {'P(net)':>8} {'raw_acc':>9} {'eff(bias)':>10} "
          f"{'k':>3} {'hw':>7} {'model':>7} {'err':>4}")

    # families: each holds net P roughly constant, raw_acc grows left->right.
    families = {
        "net=1000, k=-1 (>>1, target 500)": [
            (100, 10, 1, -1),   # raw 22800
            (50, 20, 1, -1),    # raw 35600
            (20, 50, 1, -1),    # raw 74000
            (10, 100, 1, -1),   # raw 138000
            (5, 100, 2, -1),    # raw 266000
            (2, 100, 5, -1),    # raw 665000
            (1, 100, 10, -1),   # raw 1290000
        ],
        "net=30000, k=-1 (>>1, target 15000)": [
            (100, 100, 3, -1),  # raw 68400,  net 30000
            (60, 100, 5, -1),   # raw 94000,  net 30000
            (30, 100, 10, -1),  # raw 158000, net 30000
        ],
        "net=1000, k=-4 (>>4, target ~62)": [
            (100, 10, 1, -4),
            (10, 100, 1, -4),
            (1, 100, 10, -4),
        ],
    }

    bad = 0
    total = 0
    for fam, cases in families.items():
        print(f"\n-- {fam} --")
        for a, b, r, k in cases:
            res = run(a, b, r, k)
            total += 1
            if res is None:
                print(f"{a:4d} {b:4d} {r:3d}  FAULT"); bad += 1; continue
            flag = "" if res["err"] == 0 else f"  <-- err {res['err']:+d}"
            if res["err"] != 0:
                bad += 1
            print(f"{res['a']:4d} {res['b']:4d} {res['r']:3d} {res['P']:8d} {res['raw_acc']:9d} "
                  f"{res['eff']:10d} {res['k']:3d} {str(res['hw']):>7} {res['model']:7d} "
                  f"{str(res['err']):>4}{flag}")

    print(f"\n{total-bad}/{total} exact (err==0).  If error appears as raw_acc grows while "
          f"net P is fixed -> RAW-acc-driven (minimize zp bias).  If error stays 0 -> NET-driven (safe).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
