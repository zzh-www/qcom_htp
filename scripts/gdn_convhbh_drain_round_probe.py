#!/usr/bin/env python3
"""Step-2 待办1b: nail the EXACT cvt-drain rounding rule for convhbh downshift.

The exponent sweep proved gain = 2^(exp-16), but k=-1/-2 carried a +1/+2 excess at
large P.  Here we probe SMALL P at k=-1 (>>1), where the f16-representable region is
exact, so any deviation from pure floor reveals the true drain rounding rule
(floor vs round-half-up vs round-half-even vs +bias).

Uniform-P probe (zero depack): act-128=a, w[:r,:]=b => P=a*b*r everywhere; read the
dominant drained u16 = clip(zp + drain(P, k), 0, 65535).

Reproduce: source scripts/env.sh && python scripts/gdn_convhbh_drain_round_probe.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gdn_convhbh_control_word_sweep as S

ZP = 0x8000


def realize_P(P):
    """pick (a,b,r) with a,b<=127 s.t. a*b*r == P (r small)."""
    for a in range(min(P, 127), 0, -1):
        if P % a == 0:
            q = P // a
            for b in range(min(q, 127), 0, -1):
                if q % b == 0:
                    return a, b, q // b
    return None


def main():
    # k=-1 (>>1), word2=0 (no round flag).  Probe small P incl. odd to see floor vs round.
    k = -1
    word0 = 0x4040 + k * 0x400
    probes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 30, 31, 100, 101, 254, 255, 1000, 1001]
    print(f"=== convhbh drain rounding probe: k={k} (>>1), word0={word0:#x}, word2=0 ===")
    print(f"{'P':>6} {'a,b,r':>10} {'out_hw-zp':>10} {'floor':>6} {'roundHU':>8} {'rule'}")
    rules = []
    for P in probes:
        abr = realize_P(P)
        if abr is None:
            print(f"{P:6d}  (unrealizable)"); continue
        a, b, r = abr
        Pc, o16, raw = S.run_uniform(a, b, r, [word0, 0x8040, 0x0000, 0x4000])
        if o16 is None:
            print(f"{P:6d} {str(abr):>10}  FAULT"); continue
        val, cnt = S.dominant_value(o16)
        d = val - ZP if val is not None else None
        fl = P >> 1
        rh = (P + 1) >> 1
        rule = "floor" if d == fl else ("roundHU" if d == rh else f"other(+{d-fl})")
        rules.append((P, d, fl, rh, rule))
        print(f"{P:6d} {str((a,b,r)):>10} {str(d):>10} {fl:6d} {rh:8d} {rule}")

    # summary
    floors = sum(1 for _, d, fl, _, _ in rules if d == fl)
    rounds = sum(1 for _, d, _, rh, _ in rules if d == rh)
    print(f"\nfloor matches {floors}/{len(rules)}, roundHU matches {rounds}/{len(rules)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
