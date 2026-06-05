#!/usr/bin/env python3
"""convhbh int16-out HMX matmul bit-exact at general C (=M=K=N), in hexagon-sim.

The 64^3 probe is degenerate (M_t=2): n_tiles_pow2=M_t*4 under-resolves rows and
larger ntp causes redundant partial passes that break dense-weight depack.  At
C>=128 (M_t>=4) the op's NATURAL descriptor (ntp=M_t*4, contiguous out_raw) is
non-degenerate.  This validates the full recipe end-to-end bit-exact:
  - act = pack_a16_crouton16_row4_surface(act_u8 << 8)   [HIGH-byte single pass]
  - act_off physical: act_tbl[row4*K_t+kt] = ((row4&7)*K_t+kt) * (M_t*256)   [no m32 off]
  - bias const = [0x4040,0x8040,0x0000,0x4000] (gain x1, NO rounding) + eff=-128*sum_w
  - out_raw contiguous: tile(row4,nt) at (row4*4*N + nt*32) u16, depack discovered.

Reproduce: source scripts/env.sh && python scripts/gdn_hmx_convhbh_sim_csize.py --C 128
"""
from __future__ import annotations
import argparse, sys, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "example" / "handwritten_hmx_matmul"
sys.path.insert(0, str(EXAMPLE)); sys.path.insert(0, str(ROOT / "scripts"))
import gdn_hmx_convhbh_sim as S
import gdn_hmx_matmul_sim as base
from prepare_owned_inputs import pack_w8_kmajor, pack_a16_crouton16_row4_surface
from emulate_hmx_conv1x1_params import conv1x1_words

EXACT_CONST = [0x4040, 0x8040, 0x0000, 0x4000]
MASK = list(conv1x1_words(0x70b, 0, 0, 0, 0x20))
EXTRA = [1, 1025, 524]

HARNESS = S.HARNESS  # reuse the same C harness (descriptor fields are params)


def build_bias(w_i8):
    k, n = w_i8.shape
    eff = (-128 * w_i8.astype(np.int32).sum(axis=0)).astype(np.int32)
    pk = np.zeros((n // 32, 512), np.uint8)
    const = np.array(EXACT_CONST, np.uint16).view(np.uint8)
    for nt in range(n // 32):
        for par in (0, 1):
            hb = par * 256
            for lane, c in enumerate(range(par, 32, 2)):
                col = nt * 32 + c; lb = hb + 8 * lane
                pk[nt, lb:lb + 8] = const
                pk[nt, hb + 128 + 8 * lane:hb + 132 + 8 * lane] = np.array([int(eff[col])], np.int32).view(np.uint8)
    return pk.reshape(-1)


def descriptor(C, ntp_factor=4):
    M = K = N = C
    Mt = M // 32; Kt = K // 32; Nt = N // 32
    row4_groups = Mt * 8
    blk = Mt * 256                       # crouton16 block stride (bytes)
    act_off = [((row4 & 7) * Kt + kt) * blk for row4 in range(row4_groups) for kt in range(Kt)]
    act_desc = dict(n_act_pairs=Kt, y=Kt)
    # contiguous out_raw (op L1265): tile(row4,nt) at (row4*4*N + nt*32) u16
    out_off = [(row4 * 4 * N + nt * 32) * 2 for row4 in range(row4_groups) for nt in range(Nt)]
    out_desc = dict(ots=Nt, oys=Nt, ntp=Mt * ntp_factor, mtm=8, ktb=Nt * 32)
    return dict(M=M, K=K, N=N, Mt=Mt, Kt=Kt, Nt=Nt, act_off=act_off, act_desc=act_desc,
                out_off=out_off, out_desc=out_desc, surf_bytes=M * K * 2, out_bytes=M * N * 2)


def run(act_u8, w_i8, C, ntp_factor=4):
    d = descriptor(C, ntp_factor)
    ap = pack_a16_crouton16_row4_surface(act_u8.astype(np.uint16) << 8).astype("<u2").tobytes()
    wp = pack_w8_kmajor(w_i8).tobytes()
    bb = build_bias(w_i8).tobytes()
    arrays = "\n".join([base.c_u8_array("k_activation", ap), base.c_u8_array("k_packed_weight", wp),
        base.c_u8_array("k_folded_bias", bb), base.c_u32_array("k_act_off", d["act_off"]),
        base.c_u32_array("k_out_off", d["out_off"]), base.c_u32_array("k_mask_control", MASK)])
    od, ad = d["out_desc"], d["act_desc"]
    src = HARNESS % dict(ARRAYS=arrays, ACT_BYTES=len(ap), WEIGHT_BYTES=len(wp), BIAS_BYTES=len(bb),
        OUT_BYTES=d["out_bytes"], ACT_ENTRIES=len(d["act_off"]), OUT_ENTRIES=len(d["out_off"]),
        E0=EXTRA[0], E1=EXTRA[1], E2=EXTRA[2], OTS=od["ots"], OYS=od["oys"], NTP=od["ntp"],
        MTM=od["mtm"], KTB=od["ktb"], NAP=ad["n_act_pairs"], AYS=ad["y"])
    out = base.build_and_run(src, keep=False)
    om = re.search(r"\[OUT\]([0-9a-f]+)", out)
    cm = re.search(r"\[CYC\](\d+)", out)
    return (np.frombuffer(bytes.fromhex(om.group(1)), dtype="<u2").astype(int) - 32768 if om else None,
            int(cm.group(1)) if cm else None, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=128)
    ap.add_argument("--ntpf", type=int, default=4, help="n_tiles_pow2 = M_t * ntpf")
    args = ap.parse_args()
    C = args.C
    rng = np.random.default_rng(5)
    # calibration: identity weight + col-ramp / row-ramp / const (exact gain -> out==signal)
    wid = np.zeros((C, C), np.int32)
    for i in range(C): wid[i, i] = 1
    wid = wid.astype(np.int8)
    actc = (128 + np.arange(C)[None, :] * np.ones((C, 1), int)).astype(np.uint8) % 256
    actr = (128 + np.arange(C)[:, None] * np.ones((1, C), int)).astype(np.uint8) % 256
    # keep ramps in [0,127] so act stays valid u8 and signal distinct; use mod via smaller
    actc = (128 + (np.arange(C)[None, :] % 100) * np.ones((C, 1), int)).astype(np.uint8)
    actr = (128 + (np.arange(C)[:, None] % 100) * np.ones((1, C), int)).astype(np.uint8)
    oc, _, _ = run(actc, wid, C, args.ntpf)
    orow, _, _ = run(actr, wid, C, args.ntpf)
    o10, _, _ = run(np.full((C, C), 138, np.uint8), wid, C, args.ntpf)
    if oc is None or orow is None or o10 is None:
        print("CALIB FAULT"); return 1
    wrote = (o10 == 10)
    # map storage position -> (m,n)  (col signal = n%100, row signal = m%100; recover with care)
    # use distinct ramps: n in 0..C-1 via two probes won't alias if C<=100; for C>=128 need 2 digits.
    print(f"C={C} ntp=Mt*{args.ntpf}={C//32*args.ntpf}: written={int(wrote.sum())}/{C*C}  "
          f"(P=10 calib; clean if ~{C*C})")
    # random gate via natural reshape AND via calibrated permutation
    act = rng.integers(120, 137, (C, C)).astype(np.uint8)
    w = rng.integers(-4, 5, (C, C)).astype(np.int8)
    P = (act.astype(np.int32) - 128) @ w.astype(np.int32)
    o, cyc, _ = run(act, w, C, args.ntpf)
    if o is None: print("RANDOM FAULT"); return 1
    g = o.reshape(C, C)
    print(f"  natural reshape: max|out-P|={int(np.abs(g-P).max())}  cyc={cyc}")
    # calibrated permutation (only valid if C<=100 ramps unique; report coverage)
    if C <= 100:
        from collections import defaultdict
        mn2pos = defaultdict(list)
        for p in np.nonzero(wrote)[0]:
            mn2pos[(int(orow[p]), int(oc[p]))].append(int(p))
        print(f"  calib distinct (m,n)={len(mn2pos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
