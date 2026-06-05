#!/usr/bin/env python3
"""待办4 sim-first gate: convhbh 64^3 int16-act matmul BIT-EXACT + 2-pass int16-weight.

Uses the PROVEN baremetal 64^3 descriptor (== GDN_BR: ots=2,oys=8,ntp=8,mtm=8,ktb=64;
n_act_pairs=2, act_y_stride=8) and the w8a16 crouton16_row4 act/out tables (32 entries,
((mt&7)*tiles+t)*m_tiles*128*itemsize), with CONTIGUOUS output de-packed by the proven
deblock_a16_crouton16_row4 (verified the exact inverse of the activation packer at 64^3).

Phase 1: nail the int16-activation accumulator contract via a uniform probe.
Phase 2: dense int16-act x int8-weight, bit-exact vs numpy reference.
Phase 3: 2-pass convhbh (weight hi/lo int8 split) -> int16-act x int16-weight bit-exact.

Reproduce: source scripts/env.sh && python scripts/gdn_convhbh_i16_block_sim.py
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "example" / "handwritten_hmx_matmul"))

import gdn_hmx_convhbh_sim as C
import gdn_hmx_matmul_sim as base
from prepare_owned_inputs import pack_w8_kmajor, pack_a16_crouton16_row4_surface
import gdn_merge_layout_directpipe_probe as LP   # deblock_c_harness (proven inverse)

M = K = N = 64
ZP = 0x8000
ITEM = 2  # int16


NTP_FACTOR = 16   # int16 :2x2 drain needs M_t*16 (=32 @64^3) to resolve all 64 rows


def descriptor_64():
    m_tiles = M // 32; k_tiles = K // 32; n_tiles = N // 32   # 2,2,2
    tmg = m_tiles * 8                                          # 16 row4-groups
    # w8a16 crouton16_row4 physical offsets (contiguous surface)
    act_off = [((mt & 7) * k_tiles + kt) * m_tiles * 4 * 32 * ITEM
               for mt in range(tmg) for kt in range(k_tiles)]   # 32 entries, stride 512
    out_off = [((mt & 7) * n_tiles + nt) * m_tiles * 4 * 32 * ITEM
               for mt in range(tmg) for nt in range(n_tiles)]   # 32 entries
    out_desc = dict(out_table_stride_dwords=n_tiles, out_y_stride_words=8,
                    n_tiles_pow2=m_tiles * NTP_FACTOR, m_total_minus_step=8, k_total_bytes=n_tiles * 32)
    act_desc = dict(n_act_pairs=k_tiles, act_table_y_stride_words=8)
    return dict(act_off=act_off, out_off=out_off, out_desc=out_desc, act_desc=act_desc,
                act_surface_bytes=M * K * ITEM, out_buf_bytes=M * N * ITEM)


def build_bias(w_i8, const_words):
    k, n = w_i8.shape
    eff = (-128 * w_i8.astype(np.int32).sum(axis=0)).astype(np.int32)
    pk = np.zeros((n // 32, 512), np.uint8)
    const = np.array(const_words, np.uint16).view(np.uint8)
    for nt in range(n // 32):
        for par in (0, 1):
            hb = par * 256
            for lane, c in enumerate(range(par, 32, 2)):
                col = nt * 32 + c; lb = hb + 8 * lane
                pk[nt, lb:lb + 8] = const
                pk[nt, hb + 128 + 8 * lane:hb + 132 + 8 * lane] = np.array([int(eff[col])], np.int32).view(np.uint8)
    return pk.reshape(-1)


def run(act_stored_u16, w_i8, const_words=None):
    """act_stored_u16: [M,K] uint16 (=int16+32768).  w_i8: [K,N] int8.  Returns out logical [M,N] u16."""
    if const_words is None:
        const_words = C.EXACT_GAIN_CONST
    d = descriptor_64()
    act_packed = pack_a16_crouton16_row4_surface(act_stored_u16.astype(np.uint16)).astype("<u2").tobytes()
    w_packed = pack_w8_kmajor(w_i8).tobytes()
    bias_bytes = build_bias(w_i8, const_words).tobytes()
    arrays = "\n".join([
        base.c_u8_array("k_activation", act_packed), base.c_u8_array("k_packed_weight", w_packed),
        base.c_u8_array("k_folded_bias", bias_bytes), base.c_u32_array("k_act_off", d["act_off"]),
        base.c_u32_array("k_out_off", d["out_off"]), base.c_u32_array("k_mask_control", C.CONVHBH_MASK)])
    od, ad = d["out_desc"], d["act_desc"]
    src = C.HARNESS % dict(
        ARRAYS=arrays, ACT_BYTES=len(act_packed), WEIGHT_BYTES=len(w_packed), BIAS_BYTES=len(bias_bytes),
        OUT_BYTES=d["out_buf_bytes"], ACT_ENTRIES=len(d["act_off"]), OUT_ENTRIES=len(d["out_off"]),
        E0=C.CONVHBH_EXTRA[0], E1=C.CONVHBH_EXTRA[1], E2=C.CONVHBH_EXTRA[2],
        OTS=od["out_table_stride_dwords"], OYS=od["out_y_stride_words"], NTP=od["n_tiles_pow2"],
        MTM=od["m_total_minus_step"], KTB=od["k_total_bytes"],
        NAP=ad["n_act_pairs"], AYS=ad["act_table_y_stride_words"])
    out = base.build_and_run(src, keep=False)
    if out is None:
        return None, None
    om = re.search(r"\[OUT\]([0-9a-f]+)", out)
    if not om:
        return None, out
    surf = np.frombuffer(bytes.fromhex(om.group(1)), dtype="<u2")
    logical = LP.deblock_c_harness(surf, M, N)   # de-crouton to logical [M,N] u16
    return logical.astype(np.int64), out


GAIN_UNITY = [0x6040, 0x8040, 0x0000, 0x4000]   # exp=24 -> out = P_true exactly (int16-act)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", type=int, default=2); args = ap.parse_args()

    if args.phase == 1:
        # DECISIVE test: does the PROVEN body-sim const [0x4440,..] (bit-exact int16-act
        # x int8-wt at 256^3) make a single call track FULL P_true, or high-byte-only?
        BODY_CONST = [0x4440, 0x8040, 0x0008, 0x4000]
        for tag, const in [("EXACT exp16 [0x4040]", C.EXACT_GAIN_CONST), ("BODY-SIM [0x4440]", BODY_CONST)]:
            print(f"=== Phase 1: int16-act contract (uniform), const={tag} ===")
            print(f"{'av':>7} {'b':>4} {'r':>3} {'P_true':>9} {'out-zp':>8}  note")
            for (av, b, r) in [(1, 1, 1), (3, 1, 1), (100, 1, 1), (257, 1, 1), (1000, 1, 1)]:
                act_i16 = np.full((M, K), av, np.int64)
                w = np.zeros((K, N), np.int32); w[:r, :] = b; w_i8 = w.astype(np.int8)
                logical, raw = run((act_i16 + 32768).astype(np.uint16), w_i8, const)
                if logical is None:
                    print(f"{av:7d} FAULT"); continue
                vals, cnts = np.unique(logical, return_counts=True); dom = int(vals[np.argmax(cnts)])
                note = "tracks low byte!" if (av in (1, 3, 257) and (dom - ZP) not in (0,)) else ""
                print(f"{av:7d} {b:4d} {r:3d} {av*b*r:9d} {dom-ZP:8d}  {note}")
        return 0

    # gain sanity: uniform with GAIN_UNITY (exp24) must give out == P_true
    print("=== gain check: uniform + GAIN_UNITY (exp24) should give out==P_true ===")
    for (av, b, r) in [(1, 1, 1), (100, 1, 1), (1000, 1, 1)]:
        act_i16 = np.full((M, K), av, np.int64)
        w = np.zeros((K, N), np.int32); w[:r, :] = b
        logical, _ = run((act_i16 + 32768).astype(np.uint16), w.astype(np.int8), GAIN_UNITY)
        if logical is not None:
            vals, cnts = np.unique(logical, return_counts=True); dom = int(vals[np.argmax(cnts)]) - ZP
            print(f"  av={av} P={av*b*r}: out-zp={dom}  {'OK' if dom == av*b*r else 'GAIN-OFF'}")

    # ---- Phase 2: DENSE int16-act x int8-weight, bit-exact vs numpy (gain=unity) ----
    print(f"=== Phase 2: dense convhbh 64^3 int16-act x int8-wt, bit-exact (ntp=M_t*{NTP_FACTOR}) ===")
    rng = np.random.default_rng(7)
    npass = 0
    for seed in range(3):
        rng = np.random.default_rng(seed)
        act_i16 = rng.integers(-100, 101, (M, K)).astype(np.int64)        # small -> |P| < 32767
        w_i8 = rng.integers(-4, 5, (K, N)).astype(np.int8)
        P_true = (act_i16 @ w_i8.astype(np.int64))
        logical, raw = run((act_i16 + 32768).astype(np.uint16), w_i8, GAIN_UNITY)
        if logical is None:
            print(f"  seed{seed}: FAULT {str(raw)[:50]}"); continue
        got = logical - ZP
        # signed wrap: u16 stored as zp+signed; recover signed
        got = np.where(got > 32767, got - 65536, got)
        diff = np.abs(got - P_true)
        md = int(diff.max()); nmis = int((diff != 0).sum())
        ok = md == 0
        npass += ok
        sorted_match = np.array_equal(np.sort(got.reshape(-1)), np.sort(P_true.reshape(-1)))
        print(f"  seed{seed}: max|diff|={md}  mism={nmis}/{M*N}  Pmax={int(np.abs(P_true).max())}  "
              f"sorted_match={sorted_match} (->layout perm)  {'BIT-EXACT' if ok else 'MISMATCH'}")
    print(f"\nPhase 2: {npass}/3 bit-exact")
    if npass < 3:
        return 1

    # ---- Phase 3: 2-pass convhbh = int16-act x int16-WEIGHT (hi/lo split) bit-exact ----
    print(f"\n=== Phase 3: 2-pass convhbh -> int16-act x int16-weight bit-exact ===")
    npass3 = 0
    # small act so each per-pass P (act@hi, act@lo) stays in int16 (read at unity gain);
    # device keeps int32 across passes, here we read each drained int16 and reconstruct.
    for seed in range(3):
        rng = np.random.default_rng(100 + seed)
        act_i16 = rng.integers(-3, 4, (M, K)).astype(np.int64)
        w_i16 = rng.integers(-32512, 32513, (K, N)).astype(np.int64)      # int16 weight, qmax 32512
        P_true = act_i16 @ w_i16                                          # int16 x int16
        hi = (w_i16 + 128) >> 8                                           # int8 each
        lo = w_i16 - hi * 256
        assert (np.abs(hi) <= 127).all() and (np.abs(lo) <= 127).all(), "hi/lo not int8"
        assert np.array_equal(256 * hi + lo, w_i16)
        stored = (act_i16 + 32768).astype(np.uint16)
        # gain so that the per-pass P fits int16 then we combine in numpy (the int32
        # combine 256*P_hi+P_lo is what the kernel/HVX would do; here we read each P exactly).
        # Per-pass P can exceed int16 (|act*hi*K| up to 90*127*64=731520) -> need a downscale
        # gain, but to PROVE correctness read each pass at a gain that keeps it exact... instead
        # validate the math: run both passes at unity-ish and reconstruct.  Use a gain that
        # keeps per-pass P in range: |P_hi|,|P_lo| <= 90*127*64=731520 > 32767 -> must downscale.
        # So validate at SMALL weight to keep per-pass in range AND show reconstruction.
        # (full-range handled by the int32 accumulator on device; here prove the algebra+kernel.)
        Phi, _ = run(stored, hi.astype(np.int8), GAIN_UNITY)
        Plo, _ = run(stored, lo.astype(np.int8), GAIN_UNITY)
        if Phi is None or Plo is None:
            print(f"  seed{seed}: FAULT"); continue
        def sgn(x): x = x - ZP; return np.where(x > 32767, x - 65536, x)
        Phi_s, Plo_s = sgn(Phi), sgn(Plo)
        P_recon = 256 * Phi_s + Plo_s
        diff = np.abs(P_recon - P_true); md = int(diff.max())
        ok = md == 0
        npass3 += ok
        print(f"  seed{seed}: max|diff|={md}  Pmax={int(np.abs(P_true).max())}  "
              f"per-pass max={int(max(np.abs(Phi_s).max(), np.abs(Plo_s).max()))}  "
              f"{'BIT-EXACT' if ok else 'MISMATCH'}")
    print(f"\nPhase 3: {npass3}/3 bit-exact (2-pass int16-weight)")
    return 0 if npass3 == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
