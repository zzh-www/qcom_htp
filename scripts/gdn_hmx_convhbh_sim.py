#!/usr/bin/env python3
"""Phase-3 Step-1: make the int16-OUTPUT HMX kernel (convhbh, `cvt.uh:2x2`)
bit-exact on a 64x64x64 matmul in hexagon-sim (NO device, NO SSR risk).

KEY FACT (Agent/current/gdn_solve_NEXT_AGENT.md + disasm @0x2f5200):
  convhbh reads `activation.ub = mxmem(r6,r7)` and `weight.b = mxmem(...):deep`
  -- the INPUT path is byte-IDENTICAL to the proven-bit-exact u8i8 kernel
  (gdn_hmx_matmul_sim.py).  The ONLY difference is the DRAIN:
    u8i8   : cvt.ub = acc(r25)         -> mxmem:cm = cvt   (u8, 1 tile/cvt)
    convhbh: cvt.uh = acc(r25):2x2     -> mxmem    = cvt   (u16, 2x2 dense)
  The drain gain regs are FIXED literals (0x120c/0x2401/0x320c) so the
  output scale is data-independent.

So: reuse u8i8 crouton8 activation packing + u8i8 activation descriptor/offsets
(the known-correct input path), swap in the convhbh kernel + its mask(0x70b)/
extra({1,1025,524}), and DISCOVER the u16 :2x2 output layout + scale empirically
with controlled probes (exactly how u8i8's output depack was cracked).

Reference numpy product the HMX computes:
  P[m,n] = sum_k (act_u8[m,k]-128) * w[k,n]          (== raw_acc + effective,
  effective = -128*sum_k w[k,n], folded into the bias surface)

Modes:
  --mode probe_m : P[m,n]=m  (row-distinct, n-independent) -> discover row map
  --mode probe_n : P[m,n]=n  (col-distinct, m-independent) -> discover col map
  --mode random  : full random, gated against ref via discovered depack
Run probe_m + probe_n first, derive the depack closed-form, then gate random.

Reproduce: source scripts/env.sh && python scripts/gdn_hmx_convhbh_sim.py --mode id_coln

================================ FINDINGS (2026-06-06) =========================
CRACKED (all sim, zero SSR):
* ACTIVATION (the hard part, fully solved): pack u8 act into the Crouton16 row4
  surface HIGH byte -> pack_a16_crouton16_row4_surface(act_u8.astype(u16) << 8).
  The convhbh single-u8 pass reads the HI byte (QNN u16 path = lo+hi 2 passes).
  - act_off uses the PHYSICAL contract: act_tbl[row4*K_t+kt] = block_base, where
    block_base = ((row4&7)*K_t + kt)*512.  NO (row4>>3)*256 m32 offset (kernel adds
    it internally via m_total_minus_step).
  - n_tiles_pow2 must be large enough to iterate all M rows: M_t*4 (=8 at 64^3)
    only resolves 16 of 64 rows; M_t*16 (=32) resolves all 64.  NOTE: M_t*4 is the
    op default and is CORRECT for the real merge size (C=256, M_t=8 -> 32); the
    64^3 probe (M_t=2) hits a small-size scaling edge.
  - VERIFIED full m+n resolution: id_coln=64, id_rowm=64, id_grid=120 distinct.
* GAIN/SCALE: effective bias = -128*sum_w (u8 zp; -32768 is WRONG, cvt handles the
  <<8 scaling).  Output gain set by bias const word0: 0x4040 -> ~x1, 0x4440 -> ~x2.
  out_u16 = 0x8000 + round(P * gain) with per-channel cvt rounding (~+-1 LSB at x1).
* OUTPUT depack: the isolated-2KB-slot map was a RED HERRING (identity-weight
  coincidence; replicas disagree for dense weight).  Real output = contiguous
  out_raw (op L1265: tile(row4,nt) at (row4*4*N+nt*32) u16), permuted by :2x2 +
  crouton; at 64^3 the 32x32 storage tiles overlap the 4-row natural regions ->
  needs the real-size (256) tiling or a non-overlapping placement to depack.
REMAINING: real-size output depack + exact cvt rounding -> full bit-exact gate.
The KERNEL ITSELF computes the int16-out matmul correctly (identity probes).
==============================================================================
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "example" / "handwritten_hmx_matmul"
sys.path.insert(0, str(EXAMPLE))
sys.path.insert(0, str(ROOT / "scripts"))

import gdn_hmx_matmul_sim as base          # build_and_run, pack_act_crouton8, descriptor_tables, c_*_array
from prepare_owned_inputs import (pack_w8_kmajor, pack_native_a16_bias,
                                  pack_a16_crouton16_row4_surface)
from emulate_hmx_conv1x1_params import conv1x1_words

M = K = N = 64

# convhbh-specific control (decoded @0x2f5200; differs from u8i8's 0x700/{1,0})
CONVHBH_MASK = list(conv1x1_words(0x70b, 0, 0, 0, 0x20))
CONVHBH_EXTRA = [1, 1025, 524]

HARNESS = r"""
#include <stdint.h>
#include <stdio.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_mxaccess.h>
#include <h2_cycles.h>
#include "handwritten_hmx_w8a16_kernel.h"
%(ARRAYS)s
#define ACT_BYTES %(ACT_BYTES)du
#define WEIGHT_BYTES %(WEIGHT_BYTES)du
#define BIAS_BYTES %(BIAS_BYTES)du
#define OUT_BYTES %(OUT_BYTES)du
#define ACT_ENTRIES %(ACT_ENTRIES)du
#define OUT_ENTRIES %(OUT_ENTRIES)du
static void cp(uint8_t*d,const uint8_t*s,uint32_t n){for(uint32_t i=0;i<n;++i)d[i]=s[i];}
int main(void){
  unsigned int vb=h2_info(INFO_VTCM_BASE), vs=h2_info(INFO_VTCM_SIZE);
  printf("convhbh 64cubed int16-out sim\n");
  if(vb==0||vs<1024u){h2_thread_stop(1);return 1;}
  h2_mxaccess_state_t mx; h2_mxaccess_unit_init(&mx,CFG_TYPE_VXU0,CFG_SUBTYPE_VXU0,CFG_HMX_CONTEXTS,0x1);
  printf("[Init] HMX acquired (%%d)\n", h2_mxaccess_acquire(&mx));
  uint8_t*basep=(uint8_t*)(uintptr_t)(vb+0x10000u);
  uint8_t*act=basep+0x00000u,*weight=basep+0x20000u,*bias=basep+0x40000u,*out=basep+0x60000u;
  int32_t*act_table=(int32_t*)(basep+0xc0000u),*out_table=(int32_t*)(basep+0xc1000u);
  cp(act,k_activation,ACT_BYTES); cp(weight,k_packed_weight,WEIGHT_BYTES);
  cp(bias,k_folded_bias,BIAS_BYTES); for(uint32_t i=0;i<OUT_BYTES;++i)out[i]=0u;
  uint32_t mask_words[16] __attribute__((aligned(16))); cp((uint8_t*)mask_words,(const uint8_t*)k_mask_control,64u);
  uint32_t extra[3] __attribute__((aligned(16)))={%(E0)du,%(E1)du,%(E2)du};
  for(uint32_t i=0;i<ACT_ENTRIES;++i)act_table[i]=(int32_t)(uintptr_t)(act+k_act_off[i]);
  for(uint32_t i=0;i<OUT_ENTRIES;++i)out_table[i]=(int32_t)(uintptr_t)(out+k_out_off[i]);
  HmW8A16OutDesc od={out_table,%(OTS)du,%(OYS)du,%(NTP)du,%(MTM)d,%(KTB)du};
  HmW8A16ActDesc ad={act_table,%(NAP)du,%(AYS)du};
  unsigned long long t0=h2_get_pcycles();
  hm_w8a16_v75deep_kernel(&od,&ad,weight,bias,(const HmW8A16MaskDesc*)mask_words,extra);
  unsigned long long t1=h2_get_pcycles();
  printf("[CYC]%%llu\n",(unsigned long long)(t1-t0));
  printf("[OUT]"); for(uint32_t i=0;i<OUT_BYTES;++i)printf("%%02x",out[i]); printf("\n[PASS] convhbh returned\n");
  h2_thread_stop(0); return 0;
}
"""


def descriptor():
    """Reuse u8i8 input descriptor (input path byte-identical) + u16 output sizing.

    u8i8 64^3: m_tiles=2,k_tiles=2,n_tiles=2, table_m_groups=1, row_span=64.
      act_off = [kt*M*32 for mt in range(1) for kt in range(2)] = [0, 2048]
      act_desc = {n_act_pairs=2, y_stride=m_tiles*4=8}
    For the u16 :2x2 output we keep the SAME out_desc field set as u8i8 but with
    a u16 surface (itemsize 2).  out_off table layout is the unknown we discover;
    start with the u8i8 crouton layout scaled to u16 (mt*n_tiles+nt)*row_span*32*2.
    """
    m_tiles = M // 32; k_tiles = K // 32; n_tiles = N // 32   # 2,2,2
    row4_groups = m_tiles * 8                                  # 16

    # INPUT activation (convhbh Crouton16 row4 surface, u16-sized 8192B).
    # PHYSICAL contract (op L1254-1264 hmx_w8a16_crouton_row4_physical_ptr):
    #   act_tbl[row4*K_t + kt] = block_table[(row4&7)*K_t + kt]   -- NO m32 offset!
    # The kernel adds the m32 half (+256) INTERNALLY via m_total_minus_step; row4
    # and row4+8 share the same physical block pointer.
    # crouton16 sub-block (phase,kt,m32)=256B; block(phase,kt) base = (phase*K_t+kt)*512.
    ACT_BLOCK_STRIDE = 512
    act_off = []
    for row4 in range(row4_groups):
        bi = (row4 & 7) * k_tiles
        for kt in range(k_tiles):
            act_off.append((bi + kt) * ACT_BLOCK_STRIDE)
    act_desc = dict(n_act_pairs=k_tiles, act_table_y_stride_words=k_tiles)

    # OUTPUT discovery: give each of the (>=32) consumed out-table entries an
    # ISOLATED 2KB slot so sub-tiles can't overlap; observe each independently.
    OUT_SLOT = 2048
    n_out_entries = 64
    out_off = [i * OUT_SLOT for i in range(n_out_entries)]
    # n_tiles_pow2: the op uses M_t*4 (=8) which only resolves 16 of 64 rows in the
    # standalone 64^3 call; sweep shows M_t*16 (=32) resolves all 64 rows (id_rowm=64).
    out_desc = dict(out_table_stride_dwords=n_tiles, out_y_stride_words=n_tiles,
                    n_tiles_pow2=m_tiles * 16, m_total_minus_step=8, k_total_bytes=n_tiles * 32)
    return dict(act_off=act_off, out_off=out_off, out_desc=out_desc, act_desc=act_desc,
                act_surface_bytes=8192, out_buf_bytes=n_out_entries * OUT_SLOT, out_slot=OUT_SLOT)


def make_inputs(mode: str, seed: int):
    """Return (act_u8 [M,K], w_i8 [K,N], P_int32 [M,N]).

    *_uniform modes: act is CONSTANT across all (m,k) so P[m,n]=const*colsum_w[n]
    is independent of however the kernel walks the activation surface -> isolates
    the OUTPUT layout + scale without first knowing the activation layout.
    """
    act = np.full((M, K), 128, dtype=np.int32)
    w = np.zeros((K, N), dtype=np.int32)
    if mode == "uni_coln":
        # act-128 == 1 everywhere; w[0,n]=n -> P[m,n]=n  (col-distinct, m-uniform)
        act[:, :] = 129
        w[0, :] = np.arange(N)
    elif mode == "uni_all1":
        # act-128==1, w==1 -> P[m,n]=K=64 everywhere (coverage + scale check)
        act[:, :] = 129
        w[:, :] = 1
    elif mode == "uni_k1":
        # act-128==1, only k=0 row of w set to n -> P[m,n]=n via single k term
        act[:, :] = 129
        w[0, :] = np.arange(N)
    elif mode.startswith("uni_p"):
        # act-128==1, w[0,:]=pval -> P[m,n]=pval everywhere -> affine P->u16 fit
        act[:, :] = 129
        w[0, :] = int(mode[5:])
    elif mode == "uni_id":
        # uniform act (129) + identity weight -> P=1 everywhere (isolates weight pack)
        act[:, :] = 129
        for n in range(N):
            w[n, n] = 1
    elif mode == "id_coln":
        # identity weight + act[m,n]=128+n -> P[m,n]=n via ACTIVATION (layout-dep)
        for n in range(N):
            w[n, n] = 1
        act[:, :] = 128 + np.arange(N)[None, :]
    elif mode == "id_rowm":
        # identity weight + act[m,n]=128+m -> P[m,n]=m via ACTIVATION (layout-dep)
        for n in range(N):
            w[n, n] = 1
        act[:, :] = 128 + np.arange(M)[:, None]
    elif mode == "id_grid":
        # identity weight + act[m,n]=128+((m*5+n*7)%120) -> unique-ish per (m,n)
        for n in range(N):
            w[n, n] = 1
        act[:, :] = 128 + ((np.arange(M)[:, None] * 5 + np.arange(N)[None, :] * 7) % 120)
    elif mode == "probe_m":
        # P[m,n] = m : act[m,0]=128+m, w[0,:]=1
        for m in range(M):
            act[m, 0] = 128 + m
        w[0, :] = 1
    elif mode == "probe_n":
        # P[m,n] = n : act[*,0]=129, w[0,n]=n
        act[:, 0] = 129
        w[0, :] = np.arange(N)
    elif mode == "probe_const":
        # P[m,n] = 5 everywhere : act[*,0]=133, w[0,:]=1  -> discover scale/zp
        act[:, 0] = 133
        w[0, :] = 1
    else:  # random (small range -> P fits comfortably, no clamp ambiguity)
        rng = np.random.default_rng(seed)
        act = rng.integers(120, 137, size=(M, K)).astype(np.int32)
        w = rng.integers(-4, 5, size=(K, N)).astype(np.int32)
    act_u8 = act.astype(np.uint8)
    w_i8 = w.astype(np.int8)
    P = (act_u8.astype(np.int32) - 128) @ w_i8.astype(np.int32)
    return act_u8, w_i8, P


def build_src(act_u8, w_i8, mode):
    d = descriptor()
    if mode.startswith("uni"):
        # Layout-independent: every activation byte == the (constant) act value,
        # so the kernel reads the same act regardless of how it walks the surface.
        const = int(act_u8.flat[0])
        act_packed = bytes([const]) * d["act_surface_bytes"]
    else:
        # CONFIRMED: the single u8 pass reads the HIGH byte of the Crouton16
        # surface (QNN's u16 path runs lo+hi passes; static u8 uses the hi pass).
        # Pack u8 act values into u16 high byte (lo=0, hi=value) via crouton16 walk.
        surf16 = pack_a16_crouton16_row4_surface(act_u8.astype(np.uint16) << 8)
        act_packed = surf16.astype("<u2").tobytes()   # 8192 bytes, value in high byte
    w_packed = pack_w8_kmajor(w_i8).tobytes()
    bias_packed, _eff = pack_native_a16_bias(8, w_i8)
    bias_bytes = bias_packed.tobytes()
    arrays = "\n".join([
        base.c_u8_array("k_activation", act_packed),
        base.c_u8_array("k_packed_weight", w_packed),
        base.c_u8_array("k_folded_bias", bias_bytes),
        base.c_u32_array("k_act_off", d["act_off"]),
        base.c_u32_array("k_out_off", d["out_off"]),
        base.c_u32_array("k_mask_control", CONVHBH_MASK),
    ])
    od, ad = d["out_desc"], d["act_desc"]
    src = HARNESS % dict(
        ARRAYS=arrays, ACT_BYTES=len(act_packed), WEIGHT_BYTES=len(w_packed),
        BIAS_BYTES=len(bias_bytes), OUT_BYTES=d["out_buf_bytes"], ACT_ENTRIES=len(d["act_off"]),
        OUT_ENTRIES=len(d["out_off"]), E0=CONVHBH_EXTRA[0], E1=CONVHBH_EXTRA[1], E2=CONVHBH_EXTRA[2],
        OTS=od["out_table_stride_dwords"], OYS=od["out_y_stride_words"], NTP=od["n_tiles_pow2"],
        MTM=od["m_total_minus_step"], KTB=od["k_total_bytes"],
        NAP=ad["n_act_pairs"], AYS=ad["act_table_y_stride_words"])
    return src, d


def analyze(o16, P, mode):
    print(f"\n--- analyze mode={mode} ---")
    print(f"u16 out: nonzero {int((o16!=0).sum())}/{o16.size}  min/max {int(o16.min())}/{int(o16.max())}")
    print(f"ref  P : min/max {int(P.min())}/{int(P.max())}  unique {np.unique(P).size}")
    # how many distinct nonzero u16 values, and their multiplicities
    vals, cnts = np.unique(o16[o16 != 0], return_counts=True)
    print(f"distinct nonzero u16: {vals.size}  (showing first 12): "
          + ", ".join(f"{int(v)}x{int(c)}" for v, c in list(zip(vals, cnts))[:12]))
    # if probe_m: expected 64 distinct values for m=1..63 (m=0 ->0). map value->byte offsets.
    if mode in ("probe_m", "probe_n"):
        idx = np.arange(o16.size)
        # show byte offset (==idx*2) of the first occurrence of a few small values
        for target in [1, 2, 3, 4, 63]:
            where = idx[o16 == target]
            print(f"  u16=={target}: count {where.size}  first idx {where[:6].tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="uni_coln",
                    help="uni_coln|uni_all1|uni_k1|uni_p<N>|probe_m|probe_n|random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    act_u8, w_i8, P = make_inputs(args.mode, args.seed)
    src, d = build_src(act_u8, w_i8, args.mode)
    out = base.build_and_run(src, keep=args.keep)
    if out is None:
        print("BUILD/RUN FAIL")
        return 1
    Path("/tmp/convhbh_full.txt").write_text(out)
    cm = re.search(r"\[CYC\](\d+)", out)
    om = re.search(r"\[OUT\]([0-9a-f]+)", out)
    pas = "[PASS]" in out
    print(f"PASS:{pas}  CYC:{cm.group(1) if cm else '??'}  (u8i8 sim ~417, vrmpy 8325)")
    print(f"descriptor: act_off={d['act_off']} out_off={d['out_off']}")
    print(f"  out_desc={d['out_desc']}  act_desc={d['act_desc']}")
    if not om:
        print("NO [OUT] -> faulted; tail:\n" + out[-500:])
        return 1
    o16 = np.frombuffer(bytes.fromhex(om.group(1)), dtype="<u2")
    np.save("/tmp/convhbh_out_u16.npy", o16)
    np.save("/tmp/convhbh_ref_P.npy", P)
    analyze(o16, P, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
