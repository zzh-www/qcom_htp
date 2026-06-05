#!/usr/bin/env python3
"""De-risk phase-3: run the int16-OUTPUT HMX kernel (convhbh, cvt.uh:2x2) on a
64x64x64 matmul in hexagon-sim (NO device, NO SSR risk), measure cyc, and work
toward bit-exact u16 output.

Kernel = the byte-verified `hm_w8a16_v75deep_kernel` (== libQnnHtpV75Skel
hmx_v75_convhbh1x1deep_stride1 @0x2f5200).  Operands packed with the existing
handwritten w8a16 packers.  Descriptor = the decoded w8a16 64^3 tuple
(row4_groups=16, 32-entry act/out tables, mask conv1x1_words(0x70b,0,0,0,0x20),
extra={1,1025,524}).

Goal 1 (cyc): a non-faulting run gives the int16-output matmul cyc (data-
independent) -> answers "is int16-out matmul ~= int8-out matmul cost".
Goal 2 (correctness): de-pack u16 output, compare to numpy ref.

Reproduce: source scripts/env.sh && python scripts/gdn_hmx_convhbh_sim.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "example" / "handwritten_hmx_matmul"
sys.path.insert(0, str(EXAMPLE))
sys.path.insert(0, str(ROOT / "scripts"))
import gdn_hmx_matmul_sim as base               # reuse build_and_run, tool, parse_out, c_* arrays
from prepare_owned_inputs import (pack_w8_kmajor, pack_a16_crouton16_row4_surface,
                                  pack_native_a16_bias)
from emulate_hmx_conv1x1_params import conv1x1_words

M = K = N = 64
KT = K // 32; NT = N // 32; MT = M // 32      # 2,2,2
ROW4_GROUPS = MT * 8                            # 16

def c_u8(name, b):  return base.c_u8_array(name, bytes(b))
def c_u32(name, v): return base.c_u32_array(name, list(v))

# ---- descriptor (decoded w8a16 64^3) ----
def descriptor():
    act_stride = KT                             # =2
    out_stride = NT                             # =2
    # act table: 32 entries, entry[row4*act_stride+kt] -> act tile (row4_phase, kt)
    # crouton16 surface tile(row4_phase,kt) offset = (row4_phase*KT+kt) * tile_bytes,
    # tile_bytes = MT*2(row_pair)*32(col)*2(u16 pair) = 512
    tile_bytes = MT * 2 * 32 * 2
    act_off = []
    for row4 in range(ROW4_GROUPS):
        ph = row4 & 7
        hi = (row4 >> 3)                         # second half adds one m32 row-block
        for kt in range(act_stride):
            act_off.append(((ph * KT + kt) * tile_bytes + hi * 256))
    # out table: 32 entries, out tile ptr = out_raw + row4*4*n_cols + nt*32 (u16 -> *2 bytes), n_cols=N
    out_off = []
    for row4 in range(ROW4_GROUPS):
        for nt in range(out_stride):
            out_off.append((row4 * 4 * N + nt * 32) * 2)
    out_desc = dict(out_table_stride_dwords=out_stride, out_y_stride_words=out_stride,
                    n_tiles_pow2=MT * 4, m_total_minus_step=8, k_total_bytes=NT * 32)
    act_desc = dict(n_act_pairs=KT, act_table_y_stride_words=KT)
    return dict(act_off=act_off, out_off=out_off, out_desc=out_desc, act_desc=act_desc,
                mask=list(conv1x1_words(0x70b, 0, 0, 0, 0x20)), extra=[1, 1025, 524])

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

def main():
    rng = np.random.default_rng(0)
    # u8 act (zp128-ish small), i8 weight, small so int32 acc + u16 out are in range.
    act = rng.integers(120, 136, size=(M, K), dtype=np.int32).astype(np.uint16)   # u16 holder for crouton16 packer
    w   = rng.integers(-4, 5, size=(K, N), dtype=np.int32).astype(np.int8)
    act_packed = pack_a16_crouton16_row4_surface(act).astype(np.uint16).tobytes()
    w_packed   = pack_w8_kmajor(w).tobytes()
    bias_packed, eff = pack_native_a16_bias(8, w)
    d = descriptor()
    arrays = "\n".join([
        c_u8("k_activation", act_packed), c_u8("k_packed_weight", w_packed),
        c_u8("k_folded_bias", bytes(bias_packed.tobytes())),
        c_u32("k_act_off", d["act_off"]), c_u32("k_out_off", d["out_off"]),
        c_u32("k_mask_control", d["mask"]),
    ])
    od, ad = d["out_desc"], d["act_desc"]
    src = HARNESS % dict(ARRAYS=arrays, ACT_BYTES=len(act_packed), WEIGHT_BYTES=len(w_packed),
        BIAS_BYTES=len(bias_packed.tobytes()), OUT_BYTES=N*M*2, ACT_ENTRIES=len(d["act_off"]),
        OUT_ENTRIES=len(d["out_off"]), E0=d["extra"][0], E1=d["extra"][1], E2=d["extra"][2],
        OTS=od["out_table_stride_dwords"], OYS=od["out_y_stride_words"], NTP=od["n_tiles_pow2"],
        MTM=od["m_total_minus_step"], KTB=od["k_total_bytes"], NAP=ad["n_act_pairs"], AYS=ad["act_table_y_stride_words"])
    out = base.build_and_run(src, keep=False)
    if out is None: print("BUILD FAIL"); return
    Path("/tmp/convhbh_full.txt").write_text(out)
    import re
    m = re.search(r"\[CYC\](\d+)", out)
    pas = "[PASS]" in out
    om = re.search(r"\[OUT\]([0-9a-f]+)", out)
    print("PASS:", pas, " CYC:", m.group(1) if m else "??", " (u8i8 floor ~214, vrmpy 8325)")
    if om:
        ob = bytes.fromhex(om.group(1)); o16 = np.frombuffer(ob, dtype="<u2")
        print("OUT bytes:", len(ob), " nonzero u16:", int((o16 != 0).sum()), "/", o16.size,
              " min/max:", int(o16.min()), int(o16.max()))
        # numpy ref: P[m,n] = sum_k (act_u8[m,k]-128)*w[k,n]  (the int product the HMX computes)
        P = ((act.astype(np.int32) - 128) @ w.astype(np.int32))
        print("ref P int32 min/max:", int(P.min()), int(P.max()))
        np.save("/tmp/convhbh_out_u16.npy", o16); np.save("/tmp/convhbh_ref_P.npy", P)
    else:
        print("NO [OUT] -> kernel likely faulted before output dump; see /tmp/convhbh_full.txt tail")
        print(out[-400:])

if __name__ == "__main__":
    main()
