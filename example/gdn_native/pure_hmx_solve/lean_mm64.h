/* lean_mm64.h — LEAN bit-exact w16a16 consumer kernel (Stage B, cron#82).
 *
 * A from-scratch, flat-loop replica of the ALIGNED MAIN PATH of QNN
 * `hmx_v73_convhhh1x1_stride1` for the EXACT GP_CROUTON8 solve descriptor
 * (M=64,K=64,N=64, n_tiles=8, n_act_pairs=2, out_table_stride=2, out_y=4,
 *  m_total=8, k_total=64; atab[i]=otab[i]=(i&3)*2048; weight=pack_wt_kmajor,
 *  bias=pack_bias). The goal: strip the convhhh bias-staircase + M-loop bloat
 *  (~1213 cyc/conv of non-MAC packet stall) down to the bare 8-tile MAC walk.
 *
 * It REPLACES our_v73deep_kernel_i16 as the PRODUCTION DEFAULT (cron#83);
 * build with -DGP_NO_LEANMM to escape to native. The output is byte-identical
 * to native. Bit-exactness is self-checked under -DGP_LEANCHK (max|d| vs the
 * native kernel must be 0) and live per-conv under -DGP_LEANCHK_LIVE.
 *
 * Register-traced walk (analyzer, byte-verified vs .inc main path L28-47):
 *   8 output tiles, each = 2 dilate MACs (hi-wt @0x0, lo-wt @0x400 into ONE acc)
 *   + bias=mxmem2(group) + cvt.uh=acc:2x2 + crouton16 store.
 *   tile -> (act_hi, act_lo, out_addr, bias_group):
 *     0: atab0,atab1 | otab0+0x00  | g0
 *     1: atab2,atab3 | otab2+0x00  | g0
 *     2: atab0,atab1 | otab0+0x40  | g1
 *     3: atab2,atab3 | otab2+0x40  | g1
 *     4: atab0,atab1 | otab1+0x00  | g2
 *     5: atab2,atab3 | otab3+0x00  | g2
 *     6: atab0,atab1 | otab1+0x40  | g3
 *     7: atab2,atab3 | otab3+0x40  | g3
 *   ART(act r24)=0x77c  WRT(wt:dilate r25)=0x3ff  CRT(cvt ctrl r31=ep[1])=1536(0x600)
 *   OUTM(out r11=mb[1])=0x700
 */
#ifndef LEAN_MM64_H
#define LEAN_MM64_H
#include <stdint.h>
#include "w16a16_mm.h"

#if defined(__hexagon__)

/* one 64^3 w16a16 matmul on already-packed VTCM operands (same ABI consumer as w16a16_mm_run). */
__attribute__((always_inline))
static inline void lean_mm64(w16a16_mm_t *b) {
    /* lean only models the GP_CROUTON8 n_tiles=8 dense 64^3 shape. The diagnostic micro-benches
     * (NTSWEEP/M-fanout/M=256 carrier) call w16a16_mm_run with OTHER descriptors — fall back to the
     * native kernel for those so they stay valid (they don't gate the Stage-B verdict). */
    {
        const hmx_conv_out_desc_t *od = (const hmx_conv_out_desc_t *)b->od;
        const hmx_conv_act_desc_t *ad = (const hmx_conv_act_desc_t *)b->ad;
        if (od->n_tiles_pow2 != 8u || od->m_total_minus_step != 8 || od->out_y_stride_words != 4u ||
            od->k_total_bytes != 64u || ad->n_act_pairs != 2u) {   /* k_total guard: future "nt8-shaped but K!=64" -> native */
            our_v73deep_kernel_i16(od, ad, b->wt, (const uint8_t *)b->bias,
                                   (const hmx_conv_mask_desc_t *)b->mb, b->ep);
            return;
        }
    }
    const uint32_t ART = 0x77cu, WRT = 0x3ffu, CRT = 0x600u, OUTM = 0x700u;
    uint8_t *wt   = b->wt;
    uint8_t *bias = (uint8_t *)b->bias;
    /* act/out absolute VTCM tile pointers (atab/otab hold base+(i&3)*2048). */
    uint8_t *a0 = (uint8_t *)(uintptr_t)b->atab[0];
    uint8_t *a1 = (uint8_t *)(uintptr_t)b->atab[1];
    uint8_t *a2 = (uint8_t *)(uintptr_t)b->atab[2];
    uint8_t *a3 = (uint8_t *)(uintptr_t)b->atab[3];
    uint8_t *o0 = (uint8_t *)(uintptr_t)b->otab[0];
    uint8_t *o1 = (uint8_t *)(uintptr_t)b->otab[1];
    uint8_t *o2 = (uint8_t *)(uintptr_t)b->otab[2];
    uint8_t *o3 = (uint8_t *)(uintptr_t)b->otab[3];

    /* tile descriptor table: {act_hi, act_lo, out_addr, bias_group_byte_off, wt_hi_off, wt_lo_off}.
     * cron#82 fix: wt base advances 0x800 per outer (nt,half) pair — NOT constant 0x0/0x400 (analyzer:
     * r2=r27=r8_after_MACs, +0x800/outer; the 4 (nt,half) pairs fill the 0x2000 wt buffer exactly). */
    struct { uint8_t *ah, *al, *out; uint32_t bg, wh, wl; } T[8] = {
        { a0, a1, o0 + 0x00, 0x000u, 0x0000u, 0x0400u },
        { a2, a3, o2 + 0x00, 0x000u, 0x0000u, 0x0400u },
        { a0, a1, o0 + 0x40, 0x100u, 0x0800u, 0x0C00u },
        { a2, a3, o2 + 0x40, 0x100u, 0x0800u, 0x0C00u },
        { a0, a1, o1 + 0x00, 0x200u, 0x1000u, 0x1400u },
        { a2, a3, o3 + 0x00, 0x200u, 0x1000u, 0x1400u },
        { a0, a1, o1 + 0x40, 0x300u, 0x1800u, 0x1C00u },
        { a2, a3, o3 + 0x40, 0x300u, 0x1800u, 0x1C00u },
    };

    for (int t = 0; t < 8; ++t) {
        uint8_t *whi = wt + T[t].wh, *wlo = wt + T[t].wl;
        uint8_t *bp  = bias + T[t].bg;
        uint8_t *op  = T[t].out;
        asm volatile(
            "mxclracc\n"
            "{ activation.ub = mxmem(%0,%4); weight.b = mxmem(%2,%5):dilate }\n"
            "{ activation.ub = mxmem(%1,%4); weight.b = mxmem(%3,%5):dilate }\n"
            "bias = mxmem2(%6)\n"
            "cvt.uh = acc(%7):2x2\n"
            "mxmem(%8,%9):2x2 = cvt\n"
            :
            : "r"(T[t].ah), "r"(T[t].al), "r"(whi), "r"(wlo),
              "r"(ART), "r"(WRT), "r"(bp), "r"(CRT), "r"(op), "r"(OUTM)
            : "memory");
    }
}

#endif /* __hexagon__ */
#endif /* LEAN_MM64_H */
