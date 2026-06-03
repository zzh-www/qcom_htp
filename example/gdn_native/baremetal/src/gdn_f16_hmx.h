/* gdn_f16_hmx.h — f16 HMX building blocks for the GDN block-recursive triangular inverse.
 *
 * THE PATH THAT WAS NEVER TRIED (see reference_hmx_f16_vtcm_resident_path): instead of the u8xi8
 * v73deep kernel (8-bit accuracy cliff + per-merge quant glue) this uses native f16 HMX
 * (qf32-precision accumulate) with all blocks VTCM-resident in 2-row tile format.  On v75 f16 the
 * activation (AH) and weight (WH) tile formats are IDENTICAL, so a merge's output (written by
 * hmx_store_acc as an AH tile) is directly reusable as the next merge's input -> ZERO conversion
 * between the 16 64x64x64 merges.  Pure HMX compute is ~free when VTCM-resident (~28k GFLOPS);
 * the only real cost is the AH<->RM conversion at the solve boundaries + the diagonal forward-subst.
 *
 * Ported from docs/hexagon-tutorial/hmx-tutorial/ch05-hmx/src/exp5_standalone_asm.c (htp-ops-lib path).
 * Blocks are fixed 64x64 -> 2x2 = 4 tiles per block, K=2 tiles per output tile (one :deep load of 2). */
#ifndef GDN_F16_HMX_H
#define GDN_F16_HMX_H
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>
#include <stdint.h>

#define HMX_TILE      32
#define HMX_TILE_B    2048u            /* 32*32 f16 = 2048 bytes per tile */
#define HMX_F16_ONE   0x3C00           /* f16 1.0 */
#define BL            64               /* GDN block size */
#define BL_T          (BL/HMX_TILE)    /* 2 tiles per block side */
#define BL_NTILES     (BL_T*BL_T)      /* 4 tiles per 64x64 block */

/* ---- the 5 HMX ASM ops (f16, no hexkl) ---- */
static inline __attribute__((always_inline)) void hmx_clear_acc(void) {
    asm volatile("mxclracc.hf" ::: "memory");
}
static inline __attribute__((always_inline)) void hmx_set_scales(const void *scales) {
    asm volatile("bias = mxmem2(%0)" :: "r"(scales) : "memory");
}
/* load n_tiles activation+weight tile pairs and MAC them into the accumulator (:deep). */
static inline __attribute__((always_inline)) void hmx_load_tiles(
        const void *act, const void *wt, uint32_t n_tiles) {
    uint32_t limit = n_tiles * HMX_TILE_B - 1;
    asm volatile(
        "{ activation.hf = mxmem(%0, %1):deep\n"
        "  weight.hf = mxmem(%2, %3) }\n"
        :: "r"(act), "r"(limit), "r"(wt), "r"(limit) : "memory");
}
/* convert accumulator -> f16 and store as an AH tile to VTCM (arg 2 = f16 output format). */
static inline __attribute__((always_inline)) void hmx_store_acc(_Float16 *out) {
    asm volatile("cvt.hf = acc(%0)\n  mxmem(%1, %2) = cvt\n"
                 :: "r"(2), "r"(out), "r"(0) : "memory");
}

/* write scale=1.0 / bias=0 into a 256-byte VTCM region; call hmx_set_scales(it) once after lock. */
static inline void hmx_init_scales(_Float16 *scales /*>=128 f16, 128B aligned*/) {
    HVX_Vector *pv = (HVX_Vector *)scales;
    pv[0] = Q6_Vh_vsplat_R(HMX_F16_ONE);
    pv[1] = Q6_V_vzero();
}

/* ---- 64x64 RM<->tile conversion (2-row interleave, vshuff/vdeal -2) ----
 * Tile layout in VTCM for a 64x64 block: 4 tiles, row-major tile index ti = trow*BL_T + tcol,
 * each tile HMX_TILE_B bytes.  AH (activation) and WH (weight) are the SAME bytes on v75 f16. */

/* row-major f16 [64x64] (stride `ld`) -> 4 AH/WH tiles at `tiles`. */
static inline void gdn_rm_to_tiles64(_Float16 *tiles, const _Float16 *src, uint32_t ld) {
    for (uint32_t trow = 0; trow < BL_T; ++trow)
        for (uint32_t tcol = 0; tcol < BL_T; ++tcol) {
            HVX_Vector *out = (HVX_Vector *)((uint8_t *)tiles + (trow * BL_T + tcol) * HMX_TILE_B);
            uint32_t r0 = trow * HMX_TILE, c0 = tcol * HMX_TILE;
            for (uint32_t rr = 0; rr < HMX_TILE / 2; ++rr) {
                HVX_Vector v_e = Q6_V_vzero(), v_o = Q6_V_vzero();
                memcpy(&v_e, src + (size_t)(r0 + rr * 2) * ld + c0, HMX_TILE * sizeof(_Float16));
                memcpy(&v_o, src + (size_t)(r0 + rr * 2 + 1) * ld + c0, HMX_TILE * sizeof(_Float16));
                out[rr] = Q6_V_lo_W(Q6_W_vshuff_VVR(v_o, v_e, -2));
            }
        }
}
/* 4 AH tiles at `tiles` -> row-major f16 [64x64] (stride `ld`). */
static inline void gdn_tiles_to_rm64(_Float16 *dst, uint32_t ld, const _Float16 *tiles) {
    for (uint32_t trow = 0; trow < BL_T; ++trow)
        for (uint32_t tcol = 0; tcol < BL_T; ++tcol) {
            const HVX_Vector *tile = (const HVX_Vector *)((uint8_t *)tiles + (trow * BL_T + tcol) * HMX_TILE_B);
            uint32_t r0 = trow * HMX_TILE, c0 = tcol * HMX_TILE;
            for (uint32_t rr = 0; rr < HMX_TILE / 2; ++rr) {
                HVX_VectorPair d = Q6_W_vdeal_VVR(Q6_V_vzero(), tile[rr], -2);
                HVX_Vector v_e = Q6_V_lo_W(d), v_o = Q6_V_hi_W(d);
                memcpy(dst + (size_t)(r0 + rr * 2) * ld + c0, &v_e, HMX_TILE * sizeof(_Float16));
                memcpy(dst + (size_t)(r0 + rr * 2 + 1) * ld + c0, &v_o, HMX_TILE * sizeof(_Float16));
            }
        }
}

/* ---- 64x64x64 f16 HMX matmul: C[4 tiles] = A_tiles @ B_tiles, all VTCM-resident, ACCUMULATE.
 * If `clear` is set, the accumulator is zeroed first; otherwise C += A@B (for the S_ij sum).
 * Tile-resident in, tile-resident out (out is an AH tile, directly reusable as the next input).
 * Note: tiles for the matmul must be laid out so that, for output tile (rt,ct), the K tiles of A
 * are A[rt*K + kt] and the K tiles of B are B[ct*K + kt] (B in column-major tile order = WH). */
static inline void gdn_hmx_mm64(_Float16 *Ctiles, const _Float16 *Atiles_ah,
                                const _Float16 *Btiles_wh, int clear) {
    const uint32_t K = BL_T; /* 2 */
    for (uint32_t rt = 0; rt < BL_T; ++rt)
        for (uint32_t ct = 0; ct < BL_T; ++ct) {
            if (clear) hmx_clear_acc();
            const uint8_t *ap = (const uint8_t *)Atiles_ah + rt * K * HMX_TILE_B;
            const uint8_t *bp = (const uint8_t *)Btiles_wh + ct * K * HMX_TILE_B;
            hmx_load_tiles(ap, bp, K);                 /* :deep loads both K tiles, MAC */
            hmx_store_acc((_Float16 *)((uint8_t *)Ctiles + (rt * BL_T + ct) * HMX_TILE_B));
        }
}

#endif /* GDN_F16_HMX_H */
