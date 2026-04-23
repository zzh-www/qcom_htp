/*
 * hmx_core_v2.c — HMX MatMul kernel using `:cm` + row-major activation.
 * Implements hmx_core_v2.h API.
 */
#include "hmx_core_v2.h"
#include <string.h>

#ifndef HMX_RT_ACT_CM
/* r7 | 0x1c required; 2047 already covers 0x1c (= 0x7ff), so 2047 works. */
#define HMX_RT_ACT_CM (2047 | 0x1C)
#endif
#ifndef HMX_RT_WT
#define HMX_RT_WT  0x3FF
#endif

#if defined(__hexagon__)

static inline __attribute__((always_inline)) void hmx_clracc_i(void)
{ __asm__ volatile("mxclracc" ::: "memory"); }

static inline __attribute__((always_inline))
void hmx_load_bias_i(const void *p)
{ __asm__ volatile("bias = mxmem(%0)" :: "r"(p) : "memory"); }

/* `:cm` activation + plain weight — Agent A probe P=C (7.92 cyc/MAC). */
static inline __attribute__((always_inline))
void hmx_load_pair_cm(const void *act, const void *wt)
{
    __asm__ volatile(
        "{ activation.ub = mxmem(%0, %1):cm\n"
        "  weight.b      = mxmem(%2, %3) }\n"
        :: "r"(act), "r"(HMX_RT_ACT_CM),
           "r"(wt),  "r"(HMX_RT_WT)
        : "memory");
}

/* Phase-2 plain mxmem pair (2-stream activation, packed weight). */
static inline __attribute__((always_inline))
void hmx_load_pair_plain(const void *act, const void *wt)
{
    __asm__ volatile(
        "{ activation.ub = mxmem(%0, %1)\n"
        "  weight.b      = mxmem(%2, %3) }\n"
        :: "r"(act), "r"(2047),
           "r"(wt),  "r"(HMX_RT_WT)
        : "memory");
}

static inline __attribute__((always_inline))
void hmx_store_acc_uh_2x1_retain(void *out)
{ __asm__ volatile("mxmem(%0, %1):after:retain.uh = acc:2x1\n"
               :: "r"(out), "r"(0) : "memory"); }

static inline __attribute__((always_inline))
void hmx_store_acc_uh_2x1(void *out)
{ __asm__ volatile("mxmem(%0, %1):after.uh = acc:2x1\n"
               :: "r"(out), "r"(0) : "memory"); }

#endif /* __hexagon__ */


void hmx_core_v2_fill_bias(void *bias_vtcm)
{
    uint16_t *blo = (uint16_t *)bias_vtcm;
    uint16_t *bhi = blo + 128;
    for (int i = 0; i < 128; i++) { blo[i] = 0x4000; bhi[i] = 0x2000; }
}

/* Phase 2 activation pack: 2-stream interleaved (bytes [0, s0, 0, s1]
 * per 4B cell). 32 logical rows × 32 K-cols → 16 phys_rows × 128 B tile
 * = 2 KiB. Scalar here; could be HVX'd in a separate upstream op. */
void hmx_core_v2_gather_act_tile(
    uint8_t       *tile_vtcm,
    const uint8_t *au,
    uint32_t       K_full,
    uint32_t       m0,
    uint32_t       k0)
{
    memset(tile_vtcm, 0, 2048);
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        uint32_t *dst = (uint32_t *)(tile_vtcm + 128 * phys_row);
        const uint8_t *s0 = &au[(m0 + phys_row)      * K_full + k0];
        const uint8_t *s1 = &au[(m0 + phys_row + 16) * K_full + k0];
        for (int k = 0; k < 32; k++)
            dst[k] = ((uint32_t)s1[k] << 24) | ((uint32_t)s0[k] << 8);
    }
}

/* Gather 32 K-rows × 32 N-cols of weight into VTCM tile.
 *
 * Key layout decision: Agent A's silicon probe Scenario C used a
 * "row-major weight" implicitly (all-1 bytes in 1 KiB), same answer as
 * pre-Phase 3 weight packed format. But the PRM says HMX weight.b
 * expects a specific per-K-group packing (4 K-rows × 32 cols per 128 B).
 *
 * For this first Phase 3B cut, use Phase 2's pack_weight_32x32 layout:
 *   128 bytes per K-group (4 consecutive K-rows × 32 N-cols)
 *   8 K-groups per tile → 1024 B
 * This is the SAME as hmx_int4_matmul.c pack_weight_32x32 output.
 * Scalar loop here; HVX-ify via the upstream PackWeightToHmxTile op
 * in a later phase if profile shows it matters. */
void hmx_core_v2_gather_wt_tile(
    uint8_t       *tile_vtcm,
    const int8_t  *wu,
    uint32_t       N_full,
    uint32_t       k0,
    uint32_t       n0)
{
    /* Phase 2 packed format: 8 K-groups × 32 cells × 4 bytes each.
     * Each cell encodes 4 consecutive K-rows at one N col. Empirically
     * required for `weight.b = mxmem` even with `:cm` on activation
     * (Agent A probe used uniform all-1 weight so didn't discriminate). */
    for (int kg = 0; kg < 8; kg++) {
        uint32_t *dst = (uint32_t *)(tile_vtcm + 128 * kg);
        const uint8_t *r0 = (const uint8_t *)&wu[(k0 + kg * 4 + 0) * N_full + n0];
        const uint8_t *r1 = (const uint8_t *)&wu[(k0 + kg * 4 + 1) * N_full + n0];
        const uint8_t *r2 = (const uint8_t *)&wu[(k0 + kg * 4 + 2) * N_full + n0];
        const uint8_t *r3 = (const uint8_t *)&wu[(k0 + kg * 4 + 3) * N_full + n0];
        for (int col = 0; col < 32; col++) {
            dst[col] =  (uint32_t)r0[col]
                     | ((uint32_t)r1[col] << 8)
                     | ((uint32_t)r2[col] << 16)
                     | ((uint32_t)r3[col] << 24);
        }
    }
}

void hmx_matmul_v2_core_mn(
    const uint8_t *act_tiles,
    const uint8_t *wt_tiles,
    uint32_t       K_tiles,
    void          *bias_vtcm,
    uint16_t      *out_top_lo,
    uint16_t      *out_top_hi,
    uint16_t      *out_bot_lo,
    uint16_t      *out_bot_hi)
{
#if defined(__hexagon__)
    uint16_t *blo = (uint16_t *)bias_vtcm;
    uint16_t *bhi = blo + 128;

    /* Phase 2 plain mxmem path: single MAC sequence covers 32 logical
     * rows via 2-stream activation tile; dual-scale readback gives full
     * int24 range. out_bot buffers unused in this path. */
    (void)out_bot_lo; (void)out_bot_hi;

    hmx_load_bias_i(blo);
    hmx_clracc_i();
    for (uint32_t kt = 0; kt < K_tiles; kt++) {
        /* act_tile stride = 2 KiB (2-stream pack); wt_tile = 1 KiB. */
        hmx_load_pair_plain(act_tiles + kt * 2048,
                            wt_tiles  + kt * 1024);
    }
    hmx_store_acc_uh_2x1_retain(out_top_lo);
    hmx_load_bias_i(bhi);
    hmx_store_acc_uh_2x1(out_top_hi);
#else
    (void)act_tiles; (void)wt_tiles; (void)K_tiles; (void)bias_vtcm;
    (void)out_top_lo; (void)out_top_hi; (void)out_bot_lo; (void)out_bot_hi;
#endif
}
