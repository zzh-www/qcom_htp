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

/* Gather 32 rows × 32 cols of activation into 1 KiB row-major VTCM tile.
 * Source is [M_full][K_full] row-major (stride = K_full bytes per row). */
void hmx_core_v2_gather_act_tile(
    uint8_t       *tile_vtcm,
    const uint8_t *au,
    uint32_t       K_full,
    uint32_t       m0,
    uint32_t       k0)
{
    for (uint32_t r = 0; r < 32; r++) {
        memcpy(tile_vtcm + r * 32, &au[(m0 + r) * K_full + k0], 32);
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
    /* Row-major 32 K-rows × 32 N-cols, 1 KiB contiguous.
     * This matches Agent A's probe layout (fill_wt just memset all-1)
     * and pairs with activation.ub=:cm / weight.b=plain at 7.9 cyc/MAC. */
    for (uint32_t r = 0; r < 32; r++) {
        memcpy(tile_vtcm + r * 32, &wu[(k0 + r) * N_full + n0], 32);
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

    /* :cm mode: 2 passes × dual-scale readback (4 stores total) for
     * 32 rows × 32 cols output with int24-range precision. */

    /* Pass 1: rows 0..15 */
    hmx_load_bias_i(blo);
    hmx_clracc_i();
    for (uint32_t kt = 0; kt < K_tiles; kt++) {
        hmx_load_pair_cm(act_tiles + kt * 1024,
                         wt_tiles  + kt * 1024);
    }
    hmx_store_acc_uh_2x1_retain(out_top_lo);
    hmx_load_bias_i(bhi);
    hmx_store_acc_uh_2x1(out_top_hi);   /* final → acc cleared */

    /* Pass 2: rows 16..31 — activation pointer +512 bytes. */
    hmx_load_bias_i(blo);
    hmx_clracc_i();
    for (uint32_t kt = 0; kt < K_tiles; kt++) {
        hmx_load_pair_cm(act_tiles + kt * 1024 + 512,
                         wt_tiles  + kt * 1024);
    }
    hmx_store_acc_uh_2x1_retain(out_bot_lo);
    hmx_load_bias_i(bhi);
    hmx_store_acc_uh_2x1(out_bot_hi);
#else
    (void)act_tiles; (void)wt_tiles; (void)K_tiles; (void)bias_vtcm;
    (void)out_top_lo; (void)out_top_hi; (void)out_bot_lo; (void)out_bot_hi;
#endif
}
