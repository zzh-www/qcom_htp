/*
 * hmx_core.c — Pure HMX MatMul kernel (Phase 3B core).
 *
 * Implements HMX_CORE_H API. Zero HVX, zero scalar data movement in hot
 * loop. Consumes pre-packed HMX tiles written by upstream HVX ops.
 */
#include "hmx_core.h"
#include <string.h>

#ifndef HMX_RT_ACT
#define HMX_RT_ACT 2047
#endif
#ifndef HMX_RT_WT
#define HMX_RT_WT  0x3FF
#endif

#if defined(__hexagon__)

/* HMX inline-asm wrappers (identical to Phase 2 kernels). */
static inline __attribute__((always_inline)) void hmx_clracc_i(void)
{ asm volatile("mxclracc" ::: "memory"); }

static inline __attribute__((always_inline)) void hmx_swapacc_i(void)
{ asm volatile("mxswapacc" ::: "memory"); }

static inline __attribute__((always_inline))
void hmx_load_bias_i(const void *p)
{ asm volatile("bias = mxmem(%0)" :: "r"(p) : "memory"); }

static inline __attribute__((always_inline))
void hmx_load_pair_u8_i8(const void *act, const void *wt)
{
    asm volatile(
        "{ activation.ub = mxmem(%0, %1)\n"
        "  weight.b      = mxmem(%2, %3) }\n"
        :: "r"(act), "r"(HMX_RT_ACT),
           "r"(wt),  "r"(HMX_RT_WT)
        : "memory");
}

static inline __attribute__((always_inline))
void hmx_store_acc_uh_2x1_retain(void *out)
{ asm volatile("mxmem(%0, %1):after:retain.uh = acc:2x1\n"
               :: "r"(out), "r"(0) : "memory"); }

static inline __attribute__((always_inline))
void hmx_store_acc_uh_2x1(void *out)
{ asm volatile("mxmem(%0, %1):after.uh = acc:2x1\n"
               :: "r"(out), "r"(0) : "memory"); }

#endif /* __hexagon__ */


void hmx_core_fill_bias(void *bias_vtcm)
{
    /* Dual-scale bias table: 128 uint16 of 0x4000 (low scale), then
     * 128 uint16 of 0x2000 (high scale). Filled once; re-used across all
     * (m,n) tile invocations.  Scalar fill is cheap (done once). */
    uint16_t *blo = (uint16_t *)bias_vtcm;
    uint16_t *bhi = blo + 128;
    for (int i = 0; i < 128; i++) { blo[i] = 0x4000; bhi[i] = 0x2000; }
}

void hmx_matmul_core_mn(
    const uint8_t *packed_act,
    const uint8_t *packed_wt,
    uint32_t       M_tiles,
    uint32_t       K_tiles,
    uint32_t       N_tiles,
    uint32_t       mt,
    uint32_t       nt,
    void          *bias_vtcm,
    uint16_t      *out_lo,
    uint16_t      *out_hi)
{
    (void)M_tiles;  /* validated upstream */
#if defined(__hexagon__)
    uint16_t *blo = (uint16_t *)bias_vtcm;
    uint16_t *bhi = blo + 128;

    /* Load low-scale bias and clear acc. */
    hmx_load_bias_i(blo);
    hmx_clracc_i();

    /* Inner MAC loop — pure HMX. Row-major tile memory:
     *   act_tile[kt] = packed_act + (mt * K_tiles + kt) * 2048
     *   wt_tile[kt]  = packed_wt  + (kt * N_tiles + nt) * 1024   */
    for (uint32_t kt = 0; kt < K_tiles; kt++) {
        const uint8_t *act_tile = packed_act + ((uint64_t)mt * K_tiles + kt) * 2048;
        const uint8_t *wt_tile  = packed_wt  + ((uint64_t)kt * N_tiles + nt) * 1024;
        hmx_load_pair_u8_i8(act_tile, wt_tile);
    }

    /* Dual-scale readback: store with low-scale bias (retain), swap bias
     * to high, store high-scale. Final store without :retain clears acc. */
    hmx_store_acc_uh_2x1_retain(out_lo);
    hmx_load_bias_i(bhi);
    hmx_store_acc_uh_2x1(out_hi);
#else
    (void)packed_act; (void)packed_wt; (void)K_tiles; (void)N_tiles;
    (void)mt; (void)nt; (void)bias_vtcm; (void)out_lo; (void)out_hi;
#endif
}


void hmx_matmul_core_mn_dualacc(
    const uint8_t *packed_act_hi,
    const uint8_t *packed_act_lo,
    const uint8_t *packed_wt,
    uint32_t       M_tiles,
    uint32_t       K_tiles,
    uint32_t       N_tiles,
    uint32_t       mt,
    uint32_t       nt,
    void          *bias_vtcm,
    uint16_t      *out_hi_lo_A,
    uint16_t      *out_hi_hi_A,
    uint16_t      *out_lo_lo_B,
    uint16_t      *out_lo_hi_B)
{
    (void)M_tiles;
#if defined(__hexagon__)
    uint16_t *blo = (uint16_t *)bias_vtcm;
    uint16_t *bhi = blo + 128;

    /* Dualacc setup: clear both accumulators (current=A → clracc; swap →
     * current=B → clracc; swap back → current=A). */
    hmx_load_bias_i(blo);
    hmx_clracc_i();
    hmx_swapacc_i();
    hmx_clracc_i();
    hmx_swapacc_i();     /* current=A */

    /* Fused K-loop: hi stream → acc A; lo stream → acc B. Weight shared.
     * After each K iter we've swapped 2× (net zero), so current=A again. */
    for (uint32_t kt = 0; kt < K_tiles; kt++) {
        const uint8_t *wt_tile  = packed_wt     + ((uint64_t)kt * N_tiles + nt) * 1024;
        const uint8_t *ah_tile  = packed_act_hi + ((uint64_t)mt * K_tiles + kt) * 2048;
        const uint8_t *al_tile  = packed_act_lo + ((uint64_t)mt * K_tiles + kt) * 2048;

        hmx_load_pair_u8_i8(ah_tile, wt_tile);   /* → A */
        hmx_swapacc_i();
        hmx_load_pair_u8_i8(al_tile, wt_tile);   /* → B */
        hmx_swapacc_i();
    }
    /* current=A (swapped 2K times total, K even or K odd same). */

    /* Dual-scale readback A (hi partial). */
    hmx_store_acc_uh_2x1_retain(out_hi_lo_A);
    hmx_load_bias_i(bhi);
    asm volatile("mxmem(%0, %1):after:retain.uh = acc:2x1\n"
                 :: "r"(out_hi_hi_A), "r"(0) : "memory");

    /* Swap to B (lo partial), readback. */
    hmx_swapacc_i();
    hmx_load_bias_i(blo);
    hmx_store_acc_uh_2x1_retain(out_lo_lo_B);
    hmx_load_bias_i(bhi);
    hmx_store_acc_uh_2x1(out_lo_hi_B);
#else
    (void)packed_act_hi; (void)packed_act_lo; (void)packed_wt;
    (void)K_tiles; (void)N_tiles; (void)mt; (void)nt; (void)bias_vtcm;
    (void)out_hi_lo_A; (void)out_hi_hi_A; (void)out_lo_lo_B; (void)out_lo_hi_B;
#endif
}
