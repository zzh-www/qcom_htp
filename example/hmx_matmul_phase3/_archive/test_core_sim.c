/*
 * test_core_sim.c — standalone sim validation of hmx_core.c Phase 3B
 * HMX-only kernel.
 *
 * Strategy: build packed tiles using Phase 2's known-good pack functions
 * (copied in here from hmx_int4xint8_matmul.c), call hmx_matmul_core_mn,
 * unpack dual-scale readback, compare to scalar reference.
 *
 * This isolates hmx_core's HMX MAC correctness from upstream HVX op
 * correctness. If this passes, the Phase 3B core is verified and we can
 * focus on Phase 3B upstream ops + graph integration.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>

#include "kernel/hmx_core.h"

#define TILE 32

/* --- Phase-2 packers (copied to break dep on hmx_matmul_qnn) --------------- */
static void pack_weight_32x32_ref(uint8_t *tile, const int8_t *w_32x32)
{
    for (int kg = 0; kg < 8; kg++) {
        uint32_t *dst = (uint32_t *)(tile + 128 * kg);
        const uint8_t *r0 = (const uint8_t *)&w_32x32[(kg * 4 + 0) * 32];
        const uint8_t *r1 = (const uint8_t *)&w_32x32[(kg * 4 + 1) * 32];
        const uint8_t *r2 = (const uint8_t *)&w_32x32[(kg * 4 + 2) * 32];
        const uint8_t *r3 = (const uint8_t *)&w_32x32[(kg * 4 + 3) * 32];
        for (int col = 0; col < 32; col++) {
            dst[col] =  (uint32_t)r0[col]
                     | ((uint32_t)r1[col] << 8)
                     | ((uint32_t)r2[col] << 16)
                     | ((uint32_t)r3[col] << 24);
        }
    }
}

/* Pack 32x32 u8 activation into HMX tile format (bytes [0,s0,0,s1]). */
static void pack_activation_32x32_u8_ref(uint8_t *tile, const uint8_t *au, int K)
{
    memset(tile, 0, 2048);
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        uint32_t *dst = (uint32_t *)(tile + 128 * phys_row);
        const uint8_t *s0 = &au[phys_row * K];
        const uint8_t *s1 = &au[(phys_row + 16) * K];
        for (int k = 0; k < 32; k++)
            dst[k] = ((uint32_t)s1[k] << 24) | ((uint32_t)s0[k] << 8);
    }
}

/* Scalar ref (int8 × int8 accumulator; we treat activation as uint8 so
 * reference is sum of (uint8) * (int8) = acc pre-offset-correction. */
static void ref_matmul_u8_i8(int32_t *out, const uint8_t *au, const int8_t *w,
                              int M, int K, int N)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++)
                s += (int32_t)au[i * K + k] * (int32_t)w[k * N + j];
            out[i * N + j] = s;
        }
}

/* Unpack HMX dual-scale readback into int32 (matches Phase 2 kernel). */
static void unpack_dual_scale(int32_t *P, const uint16_t *out_lo, const uint16_t *out_hi)
{
    for (int ir = 0; ir < 32; ir++) {
        int phys_row = ir & 15, stream = ir >> 4;
        for (int jc = 0; jc < 32; jc++) {
            int idx = phys_row * 64 + 2 * jc + stream;
            uint16_t lo = out_lo[idx], hi = out_hi[idx];
            P[ir * 32 + jc] = ((int32_t)(int16_t)hi << 8) | ((int32_t)lo & 0xFF);
        }
    }
}

int main(void)
{
    printf("=== Phase 3B hmx_core sim test (32x32x32 int8) ===\n");

    unsigned int vtcm_base = h2_info(INFO_VTCM_BASE);
    if (!vtcm_base) { h2_thread_stop(1); return 1; }

    h2_vecaccess_state_t vacc;
    h2_vecaccess_unit_init(&vacc, H2_VECACCESS_HVX_128,
                           CFG_TYPE_VXU0, CFG_SUBTYPE_VXU0,
                           CFG_HVX_CONTEXTS, 0x1);
    h2_vecaccess_acquire(&vacc);
    h2_mxaccess_state_t mxacc;
    h2_mxaccess_unit_init(&mxacc, CFG_TYPE_VXU0, CFG_SUBTYPE_VXU0,
                          CFG_HMX_CONTEXTS, 0x1);
    h2_mxaccess_acquire(&mxacc);

    uint8_t *vt = (uint8_t *)(unsigned long)vtcm_base;

    /* VTCM layout for test:
     *   [0,     512)     bias scratch
     *   [512,   2560)    packed_act tile (2 KiB)
     *   [2560,  3584)    packed_wt tile (1 KiB)
     *   [4096,  6144)    out_lo (2 KiB, 2-KB aligned)
     *   [6144,  8192)    out_hi (2 KiB) */
    void    *bias_vtcm = vt + 0;
    uint8_t *packed_act = vt + 512;
    uint8_t *packed_wt  = vt + 2560;
    uint16_t *out_lo    = (uint16_t *)(vt + 4096);
    uint16_t *out_hi    = (uint16_t *)(vt + 6144);

    hmx_core_fill_bias(bias_vtcm);

    int fails = 0;

    /* ---- Scenario S1: constant (au=1, w=1) ---- */
    {
        printf("\n-- S1 (au=1, w=1) --\n");
        static uint8_t au[TILE*TILE];
        static int8_t  w [TILE*TILE];
        static int32_t out_ref[TILE*TILE];
        static int32_t out_hmx[TILE*TILE];
        for (int i = 0; i < TILE*TILE; i++) { au[i] = 1; w[i] = 1; }

        pack_activation_32x32_u8_ref(packed_act, au, TILE);
        pack_weight_32x32_ref(packed_wt, w);

        hmx_matmul_core_mn(packed_act, packed_wt, 1, 1, 1, 0, 0,
                           bias_vtcm, out_lo, out_hi);

        unpack_dual_scale(out_hmx, out_lo, out_hi);
        ref_matmul_u8_i8(out_ref, au, w, TILE, TILE, TILE);

        int d = 0; int max_err = 0;
        for (int i = 0; i < TILE*TILE; i++) {
            int e = (int)(out_hmx[i] - out_ref[i]);
            if (e) { d++; if (e<0) e=-e; if (e>max_err) max_err=e; }
        }
        if (d == 0) {
            printf("  [PASS] bit-exact, out[0..3] = %ld %ld %ld %ld\n",
                   (long)out_hmx[0], (long)out_hmx[1], (long)out_hmx[2], (long)out_hmx[3]);
        } else {
            printf("  [FAIL] %d/1024 differ, max_err=%d\n", d, max_err);
            for (int i = 0; i < 4 && i < TILE; i++)
                printf("    [0,%d] ref=%ld hmx=%ld\n", i,
                       (long)out_ref[i], (long)out_hmx[i]);
            fails++;
        }
    }

    /* ---- S2: random (au in [0,255], w in [-7,7]) ---- */
    {
        printf("\n-- S2 (random u8 × i4 range) --\n");
        static uint8_t au[TILE*TILE];
        static int8_t  w [TILE*TILE];
        static int32_t out_ref[TILE*TILE];
        static int32_t out_hmx[TILE*TILE];
        uint32_t rng = 0xCAFEBABEu;
        for (int i = 0; i < TILE*TILE; i++) {
            rng = rng * 1664525u + 1013904223u;
            au[i] = (uint8_t)(rng >> 24);
        }
        for (int i = 0; i < TILE*TILE; i++) {
            rng = rng * 1664525u + 1013904223u;
            w[i] = (int8_t)(((rng >> 24) % 15) - 7);
        }

        pack_activation_32x32_u8_ref(packed_act, au, TILE);
        pack_weight_32x32_ref(packed_wt, w);

        hmx_matmul_core_mn(packed_act, packed_wt, 1, 1, 1, 0, 0,
                           bias_vtcm, out_lo, out_hi);

        unpack_dual_scale(out_hmx, out_lo, out_hi);
        ref_matmul_u8_i8(out_ref, au, w, TILE, TILE, TILE);

        int d = 0; int max_err = 0;
        for (int i = 0; i < TILE*TILE; i++) {
            int e = (int)(out_hmx[i] - out_ref[i]);
            if (e) { d++; if (e<0) e=-e; if (e>max_err) max_err=e; }
        }
        if (d == 0)
            printf("  [PASS] bit-exact\n");
        else {
            printf("  [FAIL] %d/1024 differ, max_err=%d\n", d, max_err);
            fails++;
        }
    }

    printf("\n%s\n", fails == 0 ? "ALL PASS" : "FAIL");
    h2_thread_stop(fails);
    return fails;
}
