/*
 * test_w16a16_matmul_sim.c — standalone hexagon-sim harness for the
 * w16a16 HMX kernel. P5 sim↔device cycle parity measurement.
 */

#include <stdio.h>
#include <stdint.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>
#include "kernel/hmx_int16x16_matmul.h"

#define TILE 32

static uint32_t rng_state;
static inline uint32_t rng_next(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static void ref_matmul_w16a16(int32_t *out,
                               const uint16_t *au, const uint16_t *wu,
                               int M, int K, int N)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++) {
                int32_t a = (int32_t)au[i*K + k] - 32768;
                int32_t w = (int32_t)wu[k*N + j] - 32768;
                s += a * w;
            }
            out[i*N + j] = s;
        }
}

static int scenario(const char *title,
                    const uint16_t *au, const uint16_t *wu, void *vtcm_ws)
{
    printf("\n--- %s ---\n", title);
    static int32_t out_ref[TILE*TILE];
    static int32_t out_hmx[TILE*TILE];
    ref_matmul_w16a16(out_ref, au, wu, TILE, TILE, TILE);
    hmx_int16x16_matmul_mn(out_hmx, au, wu, TILE, TILE, TILE, 0, 0, vtcm_ws);

    int diff = 0, max_abs = 0;
    for (int i = 0; i < TILE*TILE; i++) {
        int d = (int)(out_ref[i] - out_hmx[i]);
        if (d) { diff++; int ad = d<0?-d:d; if (ad>max_abs) max_abs=ad; }
    }
    if (diff == 0) {
        printf("  [PASS] HMX == ref bit-exact (%d/%d)\n", TILE*TILE, TILE*TILE);
        printf("  out[0..3] = %ld %ld %ld %ld\n",
               (long)out_hmx[0], (long)out_hmx[1], (long)out_hmx[2], (long)out_hmx[3]);
        return 0;
    }
    printf("  [FAIL] %d/%d differ, max_abs=%d\n", diff, TILE*TILE, max_abs);
    int shown = 0;
    for (int idx = 0; idx < TILE*TILE && shown < 5; idx++) {
        if (out_ref[idx] != out_hmx[idx]) {
            printf("    [%d,%d]  ref=%ld  hmx=%ld\n",
                   idx/TILE, idx%TILE, (long)out_ref[idx], (long)out_hmx[idx]);
            shown++;
        }
    }
    return 1;
}

int main(void)
{
    int fail = 0;
    printf("========================================\n");
    printf("  w16a16 HMX kernel (int16×int16, 4-term decomp)\n");
    printf("========================================\n\n");

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
    void *vtcm_ws = (void *)(unsigned long)vtcm_base;

    /* S1 small */
    {
        static uint16_t au[TILE*TILE], wu[TILE*TILE];
        for (int i = 0; i < TILE*TILE; i++) {
            au[i] = 32768 + 256; wu[i] = 32768 + 256;
        }
        fail += scenario("S1 (a=256,w=256 signed)", au, wu, vtcm_ws);
    }
    /* S2 random */
    {
        static uint16_t au[TILE*TILE], wu[TILE*TILE];
        rng_state = 0xC0DEBABEu;
        for (int i = 0; i < TILE*TILE; i++)
            au[i] = (uint16_t)(rng_next() >> 16);
        for (int i = 0; i < TILE*TILE; i++)
            wu[i] = (uint16_t)(rng_next() >> 16);
        fail += scenario("S2 random int16·int16", au, wu, vtcm_ws);
    }
    /* S3 edge */
    {
        static uint16_t au[TILE*TILE], wu[TILE*TILE];
        for (int i = 0; i < TILE*TILE; i++) {
            au[i] = 32768 - 10000;  /* signed -10000 */
            wu[i] = 32768 + 20000;  /* signed +20000 */
        }
        fail += scenario("S3 (a=-10000, w=20000)", au, wu, vtcm_ws);
    }

    printf("\n%s\n", fail == 0 ? "ALL PASS" : "FAIL");
    h2_thread_stop(fail);
    return fail;
}
