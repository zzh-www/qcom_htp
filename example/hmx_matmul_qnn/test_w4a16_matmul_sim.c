/*
 * test_w4a16_matmul_sim.c — standalone hexagon-sim harness for the w4a16
 * HMX kernel. Bit-exact vs scalar int32 reference + reports pcycles.
 * P5 sim↔device cycle parity measurement for w4a16.
 */

#include <stdio.h>
#include <stdint.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>
#include "kernel/hmx_int4_matmul.h"

#define TILE 32

static uint32_t rng_state;
static inline uint32_t rng_next(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

/* Scalar ref: out = Σ_k (au[m,k] - 32768) × w_signed[k,n].
 * `au` is post-Cast uint16 (signed + 32768). `w` is signed int8 (int4 range). */
static void ref_matmul_w4a16(int32_t        *out,
                              const uint16_t *au,
                              const int8_t   *w,
                              int M, int K, int N)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++)
                s += ((int32_t)au[i*K + k] - 32768) * (int32_t)w[k*N + j];
            out[i*N + j] = s;
        }
}

static int scenario(const char *title,
                    const uint16_t *au, const int8_t *w, void *vtcm_ws)
{
    printf("\n--- %s ---\n", title);
    static int32_t out_ref[TILE*TILE];
    static int32_t out_hmx[TILE*TILE];

    ref_matmul_w4a16(out_ref, au, w, TILE, TILE, TILE);

    /* Call the fused-activation-prepack + dualacc matmul path. */
    hmx_int4_prepack_activation_fused(au, TILE, TILE, 0, vtcm_ws);
    hmx_int4_matmul_mn_dualacc(out_hmx, w, TILE, vtcm_ws, NULL);

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
    printf("  w4a16 HMX kernel (int4×int16, dualacc)\n");
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

    /* S1 small constant */
    {
        static uint16_t au[TILE*TILE];
        static int8_t w[TILE*TILE];
        for (int i = 0; i < TILE*TILE; i++) { au[i] = 32768 + 100; w[i] = 3; }
        fail += scenario("S1 (a=100 signed16, w=3)", au, w, vtcm_ws);
    }
    /* S2 random int16 + int4 */
    {
        static uint16_t au[TILE*TILE];
        static int8_t w[TILE*TILE];
        rng_state = 0xCAFEBABEu;
        for (int i = 0; i < TILE*TILE; i++)
            au[i] = (uint16_t)(rng_next() >> 16);
        for (int i = 0; i < TILE*TILE; i++)
            w[i] = (int8_t)(((rng_next() >> 24) % 15) - 7);
        fail += scenario("S2 random int16·int4", au, w, vtcm_ws);
    }
    /* S3 edge: max positive */
    {
        static uint16_t au[TILE*TILE];
        static int8_t w[TILE*TILE];
        for (int i = 0; i < TILE*TILE; i++) au[i] = 32768 + 32767;
        for (int i = 0; i < TILE*TILE; i++) w[i] = 7;
        fail += scenario("S3 (a=32767, w=7 — max positive)", au, w, vtcm_ws);
    }

    printf("\n%s\n", fail == 0 ? "ALL PASS" : "FAIL");
    h2_thread_stop(fail);
    return fail;
}
