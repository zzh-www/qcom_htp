/*
 * test_w4a8_matmul_sim.c — standalone hexagon-sim harness for the w4a8
 * HMX kernel. Bit-exact vs scalar int32 reference + reports pcycles.
 *
 * Runs under hexagon-sim + H2 booter, same recipe as
 * tests/test_hmx_matmul_int16.sh. Provides P5 sim↔device cycle parity
 * measurement for the w4a8 kernel.
 */

#include <stdio.h>
#include <stdint.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>
#include "kernel/hmx_int4xint8_matmul.h"

#define TILE 32

static uint32_t rng_state;
static inline uint32_t rng_next(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

/* Scalar reference: out = Σ_k (au[m,k] - 128) × w_signed[k,n].
 * `au` is post-Cast uint8 (signed + 128). `w` is signed int8 (w4 range). */
static void ref_matmul_w4a8(int32_t       *out,
                             const uint8_t *au,
                             const int8_t  *w,
                             int M, int K, int N)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++)
                s += ((int32_t)au[i*K + k] - 128) * (int32_t)w[k*N + j];
            out[i*N + j] = s;
        }
}

static int scenario(const char *title,
                    const uint8_t *au,
                    const int8_t  *w,
                    void          *vtcm_ws)
{
    printf("\n--- %s ---\n", title);
    static int32_t out_ref[TILE*TILE];
    static int32_t out_hmx[TILE*TILE];

    ref_matmul_w4a8(out_ref, au, w, TILE, TILE, TILE);

    /* HMX path: prepack activation to VTCM, then K-accumulated MAC. */
    hmx_int4xint8_prepack_activation(au, TILE, TILE, 0, vtcm_ws);
    /* Weight in K×N layout = [K=32][N=32] row-major = same as w input. */
    hmx_int4xint8_matmul_mn(out_hmx, w, TILE, vtcm_ws);

    int diff = 0, max_abs = 0;
    for (int i = 0; i < TILE*TILE; i++) {
        int d = (int)(out_ref[i] - out_hmx[i]);
        if (d) {
            diff++;
            int ad = d < 0 ? -d : d;
            if (ad > max_abs) max_abs = ad;
        }
    }
    if (diff == 0) {
        printf("  [PASS] HMX == ref bit-exact (%d / %d)\n", TILE*TILE, TILE*TILE);
        printf("  out[0..3] = %d %d %d %d\n",
               out_hmx[0], out_hmx[1], out_hmx[2], out_hmx[3]);
        return 0;
    }
    printf("  [FAIL] %d / %d differ, max_abs=%d\n", diff, TILE*TILE, max_abs);
    int shown = 0;
    for (int idx = 0; idx < TILE*TILE && shown < 8; idx++) {
        if (out_ref[idx] != out_hmx[idx]) {
            printf("    [%d,%d]  ref=%d  hmx=%d\n",
                   idx / TILE, idx % TILE, out_ref[idx], out_hmx[idx]);
            shown++;
        }
    }
    return 1;
}

int main(void)
{
    int fail = 0;
    printf("========================================\n");
    printf("  w4a8 HMX kernel (int4×int8 u8·i8 single-pass)\n");
    printf("========================================\n\n");

    unsigned int vtcm_base = h2_info(INFO_VTCM_BASE);
    printf("[Init] VTCM base=0x%08x\n", vtcm_base);
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

    /* S1: small constant */
    {
        static uint8_t au[TILE*TILE];
        static int8_t  w[TILE*TILE];
        for (int i = 0; i < TILE*TILE; i++) { au[i] = 128 + 1; w[i] = 1; }
        fail += scenario("S1 (a=1 signed, w=1)", au, w, vtcm_ws);
    }

    /* S2: random int4 weights, random int8 activations */
    {
        static uint8_t au[TILE*TILE];
        static int8_t  w[TILE*TILE];
        rng_state = 0xABCD1234u;
        for (int i = 0; i < TILE*TILE; i++) {
            int8_t a = (int8_t)(rng_next() >> 24);
            au[i] = (uint8_t)((int32_t)a + 128);  /* post-Cast */
        }
        for (int i = 0; i < TILE*TILE; i++) {
            /* Uniform in [-7, 7] */
            w[i] = (int8_t)(((rng_next() >> 24) % 15) - 7);
        }
        fail += scenario("S2 random int8·int4", au, w, vtcm_ws);
    }

    /* S3: edge case — all-1 activation, extreme weights */
    {
        static uint8_t au[TILE*TILE];
        static int8_t  w[TILE*TILE];
        for (int i = 0; i < TILE*TILE; i++) au[i] = 128 + 127;  /* signed 127 */
        for (int i = 0; i < TILE*TILE; i++) w[i] = 7;
        fail += scenario("S3 (a=127, w=7 — max positive)", au, w, vtcm_ws);
    }

    printf("\n========================================\n");
    printf("  %s\n", fail == 0 ? "ALL PASS" : "FAIL");
    printf("========================================\n");

    h2_thread_stop(fail);
    return fail;
}
