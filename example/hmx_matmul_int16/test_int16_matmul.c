/*
 * test_int16_matmul.c — end-to-end test for the HMX int16 matmul kernel.
 *
 * The HMX kernel now does the full 4-term u8×i8 decomposition with
 * bit-correction for the u8×u8 partial products.  It MUST match the
 * exact int16 reference bit-exactly across all scenarios.
 *
 *   S1  constant small:   A=256, W=256            (simple sanity)
 *   S2  constant mid:     A=−10000, W=20000       (uint16 wrap territory)
 *   S3  random int16:     full range              (stresses layout + combine)
 */

#include <stdio.h>
#include <stdint.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>
#include "int16_matmul.h"

static uint32_t rng_state;
static inline uint32_t rng_next(void) {
    rng_state = rng_state * 1664525u + 1013904223u; return rng_state;
}
static inline int16_t rng_i16(void) { return (int16_t)(rng_next() >> 16); }

static void preview(const char *label, const int16_t *buf)
{
    printf("  %-16s:", label);
    for (int i = 0; i < 8; i++) printf(" %6d", buf[i]);
    printf("\n");
}

static int cmp_tiles(const int16_t *a, const int16_t *b, int *max_abs)
{
    int diff = 0, m = 0;
    for (int i = 0; i < IM_TILE_MN; i++) {
        int d = (int)a[i] - (int)b[i]; if (d < 0) d = -d;
        if (a[i] != b[i]) diff++;
        if (d > m) m = d;
    }
    if (max_abs) *max_abs = m;
    return diff;
}

static int scenario(const char *title,
                    const int16_t *A, const int16_t *W,
                    im_requant_t rq, void *vtcm_ws)
{
    printf("\n--- %s ---\n", title);
    static int16_t out_ref[IM_TILE_MN];
    static int16_t out_hmx[IM_TILE_MN];

    im_matmul_ref    (out_ref, A, W, rq);
    im_matmul_hmx_i8 (out_hmx, A, W, rq, vtcm_ws);

    preview("ref_exact", out_ref);
    preview("hmx      ", out_hmx);

    int m; int d = cmp_tiles(out_hmx, out_ref, &m);
    if (d == 0) {
        printf("  [PASS] HMX == ref bit-exact (%d / %d)\n", IM_TILE_MN, IM_TILE_MN);
        return 0;
    }
    printf("  [FAIL] %d / %d differ, max_abs=%d\n", d, IM_TILE_MN, m);
    /* Show first 8 disagreements. */
    int shown = 0;
    for (int idx = 0; idx < IM_TILE_MN && shown < 8; idx++) {
        if (out_ref[idx] != out_hmx[idx]) {
            printf("    [%d,%d]  ref=%6d  hmx=%6d  diff=%+d\n",
                   idx / 32, idx % 32, out_ref[idx], out_hmx[idx],
                   out_hmx[idx] - out_ref[idx]);
            shown++;
        }
    }
    return 1;
}

int main(void)
{
    int fail = 0;

    printf("========================================\n");
    printf("  int16 per-tensor quantized matmul (HMX u8×i8 full decomp)\n");
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

    /* S1 */
    {
        static int16_t A[IM_TILE_MK], W[IM_TILE_KN];
        for (int i = 0; i < IM_TILE_MK; i++) A[i] = 256;
        for (int i = 0; i < IM_TILE_KN; i++) W[i] = 256;
        im_requant_t rq = { .mul = 1, .shift = 16 };
        fail += scenario("S1 constant small  (A=256,  W=256,  shift=16)",
                         A, W, rq, vtcm_ws);
    }

    /* S2 */
    {
        static int16_t A[IM_TILE_MK], W[IM_TILE_KN];
        for (int i = 0; i < IM_TILE_MK; i++) A[i] = -10000;
        for (int i = 0; i < IM_TILE_KN; i++) W[i] =  20000;
        im_requant_t rq = { .mul = 1, .shift = 25 };
        fail += scenario("S2 constant mid    (A=-10000, W=20000, shift=25)",
                         A, W, rq, vtcm_ws);
    }

    /* S3 */
    {
        static int16_t A[IM_TILE_MK], W[IM_TILE_KN];
        rng_state = 0xC0DEBABEu;
        for (int i = 0; i < IM_TILE_MK; i++) A[i] = rng_i16();
        for (int i = 0; i < IM_TILE_KN; i++) W[i] = rng_i16();
        im_requant_t rq = { .mul = 1, .shift = 20 };
        fail += scenario("S3 random int16    (full range, shift=20)",
                         A, W, rq, vtcm_ws);
    }

    printf("\n========================================\n");
    printf("  %s\n", fail == 0 ? "ALL PASS" : "FAIL");
    printf("========================================\n");

    h2_thread_stop(fail);
    return fail;
}
