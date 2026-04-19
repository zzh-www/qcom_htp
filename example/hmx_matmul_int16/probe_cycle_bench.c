/*
 * probe_cycle_bench.c — measure HMX / HVX / CPU cycles for one 32x32x32
 * tile of fp16 matmul vs int16 matmul.
 *
 * Uses the sim's `hexagon_sim_read_pcycles()` around narrowly-scoped
 * regions so we can attribute cycles to specific phases.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>
#include <hexagon_types.h>
#include <hexagon_sim_timer.h>

#include "int16_matmul.h"

#define TILE_ELEMS 1024
#define F16_ONE 0x3C00

static void fill_f16(uint16_t *buf, uint16_t val)
{
    int s = ((int)val << 16) | val;
    HVX_Vector v = Q6_V_vsplat_R(s);
    for (int i = 0; i < TILE_ELEMS / 64; i++) ((HVX_Vector *)buf)[i] = v;
}

static void fill_scales_f16(uint8_t *buf, uint16_t val)
{
    int s = ((int)val << 16) | val;
    ((HVX_Vector *)buf)[0] = Q6_V_vsplat_R(s);
    ((HVX_Vector *)buf)[1] = Q6_V_vzero();
}

static void fp16_matmul_tile(const uint16_t *act, const uint16_t *wt,
                             const uint8_t *scl, uint16_t *out)
{
    asm volatile("bias = mxmem2(%0)" :: "r"(scl) : "memory");
    asm volatile("mxclracc.hf" ::: "memory");
    asm volatile("{ activation.hf = mxmem(%0,%1)\n"
                 "  weight.hf     = mxmem(%2,%3) }"
                 :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
    asm volatile("mxmem(%0,%1):after.hf = acc" :: "r"(out), "r"(0) : "memory");
}

int main(void)
{
    unsigned int vtcm_base = h2_info(INFO_VTCM_BASE);
    if (!vtcm_base) { h2_thread_stop(1); return 1; }

    h2_vecaccess_state_t vacc;
    h2_vecaccess_unit_init(&vacc, H2_VECACCESS_HVX_128, CFG_TYPE_VXU0,
                           CFG_SUBTYPE_VXU0, CFG_HVX_CONTEXTS, 0x1);
    h2_vecaccess_acquire(&vacc);
    h2_mxaccess_state_t mxacc;
    h2_mxaccess_unit_init(&mxacc, CFG_TYPE_VXU0, CFG_SUBTYPE_VXU0,
                          CFG_HMX_CONTEXTS, 0x1);
    h2_mxaccess_acquire(&mxacc);

    uint8_t  *vt   = (uint8_t *)(unsigned long)vtcm_base;
    /* fp16 buffers */
    uint16_t *f16_act  = (uint16_t *)(vt + 0 * 2048);
    uint16_t *f16_wt   = (uint16_t *)(vt + 1 * 2048);
    uint8_t  *f16_scl  =              vt + 2 * 2048;
    uint16_t *f16_out  = (uint16_t *)(vt + 3 * 2048);
    /* int16 kernel VTCM workspace starts at offset 10 x 2048 to stay clear */
    void     *int16_vtcm = vt + 10 * 2048;
    /* int16 kernel args (normal RAM, stack) */
    static int16_t a16[1024], w16[1024], out16[1024];

    hexagon_sim_init_timer();

    /* ===== fp16 cycle measurements ===== */

    /* Prepare data once (outside timed region). */
    fill_f16(f16_act, F16_ONE);
    fill_f16(f16_wt,  F16_ONE);
    fill_scales_f16(f16_scl, F16_ONE);

    /* Warmup call to prime caches / mx contexts. */
    fp16_matmul_tile(f16_act, f16_wt, f16_scl, f16_out);

    printf("=== fp16 32x32x32 matmul ===\n");
    {
        uint64_t c0 = hexagon_sim_read_pcycles();
        fp16_matmul_tile(f16_act, f16_wt, f16_scl, f16_out);
        uint64_t c1 = hexagon_sim_read_pcycles();
        printf("  single tile (HMX only): %llu pcycles\n",
               (unsigned long long)(c1 - c0));
    }
    /* 10-iteration average to smooth out first-call / cache effects. */
    {
        uint64_t c0 = hexagon_sim_read_pcycles();
        for (int i = 0; i < 10; i++)
            fp16_matmul_tile(f16_act, f16_wt, f16_scl, f16_out);
        uint64_t c1 = hexagon_sim_read_pcycles();
        printf("  10x tile:               %llu pcycles  (avg %llu / tile)\n",
               (unsigned long long)(c1 - c0),
               (unsigned long long)((c1 - c0) / 10));
    }

    /* ===== int16 cycle measurements ===== */

    /* Initialize int16 inputs once (small known pattern). */
    for (int i = 0; i < 1024; i++) {
        a16[i] = (int16_t)((i * 37) & 0x7FFF);
        w16[i] = (int16_t)((i * 53) & 0x7FFF);
    }
    im_requant_t rq = { .mul = 1, .shift = 20 };

    /* Warmup. */
    im_matmul_hmx_i8(out16, a16, w16, rq, int16_vtcm);

    printf("\n=== int16 32x32x32 matmul (current kernel) ===\n");
    {
        uint64_t c0 = hexagon_sim_read_pcycles();
        im_matmul_hmx_i8(out16, a16, w16, rq, int16_vtcm);
        uint64_t c1 = hexagon_sim_read_pcycles();
        printf("  single tile (full kernel): %llu pcycles\n",
               (unsigned long long)(c1 - c0));
    }
    {
        uint64_t c0 = hexagon_sim_read_pcycles();
        for (int i = 0; i < 10; i++)
            im_matmul_hmx_i8(out16, a16, w16, rq, int16_vtcm);
        uint64_t c1 = hexagon_sim_read_pcycles();
        printf("  10x tile:                  %llu pcycles  (avg %llu / tile)\n",
               (unsigned long long)(c1 - c0),
               (unsigned long long)((c1 - c0) / 10));
    }

    /* ===== breakdown: time the int16 phases separately =====
     * We re-run the kernel but spy on phases by splitting them into
     * separate inline regions. Simpler: measure the inline asm-only
     * part (HMX packets) vs scalar part (decompose+combine).  For that
     * we call a stripped function that only does the 24 HMX packets
     * on pre-packed tiles. */

    printf("\n=== int16 HMX-only (4 partials, 24 packets) ===\n");
    {
        /* Use the same pre-packed f16 tiles as dummy u8/i8 — we only
         * care about cycle count, correctness doesn't matter here. */
        uint8_t  *act_tile = vt + 10 * 2048;
        int8_t   *wt_tile  = (int8_t *)(vt + 11 * 2048);
        uint16_t *bias_lo  = (uint16_t *)(vt + 12 * 2048);
        uint16_t *bias_hi  = (uint16_t *)(vt + 12 * 2048 + 256);
        uint16_t *out_lo   = (uint16_t *)(vt + 13 * 2048);
        uint16_t *out_hi   = (uint16_t *)(vt + 14 * 2048);

        memset(act_tile, 0, 2048);
        memset(wt_tile,  0, 1024);
        for (int i = 0; i < 128; i++) { bias_lo[i] = 0x4000; bias_hi[i] = 0x2000; }

        /* One partial = 6 HMX packets */
#define ONE_PARTIAL()                                                        \
    do {                                                                     \
        asm volatile("bias = mxmem(%0)" :: "r"(bias_lo) : "memory");         \
        asm volatile("mxclracc" ::: "memory");                               \
        asm volatile("{ activation.ub = mxmem(%0,%1)\n"                      \
                     "  weight.b      = mxmem(%2,%3) }"                      \
                     :: "r"(act_tile), "r"(2047),                            \
                        "r"(wt_tile),  "r"(2047) : "memory");                \
        asm volatile("mxmem(%0,%1):after:retain.uh = acc:2x1"                \
                     :: "r"(out_lo), "r"(0) : "memory");                     \
        asm volatile("bias = mxmem(%0)" :: "r"(bias_hi) : "memory");         \
        asm volatile("mxmem(%0,%1):after.uh = acc:2x1"                       \
                     :: "r"(out_hi), "r"(0) : "memory");                     \
    } while (0)

        /* Warmup */
        ONE_PARTIAL();

        uint64_t c0 = hexagon_sim_read_pcycles();
        for (int i = 0; i < 10; i++) {
            ONE_PARTIAL(); ONE_PARTIAL(); ONE_PARTIAL(); ONE_PARTIAL();
        }
        uint64_t c1 = hexagon_sim_read_pcycles();
        printf("  10x (4 partials each): %llu pcycles  (avg %llu / tile, %llu / partial)\n",
               (unsigned long long)(c1 - c0),
               (unsigned long long)((c1 - c0) / 10),
               (unsigned long long)((c1 - c0) / 40));
    }

    h2_thread_stop(0);
    return 0;
}
