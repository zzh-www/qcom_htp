/*
 * Probe P1: dual-scale readback via `:retain`.
 *
 * Core question: can we recover a full int32 accumulator by issuing ONE
 * MAC packet and then TWO convert-stores on the same acc, with different
 * biases — the first convert uses `:after:retain.uh=acc:2x1` (acc
 * survives), the second uses `:after.uh=acc:2x1` (standard)?
 *
 * If yes, the full int16 matmul can run with 4 MACs + 8 converts per
 * 32x32x32 tile instead of today's 128 MACs.
 *
 * Expected formula (to be verified):
 *    output_lo = acc mod 2^16           (bias scale = 1.0)
 *    output_hi = floor(acc / 2^16)      (bias scale = 2^-16)
 *    int32 acc = (int16)output_hi * 65536 + (uint16)output_lo
 *                  [with carry correction if needed]
 *
 * Test pinning:
 *    A = a_val (u8 broadcast), W = w_val (i8 broadcast), K = 32
 *    acc = 32 * a_val * w_val per output cell.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>
#include <hexagon_types.h>

#define DO_CLRACC()  asm volatile("mxclracc" ::: "memory")
#define DO_MM(A, W)  asm volatile("{ activation.ub = mxmem(%0,%1)\n" \
                                  "  weight.b     = mxmem(%2,%3) }"  \
                                  :: "r"(A), "r"(2047), "r"(W), "r"(2047) : "memory")
#define DO_BIAS(P)   asm volatile("bias = mxmem(%0)" :: "r"(P) : "memory")
#define DO_ST_R(O)   asm volatile("mxmem(%0,%1):after:retain.uh = acc:2x1" \
                                  :: "r"(O), "r"(0) : "memory")
#define DO_ST(O)     asm volatile("mxmem(%0,%1):after.uh = acc:2x1" \
                                  :: "r"(O), "r"(0) : "memory")

/* Fill all 512 u16s (128 cols × 4 slots) with `v`. Splat layout. */
static void fill_bias(uint16_t *b, uint16_t v)
{
    int splat = ((int)v << 16) | (int)v;
    HVX_Vector x = Q6_V_vsplat_R(splat);
    for (int i = 0; i < 32; i++) ((HVX_Vector *)b)[i] = x;
}

/* Fill bias so ONLY slot 0 of each column carries the f16 scale.
 * Slots 1, 2, 3 = 0. This lets us isolate the effect of slot 0. */
static void fill_bias_slot0(uint16_t *b, uint16_t v)
{
    memset(b, 0, 1024);                   /* 128 cols × 4 slots × 2 bytes */
    for (int col = 0; col < 128; col++) {
        b[col * 4 + 0] = v;
    }
}

/* Pack 32x32 activation at all 32 K slots.
 * For this probe, all 32 logical rows get the same value in every K. */
static void pack_act_full(uint8_t *tile, uint8_t v)
{
    memset(tile, 0, 2048);
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        for (int K = 0; K < 32; K++) {
            tile[128 * phys_row + 4 * K + 1] = v;  /* stream 0 */
            tile[128 * phys_row + 4 * K + 3] = v;  /* stream 1 */
        }
    }
}

/* Pack 32x32 weight at all 32 K slots, same value per cell. */
static void pack_wt_full(int8_t *tile, int8_t v)
{
    memset(tile, 0, 1024);
    for (int K = 0; K < 32; K++) {
        for (int col = 0; col < 32; col++) {
            tile[128 * (K >> 2) + 4 * col + (K & 3)] = v;
        }
    }
}

/* Logical (ir=0, jc=0) lives at phys_row 0, stream 0, col 0
 *   → index 0*64 + 2*0 + 0 = 0 in the u16 output tile. */
static uint16_t read_cell(const uint16_t *out, int ir, int jc)
{
    int phys_row = ir & 15;
    int stream   = ir >> 4;
    return out[phys_row * 64 + 2 * jc + stream];
}

/* Two bias-fill modes: "splat" (all 512 u16 = v, the old behavior) and
 * "slot0" (only slot 0 of each column = v, slots 1/2/3 = 0).
 * If `mode` == 1, use splat; if 0, use slot0-only. */
static void do_trial(const char *label,
                     uint8_t a_val, int8_t w_val,
                     uint16_t bias_lo_f16, uint16_t bias_hi_f16,
                     int mode,
                     uint8_t *A, int8_t *W,
                     uint16_t *BIAS, uint16_t *OUT_LO, uint16_t *OUT_HI)
{
    pack_act_full(A, a_val);
    pack_wt_full(W, w_val);

    int32_t acc = 32 * (int32_t)a_val * (int32_t)w_val;
    uint16_t want_lo = (uint16_t)(acc & 0xFFFF);
    uint16_t want_hi = (uint16_t)((acc >> 16) & 0xFFFF);

    printf("--- %s (mode=%s) ---\n", label, mode ? "splat" : "slot0");
    printf("  a=%u w=%d acc=%ld=0x%08x  lo=0x%04x hi=0x%04x\n",
           a_val, w_val, (long)acc, (uint32_t)acc, want_lo, want_hi);

    memset(OUT_LO, 0xEE, 2048);
    memset(OUT_HI, 0xEE, 2048);

    if (mode) fill_bias(BIAS, bias_lo_f16); else fill_bias_slot0(BIAS, bias_lo_f16);
    DO_CLRACC();
    DO_BIAS(BIAS);
    DO_MM(A, W);
    DO_ST_R(OUT_LO);

    if (mode) fill_bias(BIAS, bias_hi_f16); else fill_bias_slot0(BIAS, bias_hi_f16);
    DO_BIAS(BIAS);
    DO_ST(OUT_HI);

    uint16_t got_lo = read_cell(OUT_LO, 0, 0);
    uint16_t got_hi = read_cell(OUT_HI, 0, 0);

    printf("  bias_lo=0x%04x → OUT_LO[0,0] = 0x%04x  (want 0x%04x)  %s\n",
           bias_lo_f16, got_lo, want_lo,
           got_lo == want_lo ? "OK" : "MISS");
    printf("  bias_hi=0x%04x → OUT_HI[0,0] = 0x%04x  (want 0x%04x)  diff=%ld\n",
           bias_hi_f16, got_hi, want_hi,
           (long)((int32_t)got_hi - (int32_t)want_hi));

    int32_t recon = ((int32_t)((int16_t)got_hi) << 16) | got_lo;
    printf("  recon = %ld  (want %ld)  %s\n\n",
           (long)recon, (long)acc, recon == acc ? "BIT-EXACT" : "MISMATCH");
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

    uint8_t  *vt     = (uint8_t *)(unsigned long)vtcm_base;
    uint8_t  *A      =           vt + 0 * 2048;
    int8_t   *W      = (int8_t *)(vt + 1 * 2048);
    uint16_t *BIAS   = (uint16_t*)(vt + 2 * 2048);
    uint16_t *OUT_LO = (uint16_t*)(vt + 4 * 2048);
    uint16_t *OUT_HI = (uint16_t*)(vt + 6 * 2048);

    printf("=== P1 dual-scale readback ===\n\n");

    /* Compare splat (slots 0-3 all = scale) vs slot0-only (slots 1-3 = 0)
     * to diagnose whether slots 1-3 perturb the convert. */

    do_trial("acc=32, bias_hi=0x0200",
             1, 1, 0x4000, 0x0200, 1, A, W, BIAS, OUT_LO, OUT_HI);
    do_trial("acc=32, bias_hi=0x0200",
             1, 1, 0x4000, 0x0200, 0, A, W, BIAS, OUT_LO, OUT_HI);

    do_trial("acc=160000, bias_hi=0x0200",
             100, 50, 0x4000, 0x0200, 1, A, W, BIAS, OUT_LO, OUT_HI);
    do_trial("acc=160000, bias_hi=0x0200",
             100, 50, 0x4000, 0x0200, 0, A, W, BIAS, OUT_LO, OUT_HI);

    do_trial("acc=1036320, bias_hi=0x0200",
             255, 127, 0x4000, 0x0200, 1, A, W, BIAS, OUT_LO, OUT_HI);
    do_trial("acc=1036320, bias_hi=0x0200",
             255, 127, 0x4000, 0x0200, 0, A, W, BIAS, OUT_LO, OUT_HI);

    /* slot0 mode with different bias_hi values — find cleanest scale. */
    do_trial("acc=160000, bias_hi=0x0400",
             100, 50, 0x4000, 0x0400, 0, A, W, BIAS, OUT_LO, OUT_HI);
    do_trial("acc=160000, bias_hi=0x0800",
             100, 50, 0x4000, 0x0800, 0, A, W, BIAS, OUT_LO, OUT_HI);
    do_trial("acc=160000, bias_hi=0x3C00 (f16 1.0 → scale 0.5)",
             100, 50, 0x4000, 0x3C00, 0, A, W, BIAS, OUT_LO, OUT_HI);

    /* The winner: bias_hi=0x2000 = f16 2^-7 (normal) → scale 2^-8.
     * OUT_B should equal (acc >> 8) mod 2^16, clean floor.
     * Reconstruct: acc = ((int16)OUT_B << 8) | (OUT_A & 0xFF). */
    printf("=== scale 2^-8 winner check ===\n\n");
    do_trial("acc=160000, bias_hi=0x2000 (f16 2^-7, scale 2^-8)",
             100, 50, 0x4000, 0x2000, 0, A, W, BIAS, OUT_LO, OUT_HI);
    do_trial("acc=1036320, bias_hi=0x2000",
             255, 127, 0x4000, 0x2000, 0, A, W, BIAS, OUT_LO, OUT_HI);
    do_trial("acc=-3200, bias_hi=0x2000",
             100, -1, 0x4000, 0x2000, 0, A, W, BIAS, OUT_LO, OUT_HI);
    do_trial("acc=-1044480, bias_hi=0x2000",
             255, -128, 0x4000, 0x2000, 0, A, W, BIAS, OUT_LO, OUT_HI);
    do_trial("acc=32, bias_hi=0x2000",
             1, 1, 0x4000, 0x2000, 0, A, W, BIAS, OUT_LO, OUT_HI);

    printf("=== Reconstruction test ===\n");
    {
        /* Print dedicated reconstruction based on (acc>>8) hi read. */
        struct { uint8_t a; int8_t w; } cases[] = {
            {100, 50}, {255, 127}, {100, -1}, {255, -128}, {1, 1}, {128, 127}
        };
        for (unsigned t = 0; t < sizeof(cases)/sizeof(cases[0]); t++) {
            uint8_t a = cases[t].a; int8_t w = cases[t].w;
            pack_act_full(A, a);
            pack_wt_full(W, w);
            int32_t acc = 32 * (int32_t)a * (int32_t)w;

            fill_bias_slot0(BIAS, 0x4000);
            DO_CLRACC(); DO_BIAS(BIAS); DO_MM(A, W); DO_ST_R(OUT_LO);
            fill_bias_slot0(BIAS, 0x2000);
            DO_BIAS(BIAS); DO_ST(OUT_HI);

            uint16_t out_a = read_cell(OUT_LO, 0, 0);
            uint16_t out_b = read_cell(OUT_HI, 0, 0);
            int32_t recon = ((int32_t)(int16_t)out_b << 8)
                          | ((uint32_t)out_a & 0xFFu);
            printf("  a=%3u w=%4d acc=%8ld  OUT_A=0x%04x OUT_B=0x%04x  recon=%ld  %s\n",
                   a, w, (long)acc, out_a, out_b, (long)recon,
                   recon == acc ? "OK" : "FAIL");
        }
    }

    /* Negative accs in slot0 mode. */
    do_trial("acc=-3200, bias_hi=0x0200",
             100, -1, 0x4000, 0x0200, 0, A, W, BIAS, OUT_LO, OUT_HI);
    do_trial("acc=-1044480, bias_hi=0x0200",
             255, -128, 0x4000, 0x0200, 0, A, W, BIAS, OUT_LO, OUT_HI);

    printf("=== done ===\n");
    h2_thread_stop(0);
    return 0;
}
