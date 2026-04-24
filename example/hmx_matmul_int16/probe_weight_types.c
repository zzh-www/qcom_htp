/*
 * probe_weight_types.c — measure HMX cycles across weight widths to
 * verify the "sub-byte MAC" architecture hypothesis.
 *
 * Hypothesis (from Agent/hmx_u8xi8_matmul_layers.md §0.5.8/§0.5.9):
 *   one byte of weight decomposes into N sub-byte values, HMX produces
 *   N MACs per cell per cycle. Expected throughput ratios vs weight.b:
 *     weight.b  (int8): 1x
 *     weight.n  (int4): 2x
 *     weight.n:2x     : 4x (2-byte wide load)
 *     weight.c  (int2): 4x
 *     weight.ubit(int1): 8x
 *     weight.hf (fp16): different path (xfp), baseline ~1x
 *
 * Experiment: back-to-back MAC packets using raw asm for each weight
 * type. Inputs are all-zero buffers — we only care about cycle count,
 * not correctness.
 *
 * Measurement: hexagon_sim_read_pcycles() around 100-iter loop. If the
 * ISS models HMX pipeline (it exposes HMX_INT8_OPS_PER_CYCLE stats so
 * it should), pcycles should decrease proportional to sub-byte count.
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

#define ITERS 100

/* Emit one full MAC sequence: clear, bias, act+wt, readback.
 * We use raw asm so we can vary the weight.* suffix and :2x modifier. */

#define MAC_BODY(WT_SUFFIX, WT_MOD)                                 \
    asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");       \
    asm volatile("mxclracc" ::: "memory");                          \
    asm volatile("{ activation.ub = mxmem(%0, %1)\n"                \
                 "  weight." WT_SUFFIX "  = mxmem(%2, %3)" WT_MOD " }" \
                 :: "r"(act), "r"(2047),                            \
                    "r"(wt),  "r"(2047) : "memory");                \
    asm volatile("mxmem(%0, %1):after.uh = acc:2x1"                 \
                 :: "r"(out), "r"(0) : "memory")

static void do_wt_b(const uint8_t *act, const int8_t *wt,
                    const uint16_t *bias, uint16_t *out, int n)
{
    for (int i = 0; i < n; i++) { MAC_BODY("b", ""); }
}
static void do_wt_n(const uint8_t *act, const int8_t *wt,
                    const uint16_t *bias, uint16_t *out, int n)
{
    for (int i = 0; i < n; i++) { MAC_BODY("n", ""); }
}
static void do_wt_n_2x(const uint8_t *act, const int8_t *wt,
                       const uint16_t *bias, uint16_t *out, int n)
{
    for (int i = 0; i < n; i++) { MAC_BODY("n", ":2x"); }
}
static void do_wt_c(const uint8_t *act, const int8_t *wt,
                    const uint16_t *bias, uint16_t *out, int n)
{
    for (int i = 0; i < n; i++) { MAC_BODY("c", ""); }
}
static void do_wt_ubit(const uint8_t *act, const int8_t *wt,
                       const uint16_t *bias, uint16_t *out, int n)
{
    for (int i = 0; i < n; i++) { MAC_BODY("ubit", ""); }
}

/* fp16 path uses a different clear + convert. */
static void do_wt_hf(const uint16_t *act, const uint16_t *wt,
                     const uint8_t *scl, uint16_t *out, int n)
{
    for (int i = 0; i < n; i++) {
        asm volatile("bias = mxmem2(%0)" :: "r"(scl) : "memory");
        asm volatile("mxclracc.hf" ::: "memory");
        asm volatile("{ activation.hf = mxmem(%0, %1)\n"
                     "  weight.hf     = mxmem(%2, %3) }"
                     :: "r"(act), "r"(2047),
                        "r"(wt),  "r"(2047) : "memory");
        asm volatile("mxmem(%0, %1):after.hf = acc"
                     :: "r"(out), "r"(0) : "memory");
    }
}

static void fill_splat(void *p, int bytes, uint32_t v)
{
    HVX_Vector vv = Q6_V_vsplat_R(v);
    for (int i = 0; i < bytes / 128; i++) ((HVX_Vector *)p)[i] = vv;
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

    uint8_t  *vt       = (uint8_t *)(unsigned long)vtcm_base;
    uint8_t  *act      = vt + 0 * 4096;   /* 2 KiB act tile   */
    int8_t   *wt       = (int8_t *)(vt + 1 * 4096); /* up to 2 KiB wt */
    uint16_t *bias_int = (uint16_t *)(vt + 2 * 4096);
    uint8_t  *scl_fp   = vt + 2 * 4096;    /* aliases bias for fp path */
    uint16_t *out      = (uint16_t *)(vt + 3 * 4096);

    /* All-zero buffers — only cycle count matters. */
    memset(act, 0, 2048);
    memset(wt,  0, 2048);
    memset(out, 0, 2048);
    for (int i = 0; i < 128; i++) bias_int[i] = 0x4000;  /* f16 2.0 */
    ((HVX_Vector *)scl_fp)[0] = Q6_V_vsplat_R((0x3C00 << 16) | 0x3C00);
    ((HVX_Vector *)scl_fp)[1] = Q6_V_vzero();

    hexagon_sim_init_timer();

    /* Warmup each path once. */
    do_wt_b(act, wt, bias_int, out, 1);
    do_wt_n(act, wt, bias_int, out, 1);
    do_wt_n_2x(act, wt, bias_int, out, 1);
    do_wt_c(act, wt, bias_int, out, 1);
    do_wt_ubit(act, wt, bias_int, out, 1);
    do_wt_hf((uint16_t *)act, (uint16_t *)wt, scl_fp, out, 1);

    printf("=== HMX weight-type throughput probe (K=32, %d MAC packets per run) ===\n", ITERS);
    printf("%-14s %-12s %-8s %-16s\n", "weight type", "pcycles", "cyc/MAC", "ratio vs .b");

    uint64_t c0, c1, baseline = 0;

#define RUN(NAME, FN)                                                          \
    do {                                                                       \
        c0 = hexagon_sim_read_pcycles();                                       \
        FN(act, wt, bias_int, out, ITERS);                                     \
        c1 = hexagon_sim_read_pcycles();                                       \
        uint64_t cycles = c1 - c0;                                             \
        if (baseline == 0) baseline = cycles;                                  \
        printf("%-14s %-12llu %-8.2f %.3fx\n", NAME,                           \
               (unsigned long long)cycles,                                     \
               (double)cycles / (double)ITERS,                                 \
               (double)baseline / (double)cycles);                             \
    } while (0)

    RUN("weight.b",     do_wt_b);
    RUN("weight.n",     do_wt_n);
    RUN("weight.n:2x",  do_wt_n_2x);
    RUN("weight.c",     do_wt_c);
    RUN("weight.ubit",  do_wt_ubit);

    /* fp16 path has different args — time separately. */
    c0 = hexagon_sim_read_pcycles();
    do_wt_hf((uint16_t *)act, (uint16_t *)wt, scl_fp, out, ITERS);
    c1 = hexagon_sim_read_pcycles();
    {
        uint64_t cycles = c1 - c0;
        printf("%-14s %-12llu %-8.2f %.3fx (xfp path)\n", "weight.hf",
               (unsigned long long)cycles,
               (double)cycles / (double)ITERS,
               (double)baseline / (double)cycles);
    }

    printf("\nExpected ratios (if sub-byte MAC hypothesis holds):\n");
    printf("  weight.b   : 1.00x  (baseline)\n");
    printf("  weight.n   : 2.00x  (1 byte -> 2 nibbles)\n");
    printf("  weight.n:2x: 4.00x  (2x wider load)\n");
    printf("  weight.c   : 4.00x  (1 byte -> 4 crumbs)\n");
    printf("  weight.ubit: 8.00x  (1 byte -> 8 bits)\n");

    h2_thread_stop(0);
    return 0;
}
