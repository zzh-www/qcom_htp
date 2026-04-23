/*
 * probe_cm_row_major.c — can HMX `:cm` consume a row-major byte tile?
 *
 * Strategic question (Phase 3 kernel design): QNN's ForceFormat_Crouton_b
 * lays out int8 MatMul activations row-major (32 rows x 32 bytes = 1024 B
 * contiguous). Our Phase 2 kernel uses a 2-stream interleaved 2 KiB tile.
 * Prior RE (Agent/qnn_hmx_pipelining.md) showed that `:cm` on the 2-stream
 * tile produces HALF the expected MAC output (16 vs 32) — suggesting `:cm`
 * consumes only one stream's worth per MAC.
 *
 * HYPOTHESIS: on a ROW-MAJOR activation tile, `:cm` may natively consume
 * the full 32x32 layout with no HVX pre-pack — which would mean QNN uses
 * `:cm` exactly because its activation storage is row-major Crouton_b.
 *
 * Scenarios:
 *   A: 2-stream interleaved + plain mxmem       (baseline, expect 32)
 *   B: row-major layout     + plain mxmem       (layout mismatch, garbage)
 *   C: row-major layout     + :cm  mxmem        (KEY TEST — does :cm eat row-major?)
 *   D: Crouton_8 mini       + :cm  mxmem        (8 real rows, 24 pad)
 *   E: row-major layout     + :cm  with Rt sweep (does Rt encoding matter?)
 *
 * For each scenario: issue 1 MAC (functional check) and N MACs (cyc/MAC).
 * Check out[0], out[2], out[4] (first three 16-bit dual-scale lo bytes for
 * m=0, n=0,1,2) to see if output is uniform across columns.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <hexagon_types.h>
#include "HAP_farf.h"
#include "HAP_compute_res.h"
#include "HAP_power.h"
#include "HAP_perf.h"

static FILE *g_out;
#define LOG(...) do { \
    FARF(ALWAYS, __VA_ARGS__); \
    if (g_out) { fprintf(g_out, __VA_ARGS__); fprintf(g_out, "\n"); fflush(g_out); } \
} while (0)

static int power_ctx;
static int power_on_hvx_hmx(void)
{
    HAP_power_request_t req;
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_apptype;
    req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    if (HAP_power_set((void *)&power_ctx, &req) != 0) return -1;

    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_DCVS_v3;
    req.dcvs_v3.set_dcvs_enable    = 1;
    req.dcvs_v3.dcvs_enable        = 1;
    req.dcvs_v3.dcvs_option        = HAP_DCVS_V2_PERFORMANCE_MODE;
    req.dcvs_v3.set_bus_params     = 1;
    req.dcvs_v3.bus_params.min_corner    = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.max_corner    = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_core_params    = 1;
    req.dcvs_v3.core_params.min_corner    = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.max_corner    = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_sleep_disable  = 1;
    req.dcvs_v3.sleep_disable      = 1;
    if (HAP_power_set((void *)&power_ctx, &req) != 0) return -2;

    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HVX;
    req.hvx.power_up = 1;
    if (HAP_power_set((void *)&power_ctx, &req) != 0) return -3;

    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HMX;
    req.hmx.power_up = 1;
    if (HAP_power_set((void *)&power_ctx, &req) != 0) return -4;
    return 0;
}

/* 2-stream tile (2 KiB): byte at [128*phys_row + 4*k + {1,3}] = val, else 0. */
static void fill_act_twostream(uint8_t *tile, uint8_t val)
{
    memset(tile, 0, 2048);
    for (int phys_row = 0; phys_row < 16; phys_row++)
        for (int k = 0; k < 32; k++) {
            tile[128 * phys_row + 4 * k + 1] = val;
            tile[128 * phys_row + 4 * k + 3] = val;
        }
}

/* Row-major 1 KiB: 32 rows x 32 bytes contiguous (row r byte k at 32*r+k). */
static void fill_act_rowmajor(uint8_t *tile, uint8_t val)
{
    memset(tile, val, 1024);
    /* Zero the upper 1 KiB in case HMX reads a 2 KiB window. */
    memset(tile + 1024, 0, 1024);
}

/* Crouton_8-like: 8 rows x 32 bytes real data, rest zero (in 2 KiB region). */
static void fill_act_crouton8(uint8_t *tile, uint8_t val)
{
    memset(tile, 0, 2048);
    memset(tile, val, 8 * 32);
}

static void fill_wt(int8_t *tile, int8_t val) { memset(tile, val, 1024); }
static void fill_bias(uint16_t *b, uint16_t v) { for (int i = 0; i < 128; i++) b[i] = v; }

/* 1-packet MAC + dual-scale readback. */
#define RUN_PLAIN(bias, act, wt, rt_a, rt_w, out)                         \
    do {                                                                  \
        asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");         \
        asm volatile("mxclracc" ::: "memory");                            \
        asm volatile("{ activation.ub = mxmem(%0,%1)\n"                   \
                     "  weight.b      = mxmem(%2,%3) }"                   \
                     :: "r"(act), "r"(rt_a), "r"(wt), "r"(rt_w)           \
                     : "memory");                                         \
        asm volatile("mxmem(%0,%1):after.uh = acc:2x1"                    \
                     :: "r"(out), "r"(0) : "memory");                     \
    } while (0)

#define RUN_CM(bias, act, wt, rt_a, rt_w, out)                            \
    do {                                                                  \
        asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");         \
        asm volatile("mxclracc" ::: "memory");                            \
        asm volatile("{ activation.ub = mxmem(%0,%1):cm\n"                \
                     "  weight.b      = mxmem(%2,%3) }"                   \
                     :: "r"(act), "r"(rt_a), "r"(wt), "r"(rt_w)           \
                     : "memory");                                         \
        asm volatile("mxmem(%0,%1):after.uh = acc:2x1"                    \
                     :: "r"(out), "r"(0) : "memory");                     \
    } while (0)

/* N-MAC tight loops for cyc/MAC measurement. */
static inline void acc_plain(const uint16_t *bias, const uint8_t *act,
                              const int8_t *wt, uint16_t *out, int n,
                              int rt_a, int rt_w)
{
    asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
    asm volatile("mxclracc" ::: "memory");
    for (int i = 0; i < n; i++) {
        asm volatile("{ activation.ub = mxmem(%0,%1)\n"
                     "  weight.b      = mxmem(%2,%3) }"
                     :: "r"(act), "r"(rt_a), "r"(wt), "r"(rt_w) : "memory");
    }
    asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
                 :: "r"(out), "r"(0) : "memory");
}

static inline void acc_cm(const uint16_t *bias, const uint8_t *act,
                           const int8_t *wt, uint16_t *out, int n,
                           int rt_a, int rt_w)
{
    asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
    asm volatile("mxclracc" ::: "memory");
    for (int i = 0; i < n; i++) {
        asm volatile("{ activation.ub = mxmem(%0,%1):cm\n"
                     "  weight.b      = mxmem(%2,%3) }"
                     :: "r"(act), "r"(rt_a), "r"(wt), "r"(rt_w) : "memory");
    }
    asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
                 :: "r"(out), "r"(0) : "memory");
}

/* Helper: log out[0..5] which is dual-scale lo-halves of m=0, n=0..2. */
static void dump_out(const char *tag, const uint16_t *out)
{
    LOG("    %-40s out[m=0,n=0]=%u  n=1=%u  n=2=%u  n=3=%u  n=8=%u  n=16=%u",
        tag, out[0], out[2], out[4], out[6], out[16], out[32]);
}

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    g_out = fopen("./probe_cm_row_major_result.txt", "w");

    LOG("=== HMX :cm + row-major probe (SM8650 v75) ===");
    if (power_on_hvx_hmx()) { LOG("[Power] FAILED"); return 1; }

    unsigned int vtcm_size = 8 * 1024 * 1024;
    HAP_compute_res_query_VTCM(0, &vtcm_size, NULL, NULL, NULL);
    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_vtcm_param(&attr, vtcm_size, 1);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);
    unsigned int ctx_id = HAP_compute_res_acquire(&attr, 100000);
    if (!ctx_id) { LOG("[Init] acquire FAIL"); return 1; }
    uint8_t *vtcm = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr);
    if (!vtcm) { LOG("[Init] NULL vtcm"); return 1; }
    if (HAP_compute_res_hmx_lock(ctx_id)) { LOG("[Init] hmx_lock FAIL"); return 1; }

    /* Use 3 activation slots so we can preload A/B/C layouts and reuse. */
    uint8_t  *act_2s  = vtcm + 0;           /* 2 KiB */
    uint8_t  *act_rm  = vtcm + 2048;        /* 2 KiB (1 KiB data + 1 KiB zero) */
    uint8_t  *act_cr8 = vtcm + 4096;        /* 2 KiB */
    int8_t   *wt      = (int8_t  *)(vtcm + 6144);   /* 1 KiB */
    uint16_t *bias    = (uint16_t *)(vtcm + 7168);  /* 256 B */
    uint16_t *out     = (uint16_t *)(vtcm + 8192);  /* 2 KiB */

    fill_act_twostream(act_2s, 1);
    fill_act_rowmajor (act_rm, 1);
    fill_act_crouton8 (act_cr8, 1);
    fill_wt(wt, 1);
    fill_bias(bias, 0x4000);  /* fp16 1.0 → lo-byte readback = raw int */

    LOG("[Init] VTCM=%p  act_2s=+0  act_rm=+2048  act_cr8=+4096  wt=+6144  bias=+7168  out=+8192",
        vtcm);
    LOG("[Init] activations all 1; weight all +1; bias 0x4000 (scale 1.0)");

    /* ---------------- Part 1: functional (1-packet) ---------------- */
    LOG("");
    LOG("--- Part 1: 1-MAC functional check (expected: 32 for correct A=W=1 dot) ---");

    /* A: 2-stream + plain */
    memset(out, 0, 2048);
    RUN_PLAIN(bias, act_2s, wt, 2047, 2047, out);
    dump_out("A  2-stream + plain", out);

    /* B: row-major + plain */
    memset(out, 0, 2048);
    RUN_PLAIN(bias, act_rm, wt, 2047, 2047, out);
    dump_out("B  row-major + plain", out);

    /* C: row-major + :cm (THE KEY TEST) */
    memset(out, 0, 2048);
    RUN_CM(bias, act_rm, wt, 2047 | 0x1c, 0x3ff, out);
    dump_out("C  row-major + :cm  (QNN pattern)", out);

    memset(out, 0, 2048);
    RUN_CM(bias, act_rm, wt, 2047, 0x3ff, out);
    dump_out("C' row-major + :cm  Rt_a=2047", out);

    memset(out, 0, 2048);
    RUN_CM(bias, act_rm, wt, 2047 | 0x1c, 2047, out);
    dump_out("C'' row-major + :cm  Rt_w=2047", out);

    /* D: Crouton_8 mini + :cm */
    memset(out, 0, 2048);
    RUN_CM(bias, act_cr8, wt, 2047 | 0x1c, 0x3ff, out);
    dump_out("D  crouton_8 + :cm  (8 real rows)", out);

    /* E: row-major + :cm with Rt sweep */
    LOG("");
    LOG("--- Part 1b: row-major + :cm, Rt_a sweep (Rt_w=0x3FF) ---");
    static const int rt_sweep[] = {
        0x000, 0x010, 0x01c, 0x020, 0x0ff, 0x3ff, 0x7ff,
        2047, 2047|0x04, 2047|0x08, 2047|0x10, 2047|0x1c,
    };
    for (unsigned i = 0; i < sizeof(rt_sweep)/sizeof(rt_sweep[0]); i++) {
        memset(out, 0, 2048);
        RUN_CM(bias, act_rm, wt, rt_sweep[i], 0x3ff, out);
        char tag[64];
        snprintf(tag, sizeof(tag), "E  row-major +:cm Rt_a=0x%03x", rt_sweep[i]);
        dump_out(tag, out);
    }

    /* Baseline reference: 2-stream + :cm (prior RE result was 16) */
    memset(out, 0, 2048);
    RUN_CM(bias, act_2s, wt, 2047 | 0x1c, 0x3ff, out);
    dump_out("REF 2-stream + :cm  (prior RE = 16)", out);

    /* ---------------- Part 2: throughput ---------------- */
    LOG("");
    LOG("--- Part 2: N-MAC throughput (cyc/MAC) ---");

    const int N_WARMUP = 16;
    const int N_MEAS   = 400;

    /* Use weight=0 during throughput to avoid accumulator saturation. */
    fill_wt(wt, 0);

    uint64_t c0, c1;

    acc_plain(bias, act_2s, wt, out, N_WARMUP, 2047, 2047);
    c0 = HAP_perf_get_pcycles();
    acc_plain(bias, act_2s, wt, out, N_MEAS, 2047, 2047);
    c1 = HAP_perf_get_pcycles();
    LOG("  A  2-stream + plain Rt=2047,2047    : %10llu pcyc  %.3f cyc/MAC",
        (unsigned long long)(c1 - c0), (double)(c1 - c0) / N_MEAS);

    acc_plain(bias, act_rm, wt, out, N_WARMUP, 2047, 2047);
    c0 = HAP_perf_get_pcycles();
    acc_plain(bias, act_rm, wt, out, N_MEAS, 2047, 2047);
    c1 = HAP_perf_get_pcycles();
    LOG("  B  row-major + plain Rt=2047,2047   : %10llu pcyc  %.3f cyc/MAC",
        (unsigned long long)(c1 - c0), (double)(c1 - c0) / N_MEAS);

    acc_cm(bias, act_rm, wt, out, N_WARMUP, 2047 | 0x1c, 0x3ff);
    c0 = HAP_perf_get_pcycles();
    acc_cm(bias, act_rm, wt, out, N_MEAS, 2047 | 0x1c, 0x3ff);
    c1 = HAP_perf_get_pcycles();
    LOG("  C  row-major + :cm Rt_a=2063,0x3FF  : %10llu pcyc  %.3f cyc/MAC",
        (unsigned long long)(c1 - c0), (double)(c1 - c0) / N_MEAS);

    acc_cm(bias, act_cr8, wt, out, N_WARMUP, 2047 | 0x1c, 0x3ff);
    c0 = HAP_perf_get_pcycles();
    acc_cm(bias, act_cr8, wt, out, N_MEAS, 2047 | 0x1c, 0x3ff);
    c1 = HAP_perf_get_pcycles();
    LOG("  D  crouton_8 + :cm  Rt_a=2063,0x3FF : %10llu pcyc  %.3f cyc/MAC",
        (unsigned long long)(c1 - c0), (double)(c1 - c0) / N_MEAS);

    acc_cm(bias, act_2s, wt, out, N_WARMUP, 2047 | 0x1c, 0x3ff);
    c0 = HAP_perf_get_pcycles();
    acc_cm(bias, act_2s, wt, out, N_MEAS, 2047 | 0x1c, 0x3ff);
    c1 = HAP_perf_get_pcycles();
    LOG("  REF 2-stream + :cm Rt_a=2063,0x3FF  : %10llu pcyc  %.3f cyc/MAC",
        (unsigned long long)(c1 - c0), (double)(c1 - c0) / N_MEAS);

    LOG("");
    LOG("=== done ===");

    HAP_compute_res_hmx_unlock(ctx_id);
    HAP_compute_res_release(ctx_id);
    if (g_out) fclose(g_out);
    return 0;
}
