/*
 * probe_cm_weight_layout.c — silicon RE for HMX `:cm` weight layout
 *
 * Agent A's earlier probe (probe_cm_row_major.c) proved that
 *   activation.ub = mxmem(p, 2047|0x1c):cm
 *   weight.b      = mxmem(q, 0x3FF)
 * consumes a ROW-MAJOR 1 KiB activation tile at 7.92 cyc/MAC. But that
 * probe used uniform all-1 activation and all-1 weight, so it could not
 * distinguish which weight-tile byte layout HMX actually expects.
 *
 * This probe varies weight patterns + weight layouts, with uniform act,
 * to pin down:
 *   1. Which weight layout gives CORRECT output under :cm + row-major
 *      activation?
 *   2. Is the K-accumulation actually 32 values per MAC as expected?
 *   3. How does weight[k][n] map to output[m][n]?
 *
 * Strategy:
 *   Act = all 1 (row-major 32x32). Then output[m][n] = sum_k w[k][n].
 *   For a "K-ramp" pattern w[k][0] = k+1 (k=0..31), rest 0:
 *     expected output[m][0] = 0+1+2+...+31+32 = 528 for all m
 *     expected output[m][n] = 0 for n != 0
 *   If we see 528 at column 0 → K-accumulation works end-to-end.
 *   If we see 528 at a DIFFERENT column → weight layout has N transposed.
 *   If we see partial sums (eg 16 or 276) → K-ordering is non-trivial.
 *
 * Layouts tested (weight is 32 K-rows x 32 N-cols, 1 KiB):
 *   RM   : row-major               tile[32*k + n] = w[k][n]
 *   P2   : Phase 2 4-K-row packed  tile[128*kg + 4*col + row4] = w[4kg+row4][col]
 *          where kg = 0..7, row4 = 0..3
 *   NT   : N-transposed row-major  tile[32*n + k] = w[k][n]
 *
 * All runs use Rt_a = 2047|0x1c, Rt_w = 0x3FF, :cm activation.
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

/* ----- activation: row-major all-ones 1 KiB (+ 1 KiB zero pad) ----- */
static void fill_act_rowmajor_ones(uint8_t *tile)
{
    memset(tile, 1, 1024);
    memset(tile + 1024, 0, 1024);
}

/* ----- weight layouts -----
 * Input: w[K=32][N=32] logical — fill pattern defined by caller.
 * Each layout lays the 1 KiB tile differently. */
typedef void (*w_layout_fn)(uint8_t *tile, const int8_t w[32][32]);

static void lay_rm(uint8_t *tile, const int8_t w[32][32])
{
    /* Row-major: tile[32*k + n] = w[k][n] */
    for (int k = 0; k < 32; k++)
        for (int n = 0; n < 32; n++)
            tile[32 * k + n] = (uint8_t)w[k][n];
}

static void lay_p2(uint8_t *tile, const int8_t w[32][32])
{
    /* Phase 2 4-K-row packed: per 128 B kg-group of 4 K-rows x 32 N-cols,
     * each 4-byte cell at offset (128*kg + 4*col) = [r0, r1, r2, r3]
     * where ri = w[4*kg + i][col]. */
    memset(tile, 0, 1024);
    for (int kg = 0; kg < 8; kg++) {
        for (int col = 0; col < 32; col++) {
            tile[128 * kg + 4 * col + 0] = (uint8_t)w[4 * kg + 0][col];
            tile[128 * kg + 4 * col + 1] = (uint8_t)w[4 * kg + 1][col];
            tile[128 * kg + 4 * col + 2] = (uint8_t)w[4 * kg + 2][col];
            tile[128 * kg + 4 * col + 3] = (uint8_t)w[4 * kg + 3][col];
        }
    }
}

static void lay_nt(uint8_t *tile, const int8_t w[32][32])
{
    /* N-outer transpose: tile[32*n + k] = w[k][n] */
    for (int n = 0; n < 32; n++)
        for (int k = 0; k < 32; k++)
            tile[32 * n + k] = (uint8_t)w[k][n];
}

/* ----- weight fill patterns -----
 * Each fn fills w[K][N] with a specific pattern. */
static void wp_single(int8_t w[32][32], int k0, int n0, int val)
{
    memset(w, 0, 32 * 32);
    w[k0][n0] = (int8_t)val;
}

static void wp_kramp(int8_t w[32][32], int n0)
{
    /* w[k][n0] = k+1 for k=0..31, else 0. Expected K-sum at col n0 = 528. */
    memset(w, 0, 32 * 32);
    for (int k = 0; k < 32; k++) w[k][n0] = (int8_t)(k + 1);
}

static void wp_nramp(int8_t w[32][32], int k0)
{
    /* w[k0][n] = n+1 for n=0..31, else 0. Expected at row k0: one-entry
     * col-scan; output[m][n] = w[k0][n] (single K contribution). */
    memset(w, 0, 32 * 32);
    for (int n = 0; n < 32; n++) w[k0][n] = (int8_t)(n + 1);
}

static void wp_all1(int8_t w[32][32])
{
    for (int k = 0; k < 32; k++)
        for (int n = 0; n < 32; n++) w[k][n] = 1;
}

/* 1-MAC dual-scale readback (plain-mxmem weight consumer). */
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

/* Read dual-scale readback (2-stream format) as int32 out[m][n].
 * readback layout for dual-scale :2x1 plain-mxmem is the Phase 2 format:
 *   out[phys_row * 64 + 2*n + stream] (stream 0..1 → rows phys_row and phys_row+16)
 * We decode for m=0..31, n=0..31. */
static void decode_out(const uint16_t *rb, int32_t out[32][32])
{
    for (int ir = 0; ir < 32; ir++) {
        int phys_row = ir & 15;
        int stream = ir >> 4;
        for (int jc = 0; jc < 32; jc++) {
            int idx = phys_row * 64 + 2 * jc + stream;
            uint16_t v = rb[idx];
            /* Plain single-scale readback (we used :after.uh, not :retain
             * + :after). The uint16 value is the raw low-16 of the int32
             * accumulator; for our tests K <= 528 so no sign/range issue. */
            out[ir][jc] = (int32_t)(int16_t)v;
        }
    }
}

/* Summarize output: total sum, positions of non-zero cells, sample rows. */
static void summarize(const char *tag, const int32_t out[32][32], int expect_val, int expect_col)
{
    long long total = 0;
    int nnz = 0;
    int first_nz_m = -1, first_nz_n = -1, first_nz_v = 0;
    int max_abs = 0;
    int row0_col0 = out[0][0];
    for (int m = 0; m < 32; m++)
        for (int n = 0; n < 32; n++) {
            int v = out[m][n];
            total += v;
            if (v != 0) {
                nnz++;
                int av = v < 0 ? -v : v;
                if (av > max_abs) max_abs = av;
                if (first_nz_m < 0) { first_nz_m = m; first_nz_n = n; first_nz_v = v; }
            }
        }

    /* Check expected column has the expected value on row 0. */
    int match = (expect_col >= 0)
              ? (out[0][expect_col] == expect_val)
              : 1;

    LOG("  %-32s nnz=%-4d total=%-8lld out[0][0]=%-5d first_nz=(%d,%d)=%d max|v|=%d match[exp_col=%d,val=%d]=%d",
        tag, nnz, total, row0_col0, first_nz_m, first_nz_n, first_nz_v, max_abs,
        expect_col, expect_val, match);

    /* Extra: dump row 0 fully if tag is verbose (marked with '!'). */
    if (tag[0] == '!') {
        LOG("    row 0:");
        for (int n = 0; n < 32; n += 8)
            LOG("      [%02d..%02d] %d %d %d %d %d %d %d %d",
                n, n+7,
                out[0][n+0], out[0][n+1], out[0][n+2], out[0][n+3],
                out[0][n+4], out[0][n+5], out[0][n+6], out[0][n+7]);
    }
}

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    g_out = fopen("./probe_cm_weight_layout_result.txt", "w");

    LOG("=== HMX :cm + row-major weight layout probe (SM8650 v75) ===");
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

    uint8_t  *act   = vtcm + 0;             /* 2 KiB (1 KiB row-major + pad) */
    uint8_t  *wt    = vtcm + 2048;          /* 1 KiB weight tile */
    uint16_t *bias  = (uint16_t *)(vtcm + 3072);  /* 256 B */
    uint16_t *rb    = (uint16_t *)(vtcm + 4096);  /* 2 KiB readback */

    fill_act_rowmajor_ones(act);
    for (int i = 0; i < 128; i++) bias[i] = 0x4000;   /* fp16 1.0 scale */

    LOG("[Init] act=+0 (1 KiB row-major all-1) wt=+2048 bias=+3072 rb=+4096");

    int8_t  w[32][32] __attribute__((aligned(128)));
    int32_t out[32][32] __attribute__((aligned(128)));

    const int rt_a = 2047 | 0x1c;
    const int rt_w = 0x3FF;

    /* Scoped helper to run one test.
     * Takes pattern (fills w), layout (arranges tile from w), expected
     * scalar result at (expect_row, expect_col). */
    struct { const char *name; w_layout_fn lay; } layouts[] = {
        {"RM ", lay_rm},
        {"P2 ", lay_p2},
        {"NT ", lay_nt},
    };

    /* ---- Test A: single-cell weight w[0][0]=1, act all 1 ----
     * Expected dot at output[m][0] = sum_k act[m][k]*w[k][0]
     *                               = 1*1 + 0*0 + ... = 1 (for any m)
     * (one non-zero weight at k=0,n=0 contributes 1 via act[m][0]=1.)
     * Expected other cells = 0. */
    LOG("");
    LOG("--- Test A: weight[0][0]=1, rest 0 — any layout should give out[m][0]=1 ---");
    for (unsigned L = 0; L < sizeof(layouts)/sizeof(layouts[0]); L++) {
        wp_single(w, 0, 0, 1);
        layouts[L].lay(wt, w);
        memset(rb, 0, 2048);
        RUN_CM(bias, act, wt, rt_a, rt_w, rb);
        decode_out(rb, out);
        char tag[64]; snprintf(tag, sizeof(tag), "A  %s w[0][0]=1", layouts[L].name);
        summarize(tag, out, 1, 0);
    }

    /* ---- Test A': single-cell at (k0, n0) — discriminates K-ordering
     * of each layout. If layout is faithful, moving the 1 to a different
     * K row should STILL produce out[m][0]=1 (because act[m][k]=1 for all k).
     * Changing N moves the output column. */
    LOG("");
    LOG("--- Test A': single w[k][n]=1 sweep (expect out[m][n]=1 only) ---");
    static const int k0_sweep[] = {0, 3, 7, 16, 31};
    static const int n0_sweep[] = {0, 5, 31};
    for (unsigned L = 0; L < sizeof(layouts)/sizeof(layouts[0]); L++) {
        for (unsigned ki = 0; ki < sizeof(k0_sweep)/sizeof(k0_sweep[0]); ki++) {
            for (unsigned ni = 0; ni < sizeof(n0_sweep)/sizeof(n0_sweep[0]); ni++) {
                int k0 = k0_sweep[ki];
                int n0 = n0_sweep[ni];
                wp_single(w, k0, n0, 1);
                layouts[L].lay(wt, w);
                memset(rb, 0, 2048);
                RUN_CM(bias, act, wt, rt_a, rt_w, rb);
                decode_out(rb, out);
                char tag[64];
                snprintf(tag, sizeof(tag), "A' %s k=%d n=%d", layouts[L].name, k0, n0);
                summarize(tag, out, 1, n0);
            }
        }
    }

    /* ---- Test B: K-ramp at n=0, w[k][0]=k+1, rest 0 ----
     * Expected out[m][0] = sum_{k=0..31} (k+1) = 528, rest 0. */
    LOG("");
    LOG("--- Test B: w[k][0]=k+1 (K-ramp) — correct K-sum = 528 at col 0 ---");
    for (unsigned L = 0; L < sizeof(layouts)/sizeof(layouts[0]); L++) {
        wp_kramp(w, 0);
        layouts[L].lay(wt, w);
        memset(rb, 0, 2048);
        RUN_CM(bias, act, wt, rt_a, rt_w, rb);
        decode_out(rb, out);
        char tag[64]; snprintf(tag, sizeof(tag), "!B %s K-ramp n0=0", layouts[L].name);
        summarize(tag, out, 528, 0);
    }

    /* ---- Test C: N-ramp at k=0, w[0][n]=n+1, rest 0 ----
     * Expected out[m][n] = act[m][0]*w[0][n] = (n+1), for all m, n=0..31. */
    LOG("");
    LOG("--- Test C: w[0][n]=n+1 (N-ramp at k=0) — single-K contribution ---");
    for (unsigned L = 0; L < sizeof(layouts)/sizeof(layouts[0]); L++) {
        wp_nramp(w, 0);
        layouts[L].lay(wt, w);
        memset(rb, 0, 2048);
        RUN_CM(bias, act, wt, rt_a, rt_w, rb);
        decode_out(rb, out);
        char tag[64]; snprintf(tag, sizeof(tag), "!C %s N-ramp k0=0", layouts[L].name);
        summarize(tag, out, 1, 0);  /* expect row 0 col 0 = 1 */
    }

    /* ---- Test D: reference — all-1 weight ----
     * Expected out[m][n] = 32 for all m, n (confirms 32 Ks consumed). */
    LOG("");
    LOG("--- Test D: all-ones weight — expected 32 everywhere ---");
    for (unsigned L = 0; L < sizeof(layouts)/sizeof(layouts[0]); L++) {
        wp_all1(w);
        layouts[L].lay(wt, w);
        memset(rb, 0, 2048);
        RUN_CM(bias, act, wt, rt_a, rt_w, rb);
        decode_out(rb, out);
        char tag[64]; snprintf(tag, sizeof(tag), "D  %s all-1", layouts[L].name);
        summarize(tag, out, 32, 0);
    }

    LOG("");
    LOG("=== done ===");

    HAP_compute_res_hmx_unlock(ctx_id);
    HAP_compute_res_release(ctx_id);
    if (g_out) fclose(g_out);
    return 0;
}
