/*
 * probe_cm_readback.c — silicon RE: how does the dual-scale `:after.uh
 * acc:2x1` readback map accumulator cells to readback bytes when the
 * MAC was issued via `:cm` row-major activation?
 *
 * Background: V3 (Phase 2 2-stream activation + plain mxmem MAC)
 * decodes its 2 KiB dual-scale readback as
 *   idx       = phys_row*64 + 2*jc + stream
 *   out[m][n] = ((int16)hi[idx] << 8) | (lo[idx] & 0xFF)
 *   where m = phys_row + 16*stream, phys_row = m & 15, stream = m >> 4.
 * V4 (`:cm` row-major + P2 weight) reuses the same decode and produces
 * garbage. This probe pins down the actual mapping.
 *
 * Three sub-tests:
 *   T1 (row-code, single readback): act[r][0] = r+1, weight[0][0] = 1,
 *                                   else 0. Expected acc[m][0] = m+1, rest 0.
 *                                   Dump all 1024 lo[idx] values and locate
 *                                   the indices that hold 1..32 → row→idx
 *                                   mapping for n=0.
 *   T2 (col-code, single readback): act[0][0] = 1, weight[0][n] = n+1,
 *                                   else 0. Expected acc[0][n] = n+1, rest 0.
 *                                   Locate indices that hold 1..32 → col→idx
 *                                   mapping for m=0.
 *   T3 (dual readback, large value): pattern producing acc[m][n] up to ~10000
 *                                   (single int16 OK, but exercises both lo
 *                                   and hi). Compare V3 decode and several
 *                                   candidate alternative decodes against the
 *                                   reference int32 values.
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

/* Row-major 1 KiB act tile + 1 KiB zero pad. */
static void fill_act_rm(uint8_t *tile, uint8_t (*pat)(int r, int k))
{
    memset(tile, 0, 2048);
    for (int r = 0; r < 32; r++)
        for (int k = 0; k < 32; k++)
            tile[32 * r + k] = pat(r, k);
}

/* Phase 2 4-K-row packed weight tile. */
static void fill_wt_p2(uint8_t *tile, int8_t (*pat)(int k, int n))
{
    memset(tile, 0, 1024);
    for (int kg = 0; kg < 8; kg++) {
        for (int col = 0; col < 32; col++) {
            tile[128 * kg + 4 * col + 0] = (uint8_t)pat(4 * kg + 0, col);
            tile[128 * kg + 4 * col + 1] = (uint8_t)pat(4 * kg + 1, col);
            tile[128 * kg + 4 * col + 2] = (uint8_t)pat(4 * kg + 2, col);
            tile[128 * kg + 4 * col + 3] = (uint8_t)pat(4 * kg + 3, col);
        }
    }
}

/* Patterns. Use file-scope globals because pat fns can't take params. */
static int g_row, g_col;

static uint8_t pat_act_rowcode(int r, int k)  { return (k == 0) ? (uint8_t)(r + 1) : 0; }
static int8_t  pat_wt_unitNW (int k, int n)   { return (k == 0 && n == 0) ? 1 : 0; }
static uint8_t pat_act_unitNW(int r, int k)   { return (r == 0 && k == 0) ? 1 : 0; }
static int8_t  pat_wt_colcode(int k, int n)   { return (k == 0) ? (int8_t)(n + 1) : 0; }
/* For T3: act all-1 (32 rows × 32 K), weight[k][n] = ((n+1) << some-bits). */
static uint8_t pat_act_all1  (int r, int k)   { (void)r; (void)k; return 1; }
static int8_t  pat_wt_t3     (int k, int n)   {
    /* Yields acc[m][n] = sum_k w[k][n] for any m. We want some values to
     * exceed int16 / int8-low byte to exercise dual-scale combine.
     * Simple choice: w[k][n] = (n+1) for k=0..3 (4 K-rows); else 0.
     * → acc[m][n] = 4*(n+1). Range [4, 128]. Fits int8 readback bytes too —
     * not yet exercising dual-scale. Bump to k=0..15 → acc=16(n+1). Range
     * [16, 512]. Still within int16. K=0..31 (all rows) → acc=32(n+1).
     * Range [32, 1024]. */
    (void)k;
    return (int8_t)(n + 1);
}

#define BIAS_LO 0x4000  /* fp16 1.0 — used for the LO dual-scale readback */
#define BIAS_HI 0x2000  /* fp16 0.5 — used for the HI dual-scale readback */

static void fill_bias(uint16_t *b, uint16_t v)
{
    for (int i = 0; i < 128; i++) b[i] = v;
}

#define RT_A    (2047 | 0x1c)
#define RT_W    0x3FF

/* ---- Single :cm MAC + single :2x1 readback (no :retain). ---- */
#define RUN_CM_SINGLE(bias_p, act, wt, out_p)                              \
    do {                                                                   \
        asm volatile("bias = mxmem(%0)" :: "r"(bias_p) : "memory");        \
        asm volatile("mxclracc" ::: "memory");                             \
        asm volatile("{ activation.ub = mxmem(%0,%1):cm\n"                 \
                     "  weight.b      = mxmem(%2,%3) }"                    \
                     :: "r"(act), "r"(RT_A), "r"(wt), "r"(RT_W)            \
                     : "memory");                                          \
        asm volatile("mxmem(%0,%1):after.uh = acc:2x1"                     \
                     :: "r"(out_p), "r"(0) : "memory");                    \
    } while (0)

/* ---- Single :cm MAC + dual readback (V3-style). ---- */
#define RUN_CM_DUAL(bias_lo_p, bias_hi_p, act, wt, out_lo_p, out_hi_p)     \
    do {                                                                   \
        asm volatile("bias = mxmem(%0)" :: "r"(bias_lo_p) : "memory");     \
        asm volatile("mxclracc" ::: "memory");                             \
        asm volatile("{ activation.ub = mxmem(%0,%1):cm\n"                 \
                     "  weight.b      = mxmem(%2,%3) }"                    \
                     :: "r"(act), "r"(RT_A), "r"(wt), "r"(RT_W)            \
                     : "memory");                                          \
        asm volatile("mxmem(%0,%1):after:retain.uh = acc:2x1"              \
                     :: "r"(out_lo_p), "r"(0) : "memory");                 \
        asm volatile("bias = mxmem(%0)" :: "r"(bias_hi_p) : "memory");     \
        asm volatile("mxmem(%0,%1):after.uh = acc:2x1"                     \
                     :: "r"(out_hi_p), "r"(0) : "memory");                 \
    } while (0)

/* Locate the indices in rb[0..1023] holding the values 1..32. Reports
 * how many indices each value lives at, and the LOW idx for each. */
static void find_distinct_values(const uint16_t *rb, const char *tag, int max_val)
{
    LOG("  [%s] scan rb[0..1023] for values 1..%d:", tag, max_val);
    for (int v = 1; v <= max_val; v++) {
        int n_hits = 0;
        int first_idx = -1;
        for (int i = 0; i < 1024; i++) {
            if ((int16_t)rb[i] == v) {
                if (first_idx < 0) first_idx = i;
                n_hits++;
            }
        }
        LOG("    val=%-3d hits=%-3d first_idx=%d  (idx mod 64 = %d, idx/64 = %d, idx mod 2 = %d)",
            v, n_hits, first_idx,
            first_idx >= 0 ? first_idx % 64 : -1,
            first_idx >= 0 ? first_idx / 64 : -1,
            first_idx >= 0 ? first_idx & 1 : -1);
    }
    /* Total nonzero count. */
    int nnz = 0; long long total = 0;
    for (int i = 0; i < 1024; i++) {
        int v = (int16_t)rb[i];
        if (v != 0) { nnz++; total += v; }
    }
    LOG("    nnz=%d  total=%lld", nnz, total);
}

/* Try several candidate decoders against expected values, count matches.
 * Decoders take (lo, hi) and produce int32. */
typedef int32_t (*decoder_fn)(uint16_t lo, uint16_t hi);

static int32_t dec_v3_combine (uint16_t lo, uint16_t hi) { return ((int32_t)(int16_t)hi << 8) | ((int32_t)lo & 0xFF); }
static int32_t dec_lo_only_s  (uint16_t lo, uint16_t hi) { (void)hi; return (int32_t)(int16_t)lo; }
static int32_t dec_lo_only_u  (uint16_t lo, uint16_t hi) { (void)hi; return (int32_t)(uint16_t)lo; }
static int32_t dec_hi_only_s  (uint16_t lo, uint16_t hi) { (void)lo; return (int32_t)(int16_t)hi; }
static int32_t dec_lo_plus_hi (uint16_t lo, uint16_t hi) { return (int32_t)(int16_t)lo + ((int32_t)(int16_t)hi << 16); }
static int32_t dec_hi_lshift8 (uint16_t lo, uint16_t hi) { return ((int32_t)(int16_t)hi << 8) + (int32_t)(int16_t)lo; }

static void test_decoders_against_expected(
    const uint16_t *rb_lo, const uint16_t *rb_hi,
    /* expected[m][n] under V3 idx formula or whatever fmt we adopt */
    int (*expected)(int m, int n),
    int (*idx_fn)(int m, int n),
    const char *tag)
{
    static const struct { const char *n; decoder_fn f; } dcs[] = {
        {"V3combine ((s16)hi<<8)|(lo&0xFF) ", dec_v3_combine},
        {"lo_only signed                   ", dec_lo_only_s},
        {"lo_only unsigned                 ", dec_lo_only_u},
        {"hi_only signed                   ", dec_hi_only_s},
        {"lo+hi<<16                        ", dec_lo_plus_hi},
        {"hi<<8 + lo                       ", dec_hi_lshift8},
    };
    LOG("  [%s] decoder match counts (expected vs decoded over 32x32 cells):", tag);
    for (unsigned d = 0; d < sizeof(dcs)/sizeof(dcs[0]); d++) {
        int match = 0; int total = 0;
        int first_diff_m = -1, first_diff_n = -1;
        int first_diff_exp = 0, first_diff_got = 0;
        for (int m = 0; m < 32; m++)
            for (int n = 0; n < 32; n++) {
                int idx = idx_fn(m, n);
                if (idx < 0 || idx >= 1024) continue;
                int got = dcs[d].f(rb_lo[idx], rb_hi[idx]);
                int exp = expected(m, n);
                total++;
                if (got == exp) match++;
                else if (first_diff_m < 0) {
                    first_diff_m = m; first_diff_n = n;
                    first_diff_exp = exp; first_diff_got = got;
                }
            }
        LOG("    %s match=%d/%d  first diff at (m=%d,n=%d): exp=%d got=%d",
            dcs[d].n, match, total, first_diff_m, first_diff_n,
            first_diff_exp, first_diff_got);
    }
}

/* V3-style index formula. */
static int idx_v3(int m, int n) { int phys_row = m & 15, stream = m >> 4; return phys_row * 64 + 2 * n + stream; }

/* Linear (m, n) -> m*32 + n (as if rb were just row-major). */
static int idx_lin(int m, int n) { return m * 32 + n; }

/* Reverse stream: stream = m & 1, phys_row = m >> 1. */
static int idx_alt1(int m, int n) { int phys_row = m >> 1, stream = m & 1; return phys_row * 64 + 2 * n + stream; }

/* T3 expected: acc[m][n] = 32*(n+1) with all-1 act and pat_wt_t3. */
static int exp_t3(int m, int n) { (void)m; return 32 * (n + 1); }

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    g_out = fopen("./probe_cm_readback_result.txt", "w");

    LOG("=== HMX :cm readback layout probe (SM8650 v75) ===");
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

    uint8_t  *act     = vtcm + 0;             /* 2 KiB row-major + pad */
    uint8_t  *wt      = vtcm + 2048;          /* 1 KiB weight tile */
    uint16_t *bias_lo = (uint16_t *)(vtcm + 3072);
    uint16_t *bias_hi = (uint16_t *)(vtcm + 3328);
    uint16_t *rb_lo   = (uint16_t *)(vtcm + 4096);
    uint16_t *rb_hi   = (uint16_t *)(vtcm + 6144);

    fill_bias(bias_lo, BIAS_LO);
    fill_bias(bias_hi, BIAS_HI);

    LOG("[Init] act=+0  wt=+2048  bias_lo=+3072  bias_hi=+3328  rb_lo=+4096  rb_hi=+6144");

    /* ---- T1: row-code (acc[m][0] = m+1, rest 0) ---- */
    LOG("");
    LOG("--- T1: row-code: act[r][0]=r+1 (else 0), wt[0][0]=1 (else 0). Expect acc[m][0]=m+1 ---");
    fill_act_rm(act, pat_act_rowcode);
    fill_wt_p2(wt,  pat_wt_unitNW);
    memset(rb_lo, 0, 2048);
    RUN_CM_SINGLE(bias_lo, act, wt, rb_lo);
    find_distinct_values(rb_lo, "T1 row-code", 32);

    /* ---- T2: col-code (acc[0][n] = n+1, rest 0) ---- */
    LOG("");
    LOG("--- T2: col-code: act[0][0]=1 (else 0), wt[0][n]=n+1 (else 0). Expect acc[0][n]=n+1 ---");
    fill_act_rm(act, pat_act_unitNW);
    fill_wt_p2(wt,  pat_wt_colcode);
    memset(rb_lo, 0, 2048);
    RUN_CM_SINGLE(bias_lo, act, wt, rb_lo);
    find_distinct_values(rb_lo, "T2 col-code", 32);

    /* ---- T3: full pattern, dual readback, test V3 + alt decoders ---- */
    LOG("");
    LOG("--- T3: act all-1, wt[k][n]=n+1 for all k. Expect acc[m][n]=32*(n+1) for all m ---");
    fill_act_rm(act, pat_act_all1);
    fill_wt_p2(wt,  pat_wt_t3);
    memset(rb_lo, 0, 2048);
    memset(rb_hi, 0, 2048);
    RUN_CM_DUAL(bias_lo, bias_hi, act, wt, rb_lo, rb_hi);

    LOG("  T3 lo dump (idx 0..63):");
    for (int i = 0; i < 64; i += 8)
        LOG("    [%02d..%02d] %5d %5d %5d %5d %5d %5d %5d %5d", i, i+7,
            (int16_t)rb_lo[i+0], (int16_t)rb_lo[i+1], (int16_t)rb_lo[i+2], (int16_t)rb_lo[i+3],
            (int16_t)rb_lo[i+4], (int16_t)rb_lo[i+5], (int16_t)rb_lo[i+6], (int16_t)rb_lo[i+7]);
    LOG("  T3 hi dump (idx 0..63):");
    for (int i = 0; i < 64; i += 8)
        LOG("    [%02d..%02d] %5d %5d %5d %5d %5d %5d %5d %5d", i, i+7,
            (int16_t)rb_hi[i+0], (int16_t)rb_hi[i+1], (int16_t)rb_hi[i+2], (int16_t)rb_hi[i+3],
            (int16_t)rb_hi[i+4], (int16_t)rb_hi[i+5], (int16_t)rb_hi[i+6], (int16_t)rb_hi[i+7]);

    LOG("");
    LOG("  T3 expected acc[m][n] = 32*(n+1), m irrelevant. n=0..31 → 32, 64, 96, ..., 1024");
    LOG("  T3 try V3 idx formula:");
    test_decoders_against_expected(rb_lo, rb_hi, exp_t3, idx_v3, "T3 idx_v3");
    LOG("  T3 try linear m*32+n:");
    test_decoders_against_expected(rb_lo, rb_hi, exp_t3, idx_lin, "T3 idx_lin");
    LOG("  T3 try alt1 stream=m&1, phys=m>>1:");
    test_decoders_against_expected(rb_lo, rb_hi, exp_t3, idx_alt1, "T3 idx_alt1");

    /* ---- T4: same as T3 but SINGLE readback (no :retain), see if we get
     * the right value with just lo (since 1024 < 32K fits int16). ---- */
    LOG("");
    LOG("--- T4: T3 pattern with single readback (lo only) ---");
    memset(rb_lo, 0, 2048);
    RUN_CM_SINGLE(bias_lo, act, wt, rb_lo);
    LOG("  T4 lo dump (idx 0..63):");
    for (int i = 0; i < 64; i += 8)
        LOG("    [%02d..%02d] %5d %5d %5d %5d %5d %5d %5d %5d", i, i+7,
            (int16_t)rb_lo[i+0], (int16_t)rb_lo[i+1], (int16_t)rb_lo[i+2], (int16_t)rb_lo[i+3],
            (int16_t)rb_lo[i+4], (int16_t)rb_lo[i+5], (int16_t)rb_lo[i+6], (int16_t)rb_lo[i+7]);
    LOG("  T4 try V3 idx formula, lo-only signed:");
    test_decoders_against_expected(rb_lo, rb_lo, exp_t3, idx_v3, "T4 idx_v3 lo");
    LOG("  T4 try linear m*32+n, lo-only signed:");
    test_decoders_against_expected(rb_lo, rb_lo, exp_t3, idx_lin, "T4 idx_lin lo");

    LOG("");
    LOG("=== done ===");

    HAP_compute_res_hmx_unlock(ctx_id);
    HAP_compute_res_release(ctx_id);
    if (g_out) fclose(g_out);

    /* silence unused-fn warnings */
    (void)g_row; (void)g_col;
    return 0;
}
