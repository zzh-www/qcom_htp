/*
 * probe_pair_lane.c — T12: pair-lane bias interaction under non-trivial acc.
 *
 * Context: T9/T9b established that under UNIFORM acc, col c reads bias[2c+1]
 * and lane 2c is ignored.  V8 integration test with per-col-varied bias +
 * varied act/wt showed col 0 bit-exact but cols 1..31 off (silicon_acc ≈
 * ½·ref_acc or worse).  T12 sweeps (bias[2c], bias[2c+1]) pairings under
 * non-trivial acc to decode how the pair actually interacts.
 *
 * Setup for all tests:
 *   act[r][k] = (r + 3k) % 128     (varied, range 0..127)
 *   wt[k][n]  = ((k + 5n) % 7) - 3 (varied, range [-3, 3])
 *   acc[m][n] = Σ_k act[m][k] × wt[k][n]  (non-trivial, per-cell unique)
 *
 * For each test we dump one row of silicon output (m=0) and compare to
 * several candidate formulas.
 *
 * T12a — bias[2c] = 0x0000, bias[2c+1] = fp16(2 + 0.05*c) (varying, >=2.0)
 *        Tests: does the "even lane = 0" cause silicon to use a zero scale?
 *
 * T12b — bias[2c] = bias[2c+1] = fp16(2 + 0.05*c)    (mirrored)
 *        This is V8's current layout.  Tests: does mirroring fully fix it?
 *
 * T12c — bias[2c] = fp16(2 + 0.05*c), bias[2c+1] = 0x0000
 *        Reverse mirror.  Tests: if even lane carries the scale, this
 *        would give varying contributions; if odd lane carries it, flat.
 *
 * T12d — bias[2c] = fp16(2.0) uniform, bias[2c+1] = fp16(2 + 0.05*c) varying
 *        Tests: separates baseline-source (bias[2c+1]) from scale-source.
 *        If scale comes from bias[2c], silicon would give uniform
 *        contribution per col (bias=2.0); varied baseline only.
 *
 * T12e — bias[2c] = fp16(2 + 0.05*c) varying, bias[2c+1] = fp16(2.0) uniform
 *        Dual of T12d.  If baseline from odd lane: uniform baseline (128).
 *        Varied scale would show varied acc contribution.
 *
 * T12f — bias[2c] = fp16(4.0) uniform, bias[2c+1] = fp16(2.0) uniform
 *        Cleanest test: both lanes nonzero uniform but DIFFERENT values.
 *        If silicon averages or swaps, output clearly flags it.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
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

#define RT_A    (2047 | 0x1C)
#define RT_W    0x3FF

#define RUN_CM_MAC_SAT_UB(bias_p, act, wt, out_p)                          \
    do {                                                                   \
        asm volatile("bias = mxmem(%0)" :: "r"(bias_p) : "memory");        \
        asm volatile("mxclracc" ::: "memory");                             \
        asm volatile("{ activation.ub = mxmem(%0,%1):cm\n"                 \
                     "  weight.b      = mxmem(%2,%3) }"                    \
                     :: "r"(act), "r"(RT_A), "r"(wt), "r"(RT_W)            \
                     : "memory");                                          \
        asm volatile("mxmem(%0,%1):after:cm:sat.ub = acc"                  \
                     :: "r"(out_p), "r"(0) : "memory");                    \
    } while (0)

static float fp16_to_float(uint16_t h)
{
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    float v;
    if (exp == 0)        v = ldexpf((float)mant, -24);
    else if (exp == 0x1F) v = 0.0f;
    else                  v = ldexpf((float)(1024 + mant), (int)exp - 25);
    return sign ? -v : v;
}

static uint16_t float_to_fp16(float f)
{
    union { float f; uint32_t u; } v = { f };
    uint32_t sign = (v.u >> 31) & 1;
    int32_t  exp  = ((v.u >> 23) & 0xFF) - 127;
    uint32_t mant = v.u & 0x7FFFFF;
    if (exp <= -15) return (uint16_t)(sign << 15);
    if (exp >= 16)  return (uint16_t)((sign << 15) | 0x7C00);
    return (uint16_t)((sign << 15) | (((exp + 15) & 0x1F) << 10) | (mant >> 13));
}

/* Fill 1 KiB row-major activation tile. */
static void fill_act(uint8_t *tile)
{
    memset(tile, 0, 2048);
    for (int r = 0; r < 32; r++)
        for (int k = 0; k < 32; k++)
            tile[32 * r + k] = (uint8_t)((r + 3 * k) % 128);  /* 0..127 */
}

/* Fill P2 4-K-row × 32-col weight tile. */
static void fill_wt(uint8_t *tile)
{
    memset(tile, 0, 1024);
    for (int kg = 0; kg < 8; kg++)
        for (int n = 0; n < 32; n++)
            for (int k4 = 0; k4 < 4; k4++) {
                int k = 4 * kg + k4;
                int8_t v = (int8_t)(((k + 5 * n) % 7) - 3);   /* [-3, 3] */
                tile[128 * kg + 4 * n + k4] = (uint8_t)v;
            }
}

/* Compute scalar reference acc for each (m, n) cell. */
static void compute_ref_acc(int32_t acc_out[32][32])
{
    for (int m = 0; m < 32; m++) {
        for (int n = 0; n < 32; n++) {
            int32_t s = 0;
            for (int k = 0; k < 32; k++) {
                int a = (m + 3 * k) % 128;
                int w = ((k + 5 * n) % 7) - 3;
                s += a * w;
            }
            acc_out[m][n] = s;
        }
    }
}

/* Silicon-formula prediction from probe_hmx_formula.c (baseline + floor scale). */
static int predict(int32_t acc, uint16_t bias_raw_baseline, uint16_t bias_raw_scale)
{
    int baseline = (int)(bias_raw_baseline >> 7);
    float bv = fp16_to_float(bias_raw_scale);
    float scaled = (float)acc * bv / 512.0f;
    int sv = (int)floorf(scaled);
    int v = baseline + sv;
    if (v < 0) v = 0;
    else if (v > 255) v = 255;
    return v;
}

static void compare_col0_detail(const uint8_t *out,
                                const int32_t acc_ref[32][32],
                                const uint16_t bias[128],
                                const char *tag)
{
    LOG("  [%s] m=0 per-col observed vs predictions:", tag);
    LOG("    c  obs   pr_2c+1   pr_2c     pr_(b=2c+1,s=2c)   pr_(b=2c,s=2c+1)   acc[0][c]");
    for (int c = 0; c < 32; c++) {
        int obs = out[c];
        int p1 = predict(acc_ref[0][c], bias[2*c+1], bias[2*c+1]);
        int p2 = predict(acc_ref[0][c], bias[2*c],   bias[2*c]);
        int p3 = predict(acc_ref[0][c], bias[2*c+1], bias[2*c]);
        int p4 = predict(acc_ref[0][c], bias[2*c],   bias[2*c+1]);
        LOG("    %2d  %3d   %3d       %3d       %3d               %3d               %d",
            c, obs, p1, p2, p3, p4, acc_ref[0][c]);
    }
}

/* Tally how many cells match each candidate formula across (m=0..31, c=0..31). */
static void count_formula_match(const uint8_t *out,
                                const int32_t acc_ref[32][32],
                                const uint16_t bias[128],
                                const char *tag)
{
    int m1=0, m2=0, m3=0, m4=0;   /* 4 candidates */
    for (int m = 0; m < 32; m++) {
        for (int c = 0; c < 32; c++) {
            int obs = out[32 * m + c];
            int p1 = predict(acc_ref[m][c], bias[2*c+1], bias[2*c+1]);
            int p2 = predict(acc_ref[m][c], bias[2*c],   bias[2*c]);
            int p3 = predict(acc_ref[m][c], bias[2*c+1], bias[2*c]);
            int p4 = predict(acc_ref[m][c], bias[2*c],   bias[2*c+1]);
            if (obs == p1) m1++;
            if (obs == p2) m2++;
            if (obs == p3) m3++;
            if (obs == p4) m4++;
        }
    }
    LOG("  [%s] match counts (out of 1024):", tag);
    LOG("    [b=2c+1, s=2c+1]: %4d", m1);
    LOG("    [b=2c,   s=2c  ]: %4d", m2);
    LOG("    [b=2c+1, s=2c  ]: %4d", m3);
    LOG("    [b=2c,   s=2c+1]: %4d", m4);
}

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    g_out = fopen("./probe_pair_lane_result.txt", "w");

    LOG("=== T12: :cm:sat.ub bias pair-lane probe (SM8650 v75) ===");
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

    uint8_t  *act  = vtcm + 0;
    uint8_t  *wt   = vtcm + 2048;
    uint16_t *bias = (uint16_t *)(vtcm + 3072);
    uint8_t  *out  = vtcm + 4096;

    /* Fill act, wt. */
    fill_act(act);
    fill_wt(wt);

    /* Compute scalar ref acc. */
    static int32_t acc_ref[32][32];
    compute_ref_acc(acc_ref);
    LOG("Sanity: acc_ref[0][0]=%d, acc_ref[0][1]=%d, acc_ref[0][15]=%d, acc_ref[15][15]=%d",
        acc_ref[0][0], acc_ref[0][1], acc_ref[0][15], acc_ref[15][15]);

    /* Build bias values: fp16(2 + 0.05*c) for c in 0..31 (range 2.0..3.55). */
    uint16_t bias_varied[32];
    for (int c = 0; c < 32; c++) bias_varied[c] = float_to_fp16(2.0f + 0.05f * (float)c);

    /* ---- T12a: lane 2c=0, lane 2c+1=varied ---- */
    LOG("");
    LOG("--- T12a: bias[2c]=0, bias[2c+1]=fp16(2+0.05c) ---");
    for (int i = 0; i < 128; i++) bias[i] = 0;
    for (int c = 0; c < 32; c++) bias[2*c + 1] = bias_varied[c];
    memset(out, 0xAA, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    count_formula_match(out, acc_ref, bias, "T12a");
    compare_col0_detail(out, acc_ref, bias, "T12a");

    /* ---- T12b: both lanes = varied (mirrored) ---- */
    LOG("");
    LOG("--- T12b: bias[2c]=bias[2c+1]=fp16(2+0.05c) (mirrored, V8 current) ---");
    for (int i = 0; i < 128; i++) bias[i] = 0;
    for (int c = 0; c < 32; c++) {
        bias[2*c]     = bias_varied[c];
        bias[2*c + 1] = bias_varied[c];
    }
    memset(out, 0xBB, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    count_formula_match(out, acc_ref, bias, "T12b");
    compare_col0_detail(out, acc_ref, bias, "T12b");

    /* ---- T12c: lane 2c=varied, lane 2c+1=0 ---- */
    LOG("");
    LOG("--- T12c: bias[2c]=fp16(2+0.05c), bias[2c+1]=0 ---");
    for (int i = 0; i < 128; i++) bias[i] = 0;
    for (int c = 0; c < 32; c++) bias[2*c] = bias_varied[c];
    memset(out, 0xCC, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    count_formula_match(out, acc_ref, bias, "T12c");
    compare_col0_detail(out, acc_ref, bias, "T12c");

    /* ---- T12d: lane 2c=2.0 uniform, lane 2c+1=varied ---- */
    LOG("");
    LOG("--- T12d: bias[2c]=fp16(2.0), bias[2c+1]=fp16(2+0.05c) ---");
    for (int i = 0; i < 128; i++) bias[i] = 0;
    uint16_t bias_2_0 = float_to_fp16(2.0f);
    for (int c = 0; c < 32; c++) {
        bias[2*c]     = bias_2_0;
        bias[2*c + 1] = bias_varied[c];
    }
    memset(out, 0xDD, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    count_formula_match(out, acc_ref, bias, "T12d");
    compare_col0_detail(out, acc_ref, bias, "T12d");

    /* ---- T12e: lane 2c=varied, lane 2c+1=2.0 uniform ---- */
    LOG("");
    LOG("--- T12e: bias[2c]=fp16(2+0.05c), bias[2c+1]=fp16(2.0) ---");
    for (int i = 0; i < 128; i++) bias[i] = 0;
    for (int c = 0; c < 32; c++) {
        bias[2*c]     = bias_varied[c];
        bias[2*c + 1] = bias_2_0;
    }
    memset(out, 0xEE, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    count_formula_match(out, acc_ref, bias, "T12e");
    compare_col0_detail(out, acc_ref, bias, "T12e");

    /* ---- T12f: lane 2c=4.0 uniform, lane 2c+1=2.0 uniform (both nonzero, different) ---- */
    LOG("");
    LOG("--- T12f: bias[2c]=fp16(4.0), bias[2c+1]=fp16(2.0) ---");
    for (int i = 0; i < 128; i++) bias[i] = 0;
    uint16_t bias_4_0 = float_to_fp16(4.0f);
    for (int c = 0; c < 32; c++) {
        bias[2*c]     = bias_4_0;
        bias[2*c + 1] = bias_2_0;
    }
    memset(out, 0xFA, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    count_formula_match(out, acc_ref, bias, "T12f");
    compare_col0_detail(out, acc_ref, bias, "T12f");

    /* ---- T12g: lane 2c=2.0, lane 2c+1=4.0 (swapped from T12f) ---- */
    LOG("");
    LOG("--- T12g: bias[2c]=fp16(2.0), bias[2c+1]=fp16(4.0) ---");
    for (int i = 0; i < 128; i++) bias[i] = 0;
    for (int c = 0; c < 32; c++) {
        bias[2*c]     = bias_2_0;
        bias[2*c + 1] = bias_4_0;
    }
    memset(out, 0xFB, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    count_formula_match(out, acc_ref, bias, "T12g");
    compare_col0_detail(out, acc_ref, bias, "T12g");

    LOG("");
    LOG("=== done ===");

    HAP_compute_res_hmx_unlock(ctx_id);
    HAP_compute_res_release(ctx_id);
    if (g_out) fclose(g_out);
    return 0;
}
