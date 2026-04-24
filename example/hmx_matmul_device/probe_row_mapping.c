/*
 * probe_row_mapping.c — T11: disambiguate :cm output-row <-> activation-row
 * mapping.  From T7d we saw that row-ramp activation (act[r][k]=r+1) gives
 * out_rows 0..15 = 120 and out_rows 16..31 = 121 under bias=1.0.  That's
 * incompatible with straight "out[m] uses phys[m]" at K=32.
 *
 * Strategy:
 *   T11a — single-row activation: only phys_row r is 1, all others 0.
 *          Predicts: with naive mapping, only out[r] is nonzero. Observe
 *          the actual set of nonzero out rows.
 *
 *   T11b — half-row activation: phys_row r = {1 iff r<16 else 0}. Combined
 *          with uniform wt, quickly reveals whether each out half draws
 *          from each act half.
 *
 *   T11c — dual polarity: phys_row r = (r<16 ? 0 : 1). Complement of T11b.
 *
 *   T11d — row-encoded activation: act[r][k] = r (no +1, so row 0 is 0 and
 *          acc[0] = 0).  Look at out[m] vs m plot to map m_out → r_act.
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

/* Wt = 1 for all (k,n) in P2 layout. */
static void fill_wt_all1(uint8_t *tile)
{
    memset(tile, 0, 1024);
    for (int kg = 0; kg < 8; kg++)
        for (int n = 0; n < 32; n++)
            for (int k4 = 0; k4 < 4; k4++)
                tile[128 * kg + 4 * n + k4] = 1;
}

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    g_out = fopen("./probe_row_mapping_result.txt", "w");

    LOG("=== T11: :cm out-row vs phys-row mapping (SM8650 v75) ===");
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

    /* Bias: uniform 0x4000 (2.0) so per-col baseline = 128. Per-col scale 2/512=1/256. */
    for (int i = 0; i < 128; i++) bias[i] = 0x4000;

    /* wt all-1 */
    fill_wt_all1(wt);

    /* ---------------------------------------------------------------- */
    /* T11a: per-row sweep — light up one phys row at a time.           */
    /* ---------------------------------------------------------------- */
    LOG("--- T11a: act[r_lit][k]=1 for ONE r_lit, all others 0. ---");
    LOG("          acc on the true out row should be K*1*1 = 32.");
    LOG("          With bias=2.0: (0x4000>>7) + trunc(32*2/512) = 128 + 0 = 128.");
    LOG("          All other out rows should be 128 too (baseline only). Not informative — use hi-K amplifier.");
    LOG("");
    LOG("          So instead we set act[r_lit][k]=255 (max) → acc = 255*32 = 8160");
    LOG("          trunc(8160*2/512) = trunc(31.875) = 31 → out = 128+31 = 159 on the hit row.");
    LOG("          Hit row(s) tell us mapping. Idle rows → 128.");

    for (int r_lit = 0; r_lit < 32; r_lit++) {
        memset(act, 0, 2048);
        for (int k = 0; k < 32; k++) act[32 * r_lit + k] = 255;
        memset(out, 0xAA, 1024);
        RUN_CM_MAC_SAT_UB(bias, act, wt, out);
        /* Collect which out rows are != 128 (baseline) */
        int hits[32]; int nhits = 0;
        for (int m = 0; m < 32; m++) {
            int v = out[32 * m + 0];   /* col 0 is representative (bias uniform) */
            if (v != 128) { hits[nhits++] = m; }
            if (nhits >= 32) break;
        }
        /* Print hit list + value at first hit */
        int hv = (nhits ? out[32 * hits[0]] : -1);
        char hs[128]; int hpos = 0; hs[0] = 0;
        for (int i = 0; i < nhits && hpos < 100; i++)
            hpos += snprintf(hs + hpos, sizeof(hs) - hpos, "%s%d", i ? "," : "", hits[i]);
        LOG("   phys_row=%2d lit (act=255) → out rows hit (val!=128): [%s]  val=%d",
            r_lit, hs, hv);
    }

    /* ---------------------------------------------------------------- */
    /* T11b: upper-half-lit — act[r][k]=1 if r<16 else 0                 */
    /* ---------------------------------------------------------------- */
    LOG("");
    LOG("--- T11b: act[r][k] = (r<16 ? 255 : 0). wt=1, bias=2.0. ---");
    LOG("   Naive: out rows 0..15 = 159, out rows 16..31 = 128.");
    memset(act, 0, 2048);
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < 32; k++)
            act[32 * r + k] = 255;
    memset(out, 0xBB, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    for (int m = 0; m < 32; m++) {
        LOG("   out[m=%2d][0..7] = %3u %3u %3u %3u %3u %3u %3u %3u",
            m,
            out[32*m+0], out[32*m+1], out[32*m+2], out[32*m+3],
            out[32*m+4], out[32*m+5], out[32*m+6], out[32*m+7]);
    }

    /* ---------------------------------------------------------------- */
    /* T11c: lower-half-lit — act[r][k] = (r>=16 ? 255 : 0).             */
    /* ---------------------------------------------------------------- */
    LOG("");
    LOG("--- T11c: act[r][k] = (r>=16 ? 255 : 0). wt=1, bias=2.0. ---");
    LOG("   Naive: out rows 0..15 = 128, out rows 16..31 = 159.");
    memset(act, 0, 2048);
    for (int r = 16; r < 32; r++)
        for (int k = 0; k < 32; k++)
            act[32 * r + k] = 255;
    memset(out, 0xCC, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    for (int m = 0; m < 32; m++) {
        LOG("   out[m=%2d][0..7] = %3u %3u %3u %3u %3u %3u %3u %3u",
            m,
            out[32*m+0], out[32*m+1], out[32*m+2], out[32*m+3],
            out[32*m+4], out[32*m+5], out[32*m+6], out[32*m+7]);
    }

    /* ---------------------------------------------------------------- */
    /* T11d: row-encoded amplifier — act[r][k] = (r+1)*4                */
    /* ---------------------------------------------------------------- */
    LOG("");
    LOG("--- T11d: act[r][k]=(r+1)*4, wt=1 all, bias=2.0. ---");
    LOG("   Expected per naive: acc[m][n] = (m+1)*4*32 = 128*(m+1).");
    LOG("                       out[m][n] = 128 + trunc((m+1)*128*2/512)");
    LOG("                                 = 128 + trunc((m+1)/2).");
    LOG("   Predictions: m=0:128, m=1:128(trunc(1)=1→129), m=2:129, m=3:129, m=4:130,…");
    LOG("   m=31 acc=128*32=4096; 4096*2/512=16 → out=144.");
    memset(act, 0, 2048);
    for (int r = 0; r < 32; r++)
        for (int k = 0; k < 32; k++)
            act[32 * r + k] = (uint8_t)((r + 1) * 4);
    memset(out, 0xDD, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    LOG("   observed out[m][0] for m=0..31:");
    for (int m = 0; m < 32; m++) {
        int naive_acc = (m + 1) * 4 * 32;
        int naive_pred = 128 + (naive_acc * 2) / 512;
        if (naive_pred > 255) naive_pred = 255;
        LOG("     m=%2d  obs=%3u  naive_pred=%3d (acc=%d)",
            m, out[32 * m + 0], naive_pred, naive_acc);
    }

    /* ---------------------------------------------------------------- */
    /* T11e: k-dim amplifier — act[r][k] = (k+1)*4 (SAME for all rows)  */
    /* Should give same result on all out rows (since rows are uniform) */
    /* but the value encodes acc = Σ_k (k+1)*4 * 1 = 4*(1+2+..+32)=4*528=2112 */
    /* ---------------------------------------------------------------- */
    LOG("");
    LOG("--- T11e: act[r][k]=(k+1)*4 (k-ramp, row-uniform), wt=1, bias=2.0. ---");
    LOG("   acc[m][n] = Σ_k (k+1)*4 = 4 * 528 = 2112 per cell.");
    LOG("   out = 128 + trunc(2112*2/512) = 128 + trunc(8.25) = 128+8 = 136.");
    memset(act, 0, 2048);
    for (int r = 0; r < 32; r++)
        for (int k = 0; k < 32; k++)
            act[32 * r + k] = (uint8_t)((k + 1) * 4);
    memset(out, 0xEE, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    for (int m = 0; m < 32; m += 4) {
        LOG("   out[m=%2d][0..7] = %3u %3u %3u %3u %3u %3u %3u %3u",
            m,
            out[32*m+0], out[32*m+1], out[32*m+2], out[32*m+3],
            out[32*m+4], out[32*m+5], out[32*m+6], out[32*m+7]);
    }

    LOG("");
    LOG("=== done ===");

    HAP_compute_res_hmx_unlock(ctx_id);
    HAP_compute_res_release(ctx_id);
    if (g_out) fclose(g_out);
    return 0;
}
