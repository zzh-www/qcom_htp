/*
 * probe_hmx_formula.c — T7..T10 silicon probes to decode :cm:sat.ub semantics.
 *
 * Working hypothesis from probe_sat_ub (T1..T6):
 *   out[m][c] = sat_u8( (bias_raw[2c+1] >> 7)
 *                     + round(acc[m][c] × bias_fp16_val[2c+1] / 512) )
 *   where bias[2c+1] is the fp16 entry actually used for column c, and
 *   the MAC treats activation as plain unsigned u8 (no signed shift).
 *
 *   T1 datum:  acc=32 (a=w=1, K=32), bias=2.0 → out=128
 *              check:  (0x4000>>7) + round(32*2/512) = 128 + 0 = 128 ✓
 *   T2 datum:  acc=32*(n+1), bias=2.0 → col7 out=129, col15 out=130
 *              check:  128 + round(256*2/512)=129 ✓;  128 + round(512*2/512)=130 ✓
 *   T5 datum:  acc small, bias varies → out = (bias_raw>>7)
 *              check:  0x3800>>7=112 ✓, 0x4800>>7=144 ✓
 *   T3/T6:     col 0 responds to bias index 1, col 31 responds to bias index 63
 *              → mapping  col c ↔ bias[2c+1]
 *
 * This probe locks those claims down across the full col range and across
 * activation polarity.  The matching reference saturate_u8 (pred) is computed
 * scalar-side here and compared to silicon (obs) for every cell.
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

/* Activation tile: row-major 32x32 bytes, padded to 2 KiB. */
static void fill_act_rm(uint8_t *tile, uint8_t (*pat)(int r, int k))
{
    memset(tile, 0, 2048);
    for (int r = 0; r < 32; r++)
        for (int k = 0; k < 32; k++)
            tile[32 * r + k] = pat(r, k);
}

/* Phase-2 4-K-row packed weight tile (K,N => kg*128 + 4*n + k4). */
static void fill_wt_p2(uint8_t *tile, int8_t (*pat)(int k, int n))
{
    memset(tile, 0, 1024);
    for (int kg = 0; kg < 8; kg++) {
        for (int n = 0; n < 32; n++) {
            tile[128 * kg + 4 * n + 0] = (uint8_t)pat(4 * kg + 0, n);
            tile[128 * kg + 4 * n + 1] = (uint8_t)pat(4 * kg + 1, n);
            tile[128 * kg + 4 * n + 2] = (uint8_t)pat(4 * kg + 2, n);
            tile[128 * kg + 4 * n + 3] = (uint8_t)pat(4 * kg + 3, n);
        }
    }
}

/* Decode fp16 u16 -> float. */
static float fp16_to_float(uint16_t h)
{
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    float v;
    if (exp == 0) {
        v = ldexpf((float)mant, -24);            /* subnormal */
    } else if (exp == 0x1F) {
        v = mant ? NAN : INFINITY;
    } else {
        v = ldexpf((float)(1024 + mant), (int)exp - 25);
    }
    return sign ? -v : v;
}

/* Scalar reference: predicts silicon output under working-hypothesis
 *   out[m][c] = sat_u8( (bias_raw[2c+1] >> 7)
 *                     + round(acc[m][c] × bias_fp16_val[2c+1] / 512) )
 * Returns 1 if all predictions match the observed tile; 0 otherwise.
 * Prints the first MISMATCH found and a summary count.
 */
static int verify_formula(const int32_t acc_mn[32][32],
                          const uint16_t bias[128],
                          const uint8_t obs[1024],
                          const char *tag)
{
    int mism = 0, match = 0, first_m = -1, first_c = -1;
    int first_pred = 0, first_obs = 0;
    for (int m = 0; m < 32; m++) {
        for (int c = 0; c < 32; c++) {
            uint16_t b = bias[2 * c + 1];
            float bv  = fp16_to_float(b);
            float raw = (float)(b >> 7);
            float scaled = (float)acc_mn[m][c] * bv / 512.0f;
            float sum = raw + scaled;
            int r = (int)roundf(sum);
            if (r < 0)       r = 0;
            else if (r > 255) r = 255;
            int o = obs[32 * m + c];
            if (o == r) match++;
            else {
                if (!mism) { first_m = m; first_c = c; first_pred = r; first_obs = o; }
                mism++;
            }
        }
    }
    LOG("  [%s] formula check: match=%4d  mism=%4d  first_mismatch@(m=%d,c=%d) pred=%d obs=%d",
        tag, match, mism, first_m, first_c, first_pred, first_obs);
    return mism == 0;
}

/* Print first 4 rows, a few cols for human eyeballing. */
static void dump_head(const uint8_t *obs, const char *tag)
{
    for (int m = 0; m < 4; m++) {
        LOG("  [%s] m=%d: %3u %3u %3u %3u %3u %3u %3u %3u  %3u %3u %3u %3u %3u %3u %3u %3u  ...  %3u %3u %3u %3u",
            tag, m,
            obs[32*m+0], obs[32*m+1], obs[32*m+2], obs[32*m+3],
            obs[32*m+4], obs[32*m+5], obs[32*m+6], obs[32*m+7],
            obs[32*m+8], obs[32*m+9], obs[32*m+10], obs[32*m+11],
            obs[32*m+12], obs[32*m+13], obs[32*m+14], obs[32*m+15],
            obs[32*m+28], obs[32*m+29], obs[32*m+30], obs[32*m+31]);
    }
}

/* ---- T7 patterns: activation polarity / scalar acc check ---- */
static uint8_t pat_act_one     (int r, int k) { (void)r; (void)k; return 1; }
static uint8_t pat_act_val128  (int r, int k) { (void)r; (void)k; return 128; }
static uint8_t pat_act_val255  (int r, int k) { (void)r; (void)k; return 255; }
static uint8_t pat_act_rowramp (int r, int k) { (void)k; return (uint8_t)(r + 1); }   /* act[r][k]=r+1 */
static uint8_t pat_act_kramp   (int r, int k) { (void)r; return (uint8_t)(k + 1); }   /* act[r][k]=k+1 */
static int8_t  pat_wt_one      (int k, int n) { (void)k; (void)n; return 1; }
static int8_t  pat_wt_colramp  (int k, int n) { (void)k; return (int8_t)(n + 1); }
static int8_t  pat_wt_krow0    (int k, int n) { return k == 0 ? (int8_t)(n + 1) : 0; } /* w[0][n]=n+1 */

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    g_out = fopen("./probe_hmx_formula_result.txt", "w");

    LOG("=== HMX :after:cm:sat.ub formula verification (SM8650 v75) ===");
    LOG("Hypothesis: out[m][c] = sat_u8((bias_raw[2c+1]>>7) + round(acc[m][c]*bias_fp16[2c+1]/512))");
    LOG("            activation treated as plain u8 (no signed shift).");
    LOG("            col c  <->  bias index (2c+1).");
    LOG("");

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

    int32_t acc_mn[32][32];

    /* ---------------------------------------------------------------- */
    /* T7 — activation polarity                                         */
    /* ---------------------------------------------------------------- */
    LOG("--- T7: activation polarity — does HMX treat act as u8 or (u8-128)? ---");

    /* T7a: act=k+1 (rows identical), wt[0][n]=n+1, else 0. */
    /* If u8:  acc[m][n] = act[m][0]*w[0][n] = 1*(n+1) = n+1
     * If signed (u8-128): acc = (1-128)*(n+1) = -127*(n+1)
     * Use bias=1.0 (scale = 1.0/512) so scaled contribution = acc/512.
     * At u8 hypothesis: scaled≈0 → out = (0x3C00>>7)=120 (const)
     * At signed hyp.: acc = -127*(n+1); scaled = -127*(n+1)/512; varies hugely per col. */
    LOG("");
    LOG("  T7a: act=all-1, wt[0][n]=n+1 (col ramp at k=0), bias=1.0 (0x3C00).");
    LOG("       If u8:     all cells ≈ 120 (acc tiny vs /512 scale).");
    LOG("       If signed: cells vary with col; acc=-127*(n+1).");
    fill_act_rm(act, pat_act_one);
    fill_wt_p2 (wt,  pat_wt_krow0);
    for (int i = 0; i < 128; i++) bias[i] = 0x3C00;  /* fp16 1.0 */
    memset(out, 0xAA, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    dump_head(out, "T7a");
    /* Predict under u8 hypothesis */
    for (int m = 0; m < 32; m++)
        for (int n = 0; n < 32; n++) acc_mn[m][n] = (n + 1);  /* 1*(n+1) */
    verify_formula(acc_mn, bias, out, "T7a u8-hyp");
    /* Also predict under signed hypothesis for comparison */
    LOG("       (signed-hyp would predict out[m][n]<0 for all n, giving 0 after sat.)");

    /* T7b: act[r][k]=128 everywhere, wt=1 everywhere, bias=2.0.
     * If u8:  acc = 128 * 1 * 32 = 4096 per cell
     *         out = (0x4000>>7) + round(4096*2/512) = 128 + 16 = 144
     * If signed: acc = 0 * 1 * 32 = 0
     *         out = (0x4000>>7) + 0 = 128 */
    LOG("");
    LOG("  T7b: act=all-128, wt=all-1, bias=2.0.  u8-hyp→144, signed-hyp→128.");
    fill_act_rm(act, pat_act_val128);
    fill_wt_p2 (wt,  pat_wt_one);
    for (int i = 0; i < 128; i++) bias[i] = 0x4000;
    memset(out, 0xAA, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    dump_head(out, "T7b");
    for (int m = 0; m < 32; m++)
        for (int n = 0; n < 32; n++) acc_mn[m][n] = 128 * 1 * 32;
    verify_formula(acc_mn, bias, out, "T7b u8-hyp");

    /* T7c: act[r][k]=255, wt=1 everywhere, bias=2.0.
     * u8: acc = 255 * 32 = 8160.  out = 128 + round(8160*2/512) = 128 + 32 = 160
     * signed: acc = 127*32 = 4064. out = 128 + round(4064*2/512) = 128 + 16 = 144 */
    LOG("");
    LOG("  T7c: act=all-255, wt=all-1, bias=2.0.  u8-hyp→160, signed-hyp→144.");
    fill_act_rm(act, pat_act_val255);
    fill_wt_p2 (wt,  pat_wt_one);
    for (int i = 0; i < 128; i++) bias[i] = 0x4000;
    memset(out, 0xAA, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    dump_head(out, "T7c");
    for (int m = 0; m < 32; m++)
        for (int n = 0; n < 32; n++) acc_mn[m][n] = 255 * 1 * 32;
    verify_formula(acc_mn, bias, out, "T7c u8-hyp");

    /* T7d: act[r][k] = r+1 (row ramp), wt=1 everywhere, bias=1.0.
     * u8: acc[m][n] = (m+1)*1*32 = 32*(m+1); out per row varies (m+1)/16 above 120.
     * Output should grow with m. */
    LOG("");
    LOG("  T7d: act[r][k]=r+1 (row ramp), wt=all-1, bias=1.0. Output grows with m.");
    fill_act_rm(act, pat_act_rowramp);
    fill_wt_p2 (wt,  pat_wt_one);
    for (int i = 0; i < 128; i++) bias[i] = 0x3C00;  /* 1.0 */
    memset(out, 0xAA, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    for (int m = 0; m < 32; m += 4) {
        LOG("    m=%2d: %3u %3u %3u %3u ... %3u %3u", m,
            out[32*m+0], out[32*m+1], out[32*m+2], out[32*m+3],
            out[32*m+30], out[32*m+31]);
    }
    for (int m = 0; m < 32; m++)
        for (int n = 0; n < 32; n++) acc_mn[m][n] = (m + 1) * 1 * 32;
    verify_formula(acc_mn, bias, out, "T7d u8-hyp");

    /* ---------------------------------------------------------------- */
    /* T8 — bias fine-sweep including mantissa                           */
    /* ---------------------------------------------------------------- */
    LOG("");
    LOG("--- T8: bias fine sweep — test non-power-of-2 bias values ---");
    LOG("   Uniform acc=32 (a=w=all-1, K=32). Baseline = (bias_raw>>7).");
    LOG("   Scaled term = round(32 * bias_fp16 / 512) = round(bias_fp16 / 16).");

    fill_act_rm(act, pat_act_one);
    fill_wt_p2 (wt,  pat_wt_one);
    /* Sweep bias values with nonzero mantissa:
     *   0x4000 (2.0), 0x4100 (2.125), 0x4200 (2.25), 0x4300 (2.375), 0x4400 (4.0),
     *   0x3E00 (1.5), 0x4700 (7.0), 0x4A00 (12.0), 0x4C00 (16.0), 0x4E00 (24.0)
     */
    uint16_t t8_biases[] = {
        0x4000, 0x4100, 0x4200, 0x4300, 0x4400,
        0x3E00, 0x4700, 0x4A00, 0x4C00, 0x4E00,
        0x3800, 0x3900, 0x3A00, 0x3B00, 0x3C00,
    };
    int n_t8 = sizeof(t8_biases) / sizeof(t8_biases[0]);
    for (int bi = 0; bi < n_t8; bi++) {
        uint16_t b = t8_biases[bi];
        for (int i = 0; i < 128; i++) bias[i] = b;
        memset(out, 0xCC, 1024);
        RUN_CM_MAC_SAT_UB(bias, act, wt, out);
        for (int m = 0; m < 32; m++)
            for (int n = 0; n < 32; n++) acc_mn[m][n] = 32;
        float bv = fp16_to_float(b);
        int pred = (b >> 7) + (int)roundf(32.0f * bv / 512.0f);
        if (pred > 255) pred = 255; else if (pred < 0) pred = 0;
        LOG("  bias=0x%04x (%.4f): out[0][0..3]=%3u %3u %3u %3u  (pred=%d)",
            b, bv, out[0], out[1], out[2], out[3], pred);
        char tag[48];
        snprintf(tag, sizeof tag, "T8 b=0x%04x", b);
        verify_formula(acc_mn, bias, out, tag);
    }

    /* ---------------------------------------------------------------- */
    /* T9 — bias lane mapping for all 32 cols                            */
    /* ---------------------------------------------------------------- */
    LOG("");
    LOG("--- T9: bias lane mapping. For each c in 0..31, set bias[2c+1] to a");
    LOG("        distinctive value and bias[2c] to a contrasting value. Expect col c");
    LOG("        output to use bias[2c+1]; if model wrong, other col will light up. ---");

    fill_act_rm(act, pat_act_one);   /* uniform acc=32 */
    fill_wt_p2 (wt,  pat_wt_one);
    /* Set bias[2c]   = 1.0 (0x3C00, baseline out=120)
     *     bias[2c+1] = encodes col index c as bias value:
     *       fp16(1.0 + c*0.125) for c=0..31   -> mantissa varies
     */
    for (int c = 0; c < 32; c++) {
        /* fp16(1.0 + c/8) — bias values 1.0..4.875 */
        /* Build via exponent 15 (biased for value 1..<2) for c<8, etc. Easier: compute. */
        float v = 1.0f + (float)c * 0.125f;
        /* build fp16 manually */
        uint32_t u;
        memcpy(&u, &v, 4);
        uint32_t sign = (u >> 31) & 1;
        int32_t  e32  = (int32_t)((u >> 23) & 0xFF) - 127;
        uint32_t m32  = u & 0x7FFFFF;
        uint16_t h;
        if (e32 < -14) {
            h = (uint16_t)(sign << 15);
        } else {
            int32_t eh = e32 + 15;
            uint32_t mh = m32 >> 13;
            h = (uint16_t)((sign << 15) | ((uint32_t)eh << 10) | mh);
        }
        bias[2 * c]     = 0x3C00;   /* "wrong lane" bias for col c */
        bias[2 * c + 1] = h;        /* "right lane" bias for col c */
    }
    /* Fill any upper 64 entries with a distinctive error-tag */
    for (int i = 64; i < 128; i++) bias[i] = 0x3C00;
    memset(out, 0xDD, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    /* Observed per col: expect (bias[2c+1]>>7) + round(32 * fp16(1+c/8) / 512) */
    for (int c = 0; c < 32; c++) {
        uint16_t b = bias[2 * c + 1];
        float bv = fp16_to_float(b);
        int pred = (b >> 7) + (int)roundf(32.0f * bv / 512.0f);
        if (pred > 255) pred = 255; else if (pred < 0) pred = 0;
        LOG("  col %2d: bias[2c+1]=0x%04x (%.4f)  obs[m=0]=%3u  pred=%3d  %s",
            c, b, bv, out[c], pred, (out[c] == pred) ? "OK" : "MISMATCH");
    }
    /* Full-tile verify */
    for (int m = 0; m < 32; m++)
        for (int n = 0; n < 32; n++) acc_mn[m][n] = 32;
    verify_formula(acc_mn, bias, out, "T9 per-col bias");

    LOG("");
    LOG("  T9b: lane probe — set bias[2c] distinct (wrong lane), bias[2c+1]=1.0 uniform.");
    LOG("       If mapping is 2c+1 only, output is 120 everywhere; if any col tracks 2c, mapping wrong.");
    for (int c = 0; c < 32; c++) {
        float v = 1.0f + (float)c * 0.25f;  /* 1..8.75 */
        uint32_t u; memcpy(&u, &v, 4);
        uint32_t sign = (u >> 31) & 1;
        int32_t  e32  = (int32_t)((u >> 23) & 0xFF) - 127;
        uint32_t m32  = u & 0x7FFFFF;
        uint16_t h;
        if (e32 < -14) h = (uint16_t)(sign << 15);
        else { int32_t eh = e32 + 15; uint32_t mh = m32 >> 13;
               h = (uint16_t)((sign << 15) | ((uint32_t)eh << 10) | mh); }
        bias[2 * c]     = h;         /* varies */
        bias[2 * c + 1] = 0x3C00;    /* 1.0 uniform */
    }
    for (int i = 64; i < 128; i++) bias[i] = 0x3C00;
    memset(out, 0xDD, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    int t9b_all_120 = 1;
    int t9b_any_mismatch_col = -1;
    int t9b_mismatch_val = 0;
    for (int c = 0; c < 32; c++) {
        if (out[c] != 120) {
            t9b_all_120 = 0;
            if (t9b_any_mismatch_col < 0) { t9b_any_mismatch_col = c; t9b_mismatch_val = out[c]; }
        }
    }
    LOG("    all-120? %s (first differing col=%d val=%d)",
        t9b_all_120 ? "YES (bias[2c] is the EVEN lane, unused by sat.ub)"
                    : "NO — lane mapping is not pure 2c+1",
        t9b_any_mismatch_col, t9b_mismatch_val);
    for (int m = 0; m < 2; m++) {
        LOG("    m=%d: %3u %3u %3u %3u  %3u %3u %3u %3u  %3u %3u %3u %3u  %3u %3u %3u %3u", m,
            out[32*m+0], out[32*m+1], out[32*m+2], out[32*m+3],
            out[32*m+4], out[32*m+5], out[32*m+6], out[32*m+7],
            out[32*m+8], out[32*m+9], out[32*m+10], out[32*m+11],
            out[32*m+12], out[32*m+13], out[32*m+14], out[32*m+15]);
    }

    LOG("");
    LOG("  T9c: bias[64..127] probe — are the upper 64 entries used at all?");
    LOG("       Set bias[0..63]=1.0, bias[64..127]=4.0; if upper unused, still 120 everywhere.");
    for (int i = 0; i < 64; i++)   bias[i] = 0x3C00;  /* 1.0 */
    for (int i = 64; i < 128; i++) bias[i] = 0x4400;  /* 4.0 */
    memset(out, 0xDD, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    int anydiff_9c = 0;
    for (int m = 0; m < 32; m++)
        for (int c = 0; c < 32; c++)
            if (out[32*m + c] != 120) anydiff_9c++;
    LOG("    cells != 120: %d  (0=upper 64 unused; >0=upper 64 does something)", anydiff_9c);
    dump_head(out, "T9c");

    /* ---------------------------------------------------------------- */
    /* T10 — single-cell sanity across many (m,n)                        */
    /* ---------------------------------------------------------------- */
    LOG("");
    LOG("--- T10: deterministic acc per (m,n) across several cells, predict each output. ---");
    /* Use act=all-1, wt[k][n] = 2 for k==n mod 4, else 0. acc[m][n] = 8 * 2 = 16
     * Use bias[2c+1] varying — verify formula scalarly. */
    fill_act_rm(act, pat_act_one);
    /* w[k][n] = 2 if k % 4 == n % 4, else 0. Count of k in [0,32) with k%4==n%4 = 8.
     * acc[m][n] = sum_k a[m][k]*w[k][n] = 8 * 2 = 16 for all (m,n). */
    for (int i = 0; i < 1024; i++) wt[i] = 0;
    for (int kg = 0; kg < 8; kg++) {
        for (int n = 0; n < 32; n++) {
            for (int k4 = 0; k4 < 4; k4++) {
                int k = 4 * kg + k4;
                if (k % 4 == n % 4) {
                    wt[128 * kg + 4 * n + k4] = 2;
                }
            }
        }
    }
    /* Check with one simpler weight: wt=2 everywhere. acc = 2*32 = 64 per cell. */
    fill_wt_p2(wt, pat_wt_one);
    for (int i = 0; i < 1024; i++) wt[i] *= 2;  /* every wt now =2 */
    /* Set bias[2c+1] sweeping across mantissa. */
    for (int i = 0; i < 128; i++) bias[i] = 0x3800;   /* 0.5 */
    for (int c = 0; c < 32; c++) {
        /* use exp=15 (value in 1..2), mantissa varies across cols */
        uint16_t h = (uint16_t)(0x3C00 | (c * 32));   /* 0x3C00 + c*0x20 */
        bias[2 * c + 1] = h;
    }
    memset(out, 0xEE, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    for (int m = 0; m < 32; m++)
        for (int n = 0; n < 32; n++) acc_mn[m][n] = 64;
    verify_formula(acc_mn, bias, out, "T10 wt=2 bias mantissa sweep");
    for (int c = 0; c < 16; c += 1) {
        uint16_t b = bias[2 * c + 1];
        float bv = fp16_to_float(b);
        int pred = (b >> 7) + (int)roundf(64.0f * bv / 512.0f);
        if (pred > 255) pred = 255; else if (pred < 0) pred = 0;
        LOG("    c=%2d bias=0x%04x (%.4f) obs=%3u pred=%3d %s",
            c, b, bv, out[c], pred, (out[c] == pred) ? "OK" : "MISMATCH");
    }

    /* T10b: big range — pick biases that push output near saturation. */
    LOG("");
    LOG("  T10b: saturation test — acc=32, bias=16.0 (0x4C00): expect");
    LOG("        (0x4C00>>7) + round(32*16/512) = 152 + 1 = 153");
    fill_act_rm(act, pat_act_one);
    fill_wt_p2 (wt,  pat_wt_one);
    for (int i = 0; i < 128; i++) bias[i] = 0x4C00;
    memset(out, 0xFA, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    for (int m = 0; m < 32; m++)
        for (int n = 0; n < 32; n++) acc_mn[m][n] = 32;
    verify_formula(acc_mn, bias, out, "T10b bias=16");
    LOG("    obs[0][0..3]=%3u %3u %3u %3u (pred=153)",
        out[0], out[1], out[2], out[3]);

    LOG("");
    LOG("  T10c: high acc + small bias — acc=32*255=8160, bias=1.0.");
    LOG("        (0x3C00>>7)+round(8160*1/512) = 120 + 16 = 136");
    fill_act_rm(act, pat_act_val255);
    fill_wt_p2 (wt,  pat_wt_one);
    for (int i = 0; i < 128; i++) bias[i] = 0x3C00;
    memset(out, 0xFB, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    for (int m = 0; m < 32; m++)
        for (int n = 0; n < 32; n++) acc_mn[m][n] = 255 * 32;
    verify_formula(acc_mn, bias, out, "T10c high acc");
    LOG("    obs[0][0..3]=%3u %3u %3u %3u (pred=136)",
        out[0], out[1], out[2], out[3]);

    LOG("");
    LOG("=== done ===");

    HAP_compute_res_hmx_unlock(ctx_id);
    HAP_compute_res_release(ctx_id);
    if (g_out) fclose(g_out);
    return 0;
}
