/*
 * probe_sat_ub.c — silicon RE for :after:cm:sat.ub single-byte HMX readback.
 *
 * Goal: verify whether the QNN-style single-instruction u8-output store
 * (used by q::ConvLayer_s1.opt) fills all 32 output rows or inherits the
 * 16-of-32 limitation previously found on :after.uh acc:2x1 (see
 * Agent/cm_readback_re.md).
 *
 * Tests:
 *   T1 — uniform: act[r][k] = 1, wt[k][n] = 1, bias[n] = 0x4000 (fp16 2.0).
 *        expected acc[m][n] = 32 ∀ m,n.
 *        out[m][n] = saturate_u8(acc × 2/2) = saturate_u8(32) = 32.
 *        Dump all 1024 bytes of out tile — if 32 rows × 32 cols all =32,
 *        :sat.ub fills 32 rows. If only 16 rows populated, :cm row halving
 *        applies (and QNN must be doing 2 MACs internally).
 *
 *   T2 — column ramp: wt[k][n] = (n+1), rest same. acc[m][n] = 32*(n+1).
 *        With bias[n] = 0x4000 (scale 1.0): out[m][n] = min(32*(n+1), 255).
 *        Saturation expected for n >= 7 (32*8=256 > 255).
 *        Confirms: (a) per-column bias applies uniformly across rows,
 *                  (b) saturation at 255 works.
 *
 *   T3 — per-column bias scaling: same acc as T1 (32 uniformly), but
 *        bias[n] varies per column: bias[0]=fp16(2.0), bias[1]=fp16(0.2),
 *        bias[2]=fp16(0.02), bias[3]=fp16(2.0 × 10/32) ≈ 0.625.
 *        expected out[m][n]:
 *          col 0: 32 × 2 / 2 = 32
 *          col 1: 32 × 0.2 / 2 = 3.2 → 3
 *          col 2: 32 × 0.02 / 2 = 0.32 → 0
 *          col 3: 32 × 0.625 / 2 = 10.0 → 10
 *        Verifies fp16 bias scale fold math + per-col independence.
 *
 *   T4 — bias format probe: use known asymmetric pattern (acc=100) with
 *        bias sweep 0x3800..0x4400 (fp16 ~0.5..4.0) and check output
 *        quantization for off-by-one at scale boundaries.
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

/* Row-major 1 KiB act tile. */
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

static uint8_t pat_act_all1  (int r, int k) { (void)r; (void)k; return 1; }
static int8_t  pat_wt_all1   (int k, int n) { (void)k; (void)n; return 1; }
static int8_t  pat_wt_colramp(int k, int n) { (void)k; return (int8_t)(n + 1); }

#define RT_A    (2047 | 0x1C)      /* :cm activation Rt */
#define RT_W    0x3FF              /* plain weight Rt */

/* Single :cm MAC + :after:cm:sat.ub u8 readback. out_p must be 128-B
 * aligned VTCM, size 1024 bytes (32×32 u8 cells). */
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

static void fill_bias(uint16_t *b, uint16_t v)
{
    for (int i = 0; i < 128; i++) b[i] = v;
}

/* Dump a 32×32 u8 tile as 32 rows × 32 bytes; report any nonzero rows. */
static void dump_u8_tile(const uint8_t *out, const char *tag)
{
    LOG("  [%s] u8 output tile (32 rows × 32 cols):", tag);
    int nonzero_rows = 0, zero_rows = 0;
    for (int r = 0; r < 32; r++) {
        int nz = 0;
        for (int c = 0; c < 32; c++) if (out[r * 32 + c]) nz++;
        if (nz) nonzero_rows++; else zero_rows++;
        if (r < 4 || (r >= 14 && r <= 17) || r >= 28) {
            LOG("    row %2d [nz=%2d]: %3u %3u %3u %3u %3u %3u %3u %3u  %3u %3u %3u %3u %3u %3u %3u %3u",
                r, nz,
                out[r*32+0], out[r*32+1], out[r*32+2], out[r*32+3],
                out[r*32+4], out[r*32+5], out[r*32+6], out[r*32+7],
                out[r*32+8], out[r*32+9], out[r*32+10], out[r*32+11],
                out[r*32+12], out[r*32+13], out[r*32+14], out[r*32+15]);
        }
    }
    LOG("    summary: nonzero_rows=%d  zero_rows=%d", nonzero_rows, zero_rows);
}

/* Dump output treating tile as if layout might be 2-stream interleaved.
 * If :sat.ub uses the same stream layout as :after.uh, bytes
 * at tile[phys_row*32 + col] would actually correspond to different
 * logical (m, n) than straightforward row-major. */
static void dump_u8_via_stream_layout(const uint8_t *out, const char *tag)
{
    LOG("  [%s] reinterpret as 2-stream layout (32B per phys_row) — inspecting byte frequency:", tag);
    int hits[256] = {0};
    for (int i = 0; i < 1024; i++) hits[out[i]]++;
    int distinct = 0;
    for (int v = 0; v < 256; v++) if (hits[v]) distinct++;
    LOG("    distinct byte values: %d", distinct);
    for (int v = 0; v < 256; v++) {
        if (hits[v]) LOG("      byte=%3d  count=%d", v, hits[v]);
    }
}

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    g_out = fopen("./probe_sat_ub_result.txt", "w");

    LOG("=== HMX :after:cm:sat.ub single-byte readback probe (SM8650 v75) ===");
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

    /* ---- T1: uniform, bias=0x4000 (fp16 2.0 → scale 1.0) ---- */
    LOG("");
    LOG("--- T1: act=wt=all1, bias=0x4000, expected out[m][n]=32 all cells ---");
    fill_act_rm(act, pat_act_all1);
    fill_wt_p2 (wt,  pat_wt_all1);
    fill_bias(bias, 0x4000);
    memset(out, 0xAA, 1024);   /* sentinel */
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    dump_u8_tile(out, "T1");
    dump_u8_via_stream_layout(out, "T1 byte freq");

    /* ---- T2: col-ramp wt, bias=0x4000 (scale 1.0). acc=32*(n+1) ---- */
    LOG("");
    LOG("--- T2: wt[k][n]=n+1, acc=32*(n+1), bias=0x4000, sat at 255 ---");
    fill_act_rm(act, pat_act_all1);
    fill_wt_p2 (wt,  pat_wt_colramp);
    fill_bias(bias, 0x4000);
    memset(out, 0xAA, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    dump_u8_tile(out, "T2");

    /* ---- T3: per-column bias scaling (acc=32 uniform) ---- */
    LOG("");
    LOG("--- T3: uniform acc=32, per-col bias varies. Expected: ---");
    LOG("      col 0: bias=0x4000 (2.0) → out = 32×2/2 = 32");
    LOG("      col 1: bias=0x3266 (≈0.2) → out = 32×0.2/2 = 3");
    LOG("      col 2: bias=0x2124 (≈0.02) → out = 32×0.02/2 = 0");
    LOG("      col 3: bias=0x3900 (≈0.625) → out = 32×0.625/2 = 10");
    fill_act_rm(act, pat_act_all1);
    fill_wt_p2 (wt,  pat_wt_all1);
    /* fp16 values: 2.0=0x4000, 0.2≈0x3266, 0.02≈0x2124, 0.625=0x38a0  */
    for (int i = 0; i < 128; i++) bias[i] = 0x4000;  /* default */
    bias[0] = 0x4000; bias[1] = 0x3266; bias[2] = 0x2124; bias[3] = 0x38a0;
    memset(out, 0xAA, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    LOG("  T3 out[0][0..7]: %u %u %u %u %u %u %u %u",
        out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    LOG("  T3 out[8][0..7]: %u %u %u %u %u %u %u %u",
        out[8*32+0], out[8*32+1], out[8*32+2], out[8*32+3],
        out[8*32+4], out[8*32+5], out[8*32+6], out[8*32+7]);
    LOG("  T3 out[16][0..7]: %u %u %u %u %u %u %u %u",
        out[16*32+0], out[16*32+1], out[16*32+2], out[16*32+3],
        out[16*32+4], out[16*32+5], out[16*32+6], out[16*32+7]);
    LOG("  T3 out[24][0..7]: %u %u %u %u %u %u %u %u",
        out[24*32+0], out[24*32+1], out[24*32+2], out[24*32+3],
        out[24*32+4], out[24*32+5], out[24*32+6], out[24*32+7]);

    /* ---- T4: Does :retain variant let us chain two :cm:sat.ub? Maybe
     * parity strategy: one MAC+conv for odd M, another for even M. Test
     * plain variant first; if T1 shows only 16 rows, we'll need a 2nd
     * MAC configured differently. ---- */
    LOG("");
    LOG("--- T4: MAC twice (same inputs), :retain + second :after:cm:sat.ub — see if second fills different rows ---");
    fill_act_rm(act, pat_act_all1);
    fill_wt_p2 (wt,  pat_wt_all1);
    fill_bias(bias, 0x4000);
    memset(out, 0xAA, 1024);
    /* Bias + clracc + MAC */
    asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
    asm volatile("mxclracc" ::: "memory");
    asm volatile("{ activation.ub = mxmem(%0,%1):cm\n"
                 "  weight.b      = mxmem(%2,%3) }"
                 :: "r"(act), "r"(RT_A), "r"(wt), "r"(RT_W) : "memory");
    /* Try :after:retain:cm:sat.ub first, then :after:cm:sat.ub */
    asm volatile("mxmem(%0,%1):after:retain:cm:sat.ub = acc"
                 :: "r"(out), "r"(0) : "memory");
    /* After retain, acc still valid. Try second readback to ANOTHER
     * location — if it fills 16 rows of same set, likely the EVEN rows
     * need a different MAC. */
    uint8_t *out2 = vtcm + 5120;
    memset(out2, 0xBB, 1024);
    asm volatile("mxmem(%0,%1):after:cm:sat.ub = acc"
                 :: "r"(out2), "r"(0) : "memory");
    LOG("  T4 out1 (first :after:retain:cm:sat.ub):");
    dump_u8_tile(out, "T4-out1");
    LOG("  T4 out2 (second :after:cm:sat.ub, same acc via retain):");
    dump_u8_tile(out2, "T4-out2");

    /* ---- T5: formula sweep. Fix wt+act so acc[m][n] is predictable for
     * each m,n; sweep bias values; dump specific cells to determine
     * out = f(acc, bias) relationship precisely. ---- */
    LOG("");
    LOG("--- T5: formula sweep. act=all1, wt[k][n]=(k==0?n+1:0) so acc[m][n]=n+1 ---");

    /* Simple weight: wt[0][n] = n+1, rest 0, so acc[m][n] = 1*(n+1) = n+1 per m. */
    memset(wt, 0, 1024);
    for (int col = 0; col < 32; col++) {
        /* P2 layout: kg=0, col, row4=0 (since k=0 corresponds to row4=0 in kg=0). */
        wt[128 * 0 + 4 * col + 0] = (uint8_t)(col + 1);  /* wt[k=0][col] = col+1 */
    }
    fill_act_rm(act, pat_act_all1);

    /* 5 sweeps: bias ∈ {0x3800 (0.5), 0x3C00 (1.0), 0x4000 (2.0), 0x4400 (4.0), 0x4800 (8.0)} */
    uint16_t bias_values[] = {0x3800, 0x3C00, 0x4000, 0x4400, 0x4800};
    const char *bias_names[] = {"0.5 (0x3800)", "1.0 (0x3C00)", "2.0 (0x4000)", "4.0 (0x4400)", "8.0 (0x4800)"};
    for (int bi = 0; bi < 5; bi++) {
        fill_bias(bias, bias_values[bi]);
        memset(out, 0xCC, 1024);
        RUN_CM_MAC_SAT_UB(bias, act, wt, out);
        LOG("  bias=%s: out[0][0..15]  = %3u %3u %3u %3u %3u %3u %3u %3u  %3u %3u %3u %3u %3u %3u %3u %3u",
            bias_names[bi],
            out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7],
            out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]);
        LOG("                    out[0][16..31] = %3u %3u %3u %3u %3u %3u %3u %3u  %3u %3u %3u %3u %3u %3u %3u %3u",
            out[16], out[17], out[18], out[19], out[20], out[21], out[22], out[23],
            out[24], out[25], out[26], out[27], out[28], out[29], out[30], out[31]);
    }

    /* ---- T6: confirm T3 anomaly — is per-col bias truly applied or just bias[0] repeated? ---- */
    LOG("");
    LOG("--- T6: fix act=wt=all1 (acc=32 uniform); sweep bias[n] individually to see true per-col mapping ---");
    fill_act_rm(act, pat_act_all1);
    fill_wt_p2 (wt,  pat_wt_all1);

    /* 128 bias entries; set bias[n] = 0x4000 * (n % 4 == 0 ? 4.0 : 1.0) discretely */
    for (int i = 0; i < 128; i++) bias[i] = 0x3C00;   /* 1.0 */
    /* Poke specific bias entries with distinct values */
    bias[0]  = 0x4000;  /* 2.0 */
    bias[1]  = 0x4400;  /* 4.0 */
    bias[2]  = 0x4800;  /* 8.0 */
    bias[31] = 0x4800;  /* 8.0 */
    bias[32] = 0x4800;  /* 8.0 (does bias[32..63] matter too?) */
    bias[63] = 0x4800;
    bias[64] = 0x4800;
    bias[127]= 0x4800;
    memset(out, 0xDD, 1024);
    RUN_CM_MAC_SAT_UB(bias, act, wt, out);
    LOG("  T6 bias[0]=2.0, [1]=4.0, [2]=8.0, [31]=[32]=[63]=[64]=[127]=8.0, rest=1.0");
    LOG("  out[0][0..7]:  %3u %3u %3u %3u %3u %3u %3u %3u", out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]);
    LOG("  out[0][24..31]: %3u %3u %3u %3u %3u %3u %3u %3u", out[24], out[25], out[26], out[27], out[28], out[29], out[30], out[31]);
    LOG("  out[16][0..7]: %3u %3u %3u %3u %3u %3u %3u %3u", out[16*32+0], out[16*32+1], out[16*32+2], out[16*32+3], out[16*32+4], out[16*32+5], out[16*32+6], out[16*32+7]);

    LOG("");
    LOG("=== done ===");

    HAP_compute_res_hmx_unlock(ctx_id);
    HAP_compute_res_release(ctx_id);
    if (g_out) fclose(g_out);
    return 0;
}
