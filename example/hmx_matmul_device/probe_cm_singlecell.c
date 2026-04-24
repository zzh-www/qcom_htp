/*
 * probe_cm_singlecell.c — pin down acc[m][n] → rb[idx] mapping under :cm.
 *
 * For each target (M, N) ∈ [0,32) × [0,32):
 *   activation: act[r][k] = 1 iff (r==M, k==0), else 0
 *   weight    : w[k][n]   = M+1 iff (k==0, n==N), else 0
 *   ⇒ acc[M][N] = M+1, all other acc cells = 0.
 *
 * Run :cm MAC, do BOTH dual-scale readbacks (lo + hi). Find which
 * rb_lo / rb_hi indices hold the value M+1 (and any near-misses caused
 * by the dual-scale carry). Print the mapping.
 *
 * Doing all 1024 (M, N) takes ~1024 MACs which is fine on cDSP.
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

#define RT_A    (2047 | 0x1c)
#define RT_W    0x3FF
#define BIAS_LO 0x4000
#define BIAS_HI 0x2000

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

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    g_out = fopen("./probe_cm_singlecell_result.txt", "w");

    LOG("=== HMX :cm single-cell (m,n) → rb[idx] mapping probe ===");
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

    for (int i = 0; i < 128; i++) bias_lo[i] = BIAS_LO;
    for (int i = 0; i < 128; i++) bias_hi[i] = BIAS_HI;

    LOG("M N  V  | LO_idxs (val pairs)         HI_idxs (val pairs)");

    /* Loop over single-cell (M, N) positions. Use a small N subset to
     * keep the output readable, then sweep all (M, N=0..3). */
    static const int N_sweep[] = {0, 1, 7, 15, 31};

    for (unsigned ni = 0; ni < sizeof(N_sweep)/sizeof(N_sweep[0]); ni++) {
        int N = N_sweep[ni];
        for (int M = 0; M < 32; M++) {
            int V = M + 1;

            /* fill activation: only act[M][0] = 1 */
            memset(act, 0, 2048);
            act[32 * M + 0] = 1;

            /* fill weight (P2 layout): only w[0][N] = V */
            memset(wt, 0, 1024);
            /* P2 byte: tile[128*kg + 4*col + row4] = w[4*kg+row4][col]
             * for k=0, row4=0, col=N → tile[4*N] = w[0][N] */
            wt[4 * N + 0] = (uint8_t)V;

            memset(rb_lo, 0, 2048);
            memset(rb_hi, 0, 2048);
            RUN_CM_DUAL(bias_lo, bias_hi, act, wt, rb_lo, rb_hi);

            /* Collect all nonzero idx + values from rb_lo and rb_hi. */
            char lo_buf[256] = {0};
            char hi_buf[256] = {0};
            int  lo_pos = 0, hi_pos = 0;
            int  lo_nnz = 0, hi_nnz = 0;
            for (int i = 0; i < 1024; i++) {
                int v = (int16_t)rb_lo[i];
                if (v != 0) {
                    lo_nnz++;
                    if (lo_pos < (int)sizeof(lo_buf) - 32)
                        lo_pos += snprintf(lo_buf + lo_pos,
                                           sizeof(lo_buf) - lo_pos,
                                           "[%d]=%d ", i, v);
                }
            }
            for (int i = 0; i < 1024; i++) {
                int v = (int16_t)rb_hi[i];
                if (v != 0) {
                    hi_nnz++;
                    if (hi_pos < (int)sizeof(hi_buf) - 32)
                        hi_pos += snprintf(hi_buf + hi_pos,
                                           sizeof(hi_buf) - hi_pos,
                                           "[%d]=%d ", i, v);
                }
            }
            LOG("%2d %2d %2d | LO_nnz=%d %s| HI_nnz=%d %s",
                M, N, V, lo_nnz, lo_buf, hi_nnz, hi_buf);
        }
        LOG("");
    }

    LOG("=== done ===");
    HAP_compute_res_hmx_unlock(ctx_id);
    HAP_compute_res_release(ctx_id);
    if (g_out) fclose(g_out);
    return 0;
}
