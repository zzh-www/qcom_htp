/*
 * bench_matmul_device.c — real-device pcycle bench for fp16 vs int16
 * 32x32x32 matmul on Hexagon v75 HTP (SM8650 cDSP, Unsigned PD).
 *
 * Loaded via run_main_on_hexagon. No signing needed on v75.
 *
 * Phases measured with HAP_perf_get_pcycles():
 *   A) fp16 matmul tile (4 HMX packets)
 *   B) int16 matmul kernel (our dual-scale readback, 24 HMX packets +
 *      scalar decompose/combine/requant)
 *   C) int16 HMX-only (24 HMX packets on pre-packed tiles, no CPU)
 *
 * Each phase runs a warmup call then a 10-iteration timed loop to
 * smooth out first-call cache effects.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include "HAP_farf.h"
#include "HAP_compute_res.h"
#include "HAP_power.h"
#include "HAP_perf.h"

#include "int16_matmul.h"

/* Write results to a host-side file via FastRPC-forwarded fopen.
 * FARF goes to logcat (needs .farf config) and printf isn't captured,
 * but fopen on a host path works. */
static FILE *g_out;
#define LOG(...) do { \
    FARF(ALWAYS, __VA_ARGS__); \
    if (g_out) { fprintf(g_out, __VA_ARGS__); fprintf(g_out, "\n"); fflush(g_out); } \
} while (0)

#define TILE_ELEMS 1024
#define F16_ONE    0x3C00

/* ---------- Power + VTCM init (ch02 pattern) ---------- */
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

/* ---------- fp16 matmul one 32x32x32 tile (HMX only) ---------- */
static inline __attribute__((always_inline))
void fp16_matmul_tile(const uint16_t *act, const uint16_t *wt,
                      const uint8_t *scl, uint16_t *out)
{
    asm volatile("bias = mxmem2(%0)"                         :: "r"(scl) : "memory");
    asm volatile("mxclracc.hf"                               ::: "memory");
    asm volatile("{ activation.hf = mxmem(%0,%1)\n"
                 "  weight.hf     = mxmem(%2,%3) }"
                 :: "r"(act), "r"(2047), "r"(wt), "r"(2047)  : "memory");
    asm volatile("mxmem(%0,%1):after.hf = acc"
                 :: "r"(out), "r"(0)                          : "memory");
}

/* ---------- Fill helpers (HVX) ---------- */
static void fill_f16(uint16_t *buf, uint16_t val)
{
    int s = ((int)val << 16) | val;
    HVX_Vector v = Q6_V_vsplat_R(s);
    for (int i = 0; i < TILE_ELEMS / 64; i++) ((HVX_Vector *)buf)[i] = v;
}
static void fill_scales_f16(uint8_t *buf, uint16_t val)
{
    int s = ((int)val << 16) | val;
    ((HVX_Vector *)buf)[0] = Q6_V_vsplat_R(s);
    ((HVX_Vector *)buf)[1] = Q6_V_vzero();
}

/* ---------- int16 HMX-only inner macro (exactly 6 HMX packets) ---------- */
#define ONE_INT16_PARTIAL(act_tile, wt_tile, bias_lo, bias_hi, out_lo, out_hi) \
    do {                                                                      \
        asm volatile("bias = mxmem(%0)" :: "r"(bias_lo) : "memory");          \
        asm volatile("mxclracc" ::: "memory");                                \
        asm volatile("{ activation.ub = mxmem(%0,%1)\n"                       \
                     "  weight.b      = mxmem(%2,%3) }"                       \
                     :: "r"(act_tile), "r"(2047),                             \
                        "r"(wt_tile),  "r"(2047) : "memory");                 \
        asm volatile("mxmem(%0,%1):after:retain.uh = acc:2x1"                 \
                     :: "r"(out_lo), "r"(0) : "memory");                      \
        asm volatile("bias = mxmem(%0)" :: "r"(bias_hi) : "memory");          \
        asm volatile("mxmem(%0,%1):after.uh = acc:2x1"                        \
                     :: "r"(out_hi), "r"(0) : "memory");                      \
    } while (0)

/* ---------- main ---------- */
int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    int ret;

    /* Open host-side result file (path relative to the FastRPC user PD's
     * working dir on the ARM side, which is wherever run_main_on_hexagon
     * was invoked from). */
    g_out = fopen("./bench_result.txt", "w");

    LOG("=== matmul pcycle bench (SM8650 v75 HTP) ===");

    /* Step 1: power */
    if ((ret = power_on_hvx_hmx()) != 0) {
        LOG("[Power] FAILED ret=%d", ret);
        return 1;
    }

    /* Step 2: VTCM + HMX lock */
    unsigned int vtcm_size = 8 * 1024 * 1024;
    HAP_compute_res_query_VTCM(0, &vtcm_size, NULL, NULL, NULL);

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_vtcm_param(&attr, vtcm_size, 1);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);

    unsigned int ctx_id = HAP_compute_res_acquire(&attr, 100000);
    if (ctx_id == 0) {
        LOG("[Init] compute_res_acquire returned 0");
        return 1;
    }
    void *vtcm_base = HAP_compute_res_attr_get_vtcm_ptr(&attr);
    if (!vtcm_base) {
        LOG("[Init] VTCM ptr NULL");
        HAP_compute_res_release(ctx_id);
        return 1;
    }
    if ((ret = HAP_compute_res_hmx_lock(ctx_id)) != 0) {
        LOG("[Init] hmx_lock ret=%d", ret);
        HAP_compute_res_release(ctx_id);
        return 1;
    }
    LOG("[Init] VTCM=%p size=%u HMX locked", vtcm_base, vtcm_size);

    uint8_t *vt = (uint8_t *)vtcm_base;

    /* ======================================================
     * Phase A: fp16 matmul
     * ====================================================== */
    {
        uint16_t *act = (uint16_t *)(vt + 0 * 2048);
        uint16_t *wt  = (uint16_t *)(vt + 1 * 2048);
        uint8_t  *scl =              vt + 2 * 2048;
        uint16_t *out = (uint16_t *)(vt + 3 * 2048);

        fill_f16(act, F16_ONE);
        fill_f16(wt,  F16_ONE);
        fill_scales_f16(scl, F16_ONE);

        fp16_matmul_tile(act, wt, scl, out); /* warmup */

        uint64_t c0 = HAP_perf_get_pcycles();
        fp16_matmul_tile(act, wt, scl, out);
        uint64_t c1 = HAP_perf_get_pcycles();
        LOG("[fp16] single tile: %llu pcycles (out[0]=0x%04x)",
             (unsigned long long)(c1 - c0), out[0]);

        c0 = HAP_perf_get_pcycles();
        for (int i = 0; i < 100; i++) fp16_matmul_tile(act, wt, scl, out);
        c1 = HAP_perf_get_pcycles();
        LOG("[fp16] 100x tile: %llu pcycles (avg %llu/tile)",
             (unsigned long long)(c1 - c0),
             (unsigned long long)((c1 - c0) / 100));
    }

    /* ======================================================
     * Phase B: int16 full kernel (our dual-scale readback)
     * ====================================================== */
    {
        /* Int16 kernel workspace at offset 10*2048 to stay clear of
         * the fp16 tiles above. Kernel uses vtcm_base + scratches. */
        void *int16_vtcm = vt + 10 * 2048;

        static int16_t a16[1024], w16[1024], out16[1024];
        for (int i = 0; i < 1024; i++) {
            a16[i] = (int16_t)((i * 37) & 0x7FFF);
            w16[i] = (int16_t)((i * 53) & 0x7FFF);
        }
        im_requant_t rq = { .mul = 1, .shift = 20 };

        im_matmul_hmx_i8(out16, a16, w16, rq, int16_vtcm); /* warmup */

        uint64_t c0 = HAP_perf_get_pcycles();
        im_matmul_hmx_i8(out16, a16, w16, rq, int16_vtcm);
        uint64_t c1 = HAP_perf_get_pcycles();
        LOG("[int16 full] single: %llu pcycles (out[0]=%d)",
             (unsigned long long)(c1 - c0), (int)out16[0]);

        c0 = HAP_perf_get_pcycles();
        for (int i = 0; i < 100; i++)
            im_matmul_hmx_i8(out16, a16, w16, rq, int16_vtcm);
        c1 = HAP_perf_get_pcycles();
        LOG("[int16 full] 100x: %llu pcycles (avg %llu/tile)",
             (unsigned long long)(c1 - c0),
             (unsigned long long)((c1 - c0) / 100));
    }

    /* ======================================================
     * Phase C: int16 HMX-only (24 HMX packets per "tile", no CPU)
     * ====================================================== */
    {
        uint8_t  *act_tile = vt + 20 * 2048;
        int8_t   *wt_tile  = (int8_t *)(vt + 21 * 2048);
        uint16_t *bias_lo  = (uint16_t *)(vt + 22 * 2048);
        uint16_t *bias_hi  = (uint16_t *)(vt + 22 * 2048 + 256);
        uint16_t *out_lo   = (uint16_t *)(vt + 23 * 2048);
        uint16_t *out_hi   = (uint16_t *)(vt + 24 * 2048);

        memset(act_tile, 0, 2048);
        memset(wt_tile,  0, 1024);
        for (int i = 0; i < 128; i++) { bias_lo[i] = 0x4000; bias_hi[i] = 0x2000; }

        /* warmup */
        ONE_INT16_PARTIAL(act_tile, wt_tile, bias_lo, bias_hi, out_lo, out_hi);

        uint64_t c0 = HAP_perf_get_pcycles();
        for (int i = 0; i < 100; i++) {
            ONE_INT16_PARTIAL(act_tile, wt_tile, bias_lo, bias_hi, out_lo, out_hi);
            ONE_INT16_PARTIAL(act_tile, wt_tile, bias_lo, bias_hi, out_lo, out_hi);
            ONE_INT16_PARTIAL(act_tile, wt_tile, bias_lo, bias_hi, out_lo, out_hi);
            ONE_INT16_PARTIAL(act_tile, wt_tile, bias_lo, bias_hi, out_lo, out_hi);
        }
        uint64_t c1 = HAP_perf_get_pcycles();
        LOG("[int16 HMX-only] 100x tile (4 partials): %llu pcycles  (avg %llu/tile, %llu/partial)",
             (unsigned long long)(c1 - c0),
             (unsigned long long)((c1 - c0) / 100),
             (unsigned long long)((c1 - c0) / 400));
    }

    if (g_out) { fclose(g_out); g_out = NULL; }
    HAP_compute_res_hmx_unlock(ctx_id);
    HAP_compute_res_release(ctx_id);
    LOG("=== done ===");
    return 0;
}
