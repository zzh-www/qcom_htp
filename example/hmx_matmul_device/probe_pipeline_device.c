/*
 * probe_pipeline_device.c — on-silicon isolation of HMX pipelining knobs.
 *
 * Background: we reverse-engineered `hmx_convbbb1x1_stride1 @ 0x2ea740` in
 * libQnnHtpV75Skel.so (u8·i8 1×1 conv = MatMul hot loop). The inner
 * two-packet body is:
 *
 *   { r8 = add(r8, #0x400)
 *     if (p0) r25:24 = combine(r9, r7)
 *     activation.ub = mxmem(r6, r24):cm
 *     weight.b      = mxmem(r8, r25) }
 *   { r8 += add(r25, #0x1)
 *     activation.ub = mxmem(r23, r24):cm
 *     weight.b      = mxmem(r8, r25) } :endloop0
 *
 * With Rt_act = r7|0x1c  (r7 = low-11-bits of arg struct)
 *      Rt_wt = 0x3FF     (hardcoded)
 *
 * Prior RE iterations tested :dilate, :dilate:2x, Rt sweeps, and the
 * :above+mxswapacc pattern, but never actually ran a K-accumulated loop
 * with :cm + Rt_wt=0x3FF (the QNN pattern). This probe isolates that.
 *
 * Tests (each runs one packet for functional check + N=400 packets in a
 * tight loop for cyc/packet measurement):
 *   P1  Plain                Rt_act=2047, Rt_wt=2047,  no :cm
 *   P2  Plain + Rt_wt=0x3FF  Rt_act=2047, Rt_wt=0x3FF, no :cm
 *   P3  :cm + Rt_wt=2047     Rt_act=2047|0x1c, Rt_wt=2047, :cm on act
 *   P4  :cm + Rt_wt=0x3FF    Rt_act=2047|0x1c, Rt_wt=0x3FF, :cm on act  (QNN)
 *   P5  :cm + Rt_wt=0x3FF
 *       + back-to-back       Like P4 but 2 MAC packets per loop iter
 *                            (two different activation ptrs, one weight)
 *
 * Expected if :cm is the pipelining unlock:
 *   P1 ~ P2 ~ P3 ≫ P4 ~ P5     (P4/P5 measurably faster, e.g. <3 cyc/MAC)
 *
 * Output: printed to probe_pipeline_result.txt (pulled back by runner).
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

/* Activation tile: 2 KiB, layout (phys_row in [0,16), K in [0,32), stream 0/1):
 *   byte @ 128·phys_row + 4·K + (stream ? 3 : 1).
 * Set every active byte to val; leave pad bytes 0. */
static void fill_act(uint8_t *tile, uint8_t val)
{
    memset(tile, 0, 2048);
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        for (int k = 0; k < 32; k++) {
            tile[128 * phys_row + 4 * k + 1] = val;
            tile[128 * phys_row + 4 * k + 3] = val;
        }
    }
}

/* Weight tile: 1 KiB. Every byte = val (signed int8). */
static void fill_wt(int8_t *tile, int8_t val) { memset(tile, val, 1024); }

/* Bias 128 × uint16 = scale 1.0 for integer readback. */
static void fill_bias(uint16_t *b, uint16_t v) { for (int i = 0; i < 128; i++) b[i] = v; }

/* ---------- Functional probe macros (1 packet + store) ------------------ */
/* Each variant computes one MAC packet with different Rt/modifier combos,
 * then stores result to `out` with dual-scale readback at scale 1.0. */

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

/* ---------- Throughput probe: K-accumulated MAC loop ----------------------
 *
 * One `mxclracc` + N MAC packets + readback. Models the inner K-loop of a
 * real matmul tile. Cycles/MAC = (total - overhead) / N.
 *
 * Variant naming: acc_PLAIN (no modifier), acc_CM (just :cm), acc_CM_QNN
 * (:cm + Rt_wt=0x3FF, matches QNN pattern). acc_CM_QNN_PAIR runs 2 MAC
 * packets per loop iter with two different activation pointers, mirroring
 * the actual hmx_convbbb1x1_stride1 hot loop structure.
 */

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

/* PAIR: 2 MAC packets per loop iter, each with a different activation ptr.
 * Mirrors QNN hot loop where r6 and r23 are two different M-rows. Here we
 * alternate act and act+64 (second stream's neighbor) — still valid VTCM
 * reads since activation tile is only 2 KB at `act`. */
static inline void acc_cm_pair(const uint16_t *bias, const uint8_t *act0,
                                const uint8_t *act1, const int8_t *wt,
                                uint16_t *out, int n,
                                int rt_a, int rt_w)
{
    asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
    asm volatile("mxclracc" ::: "memory");
    for (int i = 0; i < n / 2; i++) {
        asm volatile("{ activation.ub = mxmem(%0,%1):cm\n"
                     "  weight.b      = mxmem(%2,%3) }"
                     :: "r"(act0), "r"(rt_a), "r"(wt), "r"(rt_w) : "memory");
        asm volatile("{ activation.ub = mxmem(%0,%1):cm\n"
                     "  weight.b      = mxmem(%2,%3) }"
                     :: "r"(act1), "r"(rt_a), "r"(wt), "r"(rt_w) : "memory");
    }
    asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
                 :: "r"(out), "r"(0) : "memory");
}

/* ---------- Main ---------------------------------------------------------- */

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    g_out = fopen("./probe_pipeline_result.txt", "w");

    LOG("=== HMX pipelining probe (SM8650 v75) ===");
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

    /* Layout: act0 @ +0, act1 @ +2048, wt @ +4096, bias @ +5120, out @ +6144 */
    uint8_t  *act0 = vtcm + 0;
    uint8_t  *act1 = vtcm + 2048;
    int8_t   *wt   = (int8_t  *)(vtcm + 4096);
    uint16_t *bias = (uint16_t *)(vtcm + 5120);
    uint16_t *out  = (uint16_t *)(vtcm + 6144);

    fill_act(act0, 1);
    fill_act(act1, 1);
    fill_wt (wt,   1);      /* int8 +1 */
    fill_bias(bias, 0x4000);/* f16 1.0 */

    LOG("[Init] VTCM=%p  act0=+0  act1=+2048  wt=+4096  bias=+5120  out=+6144",
        vtcm);

    /* ---- Functional parity check: all variants should produce identical
     *       output bytes given A=1, W=1. Expected: acc[m,n] = 32 for all (m,n)
     *       → dual-scale readback with bias=0x4000 gives lo=32, hi=0.       */
    LOG("");
    LOG("--- Part 1: functional parity check (A=1, W=1) ---");

    memset(out, 0, 2048);
    RUN_PLAIN(bias, act0, wt, 2047,  2047,  out);
    LOG("  P1 plain Rt_a=2047  Rt_w=2047       out[0,0]=%u  [0,1]=%u",
        out[0], out[2]);

    memset(out, 0, 2048);
    RUN_PLAIN(bias, act0, wt, 2047,  0x3ff, out);
    LOG("  P2 plain Rt_a=2047  Rt_w=0x3FF      out[0,0]=%u  [0,1]=%u",
        out[0], out[2]);

    memset(out, 0, 2048);
    RUN_CM   (bias, act0, wt, 2047|0x1c, 2047,  out);
    LOG("  P3 :cm   Rt_a=2063  Rt_w=2047       out[0,0]=%u  [0,1]=%u",
        out[0], out[2]);

    memset(out, 0, 2048);
    RUN_CM   (bias, act0, wt, 2047|0x1c, 0x3ff, out);
    LOG("  P4 :cm   Rt_a=2063  Rt_w=0x3FF      out[0,0]=%u  [0,1]=%u",
        out[0], out[2]);

    /* ---- Throughput probe ----
     * N MACs accumulated into single acc, then single readback. Measures
     * the HMX MAC issue rate and any pipelining effect of :cm. */
    LOG("");
    LOG("--- Part 2: throughput (N MACs accumulated, single readback) ---");

    const int N_WARMUP = 16;
    const int N_MEAS   = 400;

    /* Warmup & measure helper — use 0-valued weight so accumulator stays
     * bounded even at N=400 (no wraparound into sat band). */
    fill_wt(wt, 0);

    uint64_t c0, c1;

    /* P1: baseline plain, Rt_a=Rt_w=2047 */
    acc_plain(bias, act0, wt, out, N_WARMUP, 2047, 2047);
    c0 = HAP_perf_get_pcycles();
    acc_plain(bias, act0, wt, out, N_MEAS, 2047, 2047);
    c1 = HAP_perf_get_pcycles();
    LOG("  P1 plain Rt_a=2047  Rt_w=2047        : %10llu pcyc  %.3f cyc/MAC",
        (unsigned long long)(c1 - c0), (double)(c1 - c0) / N_MEAS);

    /* P2: plain, Rt_w=0x3FF (QNN weight-tile-mask) */
    acc_plain(bias, act0, wt, out, N_WARMUP, 2047, 0x3ff);
    c0 = HAP_perf_get_pcycles();
    acc_plain(bias, act0, wt, out, N_MEAS, 2047, 0x3ff);
    c1 = HAP_perf_get_pcycles();
    LOG("  P2 plain Rt_a=2047  Rt_w=0x3FF       : %10llu pcyc  %.3f cyc/MAC",
        (unsigned long long)(c1 - c0), (double)(c1 - c0) / N_MEAS);

    /* P3: :cm on activation, Rt_w=2047 */
    acc_cm(bias, act0, wt, out, N_WARMUP, 2047 | 0x1c, 2047);
    c0 = HAP_perf_get_pcycles();
    acc_cm(bias, act0, wt, out, N_MEAS, 2047 | 0x1c, 2047);
    c1 = HAP_perf_get_pcycles();
    LOG("  P3 :cm   Rt_a=2063  Rt_w=2047        : %10llu pcyc  %.3f cyc/MAC",
        (unsigned long long)(c1 - c0), (double)(c1 - c0) / N_MEAS);

    /* P4: :cm + Rt_w=0x3FF (QNN pattern) */
    acc_cm(bias, act0, wt, out, N_WARMUP, 2047 | 0x1c, 0x3ff);
    c0 = HAP_perf_get_pcycles();
    acc_cm(bias, act0, wt, out, N_MEAS, 2047 | 0x1c, 0x3ff);
    c1 = HAP_perf_get_pcycles();
    LOG("  P4 :cm   Rt_a=2063  Rt_w=0x3FF (QNN) : %10llu pcyc  %.3f cyc/MAC",
        (unsigned long long)(c1 - c0), (double)(c1 - c0) / N_MEAS);

    /* P5: PAIR (2 MACs per iter, two different act ptrs) — QNN hot loop
     * structure. Tests whether alternating activation source pipelines. */
    acc_cm_pair(bias, act0, act1, wt, out, N_WARMUP, 2047 | 0x1c, 0x3ff);
    c0 = HAP_perf_get_pcycles();
    acc_cm_pair(bias, act0, act1, wt, out, N_MEAS, 2047 | 0x1c, 0x3ff);
    c1 = HAP_perf_get_pcycles();
    LOG("  P5 :cm PAIR (QNN hot loop shape)     : %10llu pcyc  %.3f cyc/MAC",
        (unsigned long long)(c1 - c0), (double)(c1 - c0) / N_MEAS);

    /* Also try :cm with Rt_a=2047 (no |0x1c) — does the 0x1c bit matter? */
    acc_cm(bias, act0, wt, out, N_WARMUP, 2047, 0x3ff);
    c0 = HAP_perf_get_pcycles();
    acc_cm(bias, act0, wt, out, N_MEAS, 2047, 0x3ff);
    c1 = HAP_perf_get_pcycles();
    LOG("  P6 :cm   Rt_a=2047  Rt_w=0x3FF       : %10llu pcyc  %.3f cyc/MAC",
        (unsigned long long)(c1 - c0), (double)(c1 - c0) / N_MEAS);

    LOG("");
    LOG("=== done ===");

    HAP_compute_res_hmx_unlock(ctx_id);
    HAP_compute_res_release(ctx_id);
    if (g_out) fclose(g_out);
    return 0;
}
