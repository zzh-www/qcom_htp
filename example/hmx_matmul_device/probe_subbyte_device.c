/*
 * probe_subbyte_device.c — on-silicon (SM8650 v75 cDSP) verification of
 * the sub-byte MAC architecture findings made on hexagon-sim ISS.
 *
 * Loaded via run_main_on_hexagon; output goes to both logcat ([DU] tag)
 * and ./probe_subbyte_result.txt in the FastRPC working dir.
 *
 * Tests:
 *   (1) K-span functional probe per weight type/byte_val pattern.
 *   (2) Cycle-per-packet throughput probe across weight types.
 *
 * Expected (if sim findings match silicon):
 *   byte_val=0x01:
 *     weight.b   out = 32
 *     weight.n   out = 16     (hi=0, lo=1)
 *     weight.c   out = 8      (c0=1, c1..3=0)
 *     weight.ubit out = 4     (bit0=1, others=0)
 *   byte_val=full:
 *     weight.n 0x11 out = 32  (both nibbles = 1)
 *     weight.c 0x55 out = 32  (all 4 crumbs = 1)
 *     weight.ubit 0xFF out = 32 (all 8 bits = 1)
 *
 *   Cycles/packet roughly equal across types (±5%).
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

/* ---- Tile filling ---- */
static void fill_act_ones(uint8_t *tile)
{
    /* Activation tile: 2 KiB, layout A(phys_row, K, stream) =
     *   128*phys_row + 4*K + (stream ? 3 : 1).  Set every logical A=1. */
    memset(tile, 0, 2048);
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        for (int K = 0; K < 32; K++) {
            tile[128 * phys_row + 4 * K + 1] = 1;
            tile[128 * phys_row + 4 * K + 3] = 1;
        }
    }
}

static void fill_bias_i(uint16_t *b, uint16_t v)
{
    for (int i = 0; i < 128; i++) b[i] = v;
}

/* ---- Functional probe helpers ---- */
#define INT_MAC_PACKET(SUFFIX, MOD, bias, act, wt, out)                 \
    do {                                                                \
        asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");       \
        asm volatile("mxclracc" ::: "memory");                          \
        asm volatile("{ activation.ub = mxmem(%0,%1)\n"                 \
                     "  weight." SUFFIX "  = mxmem(%2,%3)" MOD " }"     \
                     :: "r"(act), "r"(2047),                            \
                        "r"(wt),  "r"(2047) : "memory");                \
        asm volatile("mxmem(%0,%1):after.uh = acc:2x1"                  \
                     :: "r"(out), "r"(0) : "memory");                   \
    } while (0)

static void probe_func_bytecase(const char *name, uint8_t *wt, int nbytes,
                                uint8_t byte_val, const uint16_t *bias,
                                const uint8_t *act, uint16_t *out,
                                void (*run)(const uint16_t *, const uint8_t *,
                                            const uint8_t *, uint16_t *))
{
    memset(wt, byte_val, nbytes);
    memset(out, 0, 2048);
    run(bias, act, wt, out);
    LOG("    %s byte=0x%02x tile=%dB -> out[0,0]=%u  [0,1]=%u  [0,2]=%u  [16,0]=%u  [1,0]=%u",
        name, byte_val, nbytes,
        out[0*64 + 2*0 + 0], out[0*64 + 2*1 + 0], out[0*64 + 2*2 + 0],
        out[0*64 + 2*0 + 1], out[1*64 + 2*0 + 0]);
}

/* Per-type runners (to escape the cpp-suffix issue in cross-func macros). */
static void run_b(const uint16_t *bias, const uint8_t *act, const uint8_t *wt, uint16_t *out) {
    INT_MAC_PACKET("b", "", bias, act, wt, out);
}
static void run_n(const uint16_t *bias, const uint8_t *act, const uint8_t *wt, uint16_t *out) {
    INT_MAC_PACKET("n", "", bias, act, wt, out);
}
static void run_n_2x(const uint16_t *bias, const uint8_t *act, const uint8_t *wt, uint16_t *out) {
    INT_MAC_PACKET("n", ":2x", bias, act, wt, out);
}
static void run_c(const uint16_t *bias, const uint8_t *act, const uint8_t *wt, uint16_t *out) {
    INT_MAC_PACKET("c", "", bias, act, wt, out);
}
static void run_ubit(const uint16_t *bias, const uint8_t *act, const uint8_t *wt, uint16_t *out) {
    INT_MAC_PACKET("ubit", "", bias, act, wt, out);
}

/* ---- Throughput probe: ITERS MAC packets in a tight loop ---- */
#define ITERS 200

static uint64_t time_loop(void (*run)(const uint16_t *, const uint8_t *, const uint8_t *, uint16_t *),
                          const uint16_t *bias, const uint8_t *act, const uint8_t *wt, uint16_t *out,
                          int iters)
{
    /* Warmup. */
    run(bias, act, wt, out);

    uint64_t c0 = HAP_perf_get_pcycles();
    for (int i = 0; i < iters; i++) run(bias, act, wt, out);
    uint64_t c1 = HAP_perf_get_pcycles();
    return c1 - c0;
}

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    int ret;

    g_out = fopen("./probe_subbyte_result.txt", "w");

    LOG("=== HMX sub-byte probe on SM8650 v75 (silicon) ===");

    ret = power_on_hvx_hmx();
    if (ret) { LOG("[Power] FAILED (ret=%d)", ret); return 1; }

    unsigned int vtcm_size = 8 * 1024 * 1024;
    HAP_compute_res_query_VTCM(0, &vtcm_size, NULL, NULL, NULL);
    LOG("[Init] VTCM total = %u bytes", vtcm_size);

    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_vtcm_param(&attr, vtcm_size, 1);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);
    unsigned int ctx_id = HAP_compute_res_acquire(&attr, 100000);
    if (ctx_id == 0) { LOG("[Init] FAIL: acquire"); return 1; }

    uint8_t *vtcm_base = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&attr);
    if (!vtcm_base) { LOG("[Init] FAIL: NULL vtcm"); return 1; }
    ret = HAP_compute_res_hmx_lock(ctx_id);
    if (ret) { LOG("[Init] FAIL: hmx_lock ret=%d", ret); return 1; }
    LOG("[Init] VTCM=%p locked HMX", vtcm_base);

    uint8_t  *act  = vtcm_base + 0 * 4096;
    uint8_t  *wt   = vtcm_base + 1 * 4096;
    uint16_t *bias = (uint16_t *)(vtcm_base + 2 * 4096);
    uint16_t *out  = (uint16_t *)(vtcm_base + 3 * 4096);

    fill_act_ones(act);
    fill_bias_i(bias, 0x4000);

    /* ------ Part 1: functional K-span probe ------ */
    LOG("");
    LOG("--- Part 1: functional probe (A=1 everywhere) ---");

    LOG("  weight.b (int8):");
    probe_func_bytecase("b",   wt, 1024, 0x01, bias, act, out, run_b);
    probe_func_bytecase("b",   wt, 1024, 0x02, bias, act, out, run_b);
    probe_func_bytecase("b",   wt, 1024, 0x7F, bias, act, out, run_b);
    probe_func_bytecase("b",   wt, 1024, 0xFF, bias, act, out, run_b);

    LOG("  weight.n (int4):");
    probe_func_bytecase("n",   wt, 1024, 0x00, bias, act, out, run_n);
    probe_func_bytecase("n",   wt, 1024, 0x01, bias, act, out, run_n);
    probe_func_bytecase("n",   wt, 1024, 0x10, bias, act, out, run_n);
    probe_func_bytecase("n",   wt, 1024, 0x11, bias, act, out, run_n);
    probe_func_bytecase("n",   wt, 1024, 0x07, bias, act, out, run_n);
    probe_func_bytecase("n",   wt, 1024, 0x70, bias, act, out, run_n);
    probe_func_bytecase("n",   wt, 1024, 0x77, bias, act, out, run_n);
    probe_func_bytecase("n",   wt, 1024, 0x88, bias, act, out, run_n);

    LOG("  weight.c (int2 crumb):");
    probe_func_bytecase("c",   wt, 1024, 0x00, bias, act, out, run_c);
    probe_func_bytecase("c",   wt, 1024, 0x01, bias, act, out, run_c);
    probe_func_bytecase("c",   wt, 1024, 0x55, bias, act, out, run_c);
    probe_func_bytecase("c",   wt, 1024, 0xFF, bias, act, out, run_c);

    LOG("  weight.ubit (int1):");
    probe_func_bytecase("ubit", wt, 1024, 0x00, bias, act, out, run_ubit);
    probe_func_bytecase("ubit", wt, 1024, 0x01, bias, act, out, run_ubit);
    probe_func_bytecase("ubit", wt, 1024, 0x80, bias, act, out, run_ubit);
    probe_func_bytecase("ubit", wt, 1024, 0xFF, bias, act, out, run_ubit);

    LOG("  weight.n:2x (int4 double-width):");
    probe_func_bytecase("n:2x", wt, 2048, 0x00, bias, act, out, run_n_2x);
    probe_func_bytecase("n:2x", wt, 2048, 0x11, bias, act, out, run_n_2x);
    probe_func_bytecase("n:2x", wt, 1024, 0x11, bias, act, out, run_n_2x);

    /* ------ Part 2: cycle throughput ------ */
    LOG("");
    LOG("--- Part 2: cycle throughput (%d MAC packets per type) ---", ITERS);

    memset(wt, 0, 2048);

    uint64_t cyc_b    = time_loop(run_b,    bias, act, wt, out, ITERS);
    uint64_t cyc_n    = time_loop(run_n,    bias, act, wt, out, ITERS);
    uint64_t cyc_n_2x = time_loop(run_n_2x, bias, act, wt, out, ITERS);
    uint64_t cyc_c    = time_loop(run_c,    bias, act, wt, out, ITERS);
    uint64_t cyc_ubit = time_loop(run_ubit, bias, act, wt, out, ITERS);

    LOG("  weight.b    : %llu pcyc  (%.2f cyc/MAC)",
        (unsigned long long)cyc_b,    (double)cyc_b    / (double)ITERS);
    LOG("  weight.n    : %llu pcyc  (%.2f cyc/MAC)  ratio=%.3f",
        (unsigned long long)cyc_n,    (double)cyc_n    / (double)ITERS,
        (double)cyc_b / (double)cyc_n);
    LOG("  weight.n:2x : %llu pcyc  (%.2f cyc/MAC)  ratio=%.3f",
        (unsigned long long)cyc_n_2x, (double)cyc_n_2x / (double)ITERS,
        (double)cyc_b / (double)cyc_n_2x);
    LOG("  weight.c    : %llu pcyc  (%.2f cyc/MAC)  ratio=%.3f",
        (unsigned long long)cyc_c,    (double)cyc_c    / (double)ITERS,
        (double)cyc_b / (double)cyc_c);
    LOG("  weight.ubit : %llu pcyc  (%.2f cyc/MAC)  ratio=%.3f",
        (unsigned long long)cyc_ubit, (double)cyc_ubit / (double)ITERS,
        (double)cyc_b / (double)cyc_ubit);

    LOG("");
    LOG("=== done ===");

    HAP_compute_res_hmx_unlock(ctx_id);
    HAP_compute_res_release(ctx_id);
    if (g_out) fclose(g_out);
    return 0;
}
