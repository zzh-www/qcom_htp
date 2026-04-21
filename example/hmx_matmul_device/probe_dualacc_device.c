/*
 * probe_dualacc_device.c — isolates the semantics of `mxswapacc` and the
 * `:above` modifier on HMX activation loads. Prior RE assumed `:above`
 * routes the MAC to the "other" (non-current) accumulator; implementations
 * built on that hypothesis produce wrong output at K≥64.
 *
 * Each probe: clear acc, fire a known MAC pattern, read back with known
 * modifier, print both output words. Activation is all-1s (fill_act), weight
 * all-1s (0x01). Expected plain-MAC output = 32 (sum of 32 A·W across one
 * MAC packet).
 *
 * Readback pattern: we read two scale bands to get 16-bit lo and 16-bit hi,
 * then reconstruct the signed int24 (dual-scale readback mechanic). Sign-
 * aware so negative accumulator values are preserved.
 *
 * Test matrix (each row is one run: clear accs, do the op sequence, read both):
 *   T0  baseline              clracc; MAC_plain; store_current             → 32
 *   T1  above only            clracc; MAC_above; store_current
 *   T2  above + swap          clracc; MAC_above; swapacc; store_current
 *   T3  swap only             clracc; swapacc;   store_current
 *   T4  both-clear, MAC       clracc; swapacc; clracc; swapacc; MAC; store
 *   T5  both-clear, MAC_above clracc; swapacc; clracc; swapacc; MAC_above; store
 *   T6  both-clear, above+swap   clracc; swapacc; clracc; swapacc; MAC_above; swapacc; store
 *   T7  both-clear, swap+MAC  clracc; swapacc; clracc; swapacc; swapacc; MAC; store
 *   T8  two MACs same acc     clracc; MAC_plain; MAC_plain; store           → 64
 *   T9  plain+above, no swap  clracc; MAC_plain; MAC_above; store           → ?
 *   T10 above+plain, no swap  clracc; MAC_above; MAC_plain; store           → ?
 *   T11 pair+swap             clracc; MAC_above; MAC_plain; swapacc; store
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
    req.dcvs_v3.set_dcvs_enable = 1; req.dcvs_v3.dcvs_enable = 1;
    req.dcvs_v3.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
    req.dcvs_v3.set_bus_params = 1;
    req.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_core_params = 1;
    req.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    req.dcvs_v3.set_sleep_disable = 1; req.dcvs_v3.sleep_disable = 1;
    if (HAP_power_set((void *)&power_ctx, &req) != 0) return -2;
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HVX; req.hvx.power_up = 1;
    if (HAP_power_set((void *)&power_ctx, &req) != 0) return -3;
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_HMX; req.hmx.power_up = 1;
    if (HAP_power_set((void *)&power_ctx, &req) != 0) return -4;
    return 0;
}

static void fill_act(uint8_t *tile, uint8_t val)
{
    memset(tile, 0, 2048);
    for (int r = 0; r < 16; r++)
        for (int k = 0; k < 32; k++) {
            tile[128 * r + 4 * k + 1] = val;
            tile[128 * r + 4 * k + 3] = val;
        }
}

/* Simplified readback: read current acc via dual-scale to two u16 tiles,
 * reconstruct signed value in (phys_row=0, stream=0, col=0). */
static int32_t readback_current(uint16_t *out_lo, uint16_t *out_hi)
{
    asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
                 :: "r"(out_hi), "r"(0) : "memory");
    /* out_hi got the current acc at scale 2^-8 truncated.
     * For small values, [0,0] of out_hi = acc & 0xFFFF approx. */
    (void)out_lo;
    return (int32_t)(int16_t)out_hi[0];
}

/* Stronger readback: use two different biases so we get both the low byte
 * (scale 0x4000 = 1.0) and high byte (scale 0x2000 = 0.5) of the int24 acc,
 * then reconstruct. This matches the kernel's dual-scale readback. */
static int32_t dual_scale_readback(uint16_t *bias_lo, uint16_t *bias_hi,
                                    uint16_t *out_lo, uint16_t *out_hi)
{
    asm volatile("bias = mxmem(%0)" :: "r"(bias_lo) : "memory");
    asm volatile("mxmem(%0,%1):after:retain.uh = acc:2x1"
                 :: "r"(out_lo), "r"(0) : "memory");
    asm volatile("bias = mxmem(%0)" :: "r"(bias_hi) : "memory");
    asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
                 :: "r"(out_hi), "r"(0) : "memory");
    uint16_t lo = out_lo[0];
    uint16_t hi = out_hi[0];
    return ((int32_t)(int16_t)hi << 8) | ((int32_t)lo & 0xFF);
}

static inline void mac_plain(const uint8_t *act, const int8_t *wt)
{
    asm volatile("{ activation.ub = mxmem(%0,%1)\n"
                 "  weight.b      = mxmem(%2,%3) }"
                 :: "r"(act), "r"(2047), "r"(wt), "r"(0x3ff) : "memory");
}
static inline void mac_above(const uint8_t *act, const int8_t *wt)
{
    asm volatile("{ activation.ub = mxmem(%0,%1):above\n"
                 "  weight.b      = mxmem(%2,%3) }"
                 :: "r"(act), "r"(2047), "r"(wt), "r"(0x3ff) : "memory");
}
static inline void load_bias(const uint16_t *p) { asm volatile("bias = mxmem(%0)" :: "r"(p) : "memory"); }
static inline void clr_acc(void)  { asm volatile("mxclracc" ::: "memory"); }
static inline void swap_acc(void) { asm volatile("mxswapacc" ::: "memory"); }

/* Read CURRENT acc — does not change "other". Uses :after.uh = acc:2x1 with
 * bias=0x4000 (low byte of int24). Caller must bias-load beforehand. */
static inline void store_current(uint16_t *out)
{
    asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
                 :: "r"(out), "r"(0) : "memory");
}

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    g_out = fopen("./probe_dualacc_result.txt", "w");
    LOG("=== HMX :above/mxswapacc semantics probe (SM8650 v75) ===");
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

    uint8_t  *act0 = vtcm + 0;
    int8_t   *wt   = (int8_t  *)(vtcm + 4096);
    uint16_t *bias_lo = (uint16_t *)(vtcm + 5120);
    uint16_t *bias_hi = (uint16_t *)(vtcm + 5376);
    uint16_t *out  = (uint16_t *)(vtcm + 6144);
    uint16_t *out2 = (uint16_t *)(vtcm + 8192);

    fill_act(act0, 1);
    memset(wt, 1, 1024);
    for (int i = 0; i < 128; i++) { bias_lo[i] = 0x4000; bias_hi[i] = 0x2000; }

    LOG("  (A=1, W=1, single MAC packet → plain acc = 32)");

    /* T0: plain, just one MAC */
    load_bias(bias_lo); clr_acc();
    mac_plain(act0, wt);
    memset(out, 0, 2048); store_current(out);
    LOG("  T0  plain MAC, store current          : %u", out[0]);

    /* T1: above MAC */
    load_bias(bias_lo); clr_acc();
    mac_above(act0, wt);
    memset(out, 0, 2048); store_current(out);
    LOG("  T1  above MAC, store current          : %u", out[0]);

    /* T2: above MAC + swap, store (expectation if above→other: 32; if above→current: swap's result → 0) */
    load_bias(bias_lo); clr_acc();
    mac_above(act0, wt); swap_acc();
    memset(out, 0, 2048); store_current(out);
    LOG("  T2  above MAC, swap, store            : %u", out[0]);

    /* T3: no MAC, just swap (expectation: 0, since other acc was never cleared or MACed) */
    load_bias(bias_lo); clr_acc();
    swap_acc();
    memset(out, 0, 2048); store_current(out);
    LOG("  T3  swap only (no MAC), store         : %u (garbage if other never cleared)", out[0]);

    /* T4: both accs cleared, then one plain MAC */
    load_bias(bias_lo); clr_acc(); swap_acc(); clr_acc(); swap_acc();
    mac_plain(act0, wt);
    memset(out, 0, 2048); store_current(out);
    LOG("  T4  both-clear + plain, store current : %u", out[0]);

    /* T5: both cleared, one above MAC */
    load_bias(bias_lo); clr_acc(); swap_acc(); clr_acc(); swap_acc();
    mac_above(act0, wt);
    memset(out, 0, 2048); store_current(out);
    LOG("  T5  both-clear + above, store current : %u", out[0]);

    /* T6: both cleared, above + swap, store (if above→other, swap makes it current → store=32) */
    load_bias(bias_lo); clr_acc(); swap_acc(); clr_acc(); swap_acc();
    mac_above(act0, wt); swap_acc();
    memset(out, 0, 2048); store_current(out);
    LOG("  T6  both-clear + above + swap, store  : %u", out[0]);

    /* T7: both cleared, two swaps (no MAC). Expected 0 both accs. */
    load_bias(bias_lo); clr_acc(); swap_acc(); clr_acc(); swap_acc();
    swap_acc();
    memset(out, 0, 2048); store_current(out);
    LOG("  T7  both-clear + extra swap           : %u", out[0]);

    /* T8: 2 plain MACs (both go to current). Expected 64. */
    load_bias(bias_lo); clr_acc();
    mac_plain(act0, wt); mac_plain(act0, wt);
    memset(out, 0, 2048); store_current(out);
    LOG("  T8  plain MAC × 2, store current      : %u (expect 64)", out[0]);

    /* T9: plain then above, no swap */
    load_bias(bias_lo); clr_acc();
    mac_plain(act0, wt); mac_above(act0, wt);
    memset(out, 0, 2048); store_current(out);
    LOG("  T9  plain+above no swap, store current: %u", out[0]);

    /* T10: above then plain, no swap */
    load_bias(bias_lo); clr_acc();
    mac_above(act0, wt); mac_plain(act0, wt);
    memset(out, 0, 2048); store_current(out);
    LOG("  T10 above+plain no swap, store current: %u", out[0]);

    /* T11: above + plain + swap, store (what the pattern assumes) */
    load_bias(bias_lo); clr_acc();
    mac_above(act0, wt); mac_plain(act0, wt); swap_acc();
    memset(out, 0, 2048); store_current(out);
    LOG("  T11 above+plain+swap, store current   : %u", out[0]);

    /* T11b: and then read the OTHER acc */
    swap_acc();
    memset(out2, 0, 2048); store_current(out2);
    LOG("  T11b +swap, store other acc           : %u", out2[0]);

    /* T12 (the pattern the kernel actually uses): clear both, above+plain+swap
     * for K=2 iters, store both. */
    load_bias(bias_lo); clr_acc(); swap_acc(); clr_acc(); swap_acc();
    mac_above(act0, wt); mac_plain(act0, wt); swap_acc();  /* kt=0-1 */
    memset(out, 0, 2048); store_current(out);              /* acc A */
    swap_acc();
    memset(out2, 0, 2048); store_current(out2);            /* acc B */
    LOG("  T12 K=2 dualacc-pattern: accA=%u accB=%u sum=%u",
        out[0], out2[0], out[0] + out2[0]);
    LOG("      (expect sum=64 if pattern equivalent to 2 plain MACs)");

    /* T13: K=4 (2 pair-iters) */
    load_bias(bias_lo); clr_acc(); swap_acc(); clr_acc(); swap_acc();
    mac_above(act0, wt); mac_plain(act0, wt); swap_acc();
    mac_above(act0, wt); mac_plain(act0, wt); swap_acc();
    memset(out, 0, 2048); store_current(out);
    swap_acc();
    memset(out2, 0, 2048); store_current(out2);
    LOG("  T13 K=4 dualacc-pattern: accA=%u accB=%u sum=%u (expect 128)",
        out[0], out2[0], out[0] + out2[0]);

    /* T14: K=8 (4 pair-iters) */
    load_bias(bias_lo); clr_acc(); swap_acc(); clr_acc(); swap_acc();
    for (int i = 0; i < 4; i++) { mac_above(act0, wt); mac_plain(act0, wt); swap_acc(); }
    memset(out, 0, 2048); store_current(out);
    swap_acc();
    memset(out2, 0, 2048); store_current(out2);
    LOG("  T14 K=8 dualacc-pattern: accA=%u accB=%u sum=%u (expect 256)",
        out[0], out2[0], out[0] + out2[0]);

    /* -------- Extra: store with :retain — does it preserve BOTH accs? ---- */
    LOG("");
    LOG("--- :retain semantics ---");

    /* T15: clear both; MAC into A; swap → B; MAC into B; store B with :retain;
     *      swap → A; store A (no retain). Expected A=32, B=32. */
    load_bias(bias_lo); clr_acc(); swap_acc(); clr_acc(); swap_acc();
    mac_plain(act0, wt);         /* A = 32 */
    swap_acc();                   /* current = B */
    mac_plain(act0, wt);         /* B = 32 */
    asm volatile("mxmem(%0,%1):after:retain.uh = acc:2x1" :: "r"(out), "r"(0) : "memory"); /* store B, RETAIN */
    swap_acc();                   /* current = A */
    memset(out2, 0, 2048);
    store_current(out2);          /* store A */
    LOG("  T15 MAC into A, swap, MAC into B, store B:retain, swap, store A");
    LOG("      out(B)=%u (expect 32)   out2(A)=%u (expect 32)", out[0], out2[0]);

    /* T16: same but BOTH stores use :retain except the last. */
    load_bias(bias_lo); clr_acc(); swap_acc(); clr_acc(); swap_acc();
    mac_plain(act0, wt);         /* A = 32 */
    swap_acc();
    mac_plain(act0, wt);         /* B = 32 */
    memset(out, 0, 2048);
    store_current(out);           /* store B, NO retain — does this clear A too? */
    swap_acc();                   /* current = A */
    memset(out2, 0, 2048);
    store_current(out2);          /* store A */
    LOG("  T16 same but store B WITHOUT retain:  out(B)=%u (expect 32)   out2(A)=%u (expect 32 if A preserved, 0 if cleared)",
        out[0], out2[0]);

    LOG("");
    LOG("=== done ===");

    HAP_compute_res_hmx_unlock(ctx_id);
    HAP_compute_res_release(ctx_id);
    if (g_out) fclose(g_out);
    return 0;
}
