/*
 * demo03_bias_scales.c — 多种 bias 值扫，CPU fill 和 HVX fill 都跑。
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>
#include "hmx_hvx_common.h"

static void cpu_fill_act_wt(uint8_t *act, int8_t *wt, uint8_t av, int8_t wv) {
    memset(act, 0, 2048);
    for (int pr = 0; pr < 16; pr++)
        for (int K = 0; K < 32; K++) {
            act[128*pr + 4*K + 1] = av;
            act[128*pr + 4*K + 3] = av;
        }
    memset(wt, 0, 1024);
    for (int K = 0; K < 32; K++)
        for (int c = 0; c < 32; c++)
            wt[128*(K>>2) + 4*c + (K&3)] = wv;
}
static void hvx_fill_act_wt(uint8_t *act, int8_t *wt, uint8_t av, int8_t wv) {
    hvx_zero(act, 2048);
    HVX_Vector v_a = Q6_V_vsplat_R((int)(((uint32_t)av << 8) | ((uint32_t)av << 24)));
    for (int pr = 0; pr < 16; pr++) hvx_vstu(act + 128*pr, v_a);
    HVX_Vector v_w = Q6_V_vsplat_R((int)((uint32_t)(uint8_t)wv * 0x01010101u));
    for (int kg = 0; kg < 8; kg++) hvx_vstu((uint8_t *)wt + 128*kg, v_w);
}

static int run_scale_sweep(uint8_t *act, int8_t *wt, uint16_t *bias, uint16_t *out, int use_hvx)
{
    uint32_t acc = 160000u;   /* 32·100·50 */

    struct { uint16_t bias; double eff; } rows[] = {
        { 0x4000, 1.0 },
        { 0x3C00, 0.5 },
        { 0x2000, 1.0/256.0 },
        { 0x0800, 1.0/16384.0 },
        { 0x0400, 1.0/32768.0 },
    };
    int N = sizeof(rows)/sizeof(rows[0]);

    int fail = 0;
    for (int i = 0; i < N; i++) {
        if (use_hvx) hvx_fill_u16(bias, 128, rows[i].bias);
        else for (int k = 0; k < 128; k++) bias[k] = rows[i].bias;

        asm volatile("mxclracc" ::: "memory");
        asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
        asm volatile("{ activation.ub = mxmem(%0,%1)\n"
                     "  weight.b      = mxmem(%2,%3) }"
                     :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
        asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
                     :: "r"(out), "r"(0) : "memory");

        uint16_t exp = (uint16_t)((uint32_t)(acc * rows[i].eff) & 0xFFFF);
        if (out[0] != exp) fail++;
    }
    return fail;
}

int main(void) {
    unsigned int vtcm = h2_info(INFO_VTCM_BASE);
    if (!vtcm) { printf("[FAIL] no VTCM\n"); h2_thread_stop(1); return 1; }
    h2_vecaccess_state_t va;
    h2_vecaccess_unit_init(&va, H2_VECACCESS_HVX_128, CFG_TYPE_VXU0,
                           CFG_SUBTYPE_VXU0, CFG_HVX_CONTEXTS, 0x1);
    h2_vecaccess_acquire(&va);
    h2_mxaccess_state_t ma;
    h2_mxaccess_unit_init(&ma, CFG_TYPE_VXU0, CFG_SUBTYPE_VXU0,
                          CFG_HMX_CONTEXTS, 0x1);
    h2_mxaccess_acquire(&ma);

    uint8_t  *vt   = (uint8_t *)(unsigned long)vtcm;
    uint8_t  *act  = vt + 0*2048;
    int8_t   *wt   = (int8_t *)(vt + 2*2048);
    uint16_t *bias = (uint16_t *)(vt + 4*2048);
    uint16_t *out  = (uint16_t *)(vt + 6*2048);

    printf("--- demo03: bias scale sweep ---\n");

    int fail = 0;
    cpu_fill_act_wt(act, wt, 100, 50);
    int f_cpu = run_scale_sweep(act, wt, bias, out, 0);
    printf("  CPU fill + sweep: %d mismatches / 5\n", f_cpu);
    fail += f_cpu;

    hvx_fill_act_wt(act, wt, 100, 50);
    int f_hvx = run_scale_sweep(act, wt, bias, out, 1);
    printf("  HVX fill + sweep: %d mismatches / 5\n", f_hvx);
    fail += f_hvx;

    if (!fail) printf("  [PASS] demo03 (both paths)\n");
    else       printf("  [FAIL] demo03\n");
    h2_thread_stop(fail);
    return fail ? 1 : 0;
}
