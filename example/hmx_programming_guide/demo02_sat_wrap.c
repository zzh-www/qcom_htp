/*
 * demo02_sat_wrap.c — convert 的 :sat vs wrap 对比。
 *
 * 两路: CPU fill vs HVX fill (都跑同一 MAC 和同一 dual-convert)。
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>
#include "hmx_hvx_common.h"

static void cpu_fill_all(uint8_t *act, int8_t *wt, uint16_t *bias, uint8_t av, int8_t wv) {
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
    for (int i = 0; i < 128; i++) bias[i] = 0x4000;
}

static void hvx_fill_all(uint8_t *act, int8_t *wt, uint16_t *bias, uint8_t av, int8_t wv) {
    /* act: 4-byte pattern {0, av, 0, av} (little-endian) */
    hvx_zero(act, 2048);
    uint32_t pat_a = ((uint32_t)(uint8_t)av << 8) | ((uint32_t)(uint8_t)av << 24);
    HVX_Vector v_a = Q6_V_vsplat_R((int)pat_a);
    for (int pr = 0; pr < 16; pr++) hvx_vstu(act + 128*pr, v_a);
    /* wt: 4-byte pattern {wv,wv,wv,wv} */
    uint32_t pat_w = ((uint32_t)(uint8_t)wv * 0x01010101u);
    HVX_Vector v_w = Q6_V_vsplat_R((int)pat_w);
    for (int kg = 0; kg < 8; kg++) hvx_vstu((uint8_t *)wt + 128*kg, v_w);
    hvx_fill_u16(bias, 128, 0x4000);
}

static void run_pair(const uint8_t *act, const int8_t *wt, const uint16_t *bias,
                     uint16_t *out1, uint16_t *out2)
{
    asm volatile("mxclracc" ::: "memory");
    asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
    asm volatile("{ activation.ub = mxmem(%0,%1)\n"
                 "  weight.b      = mxmem(%2,%3) }"
                 :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
    asm volatile("mxmem(%0,%1):after:retain.uh = acc:2x1"
                 :: "r"(out1), "r"(0) : "memory");
    asm volatile("mxmem(%0,%1):after:sat.uh = acc:2x1"
                 :: "r"(out2), "r"(0) : "memory");
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
    uint16_t *out1 = (uint16_t *)(vt + 6*2048);
    uint16_t *out2 = (uint16_t *)(vt + 8*2048);

    uint32_t acc_exp = 32u * 255u * 127u;
    uint16_t wrap_exp = (uint16_t)(acc_exp & 0xFFFF);
    uint16_t sat_exp  = 0xFFFF;

    printf("--- demo02: sat vs wrap ---\n");
    printf("  acc = 32·255·127 = %u (0x%05x)\n", acc_exp, acc_exp);

    int fail = 0;

    /* Path 1: CPU fill */
    cpu_fill_all(act, wt, bias, 255, 127);
    run_pair(act, wt, bias, out1, out2);
    printf("  CPU: wrap=%u sat=%u  expect %u / %u  %s\n",
           out1[0], out2[0], wrap_exp, sat_exp,
           (out1[0] == wrap_exp && out2[0] == sat_exp) ? "OK" : "FAIL");
    if (out1[0] != wrap_exp || out2[0] != sat_exp) fail++;

    /* Path 2: HVX fill */
    hvx_fill_all(act, wt, bias, 255, 127);
    run_pair(act, wt, bias, out1, out2);
    printf("  HVX: wrap=%u sat=%u  expect %u / %u  %s\n",
           out1[0], out2[0], wrap_exp, sat_exp,
           (out1[0] == wrap_exp && out2[0] == sat_exp) ? "OK" : "FAIL");
    if (out1[0] != wrap_exp || out2[0] != sat_exp) fail++;

    if (!fail) printf("  [PASS] demo02 (both paths)\n");
    else       printf("  [FAIL] demo02\n");

    h2_thread_stop(fail);
    return fail ? 1 : 0;
}
