/*
 * demo01_hello_hmx.c — HMX 编程指南第一个 demo
 *
 * 两个路径:
 *   (1) CPU: memset + scalar for-loop 填 act/wt/bias
 *   (2) HVX: hvx_fill / hvx_zero 填 act/wt/bias
 * HMX MAC 指令完全相同。期望两路都跑出 out[0] == 32。
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>
#include "hmx_hvx_common.h"

static void hmx_mac(const uint8_t *act, const int8_t *wt,
                    const uint16_t *bias, uint16_t *out)
{
    asm volatile("mxclracc" ::: "memory");
    asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
    asm volatile("{ activation.ub = mxmem(%0,%1)\n"
                 "  weight.b      = mxmem(%2,%3) }"
                 :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
    asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
                 :: "r"(out), "r"(0) : "memory");
}

/* ---- Fillers ---- */
static void cpu_setup(uint8_t *act, int8_t *wt, uint16_t *bias)
{
    memset(act, 0, 2048);
    for (int pr = 0; pr < 16; pr++)
        for (int K = 0; K < 32; K++) {
            act[128*pr + 4*K + 1] = 1;
            act[128*pr + 4*K + 3] = 1;
        }
    memset(wt, 0, 1024);
    for (int K = 0; K < 32; K++)
        for (int c = 0; c < 32; c++)
            wt[128*(K>>2) + 4*c + (K&3)] = 1;
    for (int i = 0; i < 128; i++) bias[i] = 0x4000;
}

static void hvx_setup(uint8_t *act, int8_t *wt, uint16_t *bias)
{
    /* 用同一数学布局，但用 HVX fill 加速"写成片的常量" */
    hvx_zero(act, 2048);
    /* act 填 1 到 4k+1 和 4k+3 位置：每 4 字节 = 0x01010000 (little-endian u32) */
    /* 展开为 HVX: 每 phys_row 写 1 条 128-B 线 */
    HVX_Vector v_pat = Q6_V_vsplat_R(0x01000100);
    for (int pr = 0; pr < 16; pr++) {
        hvx_vstu(act + 128*pr, v_pat);
    }
    /* wt: 每 4 字节 = 0x01010101 (全 K 全 col = 1) */
    HVX_Vector v_wt = Q6_V_vsplat_R(0x01010101);
    for (int kg = 0; kg < 8; kg++)
        hvx_vstu((uint8_t *)wt + 128*kg, v_wt);
    /* bias 128 × u16 splat */
    hvx_fill_u16(bias, 128, 0x4000);
}

static int verify_out(const uint16_t *out)
{
    int fail = 0;
    if (out[0]              != 32) fail++;   /* (ir=0, jc=0) */
    if (out[0*64 + 2*1 + 0] != 32) fail++;   /* (ir=0, jc=1) */
    if (out[0*64 + 2*0 + 1] != 32) fail++;   /* (ir=16, jc=0) */
    if (out[15*64 + 2*31 + 1] != 32) fail++; /* (ir=31, jc=31) */
    return fail;
}

int main(void)
{
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
    uint8_t  *act  = vt + 0 * 2048;
    int8_t   *wt   = (int8_t *)(vt + 2 * 2048);
    uint16_t *bias = (uint16_t *)(vt + 4 * 2048);
    uint16_t *out  = (uint16_t *)(vt + 6 * 2048);

    printf("--- demo01: Hello HMX ---\n");
    int fail = 0;

    /* Path 1: CPU */
    cpu_setup(act, wt, bias);
    memset(out, 0, 2048);
    hmx_mac(act, wt, bias, out);
    int f1 = verify_out(out);
    printf("  CPU setup: out[0]=%u   %s\n", out[0], f1 ? "FAIL" : "OK");
    fail += f1;

    /* Path 2: HVX */
    hvx_setup(act, wt, bias);
    hvx_zero(out, 2048);
    hmx_mac(act, wt, bias, out);
    int f2 = verify_out(out);
    printf("  HVX setup: out[0]=%u   %s\n", out[0], f2 ? "FAIL" : "OK");
    fail += f2;

    if (!fail) printf("  [PASS] demo01 (both paths)\n");
    else       printf("  [FAIL] demo01 (cpu=%d hvx=%d)\n", f1, f2);

    h2_thread_stop(fail);
    return fail ? 1 : 0;
}
