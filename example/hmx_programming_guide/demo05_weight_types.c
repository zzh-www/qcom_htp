/*
 * demo05_weight_types.c — sub-byte weight types 对比。
 *
 * 两路: CPU vs HVX (fill 方式不同，其余同)。
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>
#include "hmx_hvx_common.h"

#define MAKE_RUNNER(name, SUFFIX)                                          \
static void run_##name(const uint8_t *act, const uint8_t *wt,              \
                       const uint16_t *bias, uint16_t *out)                \
{                                                                          \
    asm volatile("mxclracc" ::: "memory");                                 \
    asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");              \
    asm volatile("{ activation.ub = mxmem(%0,%1)\n"                        \
                 "  weight." SUFFIX " = mxmem(%2,%3) }"                    \
                 :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");   \
    asm volatile("mxmem(%0,%1):after.uh = acc:2x1"                         \
                 :: "r"(out), "r"(0) : "memory");                          \
}
MAKE_RUNNER(b, "b")
MAKE_RUNNER(n, "n")
MAKE_RUNNER(c, "c")
MAKE_RUNNER(ubit, "ubit")

static void cpu_fill_act_ones(uint8_t *act) {
    memset(act, 0, 2048);
    for (int pr = 0; pr < 16; pr++)
        for (int K = 0; K < 32; K++) {
            act[128*pr + 4*K + 1] = 1;
            act[128*pr + 4*K + 3] = 1;
        }
}
static void hvx_fill_act_ones(uint8_t *act) {
    hvx_zero(act, 2048);
    HVX_Vector v = Q6_V_vsplat_R(0x01000100);
    for (int pr = 0; pr < 16; pr++) hvx_vstu(act + 128*pr, v);
}

static int run_all(uint8_t *act, uint8_t *wt, uint16_t *bias, uint16_t *out, int use_hvx)
{
    struct c { const char *name; uint8_t bv; void (*run)(const uint8_t*, const uint8_t*, const uint16_t*, uint16_t*); int sz; uint16_t ex; };
    struct c cases[] = {
        {"b=0x01", 0x01, run_b,    1024, 32},
        {"b=0x7F", 0x7F, run_b,    1024, 4064},
        {"n=0x11", 0x11, run_n,    512,  32},
        {"n=0x07", 0x07, run_n,    512,  112},
        {"c=0x55", 0x55, run_c,    256,  32},
        {"u=0xFF", 0xFF, run_ubit, 128,  32},
    };
    int fail = 0;
    for (int i = 0; i < 6; i++) {
        if (use_hvx) { hvx_fill_u8(wt, 2048, cases[i].bv); hvx_zero(out, 2048); }
        else         { memset(wt, cases[i].bv, cases[i].sz); memset(out, 0, 2048); }
        cases[i].run(act, wt, bias, out);
        if (out[0] != cases[i].ex) fail++;
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
    uint8_t  *wt   = vt + 2*2048;
    uint16_t *bias = (uint16_t *)(vt + 4*2048);
    uint16_t *out  = (uint16_t *)(vt + 6*2048);

    printf("--- demo05: sub-byte weight types ---\n");
    int fail = 0;

    cpu_fill_act_ones(act);
    for (int i = 0; i < 128; i++) bias[i] = 0x4000;
    int f_cpu = run_all(act, wt, bias, out, 0);
    printf("  CPU: %d failures / 6\n", f_cpu);
    fail += f_cpu;

    hvx_fill_act_ones(act);
    hvx_fill_u16(bias, 128, 0x4000);
    int f_hvx = run_all(act, wt, bias, out, 1);
    printf("  HVX: %d failures / 6\n", f_hvx);
    fail += f_hvx;

    if (!fail) printf("  [PASS] demo05 (both paths)\n");
    else       printf("  [FAIL] demo05\n");
    h2_thread_stop(fail);
    return fail ? 1 : 0;
}
