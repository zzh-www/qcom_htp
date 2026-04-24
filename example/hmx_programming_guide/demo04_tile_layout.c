/*
 * demo04_tile_layout.c — single-hot-byte probe 验证 tile 字节布局。
 *
 * 两路: CPU (memset) vs HVX (hvx_zero)。两路都跑 5 个探针，都应 PASS。
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>
#include "hmx_hvx_common.h"

typedef void (*zero_fn)(void *, int);
static void cpu_zero(void *p, int n) { memset(p, 0, n); }
static void hvx_zero_wrap(void *p, int n) { hvx_zero(p, n); }

static int probe_single(uint8_t *act, int8_t *wt, uint16_t *bias, uint16_t *out,
                        zero_fn zf,
                        int act_off, uint8_t av, int wt_off, int8_t wv,
                        int expect_ir, int expect_jc, uint16_t expect_val)
{
    zf(act, 2048); zf(wt, 1024); zf(out, 2048);
    act[act_off] = av;
    wt[wt_off]   = wv;

    asm volatile("mxclracc" ::: "memory");
    asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
    asm volatile("{ activation.ub = mxmem(%0,%1)\n"
                 "  weight.b      = mxmem(%2,%3) }"
                 :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
    asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
                 :: "r"(out), "r"(0) : "memory");

    int want_idx = (expect_ir & 15) * 64 + 2 * expect_jc + (expect_ir >> 4);
    int nonzero = 0, first = -1;
    for (int i = 0; i < 1024; i++) {
        if (out[i]) { if (first < 0) first = i; nonzero++; }
    }
    int ok = (expect_val == 0) ? (nonzero == 0)
                               : (nonzero == 1 && first == want_idx && out[want_idx] == expect_val);
    return ok ? 0 : 1;
}

static int run_all(uint8_t *act, int8_t *wt, uint16_t *bias, uint16_t *out, zero_fn zf)
{
    int fail = 0;
    /* A(0,0)·W(0,0)=1, 应在 C[0,0] 出现 */
    fail += probe_single(act, wt, bias, out, zf, 1, 1, 0, 1, 0, 0, 1);
    /* A(16,0)·W(0,0)=1, 应在 C[16,0] */
    fail += probe_single(act, wt, bias, out, zf, 3, 1, 0, 1, 16, 0, 1);
    /* A(3,5)·W(5,7): act_off=405, wt_off=157, C[3,7]=1 */
    fail += probe_single(act, wt, bias, out, zf, 405, 1, 157, 1, 3, 7, 1);
    /* A(31,31)·W(31,31): act_off=2047, wt_off=1023, C[31,31]=1 */
    fail += probe_single(act, wt, bias, out, zf, 2047, 1, 1023, 1, 31, 31, 1);
    /* K mismatch: A 在 K=3, W 在 K=5 -> 0 */
    int a_off = 128*0 + 4*3 + 1;   /* K=3 */
    int w_off = 128*(5>>2) + 4*0 + (5&3);  /* K=5 col=0 */
    fail += probe_single(act, wt, bias, out, zf, a_off, 1, w_off, 1, 0, 0, 0);
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
    uint8_t  *act  = vt + 0*2048;
    int8_t   *wt   = (int8_t *)(vt + 2*2048);
    uint16_t *bias = (uint16_t *)(vt + 4*2048);
    uint16_t *out  = (uint16_t *)(vt + 6*2048);

    /* bias 用 HVX 填，所有 probe 共用 */
    hvx_fill_u16(bias, 128, 0x4000);

    printf("--- demo04: tile layout probe ---\n");
    int f_cpu = run_all(act, wt, bias, out, cpu_zero);
    int f_hvx = run_all(act, wt, bias, out, hvx_zero_wrap);
    printf("  CPU zero: %d failures / 5\n", f_cpu);
    printf("  HVX zero: %d failures / 5\n", f_hvx);

    int fail = f_cpu + f_hvx;
    if (!fail) printf("  [PASS] demo04 (both paths)\n");
    else       printf("  [FAIL] demo04\n");
    h2_thread_stop(fail);
    return fail ? 1 : 0;
}
