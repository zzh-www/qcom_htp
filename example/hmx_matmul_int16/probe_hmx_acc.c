/*
 * Probe: can a 2^-16 bias scale give us acc's high bits?
 * If f16 bias = 2^-15 (encoding 0x0200) → effective scale 2^-16,
 * then HMX output = acc / 65536.
 *
 * Target: A=100 u8, W=50 i8, K=32 → acc=160000 = 0x00027100
 *   high 16 bits = 2,  low 16 bits = 0x7100 = 28928
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>
#include <hexagon_types.h>

#define DO_CLRACC() asm volatile("mxclracc" ::: "memory")
#define DO_MM(A,W)  asm volatile("{ activation.ub = mxmem(%0,%1)\n" \
                                 "  weight.b     = mxmem(%2,%3) }"   \
                                 :: "r"(A),"r"(2047),"r"(W),"r"(2047):"memory")
#define DO_BIAS(P)  asm volatile("bias = mxmem(%0)" :: "r"(P):"memory")
#define DO_ST(O)    asm volatile("mxmem(%0,%1):after.uh = acc:2x1" \
                                 :: "r"(O),"r"(0):"memory")

static void fill_bias(uint16_t *b, uint16_t v)
{
    int splat = ((int)v << 16) | (int)v;
    HVX_Vector x = Q6_V_vsplat_R(splat);
    for (int i = 0; i < 16; i++) ((HVX_Vector *)b)[i] = x;
}

int main(void)
{
    unsigned int vtcm_base = h2_info(INFO_VTCM_BASE);
    if (!vtcm_base) { h2_thread_stop(1); return 1; }
    h2_vecaccess_state_t vacc;
    h2_vecaccess_unit_init(&vacc, H2_VECACCESS_HVX_128, CFG_TYPE_VXU0,
                           CFG_SUBTYPE_VXU0, CFG_HVX_CONTEXTS, 0x1);
    h2_vecaccess_acquire(&vacc);
    h2_mxaccess_state_t mxacc;
    h2_mxaccess_unit_init(&mxacc, CFG_TYPE_VXU0, CFG_SUBTYPE_VXU0,
                          CFG_HMX_CONTEXTS, 0x1);
    h2_mxaccess_acquire(&mxacc);

    uint8_t  *vt   = (uint8_t *)(unsigned long)vtcm_base;
    uint8_t  *A    = vt + 0 * 2048;  memset(A, 100, 2048);
    int8_t   *W    = (int8_t *)(vt + 1 * 2048); memset(W, 50, 1024);
    uint16_t *BIAS = (uint16_t *)(vt + 2 * 2048);
    uint16_t *OUT  = (uint16_t *)(vt + 5 * 2048);

    /* expected acc = 32 * 100 * 50 = 160000 = 0x00027100 */
    printf("A=100, W=50, K=32 -> acc=160000 (0x00027100)\n");
    printf("low16=0x7100 (28928), high16=0x0002 (2)\n\n");

    /* scale 1x (f16 2.0 = 0x4000) → low 16 bits */
    fill_bias(BIAS, 0x4000);
    memset(OUT, 0xEE, 4096);
    DO_CLRACC(); DO_BIAS(BIAS); DO_MM(A, W); DO_ST(OUT);
    printf("bias=f16(2.0),   first 4: %04x %04x %04x %04x\n",
           OUT[0], OUT[1], OUT[2], OUT[3]);

    /* scale 2^-16: need f16(X) with X/2 = 2^-16 → X = 2^-15.
     * f16 encoding for 2^-15: sign=0 exp=0 (denormal) mant=512 → 0x0200.
     * But denormals can be flushed.  Try normal 2^-14 (exp=1, mant=0) = 0x0400
     * too — gives scale 2^-15 = acc/32768. */
    fill_bias(BIAS, 0x0200);                 /* 2^-15 f16 → /65536 scale */
    memset(OUT, 0xEE, 4096);
    DO_CLRACC(); DO_BIAS(BIAS); DO_MM(A, W); DO_ST(OUT);
    printf("bias=f16(2^-15), first 4: %04x %04x %04x %04x  (expect ~2)\n",
           OUT[0], OUT[1], OUT[2], OUT[3]);

    fill_bias(BIAS, 0x0400);                 /* 2^-14 f16 → /32768 scale */
    memset(OUT, 0xEE, 4096);
    DO_CLRACC(); DO_BIAS(BIAS); DO_MM(A, W); DO_ST(OUT);
    printf("bias=f16(2^-14), first 4: %04x %04x %04x %04x  (expect ~4-5)\n",
           OUT[0], OUT[1], OUT[2], OUT[3]);

    fill_bias(BIAS, 0x0800);                 /* 2^-13 f16 → /16384 scale */
    memset(OUT, 0xEE, 4096);
    DO_CLRACC(); DO_BIAS(BIAS); DO_MM(A, W); DO_ST(OUT);
    printf("bias=f16(2^-13), first 4: %04x %04x %04x %04x  (expect ~9-10)\n",
           OUT[0], OUT[1], OUT[2], OUT[3]);

    /* Try with negative acc: A=100 W=-1 (=0xFF), K=32 -> acc=-3200 = 0xFFFFF380 */
    memset(W, 0xFF, 1024);
    printf("\nA=100, W=-1, K=32 -> acc=-3200 (0xFFFFF380)\n");
    fill_bias(BIAS, 0x4000);
    memset(OUT, 0xEE, 4096);
    DO_CLRACC(); DO_BIAS(BIAS); DO_MM(A, W); DO_ST(OUT);
    printf("bias=f16(2.0),   first 4: %04x %04x %04x %04x  (low=0xf380=-3200)\n",
           OUT[0], OUT[1], OUT[2], OUT[3]);

    fill_bias(BIAS, 0x0200);
    memset(OUT, 0xEE, 4096);
    DO_CLRACC(); DO_BIAS(BIAS); DO_MM(A, W); DO_ST(OUT);
    printf("bias=f16(2^-15), first 4: %04x %04x %04x %04x  (expect 0 or -1)\n",
           OUT[0], OUT[1], OUT[2], OUT[3]);

    h2_thread_stop(0);
    return 0;
}
