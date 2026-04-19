/*
 * probe_f16_profile.c — minimal single-tile f16 matmul for instruction
 * profiling via hexagon-sim --ihist.
 *
 * Does one 32x32x32 f16 matmul and prints a sentinel so we can locate
 * the kernel region in trace output.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>
#include <hexagon_types.h>

#define TILE_ELEMS 1024  /* 32x32 */
#define F16_ONE 0x3C00

static void fill_f16(uint16_t *buf, uint16_t val)
{
    int s = ((int)val << 16) | val;
    HVX_Vector v = Q6_V_vsplat_R(s);
    for (int i = 0; i < TILE_ELEMS / 64; i++) ((HVX_Vector *)buf)[i] = v;
}

/* Bias/scale tile for f16 HMX: 64 f16 scales + 128 zero bytes = 256 B. */
static void fill_scales(uint8_t *buf, uint16_t val)
{
    int s = ((int)val << 16) | val;
    ((HVX_Vector *)buf)[0] = Q6_V_vsplat_R(s);
    ((HVX_Vector *)buf)[1] = Q6_V_vzero();
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
    uint16_t *act  = (uint16_t *)(vt + 0 * 2048);
    uint16_t *wt   = (uint16_t *)(vt + 1 * 2048);
    uint8_t  *scl  =              vt + 2 * 2048;
    uint16_t *out  = (uint16_t *)(vt + 3 * 2048);

    fill_f16(act, F16_ONE);
    fill_f16(wt,  F16_ONE);
    fill_scales(scl, F16_ONE);

    printf("PROBE_BEGIN\n");

    /* The profiling region: exactly the 4 HMX operations that make up
     * a single 32x32x32 f16 matmul tile. */
    asm volatile("bias = mxmem2(%0)" :: "r"(scl) : "memory");
    asm volatile("mxclracc.hf" ::: "memory");
    asm volatile("{ activation.hf = mxmem(%0,%1)\n"
                 "  weight.hf     = mxmem(%2,%3) }"
                 :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
    asm volatile("mxmem(%0,%1):after.hf = acc" :: "r"(out), "r"(0) : "memory");

    printf("PROBE_END out[0]=0x%04x\n", out[0]);

    h2_thread_stop(0);
    return 0;
}
