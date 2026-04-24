/*
 * probe_subbyte_full.c — dump full output and multiple byte_val values
 * to understand the real semantics of sub-byte weight types.
 *
 * After probe_subbyte_k surprisingly showed out[0]=32 for ALL sub-byte
 * types (not the expected 2x/4x/8x), we need more data to figure out
 * what sub-byte types actually change:
 *   (A) do they halve tile footprint while keeping K=32?
 *   (B) do they keep tile size but use only part of it?
 *   (C) something else entirely?
 *
 * Test strategy: for each type, probe multiple byte_val values and dump
 * several output cells. Patterns will reveal the semantics.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>
#include <hexagon_types.h>

static void fill_act_ones(uint8_t *tile)
{
    memset(tile, 0, 2048);
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        for (int K = 0; K < 32; K++) {
            tile[128 * phys_row + 4 * K + 1] = 1;
            tile[128 * phys_row + 4 * K + 3] = 1;
        }
    }
}

static void fill_bias(uint16_t *b, uint16_t v)
{
    for (int i = 0; i < 128; i++) b[i] = v;
}

/* Sample a handful of output cells so we can see patterns. */
static void dump_out(uint16_t *out)
{
    /* out[phys_row*64 + 2*col + stream].  We sample:
     *   ir=0  (phys_row=0, stream=0), cols 0, 1, 2, 15
     *   ir=16 (phys_row=0, stream=1), cols 0, 15
     *   ir=1  (phys_row=1, stream=0), col 0
     */
    printf("    [0,0]=%u  [0,1]=%u  [0,2]=%u  [0,15]=%u  [16,0]=%u  [16,15]=%u  [1,0]=%u\n",
           out[0*64 + 2*0 + 0], out[0*64 + 2*1 + 0], out[0*64 + 2*2 + 0],
           out[0*64 + 2*15 + 0], out[0*64 + 2*0 + 1], out[0*64 + 2*15 + 1],
           out[1*64 + 2*0 + 0]);
}

#define RUN(SUFFIX, MOD, BYTE, TILE_BYTES)                            \
    do {                                                              \
        memset(wt, (BYTE), (TILE_BYTES));                             \
        memset(out, 0, 2048);                                         \
        asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");     \
        asm volatile("mxclracc" ::: "memory");                        \
        asm volatile("{ activation.ub = mxmem(%0,%1)\n"               \
                     "  weight." SUFFIX "  = mxmem(%2,%3)" MOD " }"   \
                     :: "r"(act), "r"(2047),                          \
                        "r"(wt),  "r"(2047) : "memory");              \
        asm volatile("mxmem(%0,%1):after.uh = acc:2x1"                \
                     :: "r"(out), "r"(0) : "memory");                 \
        printf("  weight." SUFFIX MOD " byte=0x%02x tile=%dB\n", (BYTE), (TILE_BYTES)); \
        dump_out(out);                                                \
    } while (0)

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
    uint8_t  *act  = vt + 0 * 4096;
    uint8_t  *wt   = vt + 1 * 4096;
    uint16_t *bias = (uint16_t *)(vt + 2 * 4096);
    uint16_t *out  = (uint16_t *)(vt + 3 * 4096);

    fill_act_ones(act);
    fill_bias(bias, 0x4000);

    printf("=== Sub-byte weight probe (A=1 everywhere) ===\n");
    printf("Output layout: [logical_row, col]=value\n\n");

    printf(" -- weight.b (int8) --\n");
    RUN("b", "", 0x01, 1024);
    RUN("b", "", 0x02, 1024);
    RUN("b", "", 0x7F, 1024);
    RUN("b", "", 0xFF, 1024);  /* -1 signed */

    printf("\n -- weight.n (int4) --\n");
    RUN("n", "", 0x00, 1024);
    RUN("n", "", 0x01, 1024);   /* low nibble=1, high=0 */
    RUN("n", "", 0x10, 1024);   /* low=0, high=1 */
    RUN("n", "", 0x11, 1024);   /* both = 1 */
    RUN("n", "", 0x07, 1024);   /* low=7, high=0 */
    RUN("n", "", 0x70, 1024);   /* low=0, high=7 */
    RUN("n", "", 0x77, 1024);   /* both = 7 */
    RUN("n", "", 0x88, 1024);   /* both = -8 signed */

    printf("\n -- weight.n w/ half tile (512 B) --\n");
    RUN("n", "", 0x11, 512);

    printf("\n -- weight.c (int2 crumb) --\n");
    RUN("c", "", 0x00, 1024);
    RUN("c", "", 0x01, 1024);   /* crumb[0]=1, others=0 */
    RUN("c", "", 0x55, 1024);   /* all four crumbs = 1 */
    RUN("c", "", 0xFF, 1024);   /* all four = 3 (-1 signed) */

    printf("\n -- weight.n:2x --\n");
    RUN("n", ":2x", 0x00, 2048);
    RUN("n", ":2x", 0x11, 2048);
    RUN("n", ":2x", 0x11, 1024);  /* same byte but only 1 KiB tile */

    printf("\n -- weight.ubit --\n");
    RUN("ubit", "", 0x00, 1024);
    RUN("ubit", "", 0x01, 1024);  /* bit 0 = 1, others = 0 */
    RUN("ubit", "", 0x80, 1024);  /* bit 7 = 1, others = 0 */
    RUN("ubit", "", 0xFF, 1024);  /* all 8 bits = 1 */

    h2_thread_stop(0);
    return 0;
}
