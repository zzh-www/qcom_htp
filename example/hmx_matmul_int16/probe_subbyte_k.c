/*
 * probe_subbyte_k.c — verify the sub-byte throughput hypothesis functionally.
 *
 * Hypothesis (Agent/hmx_u8xi8_matmul_layers.md §0.5.8): one weight byte
 * decomposes into N sub-byte values, each contributing an independent
 * MAC per HMX packet.
 *
 * If the hypothesis holds, with A=1 everywhere and W "all 1s in sub-byte
 * sense" (byte value that makes every sub-byte value = 1), a single MAC
 * packet should produce output = (N * K_base) where N is the sub-byte
 * count and K_base is how many K-rows the tile encodes at the base
 * layout.
 *
 *   weight.b  (int8):  byte = 0x01, one u8*i8 MAC per byte -> K_total = 32
 *   weight.n  (int4):  byte = 0x11 (two nibbles = 1 each), expect K_total = 64
 *   weight.c  (int2):  byte = 0x55 (four crumbs = 1 each),  expect K_total = 128
 *   weight.ubit(int1): byte = 0xFF (eight bits = 1 each),   expect K_total = 256
 *
 * Compared against:
 *   weight.b baseline: 32
 *
 * If the output values match those predictions, the "1 byte -> N sub-byte
 * MACs" hypothesis is functionally confirmed.
 *
 * Note: readback via :after.uh = acc:2x1 truncates to 16 bit; 256 fits
 * fine. We fill BIAS with identity (0x4000 = f16 2.0 -> scale 1.0) so
 * output = acc mod 2^16 with no extra scaling.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>
#include <hexagon_types.h>

static void fill_act_tile_ones(uint8_t *tile)
{
    /* Fill activation tile so every logical A[ir][K] = 1.
     * Layout: A_byte(phys_row, K, stream) at 128*phys_row + 4*K + (stream?3:1).
     * Ignored-byte positions (4K+0, 4K+2) must stay zero. */
    memset(tile, 0, 2048);
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        for (int K = 0; K < 32; K++) {
            tile[128 * phys_row + 4 * K + 1] = 1;  /* stream 0 */
            tile[128 * phys_row + 4 * K + 3] = 1;  /* stream 1 */
        }
    }
}

static void fill_wt_all(uint8_t *tile, int nbytes, uint8_t byte_val)
{
    /* For sub-byte weight types, we splat the same byte value across the
     * full tile region. Tile byte layout for sub-byte types is not yet
     * decoded, so we fill uniformly — this works as long as EVERY byte
     * in the range contributes the same sub-byte value pattern. */
    memset(tile, byte_val, nbytes);
}

static void fill_bias(uint16_t *b, uint16_t v)
{
    for (int i = 0; i < 128; i++) b[i] = v;
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
    uint8_t  *act  = vt + 0 * 4096;
    uint8_t  *wt   = vt + 1 * 4096;  /* up to 2 KiB for various types */
    uint16_t *bias = (uint16_t *)(vt + 2 * 4096);
    uint16_t *out  = (uint16_t *)(vt + 3 * 4096);

    fill_act_tile_ones(act);
    fill_bias(bias, 0x4000);   /* identity scale 1.0 */

    printf("=== HMX sub-byte K-span probe ===\n");
    printf("A = 1 everywhere (32 rows x 32 K via 16 phys_row x 2 stream)\n");
    printf("W = all 1s in sub-byte interpretation, entire weight region\n\n");
    printf("Reading out[phys_row=0, col=0, stream=0] — expect it to equal\n");
    printf("the number of K-step MACs actually performed per HMX packet.\n\n");

    /* Readback layout: out[phys_row*64 + 2*col + stream]. We check [0][0][0]. */
    #define READ_00() (out[0])

    /* --- weight.b  -- byte = 0x01, 1 KiB tile (8 lines * 128 B) -------------- */
    fill_wt_all(wt, 1024, 0x01);
    memset(out, 0, 2048);
    asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
    asm volatile("mxclracc" ::: "memory");
    asm volatile("{ activation.ub = mxmem(%0,%1)\n"
                 "  weight.b      = mxmem(%2,%3) }"
                 :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
    asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
                 :: "r"(out), "r"(0) : "memory");
    printf("  weight.b   (byte=0x01, tile=1KiB): out[0]=%u  (expect 32)\n",
           READ_00());

    /* --- weight.n  -- byte = 0x11 (two nibbles, each = 1) ------------------- */
    fill_wt_all(wt, 1024, 0x11);
    memset(out, 0, 2048);
    asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
    asm volatile("mxclracc" ::: "memory");
    asm volatile("{ activation.ub = mxmem(%0,%1)\n"
                 "  weight.n      = mxmem(%2,%3) }"
                 :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
    asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
                 :: "r"(out), "r"(0) : "memory");
    printf("  weight.n   (byte=0x11, tile=1KiB): out[0]=%u  (expect 64 if 2 nibbles/byte)\n",
           READ_00());

    /* --- weight.c  -- byte = 0x55 (four crumbs, each = 0b01 = 1) ------------ */
    fill_wt_all(wt, 1024, 0x55);
    memset(out, 0, 2048);
    asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
    asm volatile("mxclracc" ::: "memory");
    asm volatile("{ activation.ub = mxmem(%0,%1)\n"
                 "  weight.c      = mxmem(%2,%3) }"
                 :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
    asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
                 :: "r"(out), "r"(0) : "memory");
    printf("  weight.c   (byte=0x55, tile=1KiB): out[0]=%u  (expect 128 if 4 crumbs/byte)\n",
           READ_00());

    /* --- weight.ubit -- byte = 0xFF (eight 1-bit, all 1) -------------------- */
    fill_wt_all(wt, 1024, 0xFF);
    memset(out, 0, 2048);
    asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
    asm volatile("mxclracc" ::: "memory");
    asm volatile("{ activation.ub = mxmem(%0,%1)\n"
                 "  weight.ubit   = mxmem(%2,%3) }"
                 :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
    asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
                 :: "r"(out), "r"(0) : "memory");
    printf("  weight.ubit(byte=0xFF, tile=1KiB): out[0]=%u  (expect 256 if 8 bits/byte)\n",
           READ_00());

    /* --- weight.n:2x -- byte = 0x11, expect 2x wider K coverage ------------- */
    fill_wt_all(wt, 2048, 0x11);
    memset(out, 0, 2048);
    asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
    asm volatile("mxclracc" ::: "memory");
    asm volatile("{ activation.ub = mxmem(%0,%1)\n"
                 "  weight.n      = mxmem(%2,%3):2x }"
                 :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
    asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
                 :: "r"(out), "r"(0) : "memory");
    printf("  weight.n:2x(byte=0x11, tile=2KiB): out[0]=%u  (expect 128 if :2x doubles K)\n",
           READ_00());

    printf("\n=== Interpretation ===\n");
    printf("Observed value = number of independent sub-byte MACs per cell\n");
    printf("per single HMX packet, summed over the input tile.\n");
    printf("If the ratios match {32, 64, 128, 256, 128}, the sub-byte\n");
    printf("decomposition hypothesis is functionally confirmed.\n");

    h2_thread_stop(0);
    return 0;
}
