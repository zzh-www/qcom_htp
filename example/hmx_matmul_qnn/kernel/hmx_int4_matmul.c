/*
 * hmx_int4_matmul.c — int4×int16 matmul on HMX via dual-scale readback.
 *
 * Derived from example/hmx_matmul_int16/int16_matmul_hmx.c. Weight path
 * collapses from 4 partials to 2 because sign-extended int4 fits entirely
 * in the low byte of an int16; i.e. W_hi = 0 and the A·W_hi products vanish.
 *
 * Decomposition (a_u = a + 32768, w in [-8,7] ⊂ int8):
 *   a · w = (a_u - 32768) · w
 *         = 256·(A_hi · w) + (A_lo · w) - 32768·w
 *
 * Per 32×32×32 tile:
 *   P1 = sum_k A_hi[i,k] · w[k,j]   (u8 · i8, native HMX MAC)
 *   P2 = sum_k A_lo[i,k] · w[k,j]   (u8 · i8, native HMX MAC)
 *   out[i,j] = (P1 << 8) + P2 - 32768 · col_sum_w[j]
 *
 * Each partial uses the dual-scale readback (same as int16 kernel):
 *   bias_lo (0x4000, f16 2.0 -> scale 1.0)   -> lo = acc mod 2^16
 *   bias_hi (0x2000, f16 2^-7 -> scale 2^-8) -> hi = (acc>>8) mod 2^16
 *   acc_int32 = ((int16_t)hi << 8) | (lo & 0xFF)
 */

#include "hmx_int4_matmul.h"
#include <string.h>

/* ============== HMX wrappers (inline asm; same pattern as int16 kernel) ==== */
static inline __attribute__((always_inline)) void hmx_clracc_i(void)
{ asm volatile("mxclracc" ::: "memory"); }

static inline __attribute__((always_inline))
void hmx_load_bias_i(const void *p)
{ asm volatile("bias = mxmem(%0)" :: "r"(p) : "memory"); }

static inline __attribute__((always_inline))
void hmx_load_pair_u8_i8(const void *act, const void *wt)
{
    asm volatile(
        "{ activation.ub = mxmem(%0, %1)\n"
        "  weight.b      = mxmem(%2, %3) }\n"
        :: "r"(act), "r"(2047),
           "r"(wt),  "r"(2047)
        : "memory");
}

static inline __attribute__((always_inline))
void hmx_store_acc_uh_2x1_retain(void *out)
{
    asm volatile("mxmem(%0, %1):after:retain.uh = acc:2x1\n"
                 :: "r"(out), "r"(0) : "memory");
}

static inline __attribute__((always_inline))
void hmx_store_acc_uh_2x1(void *out)
{
    asm volatile("mxmem(%0, %1):after.uh = acc:2x1\n"
                 :: "r"(out), "r"(0) : "memory");
}

static void fill_bias_scale(uint16_t *buf, uint16_t v)
{
    for (int col = 0; col < 128; col++) buf[col] = v;
}

/* ============== Tile packers — same HMX layout as the int16 kernel ======== */
static void pack_activation_full(uint8_t *tile, const uint8_t *a_32x32)
{
    memset(tile, 0, 2048);
    for (int ir = 0; ir < 32; ir++) {
        int phys_row = ir & 15;
        int stream   = ir >> 4;
        int byte_off = (stream == 0) ? 1 : 3;
        for (int K = 0; K < 32; K++) {
            tile[128 * phys_row + 4 * K + byte_off] = a_32x32[ir * 32 + K];
        }
    }
}

static void pack_weight_full(int8_t *tile, const int8_t *w_32x32)
{
    memset(tile, 0, 1024);
    for (int K = 0; K < 32; K++) {
        for (int col = 0; col < 32; col++) {
            tile[128 * (K >> 2) + 4 * col + (K & 3)] = w_32x32[K * 32 + col];
        }
    }
}

/* ============== One u8·i8 partial + dual-scale readback =================== *
 * Activation is packed into act_tile here (varies per partial).  Weight must
 * be pre-packed by the caller since it's shared across both partials within
 * a 32×32×32 tile. */
static void hmx_partial_dual_scale(
    int32_t       *dst32x32,
    const uint8_t *a_32x32,
    uint8_t       *act_tile,
    const int8_t  *wt_tile,
    const uint16_t *bias_lo,
    const uint16_t *bias_hi,
    uint16_t      *out_lo,
    uint16_t      *out_hi)
{
    pack_activation_full(act_tile, a_32x32);

    hmx_load_bias_i(bias_lo);
    hmx_clracc_i();
    hmx_load_pair_u8_i8(act_tile, wt_tile);
    hmx_store_acc_uh_2x1_retain(out_lo);

    hmx_load_bias_i(bias_hi);
    hmx_store_acc_uh_2x1(out_hi);

    for (int ir = 0; ir < 32; ir++) {
        int phys_row = ir & 15;
        int stream   = ir >> 4;
        for (int jc = 0; jc < 32; jc++) {
            uint16_t lo = out_lo[phys_row * 64 + 2 * jc + stream];
            uint16_t hi = out_hi[phys_row * 64 + 2 * jc + stream];
            dst32x32[ir * 32 + jc] =
                ((int32_t)(int16_t)hi << 8) | ((int32_t)lo & 0xFF);
        }
    }
}

/* ============== Public kernel ============================================= */
void hmx_int4_matmul_tile(
    int32_t       *__restrict__ out,
    const int16_t *__restrict__ a,
    const int8_t  *__restrict__ w,
    void          *__restrict__ vtcm_base)
{
    /* VTCM layout (12 KiB total):
     *   [0, 2 KiB)     act_tile   (2 KiB)
     *   [2, 3 KiB)     wt_tile    (1 KiB, actually 1024 B)
     *   [3, 3.25 KiB)  bias_lo    (128 u16)
     *   [3.25, 3.5)    bias_hi    (128 u16)
     *   [4, 6 KiB)     out_lo     (1024 u16)
     *   [6, 8 KiB)     out_hi     (1024 u16)
     *   [8, 12 KiB)    reserved
     */
    uint8_t  *vt       = (uint8_t *)vtcm_base;
    uint8_t  *act_tile =             vt + 0 * 1024;       /* 2 KiB */
    int8_t   *wt_tile  = (int8_t *)( vt + 2 * 1024);      /* 1 KiB */
    uint16_t *bias_lo  = (uint16_t *)(vt + 3 * 1024);     /* 256 B */
    uint16_t *bias_hi  = (uint16_t *)(vt + 3 * 1024 + 256);
    uint16_t *out_lo   = (uint16_t *)(vt + 4 * 1024);     /* 2 KiB */
    uint16_t *out_hi   = (uint16_t *)(vt + 6 * 1024);     /* 2 KiB */

    fill_bias_scale(bias_lo, 0x4000);
    fill_bias_scale(bias_hi, 0x2000);

    /* Move big intermediates off the stack — QHPI kernel threads on HTP
     * have a small default stack and the Op.cpp caller also holds ~7 KiB
     * of tile buffers. Statics are safe because the kernel is
     * multithreaded=false (single slice, sequential invocations). */
    static uint8_t  A_h[32 * 32];
    static uint8_t  A_l[32 * 32];
    static int32_t  P_hi[32 * 32];
    static int32_t  P_lo[32 * 32];
    static int32_t  col_sum_w[32];

    for (int i = 0; i < 32 * 32; i++) {
        uint32_t au = (uint32_t)((int32_t)a[i] + 32768);
        A_h[i] = (uint8_t)(au >> 8);
        A_l[i] = (uint8_t)(au & 0xFF);
    }
    for (int j = 0; j < 32; j++) col_sum_w[j] = 0;
    for (int k = 0; k < 32; k++)
        for (int j = 0; j < 32; j++)
            col_sum_w[j] += (int32_t)w[k * 32 + j];

    /* Pack weight ONCE — same data feeds both A_hi·W and A_lo·W partials. */
    pack_weight_full(wt_tile, w);

    hmx_partial_dual_scale(P_hi, A_h, act_tile, wt_tile,
                           bias_lo, bias_hi, out_lo, out_hi);
    hmx_partial_dual_scale(P_lo, A_l, act_tile, wt_tile,
                           bias_lo, bias_hi, out_lo, out_hi);

    /* Combine: (A_u · W) - 32768·W  =  256·P_hi + P_lo - 32768·col_sum_w. */
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            out[i * 32 + j] = (P_hi[i * 32 + j] << 8)
                            +  P_lo[i * 32 + j]
                            - (col_sum_w[j] << 15);
        }
    }
}
