/*
 * int16_matmul_hmx.c — bit-exact int16 matmul on HMX via dual-scale readback.
 *
 * 32x32x32 per-tensor symmetric quantized int16 matmul. One full-K HMX u8.i8
 * MAC per partial product, followed by TWO converts on the same accumulator:
 *   1) `:after:retain.uh=acc:2x1` with bias scale 1.0   -> OUT_LO = acc mod 2^16
 *   2) `:after.uh=acc:2x1`        with bias scale 2^-8  -> OUT_HI = (acc>>8) mod 2^16
 * Reconstruct int32: `acc = ((int16_t)OUT_HI << 8) | (OUT_LO & 0xFF)`.
 *
 * Per 32x32x32 tile: 4 MAC packets + 8 convert packets = 12 HMX packets.
 * (Down from 128 in the K-sliced approach.)
 *
 * Decomposition (a_u = a_q + 32768):
 *   a_q.w_q  =  65536.A_h.W_h + 256.(A_h.W_l + A_l.W_h) + A_l.W_l - 32768.w_q
 *
 * M1 = sum_k A_h.W_h   u8.i8 native
 * M2 = sum_k A_h.W_l   u8.u8 via u8.(i8 reinterp) + 256.sum A_h.topbit(W_l)
 * M3 = sum_k A_l.W_h   u8.i8 native
 * M4 = sum_k A_l.W_l   u8.u8 via u8.(i8 reinterp) + 256.sum A_l.topbit(W_l)
 *
 * Tile layouts (decoded in hexagon_hmx_matmul_native_int.md):
 *   Act byte A(phys_row,K,stream) at 128.phys_row + 4.K + (stream?3:1), 2 KiB.
 *   Wt  byte W(K,col)             at 128.(K>>2) + 4.col + (K&3),        1 KiB.
 *   Out u16 (phys_row,col,stream) at phys_row.64 + 2.col + stream,      1024 u16.
 *   Logical ir -> phys_row = ir & 15, stream = ir >> 4.
 *   Bias: 128 u16 (one per output column), f16 scale, applied as scale/2
 *         inside HMX. Slots beyond the 128-u16 per-column are ignored.
 */

#include "int16_matmul.h"
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <string.h>

/* ============== HMX wrappers ============== */
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

/* First of two converts: store convert result but retain the accumulator
 * for a second convert at a different bias. */
static inline __attribute__((always_inline))
void hmx_store_acc_uh_2x1_retain(void *out)
{
    asm volatile("mxmem(%0, %1):after:retain.uh = acc:2x1\n"
                 :: "r"(out), "r"(0) : "memory");
}

/* Second of two converts: standard store (acc consumed after). */
static inline __attribute__((always_inline))
void hmx_store_acc_uh_2x1(void *out)
{
    asm volatile("mxmem(%0, %1):after.uh = acc:2x1\n"
                 :: "r"(out), "r"(0) : "memory");
}

/* Fill 128 u16s with f16 scale `v`. */
static void fill_bias_scale(uint16_t *buf, uint16_t v)
{
    for (int col = 0; col < 128; col++) buf[col] = v;
}

/* ============== Full-K tile packers ==============
 * Pack a 32-row x 32-K uint8 activation (row-major) into the HMX 2 KiB tile.
 * Pack a 32-K  x 32-col int8  weight     (row-major) into the HMX 1 KiB tile. */
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

/* ============== One HMX full-K partial product via dual-scale readback ==
 *   dst[i][j] = Σ_K A[i][K] * W[K][j]    (int32)
 * Single MAC packet + two converts on the same acc.
 *
 * bias_lo (0x4000, f16 2.0 -> scale 1.0) -> lo = acc mod 2^16
 * bias_hi (0x2000, f16 2^-7 -> scale 2^-8) -> hi = (acc>>8) mod 2^16
 * reconstruct: acc = ((int16_t)hi << 8) | (lo & 0xFF)
 */
static void hmx_partial_dual_scale(
    int32_t       *dst32x32,
    const uint8_t *a_32x32,
    const int8_t  *w_32x32,
    uint8_t       *act_tile,
    int8_t        *wt_tile,
    uint16_t      *bias_lo,
    uint16_t      *bias_hi,
    uint16_t      *out_lo,
    uint16_t      *out_hi)
{
    pack_activation_full(act_tile, a_32x32);
    pack_weight_full(wt_tile, w_32x32);

    hmx_load_bias_i(bias_lo);
    hmx_clracc_i();
    hmx_load_pair_u8_i8(act_tile, wt_tile);
    hmx_store_acc_uh_2x1_retain(out_lo);   /* OUT_LO: acc mod 2^16 */

    hmx_load_bias_i(bias_hi);
    hmx_store_acc_uh_2x1(out_hi);          /* OUT_HI: (acc>>8) mod 2^16 */

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

/* ============== Requant ============== */
static inline int16_t sat_i16_shift(int64_t prod, int shift)
{
    int64_t half = (shift == 0) ? 0 : ((int64_t)1 << (shift - 1));
    int64_t v    = (prod + half) >> shift;
    if (v >  32767) v =  32767;
    if (v < -32768) v = -32768;
    return (int16_t)v;
}

/* ============== Public kernel ============== */

void im_matmul_hmx_i8(
    int16_t       *restrict out,
    const int16_t *restrict a,
    const int16_t *restrict w,
    im_requant_t            rq,
    void          *restrict vtcm_base)
{
    uint8_t  *vt       = (uint8_t *)vtcm_base;
    uint8_t  *act_tile =            vt + 0 * 2048;
    int8_t   *wt_tile  = (int8_t *)(vt + 1 * 2048);
    uint16_t *bias_lo  = (uint16_t *)(vt + 2 * 2048);
    uint16_t *bias_hi  = (uint16_t *)(vt + 2 * 2048 + 256); /* 128 u16 after bias_lo */
    uint16_t *out_lo   = (uint16_t *)(vt + 3 * 2048);
    uint16_t *out_hi   = (uint16_t *)(vt + 5 * 2048);

    /* f16 biases: 0x4000 = f16(2.0)  -> scale 1.0    (low-byte read)
     *             0x2000 = f16(2^-7) -> scale 2^-8   (high-byte read, shifted left 8) */
    fill_bias_scale(bias_lo, 0x4000);
    fill_bias_scale(bias_hi, 0x2000);

    /* ---- Decompose int16 operands into top/bot bytes ---- */
    uint8_t A_h[32 * 32], A_l[32 * 32];
    int8_t  W_h[32 * 32];
    uint8_t W_l[32 * 32];
    int32_t col_sum_w[32];
    for (int j = 0; j < 32; j++) col_sum_w[j] = 0;

    for (int i = 0; i < 32 * 32; i++) {
        uint32_t au = (uint32_t)((int32_t)a[i] + 32768);
        A_h[i] = (uint8_t)(au >> 8);
        A_l[i] = (uint8_t)(au & 0xFF);
    }
    for (int k = 0; k < 32; k++) {
        for (int j = 0; j < 32; j++) {
            int16_t wv      = w[k * 32 + j];
            W_h[k * 32 + j] = (int8_t)(wv >> 8);
            W_l[k * 32 + j] = (uint8_t)(wv & 0xFF);
            col_sum_w[j]   += (int32_t)wv;
        }
    }

    /* ---- 4 partial products via single full-K MAC + dual-scale readback ---- */
    int32_t M1[32 * 32], M2_hmx[32 * 32], M3[32 * 32], M4_hmx[32 * 32];
    hmx_partial_dual_scale(M1,     A_h, W_h,           act_tile, wt_tile,
                           bias_lo, bias_hi, out_lo, out_hi);
    hmx_partial_dual_scale(M3,     A_l, W_h,           act_tile, wt_tile,
                           bias_lo, bias_hi, out_lo, out_hi);
    hmx_partial_dual_scale(M2_hmx, A_h, (int8_t *)W_l, act_tile, wt_tile,
                           bias_lo, bias_hi, out_lo, out_hi);
    hmx_partial_dual_scale(M4_hmx, A_l, (int8_t *)W_l, act_tile, wt_tile,
                           bias_lo, bias_hi, out_lo, out_hi);

    /* ---- Bit-correction (u8.u8 was done as u8.(i8 reinterp u8)) ---- */
    int32_t M2_corr[32 * 32], M4_corr[32 * 32];
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            int32_t c2 = 0, c4 = 0;
            for (int k = 0; k < 32; k++) {
                uint8_t tb = (uint8_t)(W_l[k * 32 + j] >> 7);
                c2 += (int32_t)A_h[i * 32 + k] * tb;
                c4 += (int32_t)A_l[i * 32 + k] * tb;
            }
            M2_corr[i * 32 + j] = c2;
            M4_corr[i * 32 + j] = c4;
        }
    }

    /* ---- Combine + requant in int64 ---- */
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            int64_t M2_true = (int64_t)M2_hmx[i * 32 + j] + 256LL * M2_corr[i * 32 + j];
            int64_t M4_true = (int64_t)M4_hmx[i * 32 + j] + 256LL * M4_corr[i * 32 + j];

            int64_t sum =
                  ((int64_t)M1[i * 32 + j] << 16)
                + (M2_true << 8)
                + ((int64_t)M3[i * 32 + j] << 8)
                +  M4_true
                - ((int64_t)col_sum_w[j] << 15);

            int64_t scaled = sum * (int64_t)rq.mul;
            out[i * 32 + j] = sat_i16_shift(scaled, rq.shift);
        }
    }
}

/* f16 API alias (kept for ABI compat). */
void im_matmul_hmx_f16(
    int16_t       *restrict out,
    const int16_t *restrict a,
    const int16_t *restrict w,
    im_requant_t            rq,
    int                     input_shift,
    void          *restrict vtcm_base)
{
    (void)input_shift;
    im_matmul_hmx_i8(out, a, w, rq, vtcm_base);
}
