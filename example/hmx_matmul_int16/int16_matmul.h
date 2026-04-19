/*
 * int16_matmul.h — per-tensor symmetric quantized int16 matmul, 32x32x32 tile.
 *
 * Quantization model (symmetric per-tensor, zero-point = 0):
 *   a_real[i][k] = s_a * a_q[i][k]        (a_q : int16)
 *   w_real[k][j] = s_w * w_q[k][j]        (w_q : int16)
 *   out_real     = sum_k a_real * w_real  =  s_a*s_w * int32_sum
 *   out_q        = sat_i16( out_real / s_out )
 *
 * The requant struct folds  (s_a * s_w / s_out)  into integer fixed-point:
 *   out_q ≈ sat_i16( (int32_sum * mul) >> shift )
 *
 * Kernels:
 *   im_matmul_ref         — exact int16 matmul in int64, fixed-point requant.
 *   im_matmul_ref_f16_path— plain-C model of the HMX f16 path; the HMX
 *                           kernel should match this bit-exactly.
 *   im_matmul_hmx_f16     — f16 HMX matmul (ch01-style), int16 I/O via
 *                           HVX dequant / requant.
 *
 * Accuracy note:
 *   f16 has ~10-11 bit mantissa, so the f16-path kernel diverges from the
 *   full-int16 reference by a few LSBs depending on magnitudes.  The HMX
 *   output *does* match im_matmul_ref_f16_path bit-exactly.
 */

#ifndef INT16_MATMUL_H_
#define INT16_MATMUL_H_

#include <stdint.h>

#define IM_TILE_M   32
#define IM_TILE_K   32
#define IM_TILE_N   32

#define IM_TILE_MK  (IM_TILE_M * IM_TILE_K)
#define IM_TILE_KN  (IM_TILE_K * IM_TILE_N)
#define IM_TILE_MN  (IM_TILE_M * IM_TILE_N)

typedef struct {
    int32_t mul;
    int32_t shift;  /* 0..31 */
} im_requant_t;

/* Exact int16 reference.  Oracle for "what the math says." */
void im_matmul_ref(
    int16_t       *restrict out,
    const int16_t *restrict a,
    const int16_t *restrict w,
    im_requant_t            rq);

/* Plain-C model of the HMX u8*i8 top-byte kernel.  Asserted bit-exact
 * vs the HMX output in the test harness. */
void im_matmul_ref_top_only(
    int16_t       *restrict out,
    const int16_t *restrict a,
    const int16_t *restrict w,
    im_requant_t            rq,
    int                     input_shift);  /* ignored; kept for ABI compat */

/* Alias, kept so older harnesses still link. */
void im_matmul_ref_f16_path(
    int16_t       *restrict out,
    const int16_t *restrict a,
    const int16_t *restrict w,
    im_requant_t            rq,
    int                     input_shift);

/* HMX int-path kernel (u8 * i8, 4-term decomposition + corrections).
 * vtcm_base: 2 KiB aligned, needs >= 18 KiB. */
void im_matmul_hmx_i8(
    int16_t       *restrict out,
    const int16_t *restrict a,
    const int16_t *restrict w,
    im_requant_t            rq,
    void          *restrict vtcm_base);

/* f16-path wrapper (kept for API compat with the test harness; internally
 * delegates to im_matmul_hmx_i8 now that the int path is in place). */
void im_matmul_hmx_f16(
    int16_t       *restrict out,
    const int16_t *restrict a,
    const int16_t *restrict w,
    im_requant_t            rq,
    int                     input_shift,
    void          *restrict vtcm_base);

#endif /* INT16_MATMUL_H_ */
