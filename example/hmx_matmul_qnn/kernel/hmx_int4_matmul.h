/*
 * hmx_int4_matmul.h — HMX int4×int16 matmul, 32×32×32 tile.
 *
 * Weight is sign-extended int4 (range [-8, 7]) passed as int8 for direct HMX
 * u8×i8 MAC consumption. Activation is signed int16; decomposed on the fly
 * into (hi u8, lo u8) via the +32768 bias shift, as in int16_matmul_hmx.c.
 *
 * Per 32×32×32 tile: 2 HMX MAC packets + dual-scale readback (2 converts each)
 *                    = 6 HMX-issue events  (vs 12 for int16×int16).
 *
 * Output is int32 accumulator (no on-kernel requantize — caller decides).
 */
#ifndef HMX_INT4_MATMUL_H
#define HMX_INT4_MATMUL_H

#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Required VTCM scratch, in bytes. Caller passes a VTCM pointer of this size. */
#define HMX_INT4_VTCM_BYTES  (12 * 1024)

/*
 * 32x32x32 int4×int16 matmul tile.
 *   out[i][j] = sum_k a[i][k] * w[k][j]    (int32)
 *
 * @param out         [32*32] int32, row-major output (not saturated).
 * @param a           [32*32] int16, row-major activation.
 * @param w           [32*32] int8, row-major weight (values must be in [-8,7]).
 * @param vtcm_base   Pointer to >= HMX_INT4_VTCM_BYTES of VTCM scratch.
 */
void hmx_int4_matmul_tile(
    int32_t       *__restrict__ out,
    const int16_t *__restrict__ a,
    const int8_t  *__restrict__ w,
    void          *__restrict__ vtcm_base);

#ifdef __cplusplus
}
#endif
#endif
