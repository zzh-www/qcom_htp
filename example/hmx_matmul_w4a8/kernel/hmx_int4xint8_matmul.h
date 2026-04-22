/*
 * hmx_int4xint8_matmul.h — int4 weight × int8 activation MatMul on HMX.
 *
 * Math:
 *   Quantization model: symmetric int4 weight in [-7, 7], signed int8
 *   activation. QNN's graph optimizer inserts Cast@FH.Fh that adds offsets
 *   to unsigned types:
 *     int8  activation → quint8  (stored = signed + 128)
 *     int4  weight     → quint8  (stored = signed + 128; int4 range [121, 135])
 *
 *   HMX native: activation.ub × weight.b. Kernel consumes:
 *     au    : uint8  (raw QNN-post-Cast, value = signed_act + 128)
 *     wu    : uint8  (raw QNN-post-Cast, value = signed_w + 128, range [121,135])
 *   and produces int32 output:
 *     out[m,n] = sum_k (au[m,k] - 128) · (wu[k,n] - 128)
 *
 *   Simpler than w4a16 because activation is single-byte (no hi/lo split).
 *   Single HMX u8·i8 MAC pass per 32×32×32 tile.
 *
 * Layout conventions:
 *   - 32×32×32 tile over HMX accumulator.
 *   - VTCM layout shared with hmx_int4_matmul (activation 2 KiB, weight 1 KiB,
 *     bias 512 B, out 4 KiB = 8 KiB fixed; pre-packed activation strips
 *     occupy 128 B × (K/32) = K·4 bytes per strip).
 */

#ifndef HMX_INT4XINT8_MATMUL_H_
#define HMX_INT4XINT8_MATMUL_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef HMX_INT4XINT8_MAX_K
#define HMX_INT4XINT8_MAX_K   8192
#endif

/* VTCM scratch requirement for one (m_tile, n_tile) strip at reduction K.
 * Fixed 8 KiB + activation prepack K·128 B + weight prepack K·32 B.
 * (Weight prepack is optional; current kernel packs per K-iter scalar.) */
#define HMX_W4A8_FIXED_VTCM        (8 * 1024)
#define HMX_W4A8_ACT_PREPACK_BYTES_FOR_K(K)  ((K) * 128)
#define HMX_W4A8_WT_PREPACK_BYTES_FOR_K(K)   ((K) * 32)
#define HMX_W4A8_VTCM_BYTES_FOR_K(K)                                    \
    (HMX_W4A8_FIXED_VTCM + HMX_W4A8_ACT_PREPACK_BYTES_FOR_K(K))

/* Pre-pack the full 32×K activation strip from raw QNN uint8 into HMX
 * tile format in VTCM. `au[M_full × K_full]` with `m0` the starting row
 * of the 32-row strip. */
void hmx_int4xint8_prepack_activation(
    const uint8_t *__restrict__ au,
    int                          M_full,
    int                          K,
    int                          m0,
    void          *__restrict__ vtcm_base);

/* Compute one (32 × N=32) output tile for K-accumulated u8·i8 MAC.
 * `w[K × 32]` is gather_w_col's output: int8 signed (post –128 un-shift).
 * `out[32 × 32]` in int32, row-major. */
void hmx_int4xint8_matmul_mn(
    int32_t       *__restrict__ out,
    const int8_t  *__restrict__ w,
    int                          K,
    void          *__restrict__ vtcm_base);

#ifdef __cplusplus
}
#endif

#endif
