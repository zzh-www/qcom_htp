/*
 * hmx_int4_matmul.h — HMX int4×int16 matmul, per-(m_tile, n_tile) K-accumulated.
 *
 * Int4 weight (sign-extended to int8, range [-7,7] for now — direct weight.n
 * nibble path is the next optimization step). Int16 activation, decomposed
 * on-the-fly into (hi u8, lo u8) streams.
 *
 * Per (m_tile=32, n_tile=32) output: 2 HMX accumulation passes (one per
 * activation byte stream), each issuing K/32 MAC-load packets into a hot
 * accumulator, with a single dual-scale readback per pass. This amortizes
 * all per-tile scalar overhead (decomposition, packing, unpacking, combine)
 * across the full K dimension.
 *
 * Saturation: with int4 weights in [-7,7], per-packet max magnitude
 * = 32·255·7 = 57,120. For K=8192, accumulated max = 14.6M, within the
 * 24-bit signed readback range (±8.4M). Actual int4 range gives comfortable
 * headroom even at K=8192 for typical non-adversarial inputs.
 */
#ifndef HMX_INT4_MATMUL_H
#define HMX_INT4_MATMUL_H

#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* VTCM scratch size per (m_tile, n_tile) iteration. Comprises:
 *   - fixed 12 KiB for bias + out_lo/out_hi + transient tiles
 *   - K/32 × 2 × 2048 for pre-packed activation tiles (hi + lo streams)
 *   - K/32 × 1024 for pre-packed weight tiles
 *   - 128 bytes for col_sum_w (32 int32)
 */
#define HMX_INT4_FIXED_VTCM        (12 * 1024)
#define HMX_INT4_ACT_PREPACK_BYTES_FOR_K(K)  ((K) * 2 * 128)  /* K/32 * 4096 */
#define HMX_INT4_WT_PREPACK_BYTES_FOR_K(K)   ((K) * 32)       /* K/32 * 1024 */
#define HMX_INT4_COLSUM_BYTES      (128)
#define HMX_INT4_VTCM_BYTES_FOR_K(K) (HMX_INT4_FIXED_VTCM                  \
    + HMX_INT4_ACT_PREPACK_BYTES_FOR_K(K)                                  \
    + HMX_INT4_WT_PREPACK_BYTES_FOR_K(K)                                   \
    + HMX_INT4_COLSUM_BYTES)

/*
 * Pre-pack the full 32×K activation strip into VTCM-resident packed tile
 * arrays (hi + lo), amortizing decomposition and pack cost across all
 * n_tiles that will consume this activation.
 *
 * @param a          [32*K] int16 row-major activation.
 * @param K          Reduction dim (multiple of 32).
 * @param vtcm_base  Pointer to HMX_INT4_VTCM_BYTES_FOR_K(K) bytes of VTCM.
 */
void hmx_int4_prepack_activation(
    const int16_t *__restrict__ a,
    int                         K,
    void          *__restrict__ vtcm_base);

/*
 * Compute out[M=32, N=32] = sum_k a[M=32, K] * w[K, N=32] using the
 * previously pre-packed activation in @p vtcm_base. Weight is packed
 * per K-iter inside this call — the interleaved scalar pack between
 * HMX MACs hides VTCM read latency (all-prepacked measured slower
 * due to back-to-back VTCM bank contention on v75).
 *
 * @param out         [32*32] int32 output tile (written).
 * @param w           [K*32]  int8 weight (sign-extended from int4).
 * @param K           Reduction dim, must match the preceding act prepack.
 * @param vtcm_base   Same VTCM pointer passed to hmx_int4_prepack_activation.
 */
void hmx_int4_matmul_mn_using_prepacked_act(
    int32_t       *__restrict__ out,
    const int8_t  *__restrict__ w,
    int                         K,
    void          *__restrict__ vtcm_base);

#ifdef __cplusplus
}
#endif
#endif
