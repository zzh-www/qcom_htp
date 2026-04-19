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
 * FUSED variant: consumes raw QNN-provided uint16 activation directly
 * (values are post-Cast = signed_int16 + 32768). Splits hi/lo bytes and
 * packs into HMX tile format in one pass, with zero DDR intermediate.
 *
 * @param au         [M_full × K_full] uint16 activation in VTCM (post-Cast).
 * @param M_full     Total M extent of source tensor (stride 0 base).
 * @param K          Reduction dim (multiple of 32).
 * @param m0         Starting M row for the 32-row strip to pack.
 * @param vtcm_base  Pointer to HMX_INT4_VTCM_BYTES_FOR_K(K) bytes of VTCM.
 */
void hmx_int4_prepack_activation_fused(
    const uint16_t *__restrict__ au,
    int                          M_full,
    int                          K,
    int                          m0,
    void           *__restrict__ vtcm_base);

/* Legacy entry for correctness reference (scalar decomp via sg_A_h scratch). */
void hmx_int4_prepack_activation(
    const int16_t *__restrict__ a,
    int                         K,
    void          *__restrict__ vtcm_base);

void hmx_int4_matmul_mn_using_prepacked_act(
    int32_t       *__restrict__ out,
    const int8_t  *__restrict__ w,
    int                         K,
    void          *__restrict__ vtcm_base);

/* Dual-accumulator variant: alternates MAC target between HMX acc A and acc B
 * via mxswapacc — breaks the per-packet data-dependency chain observed in
 * built-in ConvLayer_s1.opt, which uses this exact pattern to hit
 * ~16 cyc/packet vs our ~17K. Sums both accs at readback. */
void hmx_int4_matmul_mn_dualacc(
    int32_t       *__restrict__ out,
    const int8_t  *__restrict__ w,
    int                         K,
    void          *__restrict__ vtcm_base);

/* Fully pre-packed variant: both activation AND weight are pre-packed in
 * VTCM before the K loop. Inner loop is PURE HMX issue (no scalar) so
 * HMX packets pipeline back-to-back. Earlier attempt (iter-4) regressed;
 * this version places weight tiles in a VTCM region chosen to avoid
 * bank conflict with activation tiles. */
void hmx_int4_prepack_weight_tiles(
    const int8_t  *__restrict__ w,
    int                         K,
    void          *__restrict__ vtcm_base);

void hmx_int4_matmul_mn_all_prepacked(
    int32_t       *__restrict__ out,
    int                         K,
    void          *__restrict__ vtcm_base);

/*
 * Fused variant: reads weight directly from QNN-provided uint8 tensor
 * (post-Cast = signed_int8 + 128), eliminating the gather_w_col DDR
 * round-trip. Computes the -128 offset shift inline during pack/col_sum.
 */
void hmx_int4_matmul_mn_fused_weight(
    int32_t        *__restrict__ out,
    const uint8_t  *__restrict__ wu,
    int                          K,
    int                          N_full,
    int                          n0,
    void           *__restrict__ vtcm_base);

#ifdef __cplusplus
}
#endif
#endif
