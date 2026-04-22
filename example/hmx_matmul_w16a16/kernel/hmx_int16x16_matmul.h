/*
 * hmx_int16x16_matmul.h — int16 × int16 matmul on HMX, K-accumulated,
 * int32 output (no requant).
 *
 * Built from the 4-term u8·i8 decomposition in int16_matmul_hmx.c but
 * adapted for (a) K-accumulation across multiple K-tiles for a single
 * (m_tile, n_tile) output tile, and (b) int32 output (not int16+requant).
 *
 * Math — with au, wu post-Cast uint16 (= signed + 32768):
 *   a = au - 32768,  w = wu - 32768
 *   au·wu = (au_hi·256 + au_lo) × (wu_hi·256 + wu_lo)
 *         = au_hi·wu_hi·65536 + (au_hi·wu_lo + au_lo·wu_hi)·256 + au_lo·wu_lo
 *   where au_hi/au_lo are u8 halves, wu_hi is treated as signed int8 but
 *   bit-corrected per int16_matmul_hmx.c's scheme, wu_lo similarly.
 *
 * out[m,n] = Σ_k a·w
 *          = Σ_k (au - 32768)(wu - 32768)
 *          = Σ_k au·wu  - 32768·Σ_k au  - 32768·Σ_k wu  + 32768²·K
 *
 * So we compute Σ_k au·wu in HMX (via 4-term decomp) and correct for the
 * offsets with row_sum_au / col_sum_wu / K constant.
 *
 * VTCM budget per (m_tile, n_tile): fixed 16 KiB (HMX tiles + bias + out)
 * plus K-dependent prepack if used.
 */

#ifndef HMX_INT16X16_MATMUL_H_
#define HMX_INT16X16_MATMUL_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HMX_W16A16_FIXED_VTCM     (16 * 1024)
#define HMX_W16A16_VTCM_BYTES_FOR_K(K)   HMX_W16A16_FIXED_VTCM

/* One (32, N=32) output tile for full K via K-accumulated 4-term HMX MACs.
 * `au` is [M_full × K_full] post-Cast u16 activation.
 * `wu` is [K_full × N_full] post-Cast u16 weight (same +32768 offset).
 * `out` is [32×32] int32 row-major. */
void hmx_int16x16_matmul_mn(
    int32_t        *__restrict__ out,
    const uint16_t *__restrict__ au,
    const uint16_t *__restrict__ wu,
    int                          M_full,
    int                          K,
    int                          N_full,
    int                          m0,
    int                          n0,
    void           *__restrict__ vtcm_base);

#ifdef __cplusplus
}
#endif

#endif
