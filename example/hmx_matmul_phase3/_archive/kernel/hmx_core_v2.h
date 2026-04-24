/*
 * hmx_core_v2.h — Phase 3B kernel using `:cm` + row-major activation.
 *
 * REPLACES hmx_core.h (which assumed 2-stream packed tiles). Agent A's
 * silicon probe (Agent/cm_row_major_re.md, 2026-04-23) confirmed:
 *
 *   activation.ub = mxmem(p, Rt_a | 0x1c):cm
 *   weight.b      = mxmem(q, 0x3FF)
 *
 * natively consumes a 1 KiB (32×32 contiguous bytes) row-major
 * activation tile at 7.92 cyc/MAC — matching QNN q::ConvLayer_s1.opt
 * and BEATING our Phase 2 2-stream approach (9.03 cyc/MAC) by 1.1.
 *
 * Implications:
 *  - No HVX pre-pack needed for activation (just gather to VTCM 1 KiB)
 *  - Weight tile stays 1 KiB (same byte layout as Phase 2 pack_weight,
 *    OR simply row-major K×N — HMX weight.b tile format matches row-
 *    major for K=32 tile granularity)
 *  - Kernel body: pure HMX mxmem + mxmac in hot loop. Setup loop for
 *    gather is HVX-able in Phase 3D refinement.
 *
 * Tile layout (row-major, aligned to 2 KiB boundary in VTCM):
 *   act_tile: 1 KiB = 32 rows × 32 K-bytes contiguous
 *   wt_tile:  1 KiB = K_row × N_col row-major (same as HMX native)
 * Dual-scale readback: 2 KiB each (out_lo, out_hi).
 */
#ifndef HMX_CORE_V2_H
#define HMX_CORE_V2_H

#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Bias scratch (512 B, fill once). */
void hmx_core_v2_fill_bias(void *bias_vtcm);

/* Compute one (32×32) output tile via HMX MAC with `:cm` + row-major.
 *
 * @param act_tiles  VTCM pointer, [K_tiles][1024] row-major act tiles
 *                   for current m_tile (gathered beforehand)
 * @param wt_tiles   VTCM pointer, [K_tiles][1024] row-major wt tiles
 *                   for current n_tile
 * @param K_tiles    K / 32  (MAC inner loop count)
 * @param bias_vtcm  pre-filled via hmx_core_v2_fill_bias (512 B)
 * @param out_lo     2 KiB dual-scale readback low
 * @param out_hi     2 KiB dual-scale readback high
 *
 * Output format: HMX dual-scale readback (uint16 × 2). Downstream Combine
 * op produces int32 = ((int16)hi << 8) | (lo & 0xFF) - offset_correction. */
void hmx_matmul_v2_core_mn(
    const uint8_t *act_tiles,
    const uint8_t *wt_tiles,
    uint32_t       K_tiles,
    void          *bias_vtcm,
    uint16_t      *out_top_lo,   /* rows 0..15 dual-scale low */
    uint16_t      *out_top_hi,   /* rows 0..15 dual-scale high */
    uint16_t      *out_bot_lo,   /* rows 16..31 dual-scale low */
    uint16_t      *out_bot_hi);  /* rows 16..31 dual-scale high */

/* HVX-vectorised decode + scatter of one (m_tile, n_tile) 2-stream
 * dual-scale readback into the 32×32 int32 output submatrix.
 *
 * Replaces V3's inner scalar loop:
 *   mn_tile[ir*32 + jc] = ((int16)hi[idx] << 8) | (lo[idx] & 0xFF)
 *   where idx = (ir & 15) * 64 + 2 * jc + (ir >> 4)
 * and the subsequent scalar scatter into out[mt*32 + ir][nt*32 + jc].
 *
 * Acts directly on VTCM readback buffers (lo_rb / hi_rb, 2 KiB each),
 * writes 32 int32 rows × 32 cols to out_tile_base at N-stride.
 *
 * Assumes out_tile_base is 128-B aligned (true when N and nt*32 are
 * multiples of 32, i.e. any standard shape). */
void hmx_matmul_v2_decode_scatter_hvx(
    int32_t       *out_tile_base,
    uint32_t       N,
    const uint16_t *lo_rb,
    const uint16_t *hi_rb);

/* V6 variant: write requanted i8 output to a staging row buffer at a
 * specific 32-B column offset (in a 128-B row), instead of writing
 * directly to DDR. Used for 4-way nt batching where 4 tiles fill one
 * 128-B aligned row before the bulk-flush to DDR. `col_offset` ∈ {0,32,64,96}. */
void hmx_matmul_v2_requant_to_staging_i8_hvx(
    int8_t        *staging_rows,   /* [32][128] */
    int            col_offset,
    const uint16_t *lo_rb,
    const uint16_t *hi_rb,
    const int32_t *multiplier_tile,
    int32_t        shift);

/* V6: quantized variant. Same dual-scale decode as above but followed by
 * TFLite-style per-channel requant:
 *   stage1 = SaturatingRoundingDoublingHighMul(int24_acc, multiplier[n])
 *          = round((int24_acc × multiplier[n]) / 2^31)  with saturation
 *   stage2 = RoundingShiftRight(stage1, shift)
 *   out    = saturate_i8(stage2)
 *
 * `multiplier` is int32 per N column (per-channel weight scale folded
 * with per-tensor activation/output scales at host prep time).
 * `shift` is a single scalar shift shared across all channels.
 *
 * Writes 32 × 32 int8 tile to out_tile_base (row stride N bytes).
 * multiplier_tile points to multiplier[nt*32]. */
void hmx_matmul_v2_requant_scatter_i8_hvx(
    int8_t        *out_tile_base,
    uint32_t       N,
    const uint16_t *lo_rb,
    const uint16_t *hi_rb,
    const int32_t *multiplier_tile,
    int32_t        shift);

/* Helper: gather 32 rows × 32 K-cols from raw row-major activation
 * [M, K] (byte-strided) into a contiguous 1 KiB VTCM tile. Simple byte
 * copy; can be replaced by HVX vmemu version in Phase 3D perf pass. */
void hmx_core_v2_gather_act_tile(
    uint8_t       *tile_vtcm,           /* 1 KiB destination */
    const uint8_t *au,                  /* [M_full][K_full] row-major */
    uint32_t       K_full,
    uint32_t       m0,                  /* starting row */
    uint32_t       k0);                 /* starting K column */

/* Compute one (32×32) output tile via HMX MAC using `:cm` row-major
 * activation (1 KiB tile per K-step) and Phase 2 4-K-row packed weight
 * (1 KiB tile per K-step). Both probed as bit-exact on SM8650 v75
 * (Agent/cm_weight_layout_re.md, 2026-04-23).
 *
 * Act tile layout (1 KiB): tile[32*r + k] = act[r][k] for r,k ∈ [0,32).
 * Wt tile layout: identical to hmx_matmul_v2_core_mn's consumer. */
void hmx_matmul_v2_core_mn_cm(
    const uint8_t *act_tiles,     /* [K_tiles][1024] */
    const uint8_t *wt_tiles,      /* [K_tiles][1024] */
    uint32_t       K_tiles,
    void          *bias_vtcm,
    uint16_t      *out_top_lo,
    uint16_t      *out_top_hi,
    uint16_t      *out_bot_lo,    /* unused in 2-stream dual-scale readback */
    uint16_t      *out_bot_hi);

/* Helper: gather 32 K-rows × 32 N-cols from raw row-major weight [K, N]
 * into HMX weight tile format (row-major K×N tile for HMX weight.b
 * consumer). Actually HMX weight.b wants the 4-K-row × 32-col packed
 * format historically; Agent A's probe did NOT reprobe weight format
 * with `:cm` activation + plain weight. Keep Phase 2's pack_weight for
 * weight until further probing; only activation side gets the :cm win. */
void hmx_core_v2_gather_wt_tile(
    uint8_t       *tile_vtcm,           /* 1 KiB destination */
    const int8_t  *wu,                  /* [K_full][N_full] row-major */
    uint32_t       N_full,
    uint32_t       k0,
    uint32_t       n0);

/* Row-major [M, K] u8 activation → Phase 2 2-stream packed 2 KiB tile.
 *
 * Consumed layout by downstream `hmx_matmul_v2_core_mn` via
 *   activation.ub = mxmem(p, 2047)
 *
 * Used by V6 op to pack activation tiles inline (Phase 3D.0 replaced the
 * separate pack_act graph node). Equivalent to pack_act_u8_hvx's per-tile
 * inner code; duplicated here to avoid cross-kernel linkage. */
void hmx_core_v2_flat_pack_act_tile(
    uint8_t       *packed_tile,         /* 2 KiB destination (VTCM, aligned 128) */
    const uint8_t *act_raw,             /* [M_full][K_full] row-major */
    uint32_t       K_full,              /* row stride in bytes */
    uint32_t       mt,                  /* output M-tile index */
    uint32_t       kt);                 /* output K-tile index */

#ifdef __cplusplus
}
#endif
#endif
