/*
 * hmx_core.h — Phase 3B core HMX MatMul kernel API.
 *
 * Design principle: this kernel ONLY issues HMX mxmem + mxmac + readback.
 * No HVX. No scalar pack. No data movement. Inputs are pre-packed
 * HMX-tile-ready bytes produced by upstream HVX ops (PackActToHmxTile /
 * PackWeightToHmxTile). Output is raw HMX dual-scale readback (two uint16
 * buffers per (m,n) tile); downstream Combine HVX op converts to int32.
 *
 * This matches QNN q::ConvLayer_s1.opt architecture: inner loop is just
 * `{ activation.ub = mxmem(p, Rt):cm; weight.b = mxmem(q, 0x3FF) }`.
 *
 * Tile layout (pre-packed by upstream ops, must be VTCM-resident):
 *   packed_act: [M_tiles][K_tiles][2048]  each 2KB block = one HMX
 *               activation tile in Phase 2's 2-stream format:
 *                 byte offsets per 4B cell: [pad, s0, pad, s1]
 *                 16 phys_rows × 128 bytes = 2048 bytes
 *               For w4a16: caller provides TWO packed_act arrays
 *                 (hi stream, lo stream) and calls hmx_matmul_core_mn()
 *                 twice (once per partial).
 *   packed_wt:  [K_tiles][N_tiles][1024]  each 1KB block = one HMX
 *               weight tile in 4-K-row × 32-col byte packing:
 *                 8 K-groups × 32 cells × 4 bytes
 *                 = 128 bytes per K-group, 8 K-groups total
 *               (Same format pack_weight_32x32 produces in Phase 2.)
 *
 * Readback format:
 *   out_lo, out_hi: each [32 × 32 uint16] = 2 KiB = HMX readback dual-scale
 *     Indexed as out_lo[phys_row * 64 + 2 * jc + stream] for ir=phys_row+16*stream
 *     (exactly as Phase 2 hmx_int4_matmul_mn_dualacc readback).
 *   Downstream Combine op is responsible for int32 reconstruction.
 */
#ifndef HMX_CORE_H
#define HMX_CORE_H

#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* VTCM scratch: bias tables for dual-scale readback.
 *   [0, 256)    bias_lo: 128× uint16 of 0x4000
 *   [256, 512)  bias_hi: 128× uint16 of 0x2000
 * 512 bytes total. Can reuse across many (m,n) tile invocations. */
#define HMX_CORE_BIAS_BYTES 512

/* Fill the bias scratch region ONCE per graph-finalize. */
void hmx_core_fill_bias(void *bias_vtcm);

/* Compute one (32 × 32) output tile via pure HMX MAC loop.
 *
 * @param packed_act    VTCM pointer to [M_tiles][K_tiles][2048] packed activation
 * @param packed_wt     VTCM pointer to [K_tiles][N_tiles][1024] packed weight
 * @param M_tiles       ceil(M / 32)
 * @param K_tiles       ceil(K / 32)  (inner MAC loop trip count)
 * @param N_tiles       ceil(N / 32)
 * @param mt            current m-tile index (0..M_tiles-1)
 * @param nt            current n-tile index (0..N_tiles-1)
 * @param bias_vtcm     VTCM pointer pre-filled via hmx_core_fill_bias (256 B + 256 B)
 * @param out_lo        VTCM pointer, 2 KiB — dual-scale readback low
 * @param out_hi        VTCM pointer, 2 KiB — dual-scale readback high
 */
void hmx_matmul_core_mn(
    const uint8_t *packed_act,
    const uint8_t *packed_wt,
    uint32_t       M_tiles,
    uint32_t       K_tiles,
    uint32_t       N_tiles,
    uint32_t       mt,
    uint32_t       nt,
    void          *bias_vtcm,
    uint16_t      *out_lo,
    uint16_t      *out_hi);

/* Dual-accumulator variant for w4a16 style: consumes TWO packed_act arrays
 * (hi stream + lo stream), runs interleaved hi/lo MAC via mxswapacc,
 * produces TWO sets of dual-scale readbacks.
 *
 * Saves: one pack_weight (shared across hi & lo) and one m-stride amortization.
 * Matches Phase 2 fused dualacc pattern. */
void hmx_matmul_core_mn_dualacc(
    const uint8_t *packed_act_hi,
    const uint8_t *packed_act_lo,
    const uint8_t *packed_wt,
    uint32_t       M_tiles,
    uint32_t       K_tiles,
    uint32_t       N_tiles,
    uint32_t       mt,
    uint32_t       nt,
    void          *bias_vtcm,
    uint16_t      *out_hi_lo_A,   /* 2 KiB: hi-partial dual-scale low  */
    uint16_t      *out_hi_hi_A,   /* 2 KiB: hi-partial dual-scale high */
    uint16_t      *out_lo_lo_B,   /* 2 KiB: lo-partial dual-scale low  */
    uint16_t      *out_lo_hi_B);  /* 2 KiB: lo-partial dual-scale high */

#ifdef __cplusplus
}
#endif
#endif
