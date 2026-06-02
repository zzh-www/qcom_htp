/* gdn_solve_br_core.h - shapes, scales, and the proven HMX 64^3 merge packers/descriptors
 * for the block-recursive GDN triangular inverse op GdnSolveBR.
 *
 * C=128, BL=64, 2x2 blocks.  L=I-A=[[L11,0],[L21,L22]]:
 *   T11 = L11^-1, T22 = L22^-1          -- HVX int16 forward-subst (GdnSolveOp's C=64 path)
 *   M   = A21 @ T11                      -- HMX 64^3 u8i8 merge 1
 *   T21 = T22 @ M                        -- HMX 64^3 u8i8 merge 2
 *   T = [[T11,0],[T21,T22]]
 *
 * The packers + descriptor here are the C transcription of scripts/gdn_hmx_matmul_sim.py
 * (M1-proven bit-exact in hexagon-sim) for the fixed 64x64x64 shape, plus the signed-merge
 * drain/control-word scheme from scripts/gdn_blockrec_sim.py (M2b-proven).
 */
#ifndef GDN_SOLVE_BR_CORE_H
#define GDN_SOLVE_BR_CORE_H
#include <cstdint>
#include <cmath>

#define GDN_BR_C   128   /* chunk size handled by this op */
#define GDN_BR_BL  64    /* recursion block size (one HMX 64^3 merge per off-diag block) */
#define GDN_BR_F   15    /* diagonal fold scale (matches GdnSolveOp GDN_F) */
#define GDN_BR_TI  (2.0f/32767.0f)   /* fixed internal int16 T scale for diagonals (|T|<2) */

/* fixed 64^3 descriptor (M1 authoritative, gdn_hmx_matmul_sim.descriptor_tables(64,64,64)):
 *   m_tiles=k_tiles=n_tiles=2, table_m_groups=1
 *   out_desc = {out_table_stride_dwords:2, out_y_stride_words:8, n_tiles_pow2:8,
 *               m_total_minus_step:8, k_total_bytes:64}
 *   act_desc = {n_act_pairs:2, act_table_y_stride_words:8}
 *   activation_offsets = [0, 64*32] = [0, 2048]   (each k-tile a contiguous m*32 tile)
 *   output_offsets     = [0, 64*32] = [0, 2048]   (two N32 tiles)
 *   extra = {1,0}; mask = conv1x1_words(0x700,0,0,0,0x20)
 */
#define GDN_BR_OUT_TABLE_STRIDE 2u
#define GDN_BR_OUT_Y_STRIDE     8u
#define GDN_BR_N_TILES_POW2     8u
#define GDN_BR_M_TOTAL_MINUS_STEP 8
#define GDN_BR_K_TOTAL_BYTES    64u
#define GDN_BR_N_ACT_PAIRS      2u
#define GDN_BR_ACT_Y_STRIDE     8u

/* shape-independent conv1x1 mask words for the u8i8 deep path (verified equal to
 * conv1x1_words(0x700,0,0,0,0x20) and to set_hmx_params_conv1x1(0x700,0,0,0,0x20)). */
static const uint32_t GDN_BR_MASK_WORDS[16] = {
    0x0u, 0x700u, 0x0u, 0x71fu, 0x0u, 0x0u, 0x7ffu, 0x0u,
    0x0u, 0x0u, 0x0u, 0x0u, 0x20u, 0x0u, 0x0u, 0x0u
};

/* ---- host-side / scalar helpers (also compiled for x86 fallback) ---- */

/* symmetric int8 quant: code in [-127,127], scale = max|x|/127. */
static inline float gdn_br_qsym_scale(const float *x, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; ++i) { float a = x[i] < 0 ? -x[i] : x[i]; if (a > m) m = a; }
    return (m > 0.0f) ? (m / 127.0f) : 1e-12f;
}

#endif /* GDN_SOLVE_BR_CORE_H */
