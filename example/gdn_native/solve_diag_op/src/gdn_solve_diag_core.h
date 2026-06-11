/* gdn_solve_diag_core.h - shapes/scales for the HVX-diag op GdnSolveDiag (M6 split, Op1).
 *
 * Op1 solves the NB diagonal 64x64 blocks T_ii=inv(I-A_ii) (HVX int16 forward-subst, the proven
 * GdnSolveOp path), narrows each to int8 in 32x32 ROW-MAJOR tiles, and ALSO quantizes the off-diag
 * A_ij blocks to int8 tiles -- all written into a VTCM scratch tensor for the HMX-merge op (Op2).
 * multithreaded=true so the central tiler self-slices the diagonal work across HVX threads.
 *
 * Tile layout for the handoff = "32x32 row-major tiles": a 64x64 block = 2x2 grid of 32x32 tiles,
 * each tile 1024 bytes contiguous in tile-row-major order (tile(tr,tc) at (tr*2+tc)*1024, element
 * (r,c) within tile at r*32+c).  Key milestone fact: a 32x32 crouton8 tile == plain row-major.
 */
#ifndef GDN_SOLVE_DIAG_CORE_H
#define GDN_SOLVE_DIAG_CORE_H
#include <cstdint>
#include <cmath>

#ifndef GDN_BR_C
#define GDN_BR_C   256
#endif
#define GDN_BR_BL  64
#define GDN_BR_NB  (GDN_BR_C / GDN_BR_BL)
#define GDN_BR_F   15
#define GDN_BR_TI  (2.0f/32767.0f)

#define GDN_BR_NBLK ((GDN_BR_NB * (GDN_BR_NB + 1)) / 2)

static inline float gdn_br_qsym_scale(const float *x, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; ++i) { float a = x[i] < 0 ? -x[i] : x[i]; if (a > m) m = a; }
    return (m > 0.0f) ? (m / 127.0f) : 1e-12f;
}

#endif /* GDN_SOLVE_DIAG_CORE_H */
