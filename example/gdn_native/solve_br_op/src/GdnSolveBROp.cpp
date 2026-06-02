/*
 * GdnSolveBROp.cpp — QNN/QHPI custom op "GdnSolveBR": T = (I - A)^-1 for a strictly-lower A,
 * C=128, via 2-block (BL=64) block recursion.  HVX int16 forward-substitution for the two
 * 64x64 diagonal inverses + TWO HMX u8i8 64^3 merges driven FROM INSIDE THE OP.
 *
 *   input[0]  A : QUInt16  [B,H,128,128]  (uint16-midpoint, scale sA, zp ~32768)
 *   output[0] T : QUInt16  [B,H,128,128]  (uint16-midpoint, scale sT, zp ~32768)
 *
 * Algorithm (host-validated, scripts/gdn_blockrec_sim.py M2b):
 *   T11 = inv(I - A[0:64,0:64]) ; T22 = inv(I - A[64:128,64:128])   (HVX forward-subst, int16)
 *   A21 = A[64:128,0:64]
 *   M   = A21 @ T11      -- merge 1 (HMX 64^3 u8i8, signed operands recentred to u8)
 *   T21 = T22 @ M        -- merge 2 (HMX 64^3 u8i8)
 *   T = [[T11,0],[T21,T22]]
 *
 * The HMX kernel is the owned V73DEEP Conv1x1 replica (same one the production
 * example/qnn_hmx_matmul_u8i8 op drives).  Operands are DYNAMIC (computed in-op), so weight/
 * activation/bias are packed at RUNTIME into static VTCM-aligned scratch and the descriptors
 * are stitched in the hot callback.  Declares QHPI_RESOURCE_HVX|QHPI_RESOURCE_HMX; the backend
 * acquires HMX at graph load, so the callback just runs the inline-asm body.
 *
 * Incremental bring-up debug modes (compile-time):
 *   GDN_BR_SKIP_KERNEL : write a marker to T[0..] and return (proves package loads + runs).
 *   GDN_BR_DIAG_ONLY   : T21=0 (block-diagonal); validates the HVX diagonals + I/O.
 *   GDN_BR_DESC_DUMP   : dump merge-1 descriptors into T[0..] and return.
 *   GDN_BR_DUMP_M      : write the recovered int8 M codes (merge 1 result) into T (inspect merge1).
 *   GDN_BR_PROBE_CYCLES: record per-stage cycles (diag/pack/hmx) into T head 0.
 */
#include "HTP/core/qhpi.h"
#include "gdn_solve_br_core.h"

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

#define GDN_BR_MAX_SLICES 8

static const int C  = GDN_BR_C;   /* 128 */
static const int BL = GDN_BR_BL;  /* 64  */

#if defined(__hexagon__)
#include <hexagon_types.h>
#include <hexagon_protos.h>

#if defined(GDN_BR_THREAD_TEST)
/* Feasibility probe: can a QHPI HMX-resource callback spawn a qurt worker thread?  The worker does
 * a pure-HVX op (no HMX) and writes a sentinel; the callback joins it.  Tests whether manual qurt
 * threading (the M3-pipeline prerequisite) is permitted inside the op, and whether a spawned thread
 * can acquire an HVX context alongside the backend's own HVX threads. */
#include "qurt.h"
static volatile int g_thr_sentinel = 0;
static volatile int g_thr_hvx_ok   = 0;
static char __attribute__((aligned(128))) g_thr_stack[16384];
static void gdn_thr_worker(void *arg) {
    (void)arg;
    int rc = qurt_hvx_lock(QURT_HVX_MODE_128B);   /* try to grab an HVX context from the worker thread */
    g_thr_hvx_ok = (rc == 0) ? 1 : (1000 + rc);
    if (rc == 0) {
        HVX_Vector v = Q6_V_vsplat_R(0x12345678);
        int32_t lanes[32] __attribute__((aligned(128)));
        *(HVX_Vector *)lanes = v;
        g_thr_sentinel = lanes[7];               /* prove HVX actually executed on this thread */
        qurt_hvx_unlock();
    } else {
        g_thr_sentinel = 0x5EED;                 /* spawn worked, HVX lock failed */
    }
}
#endif

/* native HMX descriptor ABI (matches the owned kernel; identical to the production op). */
struct hmx_conv_out_desc_t {
    int32_t *out_tile_ptr_table;
    uint32_t out_table_stride_dwords;
    uint32_t out_y_stride_words;
    uint32_t n_tiles_pow2;
    int32_t  m_total_minus_step;
    uint32_t k_total_bytes;
};
struct hmx_conv_act_desc_t {
    int32_t *act_ptr_pairs;
    uint32_t n_act_pairs;
    uint32_t act_table_y_stride_words;
};
struct hmx_conv_mask_desc_t {
    int32_t  out_check;     uint32_t out_rt_mask;
    int32_t  act_check;     uint32_t act_rt_base;
    uint32_t filter_x_stride; uint32_t _pad14; uint32_t alt_rt;
};
#include "v73deep_conv1x1_kernel.h"
#endif

/* ----------------------------- scratch layout -----------------------------
 * The diagonal-solve and fp-merge scratch can live in plain BSS (HVX/scalar reads DDR fine).
 * The HMX merge SURFACES (packed activation crouton8, k-major weight, folded bias, output crouton8,
 * pointer tables) MUST be in VTCM — HMX mxmem ops fault on DDR addresses.  We carve those out of a
 * scratch tensor declared QHPI_MemLoc_TCM_Only (inputs[1]); qhpi_tensor_raw_data() returns its VTCM
 * address.  (Static BSS is NOT VTCM and faults the kernel — verified on device.) */
#if defined(__hexagon__)
/* DDR-resident working scratch (single thread, multithreaded=false). */
static int32_t __attribute__((aligned(128))) g_Tc [GDN_BR_BL * GDN_BR_BL];
static int32_t __attribute__((aligned(128))) g_Afx[GDN_BR_BL * GDN_BR_BL];
/* int32 T-codes (scale GDN_BR_TI) for the two diagonal blocks + A21 folded codes (scale 2^-F).
 * Kept as INTEGER end-to-end (no float dequant roundtrip) — the float path was the M_op bottleneck. */
static int32_t __attribute__((aligned(128))) g_Tc11[GDN_BR_BL * GDN_BR_BL];
static int32_t __attribute__((aligned(128))) g_Tc22[GDN_BR_BL * GDN_BR_BL];
static int32_t __attribute__((aligned(128))) g_A21c[GDN_BR_BL * GDN_BR_BL];   /* A21 folded codes @2^-F */
static int8_t  __attribute__((aligned(128))) g_Mi8 [GDN_BR_BL * GDN_BR_BL];
static int8_t  __attribute__((aligned(128))) g_T21i[GDN_BR_BL * GDN_BR_BL];
static uint8_t __attribute__((aligned(128))) g_actbuf[GDN_BR_BL * GDN_BR_BL]; /* u8 activation (zp128) */
static int8_t  __attribute__((aligned(128))) g_wtbuf [GDN_BR_BL * GDN_BR_BL]; /* i8 weight (k-major src) */
static int32_t __attribute__((aligned(128))) g_qbuf [GDN_BR_BL * GDN_BR_BL]; /* quant scratch (avoid DSP stack) */

/* VTCM scratch carved from the TCM_Only scratch tensor.  Buffers are spaced 0x10000 (64 KB) apart —
 * matching the proven M1 sim harness layout — so any HMX over-write/alignment slack can't clobber a
 * neighbouring buffer.  Total span < 0x60000 (384 KB; graph declares >= that). */
struct gdn_vtcm_t {
    uint8_t *act;     /* 4096 */
    int8_t  *wt;      /* 4096 */
    int32_t *bias;    /* 128 int32 = 512 B */
    uint8_t *out;     /* 4096 */
    int32_t *acttab;  /* 2 */
    int32_t *outtab;  /* 2 */
};
static gdn_vtcm_t gdn_vtcm_from(uint8_t *base) {
    gdn_vtcm_t v;
    v.act    = base + 0x00000;
    v.wt     = (int8_t *)(base + 0x10000);
    v.bias   = (int32_t *)(base + 0x20000);
    v.out    = base + 0x30000;
    v.acttab = (int32_t *)(base + 0x40000);
    v.outtab = (int32_t *)(base + 0x40080);
    return v;
}

/* ---- diagonal 64x64 forward-subst (ported from GdnSolveOp C=64 path) ---- */
static inline void gdn_fold_MS(float sA, int *pM, int *pS) {
    float sf = sA * (float)(1 << GDN_BR_F);
    int S = 14;
    while (S < 30 && sf * (float)(1 << (S + 1)) < 30000.0f) ++S;
    while (S >  0 && sf * (float)(1 <<  S)      > 32760.0f) --S;
    *pM = (int)(sf * (float)(1 << S) + 0.5f); *pS = S;
}

/* VECTORIZED fold of one 64x64 block A (uint16 codes, row-stride=C) -> int32 codes Afx (scale 2^-F),
 * with the code REPLICATED into both halfwords (so it is a plain scalar Rt for vmpyiacc_VwVwRh).
 * Ported from GdnSolveOp::gdn_fold_A; handles a row stride (A sub-block sits in a C-wide tensor). */
static void gdn_fold_block_hvx(const uint16_t *Au, int row_stride, int32_t *Afx, int zpA, int M, int S) {
    const int Mrep = (M & 0xFFFF) * 0x10001;
    const HVX_Vector vzp = Q6_V_vsplat_R(zpA), vrndS = Q6_V_vsplat_R(1 << (S - 1)), m16 = Q6_V_vsplat_R(0xFFFF);
    for (int r = 0; r < BL; ++r) {
        const HVX_UVector *Av = (const HVX_UVector *)(Au + r * row_stride);   /* 64 uint16 = 1 vec */
        HVX_Vector *Afxv = (HVX_Vector *)(Afx + r * BL);                      /* 64 int32 = 2 vecs */
        HVX_VectorPair w = Q6_Wuw_vzxt_Vuh(Av[0]);
        HVX_Vector c0 = Q6_Vw_vsub_VwVw(Q6_V_lo_W(w), vzp), c1 = Q6_Vw_vsub_VwVw(Q6_V_hi_W(w), vzp);
        HVX_Vector i0 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(c0, Mrep), vrndS), S);
        HVX_Vector i1 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(c1, Mrep), vrndS), S);
        HVX_VectorPair s = Q6_W_vshuff_VVR(i1, i0, -4);
        HVX_Vector tl = Q6_V_vand_VV(Q6_V_lo_W(s), m16), th = Q6_V_vand_VV(Q6_V_hi_W(s), m16);
        Afxv[0] = Q6_V_vor_VV(tl, Q6_Vw_vasl_VwR(tl, 16));
        Afxv[1] = Q6_V_vor_VV(th, Q6_Vw_vasl_VwR(th, 16));
    }
}

/* fold one 64x64 block A -> RAW int32 codes (scale 2^-F, sign-extended, NOT halfword-replicated).
 * Used for A21 which is a QUANT operand (needs the plain code), unlike the diagonal-solve Afx which is
 * a scalar multiplier (needs both halfwords set). */
static void gdn_fold_block_raw(const uint16_t *Au, int row_stride, int32_t *Afx, int zpA, int M, int S) {
    const int Mrep = (M & 0xFFFF) * 0x10001;
    const HVX_Vector vzp = Q6_V_vsplat_R(zpA), vrndS = Q6_V_vsplat_R(1 << (S - 1));
    for (int r = 0; r < BL; ++r) {
        const HVX_UVector *Av = (const HVX_UVector *)(Au + r * row_stride);
        HVX_Vector *Afxv = (HVX_Vector *)(Afx + r * BL);
        HVX_VectorPair w = Q6_Wuw_vzxt_Vuh(Av[0]);
        HVX_Vector c0 = Q6_Vw_vsub_VwVw(Q6_V_lo_W(w), vzp), c1 = Q6_Vw_vsub_VwVw(Q6_V_hi_W(w), vzp);
        HVX_Vector i0 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(c0, Mrep), vrndS), S);
        HVX_Vector i1 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(c1, Mrep), vrndS), S);
        HVX_VectorPair s = Q6_W_vshuff_VVR(i1, i0, -4);   /* even/odd -> natural col order */
        Afxv[0] = Q6_V_lo_W(s);
        Afxv[1] = Q6_V_hi_W(s);
    }
}

/* solve one 64x64 diagonal block: T = inv(I - A_block), int16-code forward subst, leave int32 CODES
 * (scale GDN_BR_TI) in Tcout — NO float dequant (the M_op float roundtrip was the diagonal bottleneck). */
static void gdn_solve_diag64(const uint16_t *Au, int row_stride, int zpA, int M, int S,
                             int32_t *Tcout) {
    int32_t *Tc  = Tcout;
    int32_t *Afx = g_Afx;
    gdn_fold_block_hvx(Au, row_stride, Afx, zpA, M, S);
    const HVX_Vector vrnd = Q6_V_vsplat_R(1 << (GDN_BR_F - 1));
    const int ei = (int)(1.0f / GDN_BR_TI + 0.5f);
    /* T row i (64 cols = 2 HVX vectors), acc int32, requant >>F. T genuinely lower-tri. */
    for (int i = 0; i < BL; ++i) {
        HVX_Vector e0 = Q6_V_vzero(), o0 = Q6_V_vzero(), e1 = Q6_V_vzero(), o1 = Q6_V_vzero();
        int k = 0;
        for (; k + 1 < i; k += 2) {
            int s0 = Afx[i * BL + k], s1 = Afx[i * BL + k + 1];
            const HVX_Vector *T0 = (const HVX_Vector *)(Tc + k * BL);
            const HVX_Vector *T1 = (const HVX_Vector *)(Tc + (k + 1) * BL);
            e0 = Q6_Vw_vmpyiacc_VwVwRh(e0, T0[0], s0);
            o0 = Q6_Vw_vmpyiacc_VwVwRh(o0, T1[0], s1);
            e1 = Q6_Vw_vmpyiacc_VwVwRh(e1, T0[1], s0);
            o1 = Q6_Vw_vmpyiacc_VwVwRh(o1, T1[1], s1);
        }
        for (; k < i; ++k) {
            int s0 = Afx[i * BL + k];
            const HVX_Vector *T0 = (const HVX_Vector *)(Tc + k * BL);
            e0 = Q6_Vw_vmpyiacc_VwVwRh(e0, T0[0], s0);
            e1 = Q6_Vw_vmpyiacc_VwVwRh(e1, T0[1], s0);
        }
        HVX_Vector *Ti = (HVX_Vector *)(Tc + i * BL);
        Ti[0] = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(e0, o0), vrnd), GDN_BR_F);
        Ti[1] = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(e1, o1), vrnd), GDN_BR_F);
        Tc[i * BL + i] += ei;
    }
}

/* ---- HVX integer helpers for scale-estimation + int8 packing (replace the scalar float path) ---- */

/* max |code| over a 64x64 int32 buffer (scale-free; caller scales). 64 int32/row = 2 vecs. */
static int32_t gdn_maxabs_codes(const int32_t *codes) {
    HVX_Vector vmax = Q6_V_vzero();
    const HVX_Vector *p = (const HVX_Vector *)codes;
    for (int b = 0; b < (BL * BL) / 32; ++b)
        vmax = Q6_Vw_vmax_VwVw(vmax, Q6_Vw_vabs_Vw(p[b]));
    vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4*16));
    vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4*8));
    vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4*4));
    vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4*2));
    vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4*1));
    int32_t lanes[32] __attribute__((aligned(128)));
    *(HVX_Vector *)lanes = vmax;
    return lanes[0];
}

/* quantize 64x64 int32 codes (value = code*scale_in) into int8 weight codes at symmetric scale
 * sQ = maxabs_value/127.  Returns sQ.  Vectorized: multiplier = round(scale_in/sQ * 2^Q) fixed-point. */
static float gdn_quant_i8_from_codes(const int32_t *codes, float scale_in, int8_t *out) {
    int32_t mx = gdn_maxabs_codes(codes);
    float maxval = (float)mx * scale_in;
    float sQ = (maxval > 0.0f) ? (maxval / 127.0f) : 1e-12f;
    /* code_q = round(code*g), g=scale_in/sQ.  Adaptive Q keeps Mg<2^15 so code(<2^15)*Mg < 2^30. */
    float g = scale_in / sQ;
    int Q = 14;
    while (Q > 1  && g * (float)(1 << Q) >= 32768.0f) --Q;
    while (Q < 28 && g * (float)(1 << (Q + 1)) < 16384.0f) ++Q;
    int32_t Mg = (int32_t)(g * (float)(1 << Q) + 0.5f);
    const HVX_Vector vrnd = Q6_V_vsplat_R(1 << (Q - 1));
    const HVX_Vector vlim = Q6_V_vsplat_R(127), vnlim = Q6_V_vsplat_R(-127);
    const HVX_Vector *p = (const HVX_Vector *)codes;
    int32_t *qbuf = g_qbuf;
    HVX_Vector *qp = (HVX_Vector *)qbuf;
    const int Mrep = (Mg & 0xFFFF) * 0x10001;   /* Mg fits int16; replicate into both halfwords for vmpyi_VwRh */
    for (int b = 0; b < (BL * BL) / 32; ++b) {
        HVX_Vector prod = Q6_Vw_vmpyi_VwRh(p[b], Mrep);
        HVX_Vector q = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(prod, vrnd), Q);
        q = Q6_Vw_vmin_VwVw(q, vlim); q = Q6_Vw_vmax_VwVw(q, vnlim);
        qp[b] = q;
    }
    for (int i = 0; i < BL * BL; ++i) out[i] = (int8_t)qbuf[i];
    return sQ;
}

/* quantize 64x64 int32 codes into u8 activation (zp128) at symmetric scale sQ.  Returns sQ. */
static float gdn_quant_u8_from_codes(const int32_t *codes, float scale_in, uint8_t *out) {
    int32_t mx = gdn_maxabs_codes(codes);
    float maxval = (float)mx * scale_in;
    float sQ = (maxval > 0.0f) ? (maxval / 127.0f) : 1e-12f;
    float g = scale_in / sQ;
    int Q = 14;
    while (Q > 1  && g * (float)(1 << Q) >= 32768.0f) --Q;
    while (Q < 28 && g * (float)(1 << (Q + 1)) < 16384.0f) ++Q;
    int32_t Mg = (int32_t)(g * (float)(1 << Q) + 0.5f);
    const HVX_Vector vrnd = Q6_V_vsplat_R(1 << (Q - 1));
    const HVX_Vector vlim = Q6_V_vsplat_R(127), vnlim = Q6_V_vsplat_R(-127), v128 = Q6_V_vsplat_R(128);
    const HVX_Vector *p = (const HVX_Vector *)codes;
    int32_t *qbuf = g_qbuf;
    HVX_Vector *qp = (HVX_Vector *)qbuf;
    const int Mrep = (Mg & 0xFFFF) * 0x10001;
    for (int b = 0; b < (BL * BL) / 32; ++b) {
        HVX_Vector prod = Q6_Vw_vmpyi_VwRh(p[b], Mrep);
        HVX_Vector q = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(prod, vrnd), Q);
        q = Q6_Vw_vmin_VwVw(q, vlim); q = Q6_Vw_vmax_VwVw(q, vnlim);
        qp[b] = Q6_Vw_vadd_VwVw(q, v128);
    }
    for (int i = 0; i < BL * BL; ++i) out[i] = (uint8_t)qbuf[i];
    return sQ;
}

/* EXACT scale estimate: max|P_int| where P_int[i,c] = sum_k (act_u8[i,k]-128)*wt_i8[k,c].  This is the
 * exact int product the HMX kernel computes, so the derived scale is exact (not a loose bound) AND HVX
 * fast.  act is u8 (zp128), wt is i8, both 64x64 natural.  Returns max|P_int| (int32). */
static int32_t gdn_pint_maxabs(const uint8_t *act_u8, const int8_t *wt_i8) {
    /* pre-widen wt rows to int32 [64][64] in g_Tc (reused scratch). */
    int32_t *W = g_Tc;
    for (int k = 0; k < BL; ++k)
        for (int c = 0; c < BL; ++c) W[k * BL + c] = (int32_t)wt_i8[k * BL + c];
    HVX_Vector vmax = Q6_V_vzero();
    const HVX_Vector v128 = Q6_V_vsplat_R(128);
    for (int i = 0; i < BL; ++i) {
        HVX_Vector a0 = Q6_V_vzero(), a1 = Q6_V_vzero();
        for (int k = 0; k < BL; ++k) {
            int s = (int)act_u8[i * BL + k] - 128;     /* signed scalar in [-128,127] */
            int srep = (s & 0xFFFF) * 0x10001;         /* replicate into BOTH halfwords (h0=h1) */
            const HVX_Vector *Wk = (const HVX_Vector *)(W + k * BL);
            a0 = Q6_Vw_vmpyiacc_VwVwRh(a0, Wk[0], srep);
            a1 = Q6_Vw_vmpyiacc_VwVwRh(a1, Wk[1], srep);
        }
        vmax = Q6_Vw_vmax_VwVw(vmax, Q6_Vw_vabs_Vw(a0));
        vmax = Q6_Vw_vmax_VwVw(vmax, Q6_Vw_vabs_Vw(a1));
    }
    vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4*16));
    vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4*8));
    vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4*4));
    vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4*2));
    vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4*1));
    int32_t lanes[32] __attribute__((aligned(128)));
    *(HVX_Vector *)lanes = vmax;
    return lanes[0];
}

/* requant a 64x64 int32-code block (value=code*scale_in) into the uint16 output Th at (roff,coff),
 * scale sT, zp zpT.  Vectorized over the 64 cols (2 vecs/row); fixed-point multiply g=scale_in/sT.
 * The strict-upper-of-diagonal lanes are written too (they're ~0 from the solve), harmless. */
static void gdn_requant_block_out(const int32_t *codes, float scale_in, float sT, int zpT,
                                  uint16_t *Th, int roff, int coff, int row_stride) {
    /* adaptive Q: keep multiplier Mg in [2^13,2^15) so code(<2^15)*Mg stays < 2^30 (no int32 overflow). */
    float g = scale_in / sT;
    int Q = 14;
    while (Q > 1  && g * (float)(1 << Q) >= 32768.0f) --Q;
    while (Q < 28 && g * (float)(1 << (Q + 1)) < 16384.0f) ++Q;
    int32_t Mg = (int32_t)(g * (float)(1 << Q) + 0.5f);
    const int Mrep = (Mg & 0xFFFF) * 0x10001;
    const HVX_Vector vrnd = Q6_V_vsplat_R(1 << (Q - 1));
    const HVX_Vector vzpT = Q6_V_vsplat_R(zpT);
    const HVX_Vector vlim = Q6_V_vsplat_R(32767), vnlim = Q6_V_vsplat_R(-32767);
    for (int r = 0; r < BL; ++r) {
        const HVX_Vector *cp = (const HVX_Vector *)(codes + r * BL);
        int32_t q[64] __attribute__((aligned(128)));
        HVX_Vector *qp = (HVX_Vector *)q;
        for (int b = 0; b < 2; ++b) {
            HVX_Vector prod = Q6_Vw_vmpyi_VwRh(cp[b], Mrep);
            HVX_Vector qq = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(prod, vrnd), Q);
            qq = Q6_Vw_vmin_VwVw(qq, vlim); qq = Q6_Vw_vmax_VwVw(qq, vnlim);
            qp[b] = Q6_Vw_vadd_VwVw(qq, vzpT);
        }
        uint16_t *dst = Th + (roff + r) * row_stride + coff;
        *(HVX_UVector *)dst = Q6_Vuh_vpack_VwVw_sat(qp[1], qp[0]);
    }
}

/* ---- crouton8 activation packer (C transcription of gdn_hmx_matmul_sim.pack_act_crouton8, 64^3) ---- */
static void gdn_pack_act_crouton8(const uint8_t *act_mk, uint8_t *out_buf) {
    /* For fixed (kt,row8,m32,rsub) the 8 col_words x 4 bytes = 32 OUTPUT bytes are source row `row`,
     * cols [k_base, k_base+32) in natural order (col_word*4 + b).  So each 32-byte output run is a
     * contiguous 32-col slice of one source row -> a plain 32-byte copy (no per-elem scatter). */
    int out = 0;
    for (int kt = 0; kt < 2; ++kt) {
        int k_base = kt * 32;
        for (int row8_group = 0; row8_group < 4; ++row8_group)
            for (int m32_group = 0; m32_group < 2; ++m32_group)
                for (int row_sub = 0; row_sub < 8; ++row_sub) {
                    int row = m32_group * 32 + row8_group * 8 + row_sub;
                    const uint8_t *src = act_mk + row * 64 + k_base;
                    uint8_t *dst = out_buf + out;
                    *(uint64_t *)(dst + 0)  = *(const uint64_t *)(src + 0);
                    *(uint64_t *)(dst + 8)  = *(const uint64_t *)(src + 8);
                    *(uint64_t *)(dst + 16) = *(const uint64_t *)(src + 16);
                    *(uint64_t *)(dst + 24) = *(const uint64_t *)(src + 24);
                    out += 32;
                }
    }
}

/* ---- k-major weight packer (C transcription of prepare_owned_inputs.pack_w8_kmajor, 64^3) ----
 * For a 32x32 tile: output word at (r4*128 + c*4) = bytes [row(4*r4+0)[c]..row(4*r4+3)[c]], i.e. the
 * 4 k-rows {4r4..4r4+3} byte-interleaved per column c.  We build it as: for each r4 (8 groups), gather
 * the 4 source rows (32 bytes each) into a 128-byte HVX vector via two vshuffs (4-way byte interleave)
 * then store the 128-byte result.  ~10x fewer ops than the per-byte scatter. */
static void gdn_pack_w8_kmajor(const int8_t *w_kn, int8_t *packed) {
    int out = 0;
    for (int kt = 0; kt < 2; ++kt) {
        int k_base = kt * 32;
        for (int nt = 0; nt < 2; ++nt) {
            int n_base = nt * 32;
            int8_t *tile = packed + out;   /* 1024 bytes = 8 groups of 128 */
            for (int r4 = 0; r4 < 8; ++r4) {
                /* load 4 rows (each 32 bytes) into the low 32 bytes of 4 vectors. */
                const int8_t *s0 = w_kn + (k_base + 4*r4 + 0) * 64 + n_base;
                const int8_t *s1 = w_kn + (k_base + 4*r4 + 1) * 64 + n_base;
                const int8_t *s2 = w_kn + (k_base + 4*r4 + 2) * 64 + n_base;
                const int8_t *s3 = w_kn + (k_base + 4*r4 + 3) * 64 + n_base;
                /* interleave bytes: want out[c*4 + b] = row_b[c].  Build 32 words, word c = (s0c,s1c,s2c,s3c). */
                int8_t *d = tile + r4 * 128;
                for (int c = 0; c < 32; ++c) { d[c*4+0]=s0[c]; d[c*4+1]=s1[c]; d[c*4+2]=s2[c]; d[c*4+3]=s3[c]; }
            }
            out += 1024;
        }
    }
}

/* effective[n] = -128*sum_k wt[k,n]; 64-wide.  HVX column-sum over int8 rows using a byte dot-product
 * with a +1 constant vector: Q6_Vw_vrmpyacc_VwVbVb_b? Use vmpa? Simplest reliable HVX: sum int8 rows by
 * widening with vsxt + the proper de-interleave is fiddly; the scalar form is correct and bounded (4096
 * ops/merge, ~small vs pint).  Kept scalar for correctness; not the bottleneck. */
static void gdn_effective(const int8_t *w_kn, int32_t *effective) {
    for (int n = 0; n < 64; ++n) {
        int s = 0;
        for (int k = 0; k < 64; ++k) s += w_kn[k * 64 + n];
        effective[n] = -128 * s;
    }
}

/* f16 bits of a float (round-to-nearest-even via the hardware fp16 convert is not available on x86;
 * do an explicit IEEE half conversion good enough for the gains we use). */
static uint16_t gdn_f16_bits(float v) {
    union { float f; uint32_t u; } in; in.f = v;
    uint32_t x = in.u;
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        int shift = 14 - exp;
        uint32_t half = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1u);
        if (rem > (1u << (shift - 1)) || (rem == (1u << (shift - 1)) && (half & 1))) half++;
        return (uint16_t)(sign | half);
    } else if (exp >= 0x1F) {
        return (uint16_t)(sign | 0x7C00u);
    }
    uint16_t h = (uint16_t)(sign | (exp << 10) | (mant >> 13));
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) h++;
    return h;
}

/* per-N32 control word = (baseline_u16<<16) | f16_bits(scale_f16); pack folded bias for 64^3 (2 tiles). */
static void gdn_pack_bias(const int32_t *effective, float scale_f16, int baseline_u16, int32_t *bias) {
    uint32_t ctrl = ((uint32_t)(baseline_u16 & 0xFFFF) << 16) | (uint32_t)gdn_f16_bits(scale_f16);
    int out = 0;
    for (int start = 0; start < 64; start += 32) {
        for (int i = 0; i < 32; ++i) bias[out++] = (int32_t)ctrl;
        for (int i = 0; i < 32; ++i) bias[out++] = effective[start + i];
    }
}

/* depack the 64x64 u8 crouton8 output surface (closed form, M1-validated). */
static inline uint8_t gdn_depack_out(const uint8_t *surf, int r, int c) {
    int nt = c / 32, m32 = r / 32, r8 = (r % 32) / 8, rsub = r % 8, cw = (c % 32) / 4, bsub = c % 4;
    return surf[nt * 2048 + r8 * 512 + m32 * 256 + rsub * 32 + cw * 4 + bsub];
}

/* FAST depack: for fixed (nt,r8,m32,rsub) the 32 consecutive surface bytes at
 * base = nt*2048 + r8*512 + m32*256 + rsub*32 are exactly output row (m32*32+r8*8+rsub),
 * cols [nt*32, nt*32+32) in natural order (cw*4+bsub = 0..31).  So depack = 128 copies of a
 * contiguous 32-byte run.  We subtract `base` (output zp) via a 32-byte vector op and write the
 * signed int8 code directly into out_codes[row*64 + nt*32].  ~8x faster than the per-elem closed form. */
static uint8_t __attribute__((aligned(128))) g_surf_sub[GDN_BR_BL * GDN_BR_BL];  /* base-subtracted surface */
static void gdn_depack_out_fast(const uint8_t *surf, int base, int8_t *out_codes) {
    /* 1) subtract base over the whole 4096-byte surface with HVX (32 vecs). surf is VTCM-aligned. */
    const HVX_Vector vb = Q6_Vb_vsplat_R(base);
    const HVX_Vector *sp = (const HVX_Vector *)surf;
    HVX_Vector *dp = (HVX_Vector *)g_surf_sub;
    for (int i = 0; i < (BL * BL) / 128; ++i) dp[i] = Q6_Vb_vsub_VbVb(sp[i], vb);
    /* 2) rearrange the subtracted surface into natural [64][64] via contiguous 32-byte runs. */
    int off = 0;
    for (int nt = 0; nt < 2; ++nt)
        for (int r8 = 0; r8 < 4; ++r8)
            for (int m32 = 0; m32 < 2; ++m32)
                for (int rsub = 0; rsub < 8; ++rsub) {
                    int row = m32 * 32 + r8 * 8 + rsub;
                    const int8_t *src = (const int8_t *)(g_surf_sub + off);
                    int8_t *dst = out_codes + row * 64 + nt * 32;
                    *(uint64_t *)(dst + 0)  = *(const uint64_t *)(src + 0);
                    *(uint64_t *)(dst + 8)  = *(const uint64_t *)(src + 8);
                    *(uint64_t *)(dst + 16) = *(const uint64_t *)(src + 16);
                    *(uint64_t *)(dst + 24) = *(const uint64_t *)(src + 24);
                    off += 32;
                }
}

/* run one signed 64^3 HMX merge: act_u8 (zp128 crouton8 not yet packed), wt_i8 (k-major not yet packed),
 * given as natural [64,64].  Packs into VTCM scratch, runs the kernel, writes recovered int8 codes
 * (out_u8 - 128) into out_codes.  scale_f16/baseline recentre the signed product to u8 zp128. */
#if defined(GDN_BR_PROBE_CYCLES)
uint64_t g_c_hmxpack = 0, g_c_hmxkern = 0, g_c_hmxdepack = 0, g_c_quant = 0, g_c_pint = 0;
#endif
static void gdn_hmx_merge(const gdn_vtcm_t *vt, const uint8_t *act_u8, const int8_t *wt_i8,
                          float scale_f16, int baseline_u16, int8_t *out_codes) {
#if defined(GDN_BR_PROBE_CYCLES)
    uint64_t p0; asm volatile("%0 = C15:14" : "=r"(p0));
#endif
    static int32_t __attribute__((aligned(128))) eff[64];
    gdn_effective(wt_i8, eff);
    gdn_pack_act_crouton8(act_u8, vt->act);
    gdn_pack_w8_kmajor(wt_i8, vt->wt);
    gdn_pack_bias(eff, scale_f16, baseline_u16, vt->bias);
    { HVX_Vector z = Q6_V_vzero(); HVX_Vector *op = (HVX_Vector *)vt->out;
      for (int i = 0; i < (BL * BL) / 128; ++i) op[i] = z; }   /* VTCM out surface = 4096 B = 32 vecs */
#if defined(GDN_BR_PROBE_CYCLES)
    uint64_t p1; asm volatile("%0 = C15:14" : "=r"(p1)); g_c_hmxpack += p1 - p0;
#endif

    vt->acttab[0] = (int32_t)(uintptr_t)(vt->act + 0);
    vt->acttab[1] = (int32_t)(uintptr_t)(vt->act + 64 * 32);
    vt->outtab[0] = (int32_t)(uintptr_t)(vt->out + 0);
    vt->outtab[1] = (int32_t)(uintptr_t)(vt->out + 64 * 32);

    uint32_t extra_param[2] __attribute__((aligned(16))) = {1u, 0u};
    uint32_t mask_buf[16] __attribute__((aligned(16)));
    for (int i = 0; i < 16; ++i) mask_buf[i] = GDN_BR_MASK_WORDS[i];

    hmx_conv_out_desc_t out_desc __attribute__((aligned(64))) = {
        vt->outtab, GDN_BR_OUT_TABLE_STRIDE, GDN_BR_OUT_Y_STRIDE,
        GDN_BR_N_TILES_POW2, GDN_BR_M_TOTAL_MINUS_STEP, GDN_BR_K_TOTAL_BYTES };
    hmx_conv_act_desc_t act_desc __attribute__((aligned(64))) = {
        vt->acttab, GDN_BR_N_ACT_PAIRS, GDN_BR_ACT_Y_STRIDE };

    our_v73deep_kernel(&out_desc, &act_desc, (const uint8_t *)vt->wt, (const uint8_t *)vt->bias,
                       (const hmx_conv_mask_desc_t *)mask_buf, extra_param);
#if defined(GDN_BR_PROBE_CYCLES)
    uint64_t p2; asm volatile("%0 = C15:14" : "=r"(p2)); g_c_hmxkern += p2 - p1;
#endif

    int base = baseline_u16 >> 7;   /* output zero-point in u8 */
    gdn_depack_out_fast(vt->out, base, out_codes);
#if defined(GDN_BR_PROBE_CYCLES)
    uint64_t p3; asm volatile("%0 = C15:14" : "=r"(p3)); g_c_hmxdepack += p3 - p2;
#endif
}
#endif  /* __hexagon__ */

/* ============================ host (x86) reference fallback ============================ */
/* Mirrors the device math in plain C double so the op is correct off-device too (and so the
 * standalone harness can compare). Uses GdnSolveOp's solve core for the diagonals. */
#include "../../solve_op/src/gdn_solve_core.h"
static void gdn_br_head_scalar(const uint16_t *Au, int zpA, float sA,
                               float sT, int zpT, uint16_t *Tu) {
    static int16_t As[GDN_BR_C * GDN_BR_C];
    /* diagonal solves via gdn_solve_head_q on the two 64-blocks. */
    float Tf11[BL * BL], Tf22[BL * BL], A21[BL * BL];
    int16_t sub16[BL * BL], Tcode[BL * BL];
    /* T11 */
    for (int r = 0; r < BL; ++r) for (int c = 0; c < BL; ++c)
        sub16[r * BL + c] = (int16_t)((int)Au[r * C + c] - zpA);
    /* re-quant the sub-block at its own sA? gdn_solve_head_q takes A codes at scale sA (whole tensor). */
    gdn_solve_head_q<int16_t>(sub16, BL, sA, GDN_BR_TI, 32767.0f, Tcode);
    for (int i = 0; i < BL * BL; ++i) Tf11[i] = (float)Tcode[i] * GDN_BR_TI;
    /* T22 */
    for (int r = 0; r < BL; ++r) for (int c = 0; c < BL; ++c)
        sub16[r * BL + c] = (int16_t)((int)Au[(BL + r) * C + (BL + c)] - zpA);
    gdn_solve_head_q<int16_t>(sub16, BL, sA, GDN_BR_TI, 32767.0f, Tcode);
    for (int i = 0; i < BL * BL; ++i) Tf22[i] = (float)Tcode[i] * GDN_BR_TI;
    /* A21 */
    for (int r = 0; r < BL; ++r) for (int c = 0; c < BL; ++c)
        A21[r * BL + c] = ((int)Au[(BL + r) * C + c] - zpA) * sA;
    /* merge1 M = A21 @ T11 (symmetric int8 both) */
    float sa = gdn_br_qsym_scale(A21, BL * BL), sb = gdn_br_qsym_scale(Tf11, BL * BL);
    static int M_i8[BL * BL];
    float M_dq[BL * BL];
    for (int r = 0; r < BL; ++r) for (int c = 0; c < BL; ++c) {
        long acc = 0;
        for (int k = 0; k < BL; ++k) {
            long aq = lroundf(A21[r * BL + k] / sa); if (aq > 127) aq = 127; if (aq < -127) aq = -127;
            long bq = lroundf(Tf11[k * BL + c] / sb); if (bq > 127) bq = 127; if (bq < -127) bq = -127;
            acc += aq * bq;
        }
        M_i8[r * BL + c] = (int)acc; M_dq[r * BL + c] = (float)acc * (sa * sb);
    }
    float sM = gdn_br_qsym_scale(M_dq, BL * BL);
    /* requant M into int8 codes at sM */
    int8_t Mq[BL * BL];
    for (int i = 0; i < BL * BL; ++i) { long q = lroundf(M_dq[i] / sM); if (q > 127) q = 127; if (q < -127) q = -127; Mq[i] = (int8_t)q; }
    /* merge2 T21 = T22 @ M (act=T22, wt=Mq at scale sM) */
    float sc = gdn_br_qsym_scale(Tf22, BL * BL);
    float T21[BL * BL];
    for (int r = 0; r < BL; ++r) for (int c = 0; c < BL; ++c) {
        long acc = 0;
        for (int k = 0; k < BL; ++k) {
            long aq = lroundf(Tf22[r * BL + k] / sc); if (aq > 127) aq = 127; if (aq < -127) aq = -127;
            acc += aq * Mq[k * BL + c];
        }
        T21[r * BL + c] = (float)acc * (sc * sM);
    }
    /* assemble T at output scale */
    for (int i = 0; i < C * C; ++i) Tu[i] = (uint16_t)zpT;
    auto put = [&](int r, int c, float v) {
        long q = lroundf(v / sT); long lim = 32767;
        if (q > lim) q = lim; if (q < -lim) q = -lim;
        Tu[r * C + c] = (uint16_t)((int)q + zpT);
    };
    for (int r = 0; r < BL; ++r) for (int c = 0; c < BL; ++c) put(r, c, Tf11[r * BL + c]);
    for (int r = 0; r < BL; ++r) for (int c = 0; c < BL; ++c) put(BL + r, BL + c, Tf22[r * BL + c]);
    for (int r = 0; r < BL; ++r) for (int c = 0; c < BL; ++c) put(BL + r, c, T21[r * BL + c]);
    (void)As; (void)M_i8;
}

/* ----------------------------------- the QHPI callback ----------------------------------- */
static uint32_t gdn_solve_br_kernel(
        QHPI_RuntimeHandle *handle,
        uint32_t num_outputs, QHPI_Tensor **outputs,
        uint32_t num_inputs, const QHPI_Tensor *const *inputs) {
    (void)num_outputs; (void)num_inputs;
    if (!outputs || !outputs[0] || !inputs || !inputs[0]) return QHPI_Success;
    const uint16_t *Au = (const uint16_t *)qhpi_tensor_raw_data(inputs[0]);
    uint16_t *Tu = (uint16_t *)qhpi_tensor_raw_data(outputs[0]);
    if (!Au || !Tu) return QHPI_Success;

    const QHPI_Quant_Parameters qa = qhpi_tensor_quant_parameters(inputs[0]);
    const QHPI_Quant_Parameters qt = qhpi_tensor_quant_parameters(outputs[0]);
    const float sA = qa.stepsize, sT = qt.stepsize;
    const int32_t zpA = qa.zero_offset, zpT = qt.zero_offset;

    QHPI_Shape s = qhpi_tensor_shape(inputs[0]);
    int Cc = (s.rank >= 1) ? (int)s.dims[s.rank - 1] : C;
    if (Cc != C) return QHPI_Success;
    uint32_t heads = 1;
    for (uint32_t d = 0; d + 2 < s.rank; ++d) heads *= s.dims[d];

    uint32_t h0 = 0, h1 = heads;

#if defined(__hexagon__)
    /* VTCM scratch from the TCM_Only scratch tensor inputs[1]. */
    uint8_t *vtcm_base = (num_inputs >= 2 && inputs[1]) ? (uint8_t *)qhpi_tensor_raw_data(inputs[1]) : nullptr;
#if !defined(GDN_BR_SKIP_KERNEL) && !defined(GDN_BR_DIAG_ONLY)
    if (!vtcm_base) return QHPI_Success;
#endif
    gdn_vtcm_t vt = vtcm_base ? gdn_vtcm_from(vtcm_base) : gdn_vtcm_t{};
    int M, S; gdn_fold_MS(sA, &M, &S);
#if defined(GDN_BR_THREAD_TEST)
    {
        g_thr_sentinel = 0; g_thr_hvx_ok = 0;
        qurt_thread_t tid; qurt_thread_attr_t attr;
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_name(&attr, (char *)"gdn_br_wkr");
        qurt_thread_attr_set_stack_addr(&attr, g_thr_stack);
        qurt_thread_attr_set_stack_size(&attr, sizeof(g_thr_stack));
        int crc = qurt_thread_create(&tid, &attr, gdn_thr_worker, nullptr);
        int status = 0;
        int jrc = (crc == QURT_EOK) ? qurt_thread_join(tid, &status) : -777;
        /* report into T head0: [create_rc, hvx_ok, sentinel, join_rc] */
        uint32_t *p = (uint32_t *)Tu;
        p[0] = (uint32_t)crc; p[1] = (uint32_t)g_thr_hvx_ok;
        p[2] = (uint32_t)g_thr_sentinel; p[3] = (uint32_t)jrc;
        return QHPI_Success;
    }
#endif
#if defined(GDN_BR_PROBE_CYCLES)
    uint64_t c_diag = 0, c_pack = 0, c_hmx = 0;
#endif
    for (uint32_t h = h0; h < h1; ++h) {
        const uint16_t *Ah = Au + (size_t)h * C * C;
        uint16_t *Th = Tu + (size_t)h * C * C;

#if defined(GDN_BR_SKIP_KERNEL)
        for (int i = 0; i < C * C; ++i) Th[i] = (uint16_t)zpT;
        Th[0] = 0x4252u; /* 'BR' marker */
        Th[1] = (uint16_t)h;
        continue;
#endif
        /* ---- diagonal solves: T11=inv(I-A[0:64,0:64]), T22=inv(I-A[64:,64:]); keep int32 CODES. ---- */
#if defined(GDN_BR_PROBE_CYCLES)
        uint64_t t0; asm volatile("%0 = C15:14" : "=r"(t0));
#endif
        gdn_solve_diag64(Ah, C, zpA, M, S, g_Tc11);
        gdn_solve_diag64(Ah + BL * C + BL, C, zpA, M, S, g_Tc22);
        /* A21 folded to RAW int32 codes (scale 2^-F) — quant operand, not a scalar multiplier. */
        gdn_fold_block_raw(Ah + BL * C, C, g_A21c, zpA, M, S);
#if defined(GDN_BR_PROBE_CYCLES)
        uint64_t t1; asm volatile("%0 = C15:14" : "=r"(t1)); c_diag += t1 - t0;
#endif
        /* assemble T11, T22 into output (HVX requant; codes @ TI -> uint16 @ sT). */
        { HVX_Vector vzph = Q6_Vh_vsplat_R(zpT);
          if (((uintptr_t)Th & 127) == 0) { HVX_Vector *op = (HVX_Vector *)Th;
              for (int i = 0; i < (C * C) / 64; ++i) op[i] = vzph; }
          else { HVX_UVector *op = (HVX_UVector *)Th;
              for (int i = 0; i < (C * C) / 64; ++i) op[i] = vzph; } }
        gdn_requant_block_out(g_Tc11, GDN_BR_TI, sT, zpT, Th, 0,  0,  C);
        gdn_requant_block_out(g_Tc22, GDN_BR_TI, sT, zpT, Th, BL, BL, C);
        /* re-zero the strict-upper of each diagonal block (requant wrote the full 64 cols/row). */
        for (int r = 0; r < BL; ++r) {
            for (int c = r + 1; c < BL; ++c) Th[r * C + c] = (uint16_t)zpT;
            for (int c = r + 1; c < BL; ++c) Th[(BL + r) * C + (BL + c)] = (uint16_t)zpT;
        }
#if defined(GDN_BR_DIAG_ONLY)
        continue;   /* T21 stays at zpT (=0) -> block-diagonal */
#endif
        /* ---- merge1: M = A21 @ T11.  HVX int8 pack (exact scales from code maxabs) ---- */
#if defined(GDN_BR_PROBE_CYCLES)
        uint64_t t2; asm volatile("%0 = C15:14" : "=r"(t2));
#endif
        float sA21 = gdn_quant_u8_from_codes(g_A21c, (float)(1.0 / (1 << GDN_BR_F)), g_actbuf);
        float sT11 = gdn_quant_i8_from_codes(g_Tc11, GDN_BR_TI, g_wtbuf);
#if defined(GDN_BR_PROBE_CYCLES)
        { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_quant += q - t2; }
#endif
#if defined(GDN_BR_DUMP_ACT)
        /* write recovered int8 act (A21, code = u8-128) into T21 region, wt (T11) into upper-right. */
        for (int r = 0; r < BL; ++r) for (int c = 0; c < BL; ++c) {
            Th[(BL + r) * C + c] = (uint16_t)(((int)g_actbuf[r * BL + c] - 128) + zpT);
            Th[r * C + (BL + c)] = (uint16_t)((int)g_wtbuf[r * BL + c] + zpT);
        }
        continue;
#endif
        /* EXACT sM from the int product the HMX will compute (no scalar float matmul). */
#if defined(GDN_BR_PROBE_CYCLES)
        uint64_t tp; asm volatile("%0 = C15:14" : "=r"(tp));
#endif
        int32_t pmax1 = gdn_pint_maxabs(g_actbuf, g_wtbuf);
        float sM = ((float)pmax1 * sA21 * sT11) / 127.0f; if (sM <= 0.0f) sM = 1e-12f;
#if defined(GDN_BR_PROBE_CYCLES)
        uint64_t t3; asm volatile("%0 = C15:14" : "=r"(t3)); c_pack += t3 - t2; g_c_pint += t3 - tp;
#endif
        float gain1 = (sA21 * sT11) / sM;
        gdn_hmx_merge(&vt, g_actbuf, g_wtbuf, gain1 * 512.0f, 128 << 7, g_Mi8);
#if defined(GDN_BR_PROBE_CYCLES)
        uint64_t t4; asm volatile("%0 = C15:14" : "=r"(t4)); c_hmx += t4 - t3;
#endif
#if defined(GDN_BR_DUMP_M)
        for (int r = 0; r < BL; ++r) for (int c = 0; c < BL; ++c)
            Th[(BL + r) * C + c] = (uint16_t)((int)g_Mi8[r * BL + c] + zpT);
        continue;
#endif
        /* ---- merge2: T21 = T22 @ M (act=T22 zp128, wt=M_i8 @ scale sM) ---- */
#if defined(GDN_BR_PROBE_CYCLES)
        uint64_t t5; asm volatile("%0 = C15:14" : "=r"(t5));
#endif
        float sT22 = gdn_quant_u8_from_codes(g_Tc22, GDN_BR_TI, g_actbuf);  /* act = T22 (u8 zp128) */
        /* EXACT sT21 from the int product T22q @ M_i8 (reuses gdn_pint_maxabs: act=T22, wt=M_i8). */
        int32_t pmax2 = gdn_pint_maxabs(g_actbuf, g_Mi8);
        float sT21 = ((float)pmax2 * sT22 * sM) / 127.0f; if (sT21 <= 0.0f) sT21 = 1e-12f;
#if defined(GDN_BR_PROBE_CYCLES)
        uint64_t t6; asm volatile("%0 = C15:14" : "=r"(t6)); c_pack += t6 - t5;
#endif
        float gain2 = (sT22 * sM) / sT21;
        gdn_hmx_merge(&vt, g_actbuf, g_Mi8, gain2 * 512.0f, 128 << 7, g_T21i);
#if defined(GDN_BR_PROBE_CYCLES)
        uint64_t t7; asm volatile("%0 = C15:14" : "=r"(t7)); c_hmx += t7 - t6;
#endif
        /* dequant + scatter T21 = code*sT21 into the lower-left block (HVX requant; int8 codes). */
        for (int i = 0; i < BL * BL; ++i) g_Tc[i] = (int32_t)g_T21i[i];   /* widen i8 -> i32 codes */
        gdn_requant_block_out(g_Tc, sT21, sT, zpT, Th, BL, 0, C);
    }
#if defined(GDN_BR_PROBE_CYCLES)
    if (h0 < h1) {
        uint16_t *Th0 = Tu + (size_t)h0 * C * C;
        uint32_t *p = (uint32_t *)Th0;
        p[0] = (uint32_t)c_diag; p[1] = (uint32_t)c_pack; p[2] = (uint32_t)c_hmx; p[3] = (h1 - h0);
        p[4] = (uint32_t)g_c_hmxpack; p[5] = (uint32_t)g_c_hmxkern; p[6] = (uint32_t)g_c_hmxdepack;
        p[7] = (uint32_t)g_c_quant; p[8] = (uint32_t)g_c_pint;
    }
#endif
    (void)handle;
#else  /* x86 fallback */
    for (uint32_t h = h0; h < h1; ++h)
        gdn_br_head_scalar(Au + (size_t)h * C * C, zpA, sA, sT, zpT, Tu + (size_t)h * C * C);
#endif
    return QHPI_Success;
}

static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
    /* scratch: VTCM workspace for the HMX merge surfaces (act/wt/bias/out/tables). */
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};

static float gdn_solve_br_cost(uint32_t num_inputs, const QHPI_Tensor *const *inputs) {
    if (!inputs || num_inputs < 1 || !inputs[0]) return 1.0f;
    QHPI_Shape s = qhpi_tensor_shape(inputs[0]);
    float n = 1.0f; for (uint32_t d = 0; d < s.rank; ++d) n *= (float)s.dims[d];
    return n;
}

static QHPI_Kernel_v1 sg_kernels[] = {
    {
        THIS_PKG_NAME_STR "::gdn_solve_br_kernel",
        gdn_solve_br_kernel,
        QHPI_RESOURCE_HMX,        /* HMX op; HVX intrinsics used freely inside (cf. HvxHmxOp tutorial) */
        false, false, false, false,   /* multithreaded=false: HMX ops are not self-sliced by prepare */
        2, sig_inputs,
        1, sig_outputs,
        gdn_solve_br_cost,
        0,
        0, nullptr, nullptr,
        nullptr,
    },
};

/* No central-tiler tiling for M_op: the op is HMX-resource (single matrix thread), processes all
 * heads in one callback.  Multi-thread / HVX∥HMX pipelining across heads is M3. */
static QHPI_OpInfo_v1 sg_ops[] = {
    {
        THIS_PKG_NAME_STR "::GdnSolveBR",
        1, sg_kernels,
        nullptr,
        nullptr,
        nullptr,
        0,
        nullptr,
        nullptr,
    },
};

extern "C" void register_gdn_solve_br_op(void) {
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
