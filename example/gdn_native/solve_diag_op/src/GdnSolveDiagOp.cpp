/*
 * GdnSolveDiagOp.cpp -- QHPI custom op "GdnSolveDiag" (M6 split, Op1).
 *
 * PURE-HVX op, multithreaded=true + central-tiler over the head dim (tiles of 8 heads): solves the
 * NB=C/64 diagonal 64x64 blocks T_ii=inv(I-A_ii) with the proven int16-packed forward-subst, narrows
 * each diagonal block to int8 32x32 row-major tiles into a VTCM scratch (inputs[1], TCM_Only) for the
 * HMX-merge op (Op2), quantizes the off-diag A_ij blocks to int8 tiles too, and -- for validation --
 * also dequantizes the diagonal int8 tiles into the T output so the harness can check them.
 *
 *   input[0]  A : QUInt16  [B,H,C,C]   (uint16-midpoint, scale sA, zp ~32768)
 *   input[1]  S : QUInt8   [1,1,1,N]   (VTCM scratch, TCM_Only; full buffer, NOT sliced)
 *   output[0] T : QUInt16  [B,H,C,C]   (block-diagonal: T_ii filled, off-diag = zpT)
 *
 * The central tiler self-slices the op across HVX threads -- the parallelism the fused HMX op lost.
 *
 * Debug: GDN_BR_PROBE_CYCLES -> per-stage cycle work-volume into T head 0 (uint32).
 */
#include "HTP/core/qhpi.h"
#include "gdn_solve_diag_core.h"

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

static const int C  = GDN_BR_C;
static const int BL = GDN_BR_BL;
static const int NB = GDN_BR_NB;
static inline int gdn_blk_index(int i, int j) { return (i * (i + 1)) / 2 + j; }

/* Handoff layout (per head, carried in Op1's 2nd OUTPUT tensor Hd = a real graph edge so Op2 sees it):
 * NBLK int8 blocks (4096 B each, as 4 x 32x32 row-major tiles) + a per-block float scale table.
 * Hd is [B,H,C,C] uint8 -> C*C bytes/head; for C=256 that's 65536 B = 0x10000.  Must hold
 * NBLK*4096 + NBLK*4 (NB=4 -> NBLK=10 -> 40960+40 < 65536; NB=2 -> 3 blocks, C=128 -> 16384 B fits). */
#define GDN_DIAG_BLK_BYTES   (GDN_BR_BL * GDN_BR_BL)
#define GDN_DIAG_HEAD_STRIDE  ((size_t)GDN_BR_C * GDN_BR_C)   /* = bytes per head in the uint8 Hd output */

#if defined(__hexagon__)
#include <hexagon_types.h>
#include <hexagon_protos.h>

static void gdn_fold_block_hvx(const uint16_t *Au, int row_stride, int32_t *Afx, int zpA, int M, int S) {
    const int Mrep = (M & 0xFFFF) * 0x10001;
    const HVX_Vector vzp = Q6_V_vsplat_R(zpA), vrndS = Q6_V_vsplat_R(1 << (S - 1)), m16 = Q6_V_vsplat_R(0xFFFF);
    for (int r = 0; r < BL; ++r) {
        const HVX_UVector *Av = (const HVX_UVector *)(Au + r * row_stride);
        HVX_Vector *Afxv = (HVX_Vector *)(Afx + r * BL);
        HVX_VectorPair w = Q6_Wuw_vzxt_Vuh(Av[0]);
        HVX_Vector c0 = Q6_Vw_vsub_VwVw(Q6_V_lo_W(w), vzp), c1 = Q6_Vw_vsub_VwVw(Q6_V_hi_W(w), vzp);
        HVX_Vector i0 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(c0, Mrep), vrndS), S);
        HVX_Vector i1 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(c1, Mrep), vrndS), S);
        HVX_VectorPair sh = Q6_W_vshuff_VVR(i1, i0, -4);
        HVX_Vector tl = Q6_V_vand_VV(Q6_V_lo_W(sh), m16), th = Q6_V_vand_VV(Q6_V_hi_W(sh), m16);
        Afxv[0] = Q6_V_vor_VV(tl, Q6_Vw_vasl_VwR(tl, 16));
        Afxv[1] = Q6_V_vor_VV(th, Q6_Vw_vasl_VwR(th, 16));
    }
}

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
        HVX_VectorPair sh = Q6_W_vshuff_VVR(i1, i0, -4);
        Afxv[0] = Q6_V_lo_W(sh);
        Afxv[1] = Q6_V_hi_W(sh);
    }
}

static void gdn_solve_diag64(int32_t *Afx, int16_t *Tc16, const uint16_t *Au, int row_stride,
                             int zpA, int M, int S, int32_t *Tcout) {
    gdn_fold_block_hvx(Au, row_stride, Afx, zpA, M, S);
    const int16_t ei16 = (int16_t)(int)(1.0f / GDN_BR_TI + 0.5f);
    for (int i = 0; i < BL; ++i) {
        HVX_VectorPair acc = Q6_W_vzero();
        for (int k = 0; k < i; ++k)
            acc = Q6_Ww_vmpyacc_WwVhRh(acc, ((const HVX_Vector *)(Tc16 + k * BL))[0], Afx[i * BL + k]);
        ((HVX_Vector *)(Tc16 + i * BL))[0] =
            Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(acc), Q6_V_lo_W(acc), GDN_BR_F);
        Tc16[i * BL + i] += ei16;
    }
    for (int i = 0; i < BL; ++i) {
        HVX_VectorPair w = Q6_Ww_vsxt_Vh(((const HVX_Vector *)(Tc16 + i * BL))[0]);
        HVX_VectorPair nat = Q6_W_vshuff_VVR(Q6_V_hi_W(w), Q6_V_lo_W(w), -4);
        ((HVX_Vector *)(Tcout + i * BL))[0] = Q6_V_lo_W(nat);
        ((HVX_Vector *)(Tcout + i * BL))[1] = Q6_V_hi_W(nat);
    }
}

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

/* quantize 64x64 int32 codes (value=code*scale_in) -> int8 codes at sQ=maxabs/127, natural order. */
static float gdn_quant_i8_from_codes(int32_t *qbuf, const int32_t *codes, float scale_in, int8_t *out) {
    int32_t mx = gdn_maxabs_codes(codes);
    float maxval = (float)mx * scale_in;
    float sQ = (maxval > 0.0f) ? (maxval / 127.0f) : 1e-12f;
    float g = scale_in / sQ;
    int Q = 14;
    while (Q > 1  && g * (float)(1 << Q) >= 32768.0f) --Q;
    while (Q < 28 && g * (float)(1 << (Q + 1)) < 16384.0f) ++Q;
    int32_t Mg = (int32_t)(g * (float)(1 << Q) + 0.5f);
    const HVX_Vector vrnd = Q6_V_vsplat_R(1 << (Q - 1));
    const HVX_Vector vlim = Q6_V_vsplat_R(127), vnlim = Q6_V_vsplat_R(-127);
    const HVX_Vector *p = (const HVX_Vector *)codes;
    HVX_Vector *qp = (HVX_Vector *)qbuf;
    const int Mrep = (Mg & 0xFFFF) * 0x10001;
    for (int b = 0; b < (BL * BL) / 32; ++b) {
        HVX_Vector prod = Q6_Vw_vmpyi_VwRh(p[b], Mrep);
        HVX_Vector q = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(prod, vrnd), Q);
        q = Q6_Vw_vmin_VwVw(q, vlim); q = Q6_Vw_vmax_VwVw(q, vnlim);
        qp[b] = q;
    }
    HVX_Vector *op = (HVX_Vector *)out;
    for (int v = 0; v < (BL * BL) / 128; ++v) {
        HVX_Vector h0 = Q6_Vh_vpack_VwVw_sat(qp[v*4+1], qp[v*4+0]);
        HVX_Vector h1 = Q6_Vh_vpack_VwVw_sat(qp[v*4+3], qp[v*4+2]);
        op[v] = Q6_Vb_vpack_VhVh_sat(h1, h0);
    }
    return sQ;
}

static void gdn_requant_block_out(const int32_t *codes, float scale_in, float sT, int zpT,
                                  uint16_t *Th, int roff, int coff, int row_stride) {
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

/* write a 64x64 int8 natural block as four 32x32 row-major tiles into dst (tile-row-major).
 * Vectorized: a 128B HVX vector holds two consecutive natural rows = [La Ra Lb Rb] (4x 32B groups,
 * L=cols0..31, R=cols32..63).  vdeal at 32-byte granularity -> [La Lb Ra Rb]: low half (64B) = the two
 * tileL rows, high half = the two tileR rows.  For rows 0..31 the L/R tiles are dst+0 / dst+1024; for
 * rows 32..63 they are dst+2048 / dst+3072.  Stores are 64B (half-vector) but VTCM scratch is padded
 * so a 128B store is safe; we use the proven unaligned 128B store of the dealt vector split per tr. */
static void gdn_write_tiles_64(const int8_t *blk_natural, int8_t *dst) {
#if defined(GDN_DIAG_SKIP_TILEWRITE)
    (void)blk_natural; (void)dst; return;
#endif
    const HVX_Vector *src = (const HVX_Vector *)blk_natural;   /* 32 vectors (2 rows each) */
    for (int tr = 0; tr < 2; ++tr) {
        HVX_Vector *tileL = (HVX_Vector *)(dst + (tr * 2 + 0) * 1024);  /* 8 vectors, 4 rows each */
        HVX_Vector *tileR = (HVX_Vector *)(dst + (tr * 2 + 1) * 1024);
        /* tr covers rows tr*32..+31 = source vectors tr*16..+15.  Process a PAIR of source vectors
         * (4 natural rows) per output vector: deal each -> [La Lb | Ra Rb]; valign(d1,d0,64) makes
         * [Lb Ra | Lc Rd]... instead combine the low halves of the two dealt vectors into one tileL
         * vector and the high halves into one tileR vector via shuffle of 64-byte lanes. */
        for (int p = 0; p < 8; ++p) {
            /* pair {v1,v0}: v0=[La Ra Lb Rb], v1=[Lc Rc Ld Rd] (32B groups).  vdeal at 32B granularity
             * -> lo=[La Lb Lc Ld] (4 tileL rows), hi=[Ra Rb Rc Rd] (4 tileR rows), one instruction. */
            HVX_Vector v0 = src[tr * 16 + p * 2 + 0];
            HVX_Vector v1 = src[tr * 16 + p * 2 + 1];
            HVX_VectorPair de = Q6_W_vdeal_VVR(v1, v0, -32);
            tileL[p] = Q6_V_lo_W(de);
            tileR[p] = Q6_V_hi_W(de);
        }
    }
}

#ifndef GDN_DIAG_NT
#define GDN_DIAG_NT 8
#endif
struct gdn_dscr_t {
    int32_t Afx [GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128)));
    int16_t Tc16[GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128)));
    int32_t Tcd [GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128)));
    int32_t Aoff[GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128)));
    int32_t qbuf[GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128)));
    int8_t  i8  [GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128)));
};
static gdn_dscr_t g_dscr[GDN_DIAG_NT];

#if defined(GDN_BR_PROBE_CYCLES)
volatile uint64_t g_d_diag = 0, g_d_tiles = 0, g_d_quantA = 0, g_d_requant = 0;
volatile uint32_t g_d_nslices = 0;
#endif

static inline void gdn_fold_MS(float sA, int *pM, int *pS) {
    float sf = sA * (float)(1 << GDN_BR_F);
    int S = 14;
    while (S < 30 && sf * (float)(1 << (S + 1)) < 30000.0f) ++S;
    while (S >  0 && sf * (float)(1 <<  S)      > 32760.0f) --S;
    *pM = (int)(sf * (float)(1 << S) + 0.5f); *pS = S;
}

/* slot claim for unique scratch per concurrent tile-op thread (lock-free, like GdnSolveOp). */
static volatile int g_slot_taken[GDN_DIAG_NT] = {0};
static int gdn_claim_slot() {
    for (;;)
        for (int i = 0; i < GDN_DIAG_NT; ++i)
            if (__sync_bool_compare_and_swap(&g_slot_taken[i], 0, 1)) return i;
}
static void gdn_free_slot(int s) { __sync_lock_release(&g_slot_taken[s]); }
#endif /* __hexagon__ */

/* ----------------------------- host (x86) reference fallback ----------------------------- */
#include "../../solve_op/src/gdn_solve_core.h"
static void gdn_diag_head_scalar(const uint16_t *Au, int zpA, float sA, float sT, int zpT, uint16_t *Tu) {
    int16_t sub16[GDN_BR_BL * GDN_BR_BL], Tcode[GDN_BR_BL * GDN_BR_BL];
    for (int q = 0; q < C * C; ++q) Tu[q] = (uint16_t)zpT;
    for (int i = 0; i < NB; ++i) {
        for (int r = 0; r < BL; ++r) for (int c = 0; c < BL; ++c)
            sub16[r * BL + c] = (int16_t)((int)Au[(i * BL + r) * C + (i * BL + c)] - zpA);
        gdn_solve_head_q<int16_t>(sub16, BL, sA, GDN_BR_TI, 32767.0f, Tcode);
        for (int r = 0; r < BL; ++r) for (int c = 0; c <= r; ++c) {
            float v = (float)Tcode[r * BL + c] * GDN_BR_TI;
            long q = lroundf(v / sT); if (q > 32767) q = 32767; if (q < -32767) q = -32767;
            Tu[(i * BL + r) * C + (i * BL + c)] = (uint16_t)((int)q + zpT);
        }
    }
}

/* ----------------------------------- the QHPI callback ----------------------------------- */
static uint32_t gdn_solve_diag_kernel(
        QHPI_RuntimeHandle *handle,
        uint32_t num_outputs, QHPI_Tensor **outputs,
        uint32_t num_inputs, const QHPI_Tensor *const *inputs) {
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

    /* the central tiler hands this call a head-tile (heads = tile extent); within the tile we also
     * self-slice via qhpi_num_slices in case the runtime sub-splits. */
    uint32_t ns = qhpi_num_slices(handle), sl = qhpi_slice_number(handle);
    if (ns == 0) ns = 1;
    uint32_t h0 = (uint64_t)heads * sl / ns, h1 = (uint64_t)heads * (sl + 1) / ns;

#if defined(__hexagon__)
    /* handoff via the 2nd OUTPUT tensor Hd (a real graph edge -> Op2 sees Op1's writes; a shared
     * constant did NOT survive across ops).  Hd is [B,H,C,C] uint8 == C*C bytes/head. */
    uint8_t *hd_base = (num_outputs >= 2 && outputs[1]) ? (uint8_t *)qhpi_tensor_raw_data(outputs[1]) : nullptr;
    int M, S; gdn_fold_MS(sA, &M, &S);
    int slot = gdn_claim_slot();
    gdn_dscr_t *sc = &g_dscr[slot];
#if defined(GDN_BR_PROBE_CYCLES)
    g_d_nslices = (g_d_nslices < ns) ? ns : g_d_nslices;
#endif
    for (uint32_t h = h0; h < h1; ++h) {
        const uint16_t *Ah = Au + (size_t)h * C * C;
        uint16_t *Th = Tu + (size_t)h * C * C;
        /* tile-local head h: QNN already offset Hd/T to this tile's head range. */
        int8_t *scr_h = hd_base ? (int8_t *)(hd_base + (size_t)h * GDN_DIAG_HEAD_STRIDE) : nullptr;
        float *scale_tab = scr_h ? (float *)(scr_h + (size_t)GDN_BR_NBLK * GDN_DIAG_BLK_BYTES) : nullptr;
        { HVX_Vector vzph = Q6_Vh_vsplat_R(zpT);
          if (((uintptr_t)Th & 127) == 0) { HVX_Vector *op = (HVX_Vector *)Th;
              for (int i = 0; i < (C * C) / 64; ++i) op[i] = vzph; }
          else { HVX_UVector *op = (HVX_UVector *)Th;
              for (int i = 0; i < (C * C) / 64; ++i) op[i] = vzph; } }
#if defined(GDN_BR_PROBE_CYCLES)
        uint64_t t0; asm volatile("%0 = C15:14" : "=r"(t0));
#endif
        for (int i = 0; i < NB; ++i) {
            gdn_solve_diag64(sc->Afx, sc->Tc16, Ah + (size_t)i * BL * C + i * BL, C, zpA, M, S, sc->Tcd);
#if defined(GDN_BR_PROBE_CYCLES)
            { uint64_t t; asm volatile("%0 = C15:14" : "=r"(t)); g_d_diag += t - t0; t0 = t; }
#endif
            if (scr_h) {
                int bi = gdn_blk_index(i, i);
                float sQ = gdn_quant_i8_from_codes(sc->qbuf, sc->Tcd, GDN_BR_TI, sc->i8);
                gdn_write_tiles_64(sc->i8, scr_h + (size_t)bi * GDN_DIAG_BLK_BYTES);
                if (scale_tab) scale_tab[bi] = sQ;
            }
#if defined(GDN_BR_PROBE_CYCLES)
            { uint64_t t; asm volatile("%0 = C15:14" : "=r"(t)); g_d_quantA += t - t0; t0 = t; }
#endif
            gdn_requant_block_out(sc->Tcd, GDN_BR_TI, sT, zpT, Th, i * BL, i * BL, C);
            for (int r = 0; r < BL; ++r)
                for (int c = r + 1; c < BL; ++c) Th[(i * BL + r) * C + (i * BL + c)] = (uint16_t)zpT;
#if defined(GDN_BR_PROBE_CYCLES)
            { uint64_t t; asm volatile("%0 = C15:14" : "=r"(t)); g_d_requant += t - t0; t0 = t; }
#endif
        }
#if !defined(GDN_DIAG_SKIP_OFFDIAG)
        if (scr_h) {
            for (int d = 1; d < NB; ++d)
                for (int j = 0; j + d < NB; ++j) {
                    int i = j + d, bij = gdn_blk_index(i, j);
                    gdn_fold_block_raw(Ah + (size_t)i * BL * C + j * BL, C, sc->Aoff, zpA, M, S);
                    float sQ = gdn_quant_i8_from_codes(sc->qbuf, sc->Aoff,
                                                       (float)(1.0 / (1 << GDN_BR_F)), sc->i8);
                    gdn_write_tiles_64(sc->i8, scr_h + (size_t)bij * GDN_DIAG_BLK_BYTES);
                    if (scale_tab) scale_tab[bij] = sQ;
                }
        }
#endif
#if defined(GDN_BR_PROBE_CYCLES)
        { uint64_t t; asm volatile("%0 = C15:14" : "=r"(t)); g_d_tiles += t - t0; }
#endif
    }
    gdn_free_slot(slot);
#if defined(GDN_BR_PROBE_CYCLES)
    if (h1 > h0 && sl == 0) {
        uint32_t *p = (uint32_t *)(Tu + (size_t)h0 * C * C);
        p[0] = (uint32_t)g_d_diag; p[1] = g_d_nslices; p[2] = (uint32_t)heads; p[3] = (h1 - h0);
        p[4] = (uint32_t)g_d_quantA; p[5] = (uint32_t)g_d_requant; p[6] = (uint32_t)g_d_tiles;
    }
#endif
#else  /* x86 fallback */
    for (uint32_t h = h0; h < h1; ++h)
        gdn_diag_head_scalar(Au + (size_t)h * C * C, zpA, sA, sT, zpT, Tu + (size_t)h * C * C);
#endif
    (void)handle;
    return QHPI_Success;
}

static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},   /* A */
};
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},   /* T  (block-diag) */
#if defined(GDN_DIAG_HD_DDR)
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},   /* Hd: DDR (isolated-probe build only) */
#else
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},     /* Hd handoff: keep on-chip VTCM */
#endif
};

static float gdn_solve_diag_cost(uint32_t num_inputs, const QHPI_Tensor *const *inputs) {
    if (!inputs || num_inputs < 1 || !inputs[0]) return 1.0f;
    QHPI_Shape s = qhpi_tensor_shape(inputs[0]);
    float n = 1.0f; for (uint32_t d = 0; d < s.rank; ++d) n *= (float)s.dims[d];
    return n;
}

static QHPI_Kernel_v1 sg_kernels[] = {
    {
        THIS_PKG_NAME_STR "::gdn_solve_diag_kernel",
        gdn_solve_diag_kernel,
        QHPI_RESOURCE_HVX,
        false, true,  false, false,   /* multithreaded=true */
        1, sig_inputs,
        2, sig_outputs,
        gdn_solve_diag_cost,
        0,
        0, nullptr, nullptr,
        nullptr,
    },
};

/* tile over heads (8/tile) so the op splits across HVX threads.  Slice ONLY A and T (input 0/output 0);
 * the scratch input S is left whole (Op2/Op1 index it by absolute head). */
static QHPI_Shape gdn_diag_shape_required(const QHPI_Op *op) {
    (void)op;
    QHPI_Shape req; req.rank = 4;
    req.dims[0] = 1;
    req.dims[1] = 8;
    req.dims[2] = QHPI_DO_NOT_TILE;
    req.dims[3] = QHPI_DO_NOT_TILE;
    return req;
}
static const QHPI_Op *gdn_diag_build_tile(const QHPI_Op *op, const QHPI_Shape *out_start,
                                          const QHPI_Shape *out_extent) {
    QHPI_OpRef inA = qhpi_op_input(op, 0);
    QHPI_Shape st = *out_start, ex = *out_extent;
    QHPI_OpRef inA_slice = qhpi_op_slice(inA, &st, &ex);
    QHPI_OpRef inputs[] = { inA_slice };
    QHPI_OutputDef o0 = qhpi_op_output(op, 0);
    QHPI_OutputDef o1 = qhpi_op_output(op, 1);
    /* both outputs share the head-tile extent (T and Hd are [B,H,C,C]). */
    QHPI_OutputDef outputs[] = { { o0.type, o0.quant_parameters, *out_extent },
                                 { o1.type, o1.quant_parameters, *out_extent } };
    return qhpi_op_create(op, qhpi_op_name(op), 1, inputs, 2, outputs);
}

static QHPI_OpInfo_v1 sg_ops[] = {
    {
        THIS_PKG_NAME_STR "::GdnSolveDiag",
        1, sg_kernels,
        nullptr,
        gdn_diag_shape_required,
        nullptr,
        0,
        gdn_diag_build_tile,
        nullptr,
    },
};

extern "C" void register_gdn_solve_diag_op(void) {
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
