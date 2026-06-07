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
/* GDN_BR_NO_QHPI: include just the device solve (no QHPI wrapper) — used by the bare-metal FastRPC HAP
 * (example/gdn_native/baremetal) to reuse this exact validated solve outside the QNN framework. */
#ifndef GDN_BR_NO_QHPI
#include "HTP/core/qhpi.h"
#endif
#include "gdn_solve_br_core.h"

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

/* HMX critical-section hooks around the mxmem kernel.  Default no-op (QNN op: the backend holds HMX).
 * The bare-metal HAP overrides these (before including this file) to lock/unlock HMX per-thread so the
 * HVX glue runs OUTSIDE the lock and worker threads parallelize. */
#ifndef GDN_BR_HMX_ENTER
#define GDN_BR_HMX_ENTER() ((void)0)
#endif
#ifndef GDN_BR_HMX_EXIT
#define GDN_BR_HMX_EXIT() ((void)0)
#endif

#define GDN_BR_MAX_SLICES 8

static const int C  = GDN_BR_C;    /* 128 or 256 */
static const int BL = GDN_BR_BL;   /* 64  */
static const int NB = GDN_BR_NB;   /* 2 (C=128) or 4 (C=256) */
#define GDN_BR_NBLK ((GDN_BR_NB * (GDN_BR_NB + 1)) / 2)   /* lower-tri block count (3 for nb=2, 10 for nb=4) */
static inline int gdn_blk_index(int i, int j) { return (i * (i + 1)) / 2 + j; }  /* row-major lower-tri */

#if defined(__hexagon__)
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include "qurt.h"            /* manual qurt worker threads for the HVX∥HMX head pipeline */

#if defined(GDN_BR_THREAD_TEST)
/* Feasibility probe: can a QHPI HMX-resource callback spawn a qurt worker thread?  The worker does
 * a pure-HVX op (no HMX) and writes a sentinel; the callback joins it.  Tests whether manual qurt
 * threading (the M3-pipeline prerequisite) is permitted inside the op, and whether a spawned thread
 * can acquire an HVX context alongside the backend's own HVX threads. */
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
#ifndef GDN_BR_NT
#define GDN_BR_NT 4                 /* number of worker threads (heads partitioned across them) */
#endif
/* Per-thread working scratch.  All the block buffers a single head's recursion touches live here so
 * that GDN_BR_NT heads can be processed concurrently by GDN_BR_NT qurt worker threads with no races.
 * (g_Sf is the dead float-accumulate path, retired by the int-only S accumulation.) */
struct gdn_scr_t {
    int32_t Tc  [GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128)));   /* general int32 scratch / pint W */
    int32_t Afx [GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128)));   /* diag-solve fold scratch */
    uint8_t actbuf[GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128))); /* u8 activation (zp128) */
    int8_t  wtbuf [GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128))); /* i8 weight (k-major src) */
    int32_t qbuf  [GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128))); /* quant / widen scratch */
    int32_t Tblk[GDN_BR_NBLK][GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128)));
    float   Tscl[GDN_BR_NBLK];
    int32_t Aoff[GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128)));   /* A_ik folded codes @2^-F */
    int32_t Sacc[GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128)));   /* inner-sum accumulator */
    int8_t  termi[GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128)));  /* one merge term as int8 */
    uint8_t surf_sub[GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128)));/* depack base-subtracted surface */
    int32_t eff[64] __attribute__((aligned(128)));                       /* folded-bias effective[] */
    int16_t Tc16[GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128)));   /* int16-packed diag T codes */
    int16_t a16[GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128)));    /* int16-HVX merge: 12-bit A operand */
    int16_t b16[GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128)));    /* int16-HVX merge: 12-bit B operand */
#if defined(GDN_BR_MM_I8)
    int8_t  a8 [GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128)));    /* int8 vrmpy: A operand (row-major) */
    int8_t  b8 [GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128)));    /* int8 vrmpy: B operand (row-major k) */
    int8_t  btp[GDN_BR_BL * GDN_BR_BL] __attribute__((aligned(128)));    /* int8 vrmpy: B transposed/packed [g][col][4] */
#endif
    /* ---- per-head operand-reuse cache metadata (Task 2): quantize+pack each distinct operand ONCE ----
     * the packed surfaces live in VTCM (gdn_vtcm_t acache/wcache); here we keep scales + valid flags.
     * A_ik act key gdn_blk_index(i,k) (10->6); T_ii act key i (6->3); T_kj wt key gdn_blk_index(k,j) (10->6). */
    float   sAa[GDN_BR_NBLK], sTw[GDN_BR_NBLK], sTa[GDN_BR_NB];
    char    vAa[GDN_BR_NBLK], vTw[GDN_BR_NBLK], vTa[GDN_BR_NB];
    int32_t mxdiag[GDN_BR_NB];   /* producer-tracked maxabs of each diagonal T block (for quant fusion) */
    int32_t effc[GDN_BR_NBLK][64] __attribute__((aligned(128)));  /* cached effective-bias per T-wt block */
    int     colabsc[GDN_BR_NBLK];                                 /* cached max-col-abs-sum per T-wt block (#1c) */
};
static gdn_scr_t g_scr[GDN_BR_NT];

#if defined(GDN_BR_STATIC_FULL)
/* FULL STATIC quantization (precision先不管): every operand+output uses a fixed (calibrated) scale, so
 * ALL runtime dynamic-quant work is eliminated -- no maxabs scans, no colabs, no gain-search, pure-add acc.
 * Calibrated on A_u16_h32 (global max scales). */
#define GDN_OPS_sAa 4.895068e-03f
#define GDN_OPS_sTw 7.874256e-03f
#define GDN_OPS_sTa 7.874256e-03f
#define GDN_OPS_COLABS 556
#define GDN_OPS_sSacc 3.742e-03f   /* calibrated max Sacc scale (A_u16_h32) */
static float g_cal_swS = 0.f;
/* These toggle between phase-dependent values (sTw / sSacc / 0) WITHIN a head solve, so in the multi-producer
 * GDNSolveHVXMixHMX pipeline they MUST be per-thread — else producer A's Sacc-phase write clobbers producer B's
 * off-diag-wt-phase value (data race -> wrong quant scale -> corrupt T). __thread = each producer its own copy. */
static __thread float g_ops_u8 = 0.f, g_ops_i8 = 0.f;   /* >0 => fixed output scale for u8/i8 quant */
static __thread float g_force_sP = 0.f;   /* >0 => force gdn_merge_packed output to drain at this scale (e.g. sTw) */
#endif

/* VTCM scratch carved from the TCM_Only scratch tensor.  Buffers are spaced 0x10000 (64 KB) apart —
 * matching the proven M1 sim harness layout — so any HMX over-write/alignment slack can't clobber a
 * neighbouring buffer.  Total span < 0x60000 (384 KB; graph declares >= that). */
struct gdn_vtcm_t {
    uint8_t *act;     /* 4096  (transient act crouton, used for non-cached path) */
    int8_t  *wt;      /* 4096  (transient wt k-major, used for the Sacc final-merge weight) */
    int32_t *bias;    /* 128 int32 = 512 B */
    uint8_t *out;     /* 4096 */
    int32_t *acttab;  /* 2 */
    int32_t *outtab;  /* 2 */
    /* ---- VTCM packed-operand caches (Task 2): pack each distinct operand's HMX surface ONCE/head ----
     * A-act crouton  : acache + key*0x1000  (key = gdn_blk_index(i,k), 0..NBLK-1)
     * Tdiag-act      : acache + 0xA000 + i*0x1000
     * T-wt k-major   : wcache + key*0x1000  (key = gdn_blk_index(k,j)) */
    uint8_t *acache;  /* A-act + Tdiag-act crouton cache (base+0x41000 .. +0x4F000) */
    int8_t  *wcache;  /* T-wt k-major cache (base+0x4F000 .. +0x59000) */
};
static gdn_vtcm_t gdn_vtcm_from(uint8_t *base) {
    gdn_vtcm_t v;
    v.act    = base + 0x00000;
    v.wt     = (int8_t *)(base + 0x10000);
    v.bias   = (int32_t *)(base + 0x20000);
    v.out    = base + 0x30000;
    v.acttab = (int32_t *)(base + 0x40000);
    v.outtab = (int32_t *)(base + 0x40080);
    v.acache = base + 0x41000;
    v.wcache = (int8_t *)(base + 0x4F000);
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
static void gdn_fold_block_raw(const uint16_t *Au, int row_stride, int32_t *Afx, int zpA, int M, int S,
                               int32_t *mx_out = nullptr) {
    const int Mrep = (M & 0xFFFF) * 0x10001;
    const HVX_Vector vzp = Q6_V_vsplat_R(zpA), vrndS = Q6_V_vsplat_R(1 << (S - 1));
    HVX_Vector vmax = Q6_V_vzero();
    for (int r = 0; r < BL; ++r) {
        const HVX_UVector *Av = (const HVX_UVector *)(Au + r * row_stride);
        HVX_Vector *Afxv = (HVX_Vector *)(Afx + r * BL);
        HVX_VectorPair w = Q6_Wuw_vzxt_Vuh(Av[0]);
        HVX_Vector c0 = Q6_Vw_vsub_VwVw(Q6_V_lo_W(w), vzp), c1 = Q6_Vw_vsub_VwVw(Q6_V_hi_W(w), vzp);
        HVX_Vector i0 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(c0, Mrep), vrndS), S);
        HVX_Vector i1 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(c1, Mrep), vrndS), S);
        if (mx_out) {   /* track maxabs of the produced codes so quant can skip its own maxabs pass */
            vmax = Q6_Vw_vmax_VwVw(vmax, Q6_Vw_vabs_Vw(i0));
            vmax = Q6_Vw_vmax_VwVw(vmax, Q6_Vw_vabs_Vw(i1));
        }
        HVX_VectorPair s = Q6_W_vshuff_VVR(i1, i0, -4);   /* even/odd -> natural col order */
        Afxv[0] = Q6_V_lo_W(s);
        Afxv[1] = Q6_V_hi_W(s);
    }
    if (mx_out) {
        vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4 * 16));
        vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4 * 8));
        vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4 * 4));
        vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4 * 2));
        vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4 * 1));
        int32_t lanes[32] __attribute__((aligned(128)));
        *(HVX_Vector *)lanes = vmax;
        *mx_out = lanes[0];
    }
}

/* HVX int16 forward-substitution main loop, isolated for measurement (phase-1 极致优化 study).
 * T_i = e_i + sum_{k<i} A_ik * T_k, all int16 codes (scale GDN_BR_TI).  Each term is ONE
 * Q6_Ww_vmpyacc_WwVhRh (64 int16 halfwords x scalar Afx -> 64 int32 word pair, accumulate).
 *
 * FLOOR vs MEASURED (device, CSE-proof bench -DGDN_BR_DIAG_SPLIT o[5], 2026-06-05):
 *   - pure-multiply floor: a halfword x halfword multiply is a DOUBLE-vector instruction (consumes BOTH
 *     multiply resources, HVX V75 ref 4.1.1) -> 1/cycle max.  sum_{i<64} i = 2016 vmpyacc -> ~2016 cyc.
 *   - measured ~6402 cyc/block = 3.2x the floor.  Binding constraint = the per-term SCALAR Afx load:
 *     2016 distinct A_ik, each used once, scheduled only 1 packet before the vmpyacc that consumes it as
 *     Rt, so the ~3-cyc scalar-load latency is NOT hidden.  (Tc16 row vectors are reused/hot -> off the
 *     critical path; the vasr narrow is on the shift resource -> overlaps the multiplies.)
 *   - pure-vmpyacc throughput probe (inline-asm, register-resident, NO loads/deps): ~0.4-1 cyc/op
 *     -> the multiply is NOT the limit; the ~3.2 cyc/op is the load feed.  Load-bound CONFIRMED.
 *   - idioms TRIED, all device-measured, NONE beat 6402 (so kept the simple single-acc form): 4
 *     independent accumulators 6404 (rules OUT an acc->acc latency chain); bulk scalar preload into a
 *     local array 8387 (worse); C-level 4-deep scalar-prefetch SW-pipeline 6402 (the -O2 scheduler
 *     re-collapses it back to 1-packet-ahead -> needs inline-asm to force); const-scalar 289 (folded).
 *     2x2 block recursion / byte-split only INCREASE the multiply count (3040 / 2x) -> cannot beat 2016.
 *   - DECISIVE cap-test (-DGDN_BR_FWD_CAP, cap inner work to 1/16, timing-only): the 4-THREAD full solve
 *     did NOT speed up (142.9K -> 148.3K, within run-to-run noise).  => forward-subst is FULLY HIDDEN on
 *     the real workload: its single-thread load stalls are absorbed by SMT (3 other threads' HVX fill the
 *     idle issue slots) + overlapped with the merge HMX work.  So the inline-asm scalar-prefetch fix
 *     recovers the 3x SINGLE-THREAD but gives ~0% on the real 4-thread solve (and 4-thread wall has ~10%
 *     noise >> any fwdsubst gain).  NOT WORTH the intricate variable-trip asm -> kept the clean form. */
static void gdn_diag_fwdsubst(const int32_t *__restrict Afx, int16_t *__restrict Tc16, int16_t ei16) {
    for (int i = 0; i < BL; ++i) {
        HVX_VectorPair acc = Q6_W_vzero();
        for (int k = 0; k < i; ++k)
            acc = Q6_Ww_vmpyacc_WwVhRh(acc, ((const HVX_Vector *)(Tc16 + k * BL))[0], Afx[i * BL + k]);
        ((HVX_Vector *)(Tc16 + i * BL))[0] =
            Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(acc), Q6_V_lo_W(acc), GDN_BR_F);  /* even/odd -> natural int16 */
        Tc16[i * BL + i] += ei16;
    }
}

/* solve one 64x64 diagonal block: T = inv(I - A_block), int16-code forward subst, leave int32 CODES
 * (scale GDN_BR_TI) in Tcout — NO float dequant (the M_op float roundtrip was the diagonal bottleneck). */
static void gdn_solve_diag64(gdn_scr_t *sc, const uint16_t *Au, int row_stride, int zpA, int M, int S,
                             int32_t *Tcout, int32_t *mx_out = nullptr) {
    /* int16-PACKED forward subst (ported from GdnSolveOp C>64 path): T held int16 in a tight VTCM-friendly
     * buffer, ONE Q6_Ww_vmpyacc_WwVhRh does all 64 cols/row (2x the int32 vmpyiacc + half the memory
     * traffic), the int32 even/odd acc narrows+re-interleaves in one vasr_VwVwR_rnd_sat.  Output widened to
     * int32 codes once at the end (the merge path consumes int32). ~10x faster than the int32 BSS version. */
    int32_t *Afx = sc->Afx;
    gdn_fold_block_hvx(Au, row_stride, Afx, zpA, M, S);   /* Afx[i*BL+k] = code, low 16b is the int16 mul scalar */
    int16_t *Tc16 = sc->Tc16;
    const int16_t ei16 = (int16_t)(int)(1.0f / GDN_BR_TI + 0.5f);
    gdn_diag_fwdsubst(Afx, Tc16, ei16);
    /* widen the int16 T codes -> int32 codes for the merge path.  vsxt de-interleaves (lo=even halfwords,
     * hi=odd); a word-granularity vshuff re-interleaves to natural [c0,c1,...] order (2 vecs/row). */
    HVX_Vector vmax = Q6_V_vzero();
    for (int i = 0; i < BL; ++i) {
        HVX_VectorPair w = Q6_Ww_vsxt_Vh(((const HVX_Vector *)(Tc16 + i * BL))[0]);
        HVX_VectorPair nat = Q6_W_vshuff_VVR(Q6_V_hi_W(w), Q6_V_lo_W(w), -4);
        HVX_Vector lo = Q6_V_lo_W(nat), hi = Q6_V_hi_W(nat);
        ((HVX_Vector *)(Tcout + i * BL))[0] = lo;
        ((HVX_Vector *)(Tcout + i * BL))[1] = hi;
        if (mx_out) { vmax = Q6_Vw_vmax_VwVw(vmax, Q6_Vw_vabs_Vw(lo));
                      vmax = Q6_Vw_vmax_VwVw(vmax, Q6_Vw_vabs_Vw(hi)); }
    }
    if (mx_out) {   /* maxabs of the diag T codes so the diag act/wt quant skips its maxabs pass */
        vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4 * 16));
        vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4 * 8));
        vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4 * 4));
        vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4 * 2));
        vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4 * 1));
        int32_t lanes[32] __attribute__((aligned(128)));
        *(HVX_Vector *)lanes = vmax;
        *mx_out = lanes[0];
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
static float gdn_quant_i8_from_codes(gdn_scr_t *sc, const int32_t *codes, float scale_in, int8_t *out,
                                    int32_t mx) {
    float sQ;
#if defined(GDN_BR_STATIC_FULL)
    if (g_ops_i8 > 0.f) sQ = g_ops_i8; else
#endif
    { if (mx < 0) mx = gdn_maxabs_codes(codes); float maxval = (float)mx * scale_in;
      sQ = (maxval > 0.0f) ? (maxval / 127.0f) : 1e-12f; }
    /* code_q = round(code*g), g=scale_in/sQ.  Adaptive Q keeps Mg<2^15 so code(<2^15)*Mg < 2^30. */
    float g = scale_in / sQ;
    int Q = 14;
    while (Q > 1  && g * (float)(1 << Q) >= 32768.0f) --Q;
    while (Q < 28 && g * (float)(1 << (Q + 1)) < 16384.0f) ++Q;
    int32_t Mg = (int32_t)(g * (float)(1 << Q) + 0.5f);
    const HVX_Vector vrnd = Q6_V_vsplat_R(1 << (Q - 1));
    const HVX_Vector vlim = Q6_V_vsplat_R(127), vnlim = Q6_V_vsplat_R(-127);
    const HVX_Vector *p = (const HVX_Vector *)codes;
    int32_t *qbuf = sc->qbuf;
    HVX_Vector *qp = (HVX_Vector *)qbuf;
    const int Mrep = (Mg & 0xFFFF) * 0x10001;   /* Mg fits int16; replicate into both halfwords for vmpyi_VwRh */
    for (int b = 0; b < (BL * BL) / 32; ++b) {
        HVX_Vector prod = Q6_Vw_vmpyi_VwRh(p[b], Mrep);
        HVX_Vector q = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(prod, vrnd), Q);
        q = Q6_Vw_vmin_VwVw(q, vlim); q = Q6_Vw_vmax_VwVw(q, vnlim);
        qp[b] = q;
    }
    /* vectorized int32 -> int8 narrow (natural order): pack 4 i32 vecs -> 2 i16 -> 1 i8 per 128-lane. */
    HVX_Vector *op = (HVX_Vector *)out;
    for (int v = 0; v < (BL * BL) / 128; ++v) {
        HVX_Vector h0 = Q6_Vh_vpack_VwVw_sat(qp[v*4+1], qp[v*4+0]);
        HVX_Vector h1 = Q6_Vh_vpack_VwVw_sat(qp[v*4+3], qp[v*4+2]);
        op[v] = Q6_Vb_vpack_VhVh_sat(h1, h0);
    }
    return sQ;
}

/* quantize 64x64 int32 codes into u8 activation (zp128) at symmetric scale sQ.  Returns sQ. */
static float gdn_quant_u8_from_codes(gdn_scr_t *sc, const int32_t *codes, float scale_in, uint8_t *out,
                                    int32_t mx) {
    float sQ;
#if defined(GDN_BR_STATIC_FULL)
    if (g_ops_u8 > 0.f) sQ = g_ops_u8; else
#endif
    { if (mx < 0) mx = gdn_maxabs_codes(codes); float maxval = (float)mx * scale_in;
      sQ = (maxval > 0.0f) ? (maxval / 127.0f) : 1e-12f; }
    float g = scale_in / sQ;
    int Q = 14;
    while (Q > 1  && g * (float)(1 << Q) >= 32768.0f) --Q;
    while (Q < 28 && g * (float)(1 << (Q + 1)) < 16384.0f) ++Q;
    int32_t Mg = (int32_t)(g * (float)(1 << Q) + 0.5f);
    const HVX_Vector vrnd = Q6_V_vsplat_R(1 << (Q - 1));
    const HVX_Vector vlim = Q6_V_vsplat_R(127), vnlim = Q6_V_vsplat_R(-127), v128 = Q6_V_vsplat_R(128);
    const HVX_Vector *p = (const HVX_Vector *)codes;
    int32_t *qbuf = sc->qbuf;
    HVX_Vector *qp = (HVX_Vector *)qbuf;
    const int Mrep = (Mg & 0xFFFF) * 0x10001;
    for (int b = 0; b < (BL * BL) / 32; ++b) {
        HVX_Vector prod = Q6_Vw_vmpyi_VwRh(p[b], Mrep);
        HVX_Vector q = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(prod, vrnd), Q);
        q = Q6_Vw_vmin_VwVw(q, vlim); q = Q6_Vw_vmax_VwVw(q, vnlim);
        qp[b] = Q6_Vw_vadd_VwVw(q, v128);
    }
    /* vectorized int32 -> u8 narrow (natural order): values in [1,255]. */
    HVX_Vector *op = (HVX_Vector *)out;
    for (int v = 0; v < (BL * BL) / 128; ++v) {
        HVX_Vector h0 = Q6_Vh_vpack_VwVw_sat(qp[v*4+1], qp[v*4+0]);
        HVX_Vector h1 = Q6_Vh_vpack_VwVw_sat(qp[v*4+3], qp[v*4+2]);
        op[v] = Q6_Vub_vpack_VhVh_sat(h1, h0);
    }
    return sQ;
}

#if defined(GDN_BR_STATIC_FULL)
/* FUSED fold+quant for the A operand (static only): ONE pass u16 -> u8, NO int32 Aoff round-trip and NO
 * separate quant loop.  Produces the IDENTICAL u8 output of gdn_fold_block_raw + gdn_quant_u8_from_codes
 * (fold scale 2^-F, then static quant g=(2^-F)/sQ).  Two sequential vmpyi in-register (fold M, then quant
 * Mg) avoid the M*Mg overflow while skipping the int32 memory round-trip. */
static void gdn_fold_quant_u8(gdn_scr_t *sc, const uint16_t *Au, int row_stride, uint8_t *out,
                              int zpA, int M, int S, float sQ) {
    float g = (float)(1.0 / (1 << GDN_BR_F)) / sQ;
    int Q = 14;
    while (Q > 1  && g * (float)(1 << Q) >= 32768.0f) --Q;
    while (Q < 28 && g * (float)(1 << (Q + 1)) < 16384.0f) ++Q;
    int32_t Mg = (int32_t)(g * (float)(1 << Q) + 0.5f);
    const int Mrep = (M & 0xFFFF) * 0x10001, MgRep = (Mg & 0xFFFF) * 0x10001;
    const HVX_Vector vzp = Q6_V_vsplat_R(zpA), vrndS = Q6_V_vsplat_R(1 << (S - 1)), vrndQ = Q6_V_vsplat_R(1 << (Q - 1));
    const HVX_Vector vlim = Q6_V_vsplat_R(127), vnlim = Q6_V_vsplat_R(-127), v128 = Q6_V_vsplat_R(128);
    HVX_Vector *qp = (HVX_Vector *)sc->qbuf;
    for (int r = 0; r < BL; ++r) {
        const HVX_UVector *Av = (const HVX_UVector *)(Au + r * row_stride);
        HVX_VectorPair w = Q6_Wuw_vzxt_Vuh(Av[0]);
        HVX_Vector c0 = Q6_Vw_vsub_VwVw(Q6_V_lo_W(w), vzp), c1 = Q6_Vw_vsub_VwVw(Q6_V_hi_W(w), vzp);
        HVX_Vector i0 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(c0, Mrep), vrndS), S);  /* fold */
        HVX_Vector i1 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(c1, Mrep), vrndS), S);
        HVX_VectorPair s = Q6_W_vshuff_VVR(i1, i0, -4);   /* even/odd -> natural cols 0..63 */
        HVX_Vector a = Q6_V_lo_W(s), b = Q6_V_hi_W(s);
        HVX_Vector qa = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(a, MgRep), vrndQ), Q);    /* quant */
        qa = Q6_Vw_vmin_VwVw(qa, vlim); qa = Q6_Vw_vmax_VwVw(qa, vnlim); qp[r * 2] = Q6_Vw_vadd_VwVw(qa, v128);
        HVX_Vector qb = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(b, MgRep), vrndQ), Q);
        qb = Q6_Vw_vmin_VwVw(qb, vlim); qb = Q6_Vw_vmax_VwVw(qb, vnlim); qp[r * 2 + 1] = Q6_Vw_vadd_VwVw(qb, v128);
    }
    HVX_Vector *op = (HVX_Vector *)out;                   /* int32 -> u8 narrow (same as gdn_quant_u8) */
    for (int v = 0; v < (BL * BL) / 128; ++v) {
        HVX_Vector h0 = Q6_Vh_vpack_VwVw_sat(qp[v * 4 + 1], qp[v * 4 + 0]);
        HVX_Vector h1 = Q6_Vh_vpack_VwVw_sat(qp[v * 4 + 3], qp[v * 4 + 2]);
        op[v] = Q6_Vub_vpack_VhVh_sat(h1, h0);
    }
}
#endif

/* EXACT scale estimate: max|P_int| where P_int[i,c] = sum_k (act_u8[i,k]-128)*wt_i8[k,c].  This is the
 * exact int product the HMX kernel computes, so the derived scale is exact (not a loose bound) AND HVX
 * fast.  act is u8 (zp128), wt is i8, both 64x64 natural.  Returns max|P_int| (int32). */
static int32_t gdn_pint_maxabs(gdn_scr_t *sc, const uint8_t *act_u8, const int8_t *wt_i8) {
    /* pre-widen wt rows to int32 [64][64] in sc->Tc (reused scratch). */
    int32_t *W = sc->Tc;
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
    /* PRE-SHIFT: code*Mg overflows int32 for code>~2^15; the int16-HVX matmul output reaches 2^22.  Shift
     * codes >> psh and compensate g so the fixed-point stays in range (no-op for the small-code HMX path). */
#if defined(GDN_BR_STATIC_FULL)
    int psh = 0;   /* STATIC: T codes bounded < 2^15 (off-diag @sTw <=127, diag @GDN_BR_TI <32767) -> psh==0,
                    * skip the gdn_maxabs_codes scan entirely. */
#else
    int psh = 0; { int32_t mx = gdn_maxabs_codes(codes); int32_t m = mx; while (m >= (1 << 15)) { m >>= 1; ++psh; } }
#endif
    float g = (scale_in / sT) * (float)(1 << psh);
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
            HVX_Vector c = (psh > 0) ? Q6_Vw_vasr_VwR(cp[b], psh) : cp[b];
            HVX_Vector prod = Q6_Vw_vmpyi_VwRh(c, Mrep);
            HVX_Vector qq = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(prod, vrnd), Q);
            qq = Q6_Vw_vmin_VwVw(qq, vlim); qq = Q6_Vw_vmax_VwVw(qq, vnlim);
            qp[b] = Q6_Vw_vadd_VwVw(qq, vzpT);
        }
        uint16_t *dst = Th + (roff + r) * row_stride + coff;
        *(HVX_UVector *)dst = Q6_Vuh_vpack_VwVw_sat(qp[1], qp[0]);
    }
}

/* ---- crouton8 activation packer (C transcription of gdn_hmx_matmul_sim.pack_act_crouton8, 64^3) ----
 * Each 32-byte output run is a contiguous 32-col slice (cols [k_base,k_base+32)) of one source row; 4
 * consecutive runs are 4 consecutive source rows.  PURE HVX: build each 128-byte VTCM output vector from
 * 4 unaligned source-row loads via ror+mask (NO scalar VTCM stores — the old 256-scalar-uint64-store path
 * was the dominant glue cost, ~40K cyc/head). */
static void gdn_pack_act_crouton8(const uint8_t *act_mk, uint8_t *out_buf) {
    /* m = [0xFF x32, 0 x96]; m1/m2/m3 = same 32-wide window rotated into bytes [32,64)/[64,96)/[96,128). */
    const HVX_Vector m  = Q6_V_valign_VVR(Q6_V_vzero(), Q6_Vb_vsplat_R(-1), 96);
    const HVX_Vector m1 = Q6_V_vror_VR(m, 96);
    const HVX_Vector m2 = Q6_V_vror_VR(m, 64);
    const HVX_Vector m3 = Q6_V_vror_VR(m, 32);
    HVX_Vector *dst = (HVX_Vector *)out_buf;
    int v = 0;
    for (int kt = 0; kt < 2; ++kt) {
        int k_base = kt * 32;
        const uint8_t *b = act_mk + k_base;
        for (int local = 0; local < 16; ++local) {
            /* run o0=local*4: row0 = ((o0/8)&1)*32 + (o0/16)*8 + (o0%8); next 3 runs are row0+1..+3. */
            int o0 = local * 4;
            int row0 = ((o0 / 8) & 1) * 32 + (o0 / 16) * 8 + (o0 % 8);
            HVX_Vector v0 = *(const HVX_UVector *)(b + (row0 + 0) * 64);
            HVX_Vector v1 = *(const HVX_UVector *)(b + (row0 + 1) * 64);
            HVX_Vector v2 = *(const HVX_UVector *)(b + (row0 + 2) * 64);
            HVX_Vector v3 = *(const HVX_UVector *)(b + (row0 + 3) * 64);
            HVX_Vector out = Q6_V_vand_VV(v0, m);
            out = Q6_V_vor_VV(out, Q6_V_vand_VV(Q6_V_vror_VR(v1, 96), m1));
            out = Q6_V_vor_VV(out, Q6_V_vand_VV(Q6_V_vror_VR(v2, 64), m2));
            out = Q6_V_vor_VV(out, Q6_V_vand_VV(Q6_V_vror_VR(v3, 32), m3));
            dst[v++] = out;
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
                /* 4 source rows (32 bytes each); want out word c = (s0[c],s1[c],s2[c],s3[c]).
                 * HVX: byte-interleave (s0,s1)->p01, (s2,s3)->p23 (halfword c = (s0c,s1c)/(s2c,s3c)),
                 * then halfword-interleave p01,p23 -> word c = (s0c,s1c,s2c,s3c).  Each row is 32B so the
                 * useful data is in the low 64 bytes of the shuffle results. */
                const int8_t *s0 = w_kn + (k_base + 4*r4 + 0) * 64 + n_base;
                const int8_t *s1 = w_kn + (k_base + 4*r4 + 1) * 64 + n_base;
                const int8_t *s2 = w_kn + (k_base + 4*r4 + 2) * 64 + n_base;
                const int8_t *s3 = w_kn + (k_base + 4*r4 + 3) * 64 + n_base;
                HVX_Vector v0 = *(const HVX_UVector *)s0, v1 = *(const HVX_UVector *)s1;
                HVX_Vector v2 = *(const HVX_UVector *)s2, v3 = *(const HVX_UVector *)s3;
                /* R=1: interleave bytes (shuff with element-size 1 byte). lo64 = (v0[0],v1[0],v0[1],v1[1]..) */
                HVX_Vector p01 = Q6_V_lo_W(Q6_W_vshuff_VVR(v1, v0, -1));
                HVX_Vector p23 = Q6_V_lo_W(Q6_W_vshuff_VVR(v3, v2, -1));
                /* R=-2: interleave halfwords. lo128 = word c = (s0c,s1c,s2c,s3c) for c=0..31. */
                HVX_Vector w = Q6_V_lo_W(Q6_W_vshuff_VVR(p23, p01, -2));
                *(HVX_UVector *)(tile + r4 * 128) = w;
            }
            out += 1024;
        }
    }
}

/* effective[n] = -128*sum_k wt[k,n]; 64-wide.  HVX column-sum over int8 rows using a byte dot-product
 * with a +1 constant vector: Q6_Vw_vrmpyacc_VwVbVb_b? Use vmpa? Simplest reliable HVX: sum int8 rows by
 * widening with vsxt + the proper de-interleave is fiddly; the scalar form is correct and bounded (4096
 * ops/merge, ~small vs pint).  Kept scalar for correctness; not the bottleneck. */
static void gdn_effective(const int8_t *w_kn, int32_t *effective, int *colabsmax = nullptr) {
    /* HVX: sign-extend each 64-byte row to int16 (vsxt deals even/odd cols into lo/hi 32 lanes),
     * accumulate even-col sums and odd-col sums separately (64 rows * 127 < 2^15, no overflow), then
     * vshuff to restore natural col order.  *(-128) at the end.
     * colabsmax (GDNSolveHVXMixHMX lever #1c): if requested, ALSO accumulate the per-col ABSOLUTE sum
     * Σ_k|wt[k,c]| and return its max over cols — a Holder bound for max|P| (with max|act code|<=128) that
     * lets gdn_merge_packed predict the PASS-1 gain and drop PASS-2. ~free (same row sweep). */
    HVX_Vector acc_e = Q6_V_vzero(), acc_o = Q6_V_vzero();   /* int16 partials: even cols / odd cols */
    HVX_Vector abs_e = Q6_V_vzero(), abs_o = Q6_V_vzero();   /* int16 ABS-sum partials (for colabsmax) */
    for (int k = 0; k < BL; ++k) {
        HVX_Vector row = *(const HVX_UVector *)(w_kn + k * BL);   /* 64 useful bytes in low half */
        HVX_VectorPair w16 = Q6_Wh_vsxt_Vb(row);                  /* lo=even-byte sexts, hi=odd-byte sexts */
        HVX_Vector lo = Q6_V_lo_W(w16), hi = Q6_V_hi_W(w16);
        acc_e = Q6_Vh_vadd_VhVh(acc_e, lo);
        acc_o = Q6_Vh_vadd_VhVh(acc_o, hi);
        if (colabsmax) {
            abs_e = Q6_Vh_vadd_VhVh(abs_e, Q6_Vh_vabs_Vh(lo));
            abs_o = Q6_Vh_vadd_VhVh(abs_o, Q6_Vh_vabs_Vh(hi));
        }
    }
    /* interleave even/odd halfwords back to natural [c0,c1,c2,...]; low 64 lanes hold cols 0..63. */
    HVX_VectorPair nat = Q6_W_vshuff_VVR(acc_o, acc_e, -2);
    int16_t cols[64] __attribute__((aligned(128)));
    *(HVX_Vector *)cols = Q6_V_lo_W(nat);
    for (int n = 0; n < 64; ++n) effective[n] = -128 * (int)cols[n];
    if (colabsmax) {
        HVX_VectorPair anat = Q6_W_vshuff_VVR(abs_o, abs_e, -2);
        int16_t acols[64] __attribute__((aligned(128)));
        *(HVX_Vector *)acols = Q6_V_lo_W(anat);
        int mx = 0; for (int n = 0; n < 64; ++n) if (acols[n] > mx) mx = acols[n];
        *colabsmax = mx;
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

/* per-N32 control word = (baseline_u16<<16) | f16_bits(scale_f16); pack folded bias for 64^3 (2 tiles).
 * rdelta is added to every effective (accumulator) entry: the v73deep drain computes
 * out = FLOOR((P_int+eff)*scale_f16/512), so passing rdelta = round(256/scale_f16) injects +0.5 LSB and
 * turns the FLOOR into round-to-nearest.  ONLY the output pass should round (the loose-gain measurement
 * passes have a tiny scale_f16 -> huge 256/scale_f16 -> pass rdelta=0 there). */
static void gdn_pack_bias(const int32_t *effective, float scale_f16, int baseline_u16, int32_t *bias,
                          int rdelta) {
    /* GDNSolveHVXMixHMX lever #1 (vectorized): the bias tile is exactly 4 HVX vectors —
     * [ctrl×32 | eff[0:32]+rdelta | ctrl×32 | eff[32:64]+rdelta] — so 4 vector stores replace the old
     * 128 SCALAR VTCM writes (the documented pathology that dominated the multi-pass cost: 4190->1720
     * cyc/matmul, end-to-end 1.29->2.06x). NUMERICALLY IDENTICAL to the old scalar loop (oc unchanged). */
    uint32_t ctrl = ((uint32_t)(baseline_u16 & 0xFFFF) << 16) | (uint32_t)gdn_f16_bits(scale_f16);
    HVX_Vector vctrl = Q6_V_vsplat_R((int)ctrl);
    HVX_Vector vrd = Q6_V_vsplat_R(rdelta);
    HVX_Vector e0 = *(const HVX_UVector *)(effective + 0);     /* eff[0:32]  (unaligned-safe) */
    HVX_Vector e1 = *(const HVX_UVector *)(effective + 32);    /* eff[32:64] */
    HVX_Vector *b = (HVX_Vector *)bias;                        /* vt->bias: VTCM, 128-aligned */
    b[0] = vctrl;
    b[1] = Q6_Vw_vadd_VwVw(e0, vrd);
    b[2] = vctrl;
    b[3] = Q6_Vw_vadd_VwVw(e1, vrd);
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
static void gdn_depack_out_fast(gdn_scr_t *sc, const uint8_t *surf, int base, int8_t *out_codes) {
    /* 1) subtract base over the whole 4096-byte surface with HVX (32 vecs). surf is VTCM-aligned. */
    const HVX_Vector vb = Q6_Vb_vsplat_R(base);
    const HVX_Vector *sp = (const HVX_Vector *)surf;
    HVX_Vector *dp = (HVX_Vector *)sc->surf_sub;
    for (int i = 0; i < (BL * BL) / 128; ++i) dp[i] = Q6_Vb_vsub_VbVb(sp[i], vb);
    /* 2) PURE-HVX de-crouton (was 256 scalar uint64 copies, the depack twin of the actpack hotspot):
     * a 128B crouton vector = 4 consecutive rows' 32-col slices; the nt=0/nt=1 vectors for the same 4
     * rows hold cols [0,32)/[32,64).  Interleave them via ror+mask into two 128B vectors = 4 full natural
     * rows, stored aligned (row0 always even -> row0*64 is 128-aligned). */
    const HVX_Vector m  = Q6_V_valign_VVR(Q6_V_vzero(), Q6_Vb_vsplat_R(-1), 96);
    const HVX_Vector m1 = Q6_V_vror_VR(m, 96), m2 = Q6_V_vror_VR(m, 64), m3 = Q6_V_vror_VR(m, 32);
    const uint8_t *s0 = sc->surf_sub, *s1 = sc->surf_sub + 2048;   /* nt=0 / nt=1 crouton halves */
    for (int local = 0; local < 16; ++local) {
        HVX_Vector v0 = *(const HVX_Vector *)(s0 + local * 128);   /* cols [0,32) of 4 rows */
        HVX_Vector v1 = *(const HVX_Vector *)(s1 + local * 128);   /* cols [32,64) of 4 rows */
        int o0 = local * 4;
        int row0 = ((o0 / 8) & 1) * 32 + (o0 / 16) * 8 + (o0 % 8);
        /* A = [row0 full | row1 full] = (v0[0:32],v1[0:32],v0[32:64],v1[32:64]) */
        HVX_Vector A = Q6_V_vand_VV(v0, m);
        A = Q6_V_vor_VV(A, Q6_V_vand_VV(Q6_V_vror_VR(v1, 96), m1));
        A = Q6_V_vor_VV(A, Q6_V_vand_VV(Q6_V_vror_VR(v0, 96), m2));
        A = Q6_V_vor_VV(A, Q6_V_vand_VV(Q6_V_vror_VR(v1, 64), m3));
        /* B = [row2 full | row3 full] = (v0[64:96],v1[64:96],v0[96:128],v1[96:128]) */
        HVX_Vector B = Q6_V_vand_VV(Q6_V_vror_VR(v0, 64), m);
        B = Q6_V_vor_VV(B, Q6_V_vand_VV(Q6_V_vror_VR(v1, 32), m1));
        B = Q6_V_vor_VV(B, Q6_V_vand_VV(Q6_V_vror_VR(v0, 32), m2));
        B = Q6_V_vor_VV(B, Q6_V_vand_VV(v1, m3));
        *(HVX_Vector *)(out_codes + (size_t)row0 * 64)       = A;
        *(HVX_Vector *)(out_codes + (size_t)(row0 + 2) * 64) = B;
    }
}


#if defined(GDN_BR_PROBE_CYCLES)
uint64_t g_c_hmxpack = 0, g_c_hmxkern = 0, g_c_hmxdepack = 0, g_c_quant = 0, g_c_pint = 0;
#endif
#if 0  /* retired single-pass merge (replaced by the 2-pass pack_only/run_only path) */
static void gdn_hmx_merge(const gdn_vtcm_t *vt, const uint8_t *act_u8, const int8_t *wt_i8,
                          float scale_f16, int baseline_u16, int8_t *out_codes) {
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
}
#endif  /* retired single-pass merge */

/* ---- 2-pass scale estimation: pack once, run HMX with a provisional (loose-bound) gain, read the
 * actual max |product| from the u8 surface (free), then re-run at the tight gain.  Replaces the 685K
 * HVX pint 64^3 matmul with ~2 free HMX passes. ---- */

/* loose upper bound on max|P_int| for P = (act_u8-128) @ wt_i8 (both 64x64 natural), guaranteed >= the
 * true max so the provisional pass cannot saturate: K * max|act-128| * max|wt|  (K=64).  Fully HVX
 * (two int8 maxabs reductions); validated near-exact downstream (host probe: 1.29e-2 vs exact 1.20e-2). */
static int32_t gdn_pint_loosebound(const uint8_t *act_u8, const int8_t *wt_i8) {
    HVX_Vector vwmax = Q6_V_vzero(), vamax = Q6_V_vzero();
    const HVX_Vector v128b = Q6_Vb_vsplat_R(128);
    for (int b = 0; b < (BL * BL) / 128; ++b) {
        HVX_Vector w = ((const HVX_Vector *)wt_i8)[b];
        HVX_VectorPair w16 = Q6_Wh_vsxt_Vb(w);
        vwmax = Q6_Vh_vmax_VhVh(vwmax, Q6_Vh_vabs_Vh(Q6_V_lo_W(w16)));
        vwmax = Q6_Vh_vmax_VhVh(vwmax, Q6_Vh_vabs_Vh(Q6_V_hi_W(w16)));
        /* act centered: (u8-128) as signed byte (wraps correctly mod 256), then abs via int16 widen. */
        HVX_Vector a = Q6_Vb_vsub_VbVb(((const HVX_Vector *)act_u8)[b], v128b);
        HVX_VectorPair a16 = Q6_Wh_vsxt_Vb(a);
        vamax = Q6_Vh_vmax_VhVh(vamax, Q6_Vh_vabs_Vh(Q6_V_lo_W(a16)));
        vamax = Q6_Vh_vmax_VhVh(vamax, Q6_Vh_vabs_Vh(Q6_V_hi_W(a16)));
    }
    int16_t wl[64] __attribute__((aligned(128))), al[64] __attribute__((aligned(128)));
    *(HVX_Vector *)wl = vwmax; *(HVX_Vector *)al = vamax;
    int wmax = 0, amax = 0;
    for (int i = 0; i < 64; ++i) { if (wl[i] > wmax) wmax = wl[i]; if (al[i] > amax) amax = al[i]; }
    int32_t b = (int32_t)BL * (int32_t)amax * (int32_t)wmax;
    return b > 0 ? b : 1;
}

/* max |u8_code - base| over the depacked product surface (the crouton8 out surface), scanning the raw
 * 4096-byte VTCM surface (order-independent for a max). */
static int gdn_surf_maxabs(const uint8_t *surf, int base) {
    const HVX_Vector vb = Q6_Vb_vsplat_R(base);
    HVX_Vector vmax = Q6_V_vzero();
    const HVX_Vector *sp = (const HVX_Vector *)surf;
    for (int i = 0; i < (BL * BL) / 128; ++i) {
        HVX_Vector d = Q6_Vb_vsub_VbVb(sp[i], vb);          /* signed code */
        HVX_VectorPair d16 = Q6_Wh_vsxt_Vb(d);
        vmax = Q6_Vh_vmax_VhVh(vmax, Q6_Vh_vabs_Vh(Q6_V_lo_W(d16)));
        vmax = Q6_Vh_vmax_VhVh(vmax, Q6_Vh_vabs_Vh(Q6_V_hi_W(d16)));
    }
    int16_t l[64] __attribute__((aligned(128))); *(HVX_Vector *)l = vmax;
    int m = 0; for (int i = 0; i < 64; ++i) if (l[i] > m) m = l[i];
    return m;
}

/* pack act/wt/effective once (no bias control word). */
#if defined(GDN_BR_PROBE_CYCLES)
uint64_t g_c_eff = 0, g_c_actpack = 0, g_c_wtpack = 0;
uint64_t g_c_diag = 0;
uint64_t g_c_fold = 0, g_c_acc = 0, g_c_widen = 0, g_c_requant = 0, g_c_zero = 0;
#endif
static void gdn_hmx_pack_only(const gdn_vtcm_t *vt, const uint8_t *act_u8, const int8_t *wt_i8, int32_t *eff) {
#if defined(GDN_BR_PROBE_CYCLES)
    uint64_t e0; asm volatile("%0 = C15:14" : "=r"(e0));
#endif
    gdn_effective(wt_i8, eff);
#if defined(GDN_BR_PROBE_CYCLES)
    uint64_t e1; asm volatile("%0 = C15:14" : "=r"(e1)); g_c_eff += e1 - e0;
#endif
    gdn_pack_act_crouton8(act_u8, vt->act);
#if defined(GDN_BR_PROBE_CYCLES)
    uint64_t e2; asm volatile("%0 = C15:14" : "=r"(e2)); g_c_actpack += e2 - e1;
#endif
    gdn_pack_w8_kmajor(wt_i8, vt->wt);
#if defined(GDN_BR_PROBE_CYCLES)
    uint64_t e3; asm volatile("%0 = C15:14" : "=r"(e3)); g_c_wtpack += e3 - e2;
#endif
    vt->acttab[0] = (int32_t)(uintptr_t)(vt->act + 0);
    vt->acttab[1] = (int32_t)(uintptr_t)(vt->act + 64 * 32);
    vt->outtab[0] = (int32_t)(uintptr_t)(vt->out + 0);
    vt->outtab[1] = (int32_t)(uintptr_t)(vt->out + 64 * 32);
}

/* run the kernel with a freshly-packed bias (gain+baseline); reads the k-major weight from wt_kmajor
 * and the activation tiles from vt->acttab (set by the caller); leaves the u8 product surface in vt->out. */
static void gdn_hmx_run_only(const gdn_vtcm_t *vt, const int8_t *wt_kmajor, const int32_t *eff,
                            float scale_f16, int baseline_u16, int round_out) {
    /* round_out: inject +0.5 LSB so the FLOOR drain rounds-to-nearest (output pass only). */
    int rdelta = round_out ? (int)(256.0f / scale_f16 + 0.5f) : 0;
    gdn_pack_bias(eff, scale_f16, baseline_u16, vt->bias, rdelta);
    { HVX_Vector z = Q6_V_vzero(); HVX_Vector *op = (HVX_Vector *)vt->out;
      for (int i = 0; i < (BL * BL) / 128; ++i) op[i] = z; }
    uint32_t extra_param[2] __attribute__((aligned(16))) = {1u, 0u};
    uint32_t mask_buf[16] __attribute__((aligned(16)));
    for (int i = 0; i < 16; ++i) mask_buf[i] = GDN_BR_MASK_WORDS[i];
    hmx_conv_out_desc_t out_desc __attribute__((aligned(64))) = {
        vt->outtab, GDN_BR_OUT_TABLE_STRIDE, GDN_BR_OUT_Y_STRIDE,
        GDN_BR_N_TILES_POW2, GDN_BR_M_TOTAL_MINUS_STEP, GDN_BR_K_TOTAL_BYTES };
    hmx_conv_act_desc_t act_desc __attribute__((aligned(64))) = {
        vt->acttab, GDN_BR_N_ACT_PAIRS, GDN_BR_ACT_Y_STRIDE };
    /* HMX critical section ONLY around the mxmem kernel (default no-op; the bare-metal HAP defines these to
     * HAP_compute_res_hmx_lock/unlock so the HVX glue runs unlocked -> threads parallelize, only mxmem serializes). */
    GDN_BR_HMX_ENTER();
    our_v73deep_kernel(&out_desc, &act_desc, (const uint8_t *)wt_kmajor, (const uint8_t *)vt->bias,
                       (const hmx_conv_mask_desc_t *)mask_buf, extra_param);
    GDN_BR_HMX_EXIT();
}

/* ---- operand-reuse cache getters (Task 2): quantize+PACK each distinct operand ONCE per head ----
 * The natural u8/i8 codes are transient (just feed the packer); the cache holds the HMX-ready packed
 * surface (crouton act / k-major wt) in VTCM + the operand scale (+ effective bias for weights). */

/* A_ik activation: fold + quant u8 + crouton-pack into the VTCM act cache; key gdn_blk_index(i,k). */
static const uint8_t *gdn_get_act_A(gdn_scr_t *sc, const gdn_vtcm_t *vt, const uint16_t *Ah, int i, int k,
                                    int C_, int zpA, int M, int S, float *sa_out) {
    int key = gdn_blk_index(i, k);
    uint8_t *cr = vt->acache + (size_t)key * 0x1000;
    if (!sc->vAa[key]) {
#if defined(GDN_BR_PROBE_CYCLES)
        uint64_t f0; asm volatile("%0 = C15:14" : "=r"(f0));
#endif
#if defined(GDN_BR_STATIC_FULL)
        /* FUSED fold+quant: one pass u16 -> u8, no int32 Aoff round-trip (static fixed scale). */
        gdn_fold_quant_u8(sc, Ah + (size_t)i * BL * C_ + k * BL, C_, sc->actbuf, zpA, M, S, GDN_OPS_sAa);
        sc->sAa[key] = GDN_OPS_sAa;
#if defined(GDN_BR_PROBE_CYCLES)
        { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_quant += q - f0; }
        uint64_t p0; asm volatile("%0 = C15:14" : "=r"(p0));
#endif
#else
        int32_t mxA;
        gdn_fold_block_raw(Ah + (size_t)i * BL * C_ + k * BL, C_, sc->Aoff, zpA, M, S, &mxA);
#if defined(GDN_BR_PROBE_CYCLES)
        { uint64_t f; asm volatile("%0 = C15:14" : "=r"(f)); g_c_fold += f - f0; }
        uint64_t q0; asm volatile("%0 = C15:14" : "=r"(q0));
#endif
        sc->sAa[key] = gdn_quant_u8_from_codes(sc, sc->Aoff, (float)(1.0 / (1 << GDN_BR_F)), sc->actbuf, mxA);
#if defined(GDN_BR_PROBE_CYCLES)
        { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_quant += q - q0; }
        uint64_t p0; asm volatile("%0 = C15:14" : "=r"(p0));
#endif
#endif
        gdn_pack_act_crouton8(sc->actbuf, cr);
#if defined(GDN_BR_PROBE_CYCLES)
        { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_actpack += q - p0; }
#endif
        sc->vAa[key] = 1;
    }
    *sa_out = sc->sAa[key];
    return cr;
}

/* T_ii diagonal block as final-merge activation: quant u8 + crouton-pack; cached by diagonal index i. */
static const uint8_t *gdn_get_act_Tdiag(gdn_scr_t *sc, const gdn_vtcm_t *vt, int i, float *sa_out) {
    uint8_t *cr = vt->acache + 0xA000 + (size_t)i * 0x1000;
    if (!sc->vTa[i]) {
        int bii = gdn_blk_index(i, i);
#if defined(GDN_BR_PROBE_CYCLES)
        uint64_t q0; asm volatile("%0 = C15:14" : "=r"(q0));
#endif
#if defined(GDN_BR_STATIC_FULL)
        g_ops_u8 = GDN_OPS_sTa;
#endif
        sc->sTa[i] = gdn_quant_u8_from_codes(sc, sc->Tblk[bii], sc->Tscl[bii], sc->actbuf, sc->mxdiag[i]);
#if defined(GDN_BR_PROBE_CYCLES)
        { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_quant += q - q0; }
        uint64_t p0; asm volatile("%0 = C15:14" : "=r"(p0));
#endif
        gdn_pack_act_crouton8(sc->actbuf, cr);
#if defined(GDN_BR_PROBE_CYCLES)
        { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_actpack += q - p0; }
#endif
        sc->vTa[i] = 1;
    }
    *sa_out = sc->sTa[i];
    return cr;
}

/* T_kj block as inner-merge weight: quant i8 + k-major-pack + effective bias; key gdn_blk_index(k,j). */
#if defined(GDN_BR_STATIC_FULL)
/* pure int32 -> int8 narrow (codes already in [-127,127] @sTw, drained by g_force_sP): NO rescale -> the
 * off-diag T_kj weight quant collapses to just a narrow (saves the vmpyi+clip of the rescale). */
static void gdn_narrow_i32_to_i8(const int32_t *codes, int8_t *out) {
    const HVX_Vector *cp = (const HVX_Vector *)codes; HVX_Vector *op = (HVX_Vector *)out;
    for (int v = 0; v < (BL * BL) / 128; ++v) {
        HVX_Vector h0 = Q6_Vh_vpack_VwVw_sat(cp[v * 4 + 1], cp[v * 4 + 0]);
        HVX_Vector h1 = Q6_Vh_vpack_VwVw_sat(cp[v * 4 + 3], cp[v * 4 + 2]);
        op[v] = Q6_Vb_vpack_VhVh_sat(h1, h0);
    }
}
#endif

static const int8_t *gdn_get_wt_T(gdn_scr_t *sc, const gdn_vtcm_t *vt, int k, int j,
                                  float *sw_out, const int32_t **eff_out, int *colabs_out) {
    int key = gdn_blk_index(k, j);
    int8_t *km = vt->wcache + (size_t)key * 0x1000;
    if (!sc->vTw[key]) {
#if defined(GDN_BR_PROBE_CYCLES)
        uint64_t q0; asm volatile("%0 = C15:14" : "=r"(q0));
#endif
        /* diag T block (k==j) has a producer-tracked maxabs; off-diag must compute it. */
#if defined(GDN_BR_STATIC_FULL)
        if (k != j && sc->Tscl[key] == GDN_OPS_sTw) {   /* off-diag T drained @sTw -> pure narrow (no rescale). */
            gdn_narrow_i32_to_i8(sc->Tblk[key], sc->wtbuf); sc->sTw[key] = GDN_OPS_sTw;
        } else {
            g_ops_i8 = GDN_OPS_sTw;
            sc->sTw[key] = gdn_quant_i8_from_codes(sc, sc->Tblk[key], sc->Tscl[key], sc->wtbuf,
                                                   (k == j) ? sc->mxdiag[k] : -1);
        }
#else
        sc->sTw[key] = gdn_quant_i8_from_codes(sc, sc->Tblk[key], sc->Tscl[key], sc->wtbuf,
                                               (k == j) ? sc->mxdiag[k] : -1);
#endif
#if defined(GDN_BR_PROBE_CYCLES)
        { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_quant += q - q0; }
        uint64_t e0; asm volatile("%0 = C15:14" : "=r"(e0));
#endif
#if defined(GDN_BR_STATIC_FULL)
        gdn_effective(sc->wtbuf, sc->effc[key]);              /* static gain: colabs unused -> nullptr skips its reduction */
        sc->colabsc[key] = GDN_OPS_COLABS;
#else
        gdn_effective(sc->wtbuf, sc->effc[key], &sc->colabsc[key]);   /* #1c: cache col-abs-sum max too */
#endif
#if defined(GDN_BR_PROBE_CYCLES)
        { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_eff += q - e0; }
        uint64_t p0; asm volatile("%0 = C15:14" : "=r"(p0));
#endif
        gdn_pack_w8_kmajor(sc->wtbuf, km);
#if defined(GDN_BR_PROBE_CYCLES)
        { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_wtpack += q - p0; }
#endif
        sc->vTw[key] = 1;
    }
    *sw_out = sc->sTw[key];
    *eff_out = sc->effc[key];
    *colabs_out = sc->colabsc[key];
    return km;
}

/* upper bound on max|P_int| for symmetric-quantized operands: both rail to ±127, so the analytic
 * bound K*max|act-128|*max|wt| is the constant 64*127*127 -- no per-merge HVX reduction needed. */
#define GDN_BR_LOOSE_CONST (64 * 127 * 127)

/* gdn_merge_packed = the GDNSolveHVXMixHMX matmul impl (HVX-feed + HMX-matmul); gdn_merge_hvx (below) = the
 * GDNSolveHVX baseline's. (Naming: GDNSolveHVX pure-HVX baseline / GDNSolveHVXMixHMX HVX-feed+HMX-matmul /
 * GDNSolveHMX full-HMX, refuted — see Agent/current/gdn_solve_hvxmixhmx.md top.)
 * INT-ONLY merge from PRE-PACKED VTCM surfaces: act crouton @ sa, wt k-major (+eff) @ sw; 2-pass scale,
 * returns int8 product codes + scale s_out.  Quant AND packing are hoisted to the cache getters. */
/* ============================================================================================
 * GDNSolveHVXMixHMX OPTIMIZATION PLAN (device-measured 2026-06-05; baseline GDNSolveHVX = 135K cyc/head)
 * Full write-up + figures: Agent/current/gdn_solve_hvxmixhmx.md
 *
 * THREE IMPLEMENTATIONS (per-head cycles, shorter = faster; #=~5K):
 *   GDNSolveHVX      (baseline/shipped, pure HVX)  135K  ███████████████████████████  1.00x
 *   GDNSolveHVXMixHMX (THIS path: HVX-feed + HMX)   57K  ███████████                  2.35x  <== BEST
 *   GDNSolveHMX      (full HMX)                   ~600K  ████...(~120#)               0.22x  REFUTED
 *
 * WHY 2.35x — HMX cuts the matmul 5x; diag is a SHARED floor (unchanged):
 *               matmul+prep(fold+quant+pack)   diag(shared HVX)   requant/acc
 *   HVX base:   [████████████████ 97K (72%)]   [██████ 26K]       [██ 12K]   = 135K
 *   MixHMX:     [███ 19.5K (34%)]               [██████ 26K]       [██ 12K]   =  57K
 *                  HMX kernel 215 vs vrmpy ~13K/mm        ^new floor 46%
 *
 * THIS FUNCTION (the multi-pass gain search) is the lever. Optimization ladder (measured cyc/matmul,
 * baremetal -DGDNBM_FEED_MULTIPASS, P=3 producers + 1 HVX-locked consumer, consumer-bound):
 *   scalar-bias 3-pass  (was)                4190   -> end-to-end 1.29x
 *   + VECTORIZE the bias pack (gdn_pack_bias) 1720   -> 2.06x   <== biggest lever, numerically EQUIVALENT
 *   + #1c: drop PASS-2 (norm-predicted gain)  1218   -> 2.35x   <== oc-SAFE (PASS-3 keeps measured max|P|)
 *
 * Root cause the scalar bias dominated: gdn_pack_bias did 128 SCALAR VTCM writes (the documented
 * pathology) re-run EVERY pass (~60% of multi-pass cost). The bias layout is exactly 4 HVX vectors
 * [ctrl | eff[0:32]+rdelta | ctrl | eff[32:64]+rdelta] -> 4 vector stores. See gdn_pack_bias below.
 *
 * PIPELINE (validated in baremetal GDNBM_FEED_4P): 3 HVX producers (fold+quant+pack, with idle slack to
 * hide fold/quant) + 1 HVX-locked consumer running THIS function; depack offloaded to a producer.
 * ============================================================================================ */
/* One logical 64^3 HMX matmul with dynamic-quant output scaling. NB: this runs the HMX kernel 2-3x
 * (PASS 1 + optional PASS 2 + PASS 3) — a gain search to scale the int8 output to fill [-127,127].
 *
 * >>> #1 VTCM-TRAFFIC HOTSPOT of the whole GDNSolveHVXMixHMX path (see Agent/current/gdn_solve_hvxmixhmx.md).
 *     PASS 1/2 are PURE scale-probing: run the matmul, read max|P| via gdn_surf_maxabs, THROW the output
 *     away. Per logical matmul the traffic is ~3 runs(12.5K each) + 2 maxabs-reads(4K) + 1 depack(4K) ≈ 50K,
 *     of which PASS1+2 ≈ 33K (~66%) is discarded. The bare-metal GDNSolveHVXMixHMX microbench measures ONLY PASS 3
 *     (~578 cyc/matmul), so it UNDER-counts real traffic ~2.5-3x.
 *
 *     Why multi-pass: int8 out must be scaled by 127/max|P|, but max|P| needs P itself -> chicken-and-egg.
 *     oc is hypersensitive to this scale (dist-1 blocks: oc 0.73 vs 0.01 if scale ~20% off), so the probes
 *     can't simply be dropped. TRAFFIC-REDUCTION LEVERS (each MUST be oc-validated vs golden):
 *       (a) predict max|P| from input norms (‖A_i,:‖·‖T_:,j‖) -> 0 probe runs, but oc-risky (~1% needed);
 *       (b) replace PASS-1/2 full HMX runs with a cheap HVX sub-sampled dot-product estimate of max|P|
 *           (keeps accuracy, drops 2 HMX runs + 2 out-writes ≈ 33K/matmul);
 *       (c) tighten PASS-1's initial gain (norm-based) so PASS 2 is unnecessary -> 3 passes -> 2 (saves 1/3,
 *           lowest risk). */
/* PRODUCER-CONSUMER hook (GDNSolveHVXMixHMX pipeline): when set (by the bare-metal -DGDNBM_HMX_PIPE driver),
 * gdn_merge_packed's single HMX kernel call is DELEGATED to the main-thread consumer instead of run locally
 * (1 HMX unit -> multi-thread HMX SSRs; only the main consumer touches mxmem, producers feed it).  null =
 * single-thread (run gdn_hmx_run_only locally, unchanged).  slot derived from sc - g_scr in the impl. */
typedef void (*gdn_hmx_dispatch_fn)(gdn_scr_t *sc, const gdn_vtcm_t *vt, const int8_t *wt_kmajor,
                                    const int32_t *eff, float scale_f16, int baseline_u16, int round_out);
static gdn_hmx_dispatch_fn g_hmx_dispatch = nullptr;

static void gdn_merge_packed(gdn_scr_t *sc, const gdn_vtcm_t *vt, const uint8_t *act_crouton, float sa,
                             const int8_t *wt_kmajor, const int32_t *eff, float sw,
                             int wt_colabsmax, int8_t *out_codes, float *s_out) {
    vt->acttab[0] = (int32_t)(uintptr_t)(act_crouton + 0);
    vt->acttab[1] = (int32_t)(uintptr_t)(act_crouton + 64 * 32);
    vt->outtab[0] = (int32_t)(uintptr_t)(vt->out + 0);
    vt->outtab[1] = (int32_t)(uintptr_t)(vt->out + 64 * 32);
#if defined(GDN_BR_PROBE_CYCLES)
    uint64_t es0; asm volatile("%0 = C15:14" : "=r"(es0));
#endif
    /* PASS 1 (#1c): norm-PREDICTED gain g1 = 127/(128*colabsmax) — a Holder upper bound on max|P|
     * (max|act code| <= 128, max-col-abs-sum(wt) = wt_colabsmax). The old PASS-1 used the LOOSE worst case
     * (~100x over), so its codes landed near 1 and the integer maxabs was ~20% off -> needed a PASS-2 refine.
     * The norm-predicted gain fills the u8 range, so PASS-1's maxabs is accurate in ONE shot and PASS-2 is
     * DROPPED. PASS-3 still uses the MEASURED max|P| (code1/g1), so oc is UNCHANGED. Verified oc-safe in sim
     * (scripts/gdn_solve_maxp_probe.py); fallback to LOOSE if colabsmax==0. */
    int maxP_est = 128 * wt_colabsmax;
    float g1 = (maxP_est > 0) ? (127.0f / (float)maxP_est) : (127.0f / (float)GDN_BR_LOOSE_CONST);
#if defined(GDN_BR_STATIC_GAIN)
    /* STATIC SINGLE-PASS (precision先不管): use the colabsmax-PREDICTED gain g1 directly as the OUTPUT
     * gain — skip the PASS-1 HMX run + gdn_surf_maxabs read AND the PASS-3 re-run.  g1 is a Holder upper
     * bound on max|P| (codes fill int8 without saturating; slightly looser than the dynamic tight gain).
     * Kills the multi-pass gain-search HMX tax (~2x the HMX work).  sP uses the predicted maxP_est so
     * dequant value = code*sP = round(P*g1)*(maxP_est*sa*sw/127) ~= P*sa*sw (exact up to the round). */
    {
        float maxP = (maxP_est > 0) ? (float)maxP_est : (float)GDN_BR_LOOSE_CONST;
        float sP = (maxP * sa * sw) / 127.0f; if (sP <= 0.0f) sP = 1e-12f;
#if defined(GDN_BR_STATIC_FULL)
        if (g_force_sP > 0.f) {   /* drain at a CHOSEN scale (e.g. sTw): sP=force -> maxP=127*force/(sa*sw). */
            sP = g_force_sP; maxP = (sa * sw > 0.f) ? (127.0f * sP / (sa * sw)) : maxP;
            g1 = (maxP > 0.f) ? (127.0f / maxP) : g1;
        }
#endif
        if (g_hmx_dispatch)   /* pipeline: hand the kernel to the main-thread HMX consumer, spin for result */
            g_hmx_dispatch(sc, vt, wt_kmajor, eff, g1 * 512.0f, 128 << 7, 1);
        else
            gdn_hmx_run_only(vt, wt_kmajor, eff, g1 * 512.0f, 128 << 7, 1);   /* single output pass (round) */
#if defined(GDN_BR_PROBE_CYCLES)
        { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_hmxkern += q - es0; }
        uint64_t d0; asm volatile("%0 = C15:14" : "=r"(d0));
#endif
        gdn_depack_out_fast(sc, vt->out, 128, out_codes);
#if defined(GDN_BR_PROBE_CYCLES)
        { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_hmxdepack += q - d0; }
#endif
        *s_out = sP;
        return;
    }
#endif
    gdn_hmx_run_only(vt, wt_kmajor, eff, g1 * 512.0f, 128 << 7, 0);
    int code1 = gdn_surf_maxabs(vt->out, 128);                  /* max|P|*g1 — accurate now (PASS-1 fills range) */
    float maxP = (float)code1 / g1; if (maxP < 1.0f) maxP = 1.0f;
    float sP = (maxP * sa * sw) / 127.0f; if (sP <= 0.0f) sP = 1e-12f;
#if defined(GDN_BR_PROBE_CYCLES)
    { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_pint += q - es0; }
#endif
    /* PASS 3 (OUTPUT): tight gain g2 = 127/maxP, depack into out_codes. */
    float g2 = 127.0f / maxP;
#if defined(GDN_BR_PROBE_CYCLES)
    uint64_t r0; asm volatile("%0 = C15:14" : "=r"(r0));
#endif
    gdn_hmx_run_only(vt, wt_kmajor, eff, g2 * 512.0f, 128 << 7, 1);   /* output pass: round the drain */
#if defined(GDN_BR_PROBE_CYCLES)
    { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_hmxkern += q - r0; }
    uint64_t d0; asm volatile("%0 = C15:14" : "=r"(d0));
#endif
    gdn_depack_out_fast(sc, vt->out, 128, out_codes);
#if defined(GDN_BR_PROBE_CYCLES)
    { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_hmxdepack += q - d0; }
#endif
    *s_out = sP;
}

/* ============ int16-HVX merge = the GDNSolveHVX baseline matmul (NO HMX -> threads freely; HMX is process-serial, can't hit 4-thread) ============
 * Replaces the HMX matmul with a direct int16 HVX matmul, reusing the diagonal solve's Q6_Ww_vmpyacc_WwVhRh.
 * Operands quantized to 12-bit (±2047) so the 64-term int32 accumulator can't overflow (64*2047^2 < 2^29). */

/* operand quant range: 12-bit (±2047) by default; GDN_BR_QMAX=127 for the int8-vrmpy precision pre-check. */
#ifndef GDN_BR_QMAX
#define GDN_BR_QMAX 2047
#endif
/* quantize 64x64 int32 codes (value=code*scale_in) to GDN_BR_QMAX-bounded int16 codes; returns scale sQ. */
static float gdn_quant_i12_from_codes(gdn_scr_t *sc, const int32_t *codes, float scale_in, int16_t *out,
                                      int32_t mx) {
    if (mx < 0) mx = gdn_maxabs_codes(codes);
    float maxval = (float)mx * scale_in;
    float sQ = (maxval > 0.0f) ? (maxval / (float)GDN_BR_QMAX) : 1e-12f;
    /* PRE-SHIFT: the fixed-point multiply code*Mg overflows int32 for code>~2^15, and matmul-result
     * operands (Sacc) reach 2^22.  Shift codes (and adjust mx,scale_in) so the shifted code fits <2^15. */
    const HVX_Vector *p = (const HVX_Vector *)codes;
    HVX_Vector *qp = (HVX_Vector *)sc->qbuf;
    int psh = 0; { int32_t m = mx; while (m >= (1 << 15)) { m >>= 1; ++psh; } }
    if (psh > 0) {
        for (int b = 0; b < (BL * BL) / 32; ++b) qp[b] = Q6_Vw_vasr_VwR(p[b], psh);
        p = qp; mx >>= psh; scale_in *= (float)(1 << psh);
    }
    float g = scale_in / sQ;   /* g unchanged: scale_in*2^psh / sQ, with shifted codes -> same value */
    int Q = 14;
    while (Q > 1  && g * (float)(1 << Q) >= 32768.0f) --Q;
    while (Q < 28 && g * (float)(1 << (Q + 1)) < 16384.0f) ++Q;
    int32_t Mg = (int32_t)(g * (float)(1 << Q) + 0.5f);
    const int Mrep = (Mg & 0xFFFF) * 0x10001;
    const HVX_Vector vrnd = Q6_V_vsplat_R(1 << (Q - 1));
    const HVX_Vector vlim = Q6_V_vsplat_R(GDN_BR_QMAX), vnlim = Q6_V_vsplat_R(-GDN_BR_QMAX);
    for (int b = 0; b < (BL * BL) / 32; ++b) {
        HVX_Vector prod = Q6_Vw_vmpyi_VwRh(p[b], Mrep);
        HVX_Vector q = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(prod, vrnd), Q);
        q = Q6_Vw_vmin_VwVw(q, vlim); q = Q6_Vw_vmax_VwVw(q, vnlim);
        qp[b] = q;
    }
    /* narrow int32 -> int16 (natural order): pack 2 int32 vecs (64 lanes) -> 1 int16 vec. */
    HVX_Vector *op = (HVX_Vector *)out;
    for (int v = 0; v < (BL * BL) / 64; ++v) op[v] = Q6_Vh_vpack_VwVw_sat(qp[2 * v + 1], qp[2 * v]);
    return sQ;
}

/* C[64][64] int32 = A[64][64] @ B[64][64], int16 operands.  C[i,:] = sum_k A[i,k]*B[k,:].  Reuses the
 * diagonal MAC: acc(WordPair, even/odd cols) += B_row[k](64 int16) * A[i,k](scalar); vshuff -> natural int32. */
static void gdn_matmul_i16(const int16_t *A, const int16_t *B, int32_t *C) {
    for (int i = 0; i < BL; ++i) {
        const int16_t *Ai = A + i * BL;
#if !defined(GDN_BR_NO_MM4ACC)
        /* 4 independent accumulator chains (k mod 4) break the depth-64 RAW dependency that left the
         * single-acc loop latency-bound.  Measured (real v75, H=32 4-thread): 1-thread mm 233K->211K,
         * whole-solve 4-thread ~150K->126K.  int32 add is exactly associative and 64*2047^2 ~= 268M fits
         * int32, so this is BIT-EXACT with the serial version (relerr unchanged 1.03e-4).  Default ON. */
        HVX_VectorPair a0 = Q6_W_vzero(), a1 = Q6_W_vzero(), a2 = Q6_W_vzero(), a3 = Q6_W_vzero();
        for (int k = 0; k < BL; k += 4) {
            a0 = Q6_Ww_vmpyacc_WwVhRh(a0, ((const HVX_Vector *)(B + (k+0) * BL))[0], (Ai[k+0] & 0xFFFF) * 0x10001);
            a1 = Q6_Ww_vmpyacc_WwVhRh(a1, ((const HVX_Vector *)(B + (k+1) * BL))[0], (Ai[k+1] & 0xFFFF) * 0x10001);
            a2 = Q6_Ww_vmpyacc_WwVhRh(a2, ((const HVX_Vector *)(B + (k+2) * BL))[0], (Ai[k+2] & 0xFFFF) * 0x10001);
            a3 = Q6_Ww_vmpyacc_WwVhRh(a3, ((const HVX_Vector *)(B + (k+3) * BL))[0], (Ai[k+3] & 0xFFFF) * 0x10001);
        }
        HVX_Vector lo = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(Q6_V_lo_W(a0), Q6_V_lo_W(a1)),
                                        Q6_Vw_vadd_VwVw(Q6_V_lo_W(a2), Q6_V_lo_W(a3)));
        HVX_Vector hi = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(Q6_V_hi_W(a0), Q6_V_hi_W(a1)),
                                        Q6_Vw_vadd_VwVw(Q6_V_hi_W(a2), Q6_V_hi_W(a3)));
        HVX_VectorPair nat = Q6_W_vshuff_VVR(hi, lo, -4);
#else
        HVX_VectorPair acc = Q6_W_vzero();
        for (int k = 0; k < BL; ++k)
            acc = Q6_Ww_vmpyacc_WwVhRh(acc, ((const HVX_Vector *)(B + k * BL))[0], (Ai[k] & 0xFFFF) * 0x10001);
        /* Q6_Ww_vmpyacc produces EVEN/ODD split (lo=even cols, hi=odd cols — matmul self-test confirmed);
         * vshuff at word granularity interleaves them back to natural int32 order. */
        HVX_VectorPair nat = Q6_W_vshuff_VVR(Q6_V_hi_W(acc), Q6_V_lo_W(acc), -4);
#endif
        ((HVX_Vector *)(C + i * BL))[0] = Q6_V_lo_W(nat);
        ((HVX_Vector *)(C + i * BL))[1] = Q6_V_hi_W(nat);
    }
}

#if defined(GDN_BR_MM_I8)
/* Pack B (int8, row-major [k][j]) into the vrmpy layout btp[g][col][m] = B[4g+m][col], so that one
 * vrmpy over a broadcast A-word reduces 4 consecutive k at once.  Per k-group g (rows 4g..4g+3): zip
 * (R0,R1) and (R2,R3) bytewise, then halfword-zip those -> 256B = cols 0..63 each as a 4-byte word.
 * vshuff(Vu,Vv,-1)=full byte zip (lo.b[2k]=Vv[k],lo.b[2k+1]=Vu[k]); -2 = halfword zip (keeps byte pairs). */
static void gdn_pack_b_vrmpy(const int8_t *B, int8_t *btp) {
    for (int g = 0; g < BL / 4; ++g) {
        HVX_Vector v = *(const HVX_Vector *)(B + (size_t)(4 * g) * BL);       /* [R0|R1] */
        HVX_Vector w = *(const HVX_Vector *)(B + (size_t)(4 * g + 2) * BL);   /* [R2|R3] */
        HVX_Vector R0 = v, R1 = Q6_V_vror_VR(v, 64), R2 = w, R3 = Q6_V_vror_VR(w, 64);
        HVX_Vector P  = Q6_V_lo_W(Q6_W_vshuff_VVR(R1, R0, -1));   /* (R0[0],R1[0],R0[1],R1[1],...) cols0..63 */
        HVX_Vector Qq = Q6_V_lo_W(Q6_W_vshuff_VVR(R3, R2, -1));   /* (R2[0],R3[0],...) */
        HVX_VectorPair O = Q6_W_vshuff_VVR(Qq, P, -2);           /* halfword zip -> [col][4] */
        ((HVX_Vector *)(btp + (size_t)g * 256))[0] = Q6_V_lo_W(O);   /* cols 0..31 */
        ((HVX_Vector *)(btp + (size_t)g * 256))[1] = Q6_V_hi_W(O);   /* cols 32..63 */
    }
}

/* C[64][64] int32 = A[64][64] @ B[64][64], int8 operands via vrmpy (4 MACs/lane).  B pre-packed in btp.
 * acc.w[c] += sum_m A[i][4g+m]*btp[g][col=c][m] over g -> C[i][c]. */
static void gdn_matmul_i8_vrmpy(const int8_t *A, const int8_t *btp, int32_t *C) {
    for (int i = 0; i < BL; ++i) {
        const int32_t *Aw = (const int32_t *)(A + (size_t)i * BL);   /* 16 words = A[i][4g..4g+3] */
        HVX_Vector acc0 = Q6_V_vzero(), acc1 = Q6_V_vzero();
        for (int g = 0; g < BL / 4; ++g) {
            HVX_Vector vA = Q6_V_vsplat_R(Aw[g]);
            const HVX_Vector *bt = (const HVX_Vector *)(btp + (size_t)g * 256);
            acc0 = Q6_Vw_vrmpyacc_VwVbVb(acc0, vA, bt[0]);
            acc1 = Q6_Vw_vrmpyacc_VwVbVb(acc1, vA, bt[1]);
        }
        ((HVX_Vector *)(C + (size_t)i * BL))[0] = acc0;
        ((HVX_Vector *)(C + (size_t)i * BL))[1] = acc1;
    }
}

/* Direct int8 quant of a strided u16 A sub-block -> a8 (row-major, natural cols).  Skips the int32 fold
 * intermediate: in dynamic-range int8 quant the per-tensor sA cancels, so a8 = round((code-zpA)*127/mxraw)
 * is pure-integer (independent of M,S).  Returns mxraw; caller forms the dequant scale mxraw*sA_eff/127.
 * Saves the fold stage (~16K/head 1-thread) AND the int32 Aoff write+read (~32KB/merge DDR traffic) — the
 * latter helps the bandwidth-bound 4-thread path.  Mirrors gdn_fold_block_raw's even/odd (vzxt+vshuff). */
static int32_t gdn_quant_i8_from_u16(const uint16_t *Au, int row_stride, int zpA, int8_t *out) {
    const HVX_Vector vzp = Q6_V_vsplat_R(zpA);
    HVX_Vector vmax = Q6_V_vzero();
    for (int r = 0; r < BL; ++r) {
        HVX_VectorPair w = Q6_Wuw_vzxt_Vuh(((const HVX_UVector *)(Au + (size_t)r * row_stride))[0]);
        vmax = Q6_Vw_vmax_VwVw(vmax, Q6_Vw_vabs_Vw(Q6_Vw_vsub_VwVw(Q6_V_lo_W(w), vzp)));
        vmax = Q6_Vw_vmax_VwVw(vmax, Q6_Vw_vabs_Vw(Q6_Vw_vsub_VwVw(Q6_V_hi_W(w), vzp)));
    }
    vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4*16));
    vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4*8));
    vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4*4));
    vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4*2));
    vmax = Q6_Vw_vmax_VwVw(vmax, Q6_V_vror_VR(vmax, 4*1));
    int32_t lanes[32] __attribute__((aligned(128))); *(HVX_Vector *)lanes = vmax;
    int32_t mxraw = lanes[0]; if (mxraw < 1) mxraw = 1;
    float g = 127.0f / (float)mxraw;
    int Q = 14; while (Q > 1 && g * (float)(1 << Q) >= 32768.0f) --Q;
    while (Q < 28 && g * (float)(1 << (Q + 1)) < 16384.0f) ++Q;
    const int Mrep = ((int)(g * (float)(1 << Q) + 0.5f) & 0xFFFF) * 0x10001;
    const HVX_Vector vrnd = Q6_V_vsplat_R(1 << (Q - 1)), vlim = Q6_V_vsplat_R(127), vnlim = Q6_V_vsplat_R(-127);
    for (int r = 0; r < BL; r += 2) {
        HVX_VectorPair w0 = Q6_Wuw_vzxt_Vuh(((const HVX_UVector *)(Au + (size_t)r * row_stride))[0]);
        HVX_Vector a0 = Q6_Vw_vsub_VwVw(Q6_V_lo_W(w0), vzp), a1 = Q6_Vw_vsub_VwVw(Q6_V_hi_W(w0), vzp);
        a0 = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(a0, Mrep), vrnd), Q), vnlim), vlim);
        a1 = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(a1, Mrep), vrnd), Q), vnlim), vlim);
        HVX_VectorPair s0 = Q6_W_vshuff_VVR(a1, a0, -4);
        HVX_Vector r0h = Q6_Vh_vpack_VwVw_sat(Q6_V_hi_W(s0), Q6_V_lo_W(s0));
        HVX_VectorPair w1 = Q6_Wuw_vzxt_Vuh(((const HVX_UVector *)(Au + (size_t)(r + 1) * row_stride))[0]);
        HVX_Vector b0 = Q6_Vw_vsub_VwVw(Q6_V_lo_W(w1), vzp), b1 = Q6_Vw_vsub_VwVw(Q6_V_hi_W(w1), vzp);
        b0 = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(b0, Mrep), vrnd), Q), vnlim), vlim);
        b1 = Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_Vw_vmpyi_VwRh(b1, Mrep), vrnd), Q), vnlim), vlim);
        HVX_VectorPair s1 = Q6_W_vshuff_VVR(b1, b0, -4);
        HVX_Vector r1h = Q6_Vh_vpack_VwVw_sat(Q6_V_hi_W(s1), Q6_V_lo_W(s1));
        ((HVX_Vector *)(out + (size_t)r * BL))[0] = Q6_Vb_vpack_VhVh_sat(r1h, r0h);
    }
    return mxraw;
}
#endif  /* GDN_BR_MM_I8 */


/* int16-HVX merge: C = A @ B (int32 codes in/out, no HMX/pack/depack).  out scale = sAq*sBq. */
static void gdn_merge_hvx(gdn_scr_t *sc, const int32_t *A_codes, float sA, int32_t mxA,
                          const int32_t *B_codes, float sB, int32_t mxB, int32_t *C_codes, float *s_out) {
#if defined(GDN_BR_PROBE_CYCLES)
    uint64_t q0; asm volatile("%0 = C15:14" : "=r"(q0));
#endif
#if defined(GDN_BR_MM_I8)
    /* int8 operands + vrmpy matmul (4 MACs/lane): int8 quant preserves oc (operand precision is below the
     * solve's error floor — verified 1.285e-2 == int12), vrmpy is ~4x the int16 broadcast-MAC throughput. */
    float sAq = gdn_quant_i8_from_codes(sc, A_codes, sA, sc->a8, mxA);
    float sBq = gdn_quant_i8_from_codes(sc, B_codes, sB, sc->b8, mxB);
#if defined(GDN_BR_PROBE_CYCLES)
    { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_quant += q - q0; }
    uint64_t m0; asm volatile("%0 = C15:14" : "=r"(m0));
#endif
    gdn_pack_b_vrmpy(sc->b8, sc->btp);
    gdn_matmul_i8_vrmpy(sc->a8, sc->btp, C_codes);
#else
    float sAq = gdn_quant_i12_from_codes(sc, A_codes, sA, sc->a16, mxA);
    float sBq = gdn_quant_i12_from_codes(sc, B_codes, sB, sc->b16, mxB);
#if defined(GDN_BR_PROBE_CYCLES)
    { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_quant += q - q0; }
    uint64_t m0; asm volatile("%0 = C15:14" : "=r"(m0));
#endif
    gdn_matmul_i16(sc->a16, sc->b16, C_codes);
#endif
#if defined(GDN_BR_PROBE_CYCLES)
    { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_hmxkern += q - m0; }
#endif
    *s_out = sAq * sBq;
}

/* accumulate an int32 term (scale s_term) into sc->Sacc (scale s_S): Sacc += round(term*s_term/s_S).
 * first==1 => Sacc = term (caller sets s_S=s_term).  No widen (term is already int32). */
static void gdn_acc_i32_to_codes(gdn_scr_t *sc, const int32_t *term, float s_term, float s_S, int first) {
    HVX_Vector *sp = (HVX_Vector *)sc->Sacc;
    const HVX_Vector *tp = (const HVX_Vector *)term;
    if (first) { for (int b = 0; b < (BL * BL) / 32; ++b) sp[b] = tp[b]; return; }
    /* PRE-SHIFT: term*Mg overflows int32 for term>~2^15 (matmul results reach 2^22). */
    int psh = 0; { int32_t mx = gdn_maxabs_codes(term); int32_t m = mx; while (m >= (1 << 15)) { m >>= 1; ++psh; } }
    float g = (s_term / s_S) * (float)(1 << psh);
    int Q = 14;
    while (Q > 1  && g * (float)(1 << Q) >= 32768.0f) --Q;
    while (Q < 28 && g * (float)(1 << (Q + 1)) < 16384.0f) ++Q;
    int32_t Mg = (int32_t)(g * (float)(1 << Q) + 0.5f);
    const int Mrep = (Mg & 0xFFFF) * 0x10001;
    const HVX_Vector vrnd = Q6_V_vsplat_R(1 << (Q - 1));
    for (int b = 0; b < (BL * BL) / 32; ++b) {
        HVX_Vector t = (psh > 0) ? Q6_Vw_vasr_VwR(tp[b], psh) : tp[b];
        HVX_Vector prod = Q6_Vw_vmpyi_VwRh(t, Mrep);
        sp[b] = Q6_Vw_vadd_VwVw(sp[b], Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(prod, vrnd), Q));
    }
}

/* order-preserving widen of 4096 int8 codes -> 4096 int32 in dst.  vsxt deals bytes; we re-interleave
 * the 4 int32 streams (byte indices 4j, 4j+1, 4j+2, 4j+3) with two word-granularity vshuffs. */
static void gdn_widen_i8_to_i32(const int8_t *src, int32_t *dst) {
    const HVX_Vector *sp = (const HVX_Vector *)src;   /* 128 i8 / vec */
    HVX_Vector *dp = (HVX_Vector *)dst;               /* 32 i32 / vec */
    for (int b = 0; b < (BL * BL) / 128; ++b) {
        HVX_VectorPair w16 = Q6_Wh_vsxt_Vb(sp[b]);              /* lo=byte[2i], hi=byte[2i+1] */
        HVX_VectorPair wlo = Q6_Ww_vsxt_Vh(Q6_V_lo_W(w16));     /* llo=byte[4j], lhi=byte[4j+2] */
        HVX_VectorPair whi = Q6_Ww_vsxt_Vh(Q6_V_hi_W(w16));     /* hlo=byte[4j+1], hhi=byte[4j+3] */
        /* want sequential byte[4j+0..3] = (llo,hlo,lhi,hhi).  interleave (llo,lhi) and (hlo,hhi) at word,
         * then interleave those at word again to weave the 4 streams. */
        HVX_VectorPair pA = Q6_W_vshuff_VVR(Q6_V_lo_W(whi), Q6_V_lo_W(wlo), -4); /* (llo,hlo) pairs */
        HVX_VectorPair pB = Q6_W_vshuff_VVR(Q6_V_hi_W(whi), Q6_V_hi_W(wlo), -4); /* (lhi,hhi) pairs */
        /* pA lo = llo[0],hlo[0],llo[1],hlo[1]...  pB lo = lhi[0],hhi[0],...  -> need (llo,hlo,lhi,hhi) */
        HVX_VectorPair s0 = Q6_W_vshuff_VVR(Q6_V_lo_W(pB), Q6_V_lo_W(pA), -8);
        HVX_VectorPair s1 = Q6_W_vshuff_VVR(Q6_V_hi_W(pB), Q6_V_hi_W(pA), -8);
        dp[b*4+0] = Q6_V_lo_W(s0); dp[b*4+1] = Q6_V_hi_W(s0);
        dp[b*4+2] = Q6_V_lo_W(s1); dp[b*4+3] = Q6_V_hi_W(s1);
    }
}

/* accumulate int8 term codes (scale s_term) into int32 accumulator g_Sacc (scale s_S):
 *   Sacc += round(term * s_term / s_S).   first==1 => Sacc = term (and caller sets s_S=s_term).
 * Vectorized with a fixed-point multiplier g=s_term/s_S. */
static void gdn_acc_i8_to_codes(gdn_scr_t *sc, const int8_t *term, float s_term, float s_S, int first) {
    if (first) {
        gdn_widen_i8_to_i32(term, sc->Sacc);
        return;
    }
#if defined(GDN_BR_STATIC_FULL)
    /* PURE-ADD: static scales -> all inner-sum terms share a scale (g=1) -> widen + int32 add, no rescale. */
    {
        (void)s_term; (void)s_S;
        gdn_widen_i8_to_i32(term, sc->qbuf);
        const HVX_Vector *tpa = (const HVX_Vector *)sc->qbuf; HVX_Vector *spa = (HVX_Vector *)sc->Sacc;
        for (int b = 0; b < (BL * BL) / 32; ++b) spa[b] = Q6_Vw_vadd_VwVw(spa[b], tpa[b]);
        return;
    }
#endif
    float g = s_term / s_S;
    int Q = 14;
    while (Q > 1  && g * (float)(1 << Q) >= 32768.0f) --Q;
    while (Q < 28 && g * (float)(1 << (Q + 1)) < 16384.0f) ++Q;
    int32_t Mg = (int32_t)(g * (float)(1 << Q) + 0.5f);
    const int Mrep = (Mg & 0xFFFF) * 0x10001;
    const HVX_Vector vrnd = Q6_V_vsplat_R(1 << (Q - 1));
    /* widen term (order-preserving HVX) into sc->qbuf, then vectorized multiply-add into sc->Sacc. */
    gdn_widen_i8_to_i32(term, sc->qbuf);
    const HVX_Vector *tp = (const HVX_Vector *)sc->qbuf;
    HVX_Vector *sp = (HVX_Vector *)sc->Sacc;
    for (int b = 0; b < (BL * BL) / 32; ++b) {
        HVX_Vector prod = Q6_Vw_vmpyi_VwRh(tp[b], Mrep);
        sp[b] = Q6_Vw_vadd_VwVw(sp[b], Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(prod, vrnd), Q));
    }
}
#endif  /* __hexagon__ */

/* ============================ host (x86) reference fallback ============================ */
/* Mirrors the device math in plain C double so the op is correct off-device too (and so the
 * standalone harness can compare). Uses GdnSolveOp's solve core for the diagonals. */
#include "../../solve_op/src/gdn_solve_core.h"
/* generic nb=NB host fallback (mirrors gdn_blockrec_c256_probe): diagonals int16 fwd-subst, off-diag
 * float-accumulated u8i8 merges in increasing diagonal-distance. */
static void gdn_hmerge_u8i8(const float *act, const float *wt, int n, float *out) {
    float sa = gdn_br_qsym_scale(act, n * n), sw = gdn_br_qsym_scale(wt, n * n);
    for (int r = 0; r < n; ++r) for (int c = 0; c < n; ++c) {
        long acc = 0;
        for (int k = 0; k < n; ++k) {
            long aq = lroundf(act[r * n + k] / sa); if (aq > 127) aq = 127; if (aq < -127) aq = -127;
            long bq = lroundf(wt[k * n + c] / sw); if (bq > 127) bq = 127; if (bq < -127) bq = -127;
            acc += aq * bq;
        }
        out[r * n + c] = (float)acc * (sa * sw);
    }
}
static void gdn_br_head_scalar(const uint16_t *Au, int zpA, float sA,
                               float sT, int zpT, uint16_t *Tu) {
    static float Tblk[GDN_BR_NBLK][BL * BL];
    int16_t sub16[BL * BL], Tcode[BL * BL];
    /* diagonals */
    for (int i = 0; i < NB; ++i) {
        for (int r = 0; r < BL; ++r) for (int c = 0; c < BL; ++c)
            sub16[r * BL + c] = (int16_t)((int)Au[(i * BL + r) * C + (i * BL + c)] - zpA);
        gdn_solve_head_q<int16_t>(sub16, BL, sA, GDN_BR_TI, 32767.0f, Tcode);
        int bi = gdn_blk_index(i, i);
        for (int q = 0; q < BL * BL; ++q) Tblk[bi][q] = (float)Tcode[q] * GDN_BR_TI;
    }
    /* off-diagonals */
    static float Aik[BL * BL], Sf[BL * BL], term[BL * BL];
    for (int d = 1; d < NB; ++d) for (int j = 0; j + d < NB; ++j) {
        int i = j + d;
        for (int q = 0; q < BL * BL; ++q) Sf[q] = 0.0f;
        for (int k = j; k < i; ++k) {
            for (int r = 0; r < BL; ++r) for (int c = 0; c < BL; ++c)
                Aik[r * BL + c] = ((int)Au[(i * BL + r) * C + (k * BL + c)] - zpA) * sA;
            gdn_hmerge_u8i8(Aik, Tblk[gdn_blk_index(k, j)], BL, term);
            for (int q = 0; q < BL * BL; ++q) Sf[q] += term[q];
        }
        gdn_hmerge_u8i8(Tblk[gdn_blk_index(i, i)], Sf, BL, Tblk[gdn_blk_index(i, j)]);
    }
    /* assemble */
    for (int q = 0; q < C * C; ++q) Tu[q] = (uint16_t)zpT;
    auto put = [&](int r, int c, float v) {
        long q = lroundf(v / sT); long lim = 32767;
        if (q > lim) q = lim; if (q < -lim) q = -lim;
        Tu[r * C + c] = (uint16_t)((int)q + zpT);
    };
    for (int i = 0; i < NB; ++i) for (int jj = 0; jj <= i; ++jj) {
        int bij = gdn_blk_index(i, jj);
        for (int r = 0; r < BL; ++r) for (int c = 0; c < BL; ++c)
            put(i * BL + r, jj * BL + c, Tblk[bij][r * BL + c]);
    }
}

#if defined(GDN_BR_TRACE) && defined(__hexagon__)
/* Per-stage timeline for the bare-metal multi-thread run: each worker appends {tid,stage,t0,t1} spans.
 * C15:14 PCYCLE is a GLOBAL core cycle counter -> timestamps from the 4 worker threads share one timebase,
 * so the assembled trace shows real cross-thread overlap/load-balance.  gdnbm_solve sets g_tr_base at
 * spawn time and serializes g_tr[] into the output for the host to emit a Chrome/Perfetto trace JSON. */
enum { GDN_TR_HEAD = 0, GDN_TR_DIAG = 1, GDN_TR_MERGE = 2, GDN_TR_MM = 3, GDN_TR_QUANT = 4,
       GDN_TR_PREP = 5, GDN_TR_ACC = 6, GDN_TR_REQ = 7 };   /* PREP=get_act+get_wt(fold/quant/pack), ACC, REQ=widen+requant */
struct gdn_tr_ev { uint32_t tid, stage; uint64_t t0, t1; };
#define GDN_TR_MAX 32768
gdn_tr_ev g_tr[GDN_TR_MAX];
volatile int g_tr_n;
uint64_t g_tr_base;
static inline uint64_t gdn_tr_pcyc(void) { uint64_t c; asm volatile("%0 = C15:14" : "=r"(c)); return c; }
static inline void gdn_tr_push(uint32_t tid, uint32_t stage, uint64_t t0, uint64_t t1) {
    int i = __sync_fetch_and_add(&g_tr_n, 1);
    if (i < GDN_TR_MAX) { g_tr[i].tid = tid; g_tr[i].stage = stage; g_tr[i].t0 = t0 - g_tr_base; g_tr[i].t1 = t1 - g_tr_base; }
}
#define GDN_TR_NOW() gdn_tr_pcyc()
#else
#define GDN_TR_NOW() 0
#endif

#if defined(__hexagon__)
/* compute one head's T = (I-A)^-1 (block-recursive) into Th, using per-thread scratch sc and per-thread
 * VTCM region vt.  zpA/sA->(M,S) fold params; sT/zpT output quant.  Pure HVX + HMX (no shared globals). */
static void gdn_br_one_head(gdn_scr_t *sc, const gdn_vtcm_t *vt, const uint16_t *Ah, uint16_t *Th,
                            int zpA, int M, int S, float sT, int zpT) {
#if defined(GDN_BR_TRACE)
    uint32_t _tid = (uint32_t)(sc - g_scr); uint64_t _hd0 = GDN_TR_NOW();
#endif
#if defined(GDN_BR_PROBE_CYCLES)
    uint64_t t0; asm volatile("%0 = C15:14" : "=r"(t0));
#endif
    /* reset the per-head operand-reuse caches (Task 2). */
    for (int b = 0; b < GDN_BR_NBLK; ++b) { sc->vAa[b] = 0; sc->vTw[b] = 0; }
    for (int i = 0; i < NB; ++i) sc->vTa[i] = 0;
    for (int i = 0; i < NB; ++i) {
        int bi = gdn_blk_index(i, i);
#if defined(GDN_BR_STATIC_FULL)
        gdn_solve_diag64(sc, Ah + (size_t)i * BL * C + i * BL, C, zpA, M, S, sc->Tblk[bi], nullptr);  /* static: skip mx_out (downstream T quant fixed-scale) */
#else
        gdn_solve_diag64(sc, Ah + (size_t)i * BL * C + i * BL, C, zpA, M, S, sc->Tblk[bi], &sc->mxdiag[i]);
#endif
        sc->Tscl[bi] = GDN_BR_TI;
    }
#if defined(GDN_BR_TRACE)
    gdn_tr_push(_tid, GDN_TR_DIAG, _hd0, GDN_TR_NOW());
#endif
#if defined(GDN_BR_PROBE_CYCLES)
    uint64_t t1; asm volatile("%0 = C15:14" : "=r"(t1)); g_c_diag += t1 - t0;
#endif
#if defined(GDN_BR_PROBE_CYCLES)
    uint64_t z0; asm volatile("%0 = C15:14" : "=r"(z0));
#endif
#if !defined(GDN_BR_SKIP_ZERO)   /* timing-only ablation: skip the Th zero-fill */
    /* Zero ONLY the strict-block-upper-triangle of T (block-rows bi, block-cols bj>bi).  T is block-lower-
     * triangular: the diagonal blocks AND the lower off-diagonal blocks are FULLY overwritten by
     * gdn_requant_block_out (diag incl. its zpT upper-tri), so zeroing them is wasted work.  Only the
     * upper off-diagonal blocks are never written -> need zpT.  Cuts zero-fill NB^2 -> NB(NB-1)/2 blocks
     * (C=256: 16->6, -62%; C=128: 4->1).  Each 64-wide block-col = one 128B vector per row. */
    { HVX_Vector vzph = Q6_Vh_vsplat_R(zpT);
      int aligned = (((uintptr_t)Th & 127) == 0);
      for (int bi = 0; bi < NB - 1; ++bi) {
        int nbc = NB - 1 - bi;                                   /* # upper block-cols in this block-row */
        for (int r = 0; r < BL; ++r) {
          uint16_t *rowp = Th + ((size_t)bi * BL + r) * C + (size_t)(bi + 1) * BL;
          if (aligned) { HVX_Vector  *op = (HVX_Vector  *)rowp; for (int b = 0; b < nbc; ++b) op[b] = vzph; }
          else        { HVX_UVector *op = (HVX_UVector *)rowp; for (int b = 0; b < nbc; ++b) op[b] = vzph; }
        }
      }
    }
#endif
#if defined(GDN_BR_PROBE_CYCLES)
    { uint64_t z; asm volatile("%0 = C15:14" : "=r"(z)); g_c_zero += z - z0; }
#endif
    for (int i = 0; i < NB; ++i) {
        int bi = gdn_blk_index(i, i);
#if defined(GDN_BR_PROBE_CYCLES)
        uint64_t rq0; asm volatile("%0 = C15:14" : "=r"(rq0));
#endif
        /* diag upper-triangle is exactly code 0 from the forward-subst (T_ii is unit-lower-tri), so
         * requant writes exactly zpT there — no separate scalar upper-tri cleanup needed. */
        gdn_requant_block_out(sc->Tblk[bi], sc->Tscl[bi], sT, zpT, Th, i * BL, i * BL, C);
#if defined(GDN_BR_PROBE_CYCLES)
        { uint64_t rq; asm volatile("%0 = C15:14" : "=r"(rq)); g_c_requant += rq - rq0; }
#endif
    }
#if defined(GDN_BR_DIAG_ONLY)
    return;
#endif
    for (int d = 1; d < NB; ++d) {
        for (int j = 0; j + d < NB; ++j) {
            int i = j + d;
#if defined(GDN_BR_TRACE)
            uint64_t _mg0 = GDN_TR_NOW();
#endif
#if defined(GDN_BR_PROBE_CYCLES)
            uint64_t m0; asm volatile("%0 = C15:14" : "=r"(m0));
#endif
            int bij = gdn_blk_index(i, j);
#if defined(GDN_BR_HVX_MERGE)
            /* int16-HVX merges (NO HMX -> threads freely).  Operate directly on int32 codes. */
            int bii = gdn_blk_index(i, i);
            float s_S = 0.0f; int first = 1;
            for (int k = j; k < i; ++k) {
                int bkj = gdn_blk_index(k, j);
                float sterm;
#if defined(GDN_BR_MM_I8)
                /* int8 direct path: quant A_ik straight from u16 (no int32 fold), quant B, vrmpy. */
#if defined(GDN_BR_PROBE_CYCLES)
                uint64_t f0; asm volatile("%0 = C15:14" : "=r"(f0));
#endif
#if defined(GDN_BR_TRACE)
                uint64_t _q0 = GDN_TR_NOW();
#endif
                float sA_eff = (float)M; for (int _t = 0; _t < S + GDN_BR_F; ++_t) sA_eff *= 0.5f;
                int32_t mxraw = gdn_quant_i8_from_u16(Ah + (size_t)i * BL * C + k * BL, C, zpA, sc->a8);
                float sAq = (float)mxraw * sA_eff / 127.0f;
                float sBq = gdn_quant_i8_from_codes(sc, sc->Tblk[bkj], sc->Tscl[bkj], sc->b8, -1);
#if defined(GDN_BR_PROBE_CYCLES)
                { uint64_t f; asm volatile("%0 = C15:14" : "=r"(f)); g_c_quant += f - f0; }
                uint64_t mm0; asm volatile("%0 = C15:14" : "=r"(mm0));
#endif
#if defined(GDN_BR_TRACE)
                uint64_t _mm0 = GDN_TR_NOW(); gdn_tr_push(_tid, GDN_TR_QUANT, _q0, _mm0);
#endif
                gdn_pack_b_vrmpy(sc->b8, sc->btp);
                gdn_matmul_i8_vrmpy(sc->a8, sc->btp, sc->Tc);
                sterm = sAq * sBq;
#if defined(GDN_BR_TRACE)
                gdn_tr_push(_tid, GDN_TR_MM, _mm0, GDN_TR_NOW());
#endif
#if defined(GDN_BR_PROBE_CYCLES)
                { uint64_t mm; asm volatile("%0 = C15:14" : "=r"(mm)); g_c_hmxkern += mm - mm0; }
#endif
#else  /* uncached */
#if defined(GDN_BR_PROBE_CYCLES)
                uint64_t f0; asm volatile("%0 = C15:14" : "=r"(f0));
#endif
                int32_t mxA;
                gdn_fold_block_raw(Ah + (size_t)i * BL * C + k * BL, C, sc->Aoff, zpA, M, S, &mxA);
#if defined(GDN_BR_PROBE_CYCLES)
                { uint64_t f; asm volatile("%0 = C15:14" : "=r"(f)); g_c_fold += f - f0; }
#endif
                gdn_merge_hvx(sc, sc->Aoff, (float)(1.0 / (1 << GDN_BR_F)), mxA,
                              sc->Tblk[bkj], sc->Tscl[bkj], -1, sc->Tc, &sterm);
#endif
                if (first) s_S = sterm;
#if defined(GDN_BR_PROBE_CYCLES)
                uint64_t a0; asm volatile("%0 = C15:14" : "=r"(a0));
#endif
                gdn_acc_i32_to_codes(sc, sc->Tc, sterm, s_S, first);
#if defined(GDN_BR_PROBE_CYCLES)
                { uint64_t a; asm volatile("%0 = C15:14" : "=r"(a)); g_c_acc += a - a0; }
#endif
                first = 0;
            }
            float sij;
            gdn_merge_hvx(sc, sc->Tblk[bii], GDN_BR_TI, -1, sc->Sacc, s_S, -1, sc->Tblk[bij], &sij);
            sc->Tscl[bij] = sij;
#if defined(GDN_BR_DUMP_SACC)
            if (i == 1 && j == 0) {   /* dump dist-1 FINAL result (T11@S) + scale to Th, then bail */
                int32_t *d = (int32_t *)Th;
                for (int q = 0; q < BL * BL; ++q) d[q] = sc->Tblk[bij][q];
                ((float *)Th)[BL * BL] = sij;
                return;
            }
#endif
#if defined(GDN_BR_PROBE_CYCLES)
            uint64_t rq0; asm volatile("%0 = C15:14" : "=r"(rq0));
#endif
            gdn_requant_block_out(sc->Tblk[bij], sij, sT, zpT, Th, i * BL, j * BL, C);
#if defined(GDN_BR_PROBE_CYCLES)
            { uint64_t rq; asm volatile("%0 = C15:14" : "=r"(rq)); g_c_requant += rq - rq0; }
            (void)m0;
#endif
#else  /* HMX merge path */
            float s_S = 0.0f; int first = 1;
            for (int k = j; k < i; ++k) {
                /* cached: A_ik act (fold+quant+pack once per (i,k)), T_kj wt (quant+pack+eff once per (k,j)). */
                float sa, sw, sterm; const int32_t *eff; int wcolabs;
#if defined(GDN_BR_TRACE)
                uint64_t _p0 = GDN_TR_NOW();
#endif
                const uint8_t *a = gdn_get_act_A(sc, vt, Ah, i, k, C, zpA, M, S, &sa);
                const int8_t  *w = gdn_get_wt_T(sc, vt, k, j, &sw, &eff, &wcolabs);
#if defined(GDN_BR_TRACE)
                uint64_t _p1 = GDN_TR_NOW(); gdn_tr_push(_tid, GDN_TR_PREP, _p0, _p1);
#endif
                gdn_merge_packed(sc, vt, a, sa, w, eff, sw, wcolabs, sc->termi, &sterm);
#if defined(GDN_BR_TRACE)
                uint64_t _p2 = GDN_TR_NOW(); gdn_tr_push(_tid, GDN_TR_MM, _p1, _p2);
#endif
                if (first) s_S = sterm;
#if defined(GDN_BR_PROBE_CYCLES)
                uint64_t a0; asm volatile("%0 = C15:14" : "=r"(a0));
#endif
                gdn_acc_i8_to_codes(sc, sc->termi, sterm, s_S, first);
#if defined(GDN_BR_TRACE)
                gdn_tr_push(_tid, GDN_TR_ACC, _p2, GDN_TR_NOW());
#endif
#if defined(GDN_BR_PROBE_CYCLES)
                { uint64_t a; asm volatile("%0 = C15:14" : "=r"(a)); g_c_acc += a - a0; }
#endif
                first = 0;
            }
            /* final merge T_ij = T_ii @ S_ij: cached T_ii act; S_ij (Sacc) is transient -> quant+pack into
             * the transient vt->wt surface (no cache slot — each S_ij is distinct). */
            float sa_ii, sw_S, sij; int scolabs;
#if defined(GDN_BR_TRACE)
            uint64_t _fp0 = GDN_TR_NOW();
#endif
            const uint8_t *a_ii = gdn_get_act_Tdiag(sc, vt, i, &sa_ii);
#if defined(GDN_BR_PROBE_CYCLES)
            uint64_t sq0; asm volatile("%0 = C15:14" : "=r"(sq0));
#endif
#if defined(GDN_BR_STATIC_FULL)
#if defined(GDN_BR_SACC_CAL)
            g_ops_i8 = 0.f;   /* CAL: run dynamic, track max sw_S to calibrate GDN_OPS_sSacc. */
#else
            g_ops_i8 = GDN_OPS_sSacc;   /* STATIC: fixed Sacc scale -> skip maxabs. */
#endif
#endif
            sw_S = gdn_quant_i8_from_codes(sc, sc->Sacc, s_S, sc->wtbuf, -1);
#if defined(GDN_BR_SACC_CAL)
            if (sw_S > g_cal_swS) g_cal_swS = sw_S;
#endif
#if defined(GDN_BR_PROBE_CYCLES)
            { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_quant += q - sq0; }
            uint64_t se0; asm volatile("%0 = C15:14" : "=r"(se0));
#endif
            gdn_effective(sc->wtbuf, sc->eff, &scolabs);   /* #1c: col-abs-sum max for the S_ij wt */
#if defined(GDN_BR_PROBE_CYCLES)
            { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_eff += q - se0; }
            uint64_t sp0; asm volatile("%0 = C15:14" : "=r"(sp0));
#endif
            gdn_pack_w8_kmajor(sc->wtbuf, vt->wt);
#if defined(GDN_BR_PROBE_CYCLES)
            { uint64_t q; asm volatile("%0 = C15:14" : "=r"(q)); g_c_wtpack += q - sp0; }
#endif
#if defined(GDN_BR_TRACE)
            uint64_t _fp1 = GDN_TR_NOW(); gdn_tr_push(_tid, GDN_TR_PREP, _fp0, _fp1);
#endif
#if defined(GDN_BR_STATIC_FULL)
            g_force_sP = GDN_OPS_sTw;   /* drain T_ij at sTw -> its later use as off-diag wt is a pure narrow (g=1) */
#endif
            gdn_merge_packed(sc, vt, a_ii, sa_ii, vt->wt, sc->eff, sw_S, scolabs, sc->termi, &sij);
#if defined(GDN_BR_STATIC_FULL)
            g_force_sP = 0.f;
#endif
#if defined(GDN_BR_TRACE)
            uint64_t _fp2 = GDN_TR_NOW(); gdn_tr_push(_tid, GDN_TR_MM, _fp1, _fp2);
#endif
#if defined(GDN_BR_PROBE_CYCLES)
            uint64_t w0; asm volatile("%0 = C15:14" : "=r"(w0));
#endif
            gdn_widen_i8_to_i32(sc->termi, sc->Tblk[bij]);
#if defined(GDN_BR_PROBE_CYCLES)
            { uint64_t w; asm volatile("%0 = C15:14" : "=r"(w)); g_c_widen += w - w0; }
#endif
            sc->Tscl[bij] = sij;
#if defined(GDN_BR_PROBE_CYCLES)
            (void)m0;
            uint64_t rq0; asm volatile("%0 = C15:14" : "=r"(rq0));
#endif
            gdn_requant_block_out(sc->Tblk[bij], sij, sT, zpT, Th, i * BL, j * BL, C);
#if defined(GDN_BR_TRACE)
            gdn_tr_push(_tid, GDN_TR_REQ, _fp2, GDN_TR_NOW());   /* widen + requant after the final MM */
#endif
#if defined(GDN_BR_PROBE_CYCLES)
            { uint64_t rq; asm volatile("%0 = C15:14" : "=r"(rq)); g_c_requant += rq - rq0; }
#endif
#endif  /* GDN_BR_HVX_MERGE */
#if defined(GDN_BR_TRACE)
            gdn_tr_push(_tid, GDN_TR_MERGE, _mg0, GDN_TR_NOW());
#endif
        }
    }
#if defined(GDN_BR_TRACE)
    gdn_tr_push(_tid, GDN_TR_HEAD, _hd0, GDN_TR_NOW());
#endif
}

/* worker-thread dispatch: each of GDN_BR_NT threads processes a strided subset of heads with its own
 * scratch slot and its own VTCM region (vt_slot).  Heads are independent, so this is embarrassingly
 * parallel; HMX is driven from every worker (the kernel re-locks per-call). */
struct gdn_work_t {
    int slot;
    uint32_t h0, h1, nheads;
    const uint16_t *Au; uint16_t *Tu;
    int zpA, M, S; float sT; int zpT;
    uint8_t *vtcm_base;
};
static void gdn_br_run_slot(gdn_work_t *w) {
    gdn_scr_t *sc = &g_scr[w->slot];
    /* each slot uses a distinct 0x60000-spaced VTCM region. */
    gdn_vtcm_t vt = gdn_vtcm_from(w->vtcm_base + (size_t)w->slot * 0x60000);
    for (uint32_t h = w->h0 + w->slot; h < w->h1; h += w->nheads)
        gdn_br_one_head(sc, &vt, w->Au + (size_t)h * C * C, w->Tu + (size_t)h * C * C,
                        w->zpA, w->M, w->S, w->sT, w->zpT);
}
/* spawned-worker wrapper: grabs its OWN HVX context AND its OWN HMX unit (v75 has 2 HMX units; main
 * releases the backend-granted unit before spawning so the workers can claim both).  The earlier "worker
 * HMX faults" was simply a MISSING qurt_hmx_lock — the kernel needs HMX enabled on the calling thread. */
static volatile int g_wkr_hmx_rc[GDN_BR_NT];
static void gdn_br_worker(void *arg) {
    gdn_work_t *w = (gdn_work_t *)arg;
    qurt_hvx_lock(QURT_HVX_MODE_128B);
#if defined(GDN_BR_HMX_SHARED)
    int hrc = qurt_hmx_lock2(QURT_HMX_SHARED_LOCK);
    g_wkr_hmx_rc[w->slot] = hrc;
#elif !defined(GDN_BR_NO_HMX_LOCK)
    int hrc = qurt_hmx_lock();
    g_wkr_hmx_rc[w->slot] = hrc;
#endif
    gdn_br_run_slot(w);
#if defined(GDN_BR_HMX_SHARED)
    if (hrc == QURT_EOK) qurt_hmx_unlock2(QURT_HMX_SHARED_UNLOCK);
#elif !defined(GDN_BR_NO_HMX_LOCK)
    if (hrc == QURT_EOK) qurt_hmx_unlock();
#endif
    qurt_hvx_unlock();
}
static char __attribute__((aligned(128))) g_wkr_stack[GDN_BR_NT][32768];
#endif  /* __hexagon__ (gdn_br_one_head / worker) */

#ifndef GDN_BR_NO_QHPI    /* QHPI wrapper + registration (excluded for the bare-metal HAP include) */
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

#if defined(GDN_BR_PROBE_TOTAL) && defined(__hexagon__)
    /* WALL C15:14 (PCYCLE) around the ENTIRE op work, read with the SAME register the bare-metal harness
     * uses (gdnbm_imp.cpp pcyc()).  Lets us measure THIS op in the C15:14 metric AND cross-reference it
     * against QHAS 'cycles' for the same instance -> the exact C15:14<->QHAS-cycle conversion. */
    uint64_t g_probe_total_t0; asm volatile("%0 = C15:14" : "=r"(g_probe_total_t0));
#endif

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
#if defined(GDN_BR_SKIP_KERNEL)
    for (uint32_t h = h0; h < h1; ++h) {
        uint16_t *Th = Tu + (size_t)h * C * C;
        for (int i = 0; i < C * C; ++i) Th[i] = (uint16_t)zpT;
        Th[0] = 0x4252u; Th[1] = (uint16_t)h;
    }
    return QHPI_Success;
#endif
    /* Head-parallel dispatch.  DEVICE CONSTRAINT (measured): the QNN backend acquires HMX for the MAIN
     * callback thread only; a spawned qurt worker that calls the mxmem kernel faults ("Graph Execution
     * failure"), even at NT=1.  So HMX can only be driven from the calling thread.  We therefore run the
     * per-head work INLINE on the calling thread (single HVX+HMX context).  To use >1 thread the HMX
     * kernel calls would have to be marshalled back to the main thread (HVX-only workers) — see report.
     * GDN_BR_USE_THREADS gates the (currently faulting) worker-spawn path for experimentation. */
    int nthreads = (int)((heads < (uint32_t)GDN_BR_NT) ? heads : (uint32_t)GDN_BR_NT);
    if (nthreads < 1) nthreads = 1;
    gdn_work_t work[GDN_BR_NT];
#if defined(GDN_BR_USE_THREADS)
    /* release the backend-granted HMX on this (main) thread so the spawned workers can each claim one of
     * the 2 v75 HMX units; re-lock after join so the backend's teardown sees its lock intact. */
#if defined(GDN_BR_MAIN_HMX_REL)
    qurt_hmx_unlock();
#endif
    qurt_thread_t tids[GDN_BR_NT];
    for (int t = 0; t < nthreads; ++t) {
        work[t] = gdn_work_t{ t, h0, h1, (uint32_t)nthreads, Au, Tu, zpA, M, S, sT, zpT, vtcm_base };
        qurt_thread_attr_t attr; qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_name(&attr, (char *)"gdn_br_wkr");
        qurt_thread_attr_set_stack_addr(&attr, g_wkr_stack[t]);
        qurt_thread_attr_set_stack_size(&attr, sizeof(g_wkr_stack[t]));
        if (qurt_thread_create(&tids[t], &attr, gdn_br_worker, &work[t]) != QURT_EOK) {
            gdn_br_worker(&work[t]); tids[t] = 0;
        }
    }
    for (int t = 0; t < nthreads; ++t) { int st; if (tids[t]) qurt_thread_join(tids[t], &st); }
#if defined(GDN_BR_MAIN_HMX_REL)
    qurt_hmx_lock();
#endif
#else
    /* single calling thread processes ALL heads (HMX stays on the thread the backend gave HMX to). */
    work[0] = gdn_work_t{ 0, h0, h1, 1u, Au, Tu, zpA, M, S, sT, zpT, vtcm_base };
    gdn_br_run_slot(&work[0]);
    (void)nthreads;
#endif
#if defined(GDN_BR_PROBE_TOTAL)
    if (h0 < h1) {
        uint64_t g_probe_total_t1; asm volatile("%0 = C15:14" : "=r"(g_probe_total_t1));
        uint64_t tot = g_probe_total_t1 - g_probe_total_t0;
        uint32_t *p = (uint32_t *)(Tu + (size_t)h0 * C * C);
        /* [0]=total C15:14 wall (this op, all heads, calling-thread), [1]=heads, [2]=hi32 of total */
        p[0] = (uint32_t)tot; p[1] = (uint32_t)(h1 - h0); p[2] = (uint32_t)(tot >> 32);
    }
#endif
#if defined(GDN_BR_PROBE_CYCLES)
    if (h0 < h1) {
        uint16_t *Th0 = Tu + (size_t)h0 * C * C;
        uint32_t *p = (uint32_t *)Th0;
        /* NOTE: g_c_* are summed over all worker threads (work-volume, not wall). */
        p[0] = (uint32_t)g_c_diag; p[1] = (uint32_t)nthreads; p[2] = 0; p[3] = (h1 - h0);
        p[4] = (uint32_t)g_c_hmxpack; p[5] = (uint32_t)g_c_hmxkern; p[6] = (uint32_t)g_c_hmxdepack;
        p[7] = (uint32_t)g_c_quant; p[8] = (uint32_t)g_c_pint;
        p[9] = (uint32_t)g_c_eff; p[10] = (uint32_t)g_c_actpack; p[11] = (uint32_t)g_c_wtpack;
        p[12] = (uint32_t)g_c_fold; p[13] = (uint32_t)g_c_acc; p[14] = (uint32_t)g_c_widen;
        p[15] = (uint32_t)g_c_requant; p[16] = (uint32_t)g_c_zero;
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
#endif  /* GDN_BR_NO_QHPI */
