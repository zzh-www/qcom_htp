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
static float   __attribute__((aligned(128))) g_T11[GDN_BR_BL * GDN_BR_BL];
static float   __attribute__((aligned(128))) g_T22[GDN_BR_BL * GDN_BR_BL];
static float   __attribute__((aligned(128))) g_A21[GDN_BR_BL * GDN_BR_BL];
static int8_t  __attribute__((aligned(128))) g_Mi8 [GDN_BR_BL * GDN_BR_BL];
static int8_t  __attribute__((aligned(128))) g_T21i[GDN_BR_BL * GDN_BR_BL];

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

/* fold one 64x64 block A (uint16 codes, row-stride = C) -> int32 codes Afx (scale 2^-F), into a
 * packed 64*64 buffer.  The folded code is replicated into BOTH halfwords of the int32 so it can be
 * used directly as the scalar Rt for Q6_Vw_vmpyiacc_VwVwRh (which multiplies even word-lanes by Rt.h0
 * and odd word-lanes by Rt.h1 — we want the SAME scalar for every column).  |code|<2^15 fits int16.
 * Scalar (clean + only 4096 elems, dwarfed by the HMX merge). */
static void gdn_fold_block(const uint16_t *Au, int row_stride, int32_t *Afx, int zpA, int M, int S) {
    const int rnd = 1 << (S - 1);
    for (int r = 0; r < BL; ++r)
        for (int c = 0; c < BL; ++c) {
            int code = (int)Au[r * row_stride + c] - zpA;
            int v = ((int64_t)code * M + rnd) >> S;
            uint32_t lo = (uint32_t)(v & 0xFFFF);
            Afx[r * BL + c] = (int32_t)(lo | (lo << 16));
        }
}

/* solve one 64x64 diagonal block: T = inv(I - A_block), int16-code forward subst, dequant to Tf (fp). */
static void gdn_solve_diag64(const uint16_t *Au, int row_stride, int zpA, int M, int S,
                             float *Tf) {
    int32_t *Tc  = g_Tc;
    int32_t *Afx = g_Afx;
    gdn_fold_block(Au, row_stride, Afx, zpA, M, S);
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
    /* dequant int16 codes (scale TI) -> fp. */
    for (int r = 0; r < BL; ++r)
        for (int c = 0; c < BL; ++c)
            Tf[r * BL + c] = (float)Tc[r * BL + c] * GDN_BR_TI;
}

/* note: the Afx fold here pre-replicates? No — gdn_fold_block stores raw int32 (low 16 used as the
 * scalar halfword by vmpyiacc_VwVwRh, which reads Rt.h0 for even lanes / Rt.h1 for odd). We store the
 * SAME value in both halfwords by passing the int directly; vmpyiacc_VwVwRh uses Rt as a scalar where
 * h0,h1 act on even/odd word lanes. Since our T columns are 32-per-vector (word lanes) and the multiplier
 * is the SAME scalar for the whole row, we must replicate code into both halfwords. Do it inline below. */

/* ---- crouton8 activation packer (C transcription of gdn_hmx_matmul_sim.pack_act_crouton8, 64^3) ---- */
static void gdn_pack_act_crouton8(const uint8_t *act_mk, uint8_t *out_buf) {
    /* each k-tile a separate contiguous m*32 tile; within: row8_group(4) x m32_group(2) x row_sub(8) x col_word(8). */
    int out = 0;
    for (int kt = 0; kt < 2; ++kt) {
        int k_base = kt * 32;
        for (int row8_group = 0; row8_group < 4; ++row8_group)
            for (int m32_group = 0; m32_group < 2; ++m32_group)
                for (int row_sub = 0; row_sub < 8; ++row_sub) {
                    int row = m32_group * 32 + row8_group * 8 + row_sub;
                    for (int col_word = 0; col_word < 8; ++col_word) {
                        int col = k_base + col_word * 4;
                        const uint8_t *src = act_mk + row * 64 + col;
                        out_buf[out + 0] = src[0]; out_buf[out + 1] = src[1];
                        out_buf[out + 2] = src[2]; out_buf[out + 3] = src[3];
                        out += 4;
                    }
                }
    }
}

/* ---- k-major weight packer (C transcription of prepare_owned_inputs.pack_w8_kmajor, 64^3) ---- */
static void gdn_pack_w8_kmajor(const int8_t *w_kn, int8_t *packed) {
    int out = 0;
    for (int kt = 0; kt < 2; ++kt) {
        int k_base = kt * 32;
        for (int nt = 0; nt < 2; ++nt) {
            int n_base = nt * 32;
            int8_t *tile = packed + out;   /* 1024 bytes */
            for (int t = 0; t < 1024; ++t) tile[t] = 0;
            for (int r = 0; r < 32; ++r)
                for (int c = 0; c < 32; ++c) {
                    int dst = (r / 4) * 128 + c * 4 + (r % 4);
                    tile[dst] = w_kn[(k_base + r) * 64 + (n_base + c)];
                }
            out += 1024;
        }
    }
}

/* effective[n] = -128*sum_k wt[k,n] + bias_q(0); 64-wide. */
static void gdn_effective(const int8_t *w_kn, int32_t *effective) {
    for (int n = 0; n < 64; ++n) {
        int64_t s = 0;
        for (int k = 0; k < 64; ++k) s += w_kn[k * 64 + n];
        effective[n] = (int32_t)(-128 * s);
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

/* run one signed 64^3 HMX merge: act_u8 (zp128 crouton8 not yet packed), wt_i8 (k-major not yet packed),
 * given as natural [64,64].  Packs into VTCM scratch, runs the kernel, writes recovered int8 codes
 * (out_u8 - 128) into out_codes.  scale_f16/baseline recentre the signed product to u8 zp128. */
static void gdn_hmx_merge(const gdn_vtcm_t *vt, const uint8_t *act_u8, const int8_t *wt_i8,
                          float scale_f16, int baseline_u16, int8_t *out_codes) {
    int32_t eff[64];
    gdn_effective(wt_i8, eff);
    gdn_pack_act_crouton8(act_u8, vt->act);
    gdn_pack_w8_kmajor(wt_i8, vt->wt);
    gdn_pack_bias(eff, scale_f16, baseline_u16, vt->bias);
    for (int i = 0; i < BL * BL; ++i) vt->out[i] = 0;

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

    int base = baseline_u16 >> 7;   /* output zero-point in u8 */
    for (int r = 0; r < BL; ++r)
        for (int c = 0; c < BL; ++c)
            out_codes[r * BL + c] = (int8_t)((int)gdn_depack_out(vt->out, r, c) - base);
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
        /* ---- diagonal solves: T11 = inv(I - A[0:64,0:64]), T22 = inv(I - A[64:,64:]) ---- */
#if defined(GDN_BR_PROBE_CYCLES)
        uint64_t t0; asm volatile("%0 = C15:14" : "=r"(t0));
#endif
        gdn_solve_diag64(Ah, C, zpA, M, S, g_T11);
        gdn_solve_diag64(Ah + BL * C + BL, C, zpA, M, S, g_T22);
        /* A21 dequant to fp */
        for (int r = 0; r < BL; ++r)
            for (int c = 0; c < BL; ++c)
                g_A21[r * BL + c] = ((int)Ah[(BL + r) * C + c] - zpA) * sA;
#if defined(GDN_BR_PROBE_CYCLES)
        uint64_t t1; asm volatile("%0 = C15:14" : "=r"(t1)); c_diag += t1 - t0;
#endif
        /* assemble T11, T22 into output now (lower-tri blocks) */
        for (int i = 0; i < C * C; ++i) Th[i] = (uint16_t)zpT;
        for (int r = 0; r < BL; ++r)
            for (int c = 0; c <= r; c++) {
                long q = lroundf(g_T11[r * BL + c] / sT);
                if (q > 32767) q = 32767; if (q < -32767) q = -32767;
                Th[r * C + c] = (uint16_t)((int)q + zpT);
            }
        for (int r = 0; r < BL; ++r)
            for (int c = 0; c <= r; c++) {
                long q = lroundf(g_T22[r * BL + c] / sT);
                if (q > 32767) q = 32767; if (q < -32767) q = -32767;
                Th[(BL + r) * C + (BL + c)] = (uint16_t)((int)q + zpT);
            }
#if defined(GDN_BR_DIAG_ONLY)
        continue;   /* T21 stays at zpT (=0) -> block-diagonal */
#endif
        /* ---- quantize merge operands ---- */
        float sA21 = gdn_br_qsym_scale(g_A21, BL * BL);
        float sT11 = gdn_br_qsym_scale(g_T11, BL * BL);
        float sT22 = gdn_br_qsym_scale(g_T22, BL * BL);
#if defined(GDN_BR_PROBE_CYCLES)
        uint64_t t2; asm volatile("%0 = C15:14" : "=r"(t2));
#endif
        /* estimate sM = max|A21@T11|/127 via a quick host matmul of the dequant operands. */
        float sM = 1e-12f;
        {
            for (int r = 0; r < BL; ++r)
                for (int c = 0; c < BL; ++c) {
                    float acc = 0.0f;
                    for (int k = 0; k < BL; ++k) acc += g_A21[r * BL + k] * g_T11[k * BL + c];
                    float a = acc < 0 ? -acc : acc; if (a > sM) sM = a;
                }
            sM /= 127.0f; if (sM <= 0.0f) sM = 1e-12f;
        }
        /* build u8 act (A21, zp128) + i8 wt (T11) at their symmetric scales */
        static uint8_t actbuf[BL * BL]; static int8_t wtbuf[BL * BL];
        for (int i = 0; i < BL * BL; ++i) {
            long q = lroundf(g_A21[i] / sA21); if (q > 127) q = 127; if (q < -127) q = -127;
            actbuf[i] = (uint8_t)(q + 128);
        }
        for (int i = 0; i < BL * BL; ++i) {
            long q = lroundf(g_T11[i] / sT11); if (q > 127) q = 127; if (q < -127) q = -127;
            wtbuf[i] = (int8_t)q;
        }
#if defined(GDN_BR_PROBE_CYCLES)
        uint64_t t3; asm volatile("%0 = C15:14" : "=r"(t3)); c_pack += t3 - t2;
#endif
        float gain1 = (sA21 * sT11) / sM;
        gdn_hmx_merge(&vt, actbuf, wtbuf, gain1 * 512.0f, 128 << 7, g_Mi8);
#if defined(GDN_BR_PROBE_CYCLES)
        uint64_t t4; asm volatile("%0 = C15:14" : "=r"(t4)); c_hmx += t4 - t3;
#endif
#if defined(GDN_BR_DUMP_M)
        /* write recovered M codes (int8) into T21 block region for inspection */
        for (int r = 0; r < BL; ++r) for (int c = 0; c < BL; ++c)
            Th[(BL + r) * C + c] = (uint16_t)((int)g_Mi8[r * BL + c] + zpT);
        continue;
#endif
        /* ---- merge2: T21 = T22 @ M (act=T22 zp128, wt=M_i8 at scale sM) ---- */
        float sT21 = 1e-12f;
        {
            for (int r = 0; r < BL; ++r)
                for (int c = 0; c < BL; ++c) {
                    long acc = 0;
                    for (int k = 0; k < BL; ++k) {
                        long aq = lroundf(g_T22[r * BL + k] / sT22); if (aq > 127) aq = 127; if (aq < -127) aq = -127;
                        acc += aq * g_Mi8[k * BL + c];
                    }
                    float v = (float)acc * sT22 * sM; float a = v < 0 ? -v : v; if (a > sT21) sT21 = a;
                }
            sT21 /= 127.0f; if (sT21 <= 0.0f) sT21 = 1e-12f;
        }
        static uint8_t act2[BL * BL];
        for (int i = 0; i < BL * BL; ++i) {
            long q = lroundf(g_T22[i] / sT22); if (q > 127) q = 127; if (q < -127) q = -127;
            act2[i] = (uint8_t)(q + 128);
        }
        float gain2 = (sT22 * sM) / sT21;
        gdn_hmx_merge(&vt, act2, g_Mi8, gain2 * 512.0f, 128 << 7, g_T21i);
        /* dequant + scatter T21 = (code)*sT21 into the lower-left block */
        for (int r = 0; r < BL; ++r)
            for (int c = 0; c < BL; ++c) {
                float v = (float)g_T21i[r * BL + c] * sT21;
                long q = lroundf(v / sT); if (q > 32767) q = 32767; if (q < -32767) q = -32767;
                Th[(BL + r) * C + c] = (uint16_t)((int)q + zpT);
            }
    }
#if defined(GDN_BR_PROBE_CYCLES)
    if (h0 < h1) {
        uint16_t *Th0 = Tu + (size_t)h0 * C * C;
        uint32_t *p = (uint32_t *)Th0;
        p[0] = (uint32_t)c_diag; p[1] = (uint32_t)c_pack; p[2] = (uint32_t)c_hmx; p[3] = (h1 - h0);
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
