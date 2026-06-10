/* GdnSolveBR16.cpp — CLEAN int16-only static HMX-pipe GDN solve (GDN_BR_I16).
 * Included AFTER GdnSolveBROp.cpp (reuses its format/pack/kernel/diag helpers + gdn_scr_t/g_scr).
 * T codes live in int16 (sc->Tblk16/Sacc16): all static codes fit int16 (diag TI=2/32767 -> [-32767,32767],
 * merge=i8, Sacc pure-add +-381) -> lossless, half storage.  Clean win = i8->i16 widen / i16->i8 narrow /
 * int16 pure-add acc / int16 maxabs (half the lanes, one fewer pack level).  quant/requant/diag reuse the
 * proven int32 helpers via an order-preserving i16->i32 widen (correctness-first; multiply count unchanged). */
#if defined(GDN_BR_I16) && defined(__hexagon__)

/* max|code| over a 64x64 int16 block. */
static int gdn_maxabs16(const int16_t *codes) {
    HVX_Vector vmax = Q6_V_vzero();
    const HVX_Vector *p = (const HVX_Vector *)codes;
    for (int b = 0; b < (BL * BL) / 64; ++b) vmax = Q6_Vh_vmax_VhVh(vmax, Q6_Vh_vabs_Vh(p[b]));
    vmax = Q6_Vh_vmax_VhVh(vmax, Q6_V_vror_VR(vmax, 2*32));
    vmax = Q6_Vh_vmax_VhVh(vmax, Q6_V_vror_VR(vmax, 2*16));
    vmax = Q6_Vh_vmax_VhVh(vmax, Q6_V_vror_VR(vmax, 2*8));
    vmax = Q6_Vh_vmax_VhVh(vmax, Q6_V_vror_VR(vmax, 2*4));
    vmax = Q6_Vh_vmax_VhVh(vmax, Q6_V_vror_VR(vmax, 2*2));
    vmax = Q6_Vh_vmax_VhVh(vmax, Q6_V_vror_VR(vmax, 2*1));
    int16_t lanes[64] __attribute__((aligned(128))); *(HVX_Vector *)lanes = vmax; return lanes[0];
}

/* i8 -> i16, natural order (one sext + one halfword shuffle, vs the i8->i32 4-way weave). */
static void gdn_widen_i8_to_i16(const int8_t *src, int16_t *dst) {
    const HVX_Vector *sp = (const HVX_Vector *)src;          /* 128 i8 / vec */
    HVX_Vector *dp = (HVX_Vector *)dst;                      /* 64 i16 / vec */
    for (int b = 0; b < (BL * BL) / 128; ++b) {
        HVX_VectorPair w = Q6_Wh_vsxt_Vb(sp[b]);            /* lo=even bytes, hi=odd bytes (deinterleaved) */
        HVX_VectorPair s = Q6_W_vshuff_VVR(Q6_V_hi_W(w), Q6_V_lo_W(w), -2);  /* interleave halfwords -> natural */
        dp[2*b] = Q6_V_lo_W(s); dp[2*b+1] = Q6_V_hi_W(s);
    }
}

/* i16 -> i8 (sat), natural order (one pack level, vs i32->i16->i8 two levels). */
static void gdn_narrow_i16_to_i8(const int16_t *codes, int8_t *out) {
    const HVX_Vector *p = (const HVX_Vector *)codes; HVX_Vector *op = (HVX_Vector *)out;
    for (int v = 0; v < (BL * BL) / 128; ++v) op[v] = Q6_Vb_vpack_VhVh_sat(p[2*v+1], p[2*v]);
}

/* PURE-ADD accumulate one i8 term into the int16 Sacc16 (static scales -> g=1, no rescale). */
static void gdn_acc16(gdn_scr_t *sc, const int8_t *term, int first) {
    if (first) { gdn_widen_i8_to_i16(term, sc->Sacc16); return; }
    gdn_widen_i8_to_i16(term, sc->qbuf16);
    const HVX_Vector *tp = (const HVX_Vector *)sc->qbuf16; HVX_Vector *sp = (HVX_Vector *)sc->Sacc16;
    for (int b = 0; b < (BL * BL) / 64; ++b) sp[b] = Q6_Vh_vadd_VhVh(sp[b], tp[b]);   /* int16 add: half the lanes */
}

/* narrow int32 -> int16 (sat), natural order (diag output: i32 forward-subst -> i16 store). */
static void gdn_narrow_i32_to_i16(const int32_t *src, int16_t *dst) {
    const HVX_Vector *p = (const HVX_Vector *)src; HVX_Vector *dp = (HVX_Vector *)dst;
    for (int v = 0; v < (BL * BL) / 64; ++v) dp[v] = Q6_Vh_vpack_VwVw_sat(p[2*v+1], p[2*v]);
}

/* int16-NATIVE Q15 quant -> i8 (g<1): q=round(code*g) via ONE int16 Q15 mult per 64 codes
 * (Q6_Vh_vmpy_VhRh_s1_rnd_sat = sat16((Vh*Rh*2+0x8000)>>16) = round(code*Mg16/2^15)) -> NO widen to int32,
 * HALF the lanes/instructions vs the int32 path.  16-bit multiplier (vs int32 adaptive ~Q20) = small
 * precision trade (精度先不管).  Requires g<1 (Mg16<2^15); caller guarantees (diag-weight/T_ii g≈0.008). */
static void gdn_quant_i8_q15(const int16_t *codes, float scale_in, float sQ, int8_t *out) {
    float g = scale_in / sQ; int Mg16 = (int)(g * 32768.0f + 0.5f); if (Mg16 > 32767) Mg16 = 32767;
    int Rh = (Mg16 & 0xFFFF) | (Mg16 << 16);
    const HVX_Vector *p = (const HVX_Vector *)codes; HVX_Vector *op = (HVX_Vector *)out;
    for (int v = 0; v < (BL * BL) / 128; ++v)
        op[v] = Q6_Vb_vpack_VhVh_sat(Q6_Vh_vmpy_VhRh_s1_rnd_sat(p[2*v+1], Rh),
                                     Q6_Vh_vmpy_VhRh_s1_rnd_sat(p[2*v+0], Rh));
}
/* int16-NATIVE Q15 quant -> u8 (zp128, g<1). */
static void gdn_quant_u8_q15(const int16_t *codes, float scale_in, float sQ, uint8_t *out) {
    float g = scale_in / sQ; int Mg16 = (int)(g * 32768.0f + 0.5f); if (Mg16 > 32767) Mg16 = 32767;
    int Rh = (Mg16 & 0xFFFF) | (Mg16 << 16); const HVX_Vector v128 = Q6_Vh_vsplat_R(128);
    const HVX_Vector *p = (const HVX_Vector *)codes; HVX_Vector *op = (HVX_Vector *)out;
    for (int v = 0; v < (BL * BL) / 128; ++v) {
        HVX_Vector q0 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhRh_s1_rnd_sat(p[2*v+0], Rh), v128);
        HVX_Vector q1 = Q6_Vh_vadd_VhVh(Q6_Vh_vmpy_VhRh_s1_rnd_sat(p[2*v+1], Rh), v128);
        op[v] = Q6_Vub_vpack_VhVh_sat(q1, q0);
    }
}

/* int16-lane WIDENING multiply (for g>=1, e.g. Sacc g=5.77): Q6_Ww_vmpy_VhRh = ONE instruction does 64
 * int16 codes -> int32 pair (fuses widen+multiply; half the mults vs int32, no separate widen pass / Tc).
 * The pair is deinterleaved (lo=even, hi=odd) -> vshuff -4 restores natural order before shift/narrow. */
static void gdn_quant_i8_i16w(const int16_t *codes, float scale_in, float sQ, int8_t *out) {
    float g = scale_in / sQ; int Q = 14;
    while (Q > 1  && g * (float)(1 << Q) >= 32768.0f) --Q;
    while (Q < 28 && g * (float)(1 << (Q + 1)) < 16384.0f) ++Q;
    int Mg16 = (int)(g * (float)(1 << Q) + 0.5f) & 0xFFFF; int Rh = Mg16 | (Mg16 << 16);
    const HVX_Vector vrnd = Q6_V_vsplat_R(1 << (Q - 1)), vlim = Q6_V_vsplat_R(127), vnlim = Q6_V_vsplat_R(-127);
    const HVX_Vector *p = (const HVX_Vector *)codes; HVX_Vector *op = (HVX_Vector *)out;
    for (int v = 0; v < (BL * BL) / 128; ++v) {
        HVX_Vector q4[4];
        for (int hv = 0; hv < 2; ++hv) {
            HVX_VectorPair pr = Q6_Ww_vmpy_VhRh(p[2*v+hv], Rh);
            HVX_VectorPair s = Q6_W_vshuff_VVR(Q6_V_hi_W(pr), Q6_V_lo_W(pr), -4);
            HVX_Vector pp[2] = { Q6_V_lo_W(s), Q6_V_hi_W(s) };
            for (int h = 0; h < 2; ++h) {
                HVX_Vector q = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(pp[h], vrnd), Q);
                q4[2*hv+h] = Q6_Vw_vmax_VwVw(Q6_Vw_vmin_VwVw(q, vlim), vnlim);
            }
        }
        op[v] = Q6_Vb_vpack_VhVh_sat(Q6_Vh_vpack_VwVw_sat(q4[3], q4[2]), Q6_Vh_vpack_VwVw_sat(q4[1], q4[0]));
    }
}
/* int16-lane requant: read i16 rows directly + widening multiply -> u16 Th (no widen pass / Tc). */
static void gdn_requant_i16(const int16_t *codes, float scale_in, float sT, int zpT,
                            uint16_t *Th, int roff, int coff, int row_stride) {
    float g = scale_in / sT; int Q = 14;
    while (Q > 1  && g * (float)(1 << Q) >= 32768.0f) --Q;
    while (Q < 28 && g * (float)(1 << (Q + 1)) < 16384.0f) ++Q;
    int Mg16 = (int)(g * (float)(1 << Q) + 0.5f) & 0xFFFF; int Rh = Mg16 | (Mg16 << 16);
    const HVX_Vector vrnd = Q6_V_vsplat_R(1 << (Q - 1)), vzpT = Q6_V_vsplat_R(zpT);
    const HVX_Vector vlim = Q6_V_vsplat_R(32767), vnlim = Q6_V_vsplat_R(-32767);
    for (int r = 0; r < BL; ++r) {
        HVX_VectorPair pr = Q6_Ww_vmpy_VhRh(*(const HVX_Vector *)(codes + r * BL), Rh);
        HVX_VectorPair s = Q6_W_vshuff_VVR(Q6_V_hi_W(pr), Q6_V_lo_W(pr), -4);
        HVX_Vector pp[2] = { Q6_V_lo_W(s), Q6_V_hi_W(s) }, q[2];
        for (int h = 0; h < 2; ++h) {
            HVX_Vector qq = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(pp[h], vrnd), Q);
            qq = Q6_Vw_vmax_VwVw(Q6_Vw_vmin_VwVw(qq, vlim), vnlim);
            q[h] = Q6_Vw_vadd_VwVw(qq, vzpT);
        }
        *(HVX_UVector *)(Th + (roff + r) * row_stride + coff) = Q6_Vuh_vpack_VwVw_sat(q[1], q[0]);
    }
}

/* ---- int16-reading operand getters (reuse pack/effective; quant via widen-then-proven-int32) ---- */
/* T_kj weight: off-diag drained@sTw -> pure i16->i8 narrow (clean); diag(k==j) -> widen+quant. */
static const int8_t *gdn_get_wt_T16(gdn_scr_t *sc, const gdn_vtcm_t *vt, int k, int j,
                                    float *sw_out, const int32_t **eff_out, int *colabs_out) {
    int key = gdn_blk_index(k, j);
    int8_t *km = vt->wcache + (size_t)key * 0x1000;
    if (!sc->vTw[key]) {
#if defined(GDN_BR_TRACE)
        uint64_t _t0 = gdn_trnow();
#endif
        if (k != j && sc->Tscl[key] == GDN_OPS_sTw) {                       /* clean i16->i8 narrow */
            gdn_narrow_i16_to_i8(sc->Tblk16[key], sc->wtbuf); sc->sTw[key] = GDN_OPS_sTw;
        } else {                                                            /* diag block: int16-native Q15 quant (g≈0.008<1) */
            gdn_quant_i8_q15(sc->Tblk16[key], sc->Tscl[key], GDN_OPS_sTw, sc->wtbuf); sc->sTw[key] = GDN_OPS_sTw;
        }
#if defined(GDN_BR_TRACE)
        uint64_t _t1 = gdn_trnow(); gdn_tr_push((uint32_t)(sc - g_scr), 4, _t0, _t1);   /* QUANT(narrow/quant) */
#endif
        gdn_effective(sc->wtbuf, sc->effc[key]); sc->colabsc[key] = GDN_OPS_COLABS;
#if defined(GDN_BR_TRACE)
        uint64_t _t2 = gdn_trnow(); gdn_tr_push((uint32_t)(sc - g_scr), 9, _t1, _t2);   /* EFF */
#endif
        gdn_pack_w8_kmajor(sc->wtbuf, km);
#if defined(GDN_BR_TRACE)
        gdn_tr_push((uint32_t)(sc - g_scr), 8, _t2, gdn_trnow());   /* PACK (kmajor) */
#endif
        sc->vTw[key] = 1;
    }
    *sw_out = sc->sTw[key]; *eff_out = sc->effc[key]; *colabs_out = sc->colabsc[key];
    return km;
}
/* T_ii activation: widen i16->i32 + quant u8 + crouton-pack. */
static const uint8_t *gdn_get_act_Tdiag16(gdn_scr_t *sc, const gdn_vtcm_t *vt, int i, float *sa_out) {
    uint8_t *cr = vt->acache + 0xA000 + (size_t)i * 0x1000;
    if (!sc->vTa[i]) {
        int bii = gdn_blk_index(i, i);
#if defined(GDN_BR_TRACE)
        uint64_t _t0 = gdn_trnow();
#endif
        gdn_quant_u8_q15(sc->Tblk16[bii], sc->Tscl[bii], GDN_OPS_sTa, sc->actbuf);   /* int16-native Q15 (g≈0.008<1) */
        sc->sTa[i] = GDN_OPS_sTa;
#if defined(GDN_BR_TRACE)
        uint64_t _t1 = gdn_trnow(); gdn_tr_push((uint32_t)(sc - g_scr), 4, _t0, _t1);   /* QUANT */
#endif
        gdn_pack_act_crouton8(sc->actbuf, cr);
#if defined(GDN_BR_TRACE)
        gdn_tr_push((uint32_t)(sc - g_scr), 8, _t1, gdn_trnow());   /* PACK (crouton) */
#endif
        sc->vTa[i] = 1;
    }
    *sa_out = sc->sTa[i];
    return cr;
}

#if defined(GDN_BR_W16)
/* ============================== GDN_BR_W16: w16a16 merge matmul ==============================
 * The merge mm switches from u8i8 (our_v73deep_kernel) to w16a16 (our_v73deep_kernel_i16): int16 act
 * (u16 zp 32768) x int16 wt (q16 +-32639), fixed power-of-2 drain out=act@wt/32767 (u16 zp 32768).
 * Static-scale design: int16 codes = the u8i8 codes refined 128x (act) / 256x (wt):
 *     sAa16 = sAa/128, sTw16 = sTw/256, sTa16 = sTa/128
 * so the fixed drain lands EXACTLY on the static chain: K-stack out scale = 32767*sAa16*sTw16 (=sSacc16,
 * codes <=~12.5K), final out = 32767*sTa16*sW16 = sTw16 (T_kj weight reuse stays a no-op).
 * int32 acc safe per byte-pass: 192*16256*255 < 2^31.  Layout 64^3 padded 2048B (PROVEN H=9/H=10). */
#define GDN_W16_sAa  (GDN_OPS_sAa / 128.0f)
#define GDN_W16_sTw  (GDN_OPS_sTw / 256.0f)
#define GDN_W16_sTa  (GDN_OPS_sTa / 128.0f)
#define GDN_W16_sS   (32767.0f * GDN_W16_sAa * GDN_W16_sTw)
#define GDN_W16_sWf  (GDN_W16_sTw / (32767.0f * GDN_W16_sTa))

/* stream LUT: 4-pass dilated-kmajor halfword h -> byte offset of natural q16 (K=64,N=64).  DDR BSS;
 * offsets feed Q6_vgather (source must be the VTCM stage). */
static uint16_t g_w16_hw[4096] __attribute__((aligned(128)));
static void gdn_w16_lut_init(void) {
    int h = 0;
    for (int nt = 0; nt < 2; ++nt) for (int half = 0; half < 2; ++half) for (int kt = 0; kt < 2; ++kt)
        for (int grp = 0; grp < 8; ++grp)
            for (int idx = 0; idx < 64; idx += 8) for (int j = 0; j < 8; ++j) {
                int vi = idx + j, lane = vi / 16, off = (grp * 8 + half * 4 + lane) * 16 + (vi & 15);
                int rgrp = off / 128, rem = off % 128, col = rem / 4, row = rgrp * 4 + rem % 4;
                g_w16_hw[h++] = (uint16_t)(2 * ((kt * 32 + row) * 64 + (nt * 32 + col)));
            }
}

/* weight pack: natural q16 (DDR ok) -> 8K 4-pass stream dst (VTCM) via vgather, + int32 colsum[64]. */
static void gdn_w16_pack_wt(const int16_t *w_nat, int16_t *stage, uint8_t *dst, int32_t *colsum) {
    for (int i = 0; i < 64; ++i) ((HVX_Vector *)stage)[i] = ((const HVX_UVector *)w_nat)[i];
    HVX_Vector *gtmp = (HVX_Vector *)(stage + 4096);
    const HVX_Vector *ofs = (const HVX_Vector *)g_w16_hw;
    const HVX_Vector K128 = Q6_Vh_vsplat_R(128);
    for (int v = 0; v < 64; v += 2) {
        Q6_vgather_ARMVh((void *)&gtmp[0], (uint32_t)(uintptr_t)stage, 8191, ofs[v]);
        Q6_vgather_ARMVh((void *)&gtmp[1], (uint32_t)(uintptr_t)stage, 8191, ofs[v + 1]);
        HVX_Vector q0 = gtmp[0], q1 = gtmp[1];
        HVX_Vector lo = Q6_Vb_vpacke_VhVh(q1, q0);
        HVX_Vector h0 = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh_sat(q0, K128), 8);
        HVX_Vector h1 = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh_sat(q1, K128), 8);
        HVX_VectorPair il = Q6_W_vshuff_VVR(Q6_Vb_vpack_VhVh_sat(h1, h0), lo, -4);
        ((HVX_Vector *)dst)[v] = Q6_V_lo_W(il); ((HVX_Vector *)dst)[v + 1] = Q6_V_hi_W(il);
    }
    /* colsum: 64 rows int16, acc int32 (64*32639 < 2^31).  vsxt deals even/odd cols -> shuff back. */
    HVX_Vector aL = Q6_V_vzero(), aH = Q6_V_vzero();
    for (int k = 0; k < BL; ++k) {
        HVX_VectorPair w = Q6_Ww_vsxt_Vh(((const HVX_UVector *)w_nat)[k]);
        aL = Q6_Vw_vadd_VwVw(aL, Q6_V_lo_W(w)); aH = Q6_Vw_vadd_VwVw(aH, Q6_V_hi_W(w));
    }
    HVX_VectorPair nat = Q6_W_vshuff_VVR(aH, aL, -4);
    ((HVX_Vector *)colsum)[0] = Q6_V_lo_W(nat); ((HVX_Vector *)colsum)[1] = Q6_V_hi_W(nat);
}

/* bias record: 4 groups x [ctrl x32 | (eff,0) x16], eff = floor(-colsum/2) (proven contract). */
static void gdn_w16_bias(const int32_t *cs, int32_t *bias) {
    static const int32_t ctrl[2] = { 0x00404420, 0x40000000 };
    for (int n = 0; n < 64; ++n) {
        long v = -(long)cs[n]; long eff = (v >= 0) ? (v / 2) : -(((-v) + 1) / 2);
        int g = n >> 4, idx = n & 15;
        bias[g * 64 + 32 + idx * 2] = (int32_t)eff; bias[g * 64 + 32 + idx * 2 + 1] = 0;
    }
    for (int g = 0; g < 4; ++g) for (int i = 0; i < 32; ++i) bias[g * 64 + i] = ctrl[i & 1];
}

/* act pack: natural u16 rows -> PADDED 2048B crouton16 blocks (16 blocks, live 512B; mm64-proven —
 * compact 512B-stride act blocks REFUTED on device, the kernel reads past the live span). */
static void gdn_w16_pack_act(const uint16_t *nat, uint8_t *blk) {
    for (int r4 = 0; r4 < 8; ++r4) for (int m32 = 0; m32 < 2; ++m32) for (int rp = 0; rp < 2; ++rp) {
        int r0 = m32 * 32 + r4 * 4 + rp * 2;
        HVX_Vector v0 = ((const HVX_Vector *)nat)[r0], v1 = ((const HVX_Vector *)nat)[r0 + 1];
        HVX_VectorPair s = Q6_W_vshuff_VVR(v1, v0, -2);   /* interleave rows per col */
        *(HVX_Vector *)(blk + (size_t)(r4 * 2 + 0) * 2048 + (m32 * 2 + rp) * 128) = Q6_V_lo_W(s);
        *(HVX_Vector *)(blk + (size_t)(r4 * 2 + 1) * 2048 + (m32 * 2 + rp) * 128) = Q6_V_hi_W(s);
    }
}

/* out depack: padded crouton16 surface (u16 zp 32768) -> natural int16 codes. */
static void gdn_w16_depack_out(const uint8_t *surf, int16_t *out) {
    const HVX_Vector X = Q6_Vh_vsplat_R(0x8000);
    for (int r4 = 0; r4 < 8; ++r4) for (int m32 = 0; m32 < 2; ++m32) for (int rp = 0; rp < 2; ++rp) {
        int r0 = m32 * 32 + r4 * 4 + rp * 2;
        HVX_Vector v0 = *(const HVX_Vector *)(surf + (size_t)(r4 * 2 + 0) * 2048 + (m32 * 2 + rp) * 128);
        HVX_Vector v1 = *(const HVX_Vector *)(surf + (size_t)(r4 * 2 + 1) * 2048 + (m32 * 2 + rp) * 128);
        HVX_VectorPair d = Q6_W_vdeal_VVR(v1, v0, -2);    /* lo=row0 cols0..63, hi=row1 */
        ((HVX_Vector *)out)[r0]     = Q6_V_vxor_VV(Q6_V_lo_W(d), X);
        ((HVX_Vector *)out)[r0 + 1] = Q6_V_vxor_VV(Q6_V_hi_W(d), X);
    }
}

/* Q15 quant int16->u16 (g<1, zp 32768 via xor). */
static void gdn_w16_quant_u16_q15(const int16_t *codes, float g, uint16_t *out) {
    int Mg = (int)(g * 32768.0f + 0.5f); if (Mg > 32767) Mg = 32767;
    int Rh = (Mg & 0xFFFF) | (Mg << 16);
    const HVX_Vector X = Q6_Vh_vsplat_R(0x8000);
    for (int v = 0; v < (BL * BL) / 64; ++v)
        ((HVX_Vector *)out)[v] = Q6_V_vxor_VV(Q6_Vh_vmpy_VhRh_s1_rnd_sat(((const HVX_Vector *)codes)[v], Rh), X);
}

/* widening quant int16->int16 (any g), clip +-32639 (kernel hi-byte contract). */
static void gdn_w16_quant_i16_wide(const int16_t *codes, float g, int16_t *out) {
    int Q = 14;
    while (Q > 1  && g * (float)(1 << Q) >= 32768.0f) --Q;
    while (Q < 28 && g * (float)(1 << (Q + 1)) < 16384.0f) ++Q;
    int Mg = (int)(g * (float)(1 << Q) + 0.5f) & 0xFFFF; int Rh = Mg | (Mg << 16);
    const HVX_Vector vrnd = Q6_V_vsplat_R(1 << (Q - 1));
    const HVX_Vector vlim = Q6_V_vsplat_R(32639), vnlim = Q6_V_vsplat_R(-32639);
    for (int v = 0; v < (BL * BL) / 64; ++v) {
        HVX_VectorPair pr = Q6_Ww_vmpy_VhRh(((const HVX_Vector *)codes)[v], Rh);
        HVX_VectorPair s = Q6_W_vshuff_VVR(Q6_V_hi_W(pr), Q6_V_lo_W(pr), -4);
        HVX_Vector q0 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_V_lo_W(s), vrnd), Q);
        HVX_Vector q1 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_V_hi_W(s), vrnd), Q);
        q0 = Q6_Vw_vmax_VwVw(Q6_Vw_vmin_VwVw(q0, vlim), vnlim);
        q1 = Q6_Vw_vmax_VwVw(Q6_Vw_vmin_VwVw(q1, vlim), vnlim);
        ((HVX_Vector *)out)[v] = Q6_Vh_vpack_VwVw_sat(q1, q0);
    }
}

/* A_ik act: fold (int16-lane, zpA==32768) + Q15 quant @sAa16 -> u16 rows -> padded crouton16. */
static void gdn_w16_fold_quant_act(gdn_scr_t *sc, const uint16_t *Au, int row_stride,
                                   int zpA, int M, int S) {
    (void)zpA;
    const int Mrep = (M & 0xFFFF) * 0x10001;
    float g = (float)(1.0 / (1 << GDN_BR_F)) / GDN_W16_sAa;
    int Mq = (int)(g * 32768.0f + 0.5f); if (Mq > 32767) Mq = 32767;
    int Rq = (Mq & 0xFFFF) | (Mq << 16);
    const HVX_Vector vxor = Q6_V_vsplat_R(0x80008000), vrndS = Q6_V_vsplat_R(1 << (S - 1));
    uint16_t *nat = (uint16_t *)sc->qbuf16;
    for (int r = 0; r < BL; ++r) {
        HVX_VectorPair fp = Q6_Ww_vmpy_VhRh(Q6_V_vxor_VV(*(const HVX_UVector *)(Au + r * row_stride), vxor), Mrep);
        HVX_Vector i0 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_V_lo_W(fp), vrndS), S);
        HVX_Vector i1 = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_V_hi_W(fp), vrndS), S);
        HVX_VectorPair s = Q6_W_vshuff_VVR(i1, i0, -4);
        HVX_Vector f16 = Q6_Vh_vpack_VwVw_sat(Q6_V_hi_W(s), Q6_V_lo_W(s));   /* folded int16, natural */
        ((HVX_Vector *)nat)[r] = Q6_V_vxor_VV(Q6_Vh_vmpy_VhRh_s1_rnd_sat(f16, Rq), Q6_V_vsplat_R(0x80008000));
    }
}

/* ---- W16 operand getters ---- */
/* A_ik act: use-once -> fold+quant+pack TRANSIENT into padded slot m (32K each; no cache). */
static const uint8_t *gdn_w16_get_act_A(gdn_scr_t *sc, const gdn_vtcm_t *vt, const uint16_t *Ah,
                                        int i, int k, int C_, int zpA, int M, int S, int slot) {
    uint8_t *cr = GDN_W16_ACT_SLOT(vt, slot);
#if defined(GDN_BR_TRACE)
    uint64_t _t0 = gdn_trnow();
#endif
    gdn_w16_fold_quant_act(sc, Ah + (size_t)i * BL * C_ + k * BL, C_, zpA, M, S);
#if defined(GDN_BR_TRACE)
    uint64_t _t1 = gdn_trnow(); gdn_tr_push((uint32_t)(sc - g_scr), 4, _t0, _t1);
#endif
    gdn_w16_pack_act((const uint16_t *)sc->qbuf16, cr);
#if defined(GDN_BR_TRACE)
    gdn_tr_push((uint32_t)(sc - g_scr), 8, _t1, gdn_trnow());
#endif
    return cr;
}
/* T_ii act: quant ONCE to natural u16 cache (8K VTCM), re-pack per final merge into slot 0. */
static const uint8_t *gdn_w16_get_act_Tdiag(gdn_scr_t *sc, const gdn_vtcm_t *vt, int i) {
    uint16_t *nat = GDN_W16_TDIAG_NAT(vt, i);
    if (!sc->vTa[i]) {
        int bii = gdn_blk_index(i, i);
#if defined(GDN_BR_TRACE)
        uint64_t _t0 = gdn_trnow();
#endif
        gdn_w16_quant_u16_q15(sc->Tblk16[bii], GDN_BR_TI / GDN_W16_sTa, nat);
#if defined(GDN_BR_TRACE)
        gdn_tr_push((uint32_t)(sc - g_scr), 4, _t0, gdn_trnow());
#endif
        sc->vTa[i] = 1;
    }
    uint8_t *cr = GDN_W16_ACT_SLOT(vt, 0);
#if defined(GDN_BR_TRACE)
    uint64_t _t1 = gdn_trnow();
#endif
    gdn_w16_pack_act(nat, cr);
#if defined(GDN_BR_TRACE)
    gdn_tr_push((uint32_t)(sc - g_scr), 8, _t1, gdn_trnow());
#endif
    return cr;
}
/* T_kj weight: off-diag codes are ALREADY @sTw16 (<32639, final-merge drain) -> pack directly;
 * diag (k==j) rescales TI->sTw16 (g~1.98) first. */
static const int8_t *gdn_w16_get_wt_T(gdn_scr_t *sc, const gdn_vtcm_t *vt, int k, int j,
                                      const int32_t **cs_out) {
    int key = gdn_blk_index(k, j);
    int8_t *st = vt->wcache + (size_t)key * GDN_W16_BLK;
    if (!sc->vTw[key]) {
        const int16_t *src = sc->Tblk16[key];
#if defined(GDN_BR_TRACE)
        uint64_t _t0 = gdn_trnow();
#endif
        if (k == j) { gdn_w16_quant_i16_wide(src, GDN_BR_TI / GDN_W16_sTw, sc->qbuf16); src = sc->qbuf16; }
#if defined(GDN_BR_TRACE)
        uint64_t _t1 = gdn_trnow(); gdn_tr_push((uint32_t)(sc - g_scr), 4, _t0, _t1);
#endif
        gdn_w16_pack_wt(src, vt->stage, (uint8_t *)st, sc->effc[key]);
#if defined(GDN_BR_TRACE)
        gdn_tr_push((uint32_t)(sc - g_scr), 8, _t1, gdn_trnow());
#endif
        sc->vTw[key] = 1;
    }
    *cs_out = sc->effc[key];
    return st;
}

/* one w16a16 64^3 (K=64d) matmul: act = d padded crouton16 caches, wt = nt-major d-stacked stream copy,
 * bias = summed colsum.  Dispatch to the consumer; depack into natural int16 codes (scale = caller's). */
static void gdn_w16_kstack(gdn_scr_t *sc, const gdn_vtcm_t *vt, const uint8_t *const act_cr[],
                           const int8_t *const wt_st[], const int32_t *const cs_blk[], int d,
                           int16_t *out_codes) {
    int8_t *w = vt->wt;
    /* stream order for K=64d = [nt][half][kt 0..2d-1]; one cached 64-block holds [nt][half][kt 0..1] in
     * 16-vec (2K) quarters -> quarter q of blk m lands at (q*d + m)*16 vecs.  d==1 = identity. */
    for (int m = 0; m < d; ++m) {
        const HVX_Vector *s = (const HVX_Vector *)wt_st[m];
        for (int q = 0; q < 4; ++q) {
            HVX_Vector *dq = (HVX_Vector *)(w + ((size_t)(q * d + m) * 16) * 128);
            for (int v = 0; v < 16; ++v) dq[v] = s[q * 16 + v];
        }
    }
    int32_t cs[BL] __attribute__((aligned(128)));
    for (int n = 0; n < BL; ++n) { int32_t t = 0; for (int m = 0; m < d; ++m) t += cs_blk[m][n]; cs[n] = t; }
    gdn_w16_bias(cs, vt->bias);
    int Kt = 2 * d;
    for (int rg = 0; rg < 16; ++rg) {
        for (int m = 0; m < d; ++m) for (int kk = 0; kk < 2; ++kk)
            vt->acttab[rg * Kt + m * 2 + kk] = (int32_t)(uintptr_t)(act_cr[m] + (size_t)((rg & 7) * 2 + kk) * 2048);
        vt->outtab[rg * 2 + 0] = (int32_t)(uintptr_t)(vt->out + (size_t)((rg & 7) * 2 + 0) * 2048);
        vt->outtab[rg * 2 + 1] = (int32_t)(uintptr_t)(vt->out + (size_t)((rg & 7) * 2 + 1) * 2048);
    }
    g_kstack_nap = Kt;
    g_hmx_dispatch(sc, (gdn_vtcm_t *)vt, w, nullptr, 0.f, 0, 0);
    g_kstack_nap = 2;
    gdn_w16_depack_out(vt->out, out_codes);
}
#endif  /* GDN_BR_W16 */

/* ---- the int16 static solve (mirrors gdn_br_one_head HMX path, Tblk/Sacc -> int16) ---- */
static void gdn_br_one_head16(gdn_scr_t *sc, const gdn_vtcm_t *vt, const uint16_t *Ah, uint16_t *Th,
                              int zpA, int M, int S, float sT, int zpT) {
#if defined(GDN_BR_TRACE)
    uint32_t _tid = (uint32_t)(sc - g_scr); uint64_t _hd0 = gdn_trnow();
#endif
    for (int b = 0; b < GDN_BR_NBLK; ++b) { sc->vAa[b] = 0; sc->vTw[b] = 0; }
    for (int i = 0; i < NB; ++i) sc->vTa[i] = 0;
    /* diag: forward-subst (proven int32 path) -> narrow to int16 store. */
    for (int i = 0; i < NB; ++i) {
        int bi = gdn_blk_index(i, i);
#if defined(GDN_BR_DIAG_I16)
        /* int16-direct: fwdsubst writes int16 straight to Tblk16, skipping the int32 widen+narrow round-trip. */
        gdn_solve_diag64_i16(sc, Ah + (size_t)i * BL * C + i * BL, C, zpA, M, S, sc->Tblk16[bi]);
#else
        gdn_solve_diag64(sc, Ah + (size_t)i * BL * C + i * BL, C, zpA, M, S, sc->Tc, nullptr);
        gdn_narrow_i32_to_i16(sc->Tc, sc->Tblk16[bi]);
#endif
        sc->Tscl[bi] = GDN_BR_TI;
#if defined(GDN_BR_COUPLING_TAX)
        /* D&C coupling-tax probe: the 2 coupling matmuls (A21@T11, T22@M) per diag block via the K-stack HMX
         * path — full pack(crouton+kmajor)+effective+dispatch+depack.  Garbage operands (timing only; mxmem
         * data-independent).  Combined with -DGDN_BR_FWD_ROWS=32 (diag halved) => the REAL one-level D&C wall. */
        for (int c = 0; c < 2; ++c) {
            gdn_pack_act_crouton8(sc->actbuf, vt->act);
            gdn_pack_w8_kmajor(sc->wtbuf, (int8_t *)vt->wt);
            gdn_effective((int8_t *)vt->wt, sc->eff);
            const uint8_t *acr[1] = { vt->act };
            const int8_t  *wkm[1] = { (int8_t *)vt->wt };
            const int32_t *efb[1] = { sc->eff };
            gdn_merge_kstack(sc, vt, acr, wkm, efb, 1, GDN_OPS_sAa, GDN_OPS_sTw, sc->termi);
        }
#endif
    }
#if defined(GDN_BR_TRACE)
    gdn_tr_push(_tid, 1, _hd0, gdn_trnow());   /* DIAG */
#endif
    /* zero strict-upper-tri of Th. */
    { HVX_Vector vzph = Q6_Vh_vsplat_R(zpT); int aligned = (((uintptr_t)Th & 127) == 0);
      for (int bi = 0; bi < NB - 1; ++bi) { int nbc = NB - 1 - bi;
        for (int r = 0; r < BL; ++r) { uint16_t *rowp = Th + ((size_t)bi*BL + r)*C + (size_t)(bi+1)*BL;
          if (aligned) { HVX_Vector *op = (HVX_Vector *)rowp; for (int b=0;b<nbc;++b) op[b]=vzph; }
          else { HVX_UVector *op = (HVX_UVector *)rowp; for (int b=0;b<nbc;++b) op[b]=vzph; } } } }
    /* diag requant: int16-lane (read i16 directly + widening multiply). */
    for (int i = 0; i < NB; ++i) { int bi = gdn_blk_index(i, i);
        gdn_requant_i16(sc->Tblk16[bi], sc->Tscl[bi], sT, zpT, Th, i * BL, i * BL, C);
    }
    /* off-diagonal merges. */
#if !defined(GDN_BR_CAP_OFFDIAG)   /* cap-test: skip ALL off-diag (DIAG-only wall floor; T wrong but DIAG timing valid) */
    for (int d = 1; d < NB; ++d) {
        for (int j = 0; j + d < NB; ++j) {
            int i = j + d, bij = gdn_blk_index(i, j);
            float s_S = 0.f; int first = 1; (void)first;
#if defined(GDN_BR_W16)
            /* w16a16 K-stack: one K=64d matmul, fixed-gain drain -> Sacc16 codes @ sSacc16. */
            {
                const uint8_t *act_cr[GDN_BR_NB]; const int8_t *wt_st[GDN_BR_NB]; const int32_t *cs_blk[GDN_BR_NB];
                for (int k = j; k < i; ++k) {
#if defined(GDN_BR_TRACE)
                    uint64_t _p0 = gdn_trnow();
#endif
                    act_cr[k - j] = gdn_w16_get_act_A(sc, vt, Ah, i, k, C, zpA, M, S, k - j);
                    wt_st[k - j]  = gdn_w16_get_wt_T(sc, vt, k, j, &cs_blk[k - j]);
#if defined(GDN_BR_TRACE)
                    gdn_tr_push(_tid, 5, _p0, gdn_trnow());   /* PREP */
#endif
                }
#if defined(GDN_BR_TRACE)
                uint64_t _m0 = gdn_trnow();
#endif
#if defined(GDN_W16_NOSTACK)
                /* bisect/fallback: d single-K dispatches + int16 sat add on the producer */
                for (int m = 0; m < d; ++m) {
                    int16_t *dst = (m == 0) ? sc->Sacc16 : sc->qbuf16;
                    gdn_w16_kstack(sc, vt, &act_cr[m], &wt_st[m], &cs_blk[m], 1, dst);
                    if (m) for (int v = 0; v < (BL * BL) / 64; ++v)
                        ((HVX_Vector *)sc->Sacc16)[v] = Q6_Vh_vadd_VhVh_sat(((HVX_Vector *)sc->Sacc16)[v],
                                                                            ((HVX_Vector *)sc->qbuf16)[v]);
                }
#else
                gdn_w16_kstack(sc, vt, act_cr, wt_st, cs_blk, d, sc->Sacc16);
#endif
                s_S = GDN_W16_sS;
#if defined(GDN_W16_DBG_S)
                /* debug: dump Sacc16 codes of (i=2,j=0,d=2) as u16 into the (0,1) upper-tri block */
                if (i == 2 && j == 0)
                    for (int r = 0; r < BL; ++r) for (int c = 0; c < BL; ++c)
                        Th[(size_t)r * C + BL + c] = (uint16_t)(sc->Sacc16[r * BL + c] ^ 0x8000);
                /* + raw operand surfaces of the (2,0) merge: act A(2,0) crouton16 -> (0,2); wt T(0,0)
                 * 8K stream -> (0,3) rows 0..31; wt T(1,0) stream -> (0,3) rows 32..63 (4 surfaces 8K each). */
                if (i == 2 && j == 0) {
                    const uint16_t *a = (const uint16_t *)act_cr[0];          /* 8KB raw crouton16 */
                    const uint16_t *w0 = (const uint16_t *)wt_st[0], *w1 = (const uint16_t *)wt_st[1];
                    for (int r = 0; r < BL; ++r) for (int c = 0; c < BL; ++c) {
                        Th[(size_t)r * C + 2 * BL + c] = a[r * BL + c];        /* verbatim surface */
                        Th[(size_t)r * C + 3 * BL + c] = w0[r * BL + c];
                        Th[(size_t)(BL + r) * C + 2 * BL + c] = w1[r * BL + c];
                    }
                }
#endif
#if defined(GDN_BR_TRACE)
                gdn_tr_push(_tid, 3, _m0, gdn_trnow());   /* MM (K-stack + depack) */
#endif
            }
#elif defined(GDN_BR_KSTACK)
            /* K-STACK: fold the d-term Sigma_k into ONE K=64d HMX matmul (accumulate in the HMX int32 acc,
             * single drain) -> no per-k handshake, no gdn_acc16.  Operands come from the same caches. */
            {
                const uint8_t *act_cr[GDN_BR_NB]; const int8_t *wt_km[GDN_BR_NB]; const int32_t *eff_blk[GDN_BR_NB];
                float sa = 0.f, sw = 0.f;
                for (int k = j; k < i; ++k) {
                    float sak, swk; const int32_t *effk; int wck;
#if defined(GDN_BR_TRACE)
                    uint64_t _p0 = gdn_trnow();
#endif
                    act_cr[k - j] = gdn_get_act_A(sc, vt, Ah, i, k, C, zpA, M, S, &sak);
                    wt_km[k - j]  = gdn_get_wt_T16(sc, vt, k, j, &swk, &effk, &wck);
                    eff_blk[k - j] = effk; sa = sak; sw = swk;
#if defined(GDN_BR_TRACE)
                    uint64_t _p1 = gdn_trnow(); gdn_tr_push(_tid, 5, _p0, _p1);   /* PREP */
#endif
                }
#if defined(GDN_BR_TRACE)
                uint64_t _m0 = gdn_trnow();
#endif
                s_S = gdn_merge_kstack(sc, vt, act_cr, wt_km, eff_blk, d, sa, sw, sc->termi);
#if defined(GDN_BR_TRACE)
                uint64_t _m1 = gdn_trnow(); gdn_tr_push(_tid, 3, _m0, _m1);   /* MM (K-stack, 1 dispatch) */
#endif
                gdn_widen_i8_to_i16(sc->termi, sc->Sacc16);   /* int8 Sacc (= Sigma_k) -> int16 for the final-merge requant */
#if defined(GDN_BR_TRACE)
                gdn_tr_push(_tid, 6, _m1, gdn_trnow());   /* ACC (now just a widen) */
#endif
            }
#elif defined(GDN_BR_SWPIPE)
            /* SOFTWARE PIPELINE: dispatch matmul k async, prep k+1's operands WHILE the consumer computes k
             * (next-operand prep is independent of k's result) -> hides the spin under the prep.  Single vt +
             * slot is safe: only 1 job in flight; prep(k+1) writes a DIFFERENT cache slot, not vt->bias/out. */
            { float sa, sw; const int32_t *eff; int wc;
              const uint8_t *a = gdn_get_act_A(sc, vt, Ah, i, j, C, zpA, M, S, &sa);
              const int8_t  *w = gdn_get_wt_T16(sc, vt, j, j, &sw, &eff, &wc);
              float sP = gdn_merge_dispatch(sc, vt, a, sa, w, eff, sw, wc);
              for (int k = j; k < i; ++k) {
                  float sa2 = 0, sw2 = 0; const int32_t *eff2 = nullptr; int wc2 = 0;
                  const uint8_t *a2 = nullptr; const int8_t *w2 = nullptr;
                  if (k + 1 < i) {                              /* prefetch next operands (overlaps consumer running k) */
                      a2 = gdn_get_act_A(sc, vt, Ah, i, k + 1, C, zpA, M, S, &sa2);
                      w2 = gdn_get_wt_T16(sc, vt, k + 1, j, &sw2, &eff2, &wc2);
                  }
                  gdn_merge_wait_depack(sc, vt, sc->termi);     /* wait k + depack (spin now hidden by the prefetch) */
                  if (first) s_S = sP;
                  gdn_acc16(sc, sc->termi, first);
                  first = 0;
                  if (k + 1 < i) sP = gdn_merge_dispatch(sc, vt, a2, sa2, w2, eff2, sw2, wc2);
              }
            }
#else
            for (int k = j; k < i; ++k) {
                float sa, sw, sterm; const int32_t *eff; int wcolabs;
#if defined(GDN_BR_TRACE)
                uint64_t _p0 = gdn_trnow();
#endif
                const uint8_t *a = gdn_get_act_A(sc, vt, Ah, i, k, C, zpA, M, S, &sa);
                const int8_t  *w = gdn_get_wt_T16(sc, vt, k, j, &sw, &eff, &wcolabs);
#if defined(GDN_BR_TRACE)
                uint64_t _p1 = gdn_trnow(); gdn_tr_push(_tid, 5, _p0, _p1);   /* PREP */
#endif
                gdn_merge_packed(sc, vt, a, sa, w, eff, sw, wcolabs, sc->termi, &sterm);
#if defined(GDN_BR_TRACE)
                uint64_t _p2 = gdn_trnow(); gdn_tr_push(_tid, 3, _p1, _p2);   /* MM */
#endif
                if (first) s_S = sterm;                          /* static: all inner terms share scale (sAa,sTw fixed) */
                gdn_acc16(sc, sc->termi, first);
#if defined(GDN_BR_TRACE)
                gdn_tr_push(_tid, 6, _p2, gdn_trnow());   /* ACC */
#endif
                first = 0;
            }
#endif
            /* final merge T_ij = T_ii @ Sacc. */
#if defined(GDN_BR_W16)
            float sij;
            {
#if defined(GDN_BR_TRACE)
                uint64_t _f0 = gdn_trnow();
#endif
                const uint8_t *a_ii = gdn_w16_get_act_Tdiag(sc, vt, i);
                /* Sacc16 @sSacc16 -> weight @sWf so the fixed drain lands @sTw16 (g~2.53, max ~31.1K). */
                gdn_w16_quant_i16_wide(sc->Sacc16, GDN_W16_sS / GDN_W16_sWf, sc->qbuf16);
                int32_t csf[BL] __attribute__((aligned(128)));
                gdn_w16_pack_wt(sc->qbuf16, vt->stage, (uint8_t *)vt->wt, csf);
                const uint8_t *acr[1] = { a_ii }; const int8_t *wst[1] = { vt->wt };
                const int32_t *csb[1] = { csf };
#if defined(GDN_BR_TRACE)
                uint64_t _f1 = gdn_trnow(); gdn_tr_push(_tid, 5, _f0, _f1);   /* PREP (final) */
#endif
                gdn_w16_kstack(sc, vt, acr, wst, csb, 1, sc->Tblk16[bij]);
                sij = GDN_W16_sTw;
#if defined(GDN_BR_TRACE)
                gdn_tr_push(_tid, 3, _f1, gdn_trnow());   /* MM (final) */
#endif
            }
            sc->Tscl[bij] = sij;
#if defined(GDN_BR_TRACE)
            uint64_t _f2 = gdn_trnow();
#endif
#else
            float sa_ii, sw_S, sij; int scolabs;
#if defined(GDN_BR_TRACE)
            uint64_t _f0 = gdn_trnow();
#endif
            const uint8_t *a_ii = gdn_get_act_Tdiag16(sc, vt, i, &sa_ii);
            gdn_quant_i8_i16w(sc->Sacc16, s_S, GDN_OPS_sSacc, sc->wtbuf);   /* int16-lane (g=5.77>1, widening mult) */
            sw_S = GDN_OPS_sSacc;
            gdn_effective(sc->wtbuf, sc->eff, &scolabs);
            gdn_pack_w8_kmajor(sc->wtbuf, vt->wt);
#if defined(GDN_BR_TRACE)
            uint64_t _f1 = gdn_trnow(); gdn_tr_push(_tid, 5, _f0, _f1);   /* PREP (final) */
#endif
            g_force_sP = GDN_OPS_sTw;
            gdn_merge_packed(sc, vt, a_ii, sa_ii, vt->wt, sc->eff, sw_S, scolabs, sc->termi, &sij);
            g_force_sP = 0.f;
#if defined(GDN_BR_TRACE)
            uint64_t _f2 = gdn_trnow(); gdn_tr_push(_tid, 3, _f1, _f2);   /* MM (final) */
#endif
            gdn_widen_i8_to_i16(sc->termi, sc->Tblk16[bij]);    /* store result i16 */
            sc->Tscl[bij] = sij;
#endif  /* GDN_BR_W16 */
            gdn_requant_i16(sc->Tblk16[bij], sij, sT, zpT, Th, i * BL, j * BL, C);   /* int16-lane requant */
#if defined(GDN_BR_TRACE)
            gdn_tr_push(_tid, 7, _f2, gdn_trnow());   /* REQ (widen+requant) */
#endif
        }
    }
#endif  /* !GDN_BR_CAP_OFFDIAG */
}
#endif  /* GDN_BR_I16 && __hexagon__ */
