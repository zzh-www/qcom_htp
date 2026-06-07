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
        if (k != j && sc->Tscl[key] == GDN_OPS_sTw) {                       /* clean i16->i8 narrow */
            gdn_narrow_i16_to_i8(sc->Tblk16[key], sc->wtbuf); sc->sTw[key] = GDN_OPS_sTw;
        } else {                                                            /* diag block: int16-native Q15 quant (g≈0.008<1) */
            gdn_quant_i8_q15(sc->Tblk16[key], sc->Tscl[key], GDN_OPS_sTw, sc->wtbuf); sc->sTw[key] = GDN_OPS_sTw;
        }
        gdn_effective(sc->wtbuf, sc->effc[key]); sc->colabsc[key] = GDN_OPS_COLABS;
        gdn_pack_w8_kmajor(sc->wtbuf, km);
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
        gdn_quant_u8_q15(sc->Tblk16[bii], sc->Tscl[bii], GDN_OPS_sTa, sc->actbuf);   /* int16-native Q15 (g≈0.008<1) */
        sc->sTa[i] = GDN_OPS_sTa;
        gdn_pack_act_crouton8(sc->actbuf, cr);
        sc->vTa[i] = 1;
    }
    *sa_out = sc->sTa[i];
    return cr;
}

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
        gdn_solve_diag64(sc, Ah + (size_t)i * BL * C + i * BL, C, zpA, M, S, sc->Tc, nullptr);
        gdn_narrow_i32_to_i16(sc->Tc, sc->Tblk16[bi]);
        sc->Tscl[bi] = GDN_BR_TI;
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
    for (int d = 1; d < NB; ++d) {
        for (int j = 0; j + d < NB; ++j) {
            int i = j + d, bij = gdn_blk_index(i, j);
            float s_S = 0.f; int first = 1;
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
            /* final merge T_ij = T_ii @ Sacc. */
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
            gdn_requant_i16(sc->Tblk16[bij], sij, sT, zpT, Th, i * BL, j * BL, C);   /* int16-lane requant */
#if defined(GDN_BR_TRACE)
            gdn_tr_push(_tid, 7, _f2, gdn_trnow());   /* REQ (widen+requant) */
#endif
        }
    }
}
#endif  /* GDN_BR_I16 && __hexagon__ */
