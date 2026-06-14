/* gdn_pure_solve.cpp — CLEAN single-path real-data pure-HMX w16a16 triangular inverse.
 *
 * One implementation, no harness/no garbage operands. T = (I-A)^-1 for a block-lower-triangular
 * (I-A), C=256 = 4x4 blocks of 64, H heads. Built ENTIRELY on the byte-proven w16a16_mm.h
 * primitive (M=256 x K=64 x N=64, software exponent tracking, oc 1.4e-5 vs fp64 on the primitive).
 *
 * Per head:
 *   - diag  T_ii = (I - A_ii)^-1  via Taylor(p=3) seed + Newton-Schulz(K)   [A_ii strictly-lower]
 *   - merge T_ij = T_ii @ ( sum_{k=j}^{i-1} A_ik @ T_kj )   (block forward-substitution, i>j)
 *
 * Scale model (caller-owned, the primitive only drains product/32767): a matrix V is stored as
 * int16 codes (|code| <= 32639) plus an integer exponent e, with  V = code * 2^e / 32767.
 *
 * CV-DOMAIN (O2): every intermediate (A/T/X/Z/...) lives as CROUTON16 codes (the kernel's act/out
 * layout, compact 16x256 = 4096), so a matmul does NO per-mm depack/repack: act = direct cv->surface
 * copy, out = direct surface->cv copy. Only the WEIGHT is LUT-linearized + kmajor-packed each call
 * (kmajor != crouton). Pointwise renorm/acc are layout-agnostic; the 64 diag fix-ups index via the
 * crouton LUT. A unpacks linear->cv once/head, T depacks cv->linear once/head. Uses TRUE 64^3
 * descriptors (INVARIANT 7: out_y_stride=64, n_tiles=64; padded 2KB act/out blocks, live 512B).
 *
 * THREADING (O1): the per-head math runs on a `gp_ctx` (scratch + one VTCM mm region). P>=2 = P HVX
 * producers (static head-stripe) each own a ctx and do the FULL solve EXCEPT the HMX kernel, which is
 * delegated to the single MAIN-thread consumer (1 HMX unit -> only main touches mxmem). Hand-off =
 * one job slot/producer (synchronous: pack -> arm -> spin -> use). P=1 = single-thread inline.
 *
 * 口径 (skill htp-cycle-metric, all PCYCLE): stats report graph-wall(0), HMX op-latency floor
 * (N_mm*256), per-call feed-inclusive kernel(g_cbusy), producer pack/spin. NEVER label feed-inclusive
 * kernel time as HMX compute.
 *
 * Build: EXTRA_DEFS="-DGDNBM_GDN_PURE_SOLVE" bash example/gdn_native/baremetal/build.sh
 * Entry: gdn_pure::run(P, H, A, stats, statsLen, T, TLen)
 *   A: H*(256x256 int16 q16, strictly lower)  T: H*(256x256 int16 codes + 16 int32 block exps @ +128B)
 */
#include <stdint.h>
#include <string.h>
#include "qurt.h"
#include "HAP_compute_res.h"
#include "HAP_farf.h"
#include "HAP_power.h"
#include "HAP_perf.h"
#include "w16a16_mm.h"
#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"

namespace gdn_pure {

/* HVX cv<->surface copy: int16 code <-> u16 zp-32768 is exactly XOR 0x8000. 16 blocks, 256 live u16
 * each (= 4 HVX vectors). Replaces the scalar per-element VTCM access (~4x slower than DDR, INV 8③).
 * Requires the cv side 128-byte aligned; surface (VTCM act/out) is 2KB-block aligned. */
/* DENSE (cron#47): pack the 16 cv blocks CONTIGUOUS (b*256, 4 blocks/2KB tile) instead of 16 sparse
 * ×2048 tiles. With dense atab (i&3)*2048 + n_tiles=8 the matmul streams in 8 walks (= native's
 * [1,8,8,64] n_tiles=8) -> per-call 10841->1577 = native warm HMX-busy parity, bit-exact (micro-bench
 * D1). GP_DENSE_SURF=0 keeps the old sparse layout. */
#ifndef GP_DENSE_SURF
#define GP_DENSE_SURF 0   /* WIP (cron#47): kernel parity PROVEN (D1 micro-bench 1577 bit-exact) but the
                           * solve cv-path with dense g_lut still gives oc 0.708 (n_tiles-independent =
                           * cv-build/output chain bug, not root-caused). 0 = safe cron#42 baseline. */
#endif

/* GP_CROUTON8 (cron#68, NEXT①): the CORRECT dense path — store cv in the PROVEN closed-form
 * `crouton_pos` order (cron#66-67 bit-exact to native ConvLayer_s1) + native M=64 descriptor
 * (out_y=4/n_tiles=8/m_total=8/act_y=4). Replaces the gp_cv_to_surf n_tiles=32 (4× over-walk) with
 * n_tiles=8 = the true per-matmul floor (~5547→~1577/call, consumer-busy ~4.9M→~1.2M). g_lut becomes
 * crouton_pos so cv == surface (contiguous 4096) and g_hw/g_il/g_fl auto-follow (PACKCHK stays 0).
 * Supersedes GP_DENSE_SURF/GP_DENSE_PERM which used the WRONG pack_act_crouton16 order (→ oc 0.708). */
#ifndef GP_CROUTON8
#define GP_CROUTON8 1   /* DEFAULT (cron#68): proven crouton_pos n_tiles=8 = consumer-busy 4.9M->1.5M (3.27×),
                         * graph-wall 6.61M->4.77M (1.39×, oc 4.24e-3 unchanged). =0 = legacy n_tiles=32. */
#endif
/* native M=64 act/out crouton tile = a pure bit-permutation (cron#67; bit-exact reproduces the
 * ramp-dump tables). bits of pos (lo->hi): r0,c0,c1,c2,c3,c4,r1,r3,r4,r5,c5,r2. */
#define CROUTON_POS(r,c) ( (((r)&1)<<0) | (((c)&1)<<1) | ((((c)>>1)&1)<<2) | ((((c)>>2)&1)<<3) | \
    ((((c)>>3)&1)<<4) | ((((c)>>4)&1)<<5) | ((((r)>>1)&1)<<6) | ((((r)>>3)&1)<<7) | \
    ((((r)>>4)&1)<<8) | ((((r)>>5)&1)<<9) | ((((c)>>5)&1)<<10) | ((((r)>>2)&1)<<11) )
#if GP_DENSE_SURF
#define GP_SURF_BLK 256
#else
#define GP_SURF_BLK 1024
#endif
#if GP_CROUTON8
/* cv is stored in crouton_pos order (g_lut=crouton_pos) == the kernel's act/out surface (4 contiguous
 * 2KB tiles = 4096 u16). So the copy is a flat XOR (int16 code <-> u16 zp32768). */
static inline void gp_cv_to_surf(uint16_t *surf, const int16_t *cv) {
    const HVX_Vector K = Q6_Vh_vsplat_R(0x8000);
    const HVX_Vector *s = (const HVX_Vector *)cv; HVX_Vector *d = (HVX_Vector *)surf;
    for (int k = 0; k < 64; ++k) d[k] = Q6_V_vxor_VV(s[k], K);   /* 4096 u16 = 64 HVX vectors */
}
static inline void gp_surf_to_cv(int16_t *cv, const uint16_t *surf) {
    const HVX_Vector K = Q6_Vh_vsplat_R(0x8000);
    const HVX_Vector *s = (const HVX_Vector *)surf; HVX_Vector *d = (HVX_Vector *)cv;
    for (int k = 0; k < 64; ++k) d[k] = Q6_V_vxor_VV(s[k], K);
}
#else
static inline void gp_cv_to_surf(uint16_t *surf, const int16_t *cv) {
    const HVX_Vector K = Q6_Vh_vsplat_R(0x8000);
    for (int b = 0; b < 16; ++b) {
        const HVX_Vector *s = (const HVX_Vector *)(cv + b * 256);
        HVX_Vector *d = (HVX_Vector *)(surf + b * GP_SURF_BLK);
        for (int k = 0; k < 4; ++k) d[k] = Q6_V_vxor_VV(s[k], K);
    }
}
static inline void gp_surf_to_cv(int16_t *cv, const uint16_t *surf) {
    const HVX_Vector K = Q6_Vh_vsplat_R(0x8000);
    for (int b = 0; b < 16; ++b) {
        const HVX_Vector *s = (const HVX_Vector *)(surf + b * GP_SURF_BLK);
        HVX_Vector *d = (HVX_Vector *)(cv + b * 256);
        for (int k = 0; k < 4; ++k) d[k] = Q6_V_vxor_VV(s[k], K);
    }
}
#endif

/* DENSE-PERM (cron#48): ISOLATED cv<->pack-order permutation INSIDE the surf copy only. Keeps the
 * proven original g_lut for the whole solve (acc/diag/renorm/unpack/weight untouched); only the
 * surf produced for the matmul is reordered into pack_act_crouton16 order so dense atab (i&3)*2048 +
 * n_tiles=8 = native 1577. g_qa: pack-pos p -> 2*(orig cv idx); g_qo: orig cv idx k -> 2*(pack pos). */
#ifndef GP_DENSE_PERM
#define GP_DENSE_PERM 0   /* WIP (cron#48): isolated cv<->pack perm; full-solve oc still 0.708 (identical to
                           * the g_lut-swap path => not the reorder method). 0 = safe cron#42 baseline. */
#endif
static uint16_t *g_qa, *g_qo;     /* VTCM, 4096 u16 each: byte-offset gather LUTs for the cv<->pack perm */
/* pack_act_crouton16 contiguous index for linear (r*64+c). */
static inline int gp_pack_idx(int r, int c) {
    int m32 = r >> 5, rr = r & 31, row4 = rr >> 2, rp = (rr >> 1) & 1, parity = r & 1, kt = c >> 5, cc = c & 31;
    return ((((row4 * 2 + kt) * 2 + m32) * 2 + rp) * 32 + cc) * 2 + parity;
}
static void gp_cv_to_surf_perm(uint16_t *surf, const int16_t *cv, int16_t *stage) {
    const HVX_Vector K = Q6_Vh_vsplat_R(0x8000);
    for (int i = 0; i < 64; ++i) ((HVX_Vector *)stage)[i] = ((const HVX_Vector *)cv)[i];   /* cv -> VTCM */
    HVX_Vector *g = (HVX_Vector *)(stage + 4096);
    const HVX_Vector *o = (const HVX_Vector *)g_qa;
    for (int v = 0; v < 64; ++v) {
        Q6_vgather_ARMVh((void *)g, (uint32_t)(uintptr_t)stage, 8191, o[v]);
        ((HVX_Vector *)surf)[v] = Q6_V_vxor_VV(*g, K);   /* surf[pack pos] = cv[orig] ^ zp */
    }
}
static void gp_surf_to_cv_perm(int16_t *cv, const uint16_t *surf, int16_t *stage) {
    const HVX_Vector K = Q6_Vh_vsplat_R(0x8000);
    for (int i = 0; i < 64; ++i) ((HVX_Vector *)stage)[i] = ((const HVX_Vector *)surf)[i];   /* surf -> VTCM */
    HVX_Vector *g = (HVX_Vector *)(stage + 4096);
    const HVX_Vector *o = (const HVX_Vector *)g_qo;
    for (int v = 0; v < 64; ++v) {
        Q6_vgather_ARMVh((void *)g, (uint32_t)(uintptr_t)stage, 8191, o[v]);
        ((HVX_Vector *)cv)[v] = Q6_V_vxor_VV(*g, K);     /* cv[orig] = surf[pack pos] ^ zp */
    }
}

#define GP_NB     4          /* 64-blocks per C=256 head */
#define GP_BL     64
#define GP_BB     4096       /* GP_BL*GP_BL = crouton-compact codes per 64-block */
#define GP_NT     4          /* max producer threads */
#ifndef GP_NEWTON
#define GP_NEWTON 0          /* Newton-Schulz iters. @scale0.05 oc is MONOTONIC in Newton (0:4.24e-3 < 1:8.24e-3
                              * < 2:9.66e-3): Taylor(3) is already near-exact, each Newton iter only ADDS w16a16
                              * quant noise. So 0 = best (768 mm, both faster & more accurate). ⚠️ SCALE-DEPENDENT:
                              * larger ‖A‖ (real GDN) needs Newton>=1 — re-check oc per deployment scale. */
#endif
#define GP_VSTRIDE 0x84000u  /* per-producer VTCM region: mm(~0x28000)+stage(0x8000)+resident A/T/scratch/acc/lin(0x52000) (cron#74). 4*0x84000=2.1MB < 8MB VTCM */

static inline uint64_t gp_pcyc(void) { uint64_t v; asm volatile("%0 = C15:14" : "=r"(v)); return v; }

/* ---- GP_TRACE: per-thread event trace for the ASCII timeline (scripts/gdn_pipe_timeline.py).
 * Format written into T at run() end: [magic u32=0x47545203][n u32][wall u64][base u64] then
 * n*{tid u32, stage u32, t0 u64, t1 u64}. tids: producers 0..P-1, consumer = GP_NT(=4). Stages
 * reuse the script's map: 3=MM(consumer kernel), 5=PREP(producer pack), 11=SPIN(producer wait).
 * Gated so the normal build/measure path is byte-identical (no trace overhead). ---- */
#ifdef GP_TRACE
struct gp_ev { uint32_t tid, stage; uint64_t t0, t1; };
static struct gp_ev g_ev[16384];
static volatile int g_ev_n;
static inline void gp_ev_push(uint32_t tid, uint32_t stage, uint64_t t0, uint64_t t1) {
    int i = __atomic_fetch_add(&g_ev_n, 1, __ATOMIC_RELAXED);
    if (i < (int)(sizeof(g_ev) / sizeof(g_ev[0]))) { g_ev[i].tid = tid; g_ev[i].stage = stage; g_ev[i].t0 = t0; g_ev[i].t1 = t1; }
}
#define GP_EV(tid, stage, t0, t1) gp_ev_push((tid), (stage), (t0), (t1))
#else
#define GP_EV(tid, stage, t0, t1) ((void)0)
#endif

/* crouton LUT: linear 64x64 index (r*64+c) -> compact cv index (block*256 + intra), block<16. */
static uint16_t g_lut[GP_BB];
static int32_t g_diagmask[GP_BB] __attribute__((aligned(128)));   /* cron#74: -1 at the 64 diag acc-positions, 0 else -> vectorizes gp_acc_diag_add (was 64 scalar RMW; prereq for VTCM-resident acc) */
static void gp_lut_init(void) {
#if GP_CROUTON8
    /* CROUTON8 (cron#68): cv index = native crouton_pos(r,c). Everything (g_hw weight pack, g_il/g_fl
     * unpack/pack, diag fix-up) derives from g_lut so it all follows. gp_cv_to_surf is then a flat copy. */
    for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c)
        g_lut[r * 64 + c] = (uint16_t)CROUTON_POS(r, c);
#elif GP_DENSE_SURF
    /* DENSE (cron#47): cv = w16a16_pack_act_crouton16 contiguous order (4096, 4 dense 2KB tiles). With
     * dense atab (i&3)*2048 + n_tiles=8 the kernel streams in 8 walks = native [1,8,8,64]. Everything
     * (g_hw weight pack / g_il-g_fl / diag / PACKCHK) derives from g_lut so it all follows. Proven
     * bit-exact + 1577 cyc in the GP_DENSE8 micro-bench (D1). */
    for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c) {
        int m32 = r >> 5, rr = r & 31, row4 = rr >> 2, rp = (rr >> 1) & 1, parity = r & 1, kt = c >> 5, cc = c & 31;
        g_lut[r * 64 + c] = (uint16_t)(((((row4 * 2 + kt) * 2 + m32) * 2 + rp) * 32 + cc) * 2 + parity);
    }
#else
    for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c)
        g_lut[r * 64 + c] = (uint16_t)(((((r >> 2) & 7) * 2 + (c >> 5)) * 1024) +
            (((r >> 5) * 2 + ((r >> 1) & 1)) * 64) + ((c & 31) * 2) + (r & 1));   /* -> surface u16 idx */
    for (int i = 0; i < GP_BB; ++i) { int s = g_lut[i]; g_lut[i] = (uint16_t)(((s >> 10) << 8) + (s & 1023)); }
#endif
    /* diag mask (cron#74): acc interleaved index of (d,d) = (i>>6)*64 + (off&1)*32 + (off>>1), i=g_lut[d*65]. */
    for (int i = 0; i < GP_BB; ++i) g_diagmask[i] = 0;
    for (int d = 0; d < 64; ++d) { int i = g_lut[d * 65], off = i & 63;
        g_diagmask[(i >> 6) * 64 + (off & 1) * 32 + (off >> 1)] = -1; }
}

/* ---- HVX weight pack (O4b): replaces scalar gather + kmajor + bias (all scalar VTCM writes). ----
 * g_hw[h] = byte offset (into the VTCM stage) of stream-order halfword h; vgather pulls cv codes in
 * kmajor stream order, then lo/hi byte split + 4lo|4hi shuffle -> wt; HVX colsum -> bias. Byte-identical
 * to w16a16_pack_wt_kmajor + w16a16_pack_bias (PACKCHK self-check in run()). Ported from the proven
 * pure_hmx_solve.cpp:p4v_pack_wt_bias. cv codes are already <=32639 (renorm), so no extra clip needed. */
static uint16_t *g_hw;            /* VTCM, 4096 u16 stream-order gather offsets */
static uint16_t *g_il, *g_fl;     /* VTCM, 4096 u16 each: byte-offset LUTs (linear<->cv vgather; g_fl also = colsum gather) */
static void gp_hwlut_init(void) {
    int h = 0;
    for (int nt = 0; nt < 2; ++nt)
        for (int half = 0; half < 2; ++half)
            for (int kt = 0; kt < 2; ++kt)
                for (int grp = 0; grp < 8; ++grp)
                    for (int idx = 0; idx < 64; idx += 8)
                        for (int j = 0; j < 8; ++j) {
                            int vi = idx + j, lane = vi / 16, off0 = (grp * 8 + half * 4 + lane) * 16;
                            int off = off0 + (vi & 15), rgrp = off / 128, rem = off % 128;
                            int col = rem / 4, row = rgrp * 4 + rem % 4;
                            g_hw[h++] = (uint16_t)(2 * g_lut[(kt * 32 + row) * 64 + (nt * 32 + col)]);
                        }
}
static void gp_pack_wt_bias_hvx(const int16_t *w_cv, int16_t *stage /*VTCM 8KB+*/, uint8_t *wt, int32_t *bias) {
    /* cron#74: w_cv is VTCM-resident (all block operands live in VTCM) -> gather DIRECTLY from it; the old
     * 64-vec cv->stage staging copy is gone. gtmp (gather output, 64 vec) goes at stage[0]. */
    HVX_Vector *gtmp = (HVX_Vector *)stage;
    const HVX_Vector *ofs = (const HVX_Vector *)g_hw;
    const HVX_Vector K128 = Q6_Vh_vsplat_R(128);
#if GP_GPIPE
    /* cron#72: was 2-deep gather (gtmp[0..1]) read back inside the pack loop => each pair's pack stalled on
     * its gather. Issue ALL 64 gathers into a 64-vec buffer first (engine pipelines), then pack. */
    for (int v = 0; v < 64; ++v) Q6_vgather_ARMVh((void *)&gtmp[v], (uint32_t)(uintptr_t)w_cv, 8191, ofs[v]);
    for (int v = 0; v < 64; v += 2) {
        HVX_Vector q0 = gtmp[v], q1 = gtmp[v + 1];
        HVX_Vector lo = Q6_Vb_vpacke_VhVh(q1, q0);                                  /* low bytes */
        HVX_Vector h0 = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh_sat(q0, K128), 8);
        HVX_Vector h1 = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh_sat(q1, K128), 8);
        HVX_Vector hi = Q6_Vb_vpack_VhVh_sat(h1, h0);                               /* rounded high bytes */
        HVX_VectorPair il = Q6_W_vshuff_VVR(hi, lo, -4);                            /* 4lo|4hi grains */
        ((HVX_Vector *)wt)[v] = Q6_V_lo_W(il); ((HVX_Vector *)wt)[v + 1] = Q6_V_hi_W(il);
    }
#else
    for (int v = 0; v < 64; v += 2) {
        Q6_vgather_ARMVh((void *)&gtmp[0], (uint32_t)(uintptr_t)w_cv, 8191, ofs[v]);
        Q6_vgather_ARMVh((void *)&gtmp[1], (uint32_t)(uintptr_t)w_cv, 8191, ofs[v + 1]);
        HVX_Vector q0 = gtmp[0], q1 = gtmp[1];
        HVX_Vector lo = Q6_Vb_vpacke_VhVh(q1, q0);                                  /* low bytes */
        HVX_Vector h0 = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh_sat(q0, K128), 8);
        HVX_Vector h1 = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh_sat(q1, K128), 8);
        HVX_Vector hi = Q6_Vb_vpack_VhVh_sat(h1, h0);                               /* rounded high bytes */
        HVX_VectorPair il = Q6_W_vshuff_VVR(hi, lo, -4);                            /* 4lo|4hi grains */
        ((HVX_Vector *)wt)[v] = Q6_V_lo_W(il); ((HVX_Vector *)wt)[v + 1] = Q6_V_hi_W(il);
    }
#endif
    static const int32_t ctrl[2] = { 0x00404420, 0x40000000 };   /* 1/32767 drain = the solve's scale model */
#if GP_CROUTON8
    /* colsum GATHER-FREE (cron#68): exploit the crouton_pos cv structure directly (no extra vgather — feed
     * is the wall, a 2nd 64-vec gather here cost ~3.4M). cv index k: lane=k&63 -> bit0=r0, bits1-5=c&31;
     * vector v=k>>6 -> bit4=c5. So column n=(lane>>1) | (c5<<5); group vectors by c5 (v bit4), accumulate
     * int32 per lane, then colsum[n] = lo[n&31]+hi[n&31] (even lane 2cc=lo, odd lane 2cc+1=hi). */
    {
        const HVX_Vector *cvv = (const HVX_Vector *)w_cv;
        HVX_Vector p0L = Q6_V_vzero(), p0H = Q6_V_vzero(), p1L = Q6_V_vzero(), p1H = Q6_V_vzero();
        for (int v = 0; v < 64; ++v) {
            HVX_VectorPair w = Q6_Ww_vsxt_Vh(cvv[v]);
            if ((v >> 4) & 1) { p1L = Q6_Vw_vadd_VwVw(p1L, Q6_V_lo_W(w)); p1H = Q6_Vw_vadd_VwVw(p1H, Q6_V_hi_W(w)); }
            else              { p0L = Q6_Vw_vadd_VwVw(p0L, Q6_V_lo_W(w)); p0H = Q6_Vw_vadd_VwVw(p0H, Q6_V_hi_W(w)); }
        }
        union { HVX_Vector v; int32_t w[32]; } a0L, a0H, a1L, a1H;
        a0L.v = p0L; a0H.v = p0H; a1L.v = p1L; a1H.v = p1H;
        for (int n = 0; n < 64; ++n) {
            int cc = n & 31, c5 = n >> 5;
            int32_t cs = c5 ? (a1L.w[cc] + a1H.w[cc]) : (a0L.w[cc] + a0H.w[cc]);
            int32_t eff = (-cs) >> 1;   /* == floor(-cs/2) for all cs (arith shift); was a per-elem /2 (cron#73) */
            int g = n >> 4, idx = n & 15;
            bias[g * 64 + 32 + idx * 2] = eff; bias[g * 64 + 32 + idx * 2 + 1] = 0;
        }
    }
#else
    for (int nt = 0; nt < 2; ++nt) {
        HVX_Vector aL = Q6_V_vzero(), aH = Q6_V_vzero();
        for (int k4 = 0; k4 < 8; ++k4) {
            const HVX_Vector *blk = (const HVX_Vector *)(w_cv + (k4 * 2 + nt) * 256);
            for (int sv = 0; sv < 4; ++sv) {
                HVX_VectorPair w = Q6_Ww_vsxt_Vh(blk[sv]);
                aL = Q6_Vw_vadd_VwVw(aL, Q6_V_lo_W(w)); aH = Q6_Vw_vadd_VwVw(aH, Q6_V_hi_W(w));
            }
        }
        union { HVX_Vector v; int32_t w[32]; } uL, uH; uL.v = aL; uH.v = aH;
        for (int c = 0; c < 32; ++c) {
            long cs = (long)uL.w[c] + uH.w[c];
            long v2 = -cs, eff = (v2 >= 0) ? (v2 / 2) : -(((-v2) + 1) / 2);
            int n = nt * 32 + c, g = n >> 4, idx = n & 15;
            bias[g * 64 + 32 + idx * 2] = (int32_t)eff; bias[g * 64 + 32 + idx * 2 + 1] = 0;
        }
    }
#endif
    for (int g = 0; g < 4; ++g) for (int i = 0; i < 32; ++i) bias[g * 64 + i] = ctrl[i & 1];
}

/* ---- HVX int32 acc helpers (O4c). acc layout = interleaved {lo=even hw lanes, hi=odd} from vsxt;
 * round-trip cv<->acc is consistent (vsxt in / vasr_VwVwR_sat out). Pointwise ops order-agnostic; only
 * the 64 diag fix-ups need the interleaved index. Ported from pure_hmx_solve.cpp:p4v_acc*. Replaces the
 * scalar renorm/acc loops (the lone remaining scalar bottleneck). acc must be 128-byte aligned. */
static inline void gp_acc3(int32_t *acc, const int16_t *a, const int16_t *b, const int16_t *cc) {
    const HVX_Vector *va = (const HVX_Vector *)a, *vb = (const HVX_Vector *)b, *vc = (const HVX_Vector *)cc;
    HVX_Vector *d = (HVX_Vector *)acc;
    for (int i = 0; i < 64; ++i) {
        HVX_VectorPair s = Q6_Ww_vadd_WwWw(Q6_Ww_vadd_WwWw(Q6_Ww_vsxt_Vh(va[i]), Q6_Ww_vsxt_Vh(vb[i])), Q6_Ww_vsxt_Vh(vc[i]));
        d[2 * i] = Q6_V_lo_W(s); d[2 * i + 1] = Q6_V_hi_W(s);
    }
}
static inline void gp_acc_negw(int32_t *acc, const int16_t *a) {
    const HVX_Vector *va = (const HVX_Vector *)a; HVX_Vector *d = (HVX_Vector *)acc;
    HVX_VectorPair z = Q6_Ww_vsxt_Vh(Q6_V_vzero());
    for (int i = 0; i < 64; ++i) { HVX_VectorPair s = Q6_Ww_vsxt_Vh(va[i]);
        d[2 * i] = Q6_Vw_vsub_VwVw(Q6_V_lo_W(z), Q6_V_lo_W(s));
        d[2 * i + 1] = Q6_Vw_vsub_VwVw(Q6_V_hi_W(z), Q6_V_hi_W(s)); }
}
static inline void gp_acc_zero(int32_t *acc) { HVX_Vector *d = (HVX_Vector *)acc, z = Q6_V_vzero();
    for (int i = 0; i < 128; ++i) d[i] = z; }
static inline void gp_acc_addsh(int32_t *acc, const int16_t *p, int sh) {
    const HVX_Vector *vp = (const HVX_Vector *)p; HVX_Vector *d = (HVX_Vector *)acc;
    for (int i = 0; i < 64; ++i) { HVX_VectorPair w = Q6_Ww_vsxt_Vh(vp[i]);
        d[2 * i] = Q6_Vw_vadd_VwVw(d[2 * i], Q6_Vw_vasr_VwR(Q6_V_lo_W(w), sh));
        d[2 * i + 1] = Q6_Vw_vadd_VwVw(d[2 * i + 1], Q6_Vw_vasr_VwR(Q6_V_hi_W(w), sh)); }
}
static inline void gp_acc_shr(int32_t *acc, int sh) { HVX_Vector *d = (HVX_Vector *)acc;
    for (int i = 0; i < 128; ++i) d[i] = Q6_Vw_vasr_VwR(d[i], sh); }
static inline int32_t gp_acc_absmax(const int32_t *acc) {
    const HVX_Vector *d = (const HVX_Vector *)acc; HVX_Vector mx = Q6_V_vzero();
    for (int i = 0; i < 128; ++i) mx = Q6_Vw_vmax_VwVw(mx, Q6_Vw_vabs_Vw(d[i]));
    for (int sh = 64; sh >= 4; sh >>= 1) mx = Q6_Vw_vmax_VwVw(mx, Q6_V_vror_VR(mx, sh));
    union { HVX_Vector v; int32_t w[32]; } u; u.v = mx; return u.w[0];
}
static inline void gp_acc_to_cv(int16_t *cv, const int32_t *acc, int s) {
    const HVX_Vector *d = (const HVX_Vector *)acc; HVX_Vector *o = (HVX_Vector *)cv;
    const HVX_Vector CP = Q6_Vh_vsplat_R(32639), CN = Q6_Vh_vsplat_R(-32639);
    for (int i = 0; i < 64; ++i) { HVX_Vector lo = d[2 * i], hi = d[2 * i + 1];
        if (s < 0) { lo = Q6_Vw_vasl_VwR(lo, -s); hi = Q6_Vw_vasl_VwR(hi, -s); }
        o[i] = Q6_Vh_vasr_VwVwR_sat(hi, lo, s > 0 ? (s & 31) : 0);
        o[i] = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(o[i], CN), CP); }
}
static inline int gp_renorm(int32_t *acc, int16_t *cv) {
    int32_t mx = gp_acc_absmax(acc);
    int s = 0; while ((mx >> s) > 32639) ++s;
    if (s == 0) { while (mx && (mx << 1) <= 16384) { mx <<= 1; --s; } }
    gp_acc_to_cv(cv, acc, s); return s;
}
static inline void gp_acc_diag_add(int32_t *acc, int32_t add) {   /* cron#74: vector acc += diagmask & splat(add) (was 64 scalar RMW) */
    const HVX_Vector *m = (const HVX_Vector *)g_diagmask; HVX_Vector *d = (HVX_Vector *)acc;
    HVX_Vector av = Q6_V_vsplat_R(add);
    for (int v = 0; v < GP_BB / 32; ++v) d[v] = Q6_Vw_vadd_VwVw(d[v], Q6_V_vand_VV(m[v], av));
}
static inline void gp_acc_from_cv(int32_t *acc, const int16_t *a) {
    const HVX_Vector *va = (const HVX_Vector *)a; HVX_Vector *d = (HVX_Vector *)acc;
    for (int i = 0; i < 64; ++i) { HVX_VectorPair s = Q6_Ww_vsxt_Vh(va[i]);
        d[2 * i] = Q6_V_lo_W(s); d[2 * i + 1] = Q6_V_hi_W(s); }
}

/* HVX permute (O4d): dst[j] = src[ofs[j]/2], src staged in VTCM (8KB), ofs = byte offsets. Replaces the
 * scalar per-head A-unpack / T-pack LUT-scatter. g_il: linear->cv (A-unpack), g_fl: cv->linear (T-pack).
 * Ported from pure_hmx_solve.cpp:p4v_perm. src/dst/stage 128-aligned; ofs in VTCM. */
static uint16_t *g_tr;            /* VTCM, 4096 u16: cv transpose gather LUT (cron#65 dense merge fix) */

/* GP_GPIPE (cron#72): the per-block permute was 64× {vgather -> immediate readback of the SAME slot},
 * so every gather stalled the full VTCM-gather latency (~184 cyc) before its readback => 7.56M Σ (43% of
 * producer feed, the #1 cost). FIX: issue all 64 gathers into 64 DISTINCT VTCM slots (g+v) so the
 * scatter/gather engine pipelines them at throughput, then bulk-read. Pure schedule change, bit-identical
 * output (same offsets, same source). g needs 64 vec = 8KB at stage+4096 (stage region = 32KB, src uses
 * the first 8KB) — no overlap with the [stage, stage+8191] gather source region. */
#ifndef GP_GPIPE
#define GP_GPIPE 1
#endif
static void gp_perm(int16_t *dst, const int16_t *src, const uint16_t *ofs, int16_t *stage) {
    for (int i = 0; i < 64; ++i) ((HVX_Vector *)stage)[i] = ((const HVX_Vector *)src)[i];   /* src -> VTCM */
    HVX_Vector *g = (HVX_Vector *)(stage + 4096);
    const HVX_Vector *o = (const HVX_Vector *)ofs;
#if GP_GPIPE
    for (int v = 0; v < 64; ++v) Q6_vgather_ARMVh((void *)(g + v), (uint32_t)(uintptr_t)stage, 8191, o[v]);
    for (int v = 0; v < 64; ++v) ((HVX_Vector *)dst)[v] = g[v];
#else
    for (int v = 0; v < 64; ++v) {
        Q6_vgather_ARMVh((void *)g, (uint32_t)(uintptr_t)stage, 8191, o[v]);
        ((HVX_Vector *)dst)[v] = *g;
    }
#endif
}

/* O5 scatter-kill: fuse the (was-scalar) linear↔block memcpy into gp_perm's HVX staging, so the strided
 * 64×64 block I/O is one HVX pass (no c->lin round-trip, no scalar memcpy). row r = 128B = 1 HVX vector,
 * src/dst 128-aligned (rpcmem page + 128-multiple offsets). gp_unpack_blk: strided linear -> cv (A-unpack).
 * gp_pack_blk: cv -> strided linear (T-pack), vgather output stored straight to the strided dst. */
static void gp_unpack_blk(int16_t *dst, const int16_t *src_blk, const uint16_t *ofs, int16_t *stage) {
    for (int r = 0; r < 64; ++r) ((HVX_Vector *)stage)[r] = *(const HVX_Vector *)&src_blk[r * 256];  /* strided -> VTCM */
    HVX_Vector *g = (HVX_Vector *)(stage + 4096);
    const HVX_Vector *o = (const HVX_Vector *)ofs;
#if GP_GPIPE
    for (int v = 0; v < 64; ++v) Q6_vgather_ARMVh((void *)(g + v), (uint32_t)(uintptr_t)stage, 8191, o[v]);
    for (int v = 0; v < 64; ++v) ((HVX_Vector *)dst)[v] = g[v];
#else
    for (int v = 0; v < 64; ++v) {
        Q6_vgather_ARMVh((void *)g, (uint32_t)(uintptr_t)stage, 8191, o[v]);
        ((HVX_Vector *)dst)[v] = *g;
    }
#endif
}
static void gp_pack_blk(int16_t *dst_blk, const int16_t *src_cv, const uint16_t *ofs, int16_t *stage) {
    for (int i = 0; i < 64; ++i) ((HVX_Vector *)stage)[i] = ((const HVX_Vector *)src_cv)[i];
    HVX_Vector *g = (HVX_Vector *)(stage + 4096);
    const HVX_Vector *o = (const HVX_Vector *)ofs;
#if GP_GPIPE
    for (int v = 0; v < 64; ++v) Q6_vgather_ARMVh((void *)(g + v), (uint32_t)(uintptr_t)stage, 8191, o[v]);
    for (int v = 0; v < 64; ++v) *(HVX_Vector *)&dst_blk[v * 256] = g[v];   /* gather output -> strided linear dst */
#else
    for (int v = 0; v < 64; ++v) {
        Q6_vgather_ARMVh((void *)g, (uint32_t)(uintptr_t)stage, 8191, o[v]);
        *(HVX_Vector *)&dst_blk[v * 256] = *g;   /* vgather output -> strided linear dst (To) */
    }
#endif
}

/* ---- job hand-off (producer -> main consumer): acquire/release on `state` only. ---- */
struct gp_job { volatile int state; int _pad[31]; } __attribute__((aligned(128)));
static gp_job g_job[GP_NT];
#define GP_ARM(j)   __atomic_store_n(&(j)->state, 1, __ATOMIC_RELEASE)
#define GP_POLL(j)  (__atomic_load_n(&(j)->state, __ATOMIC_ACQUIRE) == 1)
#define GP_WAIT(j)  do { while (__atomic_load_n(&(j)->state, __ATOMIC_ACQUIRE) != 2) {} } while (0)
#define GP_DONE(j)  __atomic_store_n(&(j)->state, 2, __ATOMIC_RELEASE)
#define GP_RESET(j) __atomic_store_n(&(j)->state, 0, __ATOMIC_RELAXED)

/* ---- per-head/per-producer context: cv-code scratch + one VTCM matmul region. ---- */
/* cron#74 (VTCM residency, [[feedback_vtcm_only_intermediates]]): ALL intermediate solve state lives in
 * VTCM (pointers into the producer's VTCM slice, assigned in run()); DDR only at head-load (A in) and
 * head-store (T out). Offsets within the per-producer slice, AFTER mm(<=0x28000)+stage(0x28000..0x30000). */
#define GPV_A    0x30000u   /* 16 cv blocks  = 0x20000 (128KB) */
#define GPV_T    0x50000u   /* 16 cv blocks  = 0x20000 */
#define GPV_SCR  0x70000u   /* AA,A3,M,Z,Tt,prod = 6*0x2000 = 0xC000 */
#define GPV_ACC  0x7C000u   /* acc int32[GP_BB] = 0x4000 (16KB) */
#define GPV_LIN  0x80000u   /* lin int16[GP_BB] = 0x2000 */
/* per-producer VTCM slice end = 0x82000; GP_VSTRIDE must be >= this. */
struct gp_ctx {
    w16a16_mm_t mm;
    uint8_t     descs[256] __attribute__((aligned(64)));
    int         slot;                    /* -1 = single-thread inline; >=0 = producer (delegate kernel) */
    int16_t     (*A)[GP_BB], (*T)[GP_BB];                   /* cv codes — VTCM-resident (16 blocks each) */
    int         e[16];
    int16_t     *AA, *A3, *M, *Z, *Tt, *prod;              /* cv-code scratch — VTCM-resident */
    int32_t     *acc;                    /* interleaved int32 accumulator (HVX) — VTCM-resident */
    int16_t     *lin;                    /* contiguous scratch (perm / self-check) — VTCM-resident */
    int16_t     *stage;                  /* per-ctx VTCM 8KB+ staging for the HVX vgather weight pack */
    uint64_t    spin, t_pack, t_depack, t_life;
    uint64_t    t_gather, t_kmajor, t_bias, t_scatter;   /* sub-stage probes */
    uint64_t    t_mc, t_pm, t_ms;   /* O5 scatter split: linear↔block memcpy / gp_perm vgather / memset */
};
static gp_ctx   g_ctx[GP_NT];
static uint64_t g_cbusy;                 /* consumer: per-call feed-inclusive kernel cyc (口径 ④) */
static int      g_sat;
static volatile int g_pdone;
static const uint8_t *g_Ah; static uint8_t *g_Th; static int g_H, g_P;
static char __attribute__((aligned(128))) g_stack[GP_NT][32768];

/* ---- one 64x64 cv matmul: out_cv = a_cv @ b_cv, e_out = e_a + e_b (out_code = a@b/32767). ----
 * act = a_cv direct into the padded surface (NO crouton repack); weight = b_cv LUT-linearized + kmajor;
 * out = surface direct into out_cv (NO depack). t_pack/t_depack probes split the two direct copies. */
static void mm64(gp_ctx *c, const int16_t *a_cv, const int16_t *b_cv, int16_t *out_cv) {
    uint64_t p0 = gp_pcyc();
#if GP_DENSE_PERM
    gp_cv_to_surf_perm((uint16_t *)c->mm.act, a_cv, c->stage);   /* act: cv -> pack-order surf (gather) */
#else
    gp_cv_to_surf((uint16_t *)c->mm.act, a_cv);                  /* act: cv -> surface (HVX vxor) */
#endif
    uint64_t g0 = gp_pcyc(); c->t_pack += g0 - p0;              /* act-copy only (HVX, ~0) */
    gp_pack_wt_bias_hvx(b_cv, c->stage, c->mm.wt, c->mm.bias);   /* weight: HVX vgather + kmajor + bias -> VTCM */
    uint64_t k1 = gp_pcyc(); c->t_kmajor += k1 - g0;
    if (c->slot < 0) {                                          /* single-thread: run inline */
        uint64_t t0 = gp_pcyc(); w16a16_mm_run(&c->mm); uint64_t t1 = gp_pcyc(); g_cbusy += t1 - t0;
        GP_EV(GP_NT, 3, t0, t1);                                /* MATMUL op (HMX) */
    } else {                                                    /* producer: hand to main consumer */
        /* per-op events (QNN-granular): act-format(4) + wt-pack(5) are SEPARATE feed ops, like QNN's
         * q::ForceFormat_Crouton + q::ConvLayer.opt.weights_to_vtcm. MATMUL(3) is the consumer's own op. */
        GP_EV(c->slot, 4, p0, g0);                              /* ACT_FORMAT (HVX) = ForceFormat_Crouton */
        GP_EV(c->slot, 5, g0, k1);                              /* WT_PACK (HVX) = weights_to_vtcm */
        GP_ARM(&g_job[c->slot]);
        uint64_t s0 = gp_pcyc();
        GP_WAIT(&g_job[c->slot]); GP_RESET(&g_job[c->slot]);
        uint64_t s1 = gp_pcyc(); c->spin += s1 - s0;
        GP_EV(c->slot, 11, s0, s1);                             /* SPIN (idle-wait; not emitted as a slice) */
    }
    uint64_t d0 = gp_pcyc();
#if GP_DENSE_PERM
    gp_surf_to_cv_perm(out_cv, (const uint16_t *)c->mm.out, c->stage);  /* out: pack-order surf -> cv (gather) */
#else
    gp_surf_to_cv(out_cv, (const uint16_t *)c->mm.out);          /* out: surface -> cv (HVX vxor) */
#endif
    uint64_t d1 = gp_pcyc(); c->t_depack += d1 - d0;
    if (c->slot >= 0) GP_EV(c->slot, 10, d0, d1);              /* OUT_COPY (HVX) = q::*OutputSlice */
}

#if GP_DENSE_SURF || GP_DENSE_PERM
/* cron#65: dense n_tiles=8 mis-computes when the ACT has large hi-byte deviation × full weight (data-dep).
 * Compute out = a@b via the working operand order (a@b = (b^T@a^T)^T); transposes via g_tr (HVX vgather).
 * Uses c->Tt/Z/M scratch (free at GP_NEWTON=0). a_cv/b_cv/out_cv must not alias Tt/Z/M. */
static void mm64T(gp_ctx *c, const int16_t *a_cv, const int16_t *b_cv, int16_t *out_cv) {
    gp_perm(c->Tt, a_cv, g_tr, c->stage);            /* a^T */
    gp_perm(c->Z,  b_cv, g_tr, c->stage);            /* b^T */
    mm64(c, c->Z, c->Tt, c->M);                      /* b^T @ a^T = (a@b)^T  (act=b^T, weight=a^T) */
    gp_perm(out_cv, c->M, g_tr, c->stage);           /* -> a@b */
}
#define MM64 mm64T
#else
#define MM64 mm64
#endif

/* ---- diag block inverse: X = (I - A)^-1, A strictly-lower 64x64 (e=0), all in cv. Returns eX.
 * acc/renorm all HVX (interleaved layout). ---- */
static int diag_inv(gp_ctx *c, const int16_t *A, int16_t *X) {
    mm64(c, A, A, c->AA);                                       /* A^2 (benign act=A) */
    MM64(c, c->AA, A, c->A3);                                   /* A^3 (act=AA can be extreme -> transpose under dense) */
    gp_acc3(c->acc, A, c->AA, c->A3);
    gp_acc_diag_add(c->acc, 32767);                             /* X0 = I + A + A^2 + A^3 (Taylor p=3) */
    int eX = gp_renorm(c->acc, X);
#if GP_NEWTON > 0
    gp_acc_negw(c->acc, A);
    gp_acc_diag_add(c->acc, 32767);                            /* M = I - A */
    int eM = gp_renorm(c->acc, c->M);
    for (int it = 0; it < GP_NEWTON; ++it) {                    /* Newton-Schulz: X <- X(2I - (I-A)X) */
        mm64(c, c->M, X, c->Tt);                                /* (I-A)X */
        int e = eM + eX;
        gp_acc_negw(c->acc, c->Tt);
        if (e < 31) gp_acc_diag_add(c->acc, (int32_t)(65534u >> e));   /* 2I - (I-A)X */
        int eZ = e + gp_renorm(c->acc, c->Z);
        mm64(c, X, c->Z, c->Tt);                                /* X(2I-(I-A)X) */
        gp_acc_from_cv(c->acc, c->Tt);
        eX = eX + eZ + gp_renorm(c->acc, X);
    }
#endif  /* GP_NEWTON > 0 */
    return eX;
}

/* GP_CVIO (cron#73): I/O contract = A delivered / T returned already in cv-block layout (block (bi,bj) at
 * int16 offset (bi*4+bj)*GP_BB, crouton_pos order within block). The linear<->cv permute (was scatter's
 * vgather, #1 feed cost ~1.25M/producer) moves to the HOST (numpy, free vs DSP wall): in production the
 * solve is a custom op and A arrives from upstream already in crouton layout, so the on-DSP linear<->cv was
 * a standalone-harness artifact. On-DSP only an 8KB HVX block-copy remains (could alias to zero-copy later).
 * The 6 unused upper-tri blocks need no zeroing (host reads only lower-tri+diag). Exps -> unused block 1.
 * User-ratified 2026-06-15 (the metric now excludes layout conversion — legitimate per the integration). */
#ifndef GP_CVIO
#define GP_CVIO 1
#endif
/* ---- one full C=256 head: 4 diag inverses + block forward-substitution for the off-diagonals. ----
 * cron#74 ([[feedback_vtcm_only_intermediates]]): ALL intermediate state (A/T blocks, scratch, acc) is
 * VTCM-resident (c->A/c->T/... are VTCM pointers). DDR is touched ONLY at head-load (A cv-block in) and
 * head-store (T cv-block out). The whole solve runs in VTCM — no per-matmul DDR staging, no DDR round-trip
 * for the repeated T-block reads in the merge. ABLK/TBLK are macros (not local arrays) to keep the P=1
 * inline path's stack small. */
#define ABLK(i) (c->A[i])
#define TBLK(i) (c->T[i])
static void solve_head(gp_ctx *c, const int16_t *Aq, int16_t *To) {
    uint64_t us0 = gp_pcyc();
#if GP_CVIO
    for (int bi = 0; bi < GP_NB; ++bi)                          /* head-load: A cv-blocks rpcmem(DDR) -> c->A VTCM */
        for (int bj = 0; bj <= bi; ++bj) {
            const HVX_Vector *s = (const HVX_Vector *)&Aq[(bi * 4 + bj) * GP_BB];
            HVX_Vector *d = (HVX_Vector *)c->A[bi * 4 + bj];
            for (int v = 0; v < 64; ++v) d[v] = s[v];
        }
#else
    for (int bi = 0; bi < GP_NB; ++bi)                          /* head-load: unpack A blocks linear -> c->A VTCM (HVX) */
        for (int bj = 0; bj <= bi; ++bj) {
            uint64_t m1 = gp_pcyc();
            gp_unpack_blk(c->A[bi * 4 + bj], &Aq[(bi * GP_BL) * 256 + bj * GP_BL], g_il, c->stage);
            c->t_pm += gp_pcyc() - m1;
        }
#endif
    c->t_scatter += gp_pcyc() - us0;
    for (int b = 0; b < GP_NB; ++b) c->e[b * 5] = diag_inv(c, ABLK(b * 5), TBLK(b * 5));
    for (int d = 1; d < GP_NB; ++d)
        for (int i = d; i < GP_NB; ++i) {
            int j = i - d, eAcc = 0;
            gp_acc_zero(c->acc);
            for (int k = j; k < i; ++k) {                      /* acc = sum_k A_ik @ T_kj (exp-aligned) */
                mm64(c, ABLK(i * 4 + k), TBLK(k * 4 + j), c->prod);   /* benign act=A_ik; weight may be extreme (ok via :dilate) */
                int ep = c->e[k * 4 + j];
                if (k == j) eAcc = ep;
                else if (ep > eAcc) { gp_acc_shr(c->acc, ep - eAcc); eAcc = ep; }
                int sh = eAcc - ep;
                if (sh < 31) gp_acc_addsh(c->acc, c->prod, sh);
            }
            int eS = eAcc + gp_renorm(c->acc, c->prod);
            MM64(c, TBLK(i * 4 + i), c->prod, TBLK(i * 4 + j));   /* T_ij = T_ii @ S (dense: working operand order) */
            gp_acc_from_cv(c->acc, TBLK(i * 4 + j));
            c->e[i * 4 + j] = c->e[i * 4 + i] + eS + gp_renorm(c->acc, TBLK(i * 4 + j));
        }
    uint64_t us1 = gp_pcyc();
#if GP_CVIO
    /* head-store: c->T VTCM -> To cv-blocks rpcmem(DDR) (10 lower-tri+diag; 6 upper-tri never read).
     * Exps -> unused block (0,1) @ int16 offset 1*GP_BB. */
    for (int bi = 0; bi < GP_NB; ++bi)
        for (int bj = 0; bj <= bi; ++bj) {
            const HVX_Vector *s = (const HVX_Vector *)c->T[bi * 4 + bj];
            HVX_Vector *d = (HVX_Vector *)&To[(bi * 4 + bj) * GP_BB];
            for (int v = 0; v < 64; ++v) d[v] = s[v];
        }
    memcpy((uint8_t *)&To[1 * GP_BB], c->e, 64);              /* 16 block exps in unused upper-tri block (0,1) */
    c->t_scatter += gp_pcyc() - us1;
#else
#if GP_GPIPE
    /* cron#72: zero ONLY the 6 strict-upper blocks (bj>bi). The lower-tri + diagonal blocks are fully
     * written by gp_pack_blk below (gp_pack_blk writes all 64×64, incl. the diag block's 0 upper-tri from
     * the lower-tri inverse), so the full 128KB memset (0.88M Σ) wasted ~10/16 of its stores. */
    for (int bi = 0; bi < GP_NB; ++bi)
        for (int bj = bi + 1; bj < GP_NB; ++bj) {
            int16_t *blk = &To[(bi * GP_BL) * 256 + bj * GP_BL];
            for (int r = 0; r < GP_BL; ++r) *(HVX_Vector *)&blk[r * 256] = Q6_V_vzero();
        }
#else
    memset(To, 0, 131072);                                     /* pack lower-tri blocks cv -> linear (HVX, fused strided) */
#endif
    uint64_t ms1 = gp_pcyc(); c->t_ms += ms1 - us1;
    for (int bi = 0; bi < GP_NB; ++bi)
        for (int bj = 0; bj <= bi; ++bj) {
            uint64_t p0 = gp_pcyc();
            gp_pack_blk(&To[(bi * GP_BL) * 256 + bj * GP_BL], c->T[bi * 4 + bj], g_fl, c->stage);
            c->t_pm += gp_pcyc() - p0;
        }
    memcpy((uint8_t *)To + 128, c->e, 64);                     /* 16 block exps in unused upper-tri */
    c->t_scatter += gp_pcyc() - us1;
#endif
}
#undef ABLK
#undef TBLK

/* ---- producer thread: own a head-stripe, run the full solve, delegate each kernel to the consumer. ---- */
static void producer(void *arg) {
    int slot = (int)(intptr_t)arg;
    gp_ctx *c = &g_ctx[slot];
    int hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);
    uint64_t L0 = gp_pcyc();
    for (int h = slot; h < g_H; h += g_P) {                    /* static interleave */
        const int16_t *Aq = (const int16_t *)(g_Ah + (size_t)h * 131072);
        int16_t *To = (int16_t *)(g_Th + (size_t)h * 131072);
        solve_head(c, Aq, To);
    }
    c->t_life = gp_pcyc() - L0;
    if (hvx == 0) qurt_hvx_unlock();
    __atomic_fetch_add(&g_pdone, 1, __ATOMIC_RELEASE);
}

/* ---- entry. ---- */
int run(int P, int H, const uint8_t *A, int *stats, int statsLen, void *T, int TLen) {
    if (TLen < (size_t)H * 131072) return -2;
    if (P > GP_NT) P = GP_NT; if (P < 1) P = 1; if (P > H) P = H;
    static int pwr_client; void *pctx = &pwr_client;
    HAP_power_set_core_corner(pctx, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_MAX);
    { HAP_power_request_t r; memset(&r, 0, sizeof(r)); r.type = HAP_power_set_HMX; r.hmx.power_up = TRUE; HAP_power_set(pctx, &r); }
    compute_res_attr_t va; HAP_compute_res_attr_init(&va);
#if GP_DENSE_SURF || GP_DENSE_PERM
    HAP_compute_res_attr_set_vtcm_param(&va, (unsigned)P * GP_VSTRIDE + 0xC000u, 0);   /* +48KB: g_hw/g_il/g_fl/g_qa/g_qo/g_tr */
#else
    HAP_compute_res_attr_set_vtcm_param(&va, (unsigned)P * GP_VSTRIDE + 0xA000u, 0);   /* +40KB: g_hw/g_il/g_fl/g_qa/g_qo */
#endif
    unsigned int vctx = HAP_compute_res_acquire(&va, 2000000);
    uint8_t *vbase = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&va);
    compute_res_attr_t ha; HAP_compute_res_attr_init(&ha); HAP_compute_res_attr_set_hmx_param(&ha, 1);
    unsigned int hctx = HAP_compute_res_acquire(&ha, 2000000);
    int hl = HAP_compute_res_hmx_lock(hctx);                    /* consumer = main, PURE HMX (no HVX lock) */
    if (!vbase || hl != 0) { if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
        if (vctx) HAP_compute_res_release(vctx); return -1; }

    gp_lut_init();
    g_hw = (uint16_t *)(vbase + (size_t)P * GP_VSTRIDE);                    /* shared VTCM LUTs */
    g_il = (uint16_t *)(vbase + (size_t)P * GP_VSTRIDE + 0x2000u);          /* linear->cv (A-unpack) */
    g_fl = (uint16_t *)(vbase + (size_t)P * GP_VSTRIDE + 0x4000u);          /* cv->linear (T-pack) */
    gp_hwlut_init();
    for (int i = 0; i < GP_BB; ++i) { g_il[g_lut[i]] = (uint16_t)(2 * i); g_fl[i] = (uint16_t)(2 * g_lut[i]); }
#if GP_DENSE_SURF || GP_DENSE_PERM
    /* cv transpose gather LUT (cron#65): dst_cv[g_lut[c*64+r]] = src_cv[g_lut[r*64+c]] => g_tr[g_lut[c*64+r]]=2*g_lut[r*64+c].
     * used to compute the merge's T_ii@S as (S^T @ T_ii^T)^T (working operand order under dense n_tiles=8). */
    g_tr = (uint16_t *)(vbase + (size_t)P * GP_VSTRIDE + 0xA000u);
    for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c) g_tr[g_lut[c*64+r]] = (uint16_t)(2 * g_lut[r*64+c]);
#endif
#if GP_DENSE_PERM || defined(GP_DIFF)
    /* cv<->pack-order perm LUTs (cron#48): g_qa[pack_pos p] = 2*(orig cv idx of the element at p);
     * g_qo[orig cv idx k] = 2*(pack pos of the element at k). Derived from original g_lut + pack idx. */
    g_qa = (uint16_t *)(vbase + (size_t)P * GP_VSTRIDE + 0x6000u);
    g_qo = (uint16_t *)(vbase + (size_t)P * GP_VSTRIDE + 0x8000u);
    { static uint16_t lut_inv[GP_BB], pk[GP_BB], pk_inv[GP_BB];
      for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c) { int L = r*64+c; pk[L] = (uint16_t)gp_pack_idx(r, c); }
      for (int L = 0; L < GP_BB; ++L) { lut_inv[g_lut[L]] = (uint16_t)L; pk_inv[pk[L]] = (uint16_t)L; }
      for (int p = 0; p < GP_BB; ++p) g_qa[p] = (uint16_t)(2 * g_lut[pk_inv[p]]);
      for (int k = 0; k < GP_BB; ++k) g_qo[k] = (uint16_t)(2 * pk[lut_inv[k]]); }
#endif
    for (int t = 0; t < P; ++t) {
        w16a16_mm_init(&g_ctx[t].mm, vbase + (size_t)t * GP_VSTRIDE, g_ctx[t].descs);
        { hmx_conv_out_desc_t *od = (hmx_conv_out_desc_t *)g_ctx[t].mm.od;   /* TRUE 64^3 (INVARIANT 7) */
#if GP_CROUTON8
          /* CROUTON8 (cron#68): PROVEN native M=64 descriptor (cron#66-67, bit-exact). act/out = 4
           * contiguous 2KB tiles (atab/otab=(i&3)*2048); cv stored in crouton_pos order. n_tiles=8 =
           * true per-matmul floor (out_y=4/m_total=8/act_y=4 — the EXACT GP_ALIGN descriptor). */
          uint8_t *act = (uint8_t *)g_ctx[t].mm.act, *out = (uint8_t *)g_ctx[t].mm.out;
          for (int i = 0; i < 16; ++i) {
              g_ctx[t].mm.atab[i] = (int32_t)(uintptr_t)(act + (size_t)(i & 3) * 2048);
              g_ctx[t].mm.otab[i] = (int32_t)(uintptr_t)(out + (size_t)(i & 3) * 2048);
          }
          od->out_table_stride_dwords = 2u; od->out_y_stride_words = 4u; od->n_tiles_pow2 = 8u;
          od->m_total_minus_step = 8u; od->k_total_bytes = 64u;
          { hmx_conv_act_desc_t *ad = (hmx_conv_act_desc_t *)g_ctx[t].mm.ad;
            ad->n_act_pairs = 2u; ad->act_table_y_stride_words = 4u; }
#elif GP_DENSE_SURF || GP_DENSE_PERM
          /* DENSE (cron#47/48): act/out packed contiguous (4 tiles of 2KB); atab/otab = (i&3)*2048; n_tiles=8
           * = native [1,8,8,64]. per-call ->1577 = native warm HMX-busy 1547 parity, bit-exact. */
          uint8_t *act = (uint8_t *)g_ctx[t].mm.act, *out = (uint8_t *)g_ctx[t].mm.out;
          for (int i = 0; i < 16; ++i) {
              g_ctx[t].mm.atab[i] = (int32_t)(uintptr_t)(act + (size_t)(i & 3) * 2048);
              g_ctx[t].mm.otab[i] = (int32_t)(uintptr_t)(out + (size_t)(i & 3) * 2048);
          }
          od->out_y_stride_words = 64u; od->n_tiles_pow2 = 8u;
#else
          /* n_tiles: atab/otab repeat every 8 row-groups (rg&7); n_tiles=64 does redundant MAC passes. */
#ifndef GP_NTILES
#define GP_NTILES 32u   /* solve floor = ~30 (gp_cv_to_surf layout); 32 = safe margin, oc byte-identical to 64 */
#endif
          od->out_y_stride_words = 64u; od->n_tiles_pow2 = GP_NTILES;
#endif
        }
        g_ctx[t].stage = (int16_t *)(vbase + (size_t)t * GP_VSTRIDE + 0x28000u);   /* gap after mm.out */
        {   /* cron#74: VTCM-resident intermediate state (DDR only at head load/store). */
            uint8_t *vb = vbase + (size_t)t * GP_VSTRIDE;
            g_ctx[t].A   = (int16_t (*)[GP_BB])(vb + GPV_A);
            g_ctx[t].T   = (int16_t (*)[GP_BB])(vb + GPV_T);
            g_ctx[t].AA  = (int16_t *)(vb + GPV_SCR + 0 * 0x2000u);
            g_ctx[t].A3  = (int16_t *)(vb + GPV_SCR + 1 * 0x2000u);
            g_ctx[t].M   = (int16_t *)(vb + GPV_SCR + 2 * 0x2000u);
            g_ctx[t].Z   = (int16_t *)(vb + GPV_SCR + 3 * 0x2000u);
            g_ctx[t].Tt  = (int16_t *)(vb + GPV_SCR + 4 * 0x2000u);
            g_ctx[t].prod= (int16_t *)(vb + GPV_SCR + 5 * 0x2000u);
            g_ctx[t].acc = (int32_t *)(vb + GPV_ACC);
            g_ctx[t].lin = (int16_t *)(vb + GPV_LIN);
        }
        uint16_t *act = (uint16_t *)g_ctx[t].mm.act;                         /* prefill padded act = zp */
        for (int i = 0; i < W16MM_ACT_BYTES / 2; ++i) act[i] = 32768;
        memset(g_ctx[t].mm.out, 0, W16MM_OUT_BYTES);
        g_ctx[t].spin = 0; g_ctx[t].t_pack = 0; g_ctx[t].t_depack = 0; g_ctx[t].t_life = 0;
        g_ctx[t].t_gather = 0; g_ctx[t].t_kmajor = 0; g_ctx[t].t_bias = 0; g_ctx[t].t_scatter = 0; g_job[t].state = 0;
        g_ctx[t].t_mc = 0; g_ctx[t].t_pm = 0; g_ctx[t].t_ms = 0;
    }
    int packchk = -1;
    {   /* PACKCHK: HVX weight pack must be byte-identical to the scalar packer (else oc breaks). */
        int hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);
        static int16_t wcv[GP_BB]; static uint8_t wA[W16MM_WT_BYTES], wB[W16MM_WT_BYTES];
        static int32_t bA[W16MM_BIAS_BYTES / 4], bB[W16MM_BIAS_BYTES / 4];
        for (int i = 0; i < GP_BB; ++i) { int v = (int)((i * 2654435761u >> 15) & 0xffff) - 32768;
            wcv[i] = (int16_t)(v > 32639 ? 32639 : (v < -32639 ? -32639 : v)); }
        for (int i = 0; i < GP_BB; ++i) g_ctx[0].AA[i] = wcv[i];   /* cron#74: wt-pack gathers from VTCM-resident w_cv -> stage test cv into VTCM */
        gp_pack_wt_bias_hvx(g_ctx[0].AA, g_ctx[0].stage, wA, bA);
        for (int i = 0; i < GP_BB; ++i) g_ctx[0].lin[i] = wcv[g_lut[i]];
        w16a16_pack_wt_kmajor(g_ctx[0].lin, wB, W16MM_K, W16MM_N);
        w16a16_pack_bias(g_ctx[0].lin, bB, W16MM_K, W16MM_N);
        packchk = 0;
        for (int i = 0; i < W16MM_WT_BYTES; ++i) if (wA[i] != wB[i]) ++packchk;
        for (int i = 0; i < W16MM_BIAS_BYTES / 4; ++i) if (bA[i] != bB[i]) ++packchk;
        if (hvx == 0) qurt_hvx_unlock();
        FARF(ALWAYS, "GDN_PURE PACKCHK diff=%d (0 = HVX wt-pack byte-exact vs scalar)", packchk);
    }
#ifdef GP_ALIGN
    {   /* BIT-EXACT vs NATIVE `ConvLayer_s1` (cron#66/67, 2026-06-14). dense n_tiles=8 native M=64 descriptor +
         * native control words + CLOSED-FORM crouton layout (no dumped tables). Output codes -> T; host
         * compares to native Cout DIRECTLY (no CPU). GP_ALIGN_NCASE inputs (default 4) guard false-positive.
         *
         * crouton_pos(r,c) = the native M=64 act/out tile layout, a PURE BIT-PERMUTATION (reverse-engineered
         * from ramp-dumps, cron#67; bit-exact reproduces both dumped tables). act[crouton_pos(r,c)]=Aref[r,c];
         * outlin[r*64+c]=raw_out[crouton_pos(r,c)]. CROUTON_POS macro is file-scope (shared with GP_CROUTON8). */
        w16a16_mm_t *m = &g_ctx[0].mm;
        static uint16_t Aref[4096]; static int16_t Wq[4096]; static uint16_t outlin[4096];
        hmx_conv_out_desc_t *od = (hmx_conv_out_desc_t *)m->od; hmx_conv_act_desc_t *ad = (hmx_conv_act_desc_t *)m->ad;
        uint8_t *act = (uint8_t *)m->act, *out = (uint8_t *)m->out;
        uint16_t *Tu = (uint16_t *)T;
        /* M=256 carrier per-call (口径④) for the doc comparison: default w16a16_mm_init descriptor
         * (out_y=256,m_total=1,n_tiles=256,act_y=128) = the production primitive (4x 64-row blocks/call). */
        uint64_t cyc256 = 0, cyc256n32 = 0; int n32_byteid = -1;
        {   /* n_tiles=256 = original over-walk; n_tiles=32 = shape-minimum (new default). Verify the
             * default trim is BYTE-IDENTICAL output (only faster), then time both. */
            static uint16_t out256[W16MM_OUT_BYTES/2];
            od->n_tiles_pow2 = 256u; memset(m->out,0,W16MM_OUT_BYTES); w16a16_mm_run(m);
            for (int i=0;i<W16MM_OUT_BYTES/2;++i) out256[i]=((uint16_t*)m->out)[i];
            od->n_tiles_pow2 = 32u;  memset(m->out,0,W16MM_OUT_BYTES); w16a16_mm_run(m);
            n32_byteid = 0; for (int i=0;i<W16MM_OUT_BYTES/2;++i) if (((uint16_t*)m->out)[i]!=out256[i]) ++n32_byteid;
            od->n_tiles_pow2 = 256u;
            for (int i = 0; i < 32; ++i) w16a16_mm_run(m);
            uint64_t b0 = gp_pcyc(); for (int i = 0; i < 200; ++i) w16a16_mm_run(m); cyc256 = (gp_pcyc()-b0)/200;
            od->n_tiles_pow2 = 32u;
            for (int i = 0; i < 32; ++i) w16a16_mm_run(m);
            b0 = gp_pcyc(); for (int i = 0; i < 200; ++i) w16a16_mm_run(m); cyc256n32 = (gp_pcyc()-b0)/200;
        }
        for (int i = 0; i < 16; ++i) { m->atab[i] = (int32_t)(uintptr_t)(act + (size_t)(i & 3) * 2048);
                                       m->otab[i] = (int32_t)(uintptr_t)(out + (size_t)(i & 3) * 2048); }
        od->out_table_stride_dwords = 2u; od->out_y_stride_words = 4u; od->n_tiles_pow2 = 8u;
        od->m_total_minus_step = 8u; od->k_total_bytes = 64u;
        ad->n_act_pairs = 2u; ad->act_table_y_stride_words = 4u;
#ifndef GP_ALIGN_NCASE
#define GP_ALIGN_NCASE 4
#endif
        uint64_t cyc8 = 0;
        for (int v = 0; v < GP_ALIGN_NCASE; ++v) {   /* multi-case false-positive guard (cron#47 lesson) */
            for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c) {
                unsigned a, w;
                switch (v) {   /* distinct large-range patterns + a strictly-lower structured case (v==3) */
                    case 0:  a=(unsigned)(r*97+c*53);      w=(unsigned)(r*131+c*71);      break;
                    case 1:  a=(unsigned)(r*61+c*29+7);    w=(unsigned)(r*43+c*101+13);   break;
                    case 2:  a=(unsigned)(r*199+c*7+5000); w=(unsigned)(r*17+c*251+999);  break;
                    default: a=(unsigned)(r*97+c*53);      w=(r>c)?(unsigned)(r*131+c*71):0u; break; /* strictly-lower wt */
                }
                Aref[r*64+c] = (uint16_t)(32768 + (int)(a % 30000u) - 15000);
                Wq[r*64+c]   = w16a16_clip_q16((int)(w % 30000u) - 15000);
            }
            {   uint16_t *ao = (uint16_t *)m->act;            /* closed-form act layout */
                for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c) ao[CROUTON_POS(r,c)] = Aref[r*64+c];
            }
            w16a16_pack_wt_kmajor(Wq, m->wt, 64, 64);
            w16a16_pack_bias(Wq, (int32_t *)m->bias, 64, 64);
            for (int g = 0; g < 4; ++g) for (int i = 0; i < 16; ++i) {   /* native control = sA*sB/sC drain */
                m->bias[g*64 + 2*i]     = (int32_t)0x804035F3;
                m->bias[g*64 + 2*i + 1] = (int32_t)0x4000023E;
            }
            memset(m->out, 0, W16MM_OUT_BYTES);
            w16a16_mm_run(m);
            {   const uint16_t *so = (const uint16_t *)m->out;   /* closed-form untile */
                for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c) outlin[r*64+c] = so[CROUTON_POS(r,c)];
            }
            if (v < 8) for (int i = 0; i < 4096; ++i) Tu[v*4096 + i] = outlin[i];   /* T[v*4096..] = case v out */
            if (v == 0) {   /* per-call cyc on resident operands (口径④, target native ~1577) */
                for (int i = 0; i < 64; ++i) w16a16_mm_run(m);
                uint64_t b0 = gp_pcyc(); for (int i = 0; i < 500; ++i) w16a16_mm_run(m); cyc8 = (gp_pcyc()-b0)/500;
            }
        }
        FARF(ALWAYS, "GDN_PURE GP_ALIGN: M64 dense n_tiles=8=%llu cyc (native ~1577) vs M256 carrier n_tiles=256=%llu cyc (=%llu/64^3-equiv); %d cases -> T",
             (unsigned long long)cyc8, (unsigned long long)cyc256, (unsigned long long)(cyc256/4), GP_ALIGN_NCASE);
        if (statsLen > 0) stats[0] = (int)cyc8;
        if (statsLen > 1) stats[1] = (int)cyc256;
        if (statsLen > 2) stats[2] = (int)cyc256n32;
        if (statsLen > 3) stats[3] = n32_byteid;   /* 0 = n_tiles=32 (new default) byte-identical to 256 */
        HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx); if (vctx) HAP_compute_res_release(vctx);
        return 0;
    }
#endif
    {   /* SINGLE-64³-MATMUL micro-bench (answers the hard口径 gate): back-to-back w16a16_mm_run on
         * resident VTCM operands -> true per-call throughput (NOT the consumer-loop aggregate). main
         * thread holds the HMX lock. data-independent (mxmem cyc fixed). Warmup-excluded steady state
         * ~8.2K (vs QNN compact M=64 = 3,349 -> 2.4x headroom = compact row4-crouton act layout, cron#18). */
        const int B = 2000;
        for (int i = 0; i < 64; ++i) w16a16_mm_run(&g_ctx[0].mm);   /* warmup (drop cold-start fill) */
        uint64_t b0 = gp_pcyc();
        for (int i = 0; i < B; ++i) w16a16_mm_run(&g_ctx[0].mm);
        uint64_t per = (gp_pcyc() - b0) / B;
        if (statsLen > 5) stats[5] = (int)per;
        FARF(ALWAYS, "GDN_PURE MM64-bench: %llu cyc/call (single 64^3 w16a16, resident, back-to-back, steady)", (unsigned long long)per);
    }
    {   /* SELF-MEASURED dominant-path (cron#69, no QNN): sweep descriptor n_tiles = the mxmem MAC-walk
         * count. n_tiles=8 = the EXACT 64^3 MAC; >8 = bit-identical REDUNDANT walks. So wall(nt) =
         * fixed_overhead(load act+wt into mxmem, drain out, setup) + nt * per_walk_mxmem. The SLOPE
         * isolates pure mxmem (per_walk); slope*8 = the pure-conv cost = native ConvLayer_s1's 370
         * domain; the INTERCEPT = the per-call feed (= what native splits off into weights_to_vtcm /
         * OutputSlice). Pure baremetal C15:14, data-independent. */
        hmx_conv_out_desc_t *od = (hmx_conv_out_desc_t *)g_ctx[0].mm.od;
        uint32_t save = od->n_tiles_pow2;
        const int B = 1000; const int NTS[4] = { 8, 16, 32, 64 }; uint64_t r[4];
        for (int i = 0; i < 4; ++i) {
            od->n_tiles_pow2 = (uint32_t)NTS[i];
            for (int w = 0; w < 64; ++w) w16a16_mm_run(&g_ctx[0].mm);   /* warmup */
            uint64_t b0 = gp_pcyc();
            for (int rr = 0; rr < B; ++rr) w16a16_mm_run(&g_ctx[0].mm);
            r[i] = (gp_pcyc() - b0) / B;
            if (statsLen > 20 + i) stats[20 + i] = (int)r[i];
        }
        od->n_tiles_pow2 = save;
        /* slope = per-walk mxmem; intercept = per-call fixed feed; conv8 = slope*8 (vs native 370). */
        long slope = (long)(r[3] - r[0]) / (64 - 8);
        long conv8 = slope * 8, intercept = (long)r[0] - conv8;
        FARF(ALWAYS, "GDN_PURE NTSWEEP nt8=%llu nt16=%llu nt32=%llu nt64=%llu | per-walk=%ld pure-conv(8walk)=%ld fixed-feed=%ld",
             (unsigned long long)r[0], (unsigned long long)r[1], (unsigned long long)r[2], (unsigned long long)r[3], slope, conv8, intercept);
        if (statsLen > 24) stats[24] = (int)slope;
        if (statsLen > 25) stats[25] = (int)conv8;
        if (statsLen > 26) stats[26] = (int)intercept;
    }
    {   /* DISTINCT-TILE test (cron#70, no QNN): native QNN op lays act/out into 32 DISTINCT contiguous tiles
         * (act_tbl[i]=base+i*2048, dumped via custom-op DESC_DUMP); our crouton8 collapses to 4 reused
         * (i&3)*2048. Hypothesis: reused tiles serialize mxmem (loads to the same VTCM addr/bank can't
         * pipeline) -> per-walk 165 vs native 46. Repoint atab/otab to 32 distinct tiles in the (idle,
         * pre-spawn) ctx1 VTCM region, measure n_tiles=8 per-call. Timing-only (data-independent). */
        hmx_conv_out_desc_t *od = (hmx_conv_out_desc_t *)g_ctx[0].mm.od;
        int32_t *atab = g_ctx[0].mm.atab, *otab = g_ctx[0].mm.otab;
        int32_t sa[32], so2[32];
        for (int i = 0; i < 32; ++i) { sa[i] = atab[i]; so2[i] = otab[i]; }
        uint8_t *scr = (uint8_t *)g_ctx[0].mm.act + GP_VSTRIDE;        /* ctx1 region (free pre-spawn) */
        uint8_t *scro = scr + 0x10000;                                 /* +64KB for out tiles */
        for (int i = 0; i < 32; ++i) {
            atab[i] = (int32_t)(uintptr_t)(scr + (size_t)i * 2048);
            otab[i] = (int32_t)(uintptr_t)(scro + (size_t)i * 2048);
        }
        uint32_t save2 = od->n_tiles_pow2; od->n_tiles_pow2 = 8u;
        for (int w = 0; w < 64; ++w) w16a16_mm_run(&g_ctx[0].mm);
        uint64_t b0 = gp_pcyc();
        for (int rr = 0; rr < 1000; ++rr) w16a16_mm_run(&g_ctx[0].mm);
        uint64_t per = (gp_pcyc() - b0) / 1000;
        od->n_tiles_pow2 = save2;
        for (int i = 0; i < 32; ++i) { atab[i] = sa[i]; otab[i] = so2[i]; }
        if (statsLen > 27) stats[27] = (int)per;
        FARF(ALWAYS, "GDN_PURE DISTINCT-TILE nt8 32-distinct=%llu cyc (vs 4-reused; native 370)",
             (unsigned long long)per);
    }
#if defined(GP_MMBATCH) && defined(GP_TRACE)
    {   /* PURE-MATMUL BATCH DEMO (-DGP_MMBATCH -DGP_TRACE): N independent 64³ w16a16 matmuls back-to-back
         * on resident VTCM operands — NO solve, NO producer/consumer glue. Each = one MATMUL op (stage 3,
         * HMX tid256). Serialized into T as a chrometrace trace → gdn_trace_to_chrometrace.py reports the
         * matmul-op cyc/instance, apples-to-apples vs QNN native q::ConvLayer_s1.opt (~1,204-1,430/op). */
        const int N = 128;
        for (int i = 0; i < 32; ++i) w16a16_mm_run(&g_ctx[0].mm);   /* warmup */
        g_ev_n = 0;
        uint64_t base = gp_pcyc();
        for (int i = 0; i < N; ++i) {
            uint64_t t0 = gp_pcyc(); w16a16_mm_run(&g_ctx[0].mm); uint64_t t1 = gp_pcyc();
            GP_EV(GP_NT, 3, t0, t1);                                /* MATMUL op */
        }
        uint64_t wall = gp_pcyc() - base;
        int n = g_ev_n; uint8_t *tb = (uint8_t *)T;
        *(uint32_t *)(tb + 0) = 0x47545203u; *(uint32_t *)(tb + 4) = (uint32_t)n;
        *(uint64_t *)(tb + 8) = wall; *(uint64_t *)(tb + 16) = 0;
        uint8_t *p = tb + 24;
        for (int i = 0; i < n; ++i) {
            *(uint32_t *)(p + 0) = g_ev[i].tid; *(uint32_t *)(p + 4) = g_ev[i].stage;
            *(uint64_t *)(p + 8) = g_ev[i].t0 - base; *(uint64_t *)(p + 16) = g_ev[i].t1 - base; p += 24;
        }
        FARF(ALWAYS, "GDN_PURE MMBATCH: %d matmul ops serialized into T (wall=%llu, %llu/op)",
             n, (unsigned long long)wall, (unsigned long long)(wall / N));
        if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
        if (vctx) HAP_compute_res_release(vctx);
        return 0;                                                   /* demo only — skip the solve */
    }
#endif
    {   /* O7 STEP 2: M-sweep fan-out amortization probe (cron, 2026-06-14). SAME kernel, descriptor M=128/256
         * does 2/4 independent 64-row M-blocks in ONE prologue/epilogue (M-fan-out, shared-weight). atab is
         * already ×2048 for 64 row-groups (mm_init). Linear fit cyc(M)=fixed+per_block*(M/64) -> 'fixed' =
         * the per-call overhead that batching amortizes. If cyc(M256)/4 << cyc(M64) -> fan-out lever real;
         * if cyc(M256)≈4×cyc(M64) -> cost is per-tile feed, batching won't help (skill warning gate). */
        hmx_conv_out_desc_t *od = (hmx_conv_out_desc_t *)g_ctx[0].mm.od;
        uint32_t sy = od->out_y_stride_words, sn = od->n_tiles_pow2;
        const int B = 1000; uint64_t r[2] = {0, 0}; const uint32_t Ms[2] = {128u, 256u};
        for (int mi = 0; mi < 2; ++mi) {
            od->out_y_stride_words = Ms[mi]; od->n_tiles_pow2 = Ms[mi];
            for (int i = 0; i < 32; ++i) w16a16_mm_run(&g_ctx[0].mm);   /* warmup */
            uint64_t b0 = gp_pcyc();
            for (int i = 0; i < B; ++i) w16a16_mm_run(&g_ctx[0].mm);
            r[mi] = (gp_pcyc() - b0) / B;
        }
        od->out_y_stride_words = sy; od->n_tiles_pow2 = sn;            /* restore 64³ */
        if (statsLen > 18) stats[18] = (int)r[0];                      /* M=128 call cyc */
        if (statsLen > 19) stats[19] = (int)r[1];                      /* M=256 call cyc */
        FARF(ALWAYS, "GDN_PURE O7-probe: M64=%d M128=%llu M256=%llu cyc/call; per-64block: M256/4=%llu vs M64=%d",
             stats[5], (unsigned long long)r[0], (unsigned long long)r[1],
             (unsigned long long)(r[1] / 4), stats[5]);
    }
#ifdef GP_NATIVE_DESC
    {   /* NATIVE-DESCRIPTOR SWEEP (cron, 2026-06-14, GP_NATIVE_DESC). Tests the cron#41 vendor-dumped
         * native batched-64³ convhhh descriptor {out_y=4,n_tiles=8,m_total=8,act_y=4} on OUR working
         * ×2048 atab layout, with KNOWN 64³ inputs + CPU reference (maxdiff) + cyc. Hypothesis: our
         * atab repeats every 8 row-groups (rg&7), so n_tiles=64 does 8× REDUNDANT MAC over 8 distinct
         * tiles; native's n_tiles=8 = one pass = same output ~8× faster. cron#35 only swept n_tiles
         * DOWN to 4 (< 8 distinct -> dropped tiles -> garbage); it never tried exactly 8. Each variant
         * isolates one field. Restores the working desc for the solve. */
        w16a16_mm_t *m = &g_ctx[0].mm;
        hmx_conv_out_desc_t *od = (hmx_conv_out_desc_t *)m->od;
        hmx_conv_act_desc_t *ad = (hmx_conv_act_desc_t *)m->ad;
        hmx_conv_out_desc_t od_save = *od; hmx_conv_act_desc_t ad_save = *ad;
        static uint16_t Aa[4096], Ab[4096]; static int16_t Wqa[4096], Wqb[4096];
        static uint16_t outlin[4096]; static uint16_t goldenB[4096];
        for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c) {  /* two DIFFERENT operands A,B (non-saturated) */
            Aa[r*64+c] = (uint16_t)(32768 + (((r*7 + c*3) % 97) - 48) * 5);
            Wqa[r*64+c] = (int16_t)((((r*5 + c*11) % 127) - 63) * 24);
            Ab[r*64+c] = (uint16_t)(32768 + (((r*13 + c*5) % 89) - 44) * 6);
            Wqb[r*64+c] = (int16_t)((((r*3 + c*7) % 113) - 56) * 27);
        }
        /* goldenB = correct B·Wb output (NT=64, cleared out). */
        od->n_tiles_pow2 = 64u;
        w16a16_pack_act_crouton16(Ab, (uint16_t *)m->act, 64, 64); w16a16_pack_wt_kmajor(Wqb, m->wt, 64, 64);
        w16a16_pack_bias(Wqb, (int32_t *)m->bias, 64, 64);
        memset(m->out, 0, W16MM_OUT_BYTES); w16a16_mm_run(m);
        w16a16_depack_crouton16((const uint16_t *)m->out, goldenB, 64, 64);
        /* n_tiles sweep: STALE-FAITHFUL (mirrors solve: out NOT cleared between matmuls). Run A·Wa, then
         * B·Wb WITHOUT memset; if NT writes the FULL output, A's stale tiles are fully overwritten ->
         * matches goldenB. If NT partial -> A leaks -> mismatch. = the exact condition the solve needs. */
        const uint32_t NT[] = { 64u, 48u, 40u, 32u, 24u, 20u, 18u, 16u };
        const int NV = (int)(sizeof(NT) / sizeof(NT[0])), B = 500;
        for (int v = 0; v < NV; ++v) {
            od->n_tiles_pow2 = NT[v];
            /* fill out with A's result (stale source) */
            w16a16_pack_act_crouton16(Aa, (uint16_t *)m->act, 64, 64); w16a16_pack_wt_kmajor(Wqa, m->wt, 64, 64);
            w16a16_pack_bias(Wqa, (int32_t *)m->bias, 64, 64);
            memset(m->out, 0, W16MM_OUT_BYTES); w16a16_mm_run(m);
            /* now B WITHOUT clearing out (exactly like the solve consumer) */
            w16a16_pack_act_crouton16(Ab, (uint16_t *)m->act, 64, 64); w16a16_pack_wt_kmajor(Wqb, m->wt, 64, 64);
            w16a16_pack_bias(Wqb, (int32_t *)m->bias, 64, 64);
            w16a16_mm_run(m);
            w16a16_depack_crouton16((const uint16_t *)m->out, outlin, 64, 64);
            int maxd = 0, nz = 0;
            for (int i = 0; i < 4096; ++i) { int d = (int)outlin[i] - (int)goldenB[i]; if (d < 0) d = -d;
                if (d > maxd) maxd = d; if (outlin[i] != goldenB[i]) ++nz; }   /* mismatch vs correct B = stale leak */
            for (int i = 0; i < 8; ++i) w16a16_mm_run(m);              /* warmup for timing */
            uint64_t b0 = gp_pcyc();
            for (int i = 0; i < B; ++i) w16a16_mm_run(m);
            uint64_t cyc = (gp_pcyc() - b0) / B;
            FARF(ALWAYS, "GDN_PURE NTSWEEP n_tiles=%-3u: maxdiff=%d nmismatch=%d cyc=%llu (stale-faithful)",
                 NT[v], maxd, nz, (unsigned long long)cyc);
            if (statsLen > v * 2 + 1) { stats[v * 2] = maxd; stats[v * 2 + 1] = (int)cyc; }  /* [v*2]=maxdiff [v*2+1]=cyc */
        }
        *od = od_save; *ad = ad_save;                                 /* restore working 64³ desc */
        if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
        if (vctx) HAP_compute_res_release(vctx);
        return 0;                                                     /* sweep only — skip the solve */
    }
#endif
#ifdef GP_DENSE8
    {   /* DENSE-8-TILE sweep (cron, 2026-06-14, GP_DENSE8). native warm matmul-op = 335 via n_tiles=8 over
         * a DENSE 4-tile act ([1,8,8,64], atab[0..3]=0/2048/4096/6144 = mm_init's first 4). pack_act_crouton16
         * fills exactly 4 contiguous 2KB tiles. 2 byte-passes (n_act=2) re-read them. Test dense atab variants
         * + n_tiles to find the bit-exact + fast config -> per-call toward native 335-1547. CPU-ref correctness
         * (maxd~190 = drain rounding = OK, like V0). cyc = back-to-back resident. */
        w16a16_mm_t *m = &g_ctx[0].mm;
        hmx_conv_out_desc_t *od = (hmx_conv_out_desc_t *)m->od; hmx_conv_act_desc_t *ad = (hmx_conv_act_desc_t *)m->ad;
        hmx_conv_out_desc_t od_save = *od; hmx_conv_act_desc_t ad_save = *ad;
        static int32_t a_save[128], o_save[128];
        for (int i = 0; i < 128; ++i) { a_save[i] = m->atab[i]; o_save[i] = m->otab[i]; }
        static uint16_t Aref[4096]; static int16_t Wq[4096]; static uint16_t outlin[4096]; static int16_t cvbuf[4096];
        for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c) {
            Aref[r*64+c] = (uint16_t)(32768 + (((r*7+c*3)%97)-48)*5);
            Wq[r*64+c]   = (int16_t)((((r*5+c*11)%127)-63)*24);
        }
        w16a16_pack_wt_kmajor(Wq, m->wt, 64, 64); w16a16_pack_bias(Wq, (int32_t *)m->bias, 64, 64);
        uint8_t *act = (uint8_t *)m->act, *out = (uint8_t *)m->out;
        /* variant: {label, atab-id, n_tiles, actfill}. atab: 0=mod4 1=pairs 2=linear8; -1=working16.
         * actfill: 0=pack_act_crouton16(direct) 1=cv->gp_cv_to_surf(solve act path,needs GP_DENSE_SURF). */
        struct { const char *lbl; int aid; uint32_t nt; int af; } V[] = {
            { "D0 working n64",       -1, 64u, 0 },
            { "D1 dense mod4 nt8",     0,  8u, 0 },
            { "D5 cv-path mod4 nt8",   0,  8u, 1 },   /* = solve act path (gp_cv_to_surf dense) */
            { "D6 cv-path mod4 nt64",  0, 64u, 1 },
            { "D7 cv-path 16wrap n64",-1, 64u, 1 },   /* solve act path on WORKING atab */
        };
        const int NV = (int)(sizeof(V)/sizeof(V[0])), B = 500;
        for (int v = 0; v < NV; ++v) {
            if (V[v].af == 0) w16a16_pack_act_crouton16(Aref, (uint16_t *)m->act, 64, 64);  /* direct */
            else { for (int i=0;i<4096;++i) cvbuf[g_lut[i]] = (int16_t)((int)Aref[i]-32768);  /* cv in g_lut order */
                   gp_cv_to_surf((uint16_t *)m->act, cvbuf); }                               /* solve act path */
            if (V[v].aid >= 0) {
                for (int i = 0; i < 16; ++i) {
                    int tile = (V[v].aid == 0) ? (i & 3) : (V[v].aid == 1) ? ((i >> 1) & 3) : (i & 7);
                    m->atab[i] = (int32_t)(uintptr_t)(act + (size_t)tile * 2048);
                    m->otab[i] = (int32_t)(uintptr_t)(out + (size_t)tile * 2048);
                }
                od->n_tiles_pow2 = V[v].nt; od->out_y_stride_words = 64u; od->m_total_minus_step = 1;
                ad->act_table_y_stride_words = 128u;
            } else { for (int i=0;i<128;++i){m->atab[i]=a_save[i];m->otab[i]=o_save[i];} *od=od_save;*ad=ad_save; od->n_tiles_pow2=V[v].nt; }
            memset(m->out, 0, W16MM_OUT_BYTES);
            w16a16_mm_run(m);
            w16a16_depack_crouton16((const uint16_t *)m->out, outlin, 64, 64);
            int maxd = 0;
            for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c) {
                long acc = 0; for (int kk = 0; kk < 64; ++kk) acc += (long)((int)Aref[r*64+kk]-32768)*(int)Wq[kk*64+c];
                long d = acc/32767 - ((int)outlin[r*64+c]-32768); if (d<0) d=-d; if (d>maxd) maxd=(int)d;
            }
            for (int i = 0; i < 8; ++i) w16a16_mm_run(m);
            uint64_t b0 = gp_pcyc(); for (int i = 0; i < B; ++i) w16a16_mm_run(m); uint64_t cyc = (gp_pcyc()-b0)/B;
            FARF(ALWAYS, "GDN_PURE DENSE8 %-22s: maxdiff=%d cyc=%llu (CPU-ref; ~190=OK drain)", V[v].lbl, maxd, (unsigned long long)cyc);
            if (statsLen > v*2+1) { stats[v*2] = maxd; stats[v*2+1] = (int)cyc; }
        }
        {   /* gp_surf_to_cv chain test: matmul -> gp_surf_to_cv(out_cv) -> gp_pack_blk(->linear) vs CPU.
             * tests the OUTPUT readback that feeds the next matmul (D5 used depack_crouton16, not this). */
            static int16_t out_cv[4096], To[64*256];
            w16a16_pack_act_crouton16(Aref, (uint16_t *)m->act, 64, 64);
            for (int i = 0; i < 16; ++i) { int t=i&3; m->atab[i]=(int32_t)(uintptr_t)(act+(size_t)t*2048);
                m->otab[i]=(int32_t)(uintptr_t)(out+(size_t)t*2048); }
            od->n_tiles_pow2=8u; od->out_y_stride_words=64u; od->m_total_minus_step=1; ad->act_table_y_stride_words=128u;
            memset(m->out,0,W16MM_OUT_BYTES); w16a16_mm_run(m);
            gp_surf_to_cv(out_cv, (const uint16_t *)m->out);
            for (int i=0;i<64*256;++i) To[i]=0;
            gp_pack_blk(To, out_cv, g_fl, g_ctx[0].stage);
            int sd=0; for (int r=0;r<64;++r) for (int c=0;c<64;++c){
                long acc=0; for(int kk=0;kk<64;++kk) acc+=(long)((int)Aref[r*64+kk]-32768)*(int)Wq[kk*64+c];
                long d=acc/32767 - (int)To[r*256+c]; if(d<0)d=-d; if(d>sd)sd=(int)d; }
            FARF(ALWAYS,"GDN_PURE DENSE8 surf_to_cv chain maxdiff=%d (~190=ok)", sd);
            if (statsLen>17) stats[17]=sd;
        }
        {   /* full mm64 composition test (gp_cv_to_surf + gp_pack_wt_bias_hvx weight + matmul + gp_surf_to_cv):
             * A_cv @ A_cv -> AA_cv, compare to CPU A@A. tests the SOLVE weight path inside a real matmul. */
            static int16_t Acv[4096], AAcv[4096], Taa[64*256], Aqs[64*256];
            for (int r=0;r<64;++r) for (int c=0;c<64;++c) Aqs[r*256+c]=(int16_t)((int)Aref[r*64+c]-32768);
            gp_unpack_blk(Acv, Aqs, g_il, g_ctx[0].stage);   /* SOLVE input path (not direct cvbuf) */
            { hmx_conv_out_desc_t *od2=(hmx_conv_out_desc_t*)m->od; hmx_conv_act_desc_t *ad2=(hmx_conv_act_desc_t*)m->ad;
              for (int i=0;i<16;++i){int t=i&3; m->atab[i]=(int32_t)(uintptr_t)(act+(size_t)t*2048); m->otab[i]=(int32_t)(uintptr_t)(out+(size_t)t*2048);}
              od2->n_tiles_pow2=8u; od2->out_y_stride_words=64u; od2->m_total_minus_step=1; ad2->act_table_y_stride_words=128u; }
            static int16_t A3cv[4096]; static int CPUaa[4096];
            int sl=g_ctx[0].slot; g_ctx[0].slot=-1;
            mm64(&g_ctx[0], Acv, Acv, AAcv);          /* AA = A@A */
            mm64(&g_ctx[0], AAcv, Acv, A3cv);         /* A3 = AA@A (CHAINED: AA fed back as act) */
            g_ctx[0].slot=sl;
            for (int i=0;i<64*256;++i) Taa[i]=0;
            gp_pack_blk(Taa, A3cv, g_fl, g_ctx[0].stage);
            /* CPU AA then A3 (code space, /32767 drain each) */
            for (int r=0;r<64;++r) for (int c=0;c<64;++c){ long a=0; for(int kk=0;kk<64;++kk) a+=(long)((int)Aref[r*64+kk]-32768)*((int)Aref[kk*64+c]-32768); CPUaa[r*64+c]=(int)(a/32767); }
            int md=0; for (int r=0;r<64;++r) for (int c=0;c<64;++c){
                long acc=0; for(int kk=0;kk<64;++kk) acc+=(long)CPUaa[r*64+kk]*((int)Aref[kk*64+c]-32768);
                long d=acc/32767-(int)Taa[r*256+c]; if(d<0)d=-d; if(d>md)md=(int)d; }
            FARF(ALWAYS,"GDN_PURE DENSE8 mm64 CHAIN A3=AA@A maxdiff=%d (small=ok)", md);
            if (statsLen>14) stats[14]=md;
        }
        {   /* full diag_inv test: strictly-lower A -> X=(I-A)^-1 Taylor(3). reconstruct X_v=X_code*2^eX/32767,
             * compare to CPU I+A+A²+A³ (relative). This is the actual failing unit (T_ii diag blocks). */
            static int16_t Aq2[64*256] __attribute__((aligned(128))), Acv2[4096] __attribute__((aligned(128))),
                           Xcv[4096] __attribute__((aligned(128))), Xlin[64*256] __attribute__((aligned(128)));
            static double Av[64][64], X0[64][64];
            for (int r=0;r<64;++r) for (int c=0;c<64;++c){ int v = (r>c) ? (((r*5+c*7)%41)-20)*80 : 0; /* ~scale0.05 */
                Aq2[r*256+c]=(int16_t)v; Av[r][c]=v/32767.0; }
            gp_unpack_blk(Acv2, Aq2, g_il, g_ctx[0].stage);
            int sl=g_ctx[0].slot; g_ctx[0].slot=-1;
            int eX = diag_inv(&g_ctx[0], Acv2, Xcv);
            g_ctx[0].slot=sl;
            for (int i=0;i<64*256;++i) Xlin[i]=0; gp_pack_blk(Xlin, Xcv, g_fl, g_ctx[0].stage);
            /* CPU X0 = I + A + A^2 + A^3 */
            static double A2[64][64], A3[64][64];
            for (int r=0;r<64;++r) for (int c=0;c<64;++c){ double s=0; for(int k=0;k<64;++k) s+=Av[r][k]*Av[k][c]; A2[r][c]=s; }
            for (int r=0;r<64;++r) for (int c=0;c<64;++c){ double s=0; for(int k=0;k<64;++k) s+=A2[r][k]*Av[k][c]; A3[r][c]=s; }
            for (int r=0;r<64;++r) for (int c=0;c<64;++c) X0[r][c]=(r==c?1.0:0.0)+Av[r][c]+A2[r][c]+A3[r][c];
            double num=0,den=0; double sc=1.0; for(int i=0;i<eX;++i) sc*=2.0; for(int i=0;i<-eX;++i) sc/=2.0;
            for (int r=0;r<64;++r) for (int c=0;c<64;++c){ double xv=Xlin[r*256+c]*sc/32767.0; double d=xv-X0[r][c]; num+=d*d; den+=X0[r][c]*X0[r][c]; }
            int rel = (int)(10000.0 * (den>0? (num/den):0));   /* relerr^2 *1e4 */
            FARF(ALWAYS,"GDN_PURE DENSE8 diag_inv relerr^2*1e4=%d eX=%d (small=ok)", rel, eX);
            if (statsLen>13) stats[13]=rel;
        }
        {   /* 2-BLOCK MERGE test under dense (cron#65): T10 = T11 @ (A10 @ T00), matching solve_head's
             * forward-subst. Uses a COMPUTED T as the matmul WEIGHT across blocks (no passing test does).
             * CPU: t00=I+a00+a00^2+a00^3 (Taylor3, = device), t11 likewise, s=a10@t00, t10=t11@s. */
            static int16_t q00[64*256],q11[64*256],q10[64*256],B00[4096],B11[4096],B10[4096];
            static int16_t T00[4096],T11[4096],Sm[4096],T10[4096],Tln[64*256];
            static double a00[64][64],a11[64][64],a10[64][64];
            for (int r=0;r<64;++r) for (int c=0;c<64;++c){
                int v00=(r>c)?(((r*5+c*7)%41)-20)*10:0, v11=(r>c)?(((r*3+c*11)%37)-18)*10:0, v10=(((r*7+c*5)%53)-26)*10;
                q00[r*256+c]=(int16_t)v00; q11[r*256+c]=(int16_t)v11; q10[r*256+c]=(int16_t)v10;
                a00[r][c]=v00/32767.0; a11[r][c]=v11/32767.0; a10[r][c]=v10/32767.0; }
            gp_unpack_blk(B00,q00,g_il,g_ctx[0].stage); gp_unpack_blk(B11,q11,g_il,g_ctx[0].stage); gp_unpack_blk(B10,q10,g_il,g_ctx[0].stage);
            int sl2=g_ctx[0].slot; g_ctx[0].slot=-1;
            int e00=diag_inv(&g_ctx[0],B00,T00), e11=diag_inv(&g_ctx[0],B11,T11);
            gp_acc_zero(g_ctx[0].acc); mm64(&g_ctx[0],B10,T00,Sm);    /* S=A10@T00 (T00 = computed WEIGHT) */
            gp_acc_addsh(g_ctx[0].acc,Sm,0); int eS=e00+gp_renorm(g_ctx[0].acc,Sm);
            {   /* ISOLATE: S=A10@T00 vs CPU a10@t00cpu (computed-WEIGHT matmul) — needs t00 below; deferred check */
                static int16_t Sln[64*256]; for(int i=0;i<64*256;++i)Sln[i]=0; gp_pack_blk(Sln,Sm,g_fl,g_ctx[0].stage);
                static double t00x[64][64],tmpx[64][64]; for(int r=0;r<64;++r)for(int c=0;c<64;++c){double s2=0;for(int k=0;k<64;++k)s2+=a00[r][k]*a00[k][c];tmpx[r][c]=s2;}
                for(int r=0;r<64;++r)for(int c=0;c<64;++c){double s3=0;for(int k=0;k<64;++k)s3+=tmpx[r][k]*a00[k][c];t00x[r][c]=(r==c?1.0:0.0)+a00[r][c]+tmpx[r][c]+s3;}
                double num=0,den=0,sc=1.0; for(int i=0;i<eS;++i)sc*=2.0; for(int i=0;i<-eS;++i)sc/=2.0;
                for(int r=0;r<64;++r)for(int c=0;c<64;++c){double sx=0;for(int k=0;k<64;++k)sx+=a10[r][k]*t00x[k][c];double xv=Sln[r*256+c]*sc/32767.0;double d=xv-sx;num+=d*d;den+=sx*sx;}
                if(statsLen>15) stats[15]=(int)(1000000.0*(den>0?num/den:0)); }   /* S relerr^2 *1e6 (1st mm64, computed wt) */
            {   /* ISOLATE: mm64(T00, B00) = renorm-output T00 as ACT + input weight B00, vs CPU t00@a00 */
                static int16_t Pm[4096], Pln[64*256]; static double t00y[64][64],tmpy[64][64],p00[64][64];
                for(int r=0;r<64;++r)for(int c=0;c<64;++c){double s2=0;for(int k=0;k<64;++k)s2+=a00[r][k]*a00[k][c];tmpy[r][c]=s2;}
                for(int r=0;r<64;++r)for(int c=0;c<64;++c){double s3=0;for(int k=0;k<64;++k)s3+=tmpy[r][k]*a00[k][c];t00y[r][c]=(r==c?1.0:0.0)+a00[r][c]+tmpy[r][c]+s3;}
                gp_acc_zero(g_ctx[0].acc); mm64(&g_ctx[0],T00,B00,Pm); gp_acc_addsh(g_ctx[0].acc,Pm,0); int eP=e00+gp_renorm(g_ctx[0].acc,Pm);
                for(int i=0;i<64*256;++i)Pln[i]=0; gp_pack_blk(Pln,Pm,g_fl,g_ctx[0].stage);
                double num=0,den=0,sc=1.0; for(int i=0;i<eP;++i)sc*=2.0; for(int i=0;i<-eP;++i)sc/=2.0;
                for(int r=0;r<64;++r)for(int c=0;c<64;++c){double pv=0;for(int k=0;k<64;++k)pv+=t00y[r][k]*a00[k][c];double xv=Pln[r*256+c]*sc/32767.0;double d=xv-pv;num+=d*d;den+=pv*pv;}
                if(statsLen>11) stats[11]=(int)(1000000.0*(den>0?num/den:0)); }   /* T00@B00 relerr^2 *1e6 (renorm act) */
            {   /* WORKAROUND VALIDATION: T_ii@S via transpose(mm64(S^T, T_ii^T)) — the working operand order.
                 * compare to CPU T11_code@Sm_code/32767. if 0 => bit-exact + dense-speed fix CONFIRMED. */
                static int16_t STcv[4096], TTcv[4096], Qcv[4096];
                static int16_t lin[64*256], linT[64*256], QTln[64*256], T11l[64*256], Sml[64*256];
                for(int i=0;i<64*256;++i){lin[i]=0;linT[i]=0;} gp_pack_blk(lin,Sm,g_fl,g_ctx[0].stage);
                for(int r=0;r<64;++r)for(int c=0;c<64;++c) linT[c*256+r]=lin[r*256+c]; gp_unpack_blk(STcv,linT,g_il,g_ctx[0].stage);  /* S^T */
                for(int i=0;i<64*256;++i){lin[i]=0;linT[i]=0;} gp_pack_blk(lin,T11,g_fl,g_ctx[0].stage);
                for(int r=0;r<64;++r)for(int c=0;c<64;++c) linT[c*256+r]=lin[r*256+c]; gp_unpack_blk(TTcv,linT,g_il,g_ctx[0].stage);  /* T11^T */
                mm64(&g_ctx[0],STcv,TTcv,Qcv);                          /* Q = S^T @ T11^T = (T11@S)^T */
                for(int i=0;i<64*256;++i){lin[i]=0;QTln[i]=0;} gp_pack_blk(lin,Qcv,g_fl,g_ctx[0].stage);
                for(int r=0;r<64;++r)for(int c=0;c<64;++c) QTln[r*256+c]=lin[c*256+r];   /* Q^T = T11@S */
                for(int i=0;i<64*256;++i){T11l[i]=0;Sml[i]=0;} gp_pack_blk(T11l,T11,g_fl,g_ctx[0].stage); gp_pack_blk(Sml,Sm,g_fl,g_ctx[0].stage);
                int qb=0; for(int r=0;r<64;++r)for(int c=0;c<64;++c){ long acc=0; for(int k=0;k<64;++k) acc+=(long)T11l[r*256+k]*(long)Sml[k*256+c];
                    long ref=acc/32767,got=QTln[r*256+c],d=ref-got; if(d<0)d=-d; if(d>20)qb++; }
                if(statsLen>16) stats[16]=qb; }   /* transpose-workaround T11@S code-mismatch (0 = FIX works) */
            mm64(&g_ctx[0],T11,Sm,T10);                               /* T10=T11@S (T11,S computed WEIGHTs) */
            gp_acc_from_cv(g_ctx[0].acc,T10); int e10=e11+eS+gp_renorm(g_ctx[0].acc,T10);
            g_ctx[0].slot=sl2;
            for (int i=0;i<64*256;++i) Tln[i]=0; gp_pack_blk(Tln,T10,g_fl,g_ctx[0].stage);
            {   /* DEFINITIVE: CPU-replicate the exact CODE matmul T11_code@Sm_code/32767, vs device T10 codes.
                 * removes all Taylor/exponent reference uncertainty. mismatch => mm64 itself mis-computes. */
                static int16_t T11ln[64*256], Smln[64*256];
                for(int i=0;i<64*256;++i){T11ln[i]=0;Smln[i]=0;} gp_pack_blk(T11ln,T11,g_fl,g_ctx[0].stage); gp_pack_blk(Smln,Sm,g_fl,g_ctx[0].stage);
                int nbad=0,mxd=0; for(int r=0;r<64;++r)for(int c=0;c<64;++c){ long acc=0; for(int k=0;k<64;++k) acc+=(long)T11ln[r*256+k]*(long)Smln[k*256+c];
                    long ref=acc/32767, got=Tln[r*256+c], d=ref-got; if(d<0)d=-d; if(d>mxd)mxd=(int)d; if(d>20)nbad++; }
                if(statsLen>13) stats[13]=nbad;    /* # codes mm64(T11,Sm) differs from CPU code-matmul (>20) */
                if(statsLen>9)  stats[9]=mxd;      /* max code diff */
                /* ISOLATE packer: gp_pack_wt_bias_hvx(Sm) [= c->mm.wt/bias after mm64(T11,Sm)] vs reference
                 * w16a16_pack_wt_kmajor/pack_bias(Sm_linear). large-value/full-matrix Sm path (PACKCHK only
                 * checked a specific weight). mismatch => weight/bias byte pack is the dense bug. */
                static int16_t Sm64[4096]; static uint8_t wtref[W16MM_WT_BYTES]; static int32_t bsref[W16MM_WT_BYTES/4+64];
                for(int r=0;r<64;++r)for(int c=0;c<64;++c) Sm64[r*64+c]=Smln[r*256+c];
                w16a16_pack_wt_kmajor(Sm64, wtref, 64, 64); w16a16_pack_bias(Sm64, bsref, 64, 64);
                gp_pack_wt_bias_hvx(Sm, g_ctx[0].stage, g_ctx[0].mm.wt, g_ctx[0].mm.bias);   /* re-pack Sm weight */
                int wbad=0; for(int i=0;i<(int)W16MM_WT_BYTES;++i) if(((uint8_t*)g_ctx[0].mm.wt)[i]!=wtref[i]) wbad++;
                int bbad=0; for(int i=0;i<256;++i) if(g_ctx[0].mm.bias[i]!=bsref[i]) bbad++;
                if(statsLen>8)  stats[8]=wbad;     /* weight byte mismatch (HVX vs ref) for Sm */
                if(statsLen>10) stats[10]=bbad; }  /* bias mismatch (HVX vs ref) for Sm */
            /* CPU t00,t11 Taylor3 ; s=a10@t00 ; t10=t11@s */
            static double t00[64][64],t11[64][64],sm[64][64],t10c[64][64],tmp[64][64],tmp2[64][64];
            for (int r=0;r<64;++r) for (int c=0;c<64;++c){ double s2=0,s3=0; for(int k=0;k<64;++k) s2+=a00[r][k]*a00[k][c]; tmp[r][c]=s2; }
            for (int r=0;r<64;++r) for (int c=0;c<64;++c){ double s3=0; for(int k=0;k<64;++k) s3+=tmp[r][k]*a00[k][c]; t00[r][c]=(r==c?1.0:0.0)+a00[r][c]+tmp[r][c]+s3; }
            for (int r=0;r<64;++r) for (int c=0;c<64;++c){ double s2=0; for(int k=0;k<64;++k) s2+=a11[r][k]*a11[k][c]; tmp2[r][c]=s2; }
            for (int r=0;r<64;++r) for (int c=0;c<64;++c){ double s3=0; for(int k=0;k<64;++k) s3+=tmp2[r][k]*a11[k][c]; t11[r][c]=(r==c?1.0:0.0)+a11[r][c]+tmp2[r][c]+s3; }
            for (int r=0;r<64;++r) for (int c=0;c<64;++c){ double s=0; for(int k=0;k<64;++k) s+=a10[r][k]*t00[k][c]; sm[r][c]=s; }
            for (int r=0;r<64;++r) for (int c=0;c<64;++c){ double s=0; for(int k=0;k<64;++k) s+=t11[r][k]*sm[k][c]; t10c[r][c]=s; }
            { /* EXPONENT SWEEP: is it a wrong-value bug or a wrong-exponent (scale) bug? try e10+off */
              double best=1e30; int boff=0;
              for (int off=-4; off<=4; ++off){ double sc=1.0; int ee=e10+off; for(int i=0;i<ee;++i)sc*=2.0; for(int i=0;i<-ee;++i)sc/=2.0;
                double num=0,den=0; for (int r=0;r<64;++r) for (int c=0;c<64;++c){ double xv=Tln[r*256+c]*sc/32767.0; double d=xv-t10c[r][c]; num+=d*d; den+=t10c[r][c]*t10c[r][c]; }
                double re=(den>0?num/den:0); if(re<best){best=re;boff=off;} }
              if (statsLen>12) stats[12]=(int)(1000000.0*best);    /* MERGE best relerr^2 *1e6 over exp offsets */
              if (statsLen>10) stats[10]=boff; }                   /* best exponent offset (0 = exp ok -> value bug) */
        }
        {   /* cron#65 ★: native-descriptor vs ours on EXTREME act (A≈I). native graph proved n_tiles=8 works
             * for extreme acts; native desc = out_y=4/m_total=8/act_y=4/ots=2. test if that fixes OUR kernel. */
            static uint16_t Ae[4096], oe[4096]; static int16_t We[4096];
            for (int r=0;r<64;++r) for (int c=0;c<64;++c){ Ae[r*64+c]=(uint16_t)((r==c)?48768:32768);  /* A≈I extreme */
                We[r*64+c]=(int16_t)((((r*7+c*5)%53)-26)*460); }                                       /* full ±~12000 */
            w16a16_pack_act_crouton16(Ae,(uint16_t*)m->act,64,64); w16a16_pack_wt_kmajor(We,m->wt,64,64); w16a16_pack_bias(We,(int32_t*)m->bias,64,64);
            uint8_t *act2=(uint8_t*)m->act,*out2=(uint8_t*)m->out;
            struct { const char *l; uint32_t oy,nt; int32_t mt; uint32_t ay,ots; } DD[2] = {
                {"native(oy4/mt8/ay4)",4u,8u,8,4u,2u}, {"ours(oy64/mt1/ay128)",64u,8u,1,128u,2u} };
            for (int v=0; v<2; ++v){
                for(int i=0;i<16;++i){m->atab[i]=(int32_t)(uintptr_t)(act2+(size_t)(i&3)*2048); m->otab[i]=(int32_t)(uintptr_t)(out2+(size_t)(i&3)*2048);}
                od->out_table_stride_dwords=DD[v].ots; od->out_y_stride_words=DD[v].oy; od->n_tiles_pow2=DD[v].nt; od->m_total_minus_step=DD[v].mt; od->k_total_bytes=64u;
                ad->n_act_pairs=2u; ad->act_table_y_stride_words=DD[v].ay;
                memset(m->out,0,W16MM_OUT_BYTES); w16a16_mm_run(m); w16a16_depack_crouton16((const uint16_t*)m->out,oe,64,64);
                int md=0; for(int r=0;r<64;++r)for(int c=0;c<64;++c){ long acc=0; for(int k=0;k<64;++k) acc+=(long)((int)Ae[r*64+k]-32768)*(int)We[k*64+c];
                    long ref=acc/32767, got=(int)oe[r*64+c]-32768, d=ref-got; if(d<0)d=-d; if(d>md)md=(int)d; }
                if (v==0 && statsLen>18) stats[18]=md;   /* native-desc maxd (extreme act) */
                if (v==1 && statsLen>19) stats[19]=md;   /* our-desc maxd (extreme act) */
            }
            /* cron#65 ★: replicate native EXACTLY = native act+weight+desc + NATIVE bias control
             * (0x804035f3/0x4000023e, bit-31 set) -> depack -> T. host compares to native's cout_native.raw. */
            w16a16_pack_bias(We,(int32_t*)m->bias,64,64);   /* OUR control 0x00404420 (consistent w/ CPU /32767) */
            for(int i=0;i<16;++i){m->atab[i]=(int32_t)(uintptr_t)(act2+(size_t)(i&3)*2048); m->otab[i]=(int32_t)(uintptr_t)(out2+(size_t)(i&3)*2048);}
            od->out_table_stride_dwords=2u; od->out_y_stride_words=4u; od->n_tiles_pow2=8u; od->m_total_minus_step=8; od->k_total_bytes=64u;
            ad->n_act_pairs=2u; ad->act_table_y_stride_words=4u;
            memset(m->out,0,W16MM_OUT_BYTES); w16a16_mm_run(m); (void)oe;
            memcpy(T, m->out, 8192);   /* T = RAW m->out (no depack); host compares MULTISET vs cout_native.raw (readback-independent) */
            if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
            if (vctx) HAP_compute_res_release(vctx);
            return 0;
        }
        {   /* diag-add isolation: acc=0, +1000 to diagonal, ->cv->linear. check ONLY (d,d)=1000.
             * misplaced diag (under new g_lut) corrupts whole block via renorm absmax->shift. */
            static int16_t cvd[4096], Td[64*256];
            gp_acc_zero(g_ctx[0].acc);
            gp_acc_diag_add(g_ctx[0].acc, 1000);
            gp_acc_to_cv(cvd, g_ctx[0].acc, 0);
            for (int i=0;i<64*256;++i) Td[i]=0;
            gp_pack_blk(Td, cvd, g_fl, g_ctx[0].stage);
            int dbad=0, obad=0;
            for (int r=0;r<64;++r) for (int c=0;c<64;++c) { int v=Td[r*256+c];
                if (r==c) { if (v!=1000) dbad++; } else if (v!=0) obad++; }
            FARF(ALWAYS,"GDN_PURE DENSE8 diag-add: diag-wrong=%d off-diag-nonzero=%d (0,0=ok)", dbad, obad);
            if (statsLen>15) stats[15]=dbad*10000+obad;
        }
        {   /* cv<->acc round-trip (gp_acc_from_cv vsxt in / gp_acc_to_cv vasr out): is it IDENTITY?
             * If a permutation (not identity), the OLD g_lut absorbed it but pack-order g_lut breaks. */
            static int16_t cvin[4096], cvout[4096];
            for (int i=0;i<4096;++i) cvin[i]=(int16_t)((i*131+7)%1999 - 999);
            gp_acc_from_cv(g_ctx[0].acc, cvin);
            gp_acc_to_cv(cvout, g_ctx[0].acc, 0);
            int rt=0; for (int i=0;i<4096;++i) if (cvout[i]!=cvin[i]) rt++;
            FARF(ALWAYS,"GDN_PURE DENSE8 cv<->acc round-trip mismatches=%d/4096 (0=identity)", rt);
            if (statsLen>16) stats[16]=rt;
        }
        {   /* gp_unpack_blk/gp_pack_blk round-trip under current g_lut: A(strided int16)->cv->A'. */
            static int16_t Aq[64*256], Ap[64*256], cvb[4096];
            for (int r=0;r<64;++r) for (int c=0;c<64;++c) Aq[r*256+c]=(int16_t)((int)Aref[r*64+c]-32768);
            gp_unpack_blk(cvb, Aq, g_il, g_ctx[0].stage);
            for (int i=0;i<64*256;++i) Ap[i]=0;
            gp_pack_blk(Ap, cvb, g_fl, g_ctx[0].stage);
            int rt=0; for (int r=0;r<64;++r) for (int c=0;c<64;++c){ if(Ap[r*256+c]!=Aq[r*256+c]) rt++; }
            FARF(ALWAYS, "GDN_PURE DENSE8 unpack/pack round-trip mismatches=%d/4096 (0=ok)", rt);
            if (statsLen>11) stats[11]=rt;
        }
        for (int i=0;i<128;++i){m->atab[i]=a_save[i];m->otab[i]=o_save[i];} *od=od_save;*ad=ad_save;
        if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
        if (vctx) HAP_compute_res_release(vctx);
        return 0;
    }
#endif
#ifdef GP_O6B_TEST
    {   /* O6b COMPACT-64³ correctness+timing (RE scaffold, -DGP_O6B_TEST). Premise: byte-identical kernel
         * read a COMPACT act (8KB,512B blocks) vs our padded 32KB → 3.2× consumer lever.
         * FINDINGS: V1 standalone strides (n_tiles=256,k_total=128,act_y=512) → DSP fault (n_tiles=256 reads
         * past 128-entry atab). V2 working strides + ×512 atab → runs, maxdiff 25774 (wrong) + cyc=10880
         * (NO speedup) → per-call cost = STRIDE descriptor, not atab spacing. Need QNN's exact compact stride
         * descriptor (dump via vendor patch, reference_vendor_kernel_patch_dump). Restores desc for the solve. */
        w16a16_mm_t *m = &g_ctx[0].mm;
        int32_t a_save[128], o_save[128];
        for (int i = 0; i < 128; ++i) { a_save[i] = m->atab[i]; o_save[i] = m->otab[i]; }
        hmx_conv_out_desc_t *od = (hmx_conv_out_desc_t *)m->od; hmx_conv_act_desc_t *ad = (hmx_conv_act_desc_t *)m->ad;
        hmx_conv_out_desc_t od_save = *od; hmx_conv_act_desc_t ad_save = *ad;
        static uint16_t Aref[4096]; static int16_t Wq[4096]; static uint16_t outlin[4096];
        for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c) {
            Aref[r * 64 + c] = (uint16_t)(32768 + (((r * 7 + c * 3) % 97) - 48) * 64);   /* known act, zp32768 */
            Wq[r * 64 + c]   = (int16_t)((((r * 5 + c * 11) % 127) - 63) * 240);          /* known wt q16 (±32639-ish) */
        }
        w16a16_pack_act_crouton16(Aref, (uint16_t *)m->act, 64, 64);     /* 8KB compact crouton16 */
        w16a16_pack_wt_kmajor(Wq, m->wt, 64, 64);
        w16a16_pack_bias(Wq, (int32_t *)m->bias, 64, 64);
        for (int i = 0; i < 128; ++i) {                                  /* compact atab/otab: ×512 (m_tiles=2) */
            int mt = i >> 1, kt = i & 1;
            m->atab[i] = (int32_t)(uintptr_t)((uint8_t *)m->act + (size_t)(((mt & 7) * 2 + kt) * 512));
            m->otab[i] = (int32_t)(uintptr_t)((uint8_t *)m->out + (size_t)(((mt & 7) * 2 + kt) * 512));
        }
        /* V3 (cron#20, REFUTED = DSP fault 0x8000040d): the OFFLINE byte-exact w16a16 descriptor
         * (prepare_owned_inputs.py:generated_descriptor_tables) = FIXED out_y=256,n_tiles=256,k_total=128,
         * act_y=512 + same ×512 compact table. FAULTED on device (solve crashed in this block). Those "256"
         * are M=256 values (the 256 literally IS M); they don't transpose to a true compact 64³ — same fault
         * class as V1. ⇒ Both natural hypotheses dead: V2 (shape-scaled 64/64/64/128) ran but WRONG (maxdiff
         * 25774); V3 (fixed 256/256/128/512) FAULTS. Descriptor-guessing exhausted → STEP 1a (dump the real
         * custom-op descriptor for an ACTUAL M=64 compact run) is now required. Below = V2 (bootable, wrong). */
        memset(m->out, 0, W16MM_OUT_BYTES);
        uint64_t c0 = gp_pcyc(); w16a16_mm_run(m); uint64_t ccyc = gp_pcyc() - c0;
        w16a16_depack_crouton16((const uint16_t *)m->out, outlin, 64, 64);
        int maxd = 0, nz = 0;
        for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c) {
            long acc = 0; for (int kk = 0; kk < 64; ++kk) acc += (long)((int)Aref[r * 64 + kk] - 32768) * (int)Wq[kk * 64 + c];
            long ref = acc / 32767; long got = (int)outlin[r * 64 + c] - 32768;
            long d = ref - got; if (d < 0) d = -d; if (d > maxd) maxd = (int)d; if (outlin[r*64+c] != 32768) ++nz;
        }
        if (statsLen > 15) stats[15] = maxd;
        if (statsLen > 16) stats[16] = (int)ccyc;
        if (statsLen > 17) stats[17] = nz;
        FARF(ALWAYS, "GDN_PURE O6b compact-64 test: maxdiff=%d nonzero=%d cyc=%llu (vs padded %d)", maxd, nz, (unsigned long long)ccyc, (int)stats[5]);
        for (int i = 0; i < 128; ++i) { m->atab[i] = a_save[i]; m->otab[i] = o_save[i]; }
        *od = od_save; *ad = ad_save;                                    /* restore padded descriptor for the real solve */
    }
#endif
#ifdef GP_OUTMAP
    {   /* OUTPUT-UNTILE derivation (cron#61): ramp act @ identity weight -> out = ramp (known). dense
         * descriptor (all inputs == native, confirmed). dump raw m->out to T -> host maps internal
         * out position -> (r,n) = native's output untile under n_tiles=8. */
        gp_ctx *c = &g_ctx[0]; c->slot = -1;
        hmx_conv_out_desc_t *od = (hmx_conv_out_desc_t *)c->mm.od; hmx_conv_act_desc_t *ad=(hmx_conv_act_desc_t*)c->mm.ad;
        /* real A@A (near-zp, distinct non-saturated out) — all 6 kernel args match native (confirmed) so
         * m->out is native-correct; dump raw m->out to find native's output untile by matching CPU A@A. */
        static uint16_t Aa[4096] __attribute__((aligned(128))); static int16_t Wa[4096] __attribute__((aligned(128)));
        for (int r=0;r<64;++r) for (int n=0;n<64;++n){ Aa[r*64+n]=(uint16_t)(32768+(((r*7+n*3)%97)-48)*5); Wa[r*64+n]=(int16_t)((((r*5+n*11)%127)-63)*24); }
        int hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);
        uint8_t *act=(uint8_t*)c->mm.act, *out=(uint8_t*)c->mm.out;
        for(int i=0;i<16;++i){c->mm.atab[i]=(int32_t)(uintptr_t)(act+(size_t)(i&3)*2048);c->mm.otab[i]=(int32_t)(uintptr_t)(out+(size_t)(i&3)*2048);}
        od->out_table_stride_dwords=2u; od->out_y_stride_words=4u; od->n_tiles_pow2=8u; od->m_total_minus_step=8; od->k_total_bytes=64u;
        ad->n_act_pairs=2u; ad->act_table_y_stride_words=4u;
        w16a16_pack_act_crouton16(Aa,(uint16_t*)c->mm.act,64,64);
        w16a16_pack_wt_kmajor(Wa,c->mm.wt,64,64); w16a16_pack_bias(Wa,(int32_t*)c->mm.bias,64,64);
        memset(c->mm.out,0,W16MM_OUT_BYTES); w16a16_mm_run(&c->mm);
        if (hvx==0) qurt_hvx_unlock();
        memcpy(T, c->mm.out, W16MM_OUT_BYTES);   /* raw m->out -> T for host analysis */
        /* store the inputs into T tail so host can compute CPU A@A (T is 131072 bytes/head, out used 8KB) */
        memcpy((uint8_t*)T+8192, Aa, 8192); memcpy((uint8_t*)T+16384, Wa, 8192);
        if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
        if (vctx) HAP_compute_res_release(vctx);
        return 0;
    }
#endif
#ifdef GP_DIFF
    {   /* IN-SOLVE dense-vs-sparse diff (cron#49): same input A, same aligned ctx buffers. Run the WORKING
         * sparse matmul (mm_init atab, n_tiles=GP_NTILES) and the DENSE matmul (atab i&3, n_tiles=8, perm),
         * diff A@A. Bypasses the unreliable micro-bench harness — uses the solve's own aligned buffers. */
        gp_ctx *c = &g_ctx[0]; c->slot = -1;
        hmx_conv_out_desc_t *od = (hmx_conv_out_desc_t *)c->mm.od;
        static int16_t Aq[64*256] __attribute__((aligned(128)));
        for (int r=0;r<64;++r) for (int k=0;k<64;++k) Aq[r*256+k] = (r>k)?(int16_t)((((r*5+k*7)%41)-20)*300):0;
        int hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);
        gp_unpack_blk(c->A[0], Aq, g_il, c->stage);                 /* A_cv (orig order) */
        /* SPARSE (working): current atab is mm_init sparse, n_tiles=GP_NTILES (32) */
        gp_cv_to_surf((uint16_t *)c->mm.act, c->A[0]);
        gp_pack_wt_bias_hvx(c->A[0], c->stage, c->mm.wt, c->mm.bias);
        memset(c->mm.out,0,W16MM_OUT_BYTES); w16a16_mm_run(&c->mm);
        gp_surf_to_cv(c->AA, (const uint16_t *)c->mm.out);          /* AA_sparse */
        /* DENSE: override atab (i&3) + n_tiles=8 + perm */
        int32_t asv[128], osv[128]; for(int i=0;i<128;++i){asv[i]=c->mm.atab[i];osv[i]=c->mm.otab[i];}
        uint32_t oy=od->out_y_stride_words, ntq=od->n_tiles_pow2;
        uint8_t *act=(uint8_t*)c->mm.act,*out=(uint8_t*)c->mm.out;
        for(int i=0;i<16;++i){c->mm.atab[i]=(int32_t)(uintptr_t)(act+(size_t)(i&3)*2048);c->mm.otab[i]=(int32_t)(uintptr_t)(out+(size_t)(i&3)*2048);}
        od->out_y_stride_words=64u; od->n_tiles_pow2=8u;
        gp_cv_to_surf_perm((uint16_t *)c->mm.act, c->A[0], c->stage);
        gp_pack_wt_bias_hvx(c->A[0], c->stage, c->mm.wt, c->mm.bias);
        memset(c->mm.out,0,W16MM_OUT_BYTES); w16a16_mm_run(&c->mm);
        gp_surf_to_cv_perm(c->A3, (const uint16_t *)c->mm.out, c->stage);  /* AA_dense */
        /* DENSE-DIRECT n_tiles SWEEP (cron#52): pack_act_crouton16 + 4-tile atab(i&3) + depack, sweep
         * n_tiles to find which reads the 4 tiles correctly (structured A@A vs CPU). native uses 4
         * contiguous tiles (atab[0..3]); the right n_tiles for 4 tiles is unknown ((i&3)+nt8 = refuted). */
        static uint16_t Alin[4096] __attribute__((aligned(128))), Yd2[4096] __attribute__((aligned(128)));
        static int16_t Wlin[4096] __attribute__((aligned(128)));
        for (int r=0;r<64;++r) for (int k=0;k<64;++k){ Alin[r*64+k]=(uint16_t)(Aq[r*256+k]+32768); Wlin[r*64+k]=Aq[r*256+k]; }
        static int CPUaa2[4096]; for(int r=0;r<64;++r)for(int k=0;k<64;++k){ long a=0; for(int j=0;j<64;++j) a+=(long)(Aq[r*256+j])*(long)(Aq[j*256+k]); CPUaa2[r*64+k]=(int)(a/32767); }
        /* native FULL descriptor sweep (cron#53): act layout = pack_act_crouton16 (confirmed via act dump);
         * test native's out_y/m_total/act_y (cron#41: out_y=4,m_total=8,act_y=4) not just n_tiles.
         * variants: {out_y, n_tiles, m_total, act_y, out_tbl_stride}. */
        hmx_conv_act_desc_t *ad2=(hmx_conv_act_desc_t*)c->mm.ad;
        struct { uint32_t oy, nt; int32_t mt; uint32_t ay, ots; } DV[4] = {
            {4u, 8u, 8, 4u, 2u},     /* native-exact (cron#41) */
            {4u, 8u, 8, 4u, 4u},     /* out_tbl_stride=4 */
            {64u,8u, 8, 4u, 2u},     /* our out_y, native m_total/act_y */
            {4u, 8u, 1, 4u, 2u},     /* native strides, m_total=1 */
        };
        int swres[4];
        for (int s=0;s<4;++s){
            for(int i=0;i<16;++i){c->mm.atab[i]=(int32_t)(uintptr_t)(act+(size_t)(i&3)*2048);c->mm.otab[i]=(int32_t)(uintptr_t)(out+(size_t)(i&3)*2048);}
            od->out_y_stride_words=DV[s].oy; od->n_tiles_pow2=DV[s].nt; od->m_total_minus_step=DV[s].mt;
            od->out_table_stride_dwords=DV[s].ots; od->k_total_bytes=64u;
            ad2->n_act_pairs=2u; ad2->act_table_y_stride_words=DV[s].ay;
            w16a16_pack_act_crouton16(Alin,(uint16_t*)c->mm.act,64,64);
            w16a16_pack_wt_kmajor(Wlin,c->mm.wt,64,64); w16a16_pack_bias(Wlin,(int32_t*)c->mm.bias,64,64);
            memset(c->mm.out,0,W16MM_OUT_BYTES); w16a16_mm_run(&c->mm);
            w16a16_depack_crouton16((const uint16_t*)c->mm.out, Yd2, 64, 64);
            int nd=0; for(int r=0;r<64;++r)for(int k=0;k<64;++k){ long got=(int)Yd2[r*64+k]-32768, dd=CPUaa2[r*64+k]-got; if(dd<0)dd=-dd; if(dd>20) nd++; }
            swres[s]=nd;
            if (s==0){ /* sorted-multiset: are values CORRECT but permuted (readback bug) vs wrong? */
                static int gv[4096], cv2[4096]; for(int i=0;i<4096;++i){gv[i]=(int)Yd2[i]-32768; cv2[i]=CPUaa2[i];}
                for(int a=0;a<4096;++a)for(int b2=a+1;b2<4096;++b2){ if(gv[b2]<gv[a]){int t=gv[a];gv[a]=gv[b2];gv[b2]=t;} if(cv2[b2]<cv2[a]){int t=cv2[a];cv2[a]=cv2[b2];cv2[b2]=t;} }
                int sm=0; for(int i=0;i<4096;++i){int dd=gv[i]-cv2[i]; if(dd<0)dd=-dd; if(dd>20) sm++; }
                if(statsLen>17) stats[17]=sm;   /* sorted mismatch: 0 => values correct, only PERMUTED (readback bug) */
            }
        }
        for(int i=0;i<128;++i){c->mm.atab[i]=asv[i];c->mm.otab[i]=osv[i];} od->out_y_stride_words=oy;od->n_tiles_pow2=ntq;
        if(statsLen>12) stats[12]=swres[0]; if(statsLen>13) stats[13]=swres[1];
        if(statsLen>14) stats[14]=swres[2]; if(statsLen>16) stats[16]=swres[3];   /* nt=2,4,8,16 mismatch vs CPU */
        /* diff c->AA (sparse) vs c->A3 (dense), both orig-order cv -> linear */
        static int16_t Ts[64*256] __attribute__((aligned(128))), Td[64*256] __attribute__((aligned(128)));
        for(int i=0;i<64*256;++i){Ts[i]=0;Td[i]=0;}
        gp_pack_blk(Ts, c->AA, g_fl, c->stage); gp_pack_blk(Td, c->A3, g_fl, c->stage);
        if (hvx==0) qurt_hvx_unlock();
        int nm=0,mx=0,firstr=-1,firstc=-1;
        for(int r=0;r<64;++r)for(int k=0;k<64;++k){int d=Ts[r*256+k]-Td[r*256+k]; if(d){ if(firstr<0){firstr=r;firstc=k;} nm++; if(d<0)d=-d; if(d>mx)mx=d;}}
        FARF(ALWAYS,"GDN_PURE GP_DIFF first mismatch at (r=%d,c=%d) Ts=%d Td=%d", firstr, firstc, firstr<0?0:Ts[firstr*256+firstc], firstr<0?0:Td[firstr*256+firstc]);
        FARF(ALWAYS,"GDN_PURE GP_DIFF A@A sparse-vs-dense mismatch=%d/4096 maxdiff=%d", nm, mx);
        FARF(ALWAYS,"GDN_PURE GP_DIFF first row Ts[0..7]=%d %d %d %d %d %d %d %d", Ts[0],Ts[1],Ts[2],Ts[3],Ts[4],Ts[5],Ts[6],Ts[7]);
        FARF(ALWAYS,"GDN_PURE GP_DIFF first row Td[0..7]=%d %d %d %d %d %d %d %d", Td[0],Td[1],Td[2],Td[3],Td[4],Td[5],Td[6],Td[7]);
        if(statsLen>5) stats[5]=nm; if(statsLen>15) stats[15]=mx;
        if(statsLen>6) stats[6]=firstr; if(statsLen>7) stats[7]=firstc;
        if(statsLen>8) stats[8]=(firstr<0?0:Ts[firstr*256+firstc]); if(statsLen>9) stats[9]=(firstr<0?0:Td[firstr*256+firstc]);
        if(statsLen>10) stats[10]=Ts[64]; if(statsLen>11) stats[11]=Td[64];  /* row1 col0 (Ts vs Td) */
        if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
        if (vctx) HAP_compute_res_release(vctx);
        return 0;
    }
#endif
    g_cbusy = 0; g_sat = 0; g_pdone = 0;
#ifdef GP_TRACE
    g_ev_n = 0;
#endif
    g_Ah = A; g_Th = (uint8_t *)T; g_H = H; g_P = P;

    uint64_t us0 = HAP_perf_get_time_us(), t0 = gp_pcyc();
    /* ⚠️ P==1 (single-thread) is BROKEN under the crouton8/cv-block regime (since cron#68, never re-tested —
     * the metric is P=4): the inline path runs the whole solve_head->diag_inv->mm64 chain on the small main
     * FastRPC stack and crouton8's 4×128B colsum unions overflow it; the threaded P=1 (1 producer) path
     * also faults (untested). Use P>=2. All of P=2/3/4 work and are bit-exact. */
    if (P == 1) {
        g_ctx[0].slot = -1;
        int hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);            /* main does HVX copies inline (+ HMX) */
        uint64_t L0 = gp_pcyc();
        for (int h = 0; h < H; ++h)
            solve_head(&g_ctx[0], (const int16_t *)(A + (size_t)h * 131072),
                       (int16_t *)((uint8_t *)T + (size_t)h * 131072));
        g_ctx[0].t_life = gp_pcyc() - L0;
        if (hvx == 0) qurt_hvx_unlock();
    } else {
        qurt_thread_t tid[GP_NT];
        for (int t = 0; t < P; ++t) {
            g_ctx[t].slot = t;
            qurt_thread_attr_t a; qurt_thread_attr_init(&a); qurt_thread_attr_set_name(&a, (char *)"gp_prod");
            qurt_thread_attr_set_stack_addr(&a, g_stack[t]); qurt_thread_attr_set_stack_size(&a, sizeof(g_stack[t]));
            if (qurt_thread_create(&tid[t], &a, producer, (void *)(intptr_t)t) != QURT_EOK) tid[t] = 0;
        }
        while (__atomic_load_n(&g_pdone, __ATOMIC_ACQUIRE) < P) {   /* consumer: run any armed kernel */
            for (int t = 0; t < P; ++t)
                if (GP_POLL(&g_job[t])) {
                    uint64_t k0 = gp_pcyc(); w16a16_mm_run(&g_ctx[t].mm); uint64_t k1 = gp_pcyc(); g_cbusy += k1 - k0;
                    GP_EV(GP_NT, 3, k0, k1);                    /* consumer MM (口径④ per-call wall) */
                    GP_DONE(&g_job[t]);
                }
        }
        for (int t = 0; t < P; ++t) { int s; if (tid[t]) qurt_thread_join(tid[t], &s); }
    }
    uint64_t t1 = gp_pcyc(), us1 = HAP_perf_get_time_us();

    if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
    if (vctx) HAP_compute_res_release(vctx);
    { HAP_power_request_t off; memset(&off, 0, sizeof(off)); off.type = HAP_power_set_HMX; off.hmx.power_up = FALSE; HAP_power_set(pctx, &off); }

    uint64_t wall = t1 - t0, spin = 0, actcopy = 0, kmajor = 0, scatter = 0, outcopy = 0, life = 0, lmax = 0;
    uint64_t sc_mc = 0, sc_pm = 0, sc_ms = 0;   /* O5 scatter split Σ */
    for (int t = 0; t < P; ++t) { spin += g_ctx[t].spin; actcopy += g_ctx[t].t_pack; kmajor += g_ctx[t].t_kmajor;
        scatter += g_ctx[t].t_scatter; outcopy += g_ctx[t].t_depack;
        sc_mc += g_ctx[t].t_mc; sc_pm += g_ctx[t].t_pm; sc_ms += g_ctx[t].t_ms;
        life += g_ctx[t].t_life; if (g_ctx[t].t_life > lmax) lmax = g_ctx[t].t_life; }
    uint64_t other = life - spin - actcopy - kmajor - scatter - outcopy;   /* HVX renorm+acc (mm-level) */
    uint32_t nmm = (uint32_t)H * 40u;                          /* 24 diag + 16 merge per head */
    if (statsLen > 0) stats[0] = (int)wall;                    /* ① graph wall (PCYCLE) */
    if (statsLen > 1) stats[1] = P;
    if (statsLen > 2) stats[2] = H;
    if (statsLen > 3) stats[3] = (int)g_cbusy;                 /* ④ kernel (consumer) */
    if (statsLen > 4) stats[4] = (int)spin;                    /* spin Σ */
    (void)nmm;   /* stats[5] = single-64³-matmul micro-bench cyc/call (set before solve) */
    if (statsLen > 6) stats[6] = (int)(us1 - us0);             /* wall us */
    if (statsLen > 7) stats[7] = (int)kmajor;                  /* HVX wt-pack Σ */
    if (statsLen > 8) stats[8] = packchk;                      /* HVX wt-pack byte-exact self-check (0 = ok) */
    if (statsLen > 9) stats[9] = (int)scatter;                 /* A-unpack + T-pack LUT-scatter Σ (SCALAR) */
    if (statsLen > 10) stats[10] = (int)other;                /* HVX renorm + acc (mm-level) Σ */
    if (statsLen > 11) stats[11] = (int)lmax;                  /* slowest producer life (~wall) */
    if (statsLen > 12) stats[12] = (int)sc_mc;                 /* O5 scatter split: linear↔block memcpy Σ */
    if (statsLen > 13) stats[13] = (int)sc_pm;                 /* O5 scatter split: gp_perm vgather Σ */
    if (statsLen > 14) stats[14] = (int)sc_ms;                 /* O5 scatter split: memset(To,128KB) Σ */
    FARF(ALWAYS, "GDN_PURE(cv) P=%d H=%d wall=%llu packchk=%d | PROD-Σ actcopy=%llu kmajor=%llu scatter=%llu renorm/acc=%llu outcopy=%llu spin=%llu | CONS kernel=%llu",
         P, H, (unsigned long long)wall, packchk, (unsigned long long)actcopy, (unsigned long long)kmajor,
         (unsigned long long)scatter, (unsigned long long)other, (unsigned long long)outcopy,
         (unsigned long long)spin, (unsigned long long)g_cbusy);
#ifdef GP_TRACE
    {   /* serialize the event trace into T (overwrites the result; trace runs are timeline-only).
         * [magic][n][wall u64][base u64=t0] then n*{tid u32, stage u32, t0 u64, t1 u64}. */
        int n = g_ev_n; if (n > (int)(sizeof(g_ev) / sizeof(g_ev[0]))) n = (int)(sizeof(g_ev) / sizeof(g_ev[0]));
        uint8_t *tb = (uint8_t *)T;
        if ((size_t)TLen >= 24u + (size_t)n * 24u) {
            *(uint32_t *)(tb + 0) = 0x47545203u; *(uint32_t *)(tb + 4) = (uint32_t)n;
            *(uint64_t *)(tb + 8) = wall;        *(uint64_t *)(tb + 16) = 0;   /* base 0: t0/t1 stored RELATIVE */
            uint8_t *p = tb + 24;
            for (int i = 0; i < n; ++i) {
                *(uint32_t *)(p + 0) = g_ev[i].tid;       *(uint32_t *)(p + 4) = g_ev[i].stage;
                *(uint64_t *)(p + 8) = g_ev[i].t0 - t0;   *(uint64_t *)(p + 16) = g_ev[i].t1 - t0; p += 24;
            }
        }
        FARF(ALWAYS, "GDN_PURE TRACE: %d events serialized into T (wall=%llu base=%llu)", n, (unsigned long long)wall, (unsigned long long)t0);
    }
#endif
    return 0;
}

}  /* namespace gdn_pure */
