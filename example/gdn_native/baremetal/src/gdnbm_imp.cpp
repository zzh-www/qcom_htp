/* gdnbm_imp.c — bare-metal FastRPC GDN block-recursive solve (escapes QNN; self-managed HVX/HMX/VTCM).
 * Reuses the EXACT validated device solve from the QNN op via the GDN_BR_NO_QHPI include guard. */
#include "gdnbm.h"
#include "HAP_compute_res.h"
#include "HAP_vtcm_mgr.h"
#include "HAP_power.h"
#include "HAP_perf.h"             /* HAP_perf_get_time_us: real wall µs (to convert PCYCLE -> ms + derive clock) */
#include "HAP_farf.h"
#include "qurt.h"
#if defined(GDNBM_PMU)
#include "qurt_pmu.h"              /* QuRT PMU: program PMUEVTCFG + read PMUCNT0..3 (raw v75 event codes) */
#endif
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

/* pull in the device solve (structs, helpers, gdn_br_one_head, gdn_br_run_slot, gdn_work_t, g_scr[]). */
#define GDN_BR_NO_QHPI 1
#ifndef GDN_BR_C
#define GDN_BR_C 256
#endif
#ifndef GDN_BR_NT
#define GDN_BR_NT 4
#endif
#define THIS_PKG_NAME GdnBm

/* ⛔ DISABLED (2026-06-05): the full-solve GDNSolveHVXMixHMX harness SSRs the cDSP at gdnbm_open
 * (rc=0x80000406) — verified on real v75 (the HVX baseline opens rc=0x0 right after on the same device,
 * so it's THIS build, not a wedged DSP). Its old "434K/head" number is NOT reproducible and is void.
 * The plan is NOT to fix this build but to REWRITE the whole solve from the (trusted) microbench findings.
 * Until then, only the microbenches are valid: -DGDNBM_FEED_PIPE[/_FEED_4P], -DGDNBM_HMX_BENCH,
 * -DGDNBM_GLUE_BENCH, -DGDNBM_FEED_MULTIPASS, and the GDNSolveHVX baseline (default / -DGDN_BR_MM_I8).
 * See Agent/current/gdn_solve_hvxmixhmx.md (top "⛔ 工作方式" banner). Escape hatch (debug only):
 * add -DGDNBM_ALLOW_BROKEN_HMX_MERGE to build it anyway (will SSR; needs a reboot to clear). */
#if defined(GDNBM_HMX_MERGE_PATH) && !defined(GDNBM_ALLOW_BROKEN_HMX_MERGE)
#error "GDNBM_HMX_MERGE_PATH is DISABLED: the full GDNSolveHVXMixHMX solve SSRs the cDSP at open (rc=0x80000406, void). Run only the microbenches; the solve will be rewritten from them. Override with -DGDNBM_ALLOW_BROKEN_HMX_MERGE (debug only, will SSR)."
#endif

/* GDNSolveHVX mode (default): int16-HVX matmul merges -> pure HVX + BSS scratch, threads freely.
 * Disabled for the OVERLAP probe (int8-HMX merges) and for GDNBM_HMX_MERGE_PATH (run the REAL
 * GDNSolveHVXMixHMX gdn_merge_packed on baremetal w/ PROBE_CYCLES, to get its true per-stage breakdown). */
#if !defined(GDNBM_OVERLAP_PROBE) && !defined(GDNBM_HMX_MERGE_PATH) && !defined(GDNBM_HMX_SOLVE) && !defined(GDNBM_HMX_PIPE)
#define GDN_BR_HVX_MERGE 1
#endif
/* PIPE default = int16 T codes (GdnSolveBR16.cpp): bit-exact vs int32, ~11% faster (half storage + i16-native
 * widen/narrow/acc/maxabs).  Opt out with -DGDN_BR_NO_I16.  Needs STATIC_FULL (all static codes fit int16).
 * MUST precede the includes (gates struct fields in GdnSolveBROp.cpp + the GdnSolveBR16.cpp body). */
#if defined(GDNBM_HMX_PIPE) && defined(GDN_BR_STATIC_FULL) && !defined(GDN_BR_NO_I16) && !defined(GDN_BR_I16)
#define GDN_BR_I16 1
#endif
/* int16-lane A fold (gdn_fold_quant_u8): (A_u16-32768)=(i16)(A XOR 0x8000) free zp-sub + 64-lane Q6_Ww_vmpy.
 * +1.4% bit-exact (3-round A/B).  COMPILE-TIME (no per-iter branch — a runtime branch in the hot loop cost
 * ~10%, the bug that earlier made this look "slower").  Assumes zpA==32768 (always true for GDN int16 act). */
#if defined(GDN_BR_I16) && !defined(GDN_BR_NO_I16_FOLD) && !defined(GDN_BR_I16_FOLD)
#define GDN_BR_I16_FOLD 1
#endif
/* + int16-lane QUANT (Q15) for the A fold: another ~5% (also kills the separate int32->u8 narrow pass).
 * NOT bit-exact (Q15 8-bit multiplier -> off-diag relerr 0.1094 vs 0.1093, +0.5%; precision先不管 OK).
 * Opt out with -DGDN_BR_NO_I16_FOLD_QUANT to stay bit-exact. */
#if defined(GDN_BR_I16_FOLD) && !defined(GDN_BR_NO_I16_FOLD_QUANT) && !defined(GDN_BR_I16_FOLD_QUANT)
#define GDN_BR_I16_FOLD_QUANT 1
#endif
#include "../../solve_br_op/src/GdnSolveBROp.cpp"
#include "../../solve_br_op/src/GdnSolveBR16.cpp"   /* clean int16 static solve (GDN_BR_I16) */

/* PIPE default = PURE-HMX consumer (lever 1): bias-pack+out-zero on the producer so the consumer is pure
 * mxmem and frees its HVX unit -> P=4 producers fit 4 HVX units (no SMT starvation; timeline-verified).
 * Opt out with -DGDNBM_PIPE_HVX_CONSUMER (consumer does bias-pack; then use P=3 to avoid 5-on-4 thrash). */
#if defined(GDNBM_HMX_PIPE) && !defined(GDNBM_PIPE_HVX_CONSUMER) && !defined(GDNBM_PIPE_PURE_HMX)
#define GDNBM_PIPE_PURE_HMX 1
#endif

/* FULLY-VECTORIZED eff+bias for the GDNSolveHVXMixHMX producer: replaces gdn_effective's scalar tail (64 scalar
 * -128*col) + gdn_pack_bias's 128 SCALAR VTCM writes (the documented pathology) with HVX column-sum +
 * 4 vector stores. Produces the identical bias layout: [ctrl×32][eff0:32][ctrl×32][eff32:64],
 * eff[n] = rdelta − 128·Σ_k wt[k,n]. */
static void fp_pack_effbias(const int8_t *wt, uint32_t ctrl, int rdelta, int32_t *bias) {
    HVX_Vector acc_e = Q6_V_vzero(), acc_o = Q6_V_vzero();          /* int16 col partials (64·127 < 2^15) */
    for (int k = 0; k < 64; ++k) {
        HVX_VectorPair w16 = Q6_Wh_vsxt_Vb(*(const HVX_UVector *)(wt + k * 64));
        acc_e = Q6_Vh_vadd_VhVh(acc_e, Q6_V_lo_W(w16));
        acc_o = Q6_Vh_vadd_VhVh(acc_o, Q6_V_hi_W(w16));
    }
    HVX_Vector cols = Q6_V_lo_W(Q6_W_vshuff_VVR(acc_o, acc_e, -2)); /* natural int16 cols 0..63 */
    HVX_VectorPair e = Q6_Ww_vsxt_Vh(cols);                         /* even/odd int32 */
    HVX_VectorPair nat = Q6_W_vshuff_VVR(Q6_V_hi_W(e), Q6_V_lo_W(e), -4);  /* natural int32 */
    HVX_Vector vrd = Q6_V_vsplat_R(rdelta);
    HVX_Vector elo = Q6_Vw_vsub_VwVw(vrd, Q6_Vw_vasl_VwR(Q6_V_lo_W(nat), 7));  /* rdelta − 128·col */
    HVX_Vector ehi = Q6_Vw_vsub_VwVw(vrd, Q6_Vw_vasl_VwR(Q6_V_hi_W(nat), 7));
    HVX_Vector vctrl = Q6_V_vsplat_R((int)ctrl);
    HVX_Vector *b = (HVX_Vector *)bias;
    b[0] = vctrl; b[1] = elo; b[2] = vctrl; b[3] = ehi;             /* 4 vector stores, no scalar VTCM */
}

/* 2-HEAD interleaved eff+bias: two column-sum chains (wtA/wtB) run in lockstep so the 64-iter int16
 * accumulate over each weight tile overlaps — ILP, mirrors fp_pack_act2/fp_depack2. */
static void fp_pack_effbias2(const int8_t *wtA, const int8_t *wtB, uint32_t ctrl, int rdelta,
                             int32_t *biasA, int32_t *biasB) {
    HVX_Vector ae=Q6_V_vzero(), ao=Q6_V_vzero(), be=Q6_V_vzero(), bo=Q6_V_vzero();
    for (int k = 0; k < 64; ++k) {
        HVX_VectorPair wa = Q6_Wh_vsxt_Vb(*(const HVX_UVector *)(wtA + k * 64));
        HVX_VectorPair wb = Q6_Wh_vsxt_Vb(*(const HVX_UVector *)(wtB + k * 64));
        ae = Q6_Vh_vadd_VhVh(ae, Q6_V_lo_W(wa)); ao = Q6_Vh_vadd_VhVh(ao, Q6_V_hi_W(wa));
        be = Q6_Vh_vadd_VhVh(be, Q6_V_lo_W(wb)); bo = Q6_Vh_vadd_VhVh(bo, Q6_V_hi_W(wb));
    }
    HVX_Vector vrd = Q6_V_vsplat_R(rdelta), vctrl = Q6_V_vsplat_R((int)ctrl);
    HVX_Vector acolA = Q6_V_lo_W(Q6_W_vshuff_VVR(ao, ae, -2));
    HVX_Vector acolB = Q6_V_lo_W(Q6_W_vshuff_VVR(bo, be, -2));
    HVX_VectorPair eA = Q6_Ww_vsxt_Vh(acolA), eB = Q6_Ww_vsxt_Vh(acolB);
    HVX_VectorPair nA = Q6_W_vshuff_VVR(Q6_V_hi_W(eA), Q6_V_lo_W(eA), -4);
    HVX_VectorPair nB = Q6_W_vshuff_VVR(Q6_V_hi_W(eB), Q6_V_lo_W(eB), -4);
    HVX_Vector aloA = Q6_Vw_vsub_VwVw(vrd, Q6_Vw_vasl_VwR(Q6_V_lo_W(nA), 7));
    HVX_Vector ahiA = Q6_Vw_vsub_VwVw(vrd, Q6_Vw_vasl_VwR(Q6_V_hi_W(nA), 7));
    HVX_Vector aloB = Q6_Vw_vsub_VwVw(vrd, Q6_Vw_vasl_VwR(Q6_V_lo_W(nB), 7));
    HVX_Vector ahiB = Q6_Vw_vsub_VwVw(vrd, Q6_Vw_vasl_VwR(Q6_V_hi_W(nB), 7));
    HVX_Vector *bA = (HVX_Vector *)biasA, *bB = (HVX_Vector *)biasB;
    bA[0] = vctrl; bA[1] = aloA; bA[2] = vctrl; bA[3] = ahiA;
    bB[0] = vctrl; bB[1] = aloB; bB[2] = vctrl; bB[3] = ahiB;
}

/* ONE-PASS pipeline-local depack: fuses gdn_depack_out_fast's base-subtract + de-crouton, skipping the
 * sc->surf_sub VTCM round-trip (kills ~48 of ~96 vector mem-ops). Each 128B crouton vector is read once
 * from `surf` (the HMX output surface), base-subtracted inline (byte-sub commutes with the ror/mask/or
 * byte rearrangement), then de-croutoned into row-major out_codes. Bit-equivalent to gdn_depack_out_fast. */
static void fp_depack(const uint8_t *surf, int base, int8_t *out_codes) {
    const HVX_Vector vb = Q6_Vb_vsplat_R(base);
    const HVX_Vector m  = Q6_V_valign_VVR(Q6_V_vzero(), Q6_Vb_vsplat_R(-1), 96);
    const HVX_Vector m1 = Q6_V_vror_VR(m, 96), m2 = Q6_V_vror_VR(m, 64), m3 = Q6_V_vror_VR(m, 32);
    const uint8_t *s0 = surf, *s1 = surf + 2048;                    /* nt=0 / nt=1 crouton halves */
    for (int local = 0; local < 16; ++local) {
        HVX_Vector v0 = Q6_Vb_vsub_VbVb(*(const HVX_Vector *)(s0 + local * 128), vb);
        HVX_Vector v1 = Q6_Vb_vsub_VbVb(*(const HVX_Vector *)(s1 + local * 128), vb);
        int o0 = local * 4;
        int row0 = ((o0 / 8) & 1) * 32 + (o0 / 16) * 8 + (o0 % 8);
        HVX_Vector A = Q6_V_vand_VV(v0, m);
        A = Q6_V_vor_VV(A, Q6_V_vand_VV(Q6_V_vror_VR(v1, 96), m1));
        A = Q6_V_vor_VV(A, Q6_V_vand_VV(Q6_V_vror_VR(v0, 96), m2));
        A = Q6_V_vor_VV(A, Q6_V_vand_VV(Q6_V_vror_VR(v1, 64), m3));
        HVX_Vector B = Q6_V_vand_VV(Q6_V_vror_VR(v0, 64), m);
        B = Q6_V_vor_VV(B, Q6_V_vand_VV(Q6_V_vror_VR(v1, 32), m1));
        B = Q6_V_vor_VV(B, Q6_V_vand_VV(Q6_V_vror_VR(v0, 32), m2));
        B = Q6_V_vor_VV(B, Q6_V_vand_VV(v1, m3));
        *(HVX_Vector *)(out_codes + (size_t)row0 * 64)       = A;
        *(HVX_Vector *)(out_codes + (size_t)(row0 + 2) * 64) = B;
    }
}

/* 2-HEAD interleaved fp_depack: two independent surfaces (surfA/surfB) -> two distinct out buffers, the
 * de-crouton chains interleaved (mirrors fp_pack_act2) to hide the ror/mask/or latency under ILP — same
 * lever that gave the pack ~1.35x. Distinct outA/outB avoid store aliasing. */
static void fp_depack2(const uint8_t *surfA, const uint8_t *surfB, int base, int8_t *outA, int8_t *outB) {
    const HVX_Vector vb = Q6_Vb_vsplat_R(base);
    const HVX_Vector m  = Q6_V_valign_VVR(Q6_V_vzero(), Q6_Vb_vsplat_R(-1), 96);
    const HVX_Vector m1 = Q6_V_vror_VR(m, 96), m2 = Q6_V_vror_VR(m, 64), m3 = Q6_V_vror_VR(m, 32);
    const uint8_t *as0 = surfA, *as1 = surfA + 2048, *bs0 = surfB, *bs1 = surfB + 2048;
    for (int local = 0; local < 16; ++local) {
        HVX_Vector av0 = Q6_Vb_vsub_VbVb(*(const HVX_Vector *)(as0 + local * 128), vb);
        HVX_Vector av1 = Q6_Vb_vsub_VbVb(*(const HVX_Vector *)(as1 + local * 128), vb);
        HVX_Vector bv0 = Q6_Vb_vsub_VbVb(*(const HVX_Vector *)(bs0 + local * 128), vb);
        HVX_Vector bv1 = Q6_Vb_vsub_VbVb(*(const HVX_Vector *)(bs1 + local * 128), vb);
        int o0 = local * 4;
        int row0 = ((o0 / 8) & 1) * 32 + (o0 / 16) * 8 + (o0 % 8);
        HVX_Vector AA = Q6_V_vand_VV(av0, m);
        AA = Q6_V_vor_VV(AA, Q6_V_vand_VV(Q6_V_vror_VR(av1, 96), m1));
        AA = Q6_V_vor_VV(AA, Q6_V_vand_VV(Q6_V_vror_VR(av0, 96), m2));
        AA = Q6_V_vor_VV(AA, Q6_V_vand_VV(Q6_V_vror_VR(av1, 64), m3));
        HVX_Vector BA = Q6_V_vand_VV(bv0, m);
        BA = Q6_V_vor_VV(BA, Q6_V_vand_VV(Q6_V_vror_VR(bv1, 96), m1));
        BA = Q6_V_vor_VV(BA, Q6_V_vand_VV(Q6_V_vror_VR(bv0, 96), m2));
        BA = Q6_V_vor_VV(BA, Q6_V_vand_VV(Q6_V_vror_VR(bv1, 64), m3));
        HVX_Vector AB = Q6_V_vand_VV(Q6_V_vror_VR(av0, 64), m);
        AB = Q6_V_vor_VV(AB, Q6_V_vand_VV(Q6_V_vror_VR(av1, 32), m1));
        AB = Q6_V_vor_VV(AB, Q6_V_vand_VV(Q6_V_vror_VR(av0, 32), m2));
        AB = Q6_V_vor_VV(AB, Q6_V_vand_VV(av1, m3));
        HVX_Vector BB = Q6_V_vand_VV(Q6_V_vror_VR(bv0, 64), m);
        BB = Q6_V_vor_VV(BB, Q6_V_vand_VV(Q6_V_vror_VR(bv1, 32), m1));
        BB = Q6_V_vor_VV(BB, Q6_V_vand_VV(Q6_V_vror_VR(bv0, 32), m2));
        BB = Q6_V_vor_VV(BB, Q6_V_vand_VV(bv1, m3));
        *(HVX_Vector *)(outA + (size_t)row0 * 64)       = AA;
        *(HVX_Vector *)(outA + (size_t)(row0 + 2) * 64) = AB;
        *(HVX_Vector *)(outB + (size_t)row0 * 64)       = BA;
        *(HVX_Vector *)(outB + (size_t)(row0 + 2) * 64) = BB;
    }
}

int gdnbm_open(const char *uri, remote_handle64 *h) { (void)uri; *h = 1; return 0; }
int gdnbm_close(remote_handle64 h) { (void)h; return 0; }

static inline uint64_t pcyc(void) { uint64_t c; asm volatile("%0 = C15:14" : "=r"(c)); return c; }
static inline float i2f(int b) { union { int i; float f; } u; u.i = b; return u.f; }

/* ---- UDMA: async DDR->VTCM (skill principle 3: hide the A-load under compute) ---- */
typedef struct {
    void *next; unsigned int length:24, desctype:2, dstcomp:1, srccomp:1,
        dstbypass:1, srcbypass:1, order:1, dstate:1;
    void *src, *dst;
} __attribute__((aligned(64))) dma_desc_t;
static inline void udma_start(dma_desc_t *d, void *dst, const void *src, uint32_t len) {
    memset(d, 0, sizeof(*d));
    d->length = len; d->desctype = 0; d->srcbypass = 1; /* DDR src bypasses cache */
    d->src = (void *)src; d->dst = dst;
    Q6_dmstart_A((void *)d);
}
static inline int udma_wait(void) { return Q6_R_dmwait(); }
/* fill a descriptor WITHOUT starting it (for chaining several transfers into one dmstart).
 * dir_to_ddr: src=VTCM dst=DDR (T writeback, dstbypass) vs src=DDR dst=VTCM (A load, srcbypass). */
static inline void udma_fill(dma_desc_t *d, void *dst, const void *src, uint32_t len, int dir_to_ddr) {
    memset(d, 0, sizeof(*d));
    d->length = len; d->desctype = 0;
    if (dir_to_ddr) d->dstbypass = 1; else d->srcbypass = 1;
    d->src = (void *)src; d->dst = dst;
}

/* ---- HMX-on-worker feasibility probe (kept) ---- */
static volatile int g_prc[GDN_BR_NT], g_psent[GDN_BR_NT];
static char __attribute__((aligned(128))) g_pstack[GDN_BR_NT][16384];
static void probe_worker(void *arg) {
    int id = (int)(long)arg;
    int hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);
    compute_res_attr_t a; HAP_compute_res_attr_init(&a); HAP_compute_res_attr_set_hmx_param(&a, 1);
    unsigned int ctx = HAP_compute_res_acquire(&a, 2000000);
    g_prc[id] = (int)ctx;
    if (ctx) { HVX_Vector v = Q6_V_vsplat_R(0x1000 + id); int l[32] __attribute__((aligned(128)));
               *(HVX_Vector *)l = v; g_psent[id] = l[0]; HAP_compute_res_release(ctx); }
    else g_psent[id] = 0xBAD;
    if (hvx == 0) qurt_hvx_unlock();
}
int gdnbm_hmx_probe(remote_handle64 _h, int nworkers, int *results, int resultsLen) {
    (void)_h; if (nworkers > GDN_BR_NT) nworkers = GDN_BR_NT; if (nworkers < 1) nworkers = 1;
    if (resultsLen > 0) results[0] = 1;
    qurt_thread_t tid[GDN_BR_NT];
    for (int i = 0; i < nworkers; ++i) {
        g_prc[i] = -999; g_psent[i] = 0;
        qurt_thread_attr_t a; qurt_thread_attr_init(&a); qurt_thread_attr_set_name(&a, (char *)"probe");
        qurt_thread_attr_set_stack_addr(&a, g_pstack[i]); qurt_thread_attr_set_stack_size(&a, sizeof(g_pstack[i]));
        if (qurt_thread_create(&tid[i], &a, probe_worker, (void *)(long)i) != QURT_EOK) { probe_worker((void *)(long)i); tid[i] = 0; }
    }
    for (int i = 0; i < nworkers; ++i) { int s; if (tid[i]) qurt_thread_join(tid[i], &s); }
    for (int i = 0; i < nworkers; ++i) if (1 + 2 * i + 1 < resultsLen) { results[1 + 2 * i] = g_prc[i]; results[1 + 2 * i + 1] = g_psent[i]; }
    return 0;
}

/* ---- the threaded solve ---- */
static char __attribute__((aligned(128))) g_solve_stack[GDN_BR_NT][32768];
/* HAP worker (GDNSolveHVX mode): own HVX context, run this slot's heads.  No HMX/VTCM — the int16-HVX
 * merges use only HVX + BSS scratch, so workers parallelize with no shared lock. */
#if defined(GDNBM_OVERLAP_PROBE)
/* Minimal HVX∥HMX overlap probe: one thread loops the int8-HMX solve (real mxmem), another loops the
 * HVX diagonal forward-subst. If concurrent wall ≈ max(solo times) and each thread's own cycles don't
 * inflate, HVX and HMX truly overlap in bare-metal (separate execution units). */
static char __attribute__((aligned(128))) g_ov_stack[2][65536];
static volatile uint64_t g_ov_cyc[2];
struct ov_arg { int role; uint32_t H; const uint16_t *Au; uint16_t *Tu; int zpA, M, S; float sT; int zpT; int iters; };
static void ov_hmx(void *p) {
    struct ov_arg *a = (struct ov_arg *)p;
    int hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);
    compute_res_attr_t va; HAP_compute_res_attr_init(&va); HAP_compute_res_attr_set_vtcm_param(&va, 0x60000u, 0);
    unsigned int vctx = HAP_compute_res_acquire(&va, 2000000); uint8_t *vtcm = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&va);
    compute_res_attr_t ha; HAP_compute_res_attr_init(&ha); HAP_compute_res_attr_set_hmx_param(&ha, 1);
    unsigned int hctx = HAP_compute_res_acquire(&ha, 2000000); int hl = HAP_compute_res_hmx_lock(hctx);
    uint64_t t0 = pcyc();
    if (vtcm) { gdn_work_t w = (gdn_work_t){ 0, 0u, a->H, 1u, a->Au, a->Tu, a->zpA, a->M, a->S, a->sT, a->zpT, vtcm };
                gdn_br_run_slot(&w); }
    g_ov_cyc[0] = pcyc() - t0;
    if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx); if (vctx) HAP_compute_res_release(vctx);
    if (hvx == 0) qurt_hvx_unlock();
}
static void ov_hvx(void *p) {
    struct ov_arg *a = (struct ov_arg *)p;
    int hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);
    gdn_scr_t *sc = &g_scr[1];
    uint64_t t0 = pcyc();
    for (int it = 0; it < a->iters; ++it)        /* the real HVX-role work: 64x64 forward-subst */
        gdn_solve_diag64(sc, a->Au, GDN_BR_C, a->zpA, a->M, a->S, sc->Tblk[0], nullptr);
    g_ov_cyc[1] = pcyc() - t0;
    if (hvx == 0) qurt_hvx_unlock();
}
static qurt_thread_t ov_spawn(void (*f)(void *), void *arg, int slot) {
    qurt_thread_t tid = 0; qurt_thread_attr_t at; qurt_thread_attr_init(&at);
    qurt_thread_attr_set_name(&at, (char *)"ov"); qurt_thread_attr_set_stack_addr(&at, g_ov_stack[slot]);
    qurt_thread_attr_set_stack_size(&at, sizeof(g_ov_stack[slot])); qurt_thread_attr_set_priority(&at, 0x80);
    if (qurt_thread_create(&tid, &at, f, arg) != QURT_EOK) { f(arg); return 0; }
    return tid;
}
static void ov_join(qurt_thread_t tid) { if (tid) { int s; qurt_thread_join(tid, &s); } }
#endif

#if defined(GDNBM_HMX_PIPE)
/* ===== GDNSolveHVXMixHMX producer-consumer (the real full solve, step 4) =====
 * P HVX producers run the FULL per-head solve (diag/fold/quant/pack/acc/widen/requant — all HVX); each
 * producer's per-matmul HMX kernel is DELEGATED to the single MAIN-thread consumer (1 HMX unit -> only the
 * main thread touches mxmem; multi-thread HMX SSRs).  Hand-off = one job slot/producer (synchronous: producer
 * fills slot, spins for the consumer's HMX result, then depacks).  A is VTCM-resident per-producer (step 1
 * ping-pong; producers are fold/A-bound).  Build: -DGDNBM_HMX_PIPE -DGDN_BR_STATIC_GAIN -DGDN_BR_STATIC_FULL
 * (run nthreads = #producers; consumer is the main thread, PURE HMX, frees its HVX unit for the Pth producer). */
struct hmx_job {
    volatile int state;                 /* 0=idle, 1=ready(consumer runs it), 2=done(producer depacks) */
    const gdn_vtcm_t *vt; const int8_t *wt; const int32_t *eff;
    float scale; int baseline; int round;
    int _pad[16];                       /* own cache/SMT line -> no false sharing between producer slots */
} __attribute__((aligned(128)));
static struct hmx_job g_pjob[GDN_BR_NT];
static volatile int g_pipe_pdone;       /* # producers that finished all their heads (drain-exit signal) */
static char __attribute__((aligned(128))) g_pipe_stack[GDN_BR_NT][32768];
static volatile uint64_t g_pipe_cbusy;          /* consumer: cyc inside the HMX kernel (busy, not spinning) */
static volatile uint64_t g_pipe_pspin[GDN_BR_NT];   /* producer t: cyc spinning for its matmul result */
static volatile uint32_t g_pipe_pcnt[GDN_BR_NT];    /* producer t: # matmuls dispatched */
static volatile uint64_t g_pipe_plife[GDN_BR_NT];   /* producer t: total lifetime cyc (HVX busy = life - spin) */

/* dispatch hook (installed into g_hmx_dispatch): hand the kernel to the main consumer, spin for the result. */
static void gdn_pipe_dispatch(gdn_scr_t *sc, const gdn_vtcm_t *vt, const int8_t *wt, const int32_t *eff,
                              float scale, int baseline, int round) {
    struct hmx_job *jb = &g_pjob[(int)(sc - g_scr)];           /* slot = this producer's scratch index */
#if defined(GDNBM_PIPE_PURE_HMX)
    /* LEVER 1: do the bias-pack + out-zero HERE (producer, HVX) so the consumer is PURE mxmem and never
     * touches an HVX unit -> frees its unit for a 4th producer (P=4 fits 4 HVX units w/o SMT oversubscribe). */
    { int rdelta = round ? (int)(256.0f / scale + 0.5f) : 0;
      gdn_pack_bias(eff, scale, baseline, (int32_t *)vt->bias, rdelta);
      HVX_Vector z = Q6_V_vzero(); HVX_Vector *op = (HVX_Vector *)vt->out;
      for (int i = 0; i < (64 * 64) / 128; ++i) op[i] = z; }
#endif
    jb->vt = vt; jb->wt = wt; jb->eff = eff; jb->scale = scale; jb->baseline = baseline; jb->round = round;
    __sync_synchronize();
    int slot = (int)(sc - g_scr);
    jb->state = 1;                                             /* arm: consumer may run it */
    uint64_t s0 = pcyc();
    while (jb->state != 2) { /* spin for the HMX result */ }
    g_pipe_pspin[slot] += pcyc() - s0; g_pipe_pcnt[slot]++;
    __sync_synchronize();
    jb->state = 0;                                             /* consumed; idle until next dispatch */
}

/* producer thread: full per-head solve over this slot's head-stripe, A VTCM-resident ping-pong. */
static void pipe_producer(void *arg) {
    gdn_work_t *w = (gdn_work_t *)arg;
    uint64_t _life0 = pcyc();
    int hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);
    gdn_scr_t *sc = &g_scr[w->slot];
    uint8_t *vbase = w->vtcm_base + (size_t)w->slot * 0xA0000u;
    gdn_vtcm_t vt = gdn_vtcm_from(vbase);
    const int CC = GDN_BR_C * GDN_BR_C;
    const uint32_t Abytes = (uint32_t)CC * 2u;
    uint16_t *Avt[2] = { (uint16_t *)(vbase + 0x60000), (uint16_t *)(vbase + 0x80000) };
    dma_desc_t dsc;
    uint32_t hs[64]; int n = 0;
    for (uint32_t h = w->h0 + w->slot; h < w->h1 && n < 64; h += w->nheads) hs[n++] = h;
    if (n > 0) udma_start(&dsc, Avt[0], w->Au + (size_t)hs[0] * CC, Abytes);
    for (int i = 0; i < n; ++i) {
        udma_wait();                                                                    /* A[i] ready */
        if (i + 1 < n) udma_start(&dsc, Avt[(i + 1) & 1], w->Au + (size_t)hs[i + 1] * CC, Abytes);
#if defined(GDN_BR_I16)
        gdn_br_one_head16(sc, &vt, Avt[i & 1], w->Tu + (size_t)hs[i] * CC,
                          w->zpA, w->M, w->S, w->sT, w->zpT);  /* int16 codes */
#else
        gdn_br_one_head(sc, &vt, Avt[i & 1], w->Tu + (size_t)hs[i] * CC,
                        w->zpA, w->M, w->S, w->sT, w->zpT);    /* matmuls auto-delegate via g_hmx_dispatch */
#endif
    }
    g_pipe_plife[w->slot] = pcyc() - _life0;   /* total lifetime cyc (HVX busy = lifetime - spin) */
    __sync_fetch_and_add(&g_pipe_pdone, 1);
    if (hvx == 0) qurt_hvx_unlock();
}
#endif  /* GDNBM_HMX_PIPE */

static void solve_worker(void *arg) {
    gdn_work_t *w = (gdn_work_t *)arg;
    int hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);
#if defined(GDNBM_HMX_MERGE_PATH)
    /* GDNSolveHVXMixHMX: spawned worker w/ qurt_hmx_lock (bare-metal HMX on a spawned thread IS fine —
     * like gdn_br_worker). 1 HMX unit -> mxmem serializes across workers, but diag/fold/quant/glue run
     * HVX-parallel across heads (HVX-locked threads <= 4 units). gdn_br_run_slot sets up vt's VTCM ptrs
     * (the VTCM-ping-pong path below is HVX-merge-only and leaves vt zeroed -> HMX mxmem would hit null). */
    int chmx = qurt_hmx_lock();
    gdn_br_run_slot(w);
    if (chmx == 0) qurt_hmx_unlock();
    if (hvx == 0) qurt_hvx_unlock();
    return;
#endif
#if defined(GDNBM_HMX_SOLVE)
    /* GDNSolveHVXMixHMX — SINGLE-THREAD de-risk version (precision先不管, u8i8 HMX merge).
     * One worker locks HMX once and runs the FULL solve with the HMX matmul (gdn_merge_packed, the
     * correct #else branch of gdn_br_one_head).  Acquires its OWN 0x60000 VTCM slice for vt's surfaces+
     * caches (gdn_vtcm_from layout span < 0x59000).  A read straight from DDR (uncached — slower; this is
     * the correctness+cyc baseline.  A-resident ping-pong + 4-producer/1-consumer pipeline come next).
     * Run with nthreads=1 (1 HMX unit -> no multi-thread HMX SSR). */
    {
        /* VTCM: 0xA0000 = vt surfaces+caches (gdn_vtcm_from span <0x59000 @ +0) + A ping-pong (2x128KB @ +0x60000). */
        compute_res_attr_t va; HAP_compute_res_attr_init(&va);
        HAP_compute_res_attr_set_vtcm_param(&va, 0xA0000u, 0);
        unsigned int vctx = HAP_compute_res_acquire(&va, 2000000);
        uint8_t *vtcm = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&va);
        /* HMX via the HAP compute_res lock (NOT qurt_hmx_lock — that symbol is unresolved on this
         * device -> .so load fails 0x80000406; FEED_4P's main-thread consumer uses HAP, same here). */
        compute_res_attr_t ha; HAP_compute_res_attr_init(&ha); HAP_compute_res_attr_set_hmx_param(&ha, 1);
        unsigned int hctx = HAP_compute_res_acquire(&ha, 2000000);
        int hl = HAP_compute_res_hmx_lock(hctx);
        gdn_scr_t *sc = &g_scr[w->slot];
        gdn_vtcm_t vt = vtcm ? gdn_vtcm_from(vtcm) : gdn_vtcm_t{};
        const int CC = GDN_BR_C * GDN_BR_C;
#if defined(GDNBM_HMX_A_DDR)
        if (vtcm) for (uint32_t h = w->h0 + w->slot; h < w->h1; h += w->nheads)
            gdn_br_one_head(sc, &vt, w->Au + (size_t)h * CC, w->Tu + (size_t)h * CC,
                            w->zpA, w->M, w->S, w->sT, w->zpT);
#else
        /* A VTCM-RESIDENT: DMA each head's A (128KB) DDR->VTCM, ping-pong so head h+1's A loads while head h
         * computes.  diag(fold A_ii)+merge(fold A_ik) are A-bound; uncached DDR A is ~7.8x slower (route doc). */
        const uint32_t Abytes = (uint32_t)CC * 2u;
        uint16_t *Avt[2] = { (uint16_t *)(vtcm + 0x60000), (uint16_t *)(vtcm + 0x80000) };
        dma_desc_t dsc;
        if (vtcm) {
            uint32_t hs[64]; int n = 0;
            for (uint32_t h = w->h0 + w->slot; h < w->h1 && n < 64; h += w->nheads) hs[n++] = h;
            if (n > 0) udma_start(&dsc, Avt[0], w->Au + (size_t)hs[0] * CC, Abytes);
            for (int i = 0; i < n; ++i) {
                udma_wait();                                                                  /* A[i] ready */
                if (i + 1 < n) udma_start(&dsc, Avt[(i + 1) & 1], w->Au + (size_t)hs[i + 1] * CC, Abytes);  /* prefetch A[i+1] */
                gdn_br_one_head(sc, &vt, Avt[i & 1], w->Tu + (size_t)hs[i] * CC,
                                w->zpA, w->M, w->S, w->sT, w->zpT);                            /* compute (overlaps prefetch) */
            }
        }
#endif
        if (hl == 0) HAP_compute_res_hmx_unlock(hctx);
        if (hctx) HAP_compute_res_release(hctx);
        if (vctx) HAP_compute_res_release(vctx);
#if defined(GDN_BR_SACC_CAL)
        ((float *)w->Tu)[0] = g_cal_swS;   /* dump calibrated max Sacc scale */
#endif
        if (hvx == 0) qurt_hvx_unlock();
    }
    return;
#endif
#if defined(GDNBM_VTCM_RESIDENT) && !defined(GDNBM_MM_TEST) && !defined(GDNBM_Q_TEST) && !defined(GDNBM_MERGE_TEST)
    /* SKILL principles: (2) keep A resident in VTCM (not uncached FastRPC DDR -> bare-metal diag was
     * 373K vs QNN 48K), (3) UDMA ping-pong so head h+1's A loads while head h computes. T writes go
     * straight to DDR (sequential, L2-prefetch friendly).
     * VTCM is acquired ONCE on the main thread (gdnbm_solve) and shared; each worker uses its own
     * 0x60000 slice via w->vtcm_base (per-worker HAP_compute_res_acquire SERIALIZES the workers — the
     * resource manager grants VTCM per-context, so concurrent acquires block on each other). */
    uint8_t *vtcm = w->vtcm_base ? (w->vtcm_base + (size_t)w->slot * 0x60000) : nullptr;
    /* NOTE: scratch (gdn_scr_t) STAYS in DDR BSS — measured: moving it to VTCM made the solve 7x SLOWER
     * (471K->3.48M) because the merge/diag scratch is accessed by SCALAR/data-dependent code, and VTCM
     * scalar access is catastrophically slow.  "all-in-VTCM" applies ONLY to HVX/HMX/DMA-accessed data
     * (here: A).  T writes go straight to DDR (sequential, L2-prefetch friendly). */
    gdn_scr_t *sc = &g_scr[w->slot];
    gdn_vtcm_t vt; memset(&vt, 0, sizeof(vt));
    const int CC = GDN_BR_C * GDN_BR_C;
    const uint32_t Abytes = (uint32_t)CC * 2u;              /* u16 A block = 128KB */
    uint16_t *Avt[2] = { (uint16_t *)vtcm, (uint16_t *)(vtcm + 0x20000) };
    dma_desc_t dsc;
    if (vtcm) {
        /* build this worker's head list, ping-pong over it */
        uint32_t hs[64]; int n = 0;
        for (uint32_t h = w->h0 + w->slot; h < w->h1 && n < 64; h += w->nheads) hs[n++] = h;
#if !defined(GDN_BR_T_DDR_DIRECT)
        /* DEFAULT: OUTPUT VIA VTCM + DMA WRITEBACK (skill principle 2/3: don't write T to DDR per-head;
         * keep it in VTCM and DMA it out, off the HVX store path).  Per-head T (zero-fill + requant) is
         * computed into a VTCM buffer Tvt, then DMA'd VTCM->DDR (dstbypass).  A-prefetch overlaps compute;
         * the T[i] writeback is issued AFTER A[i+1] is fully in (engine free) so the two transfers never
         * have concurrent dmstarts (a 2nd Q6_dmstart_A while one is in-flight CLOBBERS it -- device-verified
         * to corrupt heads).  Removes the 4-thread DDR-write-bandwidth contention that caps per-head
         * zero-fill+requant scaling: P=4 ~140K(jittery) -> ~122K(steady), 2.95x->3.39x, bit-exact.
         * Legacy per-head DDR-direct path: -DGDN_BR_T_DDR_DIRECT. */
        uint16_t *Tvt = (uint16_t *)(vtcm + 0x40000);
        dma_desc_t dsc_t;
        if (n > 0) udma_start(&dsc, Avt[0], w->Au + (size_t)hs[0] * CC, Abytes);   /* A[0] in */
        for (int i = 0; i < n; ++i) {
            udma_wait();                                          /* A[i] ready */
            if (i + 1 < n) udma_start(&dsc, Avt[(i + 1) & 1], w->Au + (size_t)hs[i + 1] * CC, Abytes); /* prefetch A[i+1] (overlaps compute) */
            gdn_br_one_head(sc, &vt, Avt[i & 1], Tvt, w->zpA, w->M, w->S, w->sT, w->zpT);  /* compute -> VTCM */
            if (i + 1 < n) udma_wait();                           /* A[i+1] done -> engine free (no concurrent dmstart) */
            udma_fill(&dsc_t, w->Tu + (size_t)hs[i] * CC, Tvt, Abytes, 1); dsc_t.next = nullptr;
            Q6_dmstart_A((void *)&dsc_t); udma_wait();            /* T[i] writeback (serial, off HVX store path) */
        }
#else
        if (n > 0) udma_start(&dsc, Avt[0], w->Au + (size_t)hs[0] * CC, Abytes);
        for (int i = 0; i < n; ++i) {
            udma_wait();                                    /* cur A ready */
            if (i + 1 < n) udma_start(&dsc, Avt[(i + 1) & 1],
                                      w->Au + (size_t)hs[i + 1] * CC, Abytes);  /* prefetch next */
            gdn_br_one_head(sc, &vt, Avt[i & 1], w->Tu + (size_t)hs[i] * CC,
                            w->zpA, w->M, w->S, w->sT, w->zpT);   /* reads A from VTCM */
        }
#endif
    }
    if (hvx == 0) qurt_hvx_unlock();
    return;
#endif
#if defined(GDNBM_STAGE_A) && !defined(GDNBM_MM_TEST) && !defined(GDNBM_Q_TEST) && !defined(GDNBM_MERGE_TEST)
    /* stage each head's A (256x256 u16 = 128KB) DDR->VTCM once, so the diag/fold read from TCM (the
     * FastRPC user buffer is uncached DDR -> the bare-metal diag is 7.8x QNN's). VTCM-only acquire (HVX path). */
    compute_res_attr_t va; HAP_compute_res_attr_init(&va);
    HAP_compute_res_attr_set_vtcm_param(&va, 0x40000u, 0);
    unsigned int vctx = HAP_compute_res_acquire(&va, 2000000);
    uint8_t *vtcm = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&va);
    gdn_scr_t *sc = &g_scr[w->slot];
    gdn_vtcm_t vt; memset(&vt, 0, sizeof(vt));
    const int CC = GDN_BR_C * GDN_BR_C;
    for (uint32_t h = w->h0 + w->slot; h < w->h1; h += w->nheads) {
        const uint16_t *Ah = w->Au + (size_t)h * CC;
        if (vtcm) {
            const HVX_UVector *s = (const HVX_UVector *)Ah; HVX_Vector *d = (HVX_Vector *)vtcm;
            for (int v = 0; v < CC * 2 / 128; ++v) d[v] = s[v];
            Ah = (const uint16_t *)vtcm;
        }
        gdn_br_one_head(sc, &vt, Ah, w->Tu + (size_t)h * CC, w->zpA, w->M, w->S, w->sT, w->zpT);
    }
    if (vctx) HAP_compute_res_release(vctx);
    if (hvx == 0) qurt_hvx_unlock();
    return;
#endif
#ifdef GDNBM_MM_TEST
    /* matmul isolation: A=100*I, B[k,j]=k*64+j -> C[i,j] must = 100*(i*64+j). Writes C int32 to Tu. */
    { gdn_scr_t *sc = &g_scr[w->slot];
      for (int i = 0; i < 64 * 64; ++i) sc->a16[i] = 2047;
      for (int i = 0; i < 64 * 64; ++i) sc->b16[i] = 2047;
      gdn_matmul_i16(sc->a16, sc->b16, (int32_t *)w->Tu); }  /* C[i,j] must = 64*2047*2047 = 268304896 */
#elif defined(GDNBM_MERGE_TEST)
    /* merge isolation: A=Tc, B=Sacc (small ramps, scale 1.0) -> C*s_out must ~= A@B. Writes C int32 + s_out. */
    { gdn_scr_t *sc = &g_scr[w->slot];
      for (int i = 0; i < 64 * 64; ++i) sc->Tc[i] = (i % 11) - 5;
      for (int i = 0; i < 64 * 64; ++i) sc->Sacc[i] = (i % 7) - 3;
      float s; gdn_merge_hvx(sc, sc->Tc, 1.0f, -1, sc->Sacc, 1.0f, -1, sc->Tblk[0], &s);
      for (int i = 0; i < 64 * 64; ++i) ((int32_t *)w->Tu)[i] = sc->Tblk[0][i];
      ((float *)w->Tu)[64 * 64] = s; }
#elif defined(GDNBM_MM_I8_TEST)
    /* int8 vrmpy matmul isolation: deterministic int8 A,B -> compare gdn_matmul_i8_vrmpy against the
     * trusted gdn_matmul_i16 (fed the SAME int8 values widened to int16).  Writes to Tu:
     *   [0]=max|diff|  [1]=first mismatch index (-1 if none)  [2..]=C_vrmpy[0..]  for inspection. */
    { gdn_scr_t *sc = &g_scr[w->slot];
      for (int k = 0; k < 64; ++k) for (int j = 0; j < 64; ++j) {
          int b = ((k * 3 + j) % 11) - 5; sc->b8[k * 64 + j] = (int8_t)b; sc->b16[k * 64 + j] = (int16_t)b; }
      for (int i = 0; i < 64; ++i) for (int k = 0; k < 64; ++k) {
          int a = ((i + 2 * k) % 7) - 3; sc->a8[i * 64 + k] = (int8_t)a; sc->a16[i * 64 + k] = (int16_t)a; }
      static int32_t __attribute__((aligned(128))) Cref[64 * 64], Cv[64 * 64];
      gdn_matmul_i16(sc->a16, sc->b16, Cref);
      gdn_pack_b_vrmpy(sc->b8, sc->btp);
      gdn_matmul_i8_vrmpy(sc->a8, sc->btp, Cv);
      int32_t *o = (int32_t *)w->Tu; int md = 0, fm = -1;
      for (int t = 0; t < 64 * 64; ++t) { int d = Cv[t] - Cref[t]; if (d < 0) d = -d; if (d > md) md = d;
          if (d && fm < 0) fm = t; }
      o[0] = md; o[1] = fm; for (int t = 0; t < 16; ++t) { o[2 + t] = Cv[t]; o[18 + t] = Cref[t]; } }
#elif defined(GDNBM_Q_TEST)
    /* quant isolation: codes[i]=i-2048 @ scale 1.0 -> out[i] ~ clamp(i-2048, +-2047). Writes int16->int32 to Tu. */
    { gdn_scr_t *sc = &g_scr[w->slot];
      for (int i = 0; i < 64 * 64; ++i) sc->Tc[i] = i - 2048;
      gdn_quant_i12_from_codes(sc, sc->Tc, 1.0f, sc->a16, -1);
      for (int i = 0; i < 64 * 64; ++i) ((int32_t *)w->Tu)[i] = sc->a16[i]; }
#elif defined(GDNBM_GLUE_BENCH)
    /* chain8-style steady microbench of the per-merge GLUE stages, single-head (baseline for the
     * N-head-batched comparison). Runs each stage REPS times back-to-back (cold amortized); writes
     * steady cyc/call into Tu[0..]. nthreads=1. Build: -DGDN_BR_MM_I8 -DGDNBM_GLUE_BENCH. */
    { gdn_scr_t *sc = &g_scr[w->slot];
      for (int i = 0; i < 64*64; ++i) { sc->Tc[i] = (int32_t)((i*7) % 4096) - 2048;
          sc->a8[i] = (int8_t)((i % 11) - 5); sc->b8[i] = (int8_t)(((i*3) % 11) - 5); }
      const int REPS = 200;
      int32_t mx = gdn_maxabs_codes(sc->Tc);
      uint64_t t0, t1; int32_t *o = (int32_t *)w->Tu;
      /* warm */ gdn_quant_i8_from_codes(sc, sc->Tc, 1.0f, sc->a8, mx);
      t0 = pcyc(); for (int r = 0; r < REPS; ++r) { volatile int32_t m = gdn_maxabs_codes(sc->Tc); (void)m; } t1 = pcyc();
      o[0] = (int32_t)((t1 - t0) / REPS);                                   /* maxabs reduction (64x64 i32) */
      t0 = pcyc(); for (int r = 0; r < REPS; ++r) gdn_quant_i8_from_codes(sc, sc->Tc, 1.0f, sc->a8, mx); t1 = pcyc();
      o[1] = (int32_t)((t1 - t0) / REPS);                                   /* int8 quant from i32 codes */
      t0 = pcyc(); for (int r = 0; r < REPS; ++r) gdn_pack_b_vrmpy(sc->b8, sc->btp); t1 = pcyc();
      o[2] = (int32_t)((t1 - t0) / REPS);                                   /* vrmpy B-pack */
      t0 = pcyc(); for (int r = 0; r < REPS; ++r) gdn_matmul_i8_vrmpy(sc->a8, sc->btp, sc->Tblk[0]); t1 = pcyc();
      o[3] = (int32_t)((t1 - t0) / REPS);                                   /* vrmpy 64^3 matmul */
      t0 = pcyc(); for (int r = 0; r < REPS; ++r) gdn_solve_diag64(sc, (const uint16_t *)sc->Tc, 64, 0, w->M, w->S, sc->Tblk[0]); t1 = pcyc();
      o[4] = (int32_t)((t1 - t0) / REPS);                                   /* 64 forward-subst diag */
#if defined(GDN_BR_DIAG_SPLIT)
      /* PHASE-1: CSE-PROOF forward-subst-only cyc/block. The const-input fwdsubst is otherwise hoisted
       * by -O2 across the REPS loop (gave a bogus ~300 < the 2016-vmpyacc floor). A rep-carried data
       * dependency (Tc16 last elem -> Afx[0] scalar -> next rep) forces every rep to execute. */
      { const int16_t ei16 = (int16_t)(int)(1.0f / GDN_BR_TI + 0.5f);
        gdn_fold_block_hvx((const uint16_t *)sc->Tc, 64, sc->Afx, 0, w->M, w->S);
        gdn_diag_fwdsubst(sc->Afx, sc->Tc16, ei16);                         /* warm */
        int32_t base0 = sc->Afx[0]; int16_t b16 = (int16_t)(base0 & 0xFFFF);
        volatile int32_t fb = 0;
        t0 = pcyc();
        for (int r = 0; r < REPS; ++r) {
            int16_t c = (int16_t)(b16 + (int16_t)(fb & 1));                 /* perturb -> defeat CSE */
            sc->Afx[0] = ((int32_t)(uint16_t)c) * 0x10001;                  /* keep both halfwords = scalar */
            gdn_diag_fwdsubst(sc->Afx, sc->Tc16, ei16);
            fb = sc->Tc16[63 * 64 + 63];                                    /* rep dependency */
        }
        t1 = pcyc();
        o[5] = (int32_t)((t1 - t0) / REPS);                                 /* CSE-proof fwdsubst cyc/block (floor 2016) */
        *(volatile int32_t *)sc->qbuf = fb; }
#endif
      /* --- N-HEAD-BATCHED variants: 2 heads interleaved (cross-head ILP fills VLIW slots / hides latency) --- */
      gdn_pack_b_vrmpy(sc->b8, sc->btp);                                    /* reuse same B for both "heads" (timing only) */
      const int8_t *A0 = sc->a8, *A1 = sc->a8, *bt0 = sc->btp, *bt1 = sc->btp;
      int32_t *C0 = sc->Tblk[0], *C1 = sc->Tc;                              /* distinct outputs */
      t0 = pcyc();
      for (int r = 0; r < REPS; ++r)
        for (int i = 0; i < 64; ++i) {                                      /* 2-head matmul: 4 acc chains */
          const int32_t *aw0 = (const int32_t *)(A0 + i*64), *aw1 = (const int32_t *)(A1 + i*64);
          HVX_Vector q0 = Q6_V_vzero(), q1 = Q6_V_vzero(), q2 = Q6_V_vzero(), q3 = Q6_V_vzero();
          for (int g = 0; g < 16; ++g) {
            HVX_Vector vA0 = Q6_V_vsplat_R(aw0[g]), vA1 = Q6_V_vsplat_R(aw1[g]);
            const HVX_Vector *b0 = (const HVX_Vector *)(bt0 + g*256), *b1 = (const HVX_Vector *)(bt1 + g*256);
            q0 = Q6_Vw_vrmpyacc_VwVbVb(q0, vA0, b0[0]); q1 = Q6_Vw_vrmpyacc_VwVbVb(q1, vA0, b0[1]);
            q2 = Q6_Vw_vrmpyacc_VwVbVb(q2, vA1, b1[0]); q3 = Q6_Vw_vrmpyacc_VwVbVb(q3, vA1, b1[1]);
          }
          ((HVX_Vector *)(C0 + i*64))[0] = q0; ((HVX_Vector *)(C0 + i*64))[1] = q1;
          ((HVX_Vector *)(C1 + i*64))[0] = q2; ((HVX_Vector *)(C1 + i*64))[1] = q3;
        }
      t1 = pcyc();
      o[5] = (int32_t)((t1 - t0) / REPS / 2);                               /* 2-head matmul, PER HEAD */
      const int32_t *cp0 = sc->Tblk[0], *cp1 = sc->Tc;
      t0 = pcyc();
      for (int r = 0; r < REPS; ++r) {                                      /* 2-head maxabs: 2 independent max vecs */
        HVX_Vector mx0 = Q6_V_vzero(), mx1 = Q6_V_vzero();
        const HVX_Vector *p0 = (const HVX_Vector *)cp0, *p1 = (const HVX_Vector *)cp1;
        for (int b = 0; b < (64*64)/32; ++b) {
          mx0 = Q6_Vw_vmax_VwVw(mx0, Q6_Vw_vabs_Vw(p0[b]));
          mx1 = Q6_Vw_vmax_VwVw(mx1, Q6_Vw_vabs_Vw(p1[b]));
        }
        HVX_Vector mm = Q6_Vw_vmax_VwVw(mx0, mx1);                          /* one combined reduction tail */
        mm = Q6_Vw_vmax_VwVw(mm, Q6_V_vror_VR(mm, 4*16)); mm = Q6_Vw_vmax_VwVw(mm, Q6_V_vror_VR(mm, 4*8));
        mm = Q6_Vw_vmax_VwVw(mm, Q6_V_vror_VR(mm, 4*4));  mm = Q6_Vw_vmax_VwVw(mm, Q6_V_vror_VR(mm, 4*2));
        volatile HVX_Vector s = Q6_Vw_vmax_VwVw(mm, Q6_V_vror_VR(mm, 4*1)); (void)s;
      }
      t1 = pcyc();
      o[6] = (int32_t)((t1 - t0) / REPS / 2);                               /* 2-head maxabs, PER HEAD */
      /* --- 4-head matmul: 8 acc chains (NH=4) --- */
      t0 = pcyc();
      for (int r = 0; r < REPS; ++r)
        for (int i = 0; i < 64; ++i) {
          const int32_t *aw = (const int32_t *)(sc->a8 + i*64);
          HVX_Vector q[8]; for (int t = 0; t < 8; ++t) q[t] = Q6_V_vzero();
          for (int g = 0; g < 16; ++g) {
            HVX_Vector vA = Q6_V_vsplat_R(aw[g]);
            const HVX_Vector *b = (const HVX_Vector *)(sc->btp + g*256);
            q[0] = Q6_Vw_vrmpyacc_VwVbVb(q[0], vA, b[0]); q[1] = Q6_Vw_vrmpyacc_VwVbVb(q[1], vA, b[1]);
            q[2] = Q6_Vw_vrmpyacc_VwVbVb(q[2], vA, b[0]); q[3] = Q6_Vw_vrmpyacc_VwVbVb(q[3], vA, b[1]);
            q[4] = Q6_Vw_vrmpyacc_VwVbVb(q[4], vA, b[0]); q[5] = Q6_Vw_vrmpyacc_VwVbVb(q[5], vA, b[1]);
            q[6] = Q6_Vw_vrmpyacc_VwVbVb(q[6], vA, b[0]); q[7] = Q6_Vw_vrmpyacc_VwVbVb(q[7], vA, b[1]);
          }
          ((HVX_Vector *)(sc->Tblk[0] + i*64))[0] = q[0]; ((HVX_Vector *)(sc->Tblk[0] + i*64))[1] = q[1];
          for (int t = 2; t < 8; ++t) { volatile HVX_Vector s = q[t]; (void)s; }
        }
      t1 = pcyc();
      o[7] = (int32_t)((t1 - t0) / REPS / 4);                               /* 4-head matmul, PER HEAD */
      /* --- crouton act-pack (the HVX work that FEEDS HMX): 1-head vs 2-head interleaved --- */
      uint8_t *crA = (uint8_t *)sc->Tblk[0], *crB = (uint8_t *)sc->Tc;      /* 4KB crouton tiles */
      const uint8_t *src = (const uint8_t *)sc->a8;
      t0 = pcyc(); for (int r = 0; r < REPS; ++r) gdn_pack_act_crouton8(src, crA); t1 = pcyc();
      o[8] = (int32_t)((t1 - t0) / REPS);                                   /* crouton act-pack 1-head */
      {   /* 2-head interleaved crouton pack: 2 independent load+mask+or streams */
        const HVX_Vector m  = Q6_V_valign_VVR(Q6_V_vzero(), Q6_Vb_vsplat_R(-1), 96);
        const HVX_Vector m1 = Q6_V_vror_VR(m, 96), m2 = Q6_V_vror_VR(m, 64), m3 = Q6_V_vror_VR(m, 32);
        t0 = pcyc();
        for (int r = 0; r < REPS; ++r) {
          HVX_Vector *dA = (HVX_Vector *)crA, *dB = (HVX_Vector *)crB; int v = 0;
          for (int kt = 0; kt < 2; ++kt) {
            const uint8_t *bA = src + kt*32, *bB = (const uint8_t *)sc->b8 + kt*32;   /* DISTINCT sources (no CSE) */
            for (int local = 0; local < 16; ++local) {
              int o0 = local*4, row0 = ((o0/8)&1)*32 + (o0/16)*8 + (o0%8);
              HVX_Vector A0 = *(const HVX_UVector *)(bA+(row0+0)*64), A1 = *(const HVX_UVector *)(bA+(row0+1)*64);
              HVX_Vector A2 = *(const HVX_UVector *)(bA+(row0+2)*64), A3 = *(const HVX_UVector *)(bA+(row0+3)*64);
              HVX_Vector B0 = *(const HVX_UVector *)(bB+(row0+0)*64), B1 = *(const HVX_UVector *)(bB+(row0+1)*64);
              HVX_Vector B2 = *(const HVX_UVector *)(bB+(row0+2)*64), B3 = *(const HVX_UVector *)(bB+(row0+3)*64);
              HVX_Vector oA = Q6_V_vor_VV(Q6_V_vor_VV(Q6_V_vand_VV(A0,m), Q6_V_vand_VV(Q6_V_vror_VR(A1,96),m1)),
                                          Q6_V_vor_VV(Q6_V_vand_VV(Q6_V_vror_VR(A2,64),m2), Q6_V_vand_VV(Q6_V_vror_VR(A3,32),m3)));
              HVX_Vector oB = Q6_V_vor_VV(Q6_V_vor_VV(Q6_V_vand_VV(B0,m), Q6_V_vand_VV(Q6_V_vror_VR(B1,96),m1)),
                                          Q6_V_vor_VV(Q6_V_vand_VV(Q6_V_vror_VR(B2,64),m2), Q6_V_vand_VV(Q6_V_vror_VR(B3,32),m3)));
              dA[v] = oA; dB[v] = oB; ++v;
            }
          }
        }
        t1 = pcyc();
        o[9] = (int32_t)((t1 - t0) / REPS / 2);                             /* crouton act-pack 2-head, PER HEAD */
      }
      t0 = pcyc(); for (int r = 0; r < REPS; ++r) gdn_pack_w8_kmajor(sc->a8, (int8_t *)crA); t1 = pcyc();
      o[10] = (int32_t)((t1 - t0) / REPS);                                  /* k-major wt-pack 1-head */
      t0 = pcyc(); for (int r = 0; r < REPS; ++r) gdn_depack_out_fast(sc, crA, 0, sc->a8); t1 = pcyc();
      o[11] = (int32_t)((t1 - t0) / REPS);                                  /* HMX-output depack 1-head */
      /* --- LEVER #A probe: act-crouton (vand/vor/vror = ALU+perm) vs wt-kmajor (vshuff = perm) --- */
      {   /* o[12] SEQUENTIAL 1-head: crouton(src->crA) then kmajor(b8->crB)         */
        const uint8_t *as = (const uint8_t *)sc->a8; const int8_t *ws = (const int8_t *)sc->b8;
        int8_t *kB = (int8_t *)sc->Tc;
        t0 = pcyc(); for (int r = 0; r < REPS; ++r) { gdn_pack_act_crouton8(as, crA); gdn_pack_w8_kmajor(ws, kB); } t1 = pcyc();
        o[12] = (int32_t)((t1 - t0) / REPS);                               /* act+wt SEQUENTIAL (sum) */
        /* o[13] FUSED interleaved: one 32-iter loop emitting one crouton vec + one kmajor vec per iter,
         * so the ALU(and/or) and the permute(vshuff) streams can co-issue in the same VLIW packet. */
        const HVX_Vector m  = Q6_V_valign_VVR(Q6_V_vzero(), Q6_Vb_vsplat_R(-1), 96);
        const HVX_Vector m1 = Q6_V_vror_VR(m, 96), m2 = Q6_V_vror_VR(m, 64), m3 = Q6_V_vror_VR(m, 32);
        HVX_Vector *dC = (HVX_Vector *)crA;
        t0 = pcyc();
        for (int r = 0; r < REPS; ++r) {
          for (int fi = 0; fi < 32; ++fi) {
            /* crouton index */
            int kt = fi >> 4, local = fi & 15; const uint8_t *b = as + kt*32;
            int o0 = local*4, row0 = ((o0/8)&1)*32 + (o0/16)*8 + (o0%8);
            HVX_Vector a0 = *(const HVX_UVector *)(b+(row0+0)*64), a1 = *(const HVX_UVector *)(b+(row0+1)*64);
            HVX_Vector a2 = *(const HVX_UVector *)(b+(row0+2)*64), a3 = *(const HVX_UVector *)(b+(row0+3)*64);
            /* kmajor index */
            int kt2 = fi >> 4, rem = fi & 15, nt = rem >> 3, r4 = rem & 7;
            const int8_t *wb = ws + kt2*32 + nt*32*0;  int nbase = nt*32, kbase = kt2*32;
            HVX_Vector w0 = *(const HVX_UVector *)(ws+(kbase+4*r4+0)*64+nbase), w1 = *(const HVX_UVector *)(ws+(kbase+4*r4+1)*64+nbase);
            HVX_Vector w2 = *(const HVX_UVector *)(ws+(kbase+4*r4+2)*64+nbase), w3 = *(const HVX_UVector *)(ws+(kbase+4*r4+3)*64+nbase);
            (void)wb;
            /* crouton (ALU+perm) */
            HVX_Vector oc = Q6_V_vor_VV(Q6_V_vor_VV(Q6_V_vand_VV(a0,m), Q6_V_vand_VV(Q6_V_vror_VR(a1,96),m1)),
                                        Q6_V_vor_VV(Q6_V_vand_VV(Q6_V_vror_VR(a2,64),m2), Q6_V_vand_VV(Q6_V_vror_VR(a3,32),m3)));
            /* kmajor (perm) */
            HVX_Vector p01 = Q6_V_lo_W(Q6_W_vshuff_VVR(w1, w0, -1));
            HVX_Vector p23 = Q6_V_lo_W(Q6_W_vshuff_VVR(w3, w2, -1));
            HVX_Vector ow  = Q6_V_lo_W(Q6_W_vshuff_VVR(p23, p01, -2));
            dC[fi] = oc; *(HVX_UVector *)((int8_t *)crB + fi*128) = ow;
          }
        }
        t1 = pcyc();
        o[13] = (int32_t)((t1 - t0) / REPS);                              /* act+wt FUSED interleaved */
        /* o[14]/o[15] CORRECTNESS: the fused 1-head index math (same as fp_pack_actwt2) must produce
         * byte-identical output to the validated original gdn_pack_act_crouton8 / gdn_pack_w8_kmajor. */
        static uint8_t refAct[4096], fusAct[4096]; static int8_t refWt[4096], fusWt[4096];
        gdn_pack_act_crouton8(as, refAct);
        gdn_pack_w8_kmajor(ws, refWt);
        { HVX_Vector *dC = (HVX_Vector *)fusAct;
          for (int fi = 0; fi < 32; ++fi) {
            int kt = fi >> 4, local = fi & 15; const uint8_t *b = as + kt*32;
            int o0 = local*4, row0 = ((o0/8)&1)*32 + (o0/16)*8 + (o0%8);
            HVX_Vector A0 = *(const HVX_UVector *)(b+(row0+0)*64), A1 = *(const HVX_UVector *)(b+(row0+1)*64);
            HVX_Vector A2 = *(const HVX_UVector *)(b+(row0+2)*64), A3 = *(const HVX_UVector *)(b+(row0+3)*64);
            int nt = local >> 3, r4 = local & 7, nb = nt*32, kb = kt*32, toff = (kt*2+nt)*1024 + r4*128;
            HVX_Vector w0 = *(const HVX_UVector *)(ws+(kb+4*r4+0)*64+nb), w1 = *(const HVX_UVector *)(ws+(kb+4*r4+1)*64+nb);
            HVX_Vector w2 = *(const HVX_UVector *)(ws+(kb+4*r4+2)*64+nb), w3 = *(const HVX_UVector *)(ws+(kb+4*r4+3)*64+nb);
            dC[fi] = Q6_V_vor_VV(Q6_V_vor_VV(Q6_V_vand_VV(A0,m), Q6_V_vand_VV(Q6_V_vror_VR(A1,96),m1)),
                                 Q6_V_vor_VV(Q6_V_vand_VV(Q6_V_vror_VR(A2,64),m2), Q6_V_vand_VV(Q6_V_vror_VR(A3,32),m3)));
            HVX_Vector pa = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_lo_W(Q6_W_vshuff_VVR(w3,w2,-1)), Q6_V_lo_W(Q6_W_vshuff_VVR(w1,w0,-1)), -2));
            *(HVX_UVector *)((int8_t *)fusWt + toff) = pa;
          } }
        int am = 0, wm = 0;
        for (int i = 0; i < 4096; ++i) { am += (refAct[i] != fusAct[i]); wm += (refWt[i] != fusWt[i]); }
        o[14] = am; o[15] = wm;                                          /* == 0 expected (byte-identical) */
      }
    }
#else
    gdn_br_run_slot(w);                 /* vtcm_base unused in GDNSolveHVX mode */
#endif
    if (hvx == 0) qurt_hvx_unlock();
}

#if defined(GDNBM_FEED_PIPE)
/* ===== GDNSolveHVXMixHMX PIPELINE: P HVX producers pack a VTCM ring; 1 main-thread PURE-HMX consumer drains it =====
 * (Naming: GDNSolveHVX = baseline pure-HVX; GDNSolveHVXMixHMX = this HVX-feed + HMX-matmul path;
 *  GDNSolveHMX = full-HMX, refuted. See Agent/current/gdn_solve_hvxmixhmx.md top.)
 * Throughput test of "feed a continuous HMX with cheaply-packed data". Each job = one 64^3 matmul.
 * Full state + reproduce: Agent/current/gdn_solve_hvxmixhmx.md.
 *
 * CURRENT IMPLEMENTATION (4P, -DGDNBM_FEED_4P):
 *   - 4 HVX producers (== 4 HVX units), static-stripe 2-head pairs into FP_K=16 VTCM slots (20KB each).
 *   - 1 MAIN-thread consumer, PURE HMX (hmx_lock only, NO qurt_hvx_lock -> frees an HVX unit for the 4th
 *     producer). Slot state machine g_fp_ready[k]: 0=free -> 1=packed -> 2=hmx-done.
 *   - Producer per pair: depack the 2 old occupants it's about to reuse (fp_depack2, 2-head interleaved),
 *     pack the new pair (fp_pack_act2 + fp_pack_wt2 + fp_pack_effbias2, all 2-head interleaved), signal 1.
 *   - Consumer per job: spin until state==1, run our_v73deep_kernel (mxmem), signal 2. NO out-zero (kernel
 *     fully overwrites, verified ovr_mism==0). One acquire barrier after the spin.
 *   - CAS-free: volatile ready flags + static striping (__sync CAS is unreliable on this target).
 *
 * MEASURED (real v75, C=256-shaped 64^3, DUMMY data, min-of-3): ~578 cyc/matmul = 2.09x vs the unpipelined
 * 1208, and 2.77x vs shipped vrmpy-4-thread (1601). Producer-feed-bound (consumer spins ~165/578 = 29%).
 *
 * VTCM TRAFFIC MAP (per matmul, THIS microbench = ONE HMX run/matmul):
 *     op                         VTCM read   VTCM write   (source/dest off-VTCM in [])
 *     producer pack act          [DDR src]      4K        HMX-mandated crouton layout
 *     producer pack wt           [DDR src]      4K        HMX-mandated k-major layout
 *     producer pack bias         [DDR src]      0.5K
 *     consumer HMX read          8.5K            -        act+wt+bias (the producer->consumer handoff)
 *     consumer HMX write out       -            4K        crouton u8 surface
 *     producer depack (read out) 4K           [DDR dst]   de-crouton -> int8 codes
 *     ------------------------------------------------------------------
 *     TOTAL                      12.5K        12.5K  = 25K/matmul, ALL HMX-mandated minimum for one run.
 *
 * BOTTLENECK (P-sweep P=2/3/4): producer-feed-bound + ~13% CONSERVED average-VTCM-bandwidth contention
 * (per-producer cost 2068->2340 & consumer-busy 367->418 BOTH rise with P). Scheduling/relayout
 * (phase-stagger, bank-aware, read/write split) CANNOT help -- it only shifts contention, never reduces
 * total traffic (phase-stagger tested: no-op). The pure-HMX consumer floor is ~418 (kernel 215 + desc +
 * 2 barriers), NOT the old 388; producer feed crosses it only at P~6 (impossible on 4 HVX units).
 *
 * >>> THE ONLY WAY DOWN IS TO REDUCE TRAFFIC (algorithm-level). The #1 lever is NOT here: the real solve's
 *     gdn_merge_packed (GdnSolveBROp.cpp; the GDNSolveHVXMixHMX matmul impl) runs the HMX kernel 2-3x PER
 *     LOGICAL MATMUL (a dynamic-quant gain search; PASS 1/2 outputs are thrown away) -> real traffic
 *     ~50K/matmul, ~66% of it pure scale-probing. This microbench measures only PASS 3. Killing/cheapening
 *     the multi-pass gain search is ~2-3x, dwarfing the depack round-trip and scheduling. See gdn_merge_packed
 *     and the Agent doc's traffic-reduction map. */
#define FP_K 16
#define FP_J 512
static volatile int g_fp_ready[FP_K];          /* 0=free (a packer may fill), 1=packed (owner may drain) */
static volatile int g_fp_done;                 /* OPCACHE: consumer sets after FP_J -> producers stop re-arming */
static int g_fp_T;                             /* worker count (static striping; CAS-free) */
static volatile uint64_t g_fp_pwork[GDN_BR_NT], g_fp_pspin[GDN_BR_NT];  /* producer pack-work vs slot-wait */
static uint8_t *g_fp_base;
static char __attribute__((aligned(128))) g_fp_stack[GDN_BR_NT][32768];
static uint8_t __attribute__((aligned(128))) g_fp_act[64*64];
static int8_t  __attribute__((aligned(128))) g_fp_wt[64*64];
static uint8_t __attribute__((aligned(128))) g_fp_act2[64*64];   /* 2nd head source (distinct -> real ILP, no CSE) */
static int8_t  __attribute__((aligned(128))) g_fp_wt2[64*64];
static int8_t  __attribute__((aligned(128))) g_fp_outc[GDN_BR_NT][64*64];
static int8_t  __attribute__((aligned(128))) g_fp_outc2[GDN_BR_NT][64*64];  /* 2nd depack dst (distinct -> ILP) */
static gdn_vtcm_t fp_slot(int k) {
    uint8_t *b = g_fp_base + (size_t)k * 0x5000;             /* 20KB/slot; tabs PAST the 4KB out surface */
    gdn_vtcm_t v; v.act = b; v.wt = (int8_t *)(b + 0x1000); v.bias = (int32_t *)(b + 0x2000);
    v.out = b + 0x3000; v.acttab = (int32_t *)(b + 0x4000); v.outtab = (int32_t *)(b + 0x4080);
    v.acache = nullptr; v.wcache = nullptr; return v;
}
static uint32_t fp_bias_ctrl(void) {                          /* constant control word (scale=1/64, baseline=128<<7) */
    return ((uint32_t)((128 << 7) & 0xFFFF) << 16) | (uint32_t)gdn_f16_bits(1.0f / 64.0f);
}
static void fp_pack_slot(int k, int32_t *eff) {              /* HVX feed work (vectorized eff+bias) */
    (void)eff; gdn_vtcm_t vt = fp_slot(k);
    gdn_pack_act_crouton8(g_fp_act, vt.act);
    gdn_pack_w8_kmajor(g_fp_wt, vt.wt);
    fp_pack_effbias(g_fp_wt, fp_bias_ctrl(), 0, vt.bias);    /* vectorized: was gdn_effective + gdn_pack_bias (scalar) */
    vt.acttab[0] = (int32_t)(uintptr_t)(vt.act + 0); vt.acttab[1] = (int32_t)(uintptr_t)(vt.act + 64*32);
    vt.outtab[0] = (int32_t)(uintptr_t)(vt.out + 0); vt.outtab[1] = (int32_t)(uintptr_t)(vt.out + 64*32);
}
/* 2-head interleaved crouton act-pack (distinct sources sA/sB -> outA/outB): real ILP ~1.35x. */
static void fp_pack_act2(const uint8_t *sA, const uint8_t *sB, uint8_t *outA, uint8_t *outB) {
    const HVX_Vector m = Q6_V_valign_VVR(Q6_V_vzero(), Q6_Vb_vsplat_R(-1), 96);
    const HVX_Vector m1 = Q6_V_vror_VR(m, 96), m2 = Q6_V_vror_VR(m, 64), m3 = Q6_V_vror_VR(m, 32);
    HVX_Vector *dA = (HVX_Vector *)outA, *dB = (HVX_Vector *)outB; int v = 0;
    for (int kt = 0; kt < 2; ++kt) {
        const uint8_t *bA = sA + kt*32, *bB = sB + kt*32;
        for (int local = 0; local < 16; ++local) {
            int o0 = local*4, row0 = ((o0/8)&1)*32 + (o0/16)*8 + (o0%8);
            HVX_Vector A0 = *(const HVX_UVector *)(bA+(row0+0)*64), A1 = *(const HVX_UVector *)(bA+(row0+1)*64);
            HVX_Vector A2 = *(const HVX_UVector *)(bA+(row0+2)*64), A3 = *(const HVX_UVector *)(bA+(row0+3)*64);
            HVX_Vector B0 = *(const HVX_UVector *)(bB+(row0+0)*64), B1 = *(const HVX_UVector *)(bB+(row0+1)*64);
            HVX_Vector B2 = *(const HVX_UVector *)(bB+(row0+2)*64), B3 = *(const HVX_UVector *)(bB+(row0+3)*64);
            dA[v] = Q6_V_vor_VV(Q6_V_vor_VV(Q6_V_vand_VV(A0,m), Q6_V_vand_VV(Q6_V_vror_VR(A1,96),m1)),
                                Q6_V_vor_VV(Q6_V_vand_VV(Q6_V_vror_VR(A2,64),m2), Q6_V_vand_VV(Q6_V_vror_VR(A3,32),m3)));
            dB[v] = Q6_V_vor_VV(Q6_V_vor_VV(Q6_V_vand_VV(B0,m), Q6_V_vand_VV(Q6_V_vror_VR(B1,96),m1)),
                                Q6_V_vor_VV(Q6_V_vand_VV(Q6_V_vror_VR(B2,64),m2), Q6_V_vand_VV(Q6_V_vror_VR(B3,32),m3)));
            ++v;
        }
    }
}
/* 2-head interleaved k-major wt-pack (distinct wA/wB -> pA/pB). */
static void fp_pack_wt2(const int8_t *wA, const int8_t *wB, int8_t *pA, int8_t *pB) {
    int out = 0;
    for (int kt = 0; kt < 2; ++kt) { int kb = kt*32;
        for (int nt = 0; nt < 2; ++nt) { int nb = nt*32; int8_t *tA = pA + out, *tB = pB + out;
            for (int r4 = 0; r4 < 8; ++r4) {
                HVX_Vector a0 = *(const HVX_UVector *)(wA+(kb+4*r4+0)*64+nb), a1 = *(const HVX_UVector *)(wA+(kb+4*r4+1)*64+nb);
                HVX_Vector a2 = *(const HVX_UVector *)(wA+(kb+4*r4+2)*64+nb), a3 = *(const HVX_UVector *)(wA+(kb+4*r4+3)*64+nb);
                HVX_Vector b0 = *(const HVX_UVector *)(wB+(kb+4*r4+0)*64+nb), b1 = *(const HVX_UVector *)(wB+(kb+4*r4+1)*64+nb);
                HVX_Vector b2 = *(const HVX_UVector *)(wB+(kb+4*r4+2)*64+nb), b3 = *(const HVX_UVector *)(wB+(kb+4*r4+3)*64+nb);
                HVX_Vector pa = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_lo_W(Q6_W_vshuff_VVR(a3,a2,-1)), Q6_V_lo_W(Q6_W_vshuff_VVR(a1,a0,-1)), -2));
                HVX_Vector pb = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_lo_W(Q6_W_vshuff_VVR(b3,b2,-1)), Q6_V_lo_W(Q6_W_vshuff_VVR(b1,b0,-1)), -2));
                *(HVX_UVector *)(tA + r4*128) = pa; *(HVX_UVector *)(tB + r4*128) = pb;
            }
            out += 1024;
        }
    }
}
/* LEVER #A: FUSED 2-head act-crouton + wt-kmajor in ONE 32-iter loop. crouton is ALU+perm
 * (vand/vor/vror), kmajor is pure perm (vshuff) -> interleaving the two streams co-issues the ALU and
 * permute pipelines (isolated probe: act+wt 2036->1570 = 1.30x). Produces the IDENTICAL output of
 * fp_pack_act2(...)+fp_pack_wt2(...). 4 output streams (actA/actB crouton, wtA/wtB kmajor). */
static void fp_pack_actwt2(const uint8_t *aA, const uint8_t *aB, const int8_t *wA, const int8_t *wB,
                           uint8_t *oaA, uint8_t *oaB, int8_t *owA, int8_t *owB) {
    const HVX_Vector m = Q6_V_valign_VVR(Q6_V_vzero(), Q6_Vb_vsplat_R(-1), 96);
    const HVX_Vector m1 = Q6_V_vror_VR(m, 96), m2 = Q6_V_vror_VR(m, 64), m3 = Q6_V_vror_VR(m, 32);
    HVX_Vector *dA = (HVX_Vector *)oaA, *dB = (HVX_Vector *)oaB;
    for (int fi = 0; fi < 32; ++fi) {
        int kt = fi >> 4, local = fi & 15;
        /* --- crouton (act), 2 heads --- */
        const uint8_t *bA = aA + kt*32, *bB = aB + kt*32;
        int o0 = local*4, row0 = ((o0/8)&1)*32 + (o0/16)*8 + (o0%8);
        HVX_Vector A0 = *(const HVX_UVector *)(bA+(row0+0)*64), A1 = *(const HVX_UVector *)(bA+(row0+1)*64);
        HVX_Vector A2 = *(const HVX_UVector *)(bA+(row0+2)*64), A3 = *(const HVX_UVector *)(bA+(row0+3)*64);
        HVX_Vector B0 = *(const HVX_UVector *)(bB+(row0+0)*64), B1 = *(const HVX_UVector *)(bB+(row0+1)*64);
        HVX_Vector B2 = *(const HVX_UVector *)(bB+(row0+2)*64), B3 = *(const HVX_UVector *)(bB+(row0+3)*64);
        /* --- kmajor (wt), 2 heads.  out index: tile (kt,nt)*1024 + r4*128, nt=local>>3, r4=local&7 --- */
        int nt = local >> 3, r4 = local & 7, nb = nt*32, kb = kt*32, toff = (kt*2+nt)*1024 + r4*128;
        HVX_Vector w0 = *(const HVX_UVector *)(wA+(kb+4*r4+0)*64+nb), w1 = *(const HVX_UVector *)(wA+(kb+4*r4+1)*64+nb);
        HVX_Vector w2 = *(const HVX_UVector *)(wA+(kb+4*r4+2)*64+nb), w3 = *(const HVX_UVector *)(wA+(kb+4*r4+3)*64+nb);
        HVX_Vector x0 = *(const HVX_UVector *)(wB+(kb+4*r4+0)*64+nb), x1 = *(const HVX_UVector *)(wB+(kb+4*r4+1)*64+nb);
        HVX_Vector x2 = *(const HVX_UVector *)(wB+(kb+4*r4+2)*64+nb), x3 = *(const HVX_UVector *)(wB+(kb+4*r4+3)*64+nb);
        dA[fi] = Q6_V_vor_VV(Q6_V_vor_VV(Q6_V_vand_VV(A0,m), Q6_V_vand_VV(Q6_V_vror_VR(A1,96),m1)),
                             Q6_V_vor_VV(Q6_V_vand_VV(Q6_V_vror_VR(A2,64),m2), Q6_V_vand_VV(Q6_V_vror_VR(A3,32),m3)));
        dB[fi] = Q6_V_vor_VV(Q6_V_vor_VV(Q6_V_vand_VV(B0,m), Q6_V_vand_VV(Q6_V_vror_VR(B1,96),m1)),
                             Q6_V_vor_VV(Q6_V_vand_VV(Q6_V_vror_VR(B2,64),m2), Q6_V_vand_VV(Q6_V_vror_VR(B3,32),m3)));
        HVX_Vector pa = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_lo_W(Q6_W_vshuff_VVR(w3,w2,-1)), Q6_V_lo_W(Q6_W_vshuff_VVR(w1,w0,-1)), -2));
        HVX_Vector pb = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_lo_W(Q6_W_vshuff_VVR(x3,x2,-1)), Q6_V_lo_W(Q6_W_vshuff_VVR(x1,x0,-1)), -2));
        *(HVX_UVector *)(owA + toff) = pa; *(HVX_UVector *)(owB + toff) = pb;
    }
}
static void fp_pack_slot2(int k0, int k1, int32_t *eff0, int32_t *eff1) {   /* pack 2 jobs, 2-head interleaved */
    gdn_vtcm_t v0 = fp_slot(k0), v1 = fp_slot(k1);
    (void)eff0; (void)eff1;
#if defined(GDNBM_FUSED_ACTWT)
    fp_pack_actwt2(g_fp_act, g_fp_act2, g_fp_wt, g_fp_wt2, v0.act, v1.act, (int8_t *)v0.wt, (int8_t *)v1.wt);
#else
    fp_pack_act2(g_fp_act, g_fp_act2, v0.act, v1.act);
    fp_pack_wt2(g_fp_wt, g_fp_wt2, (int8_t *)v0.wt, (int8_t *)v1.wt);
#endif
    uint32_t c = fp_bias_ctrl();
    fp_pack_effbias2(g_fp_wt, g_fp_wt2, c, 0, v0.bias, v1.bias);  /* 2-head interleaved eff+bias (ILP) */
    v0.acttab[0]=(int32_t)(uintptr_t)(v0.act+0); v0.acttab[1]=(int32_t)(uintptr_t)(v0.act+64*32);
    v0.outtab[0]=(int32_t)(uintptr_t)(v0.out+0); v0.outtab[1]=(int32_t)(uintptr_t)(v0.out+64*32);
    v1.acttab[0]=(int32_t)(uintptr_t)(v1.act+0); v1.acttab[1]=(int32_t)(uintptr_t)(v1.act+64*32);
    v1.outtab[0]=(int32_t)(uintptr_t)(v1.out+0); v1.outtab[1]=(int32_t)(uintptr_t)(v1.out+64*32);
}
static void fp_drain_slot(int k, gdn_scr_t *sc, int8_t *outc, const uint32_t *ep, const uint32_t *mb) {
    gdn_vtcm_t vt = fp_slot(k);
    { HVX_Vector z = Q6_V_vzero(); HVX_Vector *op = (HVX_Vector *)vt.out; for (int i = 0; i < (64*64)/128; ++i) op[i] = z; }
    hmx_conv_out_desc_t od __attribute__((aligned(64))) = { vt.outtab, GDN_BR_OUT_TABLE_STRIDE, GDN_BR_OUT_Y_STRIDE,
        GDN_BR_N_TILES_POW2, GDN_BR_M_TOTAL_MINUS_STEP, GDN_BR_K_TOTAL_BYTES };
    hmx_conv_act_desc_t ad __attribute__((aligned(64))) = { vt.acttab, GDN_BR_N_ACT_PAIRS, GDN_BR_ACT_Y_STRIDE };
    our_v73deep_kernel(&od, &ad, (const uint8_t *)vt.wt, (const uint8_t *)vt.bias, (const hmx_conv_mask_desc_t *)mb, ep);
    gdn_depack_out_fast(sc, vt.out, (128 << 7) >> 7, outc);
}
/* PRODUCER thread: static-stripe pack jobs into the ring (worker t -> jobs t, t+P, ...). The HMX
 * CONSUMER is the MAIN thread (see gdnbm_solve) — NOT a spawned worker (HMX wants the main/callback
 * thread; a spawned HMX consumer was flaky). CAS-free (volatile ready flags). Total = P producers +
 * main consumer; choose P = (#HVX units - 1) = 3 so total = 4 = the 4 HVX units (no oversubscription). */
static void feed_producer(void *arg) {
    int id = (int)(intptr_t)arg;
    int hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);
    int32_t eff0[64], eff1[64];
    uint32_t P = (uint32_t)g_fp_T;
    uint64_t pw = 0, ps = 0;
#if defined(GDNBM_OPCACHE)
    /* OPERAND-CACHE CEILING: pack each slot ONCE (operands resident), then the steady loop ONLY depacks +
     * re-arms (no per-matmul pack). Models the real solve reusing T_kj/A_ik across i,j. Measures whether
     * killing the producer's per-matmul pack-write traffic drops the contention-bound 507 toward the
     * consumer floor. (dummy data repeats — the CYCLES are real.) */
    {
        int8_t *poutc = g_fp_outc[id];
        for (int k = id; k < FP_K; k += P) {            /* pre-pack & arm my stripe slots ONCE */
            fp_pack_slot(k, eff0); __sync_synchronize(); g_fp_ready[k] = 1;
        }
        while (!g_fp_done) {                              /* steady: depack consumed slots + re-arm (cached) */
            for (int k = id; k < FP_K; k += (int)P) {
                if (g_fp_ready[k] == 2) {
                    __sync_synchronize();
#if !defined(GDNBM_OPCACHE_NODEP)
                    fp_depack(fp_slot(k).out, (128 << 7) >> 7, poutc);
#endif
                    __sync_synchronize();
                    g_fp_ready[k] = 1;
                }
            }
        }
    }
    (void)eff1; if (hvx == 0) qurt_hvx_unlock(); g_fp_pwork[id]=pw; g_fp_pspin[id]=ps; return;
#endif
#if defined(GDNBM_FEED_4P)
    /* 4-PRODUCER 3-stage: consumer is pure-HMX (frees its HVX unit). Slot states 0=free 1=packed 2=hmx-done.
     * This producer also DEPACKS the slot it's about to reuse (the previous occupant the consumer HMX'd). */
    int8_t *poutc = g_fp_outc[id], *poutc2 = g_fp_outc2[id];
    for (uint32_t base = (uint32_t)id; base < FP_J; base += 2*P) {   /* 2-head pairs */
        uint32_t ja = base, jb = base + P;
        int ka = ja % FP_K, kb = (jb < FP_J) ? (int)(jb % FP_K) : ka;
        while (g_fp_ready[ka] == 1 || g_fp_ready[kb] == 1) { /* wait until consumer HMX'd previous */ }
        __sync_synchronize();   /* single acquire barrier after the spin (vs one per depack — small win) */
        int da = (g_fp_ready[ka] == 2), db = (kb != ka && g_fp_ready[kb] == 2);  /* old occupants to depack */
        if (da && db) fp_depack2(fp_slot(ka).out, fp_slot(kb).out, (128<<7)>>7, poutc, poutc2);  /* 2-head interleaved */
        else { if (da) fp_depack(fp_slot(ka).out, (128<<7)>>7, poutc);
               if (db) fp_depack(fp_slot(kb).out, (128<<7)>>7, poutc2); }
        if (jb < FP_J) fp_pack_slot2(ka, kb, eff0, eff1); else fp_pack_slot(ka, eff0);
        /* NO out-surface zero: our_v73deep_kernel fully OVERWRITES the out surface (verified ovr_mism==0).
         * Phase-stagger (even depack→pack / odd pack→depack) was tried: no-op — the ~13% contention is
         * conserved average-bandwidth, not burst collision, so reordering only shifts it producer↔consumer. */
        __sync_synchronize();
        g_fp_ready[ka] = 1; if (jb < FP_J) g_fp_ready[kb] = 1;
    }
#elif defined(GDNBM_FEED_2H)
    for (uint32_t base = (uint32_t)id; base < FP_J; base += 2*P) {   /* 2-head: pack a PAIR of my stripe jobs */
        uint32_t ja = base, jb = base + P;
        if (jb < FP_J) {
            int ka = ja % FP_K, kb = jb % FP_K;
            uint64_t a = pcyc();
            while (g_fp_ready[ka] != 0) { } while (g_fp_ready[kb] != 0) { }
            uint64_t b = pcyc(); ps += b - a;
            fp_pack_slot2(ka, kb, eff0, eff1);
            uint64_t c = pcyc(); pw += c - b;
            __sync_synchronize();
            g_fp_ready[ka] = 1; g_fp_ready[kb] = 1;
        } else {
            int ka = ja % FP_K; while (g_fp_ready[ka] != 0) { } fp_pack_slot(ka, eff0);
            __sync_synchronize(); g_fp_ready[ka] = 1;
        }
    }
#else
    for (uint32_t j = (uint32_t)id; j < FP_J; j += P) {
        int k = j % FP_K;
        while (g_fp_ready[k] != 0) { /* wait for the consumer to free this slot */ }
        fp_pack_slot(k, eff0);
        __sync_synchronize();
        g_fp_ready[k] = 1;
    }
#endif
    g_fp_pwork[id] = pw; g_fp_pspin[id] = ps;
    if (hvx == 0) qurt_hvx_unlock();
}
#endif

int gdnbm_solve(remote_handle64 _h, const uint8_t *A, int ALen, int H, int C, int zpA, int zpT,
                int sA_bits, int sT_bits, int nthreads, uint8_t *T, int TLen, int *stats, int statsLen) {
    (void)_h; (void)ALen; (void)TLen;
    if (C != GDN_BR_C) return -1;
    if (nthreads > GDN_BR_NT) nthreads = GDN_BR_NT; if (nthreads < 1) nthreads = 1;
    if (H < nthreads) nthreads = H;
    const uint16_t *Au = (const uint16_t *)A;
    uint16_t *Tu = (uint16_t *)T;
    float sA = i2f(sA_bits), sT = i2f(sT_bits);
    int M, S; gdn_fold_MS(sA, &M, &S);

#if defined(GDNBM_HWINFO)
    {   /* DEVICE-CONFIRM the actual v75 chip HVX/VTCM config (not from headers). */
        int u = qurt_hvx_get_units();              /* bits15:8 = #128B units, bits7:0 = #64B units */
        unsigned int vt_total = 0, vt_avail = 0;
        HAP_compute_res_query_VTCM(0, &vt_total, nullptr, &vt_avail, nullptr);
        if (statsLen > 0) stats[0] = u;
        if (statsLen > 1) stats[1] = (u >> 8) & 0xff;   /* #128B units */
        if (statsLen > 2) stats[2] = u & 0xff;          /* #64B units */
        if (statsLen > 3) stats[3] = (int)vt_total;
        if (statsLen > 4) stats[4] = (int)vt_avail;
        FARF(ALWAYS, "HWINFO: hvx_units=0x%x (128B=%d, 64B=%d) VTCM total=%u avail=%u",
             u, (u >> 8) & 0xff, u & 0xff, vt_total, vt_avail);
        return 0;
    }
#endif

    /* POWER: vote turbo core clock + HMX power-on (the bare HMX acquire doesn't power/clock it -> mxmem hangs). */
    static int g_pwr_client; void *pctx = &g_pwr_client;
    HAP_power_set_core_corner(pctx, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_MAX);
    { HAP_power_request_t r; memset(&r,0,sizeof(r)); r.type=HAP_power_set_HMX; r.hmx.power_up=TRUE; HAP_power_set(pctx,&r); }

#if defined(GDNBM_HMX_PIPE)
    {   /* GDNSolveHVXMixHMX producer-consumer (step 4): P HVX producers feed 1 main-thread HMX consumer. */
        int P = nthreads; if (P > GDN_BR_NT) P = GDN_BR_NT; if (P < 1) P = 1;
        compute_res_attr_t va; HAP_compute_res_attr_init(&va);
        HAP_compute_res_attr_set_vtcm_param(&va, (unsigned)P * 0xA0000u, 0);   /* 0xA0000/producer: vt + A ping-pong */
        unsigned int vctx2 = HAP_compute_res_acquire(&va, 2000000);
        uint8_t *vbase = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&va);
        compute_res_attr_t ha; HAP_compute_res_attr_init(&ha); HAP_compute_res_attr_set_hmx_param(&ha, 1);
        unsigned int hctx = HAP_compute_res_acquire(&ha, 2000000);
        int hl = HAP_compute_res_hmx_lock(hctx);                               /* consumer = main, PURE HMX (no HVX lock) */
        if (!vbase || hl != 0) { FARF(ALWAYS, "PIPE: acquire failed vbase=%p hl=%d", vbase, hl);
            if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
            if (vctx2) HAP_compute_res_release(vctx2); return -1; }
        for (int t = 0; t < P; ++t) { g_pjob[t].state = 0; g_pipe_pspin[t] = 0; g_pipe_pcnt[t] = 0; g_pipe_plife[t] = 0; }
        g_pipe_cbusy = 0;
        g_pipe_pdone = 0;
        g_hmx_dispatch = gdn_pipe_dispatch;                                    /* matmuls now delegate to this consumer */
        __sync_synchronize();
        gdn_work_t work[GDN_BR_NT]; qurt_thread_t tid[GDN_BR_NT];
        uint64_t us0 = HAP_perf_get_time_us();
        uint64_t t0 = pcyc();
#if defined(GDN_BR_TRACE)
        g_tr_n = 0; g_tr_base = t0;   /* per-thread event timeline: producers push DIAG/MERGE/HEAD, consumer MM */
#endif
        for (int t = 0; t < P; ++t) {
            work[t] = (gdn_work_t){ t, 0u, (uint32_t)H, (uint32_t)P, Au, Tu, zpA, M, S, sT, zpT, vbase };
            qurt_thread_attr_t a; qurt_thread_attr_init(&a); qurt_thread_attr_set_name(&a,(char*)"pipeprod");
            qurt_thread_attr_set_stack_addr(&a,g_pipe_stack[t]); qurt_thread_attr_set_stack_size(&a,sizeof(g_pipe_stack[t]));
            if (qurt_thread_create(&tid[t],&a,pipe_producer,&work[t])!=QURT_EOK) tid[t]=0;  /* can't inline: main must drain */
        }
        /* consumer drain: run any armed job; exit when all producers finished.  A producer bumps g_pipe_pdone
         * only AFTER its last dispatch fully returned -> pdone==P implies no job is still armed. */
#if defined(GDNBM_PIPE_PURE_HMX)
        uint32_t pipe_ep[2] __attribute__((aligned(16))) = {1u, 0u};                  /* PURE-HMX consumer: bias-pack */
        uint32_t pipe_mb[16] __attribute__((aligned(16))); for (int i = 0; i < 16; ++i) pipe_mb[i] = GDN_BR_MASK_WORDS[i];
#endif
        while (g_pipe_pdone < P) {
            for (int t = 0; t < P; ++t) {
                if (g_pjob[t].state == 1) {
                    struct hmx_job *jb = &g_pjob[t];
                    uint64_t k0 = pcyc();
#if defined(GDNBM_PIPE_PURE_HMX)
                    /* LEVER 1: PURE mxmem — bias-pack+out-zero already done producer-side; consumer never
                     * touches an HVX unit -> P=4 producers fit 4 HVX units without SMT starvation. */
                    { const gdn_vtcm_t *cvt = jb->vt;
                      hmx_conv_out_desc_t od __attribute__((aligned(64))) = { cvt->outtab, GDN_BR_OUT_TABLE_STRIDE,
                          GDN_BR_OUT_Y_STRIDE, GDN_BR_N_TILES_POW2, GDN_BR_M_TOTAL_MINUS_STEP, GDN_BR_K_TOTAL_BYTES };
                      hmx_conv_act_desc_t ad __attribute__((aligned(64))) = { cvt->acttab, GDN_BR_N_ACT_PAIRS, GDN_BR_ACT_Y_STRIDE };
                      our_v73deep_kernel(&od, &ad, (const uint8_t *)jb->wt, (const uint8_t *)cvt->bias,
                                         (const hmx_conv_mask_desc_t *)pipe_mb, pipe_ep); }
#else
                    gdn_hmx_run_only(jb->vt, jb->wt, jb->eff, jb->scale, jb->baseline, jb->round);
#endif
                    uint64_t k1 = pcyc(); g_pipe_cbusy += k1 - k0;
#if defined(GDN_BR_TRACE)
                    gdn_tr_push((uint32_t)GDN_BR_NT, GDN_TR_MM, k0, k1);   /* consumer tid=GDN_BR_NT, MM span */
#endif
                    __sync_synchronize();
                    jb->state = 2;
                }
            }
        }
        uint64_t t1 = pcyc();
        uint64_t us1 = HAP_perf_get_time_us();
        for (int t = 0; t < P; ++t) { int s; if (tid[t]) qurt_thread_join(tid[t], &s); }
        g_hmx_dispatch = nullptr;
        if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
        if (vctx2) HAP_compute_res_release(vctx2);
        { HAP_power_request_t off; memset(&off,0,sizeof(off)); off.type=HAP_power_set_HMX; off.hmx.power_up=FALSE; HAP_power_set(pctx,&off); }
        if (statsLen > 0) stats[0] = (int)(t1 - t0);
        if (statsLen > 1) stats[1] = P;
        if (statsLen > 2) stats[2] = H;
        uint64_t wall = t1 - t0;                              /* domain cycle = de-overlapped real time (makespan) */
        uint64_t hvx_cyc = 0;                                 /* total HVX busy = sum over producers of (life - spin) */
        for (int t = 0; t < P; ++t) hvx_cyc += g_pipe_plife[t] - g_pipe_pspin[t];
        uint64_t hmx_cyc = g_pipe_cbusy;                      /* total HMX busy = consumer mxmem cycles */
        uint64_t us = us1 - us0;
        if (statsLen > 3) stats[3] = (int)hmx_cyc;            /* 总 HMX cycle */
        if (statsLen > 4) stats[4] = (int)hvx_cyc;            /* 总 HVX cycle */
        if (statsLen > 5) stats[5] = (int)wall;               /* 总 domain cycle (= wall) */
        if (statsLen > 6) stats[6] = (int)us;                 /* 总时间 us (real, HAP_perf) */
        FARF(ALWAYS, "PIPE 4col: HMX=%llu HVX=%llu DOMAIN=%llu cyc | %llu us | PCYCLE=%llu MHz",
             (unsigned long long)hmx_cyc, (unsigned long long)hvx_cyc, (unsigned long long)wall,
             (unsigned long long)us, (unsigned long long)(us ? wall / us : 0));
#if defined(GDN_BR_TRACE)
        /* serialize per-thread event timeline into Tu (overwrites T): [magic][n][wall u64][base u64] then
         * n*{tid u32, stage u32, t0 u64, t1 u64}.  Host renders ASCII timeline (scripts/gdn_pipe_timeline.py). */
        { int n = g_tr_n; if (n > GDN_TR_MAX) n = GDN_TR_MAX;
          uint32_t *hdr = (uint32_t *)Tu; hdr[0] = 0x47545203u; hdr[1] = (uint32_t)n;
          ((uint64_t *)(hdr + 2))[0] = wall; ((uint64_t *)(hdr + 2))[1] = t0;
          uint8_t *p = (uint8_t *)Tu + 24;
          for (int e = 0; e < n; ++e) { uint32_t *q = (uint32_t *)(p + (size_t)e * 24);
            q[0] = g_tr[e].tid; q[1] = g_tr[e].stage;
            ((uint64_t *)(q + 2))[0] = g_tr[e].t0; ((uint64_t *)(q + 2))[1] = g_tr[e].t1; }
          FARF(ALWAYS, "PIPE TRACE: %d events, wall=%llu", n, (unsigned long long)wall); }
#endif
        return 0;
    }
#endif

#if defined(GDNBM_OVERLAP_PROBE)
    {   /* 3-phase HVX∥HMX overlap probe. iters tuned so HVX-solo ≈ HMX-solo (H*~16 diag-solves). */
        struct ov_arg ha = { 0, (uint32_t)H, Au, Tu, zpA, M, S, sT, zpT, 0 };
        struct ov_arg va = { 1, (uint32_t)H, Au, Tu, zpA, M, S, sT, zpT, H * 16 };
        ov_join(ov_spawn(ov_hmx, &ha, 0));            uint64_t hmx_solo = g_ov_cyc[0];
        ov_join(ov_spawn(ov_hvx, &va, 1));            uint64_t hvx_solo = g_ov_cyc[1];
        uint64_t w0 = pcyc();
        qurt_thread_t a = ov_spawn(ov_hmx, &ha, 0), b = ov_spawn(ov_hvx, &va, 1);
        ov_join(a); ov_join(b);
        uint64_t wall_both = pcyc() - w0;
        { HAP_power_request_t off; memset(&off,0,sizeof(off)); off.type=HAP_power_set_HMX; off.hmx.power_up=FALSE; HAP_power_set(pctx,&off); }
        if (statsLen > 0) stats[0] = (int)hmx_solo;
        if (statsLen > 1) stats[1] = (int)hvx_solo;
        if (statsLen > 2) stats[2] = (int)g_ov_cyc[0];   /* HMX cycles when concurrent */
        if (statsLen > 3) stats[3] = (int)g_ov_cyc[1];   /* HVX cycles when concurrent */
        if (statsLen > 4) stats[4] = (int)wall_both;
        FARF(ALWAYS, "OVERLAP: hmx_solo=%llu hvx_solo=%llu hmx_both=%llu hvx_both=%llu WALL_both=%llu",
             (unsigned long long)hmx_solo, (unsigned long long)hvx_solo,
             (unsigned long long)g_ov_cyc[0], (unsigned long long)g_ov_cyc[1], (unsigned long long)wall_both);
        return 0;
    }
#endif

    /* VTCM is acquired ONCE here on the main thread and the base pointer shared with all workers (each
     * takes a 0x60000 slice via w->vtcm_base).  Per-worker acquire serializes the workers — the resource
     * manager grants VTCM per-context so concurrent acquires block on each other (canonical pattern:
     * Hexagon SDK HAP_compute_res + tutorial ch04 demo_vtcm_alloc = acquire once, bump-allocate). */
    uint8_t *vtcm_base = nullptr; unsigned int vctx = 0;
#if defined(GDNBM_VTCM_RESIDENT)
    { compute_res_attr_t va; HAP_compute_res_attr_init(&va);
#if defined(GDN_BR_T_DDR_DIRECT)
      HAP_compute_res_attr_set_vtcm_param(&va, (unsigned)GDN_BR_NT * 0x60000u, 0);  /* 384KB/worker: A ping-pong (legacy DDR-direct T) */
#else
      HAP_compute_res_attr_set_vtcm_param(&va, (unsigned)GDN_BR_NT * 0x80000u, 0);  /* 512KB/worker: A ping-pong + T(VTCM)+DMA writeback (default) */
#endif
      vctx = HAP_compute_res_acquire(&va, 2000000);
      vtcm_base = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&va); }
#endif
#if defined(GDNBM_HMX_BENCH)
    /* STAGE 1 de-risk: continuous HMX 64^3 throughput on the MAIN thread (HMX faults on workers).
     * pack ONCE (resident), then loop run_only [+ depack] J times — does per-call bias/zero setup
     * kill the 237-cyc kernel? Build: -DGDNBM_VTCM_RESIDENT -DGDNBM_HMX_BENCH. */
    if (vtcm_base) {
        int bhvx = qurt_hvx_lock(QURT_HVX_MODE_128B);          /* main thread must lock HVX for pack/kernel/depack */
        compute_res_attr_t bha; HAP_compute_res_attr_init(&bha); HAP_compute_res_attr_set_hmx_param(&bha, 1);
        unsigned int bhctx = HAP_compute_res_acquire(&bha, 2000000); int bhl = HAP_compute_res_hmx_lock(bhctx);  /* mxmem needs HMX lock */
        gdn_vtcm_t vt = gdn_vtcm_from(vtcm_base);
        static uint8_t __attribute__((aligned(128))) bact[64*64];
        static int8_t  __attribute__((aligned(128))) bwt[64*64];
        static int8_t  __attribute__((aligned(128))) boutc[64*64];
        static int8_t  __attribute__((aligned(128))) boutc2[64*64];   /* fused-depack output for correctness compare */
        static int32_t beff[64];
        for (int i = 0; i < 64*64; ++i) { bact[i] = (uint8_t)((i % 200) + 28); bwt[i] = (int8_t)((i % 11) - 5); }
        const float scale_f16 = 1.0f / 64.0f; const int baseline_u16 = 128 << 7;
        gdn_hmx_pack_only(&vt, bact, bwt, beff);                 /* pack act+wt ONCE -> resident in VTCM */
        const int J = 512;
        gdn_hmx_run_only(&vt, vt.wt, beff, scale_f16, baseline_u16, 0);   /* warm */
        uint64_t a0 = pcyc();
        for (int j = 0; j < J; ++j) gdn_hmx_run_only(&vt, vt.wt, beff, scale_f16, baseline_u16, 0);
        uint64_t a1 = pcyc();
        gdn_scr_t *bsc = &g_scr[0];
        for (int j = 0; j < J; ++j) { gdn_hmx_run_only(&vt, vt.wt, beff, scale_f16, baseline_u16, 0);
                                      gdn_depack_out_fast(bsc, vt.out, baseline_u16 >> 7, boutc); }
        uint64_t a2 = pcyc();
        /* pure continuous kernel rate: bias + descriptors set up ONCE, loop only the mxmem kernel */
        gdn_pack_bias(beff, scale_f16, baseline_u16, vt.bias, 0);
        uint32_t ep[2] __attribute__((aligned(16))) = {1u, 0u};
        uint32_t mb[16] __attribute__((aligned(16))); for (int i = 0; i < 16; ++i) mb[i] = GDN_BR_MASK_WORDS[i];
        hmx_conv_out_desc_t od __attribute__((aligned(64))) = { vt.outtab, GDN_BR_OUT_TABLE_STRIDE, GDN_BR_OUT_Y_STRIDE,
            GDN_BR_N_TILES_POW2, GDN_BR_M_TOTAL_MINUS_STEP, GDN_BR_K_TOTAL_BYTES };
        hmx_conv_act_desc_t ad __attribute__((aligned(64))) = { vt.acttab, GDN_BR_N_ACT_PAIRS, GDN_BR_ACT_Y_STRIDE };
        uint64_t a3 = pcyc();
        for (int j = 0; j < J; ++j) our_v73deep_kernel(&od, &ad, (const uint8_t *)vt.wt, (const uint8_t *)vt.bias,
                                                        (const hmx_conv_mask_desc_t *)mb, ep);
        uint64_t a4 = pcyc();
        /* the PIPELINE CONSUMER's exact work, clean wall-based: zero + kernel + depack (NO bias; offloaded) */
        gdn_scr_t *zsc = &g_scr[0];
        uint64_t a5 = pcyc();
        for (int j = 0; j < J; ++j) {
            { HVX_Vector z = Q6_V_vzero(); HVX_Vector *op = (HVX_Vector *)vt.out; for (int i = 0; i < (64*64)/128; ++i) op[i] = z; }
            our_v73deep_kernel(&od, &ad, (const uint8_t *)vt.wt, (const uint8_t *)vt.bias, (const hmx_conv_mask_desc_t *)mb, ep);
            gdn_depack_out_fast(zsc, vt.out, baseline_u16 >> 7, boutc);
        }
        uint64_t a6 = pcyc();
        /* ISOLATED depack component: old two-pass (gdn_depack_out_fast) vs new fused one-pass (fp_depack) on
         * the SAME surface, plus a byte-exact correctness compare (must be 0 mismatches). */
        gdn_scr_t *dsc = &g_scr[0];
        gdn_hmx_run_only(&vt, vt.wt, beff, scale_f16, baseline_u16, 0);   /* fresh HMX surface to depack */
        uint64_t d0 = pcyc();
        for (int j = 0; j < J; ++j) gdn_depack_out_fast(dsc, vt.out, baseline_u16 >> 7, boutc);
        uint64_t d1 = pcyc();
        for (int j = 0; j < J; ++j) fp_depack(vt.out, baseline_u16 >> 7, boutc2);
        uint64_t d2 = pcyc();
        int dep_mism = 0; for (int i = 0; i < 64*64; ++i) if (boutc[i] != boutc2[i]) ++dep_mism;
        /* LEVER 2 correctness: does our_v73deep_kernel OVERWRITE the out surface (so the producer pre-zero
         * is unnecessary)? Run the SAME inputs twice over DIFFERENT pre-fill values; if depacked outputs
         * match, the kernel overwrites and the zero can be dropped. (descriptors od/ad/bias set above.) */
        { HVX_Vector zf = Q6_Vb_vsplat_R(0);  HVX_Vector *op=(HVX_Vector*)vt.out; for(int i=0;i<(64*64)/128;++i) op[i]=zf; }
        our_v73deep_kernel(&od, &ad, (const uint8_t *)vt.wt, (const uint8_t *)vt.bias, (const hmx_conv_mask_desc_t *)mb, ep);
        fp_depack(vt.out, baseline_u16 >> 7, boutc);
        { HVX_Vector ff = Q6_Vb_vsplat_R(-1); HVX_Vector *op=(HVX_Vector*)vt.out; for(int i=0;i<(64*64)/128;++i) op[i]=ff; }
        our_v73deep_kernel(&od, &ad, (const uint8_t *)vt.wt, (const uint8_t *)vt.bias, (const hmx_conv_mask_desc_t *)mb, ep);
        fp_depack(vt.out, baseline_u16 >> 7, boutc2);
        int ovr_mism = 0; for (int i = 0; i < 64*64; ++i) if (boutc[i] != boutc2[i]) ++ovr_mism;
        /* LEVER 3 correctness: fp_pack_effbias2 (2-head interleaved) must == two independent fp_pack_effbias.
         * Distinct A/B inputs (bwt vs bact-as-i8) catch any cross-wiring of the two accumulator chains. */
        { uint32_t bc = ((uint32_t)((128 << 7) & 0xFFFF) << 16) | (uint32_t)gdn_f16_bits(scale_f16);
          static int32_t e_rA[128], e_rB[128], e_a[128], e_b[128];
          fp_pack_effbias(bwt, bc, 0, e_rA);
          fp_pack_effbias((int8_t *)bact, bc, 0, e_rB);
          fp_pack_effbias2(bwt, (int8_t *)bact, bc, 0, e_a, e_b);
          int em = 0; for (int i = 0; i < 128; ++i) if (e_a[i] != e_rA[i] || e_b[i] != e_rB[i]) ++em;
          ovr_mism += em; }   /* fold into stats[9]: 0 => both lever-2 overwrite AND lever-3 effbias2 OK */
        if (statsLen > 0) stats[0] = (int)((a1 - a0) / J);      /* HMX run_only cyc/matmul (kernel+bias+zero) */
        if (statsLen > 1) stats[1] = (int)((a2 - a1) / J);      /* run_only + depack cyc/matmul */
        if (statsLen > 2) stats[2] = (int)((a4 - a3) / J);      /* PURE kernel cyc/matmul (desc+bias once) */
        /* PRODUCER pack floor (1-head, tight loop): eff + act-crouton + wt-kmajor + bias */
        uint32_t bctrl = ((uint32_t)((128 << 7) & 0xFFFF) << 16) | (uint32_t)gdn_f16_bits(scale_f16);
        uint64_t a7 = pcyc();
        for (int j = 0; j < J; ++j) {                            /* OLD: scalar eff+bias */
            gdn_effective(bwt, beff);
            gdn_pack_act_crouton8(bact, vt.act);
            gdn_pack_w8_kmajor(bwt, (int8_t *)vt.wt);
            gdn_pack_bias(beff, scale_f16, baseline_u16, vt.bias, 0);
        }
        uint64_t a8 = pcyc();
        for (int j = 0; j < J; ++j) {                            /* NEW: vectorized eff+bias */
            fp_pack_effbias(bwt, bctrl, 0, vt.bias);
            gdn_pack_act_crouton8(bact, vt.act);
            gdn_pack_w8_kmajor(bwt, (int8_t *)vt.wt);
        }
        uint64_t a9 = pcyc();
        if (statsLen > 3) stats[3] = (int)((a8 - a7) / J);      /* PRODUCER PACK floor, OLD scalar eff+bias */
        if (statsLen > 4) stats[4] = (int)((a9 - a8) / J);      /* PRODUCER PACK floor, NEW vec eff+bias */
        if (statsLen > 5) stats[5] = (int)((a6 - a5) / J);      /* CONSUMER CEILING: zero+kernel+depack /matmul */
        if (statsLen > 6) stats[6] = (int)((d1 - d0) / J);      /* DEPACK old two-pass (isolated) cyc/matmul */
        if (statsLen > 7) stats[7] = (int)((d2 - d1) / J);      /* DEPACK new fused fp_depack (isolated) cyc/matmul */
        if (statsLen > 8) stats[8] = dep_mism;                  /* fp_depack vs old depack mismatch bytes (==0 expected) */
        if (statsLen > 9) stats[9] = ovr_mism;                  /* 0 => kernel overwrites (pre-zero unneeded) AND effbias2 OK */
        FARF(ALWAYS, "HMX_BENCH: run_only=%d run+depack=%d ceil=%d depack old=%d new=%d depmism=%d ovrmism=%d (J=%d)",
             (int)((a1-a0)/J), (int)((a2-a1)/J), (int)((a6-a5)/J), (int)((d1-d0)/J), (int)((d2-d1)/J), dep_mism, ovr_mism, J);
        if (bhl == 0) HAP_compute_res_hmx_unlock(bhctx); if (bhctx) HAP_compute_res_release(bhctx);
        if (bhvx == 0) qurt_hvx_unlock();
        { HAP_power_request_t off; memset(&off,0,sizeof(off)); off.type=HAP_power_set_HMX; off.hmx.power_up=FALSE; HAP_power_set(pctx,&off); }
        if (vctx) HAP_compute_res_release(vctx);
        return 0;
    }
#endif
#if defined(GDNBM_FEED_PIPE)
    if (vtcm_base) {
        g_fp_base = vtcm_base;
        g_fp_done = 0;
        for (int k = 0; k < FP_K; ++k) g_fp_ready[k] = 0;
        for (int i = 0; i < 64*64; ++i) { g_fp_act[i] = (uint8_t)((i % 200) + 28); g_fp_wt[i] = (int8_t)((i % 11) - 5);
                                          g_fp_act2[i] = (uint8_t)((i*3 % 200) + 28); g_fp_wt2[i] = (int8_t)((i*5 % 11) - 5); }
        int P = nthreads; if (P > GDN_BR_NT) P = GDN_BR_NT; if (P < 1) P = 1;   /* P PRODUCER threads + main consumer */
        g_fp_T = P;
        /* MAIN thread is the HMX consumer. In 4P mode it's PURE-HMX (no qurt_hvx_lock → frees its HVX
         * unit for a 4th producer); zero+depack moved to producers. Else it owns HVX+HMX and drains. */
#if defined(GDNBM_FEED_4P) && !defined(GDNBM_FEED_MULTIPASS)
        int chvx = -1;   /* NOT locking HVX — the kernel needs only hmx_lock */
#else
        int chvx = qurt_hvx_lock(QURT_HVX_MODE_128B);   /* MULTIPASS consumer does maxabs (HVX) -> must lock; use P=3 */
#endif
        compute_res_attr_t cha; HAP_compute_res_attr_init(&cha); HAP_compute_res_attr_set_hmx_param(&cha, 1);
        unsigned int chctx = HAP_compute_res_acquire(&cha, 2000000); int chl = HAP_compute_res_hmx_lock(chctx);
        gdn_scr_t *csc = &g_scr[GDN_BR_NT - 1]; int8_t *coutc = g_fp_outc[GDN_BR_NT - 1];
        uint32_t ep[2] __attribute__((aligned(16))) = {1u, 0u};
        uint32_t mb[16] __attribute__((aligned(16))); for (int i = 0; i < 16; ++i) mb[i] = GDN_BR_MASK_WORDS[i];
        qurt_thread_t pt[GDN_BR_NT];
        uint64_t w0 = pcyc();
        for (int t = 0; t < P; ++t) {
            qurt_thread_attr_t a; qurt_thread_attr_init(&a); qurt_thread_attr_set_name(&a, (char *)"fpprod");
            qurt_thread_attr_set_stack_addr(&a, g_fp_stack[t]); qurt_thread_attr_set_stack_size(&a, sizeof(g_fp_stack[t]));
#if defined(GDNBM_PROD_PRIO)
            qurt_thread_attr_set_priority(&a, GDNBM_PROD_PRIO);  /* lower producer prio -> consumer wins SMT issue slots */
#endif
            if (qurt_thread_create(&pt[t], &a, feed_producer, (void *)(intptr_t)t) != QURT_EOK) pt[t] = 0;
        }
        uint64_t c_spin = 0, c_zero = 0, c_kern = 0, c_dep = 0;   /* consumer breakdown */
        (void)csc; (void)coutc;
#if defined(GDNBM_CONS_PRIO)
        /* SMT issue-arbitration lever: raise the HMX consumer (this main thread) above the HVX producers so
         * it wins issue slots when co-running. No cluster-pin API on v75; priority biases the SMT arbiter. */
        qurt_thread_set_priority(qurt_thread_get_self(), GDNBM_CONS_PRIO);
#endif
        { qurt_sysenv_max_hthreads_t mh; if (qurt_sysenv_get_max_hw_threads(&mh) == 0 && statsLen > 9) stats[9] = (int)mh.max_hthreads; }
#if defined(GDNBM_PMU)
        /* DIRECT SMT-contention proof via the v75 PMU (raw event codes, Table 9-1). PMUEVTCFG packs 4
         * 8-bit event selectors for PMUCNT0..3. Deltas over the consumer window (all P producers live):
         *   cnt0 COMMITTED_PKT_ANY (0x03), cnt1 CYCLES_2_THREAD_RUNNING (0x3c),
         *   cnt2 CYCLES_5_THREAD_RUNNING (0x0a), cnt3 SMT_BANK_CONFLICT (0xb9 inter-thread SMT bank conflict). */
        qurt_pmu_set(QURT_PMUEVTCFG, (0xb9u<<24)|(0x0au<<16)|(0x03u<<8)|0x3cu);  /* cnt0 CYC2T, cnt1 COMMITTED_PKT_ANY, cnt2 CYC5T, cnt3 SMT_BANK */
        qurt_pmu_enable(1);
        unsigned int pmu_base[4] = {
            qurt_pmu_get(QURT_PMUCNT0), qurt_pmu_get(QURT_PMUCNT1),
            qurt_pmu_get(QURT_PMUCNT2), qurt_pmu_get(QURT_PMUCNT3) };
#endif
        for (uint32_t j = 0; j < FP_J; ++j) {                /* main consumer: drain in job order */
            int k = j % FP_K;
            uint64_t s0 = pcyc();
            while (g_fp_ready[k] != 1) { /* wait for a producer */ }
            __sync_synchronize();
            uint64_t s1 = pcyc();
            gdn_vtcm_t vt = fp_slot(k);
            hmx_conv_out_desc_t od __attribute__((aligned(64))) = { vt.outtab, GDN_BR_OUT_TABLE_STRIDE, GDN_BR_OUT_Y_STRIDE,
                GDN_BR_N_TILES_POW2, GDN_BR_M_TOTAL_MINUS_STEP, GDN_BR_K_TOTAL_BYTES };
            hmx_conv_act_desc_t ad __attribute__((aligned(64))) = { vt.acttab, GDN_BR_N_ACT_PAIRS, GDN_BR_ACT_Y_STRIDE };
#if defined(GDNBM_FEED_MULTIPASS)
            /* FAITHFUL GDNSolveHVXMixHMX cost: the real gdn_merge_packed runs 3 HMX passes per logical matmul
             * (PASS1/2 = dynamic-quant gain search via maxabs, output discarded; PASS3 = real output). Serial
             * dependency (PASS3's gain needs PASS1/2's maxabs). dummy values, REAL cycle count. depack still
             * offloaded to a producer (state 2). This measures what the single-run 578 microbench under-counts. */
            {
                uint32_t bc = fp_bias_ctrl();
                fp_pack_effbias(g_fp_wt, bc, 0, vt.bias);                                /* PASS 1 (vec bias) */
                our_v73deep_kernel(&od, &ad, (const uint8_t *)vt.wt, (const uint8_t *)vt.bias, (const hmx_conv_mask_desc_t *)mb, ep);
                (void)gdn_surf_maxabs(vt.out, 128);
#if !defined(GDNBM_FEED_MP_2PASS)
                fp_pack_effbias(g_fp_wt, bc, 0, vt.bias);                                /* PASS 2 (refine) */
                our_v73deep_kernel(&od, &ad, (const uint8_t *)vt.wt, (const uint8_t *)vt.bias, (const hmx_conv_mask_desc_t *)mb, ep);
                (void)gdn_surf_maxabs(vt.out, 128);
#endif
                fp_pack_effbias(g_fp_wt, bc, 0, vt.bias);                                /* PASS 3 (output) */
                our_v73deep_kernel(&od, &ad, (const uint8_t *)vt.wt, (const uint8_t *)vt.bias, (const hmx_conv_mask_desc_t *)mb, ep);
            }
            __sync_synchronize();
            g_fp_ready[k] = 2;
            c_kern += pcyc() - s1;
#elif defined(GDNBM_FEED_4P)
            our_v73deep_kernel(&od, &ad, (const uint8_t *)vt.wt, (const uint8_t *)vt.bias, (const hmx_conv_mask_desc_t *)mb, ep);
            __sync_synchronize();
            g_fp_ready[k] = 2;                               /* HMX done; a producer will depack+reuse */
            c_kern += pcyc() - s1;
#else
            { HVX_Vector z = Q6_V_vzero(); HVX_Vector *op = (HVX_Vector *)vt.out; for (int i = 0; i < (64*64)/128; ++i) op[i] = z; }
            uint64_t s2 = pcyc();
            our_v73deep_kernel(&od, &ad, (const uint8_t *)vt.wt, (const uint8_t *)vt.bias, (const hmx_conv_mask_desc_t *)mb, ep);
            uint64_t s3 = pcyc();
            gdn_depack_out_fast(csc, vt.out, (128 << 7) >> 7, coutc);
            uint64_t s4 = pcyc();
            __sync_synchronize();
            g_fp_ready[k] = 0;
            c_zero += s2 - s1; c_kern += s3 - s2; c_dep += s4 - s3;
#endif
            c_spin += s1 - s0;
        }
        uint64_t w1 = pcyc();
        g_fp_done = 1; __sync_synchronize();                 /* OPCACHE: release producers from their re-arm loop */
#if defined(GDNBM_PMU)
        unsigned int pmu_fin[4] = {
            qurt_pmu_get(QURT_PMUCNT0), qurt_pmu_get(QURT_PMUCNT1),
            qurt_pmu_get(QURT_PMUCNT2), qurt_pmu_get(QURT_PMUCNT3) };
        if (statsLen > 5) stats[5] = (int)((pmu_fin[0] - pmu_base[0]) / FP_J);  /* COMMITTED_PKT_ANY /mm */
        if (statsLen > 6) stats[6] = (int)((pmu_fin[1] - pmu_base[1]) / FP_J);  /* CYCLES_2_THREAD_RUNNING /mm */
        if (statsLen > 7) stats[7] = (int)((pmu_fin[2] - pmu_base[2]) / FP_J);  /* CYCLES_5_THREAD_RUNNING /mm */
        if (statsLen > 8) stats[8] = (int)((pmu_fin[3] - pmu_base[3]) / FP_J);  /* SMT_BANK_CONFLICT /mm */
#endif
        uint32_t per = FP_J / (uint32_t)P;                   /* matmuls packed by producer 0 */
        if (statsLen > 2 && per) stats[2] = (int)(g_fp_pwork[0] / per);  /* producer0: pack-work / matmul */
        if (statsLen > 3 && per) stats[3] = (int)(g_fp_pspin[0] / per);  /* producer0: slot-wait / matmul */
        if (statsLen > 4) stats[4] = (int)(c_spin / FP_J);   /* consumer: spin-wait */
        (void)c_zero; (void)c_kern; (void)c_dep;
        for (int t = 0; t < P; ++t) { int s; if (pt[t]) qurt_thread_join(pt[t], &s); }
        if (statsLen > 0) stats[0] = (int)((w1 - w0) / FP_J);   /* pipeline cyc/matmul */
        if (statsLen > 1) stats[1] = P;
        FARF(ALWAYS, "FEED_PIPE: %d producers + main HMX consumer -> %d cyc/matmul (J=%d)", P, (int)((w1-w0)/FP_J), FP_J);
        if (chl == 0) HAP_compute_res_hmx_unlock(chctx); if (chctx) HAP_compute_res_release(chctx);
        if (chvx == 0) qurt_hvx_unlock();
        { HAP_power_request_t off; memset(&off,0,sizeof(off)); off.type=HAP_power_set_HMX; off.hmx.power_up=FALSE; HAP_power_set(pctx,&off); }
        if (vctx) HAP_compute_res_release(vctx);
        return 0;
    }
#endif
    uint64_t t0 = pcyc();
#if defined(GDN_BR_TRACE)
    g_tr_n = 0; g_tr_base = t0;   /* all event timestamps become relative to spawn time */
#endif
    {
        gdn_work_t work[GDN_BR_NT]; qurt_thread_t tid[GDN_BR_NT];
        for (int t = 0; t < nthreads; ++t) {
            work[t] = (gdn_work_t){ t, 0u, (uint32_t)H, (uint32_t)nthreads, Au, Tu, zpA, M, S, sT, zpT, vtcm_base };
            qurt_thread_attr_t a; qurt_thread_attr_init(&a); qurt_thread_attr_set_name(&a,(char*)"gdnsolve");
            qurt_thread_attr_set_stack_addr(&a,g_solve_stack[t]); qurt_thread_attr_set_stack_size(&a,sizeof(g_solve_stack[t]));
            if (qurt_thread_create(&tid[t],&a,solve_worker,&work[t])!=QURT_EOK) { solve_worker(&work[t]); tid[t]=0; }
        }
        for (int t = 0; t < nthreads; ++t) { int s; if (tid[t]) qurt_thread_join(tid[t], &s); }
    }
    uint64_t t1 = pcyc();
#if defined(GDN_BR_TRACE)
    /* serialize the event buffer into the output T (overwrites T — trace runs don't validate T):
     * [magic u32][n u32][total_wall u64] then n*{tid u32, stage u32, t0 u64, t1 u64} (24B each). */
    {
        int n = g_tr_n; if (n > GDN_TR_MAX) n = GDN_TR_MAX;
        uint32_t *hdr = (uint32_t *)Tu;
        hdr[0] = 0x47545203u; hdr[1] = (uint32_t)n;
        ((uint64_t *)(hdr + 2))[0] = t1 - t0;   /* total wall (cycles) */
        ((uint64_t *)(hdr + 2))[1] = t0;        /* absolute base PCYCLE (for "Start Cycle") */
        uint8_t *p = (uint8_t *)Tu + 24;
        for (int e = 0; e < n; ++e) {
            uint32_t *q = (uint32_t *)(p + (size_t)e * 24);
            q[0] = g_tr[e].tid; q[1] = g_tr[e].stage;
            ((uint64_t *)(q + 2))[0] = g_tr[e].t0; ((uint64_t *)(q + 2))[1] = g_tr[e].t1;
        }
        FARF(ALWAYS, "TRACE: %d events, wall=%llu", n, (unsigned long long)(t1 - t0));
    }
#endif
    if (vctx) HAP_compute_res_release(vctx);
    { HAP_power_request_t off; memset(&off,0,sizeof(off)); off.type=HAP_power_set_HMX; off.hmx.power_up=FALSE; HAP_power_set(pctx,&off); }

    if (statsLen > 0) stats[0] = (int)(t1 - t0);
    if (statsLen > 1) stats[1] = nthreads;
    if (statsLen > 2) stats[2] = H;
    FARF(ALWAYS, "gdnbm_solve: wall=%llu cyc / %d heads / %d threads", (unsigned long long)(t1-t0), H, nthreads);
#if defined(GDN_BR_PROBE_CYCLES)
    /* per-stage cycle share (accumulated over all heads on this thread; run nthreads=1 for a clean,
     * race-free per-stage breakdown of the 1-thread base). GDNSolveHVX path stages:
     *   diag (fwd-subst) | zero (Th init) | fold (A_ik) | quant (operand quant in merge) |
     *   mm (int16 matmul) | acc (term accumulate) | requant (codes->u16 out). */
    {
        /* per-stage cyc/head. Buckets work for BOTH paths: GDNSolveHVX (pint/depack/actpack/wtpack/eff=0,
         * mergeRun=vrmpy mm) AND GDNSolveHVXMixHMX (mergeRun=all HMX runs incl probe passes, mergeGlue=HMX
         * pack/depack). Lets us see merge vs diag vs prep(fold+quant) share for either implementation. */
        uint64_t mergeRun  = g_c_hmxkern + g_c_pint;                           /* HMX runs (or vrmpy mm) */
        uint64_t mergeGlue = g_c_hmxdepack + g_c_actpack + g_c_wtpack + g_c_eff; /* HMX pack/depack (0 for HVX) */
        uint64_t other     = g_c_requant + g_c_acc + g_c_zero + g_c_widen;
        uint64_t tot = g_c_diag + mergeRun + mergeGlue + g_c_fold + g_c_quant + other;
        FARF(ALWAYS, "PROBE/head(H=%d,nt=%d): diag=%llu mergeRun=%llu mergeGlue=%llu fold=%llu quant=%llu other=%llu SUM=%llu wall=%llu",
             H, nthreads,
             (unsigned long long)(g_c_diag / H), (unsigned long long)(mergeRun / H),
             (unsigned long long)(mergeGlue / H), (unsigned long long)(g_c_fold / H),
             (unsigned long long)(g_c_quant / H), (unsigned long long)(other / H),
             (unsigned long long)(tot / H), (unsigned long long)((t1 - t0) / H));
        /* FINE per-glue-op breakdown (per head) so the dominant glue is visible, not lumped into "other". */
        if (statsLen > 3) stats[3] = (int)(g_c_diag / H);                                   /* diag fwd-subst */
        if (statsLen > 4) stats[4] = (int)(mergeRun / H);                                   /* HMX kernel (run_only) */
        if (statsLen > 5) stats[5] = (int)(g_c_hmxdepack / H);                              /* depack out u8->codes */
        if (statsLen > 6) stats[6] = (int)(g_c_fold / H);                                   /* fold A_ik */
        if (statsLen > 7) stats[7] = (int)(g_c_quant / H);                                  /* operand quant */
        if (statsLen > 8) stats[8] = (int)(g_c_acc / H);                                    /* acc terms */
        if (statsLen > 9) stats[9] = (int)(g_c_requant / H);                                /* requant codes->u16 out */
        if (statsLen > 10) stats[10] = (int)(g_c_widen / H);                               /* widen i8->i32 */
        if (statsLen > 11) stats[11] = (int)((g_c_actpack + g_c_wtpack + g_c_eff) / H);     /* act/wt pack + eff */
        (void)mergeGlue; (void)other;
    }
#endif
    return 0;
}
