/* pure_hmx_solve.cpp — SELF-CONTAINED pure-HMX (all-w16a16) GDN triangular-inverse SCHEDULE bench.
 *
 * Purpose: measure the per-thread TIMELINE + 32-head TOTAL wall of the "pure HMX" route, where the
 * diagonal 64-block inverse is done by Taylor(p=3)+Newton(K=4) = 11 w16a16 64^3 matmuls/block (44/head)
 * instead of HVX forward-subst, plus 16 off-diag merge matmuls = 60 w16a16 64^3 matmuls/head, ALL on the
 * single HMX consumer with P HVX producers feeding it.
 *
 * Numerics are KNOWN-BROKEN (the int16 matrix-power inverse overflows on ~15% of high-||A||2 diagonal
 * blocks — proven separately in scripts/gdn_solve_e2e_precision_probe.py). This bench measures SPEED ONLY:
 * the matmul work is data-independent, so garbage operands give the true cycle/timeline.
 *
 * Kept OUT of the shipping solve (GdnSolveBR16.cpp / gdnbm_imp.cpp) on purpose — this is an isolated
 * experiment. It reuses ONLY the byte-proven w16a16 kernel header (sha256-identical to the device-exact
 * custom op) and the validated 64^3 descriptors/mask (run_w16a16_standalone_device.py --shape 64,64,64 =
 * 8259 cyc/64^3). It is #included into the gdnbm skel TU under -DGDNBM_PURE_HMX_SOLVE and dispatched first.
 *
 * Build:  EXTRA_DEFS="-DGDNBM_PURE_HMX_SOLVE" bash example/gdn_native/baremetal/build.sh
 * Run:    scripts/run_pure_hmx_solve.py  (compile + device + render timeline)
 */
#include <stdint.h>
#include <string.h>
#include "qurt.h"
#include "HAP_compute_res.h"
#include "HAP_farf.h"
#include "HAP_power.h"
#include "HAP_perf.h"
#include "../baremetal/inc/v73deep_conv1x1_kernel_i16.h"
#include "../pure_hmx_solve/w16a16_mm.h"

namespace pure_hmx {

static inline uint64_t phs_pcyc(void) { uint64_t v; asm volatile("%0 = C15:14" : "=r"(v)); return v; }

/* ---- Phase-1 mm-test (H==1): REAL w16a16 M=256-carrier primitive on REAL data ----
 * payload in A: [0..32K) act u16 codes 256x64 (zp 32768)  [32K..40K) weight q16 int16 64x64 (±32639)
 * out  in T:  [0..32K) Y u16 codes 256x64, [32K..64K) raw crouton out surface (debug).
 * stats: [0]=wall [3]=kernel cyc [6]=us. */
static int mm_test(const uint8_t *A, int *stats, int statsLen, void *T, int TLen) {
    if (TLen < 0x8000) return -2;
    static int pwr_client; void *pctx = &pwr_client;
    HAP_power_set_core_corner(pctx, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_MAX);
    { HAP_power_request_t r; memset(&r, 0, sizeof(r)); r.type = HAP_power_set_HMX; r.hmx.power_up = TRUE; HAP_power_set(pctx, &r); }
    compute_res_attr_t va; HAP_compute_res_attr_init(&va);
    HAP_compute_res_attr_set_vtcm_param(&va, 0x100000u, 0);
    unsigned int vctx = HAP_compute_res_acquire(&va, 2000000);
    uint8_t *vbase = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&va);
    compute_res_attr_t ha; HAP_compute_res_attr_init(&ha); HAP_compute_res_attr_set_hmx_param(&ha, 1);
    unsigned int hctx = HAP_compute_res_acquire(&ha, 2000000);
    int hl = HAP_compute_res_hmx_lock(hctx);
    if (!vbase || hl != 0) { FARF(ALWAYS, "MMTEST: acquire failed vbase=%p hl=%d", vbase, hl);
        if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
        if (vctx) HAP_compute_res_release(vctx); return -1; }

    static uint8_t descs[256] __attribute__((aligned(64)));
    w16a16_mm_t mm; w16a16_mm_init(&mm, vbase, descs);
    memset(mm.out, 0, W16MM_OUT_BYTES);
    const uint16_t *Au = (const uint16_t *)A;
    const int16_t  *Wq = (const int16_t *)(A + W16MM_ACT_BYTES);
    uint16_t *Y = (uint16_t *)T;
    uint64_t us0 = HAP_perf_get_time_us(), t0 = phs_pcyc();
    w16a16_pack_act_crouton16(Au, (uint16_t *)mm.act, W16MM_M, W16MM_K);
    w16a16_pack_wt_kmajor(Wq, mm.wt, W16MM_K, W16MM_N);
    w16a16_pack_bias(Wq, mm.bias, W16MM_K, W16MM_N);
    uint64_t k0 = phs_pcyc();
    w16a16_mm_run(&mm);
    uint64_t k1 = phs_pcyc();
    w16a16_depack_crouton16((const uint16_t *)mm.out, Y, W16MM_M, W16MM_N);
    uint64_t t1 = phs_pcyc(), us1 = HAP_perf_get_time_us();
    if (TLen >= 2 * W16MM_OUT_BYTES) memcpy((uint8_t *)T + W16MM_OUT_BYTES, mm.out, W16MM_OUT_BYTES);

    if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
    if (vctx) HAP_compute_res_release(vctx);
    { HAP_power_request_t off; memset(&off, 0, sizeof(off)); off.type = HAP_power_set_HMX; off.hmx.power_up = FALSE; HAP_power_set(pctx, &off); }
    if (statsLen > 0) stats[0] = (int)(t1 - t0);
    if (statsLen > 3) stats[3] = (int)(k1 - k0);
    if (statsLen > 6) stats[6] = (int)(us1 - us0);
    FARF(ALWAYS, "MMTEST w16a16 64^3: wall=%llu kernel=%llu cyc %llu us",
         (unsigned long long)(t1 - t0), (unsigned long long)(k1 - k0), (unsigned long long)(us1 - us0));
    return 0;
}

#define PHS_NT       4          /* max producers */
#define PHS_DIAG_MM  11         /* Taylor(3)+Newton(4) matmuls per 64-diag-block */
#define PHS_NB       4          /* 4 diagonal 64-blocks per C=256 head */
#define PHS_MERGE_MM 16         /* off-diagonal merge matmuls per head */

/* ---- job hand-off (acquire/release on `state`, the documented fast path) ---- */
struct phs_job { volatile int state; int _pad[31]; } __attribute__((aligned(128)));
static phs_job g_job[PHS_NT];
#define PHS_ARM(j)   __atomic_store_n(&(j)->state, 1, __ATOMIC_RELEASE)
#define PHS_POLL(j)  (__atomic_load_n(&(j)->state, __ATOMIC_ACQUIRE) == 1)
#define PHS_WAIT(j)  do { while (__atomic_load_n(&(j)->state, __ATOMIC_ACQUIRE) != 2) {} } while (0)
#define PHS_DONE(j)  __atomic_store_n(&(j)->state, 2, __ATOMIC_RELEASE)
#define PHS_RESET(j) __atomic_store_n(&(j)->state, 0, __ATOMIC_RELAXED)

static volatile int g_pdone;            /* # producers finished all heads */
static volatile int g_next_head;        /* dynamic head scheduler (atomic pull) */
static int g_H, g_P;
static uint64_t g_cbusy;                /* consumer mxmem busy cyc */
static uint64_t g_plife[PHS_NT], g_pspin[PHS_NT];
static char __attribute__((aligned(128))) g_stack[PHS_NT][32768];
static uint8_t *g_packscratch[PHS_NT];  /* per-producer pack-proxy buffer (represents PREP glue) */

/* ---- per-thread event trace (same wire format scripts/gdn_pipe_timeline.py renders) ---- */
#define PHS_TR_MAX 200000
struct phs_ev { uint32_t tid, stage; uint64_t t0, t1; };
static phs_ev g_tr[PHS_TR_MAX];
static volatile int g_tr_n;
static uint64_t g_tr_base;
static inline void phs_tr(uint32_t tid, uint32_t stage, uint64_t t0, uint64_t t1) {
    int i = __atomic_fetch_add(&g_tr_n, 1, __ATOMIC_RELAXED);
    if (i < PHS_TR_MAX) { g_tr[i].tid = tid; g_tr[i].stage = stage; g_tr[i].t0 = t0 - g_tr_base; g_tr[i].t1 = t1 - g_tr_base; }
}
/* stages match the renderer: 1=DIAG 3=MM 5=PREP 11=SPIN */
#define PHS_S_DIAG 1u
#define PHS_S_MM   3u
#define PHS_S_PREP 5u
#define PHS_S_SPIN 11u

/* one producer: pull heads, issue 60 w16a16 matmuls/head to the consumer (44 diag SEQUENTIAL = Newton
 * dependency: each dispatch waits before the next; 16 merge). A small scalar "pack-proxy" before each
 * dispatch represents the crouton16/kmajor PREP glue (hidden under the much larger HMX matmul anyway). */
static void phs_producer(void *arg) {
    int slot = (int)(intptr_t)arg;
    phs_job *jb = &g_job[slot];
    uint8_t *scr = g_packscratch[slot];
    uint64_t life0 = phs_pcyc();
    uint64_t spin = 0;
    for (;;) {
        int h = __atomic_fetch_add(&g_next_head, 1, __ATOMIC_RELAXED);
        if (h >= g_H) break;
        /* 4 diagonal blocks, 11 sequential matmuls each (Taylor seed + Newton steps, dependent) */
        for (int b = 0; b < PHS_NB; ++b) {
            uint64_t d0 = phs_pcyc();
            for (int mm = 0; mm < PHS_DIAG_MM; ++mm) {
                uint64_t p0 = phs_pcyc();
                for (int i = 0; i < 16; ++i) scr[i] = (uint8_t)(scr[i] + i + mm);   /* PREP-proxy */
                uint64_t p1 = phs_pcyc(); phs_tr((uint32_t)slot, PHS_S_PREP, p0, p1);
                PHS_ARM(jb);
                uint64_t s0 = phs_pcyc();
                PHS_WAIT(jb); PHS_RESET(jb);
                uint64_t s1 = phs_pcyc(); spin += s1 - s0; phs_tr((uint32_t)slot, PHS_S_SPIN, s0, s1);
            }
            phs_tr((uint32_t)slot, PHS_S_DIAG, d0, phs_pcyc());
        }
        /* 16 merge matmuls (off-diag); same dispatch path */
        for (int mm = 0; mm < PHS_MERGE_MM; ++mm) {
            uint64_t p0 = phs_pcyc();
            for (int i = 0; i < 16; ++i) scr[i] = (uint8_t)(scr[i] + i);
            uint64_t p1 = phs_pcyc(); phs_tr((uint32_t)slot, PHS_S_PREP, p0, p1);
            PHS_ARM(jb);
            uint64_t s0 = phs_pcyc();
            PHS_WAIT(jb); PHS_RESET(jb);
            uint64_t s1 = phs_pcyc(); spin += s1 - s0; phs_tr((uint32_t)slot, PHS_S_SPIN, s0, s1);
        }
    }
    g_pspin[slot] = spin;
    g_plife[slot] = phs_pcyc() - life0;
    __atomic_fetch_add(&g_pdone, 1, __ATOMIC_RELEASE);
}

/* ---- Phase-2 diag-solve (H==2): X = (I-A)^-1 for one strictly-lower 64-block, all matmuls on
 * the REAL w16a16 primitive (M=256 carrier, rows 64..255 idle), Taylor(p=3)+Newton(4) = 11 mm.
 * Software scale tracking: matrix V == code * 2^eV / 32767 (mm: codes prod/32767, e adds).
 * payload A: [0..8K) A q16 codes (eA=0, strictly lower, |a|<1).
 * out T: [0..8K) X int16 codes; stats[7]=eX [8]=sat count [0]=wall [3]=mm cyc total [6]=us. */
static int16_t g_dsA[4096], g_dsAA[4096], g_dsA3[4096], g_dsM[4096], g_dsX[4096], g_dsZ[4096], g_dsT[4096];
static uint16_t g_dsAct[256 * 64];
static int g_sat;

/* z = x + y*2^(ey-ez) style saturating add with exponent alignment, plus auto-renorm into int16 */
static int ds_renorm(int32_t *acc, int16_t *dst) {  /* returns extra shift s so dst = acc>>s */
    int32_t mx = 0;
    for (int i = 0; i < 4096; ++i) { int32_t v = acc[i] < 0 ? -acc[i] : acc[i]; if (v > mx) mx = v; }
    int s = 0; while ((mx >> s) > 32639) ++s;
    if (s == 0) { while (mx && (mx << 1) <= 16384) { mx <<= 1; --s; } }   /* renorm up: keep codes large, e low */
    for (int i = 0; i < 4096; ++i) {
        int32_t v = (s >= 0) ? (acc[i] >> s) : (acc[i] << -s);
        if (v > 32639) { v = 32639; ++g_sat; } if (v < -32639) { v = -32639; ++g_sat; }
        dst[i] = (int16_t)v;
    }
    return s;
}

/* one w16a16 matmul on codes: out = a@w, e_out = e_a + e_w (codes = prod/32767 by the drain).
 * Only 64 act rows are live on the M=256 carrier: surface rows >=64 (slabs m32>=2) stay zp,
 * prefilled once by ds_mm_prep(); per-call we pack/depack 64 rows only. */
static uint64_t g_ds_mmcyc;
static void ds_mm_prep(w16a16_mm_t *mm) {
    uint16_t *s = (uint16_t *)mm->act;
    for (int i = 0; i < 256 * 64; ++i) s[i] = 32768;
}
static void ds_mm(w16a16_mm_t *mm, const int16_t *a, const int16_t *w, int16_t *out) {
    for (int i = 0; i < 4096; ++i) g_dsAct[i] = (uint16_t)(32768 + a[i]);
    static uint16_t Y[4096];
    uint64_t c0 = phs_pcyc();
    /* 64-row act pack into the prefilled M=256 surface (m32 0..1 slabs of each (row4,kt) block) */
    {
        uint16_t *s = (uint16_t *)mm->act;
        for (int row4 = 0; row4 < 8; ++row4)
            for (int kt = 0; kt < 2; ++kt) {
                uint16_t *blk = s + (row4 * 2 + kt) * 1024;
                for (int m32 = 0; m32 < 2; ++m32)
                    for (int rp = 0; rp < 2; ++rp) {
                        int r0 = m32 * 32 + row4 * 4 + rp * 2;
                        const uint16_t *s0 = &g_dsAct[r0 * 64 + kt * 32], *s1 = s0 + 64;
                        uint16_t *d = blk + (m32 * 2 + rp) * 64;
                        for (int c = 0; c < 32; ++c) { d[c * 2] = s0[c]; d[c * 2 + 1] = s1[c]; }
                    }
            }
        w16a16_pack_wt_kmajor(w, mm->wt, 64, 64);
        w16a16_pack_bias(w, mm->bias, 64, 64);
        w16a16_mm_run(mm);
        /* 64-row depack (m32 0..1 of each (row4,nt) out block) */
        const uint16_t *o = (const uint16_t *)mm->out;
        for (int row4 = 0; row4 < 8; ++row4)
            for (int nt = 0; nt < 2; ++nt) {
                const uint16_t *blk = o + (row4 * 2 + nt) * 1024;
                for (int m32 = 0; m32 < 2; ++m32)
                    for (int rp = 0; rp < 2; ++rp) {
                        int r0 = m32 * 32 + row4 * 4 + rp * 2;
                        uint16_t *d0 = &Y[r0 * 64 + nt * 32], *d1 = d0 + 64;
                        const uint16_t *sp = blk + (m32 * 2 + rp) * 64;
                        for (int c = 0; c < 32; ++c) { d0[c] = sp[c * 2]; d1[c] = sp[c * 2 + 1]; }
                    }
            }
    }
    g_ds_mmcyc += phs_pcyc() - c0;
    for (int i = 0; i < 4096; ++i) out[i] = (int16_t)((int)Y[i] - 32768);
}

/* core: X = (I-A)^-1 for one strictly-lower 64-block (10 w16a16 mm), returns eX. */
static int ds_diag(w16a16_mm_t *mm, const int16_t *Ablk, int16_t *X) {
    static int32_t acc[4096];
    memcpy(g_dsA, Ablk, 8192);
    ds_mm(mm, g_dsA, g_dsA, g_dsAA);
    ds_mm(mm, g_dsAA, g_dsA, g_dsA3);
    for (int i = 0; i < 4096; ++i) acc[i] = (int32_t)g_dsA[i] + g_dsAA[i] + g_dsA3[i];
    for (int d = 0; d < 64; ++d) acc[d * 65] += 32767;
    int eX = ds_renorm(acc, g_dsX);
    for (int i = 0; i < 4096; ++i) acc[i] = -(int32_t)g_dsA[i];
    for (int d = 0; d < 64; ++d) acc[d * 65] += 32767;
    int eM = ds_renorm(acc, g_dsM);
    for (int it = 0; it < 4; ++it) {
        ds_mm(mm, g_dsM, g_dsX, g_dsT);
        int e = eM + eX;
        for (int i = 0; i < 4096; ++i) acc[i] = -(int32_t)g_dsT[i];
        if (e < 31) for (int d = 0; d < 64; ++d) acc[d * 65] += (int32_t)(65534u >> e);
        int eZ = e + ds_renorm(acc, g_dsZ);
        ds_mm(mm, g_dsX, g_dsZ, g_dsT);
        for (int i = 0; i < 4096; ++i) acc[i] = g_dsT[i];
        eX = eX + eZ + ds_renorm(acc, g_dsX);
    }
    memcpy(X, g_dsX, 8192);
    return eX;
}

static int diag_solve(const uint8_t *A, int *stats, int statsLen, void *T, int TLen) {
    if (TLen < 8192) return -2;
    static int pwr_client; void *pctx = &pwr_client;
    HAP_power_set_core_corner(pctx, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_MAX);
    { HAP_power_request_t r; memset(&r, 0, sizeof(r)); r.type = HAP_power_set_HMX; r.hmx.power_up = TRUE; HAP_power_set(pctx, &r); }
    compute_res_attr_t va; HAP_compute_res_attr_init(&va);
    HAP_compute_res_attr_set_vtcm_param(&va, 0x100000u, 0);
    unsigned int vctx = HAP_compute_res_acquire(&va, 2000000);
    uint8_t *vbase = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&va);
    compute_res_attr_t ha; HAP_compute_res_attr_init(&ha); HAP_compute_res_attr_set_hmx_param(&ha, 1);
    unsigned int hctx = HAP_compute_res_acquire(&ha, 2000000);
    int hl = HAP_compute_res_hmx_lock(hctx);
    if (!vbase || hl != 0) { if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
        if (vctx) HAP_compute_res_release(vctx); return -1; }
    static uint8_t descs[256] __attribute__((aligned(64)));
    w16a16_mm_t mm; w16a16_mm_init(&mm, vbase, descs);
    memset(mm.out, 0, W16MM_OUT_BYTES);
    ds_mm_prep(&mm);

    g_sat = 0; g_ds_mmcyc = 0;
    uint64_t us0 = HAP_perf_get_time_us(), t0 = phs_pcyc();
    int eX = ds_diag(&mm, (const int16_t *)A, (int16_t *)T);
    uint64_t t1 = phs_pcyc(), us1 = HAP_perf_get_time_us();
    if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
    if (vctx) HAP_compute_res_release(vctx);
    { HAP_power_request_t off; memset(&off, 0, sizeof(off)); off.type = HAP_power_set_HMX; off.hmx.power_up = FALSE; HAP_power_set(pctx, &off); }
    if (statsLen > 0) stats[0] = (int)(t1 - t0);
    if (statsLen > 3) stats[3] = (int)g_ds_mmcyc;
    if (statsLen > 6) stats[6] = (int)(us1 - us0);
    if (statsLen > 7) stats[7] = eX;
    if (statsLen > 8) stats[8] = g_sat;
    FARF(ALWAYS, "DIAGSOLVE: wall=%llu mm=%llu cyc eX=%d sat=%d", (unsigned long long)(t1 - t0),
         (unsigned long long)g_ds_mmcyc, eX, g_sat);
    return 0;
}

/* ---- Phase-3 head solve (H==3): full C=256 T = (I-A)^-1, 4 diag (10 mm) + off-diag merges
 * T_ij = T_ii @ sum_k A_ik T_kj (16 mm) = 56 w16a16 mm/head.
 * payload A: 256x256 q16 codes (strictly lower, 128KB).
 * out T: 256x256 int16 codes (128KB) + 16 int32 exponents (per block, row-major). */
static int16_t g_p3A[16][4096], g_p3T[16][4096];
static int g_p3e[16];

static int head_solve(const uint8_t *Ah, int *stats, int statsLen, void *T, int TLen) {
    if (TLen < 131072 + 64) return -2;
    static int pwr_client; void *pctx = &pwr_client;
    HAP_power_set_core_corner(pctx, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_MAX);
    { HAP_power_request_t r; memset(&r, 0, sizeof(r)); r.type = HAP_power_set_HMX; r.hmx.power_up = TRUE; HAP_power_set(pctx, &r); }
    compute_res_attr_t va; HAP_compute_res_attr_init(&va);
    HAP_compute_res_attr_set_vtcm_param(&va, 0x100000u, 0);
    unsigned int vctx = HAP_compute_res_acquire(&va, 2000000);
    uint8_t *vbase = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&va);
    compute_res_attr_t ha; HAP_compute_res_attr_init(&ha); HAP_compute_res_attr_set_hmx_param(&ha, 1);
    unsigned int hctx = HAP_compute_res_acquire(&ha, 2000000);
    int hl = HAP_compute_res_hmx_lock(hctx);
    if (!vbase || hl != 0) { if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
        if (vctx) HAP_compute_res_release(vctx); return -1; }
    static uint8_t descs[256] __attribute__((aligned(64)));
    w16a16_mm_t mm; w16a16_mm_init(&mm, vbase, descs);
    memset(mm.out, 0, W16MM_OUT_BYTES);
    ds_mm_prep(&mm);

    const int16_t *Aq = (const int16_t *)Ah;
    for (int bi = 0; bi < 4; ++bi) for (int bj = 0; bj < 4; ++bj)
        for (int r = 0; r < 64; ++r)
            memcpy(&g_p3A[bi * 4 + bj][r * 64], &Aq[(bi * 64 + r) * 256 + bj * 64], 128);

    g_sat = 0; g_ds_mmcyc = 0;
    static int32_t acc[4096];
    static int16_t prod[4096];
    uint64_t us0 = HAP_perf_get_time_us(), t0 = phs_pcyc();
    for (int b = 0; b < 4; ++b) g_p3e[b * 5] = ds_diag(&mm, g_p3A[b * 5], g_p3T[b * 5]);
    for (int d = 1; d < 4; ++d)                                  /* off-diag by ascending depth */
        for (int i = d; i < 4; ++i) {
            int j = i - d;
            int eAcc = 0; for (int z = 0; z < 4096; ++z) acc[z] = 0;
            for (int k = j; k < i; ++k) {                         /* sum_k A_ik @ T_kj  (A e=0) */
                ds_mm(&mm, g_p3A[i * 4 + k], g_p3T[k * 4 + j], prod);
                int ep = g_p3e[k * 4 + j];
                if (k == j) eAcc = ep;
                else if (ep > eAcc) { int sh = ep - eAcc; for (int z = 0; z < 4096; ++z) acc[z] >>= sh; eAcc = ep; }
                int sh = eAcc - ep;
                for (int z = 0; z < 4096; ++z) acc[z] += (sh < 31) ? (prod[z] >> sh) : 0;
            }
            int eS = eAcc + ds_renorm(acc, prod);
            ds_mm(&mm, g_p3T[i * 4 + i], prod, g_p3T[i * 4 + j]);
            for (int z = 0; z < 4096; ++z) acc[z] = g_p3T[i * 4 + j][z];
            g_p3e[i * 4 + j] = g_p3e[i * 4 + i] + eS + ds_renorm(acc, g_p3T[i * 4 + j]);
        }
    uint64_t t1 = phs_pcyc(), us1 = HAP_perf_get_time_us();

    int16_t *To = (int16_t *)T;
    memset(To, 0, 131072);
    for (int bi = 0; bi < 4; ++bi) for (int bj = 0; bj <= bi; ++bj)
        for (int r = 0; r < 64; ++r)
            memcpy(&To[(bi * 64 + r) * 256 + bj * 64], &g_p3T[bi * 4 + bj][r * 64], 128);
    memcpy((uint8_t *)T + 131072, g_p3e, 64);

    if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
    if (vctx) HAP_compute_res_release(vctx);
    { HAP_power_request_t off; memset(&off, 0, sizeof(off)); off.type = HAP_power_set_HMX; off.hmx.power_up = FALSE; HAP_power_set(pctx, &off); }
    if (statsLen > 0) stats[0] = (int)(t1 - t0);
    if (statsLen > 3) stats[3] = (int)g_ds_mmcyc;
    if (statsLen > 6) stats[6] = (int)(us1 - us0);
    if (statsLen > 8) stats[8] = g_sat;
    FARF(ALWAYS, "HEADSOLVE: wall=%llu mm=%llu cyc sat=%d", (unsigned long long)(t1 - t0),
         (unsigned long long)g_ds_mmcyc, g_sat);
    return 0;
}

/* ---- Phase-4 crouton-domain head solve (H==5): same math as Phase-3, zero per-mm depack:
 * X/T/Z/... all live as crouton16 codes (the kernel's act/out layout, 64 live rows of the M=256
 * carrier = 16KB). act = 16KB copy; weight = LUT-driven 4-pass kmajor pack (colsum on the fly).
 * Pointwise add/renorm are layout-agnostic. */
static uint16_t g_p4lut[4096];   /* lut[r*64+c] = surface u16 idx (live rows 0..63) */
static void p4_lut_init(void) {
    for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c)
        g_p4lut[r * 64 + c] = (uint16_t)(((((r >> 2) & 7) * 2 + (c >> 5)) * 1024) +
            (((r >> 5) * 2 + ((r >> 1) & 1)) * 64) + ((c & 31) * 2) + (r & 1));
}
/* compact crouton vector cv[4096]: live 512B of each of the 16 surface blocks (slabs m32 0..1).
 * cv idx = blk*256 + intra; surface u16 idx = blk*1024 + intra. */
static void p4_lutc_init(void) {
    for (int i = 0; i < 4096; ++i) {
        int s = g_p4lut[i];
        g_p4lut[i] = (uint16_t)(((s >> 10) << 8) + (s & 1023));   /* (s&1023) < 256 for live rows */
    }
}
/* weight pack from cv: LUT linearize + proven kmajor/bias packers. */
static void p4_pack_wt_bias(const int16_t *cv, uint8_t *wt, int32_t *bias, int16_t *lin) {
    for (int i = 0; i < 4096; ++i) lin[i] = cv[g_p4lut[i]];
    w16a16_pack_wt_kmajor(lin, wt, 64, 64);
    w16a16_pack_bias(lin, bias, 64, 64);
}
/* one cv-domain matmul: out = a@w (codes prod/32767), zero depack; ~85K prep + 42K kernel. */
static void p4_mm(w16a16_mm_t *mm, const int16_t *a_cv, const int16_t *w_cv, int16_t *out_cv, int16_t *lin) {
    uint16_t *act = (uint16_t *)mm->act;
    for (int b = 0; b < 16; ++b) {
        const int16_t *s = a_cv + b * 256; uint16_t *d = act + b * 1024;
        for (int i = 0; i < 256; ++i) d[i] = (uint16_t)(32768 + s[i]);
    }
    p4_pack_wt_bias(w_cv, mm->wt, mm->bias, lin);
    uint64_t c0 = phs_pcyc();
    w16a16_mm_run(mm);
    g_ds_mmcyc += phs_pcyc() - c0;
    const uint16_t *o = (const uint16_t *)mm->out;
    for (int b = 0; b < 16; ++b) {
        const uint16_t *s = o + b * 1024; int16_t *d = out_cv + b * 256;
        for (int i = 0; i < 256; ++i) d[i] = (int16_t)((int)s[i] - 32768);
    }
}
/* cv diag inverse: X=(I-A)^-1, 10 p4_mm, exponent tracking (same math as ds_diag). */
static int p4_diag(w16a16_mm_t *mm, const int16_t *Acv, int16_t *Xcv, int16_t *scr /*4*4096*/, int16_t *lin, int32_t *acc) {
    int16_t *AA = scr, *A3 = scr + 4096, *M = scr + 8192, *Z = scr + 12288;
    p4_mm(mm, Acv, Acv, AA, lin);
    p4_mm(mm, AA, Acv, A3, lin);
    for (int i = 0; i < 4096; ++i) acc[i] = (int32_t)Acv[i] + AA[i] + A3[i];
    for (int d = 0; d < 64; ++d) acc[g_p4lut[d * 65]] += 32767;
    int eX = ds_renorm(acc, Xcv);
    for (int i = 0; i < 4096; ++i) acc[i] = -(int32_t)Acv[i];
    for (int d = 0; d < 64; ++d) acc[g_p4lut[d * 65]] += 32767;
    int eM = ds_renorm(acc, M);
    for (int it = 0; it < 4; ++it) {
        p4_mm(mm, M, Xcv, Z, lin);                       /* MX */
        int e = eM + eX;
        for (int i = 0; i < 4096; ++i) acc[i] = -(int32_t)Z[i];
        if (e < 31) for (int d = 0; d < 64; ++d) acc[g_p4lut[d * 65]] += (int32_t)(65534u >> e);
        int eZ = e + ds_renorm(acc, Z);
        p4_mm(mm, Xcv, Z, AA, lin);                      /* reuse AA as tmp */
        for (int i = 0; i < 4096; ++i) acc[i] = AA[i];
        eX = eX + eZ + ds_renorm(acc, Xcv);
    }
    return eX;
}

/* ---- Phase-4 head solve, cv-domain (H>=5): H heads, 56 mm/head, single-thread baseline.
 * payload A: H x 256x256 q16 strictly lower (128KB/head). out T: per head 256x256 int16 + exps tail. */
static int16_t g_p4A[16][4096], g_p4T[16][4096], g_p4scr[4 * 4096], g_p4lin[4096], g_p4prod[4096];
static int32_t g_p4acc[4096];

static int p4_head_solve(const uint8_t *Ah, int H, int *stats, int statsLen, void *T, int TLen) {
    if (TLen < H * 131072) return -2;
    static int pwr_client; void *pctx = &pwr_client;
    HAP_power_set_core_corner(pctx, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_MAX);
    { HAP_power_request_t r; memset(&r, 0, sizeof(r)); r.type = HAP_power_set_HMX; r.hmx.power_up = TRUE; HAP_power_set(pctx, &r); }
    compute_res_attr_t va; HAP_compute_res_attr_init(&va);
    HAP_compute_res_attr_set_vtcm_param(&va, 0x100000u, 0);
    unsigned int vctx = HAP_compute_res_acquire(&va, 2000000);
    uint8_t *vbase = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&va);
    compute_res_attr_t ha; HAP_compute_res_attr_init(&ha); HAP_compute_res_attr_set_hmx_param(&ha, 1);
    unsigned int hctx = HAP_compute_res_acquire(&ha, 2000000);
    int hl = HAP_compute_res_hmx_lock(hctx);
    if (!vbase || hl != 0) { if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
        if (vctx) HAP_compute_res_release(vctx); return -1; }
    static uint8_t descs[256] __attribute__((aligned(64)));
    w16a16_mm_t mm; w16a16_mm_init(&mm, vbase, descs);
    memset(mm.out, 0, W16MM_OUT_BYTES);
    { uint16_t *s = (uint16_t *)mm.act; for (int i = 0; i < 256 * 64; ++i) s[i] = 32768; }
    p4_lut_init(); p4_lutc_init();
    /* scratch stays in DDR BSS: scalar VTCM access measured 4x SLOWER (753M vs 187M / 8 heads). */
    g_sat = 0; g_ds_mmcyc = 0;
    uint64_t us0 = HAP_perf_get_time_us(), t0 = phs_pcyc();
    for (int h = 0; h < H; ++h) {
        const int16_t *Aq = (const int16_t *)(Ah + (size_t)h * 131072);
        int eT[16];
        for (int bi = 0; bi < 4; ++bi) for (int bj = 0; bj <= bi; ++bj) {
            int16_t *dst = g_p4A[bi * 4 + bj];
            for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c)
                dst[g_p4lut[r * 64 + c]] = Aq[(bi * 64 + r) * 256 + bj * 64 + c];
        }
        for (int b = 0; b < 4; ++b) eT[b * 5] = p4_diag(&mm, g_p4A[b * 5], g_p4T[b * 5], g_p4scr, g_p4lin, g_p4acc);
        for (int d = 1; d < 4; ++d)
            for (int i = d; i < 4; ++i) {
                int j = i - d, eAcc = 0;
                for (int z = 0; z < 4096; ++z) g_p4acc[z] = 0;
                for (int k = j; k < i; ++k) {
                    p4_mm(&mm, g_p4A[i * 4 + k], g_p4T[k * 4 + j], g_p4prod, g_p4lin);
                    int ep = eT[k * 4 + j];
                    if (k == j) eAcc = ep;
                    else if (ep > eAcc) { int sh = ep - eAcc; for (int z = 0; z < 4096; ++z) g_p4acc[z] >>= sh; eAcc = ep; }
                    int sh = eAcc - ep;
                    for (int z = 0; z < 4096; ++z) g_p4acc[z] += (sh < 31) ? (g_p4prod[z] >> sh) : 0;
                }
                int eS = eAcc + ds_renorm(g_p4acc, g_p4prod);
                p4_mm(&mm, g_p4T[i * 4 + i], g_p4prod, g_p4T[i * 4 + j], g_p4lin);
                for (int z = 0; z < 4096; ++z) g_p4acc[z] = g_p4T[i * 4 + j][z];
                eT[i * 4 + j] = eT[i * 4 + i] + eS + ds_renorm(g_p4acc, g_p4T[i * 4 + j]);
            }
        int16_t *To = (int16_t *)T + (size_t)h * 65536;
        memset(To, 0, 131072);
        for (int bi = 0; bi < 4; ++bi) for (int bj = 0; bj <= bi; ++bj) {
            const int16_t *src = g_p4T[bi * 4 + bj];
            for (int r = 0; r < 64; ++r) for (int c = 0; c < 64; ++c)
                To[(bi * 64 + r) * 256 + bj * 64 + c] = src[g_p4lut[r * 64 + c]];
        }
        memcpy((uint8_t *)To + 128, eT, 64);   /* exps in the unused upper triangle (row0,col64) */
    }
    uint64_t t1 = phs_pcyc(), us1 = HAP_perf_get_time_us();
    if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
    if (vctx) HAP_compute_res_release(vctx);
    { HAP_power_request_t off; memset(&off, 0, sizeof(off)); off.type = HAP_power_set_HMX; off.hmx.power_up = FALSE; HAP_power_set(pctx, &off); }
    if (statsLen > 0) stats[0] = (int)(t1 - t0);
    if (statsLen > 3) stats[3] = (int)g_ds_mmcyc;
    if (statsLen > 6) stats[6] = (int)(us1 - us0);
    if (statsLen > 8) stats[8] = g_sat;
    FARF(ALWAYS, "P4HEAD: H=%d wall=%llu mm=%llu sat=%d", H, (unsigned long long)(t1 - t0),
         (unsigned long long)g_ds_mmcyc, g_sat);
    return 0;
}

/* ---- HVX helpers for the producer prep (vmem on cached DDR is fine; only vscatter needs VTCM). */
#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"
#define P4V ((int)sizeof(HVX_Vector))            /* 128B */
/* dst_u16[i] = src_i16[i] + 32768 (= xor 0x8000), n multiple of 64 */
static inline void p4v_i16_to_u16(uint16_t *dst, const int16_t *src, int n) {
    const HVX_Vector K = Q6_Vh_vsplat_R(0x8000);
    HVX_Vector *d = (HVX_Vector *)dst; const HVX_Vector *s = (const HVX_Vector *)src;
    for (int i = 0; i < n / 64; ++i) d[i] = Q6_V_vxor_VV(s[i], K);
}
/* u16->i16 (xor 0x8000) — same op both directions. */
#define p4v_u16_to_i16(dst, src, n) p4v_i16_to_u16((uint16_t *)(dst), (const int16_t *)(src), (n))

/* ---- HVX int32 acc helpers. acc layout = 64 chunks x {lo(even hw lanes), hi(odd)} vec pairs.
 * Pointwise ops are order-agnostic; only the 64 diag fixups need the interleaved index. */
static uint16_t g_p4lut2[4096];   /* fwd LUT copy needed before diag fixup helper (init alias) */
static inline void p4v_acc3(int32_t *acc, const int16_t *a, const int16_t *b, const int16_t *c) {
    const HVX_Vector *va = (const HVX_Vector *)a, *vb = (const HVX_Vector *)b, *vc = (const HVX_Vector *)c;
    HVX_Vector *d = (HVX_Vector *)acc;
    for (int i = 0; i < 64; ++i) {
        HVX_VectorPair s = Q6_Ww_vadd_WwWw(Q6_Ww_vadd_WwWw(Q6_Ww_vsxt_Vh(va[i]), Q6_Ww_vsxt_Vh(vb[i])),
                                           Q6_Ww_vsxt_Vh(vc[i]));
        d[2 * i] = Q6_V_lo_W(s); d[2 * i + 1] = Q6_V_hi_W(s);
    }
}
static inline void p4v_acc_negw(int32_t *acc, const int16_t *a) {     /* acc = -a (widened) */
    const HVX_Vector *va = (const HVX_Vector *)a; HVX_Vector *d = (HVX_Vector *)acc;
    HVX_VectorPair z = Q6_Ww_vsxt_Vh(Q6_V_vzero());
    for (int i = 0; i < 64; ++i) {
        HVX_VectorPair s = Q6_Ww_vsxt_Vh(va[i]);
        d[2 * i] = Q6_Vw_vsub_VwVw(Q6_V_lo_W(z), Q6_V_lo_W(s));
        d[2 * i + 1] = Q6_Vw_vsub_VwVw(Q6_V_hi_W(z), Q6_V_hi_W(s));
    }
}
static inline void p4v_acc_zero(int32_t *acc) {
    HVX_Vector *d = (HVX_Vector *)acc; HVX_Vector z = Q6_V_vzero();
    for (int i = 0; i < 128; ++i) d[i] = z;
}
static inline void p4v_acc_addsh(int32_t *acc, const int16_t *p, int sh) {   /* acc += p>>sh */
    const HVX_Vector *vp = (const HVX_Vector *)p; HVX_Vector *d = (HVX_Vector *)acc;
    for (int i = 0; i < 64; ++i) {
        HVX_VectorPair w = Q6_Ww_vsxt_Vh(vp[i]);
        d[2 * i] = Q6_Vw_vadd_VwVw(d[2 * i], Q6_Vw_vasr_VwR(Q6_V_lo_W(w), sh));
        d[2 * i + 1] = Q6_Vw_vadd_VwVw(d[2 * i + 1], Q6_Vw_vasr_VwR(Q6_V_hi_W(w), sh));
    }
}
static inline void p4v_acc_shr(int32_t *acc, int sh) {
    HVX_Vector *d = (HVX_Vector *)acc;
    for (int i = 0; i < 128; ++i) d[i] = Q6_Vw_vasr_VwR(d[i], sh);
}
static inline int32_t p4v_acc_absmax(const int32_t *acc) {
    const HVX_Vector *d = (const HVX_Vector *)acc;
    HVX_Vector mx = Q6_V_vzero();
    for (int i = 0; i < 128; ++i) mx = Q6_Vw_vmax_VwVw(mx, Q6_Vw_vabs_Vw(d[i]));
    for (int sh = 64; sh >= 4; sh >>= 1) mx = Q6_Vw_vmax_VwVw(mx, Q6_V_vror_VR(mx, sh));
    union { HVX_Vector v; int32_t w[32]; } u; u.v = mx; return u.w[0];
}
static inline void p4v_acc_to_cv(int16_t *cv, const int32_t *acc, int s) {   /* clip ±32639 */
    const HVX_Vector *d = (const HVX_Vector *)acc; HVX_Vector *o = (HVX_Vector *)cv;
    const HVX_Vector CP = Q6_Vh_vsplat_R(32639), CN = Q6_Vh_vsplat_R(-32639);
    for (int i = 0; i < 64; ++i) {
        HVX_Vector lo = d[2 * i], hi = d[2 * i + 1];
        if (s < 0) { lo = Q6_Vw_vasl_VwR(lo, -s); hi = Q6_Vw_vasl_VwR(hi, -s); }
        o[i] = Q6_Vh_vasr_VwVwR_sat(hi, lo, s > 0 ? (s & 31) : 0);
        o[i] = Q6_Vh_vmin_VhVh(Q6_Vh_vmax_VhVh(o[i], CN), CP);
    }
}
static inline int p4v_renorm(int32_t *acc, int16_t *cv) {
    int32_t mx = p4v_acc_absmax(acc);
    int s = 0; while ((mx >> s) > 32639) ++s;
    if (s == 0) { while (mx && (mx << 1) <= 16384) { mx <<= 1; --s; } }
    p4v_acc_to_cv(cv, acc, s);
    return s;
}
static inline void p4v_acc_diag_add(int32_t *acc, int32_t add) {   /* vsxt/vasr = even/odd interleave */
    for (int d = 0; d < 64; ++d) {
        int i = g_p4lut2[d * 65], off = i & 63;
        acc[(i >> 6) * 64 + (off & 1) * 32 + (off >> 1)] += add;
    }
}
static inline void p4v_acc_from_cv(int32_t *acc, const int16_t *a) {  /* widen copy */
    const HVX_Vector *va = (const HVX_Vector *)a; HVX_Vector *d = (HVX_Vector *)acc;
    for (int i = 0; i < 64; ++i) {
        HVX_VectorPair s = Q6_Ww_vsxt_Vh(va[i]);
        d[2 * i] = Q6_V_lo_W(s); d[2 * i + 1] = Q6_V_hi_W(s);
    }
}

/* HVX weight pack: gather q16 (stream order) from a VTCM staging copy of w_cv, then lo/hi split +
 * 4lo|4hi interleave (vshuff 4B grains). g_p4hw[h] = byte offset (in staging) of stream halfword h. */
static uint16_t *g_p4hw;          /* shared VTCM LUT, 4096 u16 */
static uint16_t *g_p4il, *g_p4fl; /* inverse/forward permutation LUTs (VTCM, byte offsets) */
/* permute via vgather: dst[j] = src[ofs[j]/2]; src must be staged in VTCM (8KB) */
static void p4v_perm(int16_t *dst, const int16_t *src, const uint16_t *ofs, int16_t *stage) {
    for (int i = 0; i < 64; ++i) ((HVX_Vector *)stage)[i] = ((const HVX_Vector *)src)[i];
    HVX_Vector *g = (HVX_Vector *)(stage + 4096);
    const HVX_Vector *o = (const HVX_Vector *)ofs;
    for (int v = 0; v < 64; ++v) {
        Q6_vgather_ARMVh((void *)g, (uint32_t)(uintptr_t)stage, 8191, o[v]);
        ((HVX_Vector *)dst)[v] = *g;
    }
}
static void p4_hwlut_init(void) { /* stream order from the proven scalar packer loops */
    int h = 0;
    for (int nt = 0; nt < 2; ++nt)
        for (int half = 0; half < 2; ++half)
            for (int kt = 0; kt < 2; ++kt)
                for (int grp = 0; grp < 8; ++grp)
                    for (int idx = 0; idx < 64; idx += 8)
                        for (int j = 0; j < 8; ++j) {
                            int vi = idx + j, lane = vi / 16, off0 = (grp * 8 + half * 4 + lane) * 16;
                            int off = off0 + (vi & 15);
                            int rgrp = off / 128, rem = off % 128;
                            int col = rem / 4, row = rgrp * 4 + rem % 4;
                            g_p4hw[h++] = (uint16_t)(2 * g_p4lut[(kt * 32 + row) * 64 + (nt * 32 + col)]);
                        }
}
static void p4v_pack_wt_bias(const int16_t *w_cv, int16_t *stage /*VTCM 8KB*/, uint8_t *wt, int32_t *bias) {
    for (int i = 0; i < 4096 / 64; ++i) ((HVX_Vector *)stage)[i] = ((const HVX_Vector *)w_cv)[i];
    HVX_Vector *gtmp = (HVX_Vector *)(stage + 4096);          /* 2 gather slots right after staging */
    const HVX_Vector *ofs = (const HVX_Vector *)g_p4hw;
    for (int v = 0; v < 64; v += 2) {
        Q6_vgather_ARMVh((void *)&gtmp[0], (uint32_t)(uintptr_t)stage, 8191, ofs[v]);
        Q6_vgather_ARMVh((void *)&gtmp[1], (uint32_t)(uintptr_t)stage, 8191, ofs[v + 1]);
        HVX_Vector q0 = gtmp[0], q1 = gtmp[1];
        HVX_Vector lo = Q6_Vb_vpacke_VhVh(q1, q0);                                  /* low bytes  */
        const HVX_Vector K128 = Q6_Vh_vsplat_R(128);
        HVX_Vector h0 = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh_sat(q0, K128), 8);
        HVX_Vector h1 = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh_sat(q1, K128), 8);
        HVX_Vector hi = Q6_Vb_vpack_VhVh_sat(h1, h0);
        HVX_VectorPair il = Q6_W_vshuff_VVR(hi, lo, -4);                            /* 4lo|4hi grains */
        ((HVX_Vector *)wt)[v] = Q6_V_lo_W(il); ((HVX_Vector *)wt)[v + 1] = Q6_V_hi_W(il);
    }
    /* bias colsum HVX (vsxt order: lo=even hw lanes -> col c parity0, hi -> parity1; cs = lo+hi) */
    static const int32_t ctrl[2] = { 0x00404420, 0x40000000 };
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
    for (int g = 0; g < 4; ++g) for (int i = 0; i < 32; ++i) bias[g * 64 + i] = ctrl[i & 1];
}

/* ---- Phase-4 threaded (H>=5 && P>=2): P producers chain whole heads (scalar prep in DDR),
 * 1 main HMX consumer runs kernels on per-slot VTCM buffers. */
struct p4_slot {
    w16a16_mm_t mm; uint8_t descs[256] __attribute__((aligned(64)));
    int16_t A[16][4096] __attribute__((aligned(128))), T[16][4096] __attribute__((aligned(128)));
    int16_t scr[4 * 4096] __attribute__((aligned(128))), lin[4096], prod[4096] __attribute__((aligned(128)));
    int32_t acc[4096] __attribute__((aligned(128)));
};
static p4_slot g_p4s[PHS_NT];
static const uint8_t *g_p4Ah; static uint8_t *g_p4Th; static int g_p4H; static int g_p4trace;

static int16_t *g_p4stage[PHS_NT];   /* per-slot VTCM staging for the HVX weight gather */

/* cv >> sh with sat (for K-stack common-exponent alignment) */
static inline void p4v_cv_shr(int16_t *dst, const int16_t *src, int sh) {
    const HVX_Vector *v = (const HVX_Vector *)src; HVX_Vector *d = (HVX_Vector *)dst;
    for (int i = 0; i < 64; ++i) d[i] = Q6_Vh_vasr_VhR(v[i], sh & 15);
}
/* weight gather pack of ONE 64x64 cv into two nt segment dsts (each 32 vecs) + colsum accumulate */
static void p4v_pack_wt_seg(const int16_t *w_cv, int16_t *stage, HVX_Vector *dst0, HVX_Vector *dst1, long *cs) {
    for (int i = 0; i < 4096 / 64; ++i) ((HVX_Vector *)stage)[i] = ((const HVX_Vector *)w_cv)[i];
    HVX_Vector *gtmp = (HVX_Vector *)(stage + 4096);
    const HVX_Vector *ofs = (const HVX_Vector *)g_p4hw;
    for (int v = 0; v < 64; v += 2) {
        Q6_vgather_ARMVh((void *)&gtmp[0], (uint32_t)(uintptr_t)stage, 8191, ofs[v]);
        Q6_vgather_ARMVh((void *)&gtmp[1], (uint32_t)(uintptr_t)stage, 8191, ofs[v + 1]);
        HVX_Vector q0 = gtmp[0], q1 = gtmp[1];
        HVX_Vector lo = Q6_Vb_vpacke_VhVh(q1, q0);
        const HVX_Vector K128 = Q6_Vh_vsplat_R(128);
        HVX_Vector h0 = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh_sat(q0, K128), 8);
        HVX_Vector h1 = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh_sat(q1, K128), 8);
        HVX_Vector hi = Q6_Vb_vpack_VhVh_sat(h1, h0);
        HVX_VectorPair il = Q6_W_vshuff_VVR(hi, lo, -4);
        if (v < 32) { dst0[v] = Q6_V_lo_W(il); dst0[v + 1] = Q6_V_hi_W(il); }
        else        { dst1[v - 32] = Q6_V_lo_W(il); dst1[v - 31] = Q6_V_hi_W(il); }
    }
    for (int n = 0; n < 64; ++n) {
        long c = 0; for (int k = 0; k < 64; ++k) c += w_cv[g_p4lut[k * 64 + n]];
        cs[n] += c;
    }
}
static void p4_bias_fin(int32_t *bias, const long *cs) {
    static const int32_t ctrl[2] = { 0x00404420, 0x40000000 };
    for (int n = 0; n < 64; ++n) {
        long v2 = -cs[n], eff = (v2 >= 0) ? (v2 / 2) : -(((-v2) + 1) / 2);
        int g = n >> 4, idx = n & 15;
        bias[g * 64 + 32 + idx * 2] = (int32_t)eff; bias[g * 64 + 32 + idx * 2 + 1] = 0;
    }
    for (int g = 0; g < 4; ++g) for (int i = 0; i < 32; ++i) bias[g * 64 + i] = ctrl[i & 1];
}
/* K-stack mm: out = sum_b act_b @ wt_b (b=0..d-1), all wt pre-aligned to e_common */
static void p4_mm_stack(int slot, int d, const int16_t *const *acts, const int16_t *const *wts,
                        const int *wshift, int16_t *out_cv) {
#if defined(P4_STACK_NOOP)
    (void)slot; (void)d; (void)acts; (void)wts; (void)wshift; (void)out_cv; return;
#endif
    p4_slot *S = &g_p4s[slot];
    uint16_t *act = (uint16_t *)S->mm.act;
    uint64_t p0 = phs_pcyc();
    int Kt = 2 * d;
#if !defined(P4_STACK_NOATAB)
    for (int rg = 0; rg < 64; ++rg) for (int kt = 0; kt < Kt; ++kt)
        S->mm.atab[rg * Kt + kt] = (int32_t)(uintptr_t)((uint8_t *)act + (((rg & 7) * Kt + kt) * 2048));
#endif
#if !defined(P4_STACK_NOACT)
    for (int b = 0; b < d; ++b)
        for (int r4 = 0; r4 < 8; ++r4) for (int kk = 0; kk < 2; ++kk)
            p4v_i16_to_u16(act + ((r4 * Kt) + b * 2 + kk) * 1024, acts[b] + (r4 * 2 + kk) * 256, 256);
#endif
    static long cs_slot[PHS_NT][64];
    long *cs = cs_slot[slot];
    for (int n = 0; n < 64; ++n) cs[n] = 0;
    for (int b = 0; b < d; ++b) {
        const int16_t *w = wts[b];
#if !defined(P4_STACK_NOSHR)
        if (wshift[b] > 0) { p4v_cv_shr(g_p4stage[slot] + 4096 + 128, w, wshift[b]); w = g_p4stage[slot] + 4096 + 128; }
#endif
#if defined(P4_STACK_NOWT)
        (void)w; ++cs[0];
#elif defined(P4_STACK_SCALARWT)
        {   /* bisect: scalar packer (lin via LUT, full 8K stream) split into nt segments */
            static int16_t lin[PHS_NT][4096]; static uint8_t tmp[PHS_NT][8192];
            for (int i = 0; i < 4096; ++i) lin[slot][i] = w[g_p4lut[i]];
            w16a16_pack_wt_kmajor(lin[slot], tmp[slot], 64, 64);
            memcpy(S->mm.wt + (size_t)b * 4096, tmp[slot], 4096);
            memcpy(S->mm.wt + (size_t)(d + b) * 4096, tmp[slot] + 4096, 4096);
            for (int n = 0; n < 64; ++n) { long c = 0; for (int k = 0; k < 64; ++k) c += lin[slot][k * 64 + n]; cs[n] += c; }
        }
#else
        p4v_pack_wt_seg(w, g_p4stage[slot],
                        (HVX_Vector *)(S->mm.wt + (size_t)b * 4096),
                        (HVX_Vector *)(S->mm.wt + (size_t)(d + b) * 4096), cs);
#endif
    }
#if !defined(P4_STACK_NOBIAS)
    p4_bias_fin(S->mm.bias, cs);
#endif
    hmx_conv_act_desc_t *ad = (hmx_conv_act_desc_t *)S->mm.ad;
#if defined(P4_STACK_D1KERNEL)
    ad->n_act_pairs = 2u; ad->act_table_y_stride_words = 128u;  /* bisect: prep stacked, kernel d=1 (WRONG math) */
    for (int rg = 0; rg < 64; ++rg) for (int t2 = 0; t2 < 2; ++t2)
        S->mm.atab[rg * 2 + t2] = (int32_t)(uintptr_t)((uint8_t *)act + (((rg & 7) * Kt + t2) * 2048));
#else
    ad->n_act_pairs = (uint32_t)Kt; ad->act_table_y_stride_words = (uint32_t)(64 * Kt);
#endif
    uint64_t p1 = phs_pcyc(); phs_tr((uint32_t)slot, PHS_S_PREP, p0, p1);
#if !defined(P4_STACK_NOHMX)
    PHS_ARM(&g_job[slot]);
    uint64_t s0 = phs_pcyc(); PHS_WAIT(&g_job[slot]); PHS_RESET(&g_job[slot]);
    uint64_t s1 = phs_pcyc(); g_pspin[slot] += s1 - s0; phs_tr((uint32_t)slot, PHS_S_SPIN, s0, s1);
#endif
    const uint16_t *o = (const uint16_t *)S->mm.out;
#if !defined(P4_STACK_NOOUT)
    for (int b = 0; b < 16; ++b) p4v_u16_to_i16(out_cv + b * 256, o + b * 1024, 256);
#endif
    (void)o;
    ad->n_act_pairs = 2u; ad->act_table_y_stride_words = 128u;   /* restore d=1 for diag mms */
    for (int rg = 0; rg < 64; ++rg) for (int t2 = 0; t2 < 2; ++t2)
        S->mm.atab[rg * 2 + t2] = (int32_t)(uintptr_t)((uint8_t *)act + (((rg & 7) * 2 + t2) * 2048));
}
static void p4_mm_thr(int slot, const int16_t *a_cv, const int16_t *w_cv, int16_t *out_cv) {
    p4_slot *S = &g_p4s[slot];
    uint16_t *act = (uint16_t *)S->mm.act;
    uint64_t p0 = phs_pcyc();
    for (int b = 0; b < 16; ++b) p4v_i16_to_u16(act + b * 1024, a_cv + b * 256, 256);
    p4v_pack_wt_bias(w_cv, g_p4stage[slot], S->mm.wt, S->mm.bias);
    uint64_t p1 = phs_pcyc(); phs_tr((uint32_t)slot, PHS_S_PREP, p0, p1);
    PHS_ARM(&g_job[slot]);
    uint64_t s0 = phs_pcyc(); PHS_WAIT(&g_job[slot]); PHS_RESET(&g_job[slot]);
    uint64_t s1 = phs_pcyc(); g_pspin[slot] += s1 - s0; phs_tr((uint32_t)slot, PHS_S_SPIN, s0, s1);
    const uint16_t *o = (const uint16_t *)S->mm.out;
    for (int b = 0; b < 16; ++b) p4v_u16_to_i16(out_cv + b * 256, o + b * 1024, 256);
}
#ifndef PHS_NEWTON
#define PHS_NEWTON 2   /* device: 2 iters oc-equal both scales(9.66e-3/4.50e-3); 1 iter degrades 0.25-scale 27% */
#endif
static int p4_diag_thr(int slot, const int16_t *Acv, int16_t *Xcv) {
    p4_slot *S = &g_p4s[slot];
    int16_t *AA = S->scr, *A3 = S->scr + 4096, *M = S->scr + 8192, *Z = S->scr + 12288;
    int32_t *acc = S->acc;
    p4_mm_thr(slot, Acv, Acv, AA);
    p4_mm_thr(slot, AA, Acv, A3);
    p4v_acc3(acc, Acv, AA, A3);
    p4v_acc_diag_add(acc, 32767);
    int eX = p4v_renorm(acc, Xcv);
    p4v_acc_negw(acc, Acv);
    p4v_acc_diag_add(acc, 32767);
    int eM = p4v_renorm(acc, M);
    for (int it = 0; it < PHS_NEWTON; ++it) {
        p4_mm_thr(slot, M, Xcv, Z);
        int e = eM + eX;
        p4v_acc_negw(acc, Z);
        if (e < 31) p4v_acc_diag_add(acc, (int32_t)(65534u >> e));
        int eZ = e + p4v_renorm(acc, Z);
        p4_mm_thr(slot, Xcv, Z, AA);
        p4v_acc_from_cv(acc, AA);
        eX = eX + eZ + p4v_renorm(acc, Xcv);
    }
    return eX;
}
static void p4_producer(void *arg) {
    int slot = (int)(intptr_t)arg;
    p4_slot *S = &g_p4s[slot];
    int hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);
    uint64_t life0 = phs_pcyc();
    for (int h = slot; h < g_p4H; h += g_P) {   /* static interleave (HVXMixHMX-proven) */
        const int16_t *Aq = (const int16_t *)(g_p4Ah + (size_t)h * 131072);
        int eT[16];
        for (int bi = 0; bi < 4; ++bi) for (int bj = 0; bj <= bi; ++bj) {
            static int16_t lin[PHS_NT][4096] __attribute__((aligned(128)));
            for (int r = 0; r < 64; ++r) memcpy(&lin[slot][r * 64], &Aq[(bi * 64 + r) * 256 + bj * 64], 128);
            p4v_perm(S->A[bi * 4 + bj], lin[slot], g_p4il, g_p4stage[slot]);
        }
        for (int b = 0; b < 4; ++b) eT[b * 5] = p4_diag_thr(slot, S->A[b * 5], S->T[b * 5]);
        for (int d = 1; d < 4; ++d)
            for (int i = d; i < 4; ++i) {
                int j = i - d;
#ifndef P4_NOSTACK
#define P4_NOSTACK 1   /* K-stack REFUTED: d>=2 crashes (M=64 envelope unproven); d=1 stack costs oc 9.7e-3->1.05e-2 (loses renorm-up). acc path stays */
#endif
#if P4_NOSTACK
                if (d > P4_NOSTACK - 1) {
                int eAcc = 0;
                p4v_acc_zero(S->acc);
                for (int k = j; k < i; ++k) {
                    p4_mm_thr(slot, S->A[i * 4 + k], S->T[k * 4 + j], S->prod);
                    int ep = eT[k * 4 + j];
                    if (k == j) eAcc = ep;
                    else if (ep > eAcc) { p4v_acc_shr(S->acc, ep - eAcc); eAcc = ep; }
                    int sh = eAcc - ep;
                    if (sh < 31) p4v_acc_addsh(S->acc, S->prod, sh);
                }
                int eC = eAcc + p4v_renorm(S->acc, S->prod);
                p4_mm_thr(slot, S->T[i * 4 + i], S->prod, S->T[i * 4 + j]);
                p4v_acc_from_cv(S->acc, S->T[i * 4 + j]);
                eT[i * 4 + j] = eT[i * 4 + i] + eC + p4v_renorm(S->acc, S->T[i * 4 + j]);
                continue;
                }
                {
                const int16_t *acts[3], *wts[3]; int wsh[3];
                int eC = eT[j * 4 + j];
                for (int k = j; k < i; ++k) if (eT[k * 4 + j] > eC) eC = eT[k * 4 + j];
                for (int k = j; k < i; ++k) { acts[k - j] = S->A[i * 4 + k]; wts[k - j] = S->T[k * 4 + j]; wsh[k - j] = eC - eT[k * 4 + j]; }
                p4_mm_stack(slot, d, acts, wts, wsh, S->prod);
                p4_mm_thr(slot, S->T[i * 4 + i], S->prod, S->T[i * 4 + j]);
                p4v_acc_from_cv(S->acc, S->T[i * 4 + j]);
                eT[i * 4 + j] = eT[i * 4 + i] + eC + p4v_renorm(S->acc, S->T[i * 4 + j]);
                continue;
                }
#else
                const int16_t *acts[3], *wts[3]; int wsh[3];
                int eC = eT[j * 4 + j];
                for (int k = j; k < i; ++k) if (eT[k * 4 + j] > eC) eC = eT[k * 4 + j];
                for (int k = j; k < i; ++k) { acts[k - j] = S->A[i * 4 + k]; wts[k - j] = S->T[k * 4 + j]; wsh[k - j] = eC - eT[k * 4 + j]; }
                p4_mm_stack(slot, d, acts, wts, wsh, S->prod);                 /* e = eC */
                p4_mm_thr(slot, S->T[i * 4 + i], S->prod, S->T[i * 4 + j]);
                p4v_acc_from_cv(S->acc, S->T[i * 4 + j]);
                eT[i * 4 + j] = eT[i * 4 + i] + eC + p4v_renorm(S->acc, S->T[i * 4 + j]);
#endif
            }
        int16_t *To = (int16_t *)(g_p4Th + (size_t)h * 131072);
        memset(To, 0, 131072);
        for (int bi = 0; bi < 4; ++bi) for (int bj = 0; bj <= bi; ++bj) {
            static int16_t lin[PHS_NT][4096] __attribute__((aligned(128)));
            p4v_perm(lin[slot], S->T[bi * 4 + bj], g_p4fl, g_p4stage[slot]);
            for (int r = 0; r < 64; ++r) memcpy(&To[(bi * 64 + r) * 256 + bj * 64], &lin[slot][r * 64], 128);
        }
        memcpy((uint8_t *)To + 128, eT, 64);
    }
    g_plife[slot] = phs_pcyc() - life0;
    if (hvx == 0) qurt_hvx_unlock();
    __atomic_fetch_add(&g_pdone, 1, __ATOMIC_RELEASE);
}

static int p4_threads(const uint8_t *Ah, int P, int H, int *stats, int statsLen, void *T, int TLen) {
    if (TLen < H * 131072) return -2;
    if (P > PHS_NT) P = PHS_NT;
    static int pwr_client; void *pctx = &pwr_client;
    HAP_power_set_core_corner(pctx, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_MAX);
    { HAP_power_request_t r; memset(&r, 0, sizeof(r)); r.type = HAP_power_set_HMX; r.hmx.power_up = TRUE; HAP_power_set(pctx, &r); }
    compute_res_attr_t va; HAP_compute_res_attr_init(&va);
    HAP_compute_res_attr_set_vtcm_param(&va, 0x100000u, 0);
    unsigned int vctx = HAP_compute_res_acquire(&va, 2000000);
    uint8_t *vbase = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&va);
    compute_res_attr_t ha; HAP_compute_res_attr_init(&ha); HAP_compute_res_attr_set_hmx_param(&ha, 1);
    unsigned int hctx = HAP_compute_res_acquire(&ha, 2000000);
    int hl = HAP_compute_res_hmx_lock(hctx);
    if (!vbase || hl != 0) { if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
        if (vctx) HAP_compute_res_release(vctx); return -1; }
    p4_lut_init(); p4_lutc_init();
    g_p4hw = (uint16_t *)(vbase + 0xC8000);
    p4_hwlut_init();
    g_p4il = (uint16_t *)(vbase + 0xCA000);   /* inverse lut: cv idx -> lin byte ofs */
    g_p4fl = (uint16_t *)(vbase + 0xCC000);   /* forward lut: lin idx -> cv byte ofs */
    for (int i = 0; i < 4096; ++i) { g_p4il[g_p4lut[i]] = (uint16_t)(2 * i); g_p4fl[i] = (uint16_t)(2 * g_p4lut[i]); }
    for (int i = 0; i < 4096; ++i) g_p4lut2[i] = g_p4lut[i];
    for (int t = 0; t < P; ++t) {
        g_p4stage[t] = (int16_t *)(vbase + (size_t)t * 0x30000 + 0x28000);
        w16a16_mm_init(&g_p4s[t].mm, vbase + (size_t)t * 0x30000, g_p4s[t].descs);
        {   /* TRUE 64^3 descriptors (4x less HMX than the M=256 carrier; padded 2048B blocks, live 512B) */
            hmx_conv_out_desc_t *od = (hmx_conv_out_desc_t *)g_p4s[t].mm.od;
            od->out_y_stride_words = 64u; od->n_tiles_pow2 = 64u;
        }
        memset(g_p4s[t].mm.out, 0, W16MM_OUT_BYTES);
        uint16_t *s = (uint16_t *)g_p4s[t].mm.act;
        for (int i = 0; i < 256 * 64; ++i) s[i] = 32768;
        g_job[t].state = 0; g_pspin[t] = 0; g_plife[t] = 0;
    }
    {   /* one-shot self-check: HVX pack vs scalar pack must be byte-exact */
        static int16_t wcv[4096]; static uint8_t wA[8192], wB[8192];
        static int32_t bA[256], bB[256]; static int16_t lin[4096];
        for (int i = 0; i < 4096; ++i) wcv[i] = (int16_t)((i * 2654435761u >> 16) & 0xffff);
        for (int i = 0; i < 4096; ++i) if (wcv[i] > 32639) wcv[i] = 32639; else if (wcv[i] < -32639) wcv[i] = -32639;
        int hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);
        p4v_pack_wt_bias(wcv, g_p4stage[0], wA, bA);
        if (hvx == 0) qurt_hvx_unlock();
        p4_pack_wt_bias(wcv, wB, bB, lin);
        int dw = 0, db = 0, fw = -1;
        for (int i = 0; i < 8192; ++i) if (wA[i] != wB[i]) { if (fw < 0) fw = i; ++dw; }
        for (int i = 0; i < 256; ++i) if (bA[i] != bB[i]) ++db;
        FARF(ALWAYS, "P4 PACKCHK wt diff=%d first=%d bias diff=%d", dw, fw, db);
        if (statsLen > 9) stats[9] = dw;
        /* HVX acc helper self-check: roundtrip + acc3 + diag-add vs scalar */
        {
            static int32_t acc[4096]; static int16_t cvA[4096], cvB[4096];
            int hv = qurt_hvx_lock(QURT_HVX_MODE_128B);
            p4v_acc_from_cv(acc, wcv);
            p4v_acc_to_cv(cvA, acc, 0);
            int rt = 0; for (int i = 0; i < 4096; ++i) if (cvA[i] != wcv[i]) ++rt;
            p4v_acc3(acc, wcv, wcv, wcv);
            p4v_acc_diag_add(acc, 5000);
            int a3 = 0;
            for (int i = 0; i < 4096; ++i) {
                int off = i & 63;
                int32_t got = acc[(i >> 6) * 64 + (off & 1) * 32 + (off >> 1)];
                int32_t want = 3 * (int32_t)wcv[i];
                ++a3; --a3;
                int isdiag = 0; for (int d = 0; d < 64; ++d) if (g_p4lut2[d * 65] == i) { isdiag = 1; break; }
                if (got != want + (isdiag ? 5000 : 0)) ++a3;
            }
            if (hv == 0) qurt_hvx_unlock();
            if (statsLen > 10) stats[10] = rt;
            if (statsLen > 11) stats[11] = a3;
        }
    }
    g_p4Ah = Ah; g_p4Th = (uint8_t *)T; g_p4H = H; g_P = P; g_pdone = 0; g_next_head = 0;
    g_sat = 0; g_tr_n = 0; g_cbusy = 0;
    uint64_t us0 = HAP_perf_get_time_us(), t0 = phs_pcyc(); g_tr_base = t0;
    qurt_thread_t tid[PHS_NT];
    for (int t = 0; t < P; ++t) {
        qurt_thread_attr_t a; qurt_thread_attr_init(&a); qurt_thread_attr_set_name(&a, (char *)"p4prod");
        qurt_thread_attr_set_stack_addr(&a, g_stack[t]); qurt_thread_attr_set_stack_size(&a, sizeof(g_stack[t]));
        if (qurt_thread_create(&tid[t], &a, p4_producer, (void *)(intptr_t)t) != QURT_EOK) tid[t] = 0;
    }
    while (g_pdone < P) {
        for (int t = 0; t < P; ++t)
            if (PHS_POLL(&g_job[t])) {
                uint64_t k0 = phs_pcyc();
                w16a16_mm_run(&g_p4s[t].mm);
                uint64_t k1 = phs_pcyc(); g_cbusy += k1 - k0;
                phs_tr((uint32_t)PHS_NT, PHS_S_MM, k0, k1);
                PHS_DONE(&g_job[t]);
            }
    }
    uint64_t t1 = phs_pcyc(), us1 = HAP_perf_get_time_us();
    for (int t = 0; t < P; ++t) { int s; if (tid[t]) qurt_thread_join(tid[t], &s); }
    if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
    if (vctx) HAP_compute_res_release(vctx);
    { HAP_power_request_t off; memset(&off, 0, sizeof(off)); off.type = HAP_power_set_HMX; off.hmx.power_up = FALSE; HAP_power_set(pctx, &off); }
    uint64_t spin = 0; for (int t = 0; t < P; ++t) spin += g_pspin[t];
    if (statsLen > 0) stats[0] = (int)(t1 - t0);
    if (statsLen > 1) stats[1] = P;
    if (statsLen > 2) stats[2] = H;
    if (statsLen > 3) stats[3] = (int)g_cbusy;
    if (statsLen > 4) stats[4] = (int)spin;
    if (statsLen > 6) stats[6] = (int)(us1 - us0);
    if (statsLen > 8) stats[8] = g_sat;
    FARF(ALWAYS, "P4THR P=%d H=%d wall=%llu HMXbusy=%llu spin=%llu", P, H,
         (unsigned long long)(t1 - t0), (unsigned long long)g_cbusy, (unsigned long long)spin);
    if (g_p4trace && TLen >= 24) {   /* trace mode: overwrite T with the event stream (solve data discarded) */
        int n = g_tr_n; if (n > PHS_TR_MAX) n = PHS_TR_MAX;
        if ((int)(24 + (size_t)n * 24) > TLen) n = (TLen - 24) / 24;
        uint32_t *hdr = (uint32_t *)T; hdr[0] = 0x47545203u; hdr[1] = (uint32_t)n;
        ((uint64_t *)(hdr + 2))[0] = t1 - t0; ((uint64_t *)(hdr + 2))[1] = t0;
        uint8_t *pp = (uint8_t *)T + 24;
        for (int e2 = 0; e2 < n; ++e2) { uint32_t *q = (uint32_t *)(pp + (size_t)e2 * 24);
            q[0] = g_tr[e2].tid; q[1] = g_tr[e2].stage;
            ((uint64_t *)(q + 2))[0] = g_tr[e2].t0; ((uint64_t *)(q + 2))[1] = g_tr[e2].t1; }
    }
    return 0;
}

/* ---- mm64-test (H==9): TRUE 64^3 standalone — out blocks STRIDE 2048B(padded), live 512B (m32 0..1).
 * payload A: [0..8K) act u16 64x64, [8K..16K) wt q16. out T: Y 8K + raw out surface 32K. */
static int mm64_test(const uint8_t *A, int *stats, int statsLen, void *T, int TLen) {
    if (TLen < 8192 + 0x8000) return -2;
    static int pwr_client; void *pctx = &pwr_client;
    HAP_power_set_core_corner(pctx, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_MAX);
    { HAP_power_request_t r; memset(&r, 0, sizeof(r)); r.type = HAP_power_set_HMX; r.hmx.power_up = TRUE; HAP_power_set(pctx, &r); }
    compute_res_attr_t va; HAP_compute_res_attr_init(&va);
    HAP_compute_res_attr_set_vtcm_param(&va, 0x100000u, 0);
    unsigned int vctx = HAP_compute_res_acquire(&va, 2000000);
    uint8_t *vbase = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&va);
    compute_res_attr_t ha; HAP_compute_res_attr_init(&ha); HAP_compute_res_attr_set_hmx_param(&ha, 1);
    unsigned int hctx = HAP_compute_res_acquire(&ha, 2000000);
    int hl = HAP_compute_res_hmx_lock(hctx);
    if (!vbase || hl != 0) { if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
        if (vctx) HAP_compute_res_release(vctx); return -1; }
    uint8_t *act = vbase; uint8_t *wt = vbase + 0x8000; int32_t *bias = (int32_t *)(vbase + 0xA000);
    int32_t *atab = (int32_t *)(vbase + 0xA800), *otab = (int32_t *)(vbase + 0xA900);
    uint32_t *ep = (uint32_t *)(vbase + 0xAA00), *mb = (uint32_t *)(vbase + 0xAA40);
    uint8_t *out = vbase + 0xB000;
    for (int rg = 0; rg < 16; ++rg) for (int t = 0; t < 2; ++t) {
        atab[rg * 2 + t] = (int32_t)(uintptr_t)(act + ((rg & 7) * 2 + t) * 2048);   /* padded act blocks */
        otab[rg * 2 + t] = (int32_t)(uintptr_t)(out + ((rg & 7) * 2 + t) * 2048);   /* stride 2048, live 512 */
    }
    static const uint32_t MASK[16] = { 0x0u,0x700u,0x0u,0x77cu,0x0u,0x0u,0x3ffu,0x0u,
                                       0x0u,0x0u,0x0u,0x0u,0x80u,0x0u,0x0u,0x0u };
    ep[0] = 1u; ep[1] = 1536u;
    for (int i = 0; i < 16; ++i) mb[i] = MASK[i];
    mb[14] = (uint32_t)(uintptr_t)ep;
    static uint8_t descs[256] __attribute__((aligned(64)));
    hmx_conv_out_desc_t *od = (hmx_conv_out_desc_t *)descs;
    hmx_conv_act_desc_t *ad = (hmx_conv_act_desc_t *)(descs + 64);
    od->out_tile_ptr_table = otab; od->out_table_stride_dwords = 2u; od->out_y_stride_words = 64u;
    od->n_tiles_pow2 = 64u; od->m_total_minus_step = 1; od->k_total_bytes = 64u;   /* FORMULA: M=64 */
    ad->act_ptr_pairs = atab; ad->n_act_pairs = 2u; ad->act_table_y_stride_words = 128u;
    memset(act, 0, 0x8000u);
    /* pack 64 live rows into PADDED 2048B act blocks (slabs m32 0..1 of each (row4,kt)) */
    {
        const uint16_t *Au = (const uint16_t *)A;
        for (int r4 = 0; r4 < 8; ++r4) for (int kt = 0; kt < 2; ++kt) {
            uint16_t *blk = (uint16_t *)(act + (r4 * 2 + kt) * 2048);
            for (int m32 = 0; m32 < 2; ++m32) for (int rp = 0; rp < 2; ++rp) {
                int r0 = m32 * 32 + r4 * 4 + rp * 2;
                uint16_t *d = blk + (m32 * 2 + rp) * 64;
                for (int c = 0; c < 32; ++c) { d[c * 2] = Au[r0 * 64 + kt * 32 + c]; d[c * 2 + 1] = Au[(r0 + 1) * 64 + kt * 32 + c]; }
            }
        }
    }
    if (0) w16a16_pack_act_crouton16((const uint16_t *)A, (uint16_t *)act, 64, 64);
    w16a16_pack_wt_kmajor((const int16_t *)(A + 8192), wt, 64, 64);
    w16a16_pack_bias((const int16_t *)(A + 8192), bias, 64, 64);
    memset(out, 0, 0x8000u);
    uint64_t k0 = phs_pcyc();
    our_v73deep_kernel_i16(od, ad, wt, (const uint8_t *)bias, (const hmx_conv_mask_desc_t *)mb, ep);
    uint64_t k1 = phs_pcyc();
    /* deblock: 16 blocks stride 2048B, live = slabs m32 0..1 (512B) */
    uint16_t *Y = (uint16_t *)T;
    for (int r4 = 0; r4 < 8; ++r4) for (int nt = 0; nt < 2; ++nt) {
        const uint16_t *blk = (const uint16_t *)(out + (r4 * 2 + nt) * 2048);
        for (int m32 = 0; m32 < 2; ++m32) for (int rp = 0; rp < 2; ++rp) {
            int r0 = m32 * 32 + r4 * 4 + rp * 2;
            const uint16_t *sp = blk + (m32 * 2 + rp) * 64;
            for (int c = 0; c < 32; ++c) {
                Y[r0 * 64 + nt * 32 + c] = sp[c * 2];
                Y[(r0 + 1) * 64 + nt * 32 + c] = sp[c * 2 + 1];
            }
        }
    }
    memcpy((uint8_t *)T + 8192, out, 0x8000u);
    if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
    if (vctx) HAP_compute_res_release(vctx);
    { HAP_power_request_t off; memset(&off, 0, sizeof(off)); off.type = HAP_power_set_HMX; off.hmx.power_up = FALSE; HAP_power_set(pctx, &off); }
    if (statsLen > 3) stats[3] = (int)(k1 - k0);
    FARF(ALWAYS, "MM64 kernel=%llu cyc", (unsigned long long)(k1 - k0));
    return 0;
}

/* ---- mm64-K128 probe (H==10): single M=64 K=128 N=64 matmul, clean buffers.
 * payload A: [0..16K) act u16 64x128, [16K..32K) wt q16 128x64. out T: Y 8K. */
static int mm64_k128(const uint8_t *A, int *stats, int statsLen, void *T, int TLen) {
    if (TLen < 8192) return -2;
    static int pwr_client; void *pctx = &pwr_client;
    HAP_power_set_core_corner(pctx, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_MAX);
    { HAP_power_request_t r; memset(&r, 0, sizeof(r)); r.type = HAP_power_set_HMX; r.hmx.power_up = TRUE; HAP_power_set(pctx, &r); }
    compute_res_attr_t va; HAP_compute_res_attr_init(&va);
    HAP_compute_res_attr_set_vtcm_param(&va, 0x100000u, 0);
    unsigned int vctx = HAP_compute_res_acquire(&va, 2000000);
    uint8_t *vbase = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&va);
    compute_res_attr_t ha; HAP_compute_res_attr_init(&ha); HAP_compute_res_attr_set_hmx_param(&ha, 1);
    unsigned int hctx = HAP_compute_res_acquire(&ha, 2000000);
    int hl = HAP_compute_res_hmx_lock(hctx);
    if (!vbase || hl != 0) { if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
        if (vctx) HAP_compute_res_release(vctx); return -1; }
    uint8_t *act = vbase;                          /* 32 padded blocks x2048 = 64K */
    uint8_t *wt  = vbase + 0x10000;                /* 16K */
    int32_t *bias = (int32_t *)(vbase + 0x14000);
    int32_t *atab = (int32_t *)(vbase + 0x15000), *otab = (int32_t *)(vbase + 0x15800);
    uint32_t *ep = (uint32_t *)(vbase + 0x15C00), *mb = (uint32_t *)(vbase + 0x15C40);
    uint8_t *out = vbase + 0x16000;                /* 32K padded */
    memset(act, 0, 0x10000u); memset(out, 0, 0x8000u);
    const uint16_t *Au = (const uint16_t *)A;      /* 64x128 */
    for (int r4 = 0; r4 < 8; ++r4) for (int kt = 0; kt < 4; ++kt) {
        uint16_t *blk = (uint16_t *)(act + (r4 * 4 + kt) * 2048);
        for (int m32 = 0; m32 < 2; ++m32) for (int rp = 0; rp < 2; ++rp) {
            int r0 = m32 * 32 + r4 * 4 + rp * 2;
            uint16_t *d = blk + (m32 * 2 + rp) * 64;
            for (int c = 0; c < 32; ++c) { d[c*2] = Au[r0*128 + kt*32 + c]; d[c*2+1] = Au[(r0+1)*128 + kt*32 + c]; }
        }
    }
    w16a16_pack_wt_kmajor((const int16_t *)(A + 0x4000), wt, 128, 64);
    w16a16_pack_bias((const int16_t *)(A + 0x4000), bias, 128, 64);
    for (int rg = 0; rg < 16; ++rg) for (int kt = 0; kt < 4; ++kt)
        atab[rg * 4 + kt] = (int32_t)(uintptr_t)(act + ((rg & 7) * 4 + kt) * 2048);
    for (int rg = 0; rg < 16; ++rg) for (int t = 0; t < 2; ++t)
        otab[rg * 2 + t] = (int32_t)(uintptr_t)(out + ((rg & 7) * 2 + t) * 2048);
    static const uint32_t MASK[16] = { 0x0u,0x700u,0x0u,0x77cu,0x0u,0x0u,0x3ffu,0x0u,
                                       0x0u,0x0u,0x0u,0x0u,0x80u,0x0u,0x0u,0x0u };
    ep[0] = 1u; ep[1] = 1536u;
    for (int i = 0; i < 16; ++i) mb[i] = MASK[i];
    mb[14] = (uint32_t)(uintptr_t)ep;
    static uint8_t descs[256] __attribute__((aligned(64)));
    hmx_conv_out_desc_t *od = (hmx_conv_out_desc_t *)descs;
    hmx_conv_act_desc_t *ad = (hmx_conv_act_desc_t *)(descs + 64);
    od->out_tile_ptr_table = otab; od->out_table_stride_dwords = 2u; od->out_y_stride_words = 64u;
    od->n_tiles_pow2 = 64u; od->m_total_minus_step = 1; od->k_total_bytes = 64u;
    ad->act_ptr_pairs = atab; ad->n_act_pairs = 4u; ad->act_table_y_stride_words = 256u;
    uint64_t k0 = phs_pcyc();
    our_v73deep_kernel_i16(od, ad, wt, (const uint8_t *)bias, (const hmx_conv_mask_desc_t *)mb, ep);
    uint64_t k1 = phs_pcyc();
    uint16_t *Y = (uint16_t *)T;
    for (int r4 = 0; r4 < 8; ++r4) for (int nt = 0; nt < 2; ++nt) {
        const uint16_t *blk = (const uint16_t *)(out + (r4 * 2 + nt) * 2048);
        for (int m32 = 0; m32 < 2; ++m32) for (int rp = 0; rp < 2; ++rp) {
            int r0 = m32 * 32 + r4 * 4 + rp * 2;
            const uint16_t *sp = blk + (m32 * 2 + rp) * 64;
            for (int c = 0; c < 32; ++c) { Y[r0*64 + nt*32 + c] = sp[c*2]; Y[(r0+1)*64 + nt*32 + c] = sp[c*2+1]; }
        }
    }
    if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
    if (vctx) HAP_compute_res_release(vctx);
    { HAP_power_request_t off; memset(&off, 0, sizeof(off)); off.type = HAP_power_set_HMX; off.hmx.power_up = FALSE; HAP_power_set(pctx, &off); }
    if (statsLen > 3) stats[3] = (int)(k1 - k0);
    FARF(ALWAYS, "MM64K128 kernel=%llu", (unsigned long long)(k1 - k0));
    return 0;
}

/* ---- Phase-4 floor bench (H==4): steady-state carrier mm + pack-stage costs.
 * stats: [3]=cyc/mm steady (100 back-to-back) [4]=act64 pack [5]=wt pack [7]=bias pack [8]=depack */
static int floor_bench(const uint8_t *A, int *stats, int statsLen, void *T, int TLen) {
    (void)T; (void)TLen;
    static int pwr_client; void *pctx = &pwr_client;
    HAP_power_set_core_corner(pctx, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_MAX);
    { HAP_power_request_t r; memset(&r, 0, sizeof(r)); r.type = HAP_power_set_HMX; r.hmx.power_up = TRUE; HAP_power_set(pctx, &r); }
    compute_res_attr_t va; HAP_compute_res_attr_init(&va);
    HAP_compute_res_attr_set_vtcm_param(&va, 0x100000u, 0);
    unsigned int vctx = HAP_compute_res_acquire(&va, 2000000);
    uint8_t *vbase = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&va);
    compute_res_attr_t ha; HAP_compute_res_attr_init(&ha); HAP_compute_res_attr_set_hmx_param(&ha, 1);
    unsigned int hctx = HAP_compute_res_acquire(&ha, 2000000);
    int hl = HAP_compute_res_hmx_lock(hctx);
    if (!vbase || hl != 0) { if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
        if (vctx) HAP_compute_res_release(vctx); return -1; }
    static uint8_t descs[256] __attribute__((aligned(64)));
    w16a16_mm_t mm; w16a16_mm_init(&mm, vbase, descs);
    memset(mm.out, 0, W16MM_OUT_BYTES);
    static uint16_t Au[256 * 64]; static int16_t Wq[4096]; static uint16_t Y[256 * 64];
    memcpy(Au, A, sizeof(Au)); memcpy(Wq, A + sizeof(Au), sizeof(Wq));
    w16a16_pack_act_crouton16(Au, (uint16_t *)mm.act, 256, 64);
    w16a16_pack_wt_kmajor(Wq, mm.wt, 64, 64);
    w16a16_pack_bias(Wq, mm.bias, 64, 64);
    w16a16_mm_run(&mm);                                  /* warm */
    uint64_t t0 = phs_pcyc();
    for (int r = 0; r < 100; ++r) w16a16_mm_run(&mm);
    uint64_t mmcyc = (phs_pcyc() - t0) / 100;
    t0 = phs_pcyc(); for (int r = 0; r < 10; ++r) w16a16_pack_act_crouton16(Au, (uint16_t *)mm.act, 64, 64);
    uint64_t actc = (phs_pcyc() - t0) / 10;              /* 64-row pack only */
    t0 = phs_pcyc(); for (int r = 0; r < 10; ++r) w16a16_pack_wt_kmajor(Wq, mm.wt, 64, 64);
    uint64_t wtc = (phs_pcyc() - t0) / 10;
    t0 = phs_pcyc(); for (int r = 0; r < 10; ++r) w16a16_pack_bias(Wq, mm.bias, 64, 64);
    uint64_t bic = (phs_pcyc() - t0) / 10;
    t0 = phs_pcyc(); for (int r = 0; r < 10; ++r) w16a16_depack_crouton16((const uint16_t *)mm.out, Y, 256, 64);
    uint64_t dpc = (phs_pcyc() - t0) / 10;
    if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
    if (vctx) HAP_compute_res_release(vctx);
    { HAP_power_request_t off; memset(&off, 0, sizeof(off)); off.type = HAP_power_set_HMX; off.hmx.power_up = FALSE; HAP_power_set(pctx, &off); }
    if (statsLen > 3) stats[3] = (int)mmcyc;
    if (statsLen > 4) stats[4] = (int)actc;
    if (statsLen > 5) stats[5] = (int)wtc;
    if (statsLen > 7) stats[7] = (int)bic;
    if (statsLen > 8) stats[8] = (int)dpc;
    FARF(ALWAYS, "FLOOR: mm=%llu act64=%llu wt=%llu bias=%llu depack=%llu",
         (unsigned long long)mmcyc, (unsigned long long)actc, (unsigned long long)wtc,
         (unsigned long long)bic, (unsigned long long)dpc);
    return 0;
}

/* entry: P producers + 1 main HMX consumer, all w16a16. stats[3]=HMX busy [4]=HVX busy(life-spin)
 * [5]=wall(domain) [6]=us. Tu (>=trace bytes) receives the timeline dump. */
int run(int P, int H, const uint8_t *A, int *stats, int statsLen, void *Tu, int TLen) {
    if (H == 1) return mm_test(A, stats, statsLen, Tu, TLen);   /* Phase-1 primitive correctness mode */
    if (H == 2) return diag_solve(A, stats, statsLen, Tu, TLen); /* Phase-2 diagonal-block inverse */
    if (H == 3) return head_solve(A, stats, statsLen, Tu, TLen); /* Phase-3 full C=256 head */
    if (H == 4) return floor_bench(A, stats, statsLen, Tu, TLen); /* Phase-4 stage-cost floor */
    if (H == 9) return mm64_test(A, stats, statsLen, Tu, TLen);   /* TRUE 64^3 (padded out blocks) */
    if (H == 10) return mm64_k128(A, stats, statsLen, Tu, TLen);  /* M=64 K=128 probe */
    g_p4trace = (H == 33); if (H == 33) H = 32;
    if (H >= 5) return (P >= 2) ? p4_threads(A, P, H, stats, statsLen, Tu, TLen)
                                : p4_head_solve(A, H, stats, statsLen, Tu, TLen); /* Phase-4 */
    if (P > PHS_NT) P = PHS_NT; if (P < 1) P = 1;
    g_P = P; g_H = H; g_pdone = 0; g_next_head = 0; g_cbusy = 0; g_tr_n = 0;
    for (int t = 0; t < P; ++t) { g_job[t].state = 0; g_plife[t] = 0; g_pspin[t] = 0; }

    static int pwr_client; void *pctx = &pwr_client;
    HAP_power_set_core_corner(pctx, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_MAX);
    { HAP_power_request_t r; memset(&r, 0, sizeof(r)); r.type = HAP_power_set_HMX; r.hmx.power_up = TRUE; HAP_power_set(pctx, &r); }

    compute_res_attr_t va; HAP_compute_res_attr_init(&va);
    HAP_compute_res_attr_set_vtcm_param(&va, 0x100000u, 0);            /* 1MB: i16 buffer + producer scratch */
    unsigned int vctx = HAP_compute_res_acquire(&va, 2000000);
    uint8_t *vbase = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&va);
    compute_res_attr_t ha; HAP_compute_res_attr_init(&ha); HAP_compute_res_attr_set_hmx_param(&ha, 1);
    unsigned int hctx = HAP_compute_res_acquire(&ha, 2000000);
    int hl = HAP_compute_res_hmx_lock(hctx);                           /* consumer = main, PURE HMX (no HVX lock) */
    if (!vbase || hl != 0) { FARF(ALWAYS, "PUREHMX: acquire failed vbase=%p hl=%d", vbase, hl);
        if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
        if (vctx) HAP_compute_res_release(vctx); return -1; }

    /* ---- consumer's single valid i16 buffer (validated 64^3 descriptors + word6=0x3ff mask) ---- */
    uint8_t *i16act = vbase + 0x00000u;                /* 8KB crouton16 act */
    int8_t  *i16wt  = (int8_t *)(vbase + 0x10000u);    /* 8KB 4-pass kmajor */
    int32_t *i16bias= (int32_t *)(vbase + 0x20000u);   /* bias record       */
    uint8_t *i16out = vbase + 0x30000u;                /* 64KB out surface  */
    int32_t *i16atab= (int32_t *)(vbase + 0x70000u);
    int32_t *i16otab= (int32_t *)(vbase + 0x70200u);
    for (uint32_t i = 0; i < 0x4000u; ++i) { i16act[i] = (uint8_t)(i * 7 + 1); ((uint8_t *)i16wt)[i] = (uint8_t)(i * 13 + 3); }
    for (uint32_t i = 0; i < 256u; ++i) i16bias[i] = (int32_t)(i * 0x00404420);
    memset(i16out, 0, 0x10000u);                       /* scalar: main holds no HVX lock */
    for (int r4 = 0; r4 < 16; ++r4) for (int tt = 0; tt < 2; ++tt) {
        i16atab[r4 * 2 + tt] = (int32_t)(uintptr_t)(i16act + (((r4 & 7) * 2 + tt) * 512));
        i16otab[r4 * 2 + tt] = (int32_t)(uintptr_t)(i16out + (((r4 & 7) * 2 + tt) * 2048));
    }
    static const uint32_t MASK[16] = { 0x0u,0x700u,0x0u,0x77cu,0x0u,0x0u,0x3ffu,0x0u, 0x0u,0x0u,0x0u,0x0u,0x80u,0x0u,0x0u,0x0u };
    uint32_t ep[2] __attribute__((aligned(16))) = { 1u, 1536u };
    uint32_t mb[16] __attribute__((aligned(16))); for (int i = 0; i < 16; ++i) mb[i] = MASK[i];
    mb[14] = (uint32_t)(uintptr_t)ep;
    hmx_conv_out_desc_t od __attribute__((aligned(64))) = { i16otab, 2u, 64u, 64u, 1, 64u };
    hmx_conv_act_desc_t ad __attribute__((aligned(64))) = { i16atab, 2u, 128u };
    for (int t = 0; t < P; ++t) g_packscratch[t] = vbase + 0x80000u + (size_t)t * 0x2000u;   /* 8KB scratch/producer */

    /* ---- spawn producers + consumer drain ---- */
    uint64_t us0 = HAP_perf_get_time_us();
    uint64_t t0 = phs_pcyc();
    g_tr_base = t0;
    qurt_thread_t tid[PHS_NT];
    for (int t = 0; t < P; ++t) {
        qurt_thread_attr_t a; qurt_thread_attr_init(&a); qurt_thread_attr_set_name(&a, (char *)"phsprod");
        qurt_thread_attr_set_stack_addr(&a, g_stack[t]); qurt_thread_attr_set_stack_size(&a, sizeof(g_stack[t]));
        if (qurt_thread_create(&tid[t], &a, phs_producer, (void *)(intptr_t)t) != QURT_EOK) tid[t] = 0;
    }
    while (g_pdone < P) {
        for (int t = 0; t < P; ++t) {
            if (PHS_POLL(&g_job[t])) {
                uint64_t k0 = phs_pcyc();
                our_v73deep_kernel_i16(&od, &ad, (const uint8_t *)i16wt, (const uint8_t *)i16bias,
                                       (const hmx_conv_mask_desc_t *)mb, ep);
                uint64_t k1 = phs_pcyc(); g_cbusy += k1 - k0;
                phs_tr((uint32_t)PHS_NT, PHS_S_MM, k0, k1);    /* consumer tid = PHS_NT */
                PHS_DONE(&g_job[t]);
            }
        }
    }
    uint64_t t1 = phs_pcyc();
    uint64_t us1 = HAP_perf_get_time_us();
    for (int t = 0; t < P; ++t) { int s; if (tid[t]) qurt_thread_join(tid[t], &s); }
    if (hl == 0) HAP_compute_res_hmx_unlock(hctx); if (hctx) HAP_compute_res_release(hctx);
    if (vctx) HAP_compute_res_release(vctx);
    { HAP_power_request_t off; memset(&off, 0, sizeof(off)); off.type = HAP_power_set_HMX; off.hmx.power_up = FALSE; HAP_power_set(pctx, &off); }

    uint64_t wall = t1 - t0, hvx = 0;
    for (int t = 0; t < P; ++t) hvx += g_plife[t] - g_pspin[t];
    if (statsLen > 0) stats[0] = (int)wall;
    if (statsLen > 1) stats[1] = P;
    if (statsLen > 2) stats[2] = H;
    if (statsLen > 3) stats[3] = (int)g_cbusy;
    if (statsLen > 4) stats[4] = (int)hvx;
    if (statsLen > 5) stats[5] = (int)wall;
    if (statsLen > 6) stats[6] = (int)(us1 - us0);
    FARF(ALWAYS, "PUREHMX %d-head wall=%llu cyc (HMX busy=%llu HVX busy=%llu) %llu us  [60 w16a16 mm/head x%d]",
         H, (unsigned long long)wall, (unsigned long long)g_cbusy, (unsigned long long)hvx,
         (unsigned long long)(us1 - us0), H);

    /* dump trace: [magic][n][wall u64][base u64] then n*{tid,stage,t0,t1} (gdn_pipe_timeline.py format) */
    if (Tu && TLen >= 24) {
        int n = g_tr_n; if (n > PHS_TR_MAX) n = PHS_TR_MAX;
        if ((int)(24 + (size_t)n * 24) > TLen) n = (TLen - 24) / 24;
        uint32_t *hdr = (uint32_t *)Tu; hdr[0] = 0x47545203u; hdr[1] = (uint32_t)n;
        ((uint64_t *)(hdr + 2))[0] = wall; ((uint64_t *)(hdr + 2))[1] = t0;
        uint8_t *p = (uint8_t *)Tu + 24;
        for (int e = 0; e < n; ++e) { uint32_t *q = (uint32_t *)(p + (size_t)e * 24);
            q[0] = g_tr[e].tid; q[1] = g_tr[e].stage;
            ((uint64_t *)(q + 2))[0] = g_tr[e].t0; ((uint64_t *)(q + 2))[1] = g_tr[e].t1; }
        FARF(ALWAYS, "PUREHMX TRACE: %d events, wall=%llu", n, (unsigned long long)wall);
    }
    return 0;
}

}  // namespace pure_hmx
