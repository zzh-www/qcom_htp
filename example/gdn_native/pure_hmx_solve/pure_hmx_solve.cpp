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

/* one w16a16 matmul on codes: out = a@w, e_out = e_a + e_w (codes = prod/32767 by the drain). */
static uint64_t g_ds_mmcyc;
static void ds_mm(w16a16_mm_t *mm, const int16_t *a, const int16_t *w, int16_t *out) {
    for (int i = 0; i < 4096; ++i) g_dsAct[i] = (uint16_t)(32768 + a[i]);
    for (int i = 4096; i < 256 * 64; ++i) g_dsAct[i] = 32768;
    static uint16_t Y[256 * 64];
    uint64_t c0 = phs_pcyc();
    w16a16_mm(mm, g_dsAct, w, Y);
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

/* entry: P producers + 1 main HMX consumer, all w16a16. stats[3]=HMX busy [4]=HVX busy(life-spin)
 * [5]=wall(domain) [6]=us. Tu (>=trace bytes) receives the timeline dump. */
int run(int P, int H, const uint8_t *A, int *stats, int statsLen, void *Tu, int TLen) {
    if (H == 1) return mm_test(A, stats, statsLen, Tu, TLen);   /* Phase-1 primitive correctness mode */
    if (H == 2) return diag_solve(A, stats, statsLen, Tu, TLen); /* Phase-2 diagonal-block inverse */
    if (H == 3) return head_solve(A, stats, statsLen, Tu, TLen); /* Phase-3 full C=256 head */
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
