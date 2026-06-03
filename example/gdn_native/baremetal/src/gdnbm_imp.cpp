/* gdnbm_imp.c — bare-metal FastRPC GDN block-recursive solve (escapes QNN; self-managed HVX/HMX/VTCM).
 * Reuses the EXACT validated device solve from the QNN op via the GDN_BR_NO_QHPI include guard. */
#include "gdnbm.h"
#include "HAP_compute_res.h"
#include "HAP_vtcm_mgr.h"
#include "HAP_power.h"
#include "HAP_farf.h"
#include "qurt.h"
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
/* HVX-MERGE mode: HMX is process-serial (mxmem under compute_resource_hmx_lock can't thread — verified),
 * so replace the HMX merges with int16-HVX matmul merges -> pure HVX + BSS scratch, worker threads
 * parallelize freely.  No HMX / VTCM / mxmem / power-HMX needed. */
#define GDN_BR_HVX_MERGE 1
#include "../../solve_br_op/src/GdnSolveBROp.cpp"

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
/* HAP worker (HVX-merge mode): own HVX context, run this slot's heads.  No HMX/VTCM — the int16-HVX
 * merges use only HVX + BSS scratch, so workers parallelize with no shared lock. */
static void solve_worker(void *arg) {
    gdn_work_t *w = (gdn_work_t *)arg;
    int hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);
#if defined(GDNBM_VTCM_RESIDENT) && !defined(GDNBM_MM_TEST) && !defined(GDNBM_Q_TEST) && !defined(GDNBM_MERGE_TEST)
    /* SKILL principles: (2) keep A resident in VTCM (not uncached FastRPC DDR -> bare-metal diag was
     * 373K vs QNN 48K), (3) UDMA ping-pong so head h+1's A loads while head h computes. T writes go
     * straight to DDR (sequential, L2-prefetch friendly).
     * VTCM is acquired ONCE on the main thread (gdnbm_solve) and shared; each worker uses its own
     * 0x60000 slice via w->vtcm_base (per-worker HAP_compute_res_acquire SERIALIZES the workers — the
     * resource manager grants VTCM per-context, so concurrent acquires block on each other). */
    uint8_t *vtcm = w->vtcm_base ? (w->vtcm_base + (size_t)w->slot * 0x60000) : nullptr;
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
        if (n > 0) udma_start(&dsc, Avt[0], w->Au + (size_t)hs[0] * CC, Abytes);
        for (int i = 0; i < n; ++i) {
            udma_wait();                                    /* cur A ready */
            if (i + 1 < n) udma_start(&dsc, Avt[(i + 1) & 1],
                                      w->Au + (size_t)hs[i + 1] * CC, Abytes);  /* prefetch next */
            gdn_br_one_head(sc, &vt, Avt[i & 1], w->Tu + (size_t)hs[i] * CC,
                            w->zpA, w->M, w->S, w->sT, w->zpT);   /* reads A from VTCM */
        }
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
#elif defined(GDNBM_Q_TEST)
    /* quant isolation: codes[i]=i-2048 @ scale 1.0 -> out[i] ~ clamp(i-2048, +-2047). Writes int16->int32 to Tu. */
    { gdn_scr_t *sc = &g_scr[w->slot];
      for (int i = 0; i < 64 * 64; ++i) sc->Tc[i] = i - 2048;
      gdn_quant_i12_from_codes(sc, sc->Tc, 1.0f, sc->a16, -1);
      for (int i = 0; i < 64 * 64; ++i) ((int32_t *)w->Tu)[i] = sc->a16[i]; }
#else
    gdn_br_run_slot(w);                 /* vtcm_base unused in HVX-merge mode */
#endif
    if (hvx == 0) qurt_hvx_unlock();
}

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

    /* POWER: vote turbo core clock + HMX power-on (the bare HMX acquire doesn't power/clock it -> mxmem hangs). */
    static int g_pwr_client; void *pctx = &g_pwr_client;
    HAP_power_set_core_corner(pctx, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_TURBO, HAP_DCVS_VCORNER_MAX);
    { HAP_power_request_t r; memset(&r,0,sizeof(r)); r.type=HAP_power_set_HMX; r.hmx.power_up=TRUE; HAP_power_set(pctx,&r); }

    /* VTCM is acquired ONCE here on the main thread and the base pointer shared with all workers (each
     * takes a 0x60000 slice via w->vtcm_base).  Per-worker acquire serializes the workers — the resource
     * manager grants VTCM per-context so concurrent acquires block on each other (canonical pattern:
     * Hexagon SDK HAP_compute_res + tutorial ch04 demo_vtcm_alloc = acquire once, bump-allocate). */
    uint8_t *vtcm_base = nullptr; unsigned int vctx = 0;
#if defined(GDNBM_VTCM_RESIDENT)
    { compute_res_attr_t va; HAP_compute_res_attr_init(&va);
      HAP_compute_res_attr_set_vtcm_param(&va, (unsigned)GDN_BR_NT * 0x60000u, 0);
      vctx = HAP_compute_res_acquire(&va, 2000000);
      vtcm_base = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&va); }
#endif
    uint64_t t0 = pcyc();
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
    if (vctx) HAP_compute_res_release(vctx);
    { HAP_power_request_t off; memset(&off,0,sizeof(off)); off.type=HAP_power_set_HMX; off.hmx.power_up=FALSE; HAP_power_set(pctx,&off); }

    if (statsLen > 0) stats[0] = (int)(t1 - t0);
    if (statsLen > 1) stats[1] = nthreads;
    if (statsLen > 2) stats[2] = H;
    FARF(ALWAYS, "gdnbm_solve: wall=%llu cyc / %d heads / %d threads", (unsigned long long)(t1-t0), H, nthreads);
    return 0;
}
