/* gdnbm_imp.c — bare-metal FastRPC GDN block-recursive solve (escapes QNN; self-managed HVX/HMX/VTCM).
 * Reuses the EXACT validated device solve from the QNN op via the GDN_BR_NO_QHPI include guard. */
#include "gdnbm.h"
#include "HAP_compute_res.h"
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
#include "../../solve_br_op/src/GdnSolveBROp.cpp"

int gdnbm_open(const char *uri, remote_handle64 *h) { (void)uri; *h = 1; return 0; }
int gdnbm_close(remote_handle64 h) { (void)h; return 0; }

static inline uint64_t pcyc(void) { uint64_t c; asm volatile("%0 = C15:14" : "=r"(c)); return c; }
static inline float i2f(int b) { union { int i; float f; } u; u.i = b; return u.f; }

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
/* HAP worker: own HVX context + own HMX (HAP_compute_res acquire, the only model this device supports),
 * then run this slot's heads through the reused gdn_br_run_slot (HVX diagonals + HMX merges). */
static void solve_worker(void *arg) {
    gdn_work_t *w = (gdn_work_t *)arg;
    int hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);
    compute_res_attr_t a; HAP_compute_res_attr_init(&a); HAP_compute_res_attr_set_hmx_param(&a, 1);
    unsigned int ctx = HAP_compute_res_acquire(&a, 2000000);
    gdn_br_run_slot(w);
    if (ctx) HAP_compute_res_release(ctx);
    if (hvx == 0) qurt_hvx_unlock();
}

int gdnbm_solve(remote_handle64 _h, const uint8_t *A, int ALen, int H, int C, int zpA, int zpT,
                int sA_bits, int sT_bits, int nthreads, uint8_t *T, int TLen, int *stats, int statsLen) {
    (void)_h;
    if (C != GDN_BR_C) return -1;
    if (nthreads > GDN_BR_NT) nthreads = GDN_BR_NT; if (nthreads < 1) nthreads = 1;
    if (H < nthreads) nthreads = H;
    const uint16_t *Au = (const uint16_t *)A;
    uint16_t *Tu = (uint16_t *)T;
    (void)ALen; (void)TLen;
    float sA = i2f(sA_bits), sT = i2f(sT_bits);
    int M, S; gdn_fold_MS(sA, &M, &S);

    /* DEBUG MODE (nthreads<=1): run the whole solve INLINE on the main callback thread, with ONE
     * acquire_cached granting BOTH HMX and VTCM to this thread.  Isolates the solve (VTCM + mxmem) from
     * the worker-threading.  acquire_cached is required for get_vtcm_ptr to return a valid pointer. */
    compute_res_attr_t va; HAP_compute_res_attr_init(&va);
    HAP_compute_res_attr_set_hmx_param(&va, 1);
    HAP_compute_res_attr_set_vtcm_param(&va, (unsigned)((nthreads < 1 ? 1 : nthreads)) * 0x60000u, 0);
    int hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);
    unsigned int vctx = HAP_compute_res_acquire(&va, 2000000);
    uint8_t *vtcm = (uint8_t *)HAP_compute_res_attr_get_vtcm_ptr(&va);
    FARF(ALWAYS, "gdnbm_solve: H=%d C=%d nthreads=%d hvx=%d vctx=%u vtcm=%p", H, C, nthreads, hvx, vctx, vtcm);
    if (!vtcm) { if (vctx) HAP_compute_res_release(vctx); if (hvx == 0) qurt_hvx_unlock(); return -2; }

    uint64_t t0 = pcyc();
    if (nthreads <= 1) {
        gdn_work_t w = { 0, 0u, (uint32_t)H, 1u, Au, Tu, zpA, M, S, sT, zpT, vtcm };
        gdn_br_run_slot(&w);                 /* inline on main (holds HMX+VTCM) */
    } else {
        gdn_work_t work[GDN_BR_NT]; qurt_thread_t tid[GDN_BR_NT];
        for (int t = 0; t < nthreads; ++t) {
            work[t] = (gdn_work_t){ t, 0u, (uint32_t)H, (uint32_t)nthreads, Au, Tu, zpA, M, S, sT, zpT, vtcm };
            qurt_thread_attr_t a; qurt_thread_attr_init(&a); qurt_thread_attr_set_name(&a, (char *)"gdnsolve");
            qurt_thread_attr_set_stack_addr(&a, g_solve_stack[t]); qurt_thread_attr_set_stack_size(&a, sizeof(g_solve_stack[t]));
            if (qurt_thread_create(&tid[t], &a, solve_worker, &work[t]) != QURT_EOK) { solve_worker(&work[t]); tid[t] = 0; }
        }
        for (int t = 0; t < nthreads; ++t) { int s; if (tid[t]) qurt_thread_join(tid[t], &s); }
    }
    uint64_t t1 = pcyc();
    if (vctx) HAP_compute_res_release(vctx);
    if (hvx == 0) qurt_hvx_unlock();

    if (statsLen > 0) stats[0] = (int)(t1 - t0);
    if (statsLen > 1) stats[1] = nthreads;
    if (statsLen > 2) stats[2] = H;
    FARF(ALWAYS, "gdnbm_solve: wall=%llu cyc for %d heads / %d threads", (unsigned long long)(t1 - t0), H, nthreads);
    return 0;
}
