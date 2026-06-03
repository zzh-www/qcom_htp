/* gdnbm_imp.c — DSP-side bare-metal FastRPC HMX-threading probe.
 * Tests whether a self-spawned qurt worker, in OUR OWN HAP/PD, can acquire HMX via the proper
 * HAP_compute_res_hmx_lock3 API (where the raw qurt_hmx_lock faulted inside QNN's managed PD). */
#include "gdnbm.h"
#include "HAP_compute_res.h"
#include "HAP_farf.h"
#include "qurt.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

int gdnbm_open(const char *uri, remote_handle64 *h) { (void)uri; *h = 1; return 0; }
int gdnbm_close(remote_handle64 h) { (void)h; return 0; }

#define MAXW 4
static unsigned int g_ctx;
static volatile int g_rc[MAXW], g_sent[MAXW];
static char __attribute__((aligned(128))) g_stack[MAXW][16384];

static void worker(void *arg) {
    int id = (int)(long)arg;
    int hvx = qurt_hvx_lock(QURT_HVX_MODE_128B);
    /* lock3 is NOT_SUPPORTED on this device -> use the older model: each worker acquires its OWN
     * HMX context (v75 has 2 HMX units, so up to 2 workers can hold HMX concurrently). */
    compute_res_attr_t attr; HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);
    unsigned int ctx = HAP_compute_res_acquire(&attr, 2000000);
    g_rc[id] = (int)ctx;                       /* reuse: 0 = acquire failed, else the context id */
    if (ctx) {
        HVX_Vector v = Q6_V_vsplat_R(0x1000 + id);
        int lanes[32] __attribute__((aligned(128)));
        *(HVX_Vector *)lanes = v;
        g_sent[id] = lanes[0];
        HAP_compute_res_release(ctx);
    } else {
        g_sent[id] = 0xBAD;
    }
    if (hvx == 0) qurt_hvx_unlock();
}

int gdnbm_hmx_probe(remote_handle64 _h, int nworkers, int *results, int resultsLen) {
    (void)_h;
    if (nworkers > MAXW) nworkers = MAXW;
    if (nworkers < 1) nworkers = 1;
    /* main does NOT acquire HMX (would consume 1 of the 2 units); each worker acquires its own. */
    if (resultsLen > 0) results[0] = 1;          /* marker: probe ran */
    FARF(ALWAYS, "gdnbm: nworkers=%d (each worker acquires its own HMX context)", nworkers);
    qurt_thread_t tid[MAXW];
    for (int i = 0; i < nworkers; ++i) {
        g_rc[i] = -999; g_sent[i] = 0;
        qurt_thread_attr_t a; qurt_thread_attr_init(&a);
        qurt_thread_attr_set_name(&a, (char *)"gdnbm_wk");
        qurt_thread_attr_set_stack_addr(&a, g_stack[i]);
        qurt_thread_attr_set_stack_size(&a, sizeof(g_stack[i]));
        if (qurt_thread_create(&tid[i], &a, worker, (void *)(long)i) != QURT_EOK) { worker((void *)(long)i); tid[i] = 0; }
    }
    for (int i = 0; i < nworkers; ++i) { int s; if (tid[i]) qurt_thread_join(tid[i], &s); }
    for (int i = 0; i < nworkers; ++i) {
        if (1 + 2 * i + 1 < resultsLen) { results[1 + 2 * i] = g_rc[i]; results[1 + 2 * i + 1] = g_sent[i]; }
        FARF(ALWAYS, "gdnbm: worker %d hmx_lock3 rc=%d sentinel=0x%X", i, g_rc[i], g_sent[i]);
    }
    HAP_compute_res_release(g_ctx);
    return 0;
}
