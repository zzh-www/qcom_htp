/* gdnbm_test.c — host driver: load A.raw, run the bare-metal threaded solve, write T.raw + stats. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gdnbm.h"
#include "remote.h"

static long fsize(const char *p) { FILE *f = fopen(p, "rb"); if (!f) return -1; fseek(f, 0, SEEK_END); long n = ftell(f); fclose(f); return n; }

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);   /* unbuffered: markers survive SIGKILL (for hang localization) */
    if (argc < 10) {
        printf("usage: %s <nthreads> <A.raw> <T.raw> <H> <C> <zpA> <zpT> <sA> <sT> [probe]\n", argv[0]);
        return 2;
    }
    int nthreads = atoi(argv[1]);
    const char *Apath = argv[2], *Tpath = argv[3];
    int H = atoi(argv[4]), C = atoi(argv[5]), zpA = atoi(argv[6]), zpT = atoi(argv[7]);
    union { float f; int i; } sA, sT; sA.f = strtof(argv[8], 0); sT.f = strtof(argv[9], 0);

    struct remote_rpc_control_unsigned_module data; data.domain = CDSP_DOMAIN_ID; data.enable = 1;
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&data, sizeof(data));
    char uri[256]; snprintf(uri, sizeof(uri), "%s%s", gdnbm_URI, "&_dom=cdsp");
    remote_handle64 h = -1;
    int rc = gdnbm_open(uri, &h);
    printf("gdnbm_open rc=0x%x\n", rc); if (rc) return 1;

    if (argc > 10) { /* run the HMX-on-worker probe too */
        int n = 1 + 2 * nthreads, *res = (int *)calloc(n, sizeof(int));
        gdnbm_hmx_probe(h, nthreads, res, n);
        for (int i = 0; i < nthreads; ++i) printf("  probe worker %d: hmx ctx=%d sent=0x%X\n", i, res[1 + 2 * i], res[1 + 2 * i + 1]);
        free(res);
    }

    long abytes = (long)H * C * C * 2;
    if (fsize(Apath) < abytes) { printf("A.raw too small (%ld < %ld)\n", fsize(Apath), abytes); return 3; }
    unsigned char *A = (unsigned char *)malloc(abytes), *T = (unsigned char *)calloc(abytes, 1);
    FILE *fa = fopen(Apath, "rb"); fread(A, 1, abytes, fa); fclose(fa);

    int stats[20] = {0};
    /* REPS: run the solve N times in ONE FastRPC session (env GDNBM_REPS, default 1). Repeated single-shot
     * processes churn the unsigned-PD FastRPC session and OCCASIONALLY deadlock the cDSP; looping the
     * remote call on the SAME handle does the steady-state measurement with NO session churn. */
    const char *re = getenv("GDNBM_REPS"); int reps = re ? atoi(re) : 1; if (reps < 1) reps = 1;
    for (int r = 0; r < reps; ++r) {
        rc = gdnbm_solve(h, A, (int)abytes, H, C, zpA, zpT, sA.i, sT.i, nthreads, T, (int)abytes, stats, 20);
        printf("gdnbm_solve rc=0x%x  P=%d  H=%d  rep=%d/%d\n", rc, stats[1], stats[2], r + 1, reps);
        if (rc) break;
    }
    /* ===== cycle report — SELF-LABELED 口径 (skill htp-cycle-metric). The ONLY cross-impl number is
     * graph-wall÷N_mm (口径①). NEVER compare optrace per-op latency/busy (口径②③) to a wall (口径④).
     * A latency-floor fact (~256/mm) is NOT a gate on a wall measurement: a per-call wall ≫300 is the
     * CORRECT feed-inclusive cost, not a "口径 error". (Hardened cron#28 after the cross-口径 churn.) ===== */
    if (rc == 0) {
        int H_ = stats[2] ? stats[2] : 1;
        int nmm = H_ * 24;                       /* 24 mm/head @ Newton=0 (8 diag + 16 merge); label notes it */
        double cpu = stats[6] ? (double)stats[0] / (double)stats[6] : 0.0;   /* PCYCLE/µs clock self-check */
        printf("  ① graph-wall (口径①, THE verdict = 32-head TOTAL) = %d cyc\n", stats[0]);
        printf("     clock self-check wall/us = %d/%d = %.1f  (TURBO ~1594; >> => wrong counter, re-measure)\n",
               stats[0], stats[6], cpu);
        printf("     cross-impl compare ONLY via graph-wall/N_mm (口径①) = %d/%d = %d cyc/mm (N_mm=24/head @Newton0)\n",
               stats[0], nmm, stats[0] / nmm);
        printf("     native anchors (same 口径①): single 64^3 = 11034 wall ;  128-batch = 2020/mm  <- targets\n");
        printf("  ④ per-call kernel wall (口径④, single 64^3 resident) = %d cyc  (= matmul + mxmem feed; >300 is CORRECT, not a bug)\n",
               stats[5]);
        printf("  ② matmul latency floor (口径②) = ~256 cyc/mm (native int16 dominant-path; what HMX does, NOT a wall target)\n");
        printf("     !! NEVER compare optrace per-op (num_dominant_path_cycles / by_htp_type busy = 口径②③) to ④ or graph-wall.\n");
        printf("  consumer HMX-busy Σ(口径④)=%d  feed Σ: wt-pack=%d scatter=%d renorm/acc=%d  slowest-prod-life=%d  PACKCHK=%d(0=ok)\n",
               stats[3], stats[7], stats[9], stats[10], stats[11], stats[8]);
    }
    printf("  raw stats[0..11]: %d %d %d %d %d %d %d %d %d %d %d %d\n",
           stats[0], stats[1], stats[2], stats[3], stats[4], stats[5], stats[6], stats[7], stats[8], stats[9], stats[10], stats[11]);
    printf("  raw stats[12..19]: %d %d %d %d %d %d %d %d\n",
           stats[12], stats[13], stats[14], stats[15], stats[16], stats[17], stats[18], stats[19]);

    FILE *ft = fopen(Tpath, "wb"); fwrite(T, 1, abytes, ft); fclose(ft);
    printf("wrote %s (%ld bytes)\n", Tpath, abytes);
    gdnbm_close(h); free(A); free(T);
    return rc;
}
