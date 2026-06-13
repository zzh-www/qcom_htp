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
        printf("gdnbm_solve rc=0x%x  wall=%d cyc  nthreads=%d  heads=%d\n", rc, stats[0], stats[1], stats[2]);
        if (rc == 0 && stats[2] > 0)
            printf("  >>> %d cyc/head (%d-thread)\n", stats[0] / stats[2], stats[1]);
        if (rc) break;
    }
    printf("  stats: [0]=%d [1]=%d [2]=%d [3]=%d [4]=%d\n", stats[0], stats[1], stats[2], stats[3], stats[4]);
    printf("  stats: [5]=%d [6]=%d [7]=%d [8]=%d [9]=%d [10]=%d [11]=%d\n", stats[5], stats[6], stats[7], stats[8], stats[9], stats[10], stats[11]);
    printf("  O5 scatter split: memcpy=%d gp_perm=%d memset=%d  (sum=%d vs scatter[9]=%d)\n",
           stats[12], stats[13], stats[14], stats[12]+stats[13]+stats[14], stats[9]);
    printf("  O6b compact-64 test: maxdiff=%d nonzero=%d cyc=%d  (vs padded bench %d)\n",
           stats[15], stats[17], stats[16], stats[5]);
    if (stats[3] || stats[5] || stats[7]) {  /* PROBE_CYCLES per-stage (cyc/head) */
        int sum = stats[3]+stats[4]+stats[5]+stats[6]+stats[7]+stats[8]+stats[9];
        printf("  PROBE cyc/head: diag=%d zero=%d fold=%d quant=%d mm=%d acc=%d requant=%d  SUM=%d\n",
               stats[3], stats[4], stats[5], stats[6], stats[7], stats[8], stats[9], sum);
    }

    FILE *ft = fopen(Tpath, "wb"); fwrite(T, 1, abytes, ft); fclose(ft);
    printf("wrote %s (%ld bytes)\n", Tpath, abytes);
    gdnbm_close(h); free(A); free(T);
    return rc;
}
