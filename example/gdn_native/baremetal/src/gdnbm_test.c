/* gdnbm_test.c — host (Android aarch64) driver for the bare-metal HMX-threading probe. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gdnbm.h"
#include "remote.h"

int main(int argc, char **argv) {
    int nworkers = (argc > 1) ? atoi(argv[1]) : 2;

    /* request an UNSIGNED PD on cdsp (matches the device's QNN unsigned-PD config). */
    struct remote_rpc_control_unsigned_module data;
    data.domain = CDSP_DOMAIN_ID;
    data.enable = 1;
    int nErr = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&data, sizeof(data));
    printf("remote_session_control(UNSIGNED) rc=0x%x\n", nErr);

    char uri[256];
    snprintf(uri, sizeof(uri), "%s%s", gdnbm_URI, "&_dom=cdsp");
    remote_handle64 h = -1;
    int rc = gdnbm_open(uri, &h);
    printf("gdnbm_open(\"%s\") rc=0x%x h=%lld\n", uri, rc, (long long)h);
    if (rc) return 1;

    int n = 1 + 2 * nworkers;
    int *results = (int *)calloc(n, sizeof(int));
    rc = gdnbm_hmx_probe(h, nworkers, results, n);
    printf("gdnbm_hmx_probe rc=0x%x\n", rc);
    for (int i = 0; i < nworkers; ++i)
        printf("  worker %d: HMX acquire ctx=%d (0=fail)  HVX sentinel=0x%X\n",
               i, results[1 + 2 * i], results[1 + 2 * i + 1]);
    int ok = 1;
    for (int i = 0; i < nworkers; ++i) if (results[1 + 2 * i] == 0) ok = 0;
    printf(ok ? "RESULT: HMX-on-worker WORKS (all workers got HMX via hmx_lock3)\n"
              : "RESULT: HMX-on-worker BLOCKED/failed\n");
    gdnbm_close(h);
    free(results);
    return 0;
}
