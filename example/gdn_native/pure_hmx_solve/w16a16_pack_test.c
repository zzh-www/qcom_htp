/* Host test: REAL C w16a16 packers (w16a16_pack.h) must be BYTE-IDENTICAL to the proven Python packers.
 * Reads the 64^3 ground-truth (Python) prepared_state + the raw A/q16, packs in C, compares byte-exact.
 *   cc -O2 w16a16_pack_test.c -o /tmp/w16pt && /tmp/w16pt */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "w16a16_pack.h"

static long rdfile(const char *p, void *buf, long max) {
    FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "open fail %s\n", p); exit(2); }
    long n = (long)fread(buf, 1, max, f); fclose(f); return n;
}
static int cmp(const char *name, const uint8_t *a, const uint8_t *b, long n) {
    long d = 0, first = -1;
    for (long i = 0; i < n; ++i) if (a[i] != b[i]) { if (first < 0) first = i; d++; }
    printf("  %-16s %s  diff=%ld/%ld%s\n", name, d == 0 ? "BYTE-EXACT" : "MISMATCH", d, n,
           first >= 0 ? "" : "");
    if (d) printf("      first diff @%ld: C=0x%02x py=0x%02x\n", first, a[first], b[first]);
    return d == 0;
}

int main(void) {
    enum { M = 64, K = 64, N = 64 };
    static uint16_t A[M * K];
    static int16_t  q16[K * N];
    static uint16_t c_act[M * K];
    static uint8_t  c_wt[K * N * 2];       /* 4-pass dilated: 2x bytes (64x64 -> 8192) */
    static int32_t  c_bias[64 * 8];        /* N/16 groups * 64 int32 (64x64: 4 groups -> 256) */
    static uint8_t  py_act[M * K * 2];
    static uint8_t  py_wt[K * N * 2];
    static uint8_t  py_bias[64 * 8 * 4];

    rdfile("/tmp/w16_64_prep/prepared_state/activation_source.raw", A, sizeof(A));
    rdfile("/tmp/w16_64_q16.raw", q16, sizeof(q16));
    long la = rdfile("/tmp/w16_64_prep/prepared_state/activation.raw", py_act, sizeof(py_act));
    long lw = rdfile("/tmp/w16_64_prep/prepared_state/packed_weight.raw", py_wt, sizeof(py_wt));
    long lb = rdfile("/tmp/w16_64_prep/prepared_state/folded_bias.raw", py_bias, sizeof(py_bias));

    w16a16_pack_act_crouton16(A, c_act, M, K);
    w16a16_pack_wt_kmajor(q16, c_wt, K, N);
    w16a16_pack_bias(q16, c_bias, K, N);

    printf("w16a16 C-packer vs Python ground-truth (64x64x64):\n");
    int ok = 1;
    ok &= cmp("activation",     (uint8_t *)c_act,  py_act, la);
    ok &= cmp("packed_weight",  c_wt,              py_wt,  lw);
    ok &= cmp("folded_bias",    (uint8_t *)c_bias, py_bias, lb);
    printf("%s\n", ok ? "ALL BYTE-EXACT -> C packers are correct" : "FAIL");
    return ok ? 0 : 1;
}
