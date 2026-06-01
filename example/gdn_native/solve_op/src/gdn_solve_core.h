/* gdn_solve_core.h - pure-math GDN triangular solve (no QHPI deps); shared by the QHPI op
 * and the host test so the device kernel math is validated on host bit-for-bit vs golden.
 *
 * INTERNAL arithmetic is int16 at a FIXED fine scale (GDN_INT_SCALE) so the diagonal forward
 * substitution stays near-exact (T relerr ~3.6e-5).  The OUTPUT is quantized to the graph's T
 * scale/bitwidth (sTout, maxc) — int8 for the U/W consume (the inherent int8-in[1] ceiling).
 * "int16 internal compute, downstream-bitwidth output" — equivalent to int16-out + immediate cast
 * since T is only ever consumed as int8 in[1]. */
#ifndef GDN_SOLVE_CORE_H
#define GDN_SOLVE_CORE_H
#include <cstdint>
#include <cmath>

#define GDN_BL 16                       /* logical block (KDA sub-chunk BC); diag inverted directly */
#define GDN_CMAX 64                     /* GDN chunk size C (matmuls 64x64) */
#define GDN_INT_SCALE (2.0f/32767.0f)   /* fixed fine internal int16 scale (|T|<2) */

static inline int16_t gdn_clip16(float v) {
    long r = lroundf(v);
    if (r >  32767) r =  32767;
    if (r < -32767) r = -32767;
    return (int16_t)r;
}

/* one head: A int16 codes [C*C] (scale sA, strictly-lower) -> T codes [C*C] at scale sTout,
 * clipped to +-maxc (127 for int8 out, 32767 for int16).  OUT is int8_t or int16_t.
 * C must be a multiple of GDN_BL and <= GDN_CMAX. */
template <typename OUT>
static void gdn_solve_head_q(const int16_t *A, int C, float sA, float sTout, float maxc, OUT *T) {
    const int BL = GDN_BL, nb = C / BL;
    const float sTi = GDN_INT_SCALE, inv_sTi = 1.0f / sTi, inv_sTout = 1.0f / sTout;
    static float Tf[GDN_CMAX * GDN_CMAX];          /* real T; static (BSS, not stack) — MAIN single-thread */

    /* --- diagonal block inverses via forward substitution (int16 codes @ sTi, int32 accum) --- */
    for (int b = 0; b < nb; ++b) {
        int16_t Tb[GDN_BL * GDN_BL];
        for (int i = 0; i < BL; ++i) {
            for (int j = 0; j < BL; ++j) {
                int32_t acc = 0;                   /* sum int16*int16 -> int32 (scale sA*sTi) */
                for (int k = 0; k < i; ++k)
                    acc += (int32_t)A[(b*BL+i)*C + (b*BL+k)] * (int32_t)Tb[k*BL + j];
                float ei = (j == i) ? inv_sTi : 0.0f;            /* identity (exact) */
                Tb[i*BL + j] = gdn_clip16(ei + (float)acc * sA); /* code = round(ei + acc*sA) @ sTi */
            }
        }
        for (int r = 0; r < BL; ++r)               /* dequant diagonal block into Tf */
            for (int c = 0; c < BL; ++c)
                Tf[(b*BL+r)*C + (b*BL+c)] = (float)Tb[r*BL + c] * sTi;
    }

    /* --- block-triangular merge: T_ij = T_ii @ sum_{k=j..i-1} A_ik T_kj  (i>j), in fp --- */
    for (int j = 0; j < nb; ++j) {
        for (int i = j + 1; i < nb; ++i) {
            float acc[GDN_BL * GDN_BL];
            for (int t = 0; t < BL*BL; ++t) acc[t] = 0.0f;
            for (int k = j; k < i; ++k)            /* acc += A_ik @ Tf[k][j] */
                for (int r = 0; r < BL; ++r)
                    for (int c = 0; c < BL; ++c) {
                        float s = 0.0f;
                        for (int m = 0; m < BL; ++m)
                            s += (float)A[(i*BL+r)*C + (k*BL+m)] * sA * Tf[(k*BL+m)*C + (j*BL+c)];
                        acc[r*BL + c] += s;
                    }
            for (int r = 0; r < BL; ++r)           /* Tf[i][j] = T_ii @ acc */
                for (int c = 0; c < BL; ++c) {
                    float s = 0.0f;
                    for (int m = 0; m < BL; ++m)
                        s += Tf[(i*BL+r)*C + (i*BL+m)] * acc[m*BL + c];
                    Tf[(i*BL+r)*C + (j*BL+c)] = s;
                }
        }
    }

    /* --- scatter to output codes @ sTout; strict-upper stays zero (T is lower-triangular) --- */
    for (int r = 0; r < C; ++r)
        for (int c = 0; c < C; ++c) {
            if (c > r) { T[r*C + c] = (OUT)0; continue; }
            long q = lroundf(Tf[r*C + c] * inv_sTout);
            if (q >  (long)maxc) q =  (long)maxc;
            if (q < -(long)maxc) q = -(long)maxc;
            T[r*C + c] = (OUT)q;
        }
}

#endif /* GDN_SOLVE_CORE_H */
