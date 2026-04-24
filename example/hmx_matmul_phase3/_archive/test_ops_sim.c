/*
 * test_ops_sim.c — hexagon-sim standalone harness for the 4 Phase 3B HVX ops.
 *
 * Links directly against the kernel .c files (bypassing QHPI registration).
 * Calls each op's body function with small fixed inputs and compares to a
 * scalar reference inlined here.
 *
 * Build + run:
 *   hexagon-clang -O2 -mhvx -mhvx-length=128B -mv75 \
 *       -I $QNN_SDK_ROOT/include/QNN \
 *       -DPREPARE_DISABLED -DTHIS_PKG_NAME=HmxMatMulPhase3Package \
 *       kernel/pack_act_hvx.c kernel/pack_wt_hvx.c \
 *       kernel/combine_hi_lo_hvx.c kernel/int4_expand_hvx.c \
 *       test_ops_sim.c -o test_ops_sim
 *   hexagon-sim -mv75 test_ops_sim
 *
 * Note: because these files pull in QHPI registration machinery, the sim
 * build must provide stubs for qhpi_register_ops_v1 and the tensor
 * accessors. They are declared but unused for the body-only call path.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* Forward decls — implementations in kernel/*.c. */
void pack_act_hvx_kernel_body(const uint16_t *au, uint8_t *out_hi,
                               uint8_t *out_lo, int M, int K);
void pack_wt_hvx_kernel_body(const int8_t *w, int8_t *out, int K, int N);
void combine_hi_lo_hvx_kernel_body(const int32_t *P_hi, const int32_t *P_lo,
                                    const int32_t *col_sum, int32_t *out,
                                    int M, int N);
void int4_expand_hvx_kernel_body(const uint8_t *in, int8_t *out, int K, int N);

/* QHPI stubs — unused when we only call the body functions, but needed
 * to link if the object files reference them. */
struct QHPI_Tensor;
typedef struct { uint32_t rank; uint32_t dims[8]; } QHPI_Shape_stub;
void *qhpi_tensor_raw_data(const void *t) { (void)t; return NULL; }
QHPI_Shape_stub qhpi_tensor_shape(const void *t) { (void)t; QHPI_Shape_stub s = {0, {0}}; return s; }
int qhpi_register_ops_v1(uint32_t n, void *ops, const char *pkg) {
    (void)n; (void)ops; (void)pkg; return 0;
}

/* -------------------------------------------------------------------
 * Scalar references — recomputed here for cross-check. Kept simple and
 * slow so a bug in the HVX path can't also hide in a shared helper.
 * ------------------------------------------------------------------- */
static void ref_pack_act(const uint16_t *au, uint8_t *out_hi, uint8_t *out_lo,
                         int M, int K)
{
    int Mt = M / 32, Kt = K / 32;
    for (int mt = 0; mt < Mt; mt++) {
        for (int kt = 0; kt < Kt; kt++) {
            uint8_t *thi = out_hi + (mt * Kt + kt) * 2048;
            uint8_t *tlo = out_lo + (mt * Kt + kt) * 2048;
            for (int phys_row = 0; phys_row < 16; phys_row++) {
                for (int Kk = 0; Kk < 32; Kk++) {
                    int m0 = mt * 32 + phys_row;
                    int m1 = mt * 32 + phys_row + 16;
                    int k  = kt * 32 + Kk;
                    uint16_t a0 = au[m0 * K + k];
                    uint16_t a1 = au[m1 * K + k];
                    /* Cell at (phys_row, Kk, stream): tile[128*phys_row + 4*Kk + stream] */
                    thi[128 * phys_row + 4 * Kk + 0] = 0;
                    thi[128 * phys_row + 4 * Kk + 1] = (uint8_t)(a0 >> 8);
                    thi[128 * phys_row + 4 * Kk + 2] = 0;
                    thi[128 * phys_row + 4 * Kk + 3] = (uint8_t)(a1 >> 8);
                    tlo[128 * phys_row + 4 * Kk + 0] = 0;
                    tlo[128 * phys_row + 4 * Kk + 1] = (uint8_t)(a0 & 0xFF);
                    tlo[128 * phys_row + 4 * Kk + 2] = 0;
                    tlo[128 * phys_row + 4 * Kk + 3] = (uint8_t)(a1 & 0xFF);
                }
            }
        }
    }
}

static void ref_pack_wt(const int8_t *w, int8_t *out, int K, int N)
{
    int Kt = K / 32, Nt = N / 32;
    for (int kt = 0; kt < Kt; kt++) {
        for (int nt = 0; nt < Nt; nt++) {
            int8_t *tile = out + (kt * Nt + nt) * 1024;
            for (int kg = 0; kg < 8; kg++) {
                for (int col = 0; col < 32; col++) {
                    int k0 = kt * 32 + kg * 4;
                    int n0 = nt * 32 + col;
                    uint8_t b0 = (uint8_t)w[(k0 + 0) * N + n0];
                    uint8_t b1 = (uint8_t)w[(k0 + 1) * N + n0];
                    uint8_t b2 = (uint8_t)w[(k0 + 2) * N + n0];
                    uint8_t b3 = (uint8_t)w[(k0 + 3) * N + n0];
                    uint32_t *dst = (uint32_t *)(tile + 128 * kg);
                    dst[col] = (uint32_t)b0 | ((uint32_t)b1 << 8)
                             | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
                }
            }
        }
    }
}

static void ref_combine(const int32_t *P_hi, const int32_t *P_lo,
                        const int32_t *col_sum, int32_t *out, int M, int N)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++)
            out[m * N + n] = (P_hi[m * N + n] << 8) + P_lo[m * N + n]
                           - (col_sum[n] << 15);
}

static void ref_int4_expand(const uint8_t *in, int8_t *out, int K, int N)
{
    int Nh = N / 2;
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < Nh; j++) {
            uint8_t b = in[k * Nh + j];
            int lo = b & 0x0F;
            int hi = (b >> 4) & 0x0F;
            out[k * N + 2 * j]     = (int8_t)(lo >= 8 ? lo - 16 : lo);
            out[k * N + 2 * j + 1] = (int8_t)(hi >= 8 ? hi - 16 : hi);
        }
    }
}

/* -------------------------------------------------------------------
 * Test bodies.
 * ------------------------------------------------------------------- */
static int test_pack_act(void)
{
    const int M = 32, K = 64;
    uint16_t au[32 * 64];
    uint8_t hi[1 * 2 * 2048], lo[1 * 2 * 2048];
    uint8_t hi_ref[1 * 2 * 2048], lo_ref[1 * 2 * 2048];
    for (int i = 0; i < M * K; i++) au[i] = (uint16_t)((i * 7919) & 0xFFFF);

    pack_act_hvx_kernel_body(au, hi, lo, M, K);
    ref_pack_act(au, hi_ref, lo_ref, M, K);

    int fails = 0;
    for (int i = 0; i < 2 * 2048; i++) {
        if (hi[i] != hi_ref[i]) fails++;
        if (lo[i] != lo_ref[i]) fails++;
    }
    printf("pack_act: %s (%d byte mismatches of %d)\n",
           fails ? "FAIL" : "PASS", fails, 2 * 2 * 2048);
    return fails ? 1 : 0;
}

static int test_pack_wt(void)
{
    const int K = 32, N = 64;
    int8_t w[32 * 64];
    int8_t out[1 * 2 * 1024];
    int8_t ref[1 * 2 * 1024];
    for (int i = 0; i < K * N; i++) w[i] = (int8_t)((i * 137) & 0xFF);

    pack_wt_hvx_kernel_body(w, out, K, N);
    ref_pack_wt(w, ref, K, N);

    int fails = 0;
    for (int i = 0; i < 2 * 1024; i++)
        if (out[i] != ref[i]) fails++;
    printf("pack_wt:  %s (%d byte mismatches of %d)\n",
           fails ? "FAIL" : "PASS", fails, 2 * 1024);
    return fails ? 1 : 0;
}

static int test_combine(void)
{
    const int M = 32, N = 64;
    int32_t P_hi[32 * 64], P_lo[32 * 64], col_sum[64];
    int32_t out[32 * 64], ref[32 * 64];
    for (int i = 0; i < M * N; i++) {
        P_hi[i] = (int32_t)((i * 31) - 1000);
        P_lo[i] = (int32_t)((i * 17) + 500);
    }
    for (int n = 0; n < N; n++) col_sum[n] = (n - 32) * 11;

    combine_hi_lo_hvx_kernel_body(P_hi, P_lo, col_sum, out, M, N);
    ref_combine(P_hi, P_lo, col_sum, ref, M, N);

    int fails = 0;
    for (int i = 0; i < M * N; i++)
        if (out[i] != ref[i]) fails++;
    printf("combine:  %s (%d int32 mismatches of %d)\n",
           fails ? "FAIL" : "PASS", fails, M * N);
    return fails ? 1 : 0;
}

static int test_int4_expand(void)
{
    const int K = 4, N = 256;          /* Nh = 128 = exactly one HVX vec/row */
    uint8_t in[4 * 128];
    int8_t  out[4 * 256], ref[4 * 256];
    for (int i = 0; i < K * (N / 2); i++) in[i] = (uint8_t)(i & 0xFF);

    int4_expand_hvx_kernel_body(in, out, K, N);
    ref_int4_expand(in, ref, K, N);

    int fails = 0;
    for (int i = 0; i < K * N; i++)
        if (out[i] != ref[i]) fails++;
    printf("int4_exp: %s (%d byte mismatches of %d)\n",
           fails ? "FAIL" : "PASS", fails, K * N);
    return fails ? 1 : 0;
}

int main(void)
{
    int total_fails = 0;
    total_fails += test_pack_act();
    total_fails += test_pack_wt();
    total_fails += test_combine();
    total_fails += test_int4_expand();
    printf("\n%s: %d/4 tests failed\n",
           total_fails ? "OVERALL FAIL" : "OVERALL PASS", total_fails);
    return total_fails ? 1 : 0;
}
