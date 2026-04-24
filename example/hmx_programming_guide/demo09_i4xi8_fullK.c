/*
 * demo09_i4xi8_fullK.c — int4×int8, K=128 (4 tile 累加 + dual-scale readback)。
 *
 * 两路: CPU vs HVX (HVX 版用 hvx_add_i8_plus_128 + hvx_apply_col_sum_correction)。
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>
#include "hmx_hvx_common.h"

#define M 32
#define N 32
#define FULL_K 128
#define KT 32

static uint32_t rs = 0xABCDEF;
static uint32_t rnd(void) { rs = rs * 1103515245 + 12345; return rs; }

static void pack_act_u8_slice_cpu(uint8_t *tile, const uint8_t A[M][FULL_K], int k0) {
    memset(tile, 0, 2048);
    for (int ir = 0; ir < M; ir++) {
        int pr = ir & 15, st = ir >> 4, off = st ? 3 : 1;
        for (int k = 0; k < KT; k++) tile[128*pr + 4*k + off] = A[ir][k0 + k];
    }
}
static void pack_wt_n_slice_cpu(uint8_t *tile, const int8_t W[FULL_K][N], int k0) {
    memset(tile, 0, 512);
    for (int k = 0; k < KT; k++)
        for (int j = 0; j < N; j++) {
            int off = 128*(k>>3) + 4*j + ((k>>1)&3);
            int hi = k & 1;
            uint8_t nib = (uint8_t)(W[k0 + k][j] & 0x0F);
            if (hi) tile[off] = (uint8_t)((tile[off] & 0x0F) | (nib << 4));
            else    tile[off] = (uint8_t)((tile[off] & 0xF0) | (nib & 0x0F));
        }
}

int main(void)
{
    unsigned int vtcm = h2_info(INFO_VTCM_BASE);
    if (!vtcm) { printf("[FAIL] no VTCM\n"); h2_thread_stop(1); return 1; }
    h2_vecaccess_state_t va;
    h2_vecaccess_unit_init(&va, H2_VECACCESS_HVX_128, CFG_TYPE_VXU0,
                           CFG_SUBTYPE_VXU0, CFG_HVX_CONTEXTS, 0x1);
    h2_vecaccess_acquire(&va);
    h2_mxaccess_state_t ma;
    h2_mxaccess_unit_init(&ma, CFG_TYPE_VXU0, CFG_SUBTYPE_VXU0,
                          CFG_HMX_CONTEXTS, 0x1);
    h2_mxaccess_acquire(&ma);

    uint8_t  *vt      = (uint8_t *)(unsigned long)vtcm;
    uint8_t  *act     = vt + 0*2048;
    uint8_t  *wt      = vt + 2*2048;
    uint16_t *bias_lo = (uint16_t *)(vt + 4*2048);
    uint16_t *bias_hi = (uint16_t *)(vt + 4*2048 + 256);
    uint16_t *out_lo  = (uint16_t *)(vt + 6*2048);
    uint16_t *out_hi  = (uint16_t *)(vt + 8*2048);

    static int8_t  A_i8[M][FULL_K], W_i4[FULL_K][N];
    static uint8_t A_u8[M][FULL_K];
    static int32_t Cref[M][N], Chmx[M][N], Cfinal[M][N];
    static int32_t col_sum_w[N];

    for (int i = 0; i < M; i++)
        for (int k = 0; k < FULL_K; k++) A_i8[i][k] = (int8_t)((rnd() & 0x0F) - 8);
    for (int k = 0; k < FULL_K; k++)
        for (int j = 0; j < N; j++) W_i4[k][j] = (int8_t)((rnd() & 0x0F) - 8);

    /* Reference */
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t s = 0;
            for (int k = 0; k < FULL_K; k++) s += (int32_t)A_i8[i][k] * (int32_t)W_i4[k][j];
            Cref[i][j] = s;
        }

    printf("--- demo09: int4×int8 K=128 ---\n");
    int fail = 0;

    /* ====== Loop body shared: MAC + dual-scale readback ====== */
    #define RUN_HMX(pack_act_fn, pack_wt_fn)                                   \
        do {                                                                    \
            asm volatile("mxclracc" ::: "memory");                              \
            asm volatile("bias = mxmem(%0)" :: "r"(bias_lo) : "memory");        \
            for (int kt = 0; kt < FULL_K; kt += KT) {                           \
                pack_act_fn;                                                    \
                pack_wt_fn;                                                     \
                asm volatile("{ activation.ub = mxmem(%0,%1)\n"                 \
                             "  weight.n      = mxmem(%2,%3) }"                 \
                             :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory"); \
            }                                                                   \
            asm volatile("mxmem(%0,%1):after:retain.uh = acc:2x1"               \
                         :: "r"(out_lo), "r"(0) : "memory");                    \
            asm volatile("bias = mxmem(%0)" :: "r"(bias_hi) : "memory");        \
            asm volatile("mxmem(%0,%1):after.uh = acc:2x1"                      \
                         :: "r"(out_hi), "r"(0) : "memory");                    \
        } while (0)

    /* ====== Path 1: CPU ====== */
    for (int i = 0; i < M; i++)
        for (int k = 0; k < FULL_K; k++) A_u8[i][k] = (uint8_t)(A_i8[i][k] + 128);
    for (int j = 0; j < N; j++) {
        col_sum_w[j] = 0;
        for (int k = 0; k < FULL_K; k++) col_sum_w[j] += W_i4[k][j];
    }
    for (int i = 0; i < 128; i++) { bias_lo[i] = 0x4000; bias_hi[i] = 0x2000; }

    RUN_HMX(pack_act_u8_slice_cpu(act, A_u8, kt), pack_wt_n_slice_cpu(wt, W_i4, kt));

    for (int i = 0; i < M; i++) {
        int pr = i & 15, st = i >> 4;
        for (int j = 0; j < N; j++) {
            int idx = pr*64 + 2*j + st;
            int32_t acc = ((int32_t)(int16_t)out_hi[idx] << 8) | ((int32_t)out_lo[idx] & 0xFF);
            Chmx[i][j] = acc;
        }
    }
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            Cfinal[i][j] = Chmx[i][j] - 128 * col_sum_w[j];
    int bad_cpu = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            if (Cfinal[i][j] != Cref[i][j]) bad_cpu++;
    printf("  CPU: %d mismatches / %d\n", bad_cpu, M * N);
    fail += bad_cpu;

    /* ====== Path 2: HVX ====== */
    hvx_add_i8_plus_128((uint8_t *)A_u8, (const int8_t *)A_i8, M * FULL_K);
    /* col_sum_w 跨 4 tile 累积 */
    for (int j = 0; j < N; j++) col_sum_w[j] = 0;
    for (int j = 0; j < N; j++)
        for (int k = 0; k < FULL_K; k++) col_sum_w[j] += W_i4[k][j];
    hvx_fill_u16(bias_lo, 128, 0x4000);
    hvx_fill_u16(bias_hi, 128, 0x2000);

    RUN_HMX(hvx_pack_act_u8_32xKslice(act, (const uint8_t *)(A_u8[0] + kt), FULL_K),
            hvx_pack_wt_n_32xKslice(wt, (const int8_t *)W_i4[kt], N));

    for (int i = 0; i < M; i++) {
        int pr = i & 15, st = i >> 4;
        for (int j = 0; j < N; j++) {
            int idx = pr*64 + 2*j + st;
            int32_t acc = ((int32_t)(int16_t)out_hi[idx] << 8) | ((int32_t)out_lo[idx] & 0xFF);
            Chmx[i][j] = acc;
        }
    }
    memcpy(Cfinal, Chmx, sizeof(Chmx));
    hvx_apply_col_sum_correction((int32_t *)Cfinal, col_sum_w, 128, M);
    int bad_hvx = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            if (Cfinal[i][j] != Cref[i][j]) bad_hvx++;
    printf("  HVX: %d mismatches / %d\n", bad_hvx, M * N);
    fail += bad_hvx;

    if (!fail) printf("  [PASS] demo09 (both paths)\n");
    else       printf("  [FAIL] demo09 (cpu=%d hvx=%d)\n", bad_cpu, bad_hvx);

    h2_thread_stop(fail ? 1 : 0);
    return fail ? 1 : 0;
}
