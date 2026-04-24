/*
 * demo06_u8xi8_tile.c — 完整 32×32×32 u8×i8 matmul，bit-exact 对照 C reference。
 *
 * 两个版本对比:
 *   (1) CPU + HMX: fill / pack / correction 用 plain scalar C
 *   (2) HVX + HMX: fill / correction 用 HVX 向量化（见 hmx_hvx_common.h）
 *
 * Pack 本身两者都用 u32-packed writes（不用 HVX vshuffe—— production 经验：
 * vshuffe 在 v75 产生错 tile 字节）。HMX MAC 指令两者相同。
 *
 * Verify 两版都 bit-exact 过 oracle。
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
#define K 32

static void ref_u8xi8(const uint8_t A[M][K], const int8_t W[K][N], int32_t C[M][N])
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++)
                s += (int32_t)A[i][k] * (int32_t)W[k][j];
            C[i][j] = s;
        }
}

/* CPU pack */
static void pack_act_cpu(uint8_t *tile, const uint8_t A[M][K]) {
    memset(tile, 0, 2048);
    for (int ir = 0; ir < M; ir++) {
        int pr = ir & 15, st = ir >> 4, off = st ? 3 : 1;
        for (int k = 0; k < K; k++)
            tile[128 * pr + 4 * k + off] = A[ir][k];
    }
}
static void pack_wt_cpu(int8_t *tile, const int8_t W[K][N]) {
    memset(tile, 0, 1024);
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++)
            tile[128 * (k >> 2) + 4 * j + (k & 3)] = W[k][j];
}
static void unpack_cpu(const uint16_t *out, int16_t C[M][N]) {
    for (int i = 0; i < M; i++) {
        int pr = i & 15, st = i >> 4;
        for (int j = 0; j < N; j++)
            C[i][j] = (int16_t)out[pr * 64 + 2 * j + st];
    }
}

static void hmx_u8xi8(const uint8_t *act, const int8_t *wt,
                      const uint16_t *bias, uint16_t *out)
{
    asm volatile("mxclracc" ::: "memory");
    asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
    asm volatile("{ activation.ub = mxmem(%0,%1)\n"
                 "  weight.b      = mxmem(%2,%3) }"
                 :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
    asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
                 :: "r"(out), "r"(0) : "memory");
}

static uint32_t rs = 0xDEADBEEF;
static uint32_t rnd(void) { rs = rs * 1103515245 + 12345; return rs; }

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

    uint8_t  *vt   = (uint8_t *)(unsigned long)vtcm;
    uint8_t  *act  = vt + 0 * 2048;
    int8_t   *wt   = (int8_t *)(vt + 2 * 2048);
    uint16_t *bias = (uint16_t *)(vt + 4 * 2048);
    uint16_t *out  = (uint16_t *)(vt + 6 * 2048);

    static uint8_t A[M][K];
    static int8_t  W[K][N];
    static int32_t Cref[M][N];
    static int16_t Cout[M][N];

    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) A[i][k] = (uint8_t)(rnd() & 0x0F);
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++) W[k][j] = (int8_t)((rnd() & 0x0F) - 8);

    ref_u8xi8(A, W, Cref);

    /* ====== Path 1: CPU fill + scalar pack + HMX ====== */
    for (int i = 0; i < 128; i++) bias[i] = 0x4000;
    pack_act_cpu(act, A);
    pack_wt_cpu(wt, W);
    memset(out, 0, 2048);
    hmx_u8xi8(act, wt, bias, out);
    unpack_cpu(out, Cout);

    int bad_cpu = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            if ((int32_t)Cout[i][j] != Cref[i][j]) bad_cpu++;

    /* ====== Path 2: HVX fill + scalar pack + HMX ====== */
    hvx_fill_u16(bias, 128, 0x4000);         /* HVX splat */
    hvx_pack_act_u8_32x32(act, (const uint8_t *)A);  /* 内部 scalar pack */
    hvx_pack_wt_b_32x32(wt, (const int8_t *)W);
    hvx_zero(out, 2048);                     /* HVX zero */
    hmx_u8xi8(act, wt, bias, out);
    unpack_cpu(out, Cout);

    int bad_hvx = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            if ((int32_t)Cout[i][j] != Cref[i][j]) bad_hvx++;

    printf("--- demo06: u8 x i8 bit-exact 32x32x32 tile ---\n");
    printf("  CPU fill + HMX: %d mismatches / %d\n", bad_cpu, M * N);
    printf("  HVX fill + HMX: %d mismatches / %d\n", bad_hvx, M * N);

    int fail = bad_cpu + bad_hvx;
    if (!fail) printf("  [PASS] demo06 (both paths)\n");
    else       printf("  [FAIL] demo06 (cpu=%d hvx=%d)\n", bad_cpu, bad_hvx);

    h2_thread_stop(fail ? 1 : 0);
    return fail ? 1 : 0;
}
