/*
 * hmx_hvx_common.h — HVX helpers shared by all demos with HVX variants.
 *
 * 设计原则：
 *   - **pack/unpack 本身保持 scalar**（32-bit packed writes）。生产 kernel
 *     `example/hmx_matmul_qnn/kernel/hmx_int4_matmul.c` 明确说"HVX vshuffe
 *     pack 产生错 tile 字节"——scalar 4-byte packed 写更稳且成本 <1%。
 *   - **HVX 做 fill / zero / col-sum / correction / offset** — 这些是
 *     vector-friendly 的计算，HVX 线性加速。
 *
 * 所以 demo 的 "HVX + HMX" 版本 ≠ HVX pack；它指 HVX 加速的 glue 计算 +
 * 相同 scalar pack + HMX MAC。和 "CPU + HMX" 的区别在 glue 层。
 */

#ifndef HMX_HVX_COMMON_H
#define HMX_HVX_COMMON_H

#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

/* Unaligned 128-byte HVX load/store */
typedef struct { HVX_Vector v; } __attribute__((packed, aligned(1))) hvx_unaligned_t;
static inline HVX_Vector hvx_vldu(const void *p) { return ((const hvx_unaligned_t *)p)->v; }
static inline void       hvx_vstu(void *p, HVX_Vector v) { ((hvx_unaligned_t *)p)->v = v; }

/* ================================================================ */
/* HVX fill / zero                                                  */
/* ================================================================ */

static inline void hvx_zero(void *dst, int bytes)
{
    HVX_Vector z = Q6_V_vzero();
    HVX_Vector *p = (HVX_Vector *)dst;
    for (int i = 0; i < bytes / 128; i++) p[i] = z;
}

static inline void hvx_fill_u8(void *dst, int bytes, uint8_t v)
{
    int splat = ((int)v << 24) | ((int)v << 16) | ((int)v << 8) | v;
    HVX_Vector x = Q6_V_vsplat_R(splat);
    HVX_Vector *p = (HVX_Vector *)dst;
    for (int i = 0; i < bytes / 128; i++) p[i] = x;
}

static inline void hvx_fill_u16(void *dst, int count, uint16_t v)
{
    int splat = ((int)v << 16) | v;
    HVX_Vector x = Q6_V_vsplat_R(splat);
    HVX_Vector *p = (HVX_Vector *)dst;
    for (int i = 0; i < count / 64; i++) p[i] = x;
}

/* ================================================================ */
/* Scalar tile pack (和 CPU 版本相同，u32-packed writes)            */
/*                                                                  */
/* 这些函数名字带 "hvx_" 前缀只是为了说明"HVX 路径也调用"；内部实现  */
/* 用 u32 packed writes 而不是 HVX shuffle（见文件头注释）。        */
/* ================================================================ */

static inline void hvx_pack_act_u8_32x32(uint8_t *tile, const uint8_t *A)
{
    /* 输入: A[32][32] row-major u8 */
    hvx_zero(tile, 2048);
    for (int pr = 0; pr < 16; pr++) {
        uint32_t *dst = (uint32_t *)(tile + 128 * pr);
        const uint8_t *s0 = &A[pr * 32];
        const uint8_t *s1 = &A[(pr + 16) * 32];
        for (int k = 0; k < 32; k++) {
            /* 4-byte slot 为 { 0, s0[k], 0, s1[k] }（little-endian u32）*/
            dst[k] = ((uint32_t)s1[k] << 24) | ((uint32_t)s0[k] << 8);
        }
    }
}

/* row_stride 变种: A 是 32 行 × row_stride bytes 的片段（大 K 分片用） */
static inline void hvx_pack_act_u8_32xKslice(uint8_t *tile, const uint8_t *A, int row_stride)
{
    hvx_zero(tile, 2048);
    for (int pr = 0; pr < 16; pr++) {
        uint32_t *dst = (uint32_t *)(tile + 128 * pr);
        const uint8_t *s0 = &A[pr * row_stride];
        const uint8_t *s1 = &A[(pr + 16) * row_stride];
        for (int k = 0; k < 32; k++) {
            dst[k] = ((uint32_t)s1[k] << 24) | ((uint32_t)s0[k] << 8);
        }
    }
}

static inline void hvx_pack_wt_b_32x32(int8_t *tile, const int8_t *W)
{
    /* W[32][32] row-major int8。输出 8 条 128-B 线 × 4 K / 线 × 32 col。*/
    for (int kg = 0; kg < 8; kg++) {
        uint32_t *dst = (uint32_t *)(tile + 128 * kg);
        const uint8_t *r0 = (const uint8_t *)&W[(kg * 4 + 0) * 32];
        const uint8_t *r1 = (const uint8_t *)&W[(kg * 4 + 1) * 32];
        const uint8_t *r2 = (const uint8_t *)&W[(kg * 4 + 2) * 32];
        const uint8_t *r3 = (const uint8_t *)&W[(kg * 4 + 3) * 32];
        for (int col = 0; col < 32; col++) {
            dst[col] = ((uint32_t)r0[col]       ) |
                       ((uint32_t)r1[col] <<  8) |
                       ((uint32_t)r2[col] << 16) |
                       ((uint32_t)r3[col] << 24);
        }
    }
}

/* int4 weight packer (layout 探针验证): byte_off = 128·(K>>3) + 4·col + ((K>>1)&3), hi = K&1 */
static inline void hvx_pack_wt_n_32x32(uint8_t *tile, const int8_t *W)
{
    memset(tile, 0, 512);
    for (int k = 0; k < 32; k++) {
        for (int j = 0; j < 32; j++) {
            int off = 128 * (k >> 3) + 4 * j + ((k >> 1) & 3);
            int hi = k & 1;
            uint8_t nib = (uint8_t)(W[k * 32 + j] & 0x0F);
            if (hi) tile[off] = (uint8_t)((tile[off] & 0x0F) | (nib << 4));
            else    tile[off] = (uint8_t)((tile[off] & 0xF0) | (nib & 0x0F));
        }
    }
}
static inline void hvx_pack_wt_n_32xKslice(uint8_t *tile, const int8_t *W, int row_stride)
{
    memset(tile, 0, 512);
    for (int k = 0; k < 32; k++) {
        for (int j = 0; j < 32; j++) {
            int off = 128 * (k >> 3) + 4 * j + ((k >> 1) & 3);
            int hi = k & 1;
            uint8_t nib = (uint8_t)(W[k * row_stride + j] & 0x0F);
            if (hi) tile[off] = (uint8_t)((tile[off] & 0x0F) | (nib << 4));
            else    tile[off] = (uint8_t)((tile[off] & 0xF0) | (nib & 0x0F));
        }
    }
}

/* ================================================================ */
/* HVX activation offset: A_u8 = A_i8 + 128                         */
/*                                                                  */
/* HVX 版用 vector byte add，整个 32*32=1024 bytes 分 8 条 128-B 线 */
/* 并行加。                                                          */
/* ================================================================ */

static inline void hvx_add_i8_plus_128(uint8_t *dst, const int8_t *src, int n_bytes)
{
    /* int8 + 128 = uint8 (bit-pattern 只是 XOR 0x80 —— 翻转最高位) */
    HVX_Vector v_mask = Q6_V_vsplat_R(0x80808080);
    int nv = n_bytes / 128;
    for (int i = 0; i < nv; i++) {
        HVX_Vector v = hvx_vldu(&src[i * 128]);
        v = Q6_V_vxor_VV(v, v_mask);
        hvx_vstu(&dst[i * 128], v);
    }
    /* 尾部 scalar 处理 */
    for (int i = nv * 128; i < n_bytes; i++)
        dst[i] = (uint8_t)((int)src[i] + 128);
}

/* ================================================================ */
/* HVX col_sum_w: Σ_k W[k][j] 为每 col 一个 int32                   */
/*                                                                  */
/* 输入 W[32][32] int8，输出 int32[32]。                             */
/* 策略: 每次读一行 (32 bytes, sign-extend to int16 halfwords, 再   */
/* vadd 到 int16 累加器，最后再扩到 int32).                         */
/* ================================================================ */

static inline void hvx_col_sum_w(int32_t col_sum[32], const int8_t *W)
{
    /* 简化: 8 row × 4 chunk 用标量展开（HVX 做 reduce 到 32 col 的 int32
     * 累加器需要 vsxt.b.h + vsxt.h.w 双宽度变换，成本和 scalar 相当）。
     * 保留 HVX 风格的 memset 至少。*/
    for (int j = 0; j < 32; j++) col_sum[j] = 0;
    for (int k = 0; k < 32; k++)
        for (int j = 0; j < 32; j++)
            col_sum[j] += (int32_t)W[k * 32 + j];
}

/* ================================================================ */
/* HVX correction: C[i][j] -= scale * col_sum_w[j]                  */
/*                                                                  */
/* 每 col 32 个 int32 = 128 B，刚好一个 HVX vector。                */
/* ================================================================ */

static inline void hvx_apply_col_sum_correction(
    int32_t *C, const int32_t *col_sum_w, int scale, int rows)
{
    /* 预计算: correction[j] = scale * col_sum_w[j] */
    int32_t corr[32] __attribute__((aligned(128)));
    for (int j = 0; j < 32; j++) corr[j] = scale * col_sum_w[j];

    HVX_Vector v_corr = hvx_vldu(corr);
    for (int i = 0; i < rows; i++) {
        HVX_Vector v_row = hvx_vldu(&C[i * 32]);
        v_row = Q6_Vw_vsub_VwVw(v_row, v_corr);
        hvx_vstu(&C[i * 32], v_row);
    }
}

#endif /* HMX_HVX_COMMON_H */
