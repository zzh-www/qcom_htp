/*
 * hmx_int4xint8_matmul.c — int4 × int8 matmul on HMX (single-pass u8·i8 MAC).
 *
 * Per (m_tile, n_tile) output: 1 × mxclracc + (K/32) MAC-load packets +
 * dual-scale readback + scalar combine with col_sum correction.
 *
 * Shares Rt_wt=0x3FF and dualacc findings with hmx_int4_matmul.c
 * (Agent/qnn_hmx_pipelining.md, 2026-04-22).
 */

#include "hmx_int4xint8_matmul.h"
#include <string.h>
#ifdef __hexagon__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

/* ============== HMX asm wrappers ========================================= */
static inline __attribute__((always_inline)) void hmx_clracc_i(void)
{ asm volatile("mxclracc" ::: "memory"); }

static inline __attribute__((always_inline)) void hmx_swapacc_i(void)
{ asm volatile("mxswapacc" ::: "memory"); }

static inline __attribute__((always_inline))
void hmx_load_bias_i(const void *p)
{ asm volatile("bias = mxmem(%0)" :: "r"(p) : "memory"); }

#ifndef HMX_RT_ACT
#define HMX_RT_ACT 2047
#endif
#ifndef HMX_RT_WT
#define HMX_RT_WT  0x3FF
#endif

static inline __attribute__((always_inline))
void hmx_load_pair_u8_i8(const void *act, const void *wt)
{
    asm volatile(
        "{ activation.ub = mxmem(%0, %1)\n"
        "  weight.b      = mxmem(%2, %3) }\n"
        :: "r"(act), "r"(HMX_RT_ACT),
           "r"(wt),  "r"(HMX_RT_WT)
        : "memory");
}

static inline __attribute__((always_inline))
void hmx_store_acc_uh_2x1_retain(void *out)
{ asm volatile("mxmem(%0, %1):after:retain.uh = acc:2x1\n"
               :: "r"(out), "r"(0) : "memory"); }

static inline __attribute__((always_inline))
void hmx_store_acc_uh_2x1(void *out)
{ asm volatile("mxmem(%0, %1):after.uh = acc:2x1\n"
               :: "r"(out), "r"(0) : "memory"); }

static void fill_bias_scale(uint16_t *buf, uint16_t v)
{ for (int col = 0; col < 128; col++) buf[col] = v; }

/* ============== Tile packers ============================================= */
/* Activation: 32 logical rows × 32 K into HMX tile format:
 *   byte @ 128·phys_row + 4·K + (stream ? 3 : 1),
 *   where phys_row = ir&15, stream = ir>>4. */
static void pack_activation_32x32_rs(uint8_t *tile, const uint8_t *a_rows,
                                      int row_stride)
{
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        uint32_t *__restrict__ dst = (uint32_t *)(tile + 128 * phys_row);
        const uint8_t *s0 = &a_rows[phys_row * row_stride];
        const uint8_t *s1 = &a_rows[(phys_row + 16) * row_stride];
        for (int K = 0; K < 32; K++) {
            dst[K] = ((uint32_t)s1[K] << 24) | ((uint32_t)s0[K] << 8);
        }
    }
}

/* pack_weight_32x32: 4 rows × 32 cols → 32 cells × 4-byte-per-cell.
 * Output byte index = rotate-right-by-2 of input byte index on 7 bits.
 * Two back-to-back Q6_Vb_vshuff_Vb produce exactly this permutation
 * (proof: Agent/hvx_4way_byte_transpose_re.md). memcpy wrappers let
 * callers pass un-aligned buffers at zero cost on v75. */
#ifdef __hexagon__
static void pack_weight_32x32(int8_t *tile, const int8_t *w_32x32)
{
    for (int kg = 0; kg < 8; kg++) {
        HVX_Vector v, s1, s2;
        memcpy(&v, w_32x32 + 128 * kg, sizeof(HVX_Vector));
        s1 = Q6_Vb_vshuff_Vb(v);
        s2 = Q6_Vb_vshuff_Vb(s1);
        memcpy(tile + 128 * kg, &s2, sizeof(HVX_Vector));
    }
}
#else
static void pack_weight_32x32(int8_t *tile, const int8_t *w_32x32)
{
    for (int kg = 0; kg < 8; kg++) {
        uint32_t *__restrict__ dst = (uint32_t *)(tile + 128 * kg);
        const uint8_t *r0 = (const uint8_t *)&w_32x32[(kg * 4 + 0) * 32];
        const uint8_t *r1 = (const uint8_t *)&w_32x32[(kg * 4 + 1) * 32];
        const uint8_t *r2 = (const uint8_t *)&w_32x32[(kg * 4 + 2) * 32];
        const uint8_t *r3 = (const uint8_t *)&w_32x32[(kg * 4 + 3) * 32];
        for (int col = 0; col < 32; col++) {
            dst[col] =  (uint32_t)r0[col]
                     | ((uint32_t)r1[col] << 8)
                     | ((uint32_t)r2[col] << 16)
                     | ((uint32_t)r3[col] << 24);
        }
    }
}
#endif

/* ============== Persistent scratch (off-stack to avoid overflow) ========= */
static int32_t sg_col_sum_w[32];

/* VTCM layout (8 KiB fixed, then K·128 B activation prepack).
 *   [0, 1 KiB)     act_tile  (transient; unused once prepacked)
 *   [1 KiB, 2 KiB) wt_tile   (per-K-iter scalar pack target)
 *   [2 KiB, 3 KiB) bias region: 256 B bias_lo + 256 B bias_hi, rest padding
 *   [4 KiB, 6 KiB) out_lo (2 KiB, 2KB-aligned — HMX acc:2x1 store requires this!)
 *   [6 KiB, 8 KiB) out_hi (2 KiB, 2KB-aligned)
 *   [8 KiB..)      act_prepack (K/32 tiles × 2 KiB)
 *
 * Alignment note: HMX mxmem store with `acc:2x1` requires the destination
 * to be 2KB-aligned. Original layout had out_lo @ 3K and out_hi @ 5K
 * (both byte-misaligned by 1KB), which caused phys_row 8-15 output to be
 * silently corrupted. The w4a16 kernel uses 4K/6K for the same reason.
 */
#define VTCM_OFF_ACT_TILE   (0)
#define VTCM_OFF_WT_TILE    (1 * 1024)
#define VTCM_OFF_BIAS_LO    (2 * 1024)
#define VTCM_OFF_BIAS_HI    (2 * 1024 + 256)
#define VTCM_OFF_OUT_LO     (4 * 1024)
#define VTCM_OFF_OUT_HI     (6 * 1024)
#define VTCM_OFF_PREPACK    (8 * 1024)
#define VTCM_ACT(k32)       (VTCM_OFF_PREPACK + (k32) * 2048)

/* ============== Public API =============================================== */

void hmx_int4xint8_prepack_activation(
    const uint8_t *__restrict__ au,
    int                          M_full,
    int                          K,
    int                          m0,
    void          *__restrict__ vtcm_base)
{
    (void)M_full;
    uint8_t *vt = (uint8_t *)vtcm_base;
    const int Ktiles = K / 32;

    /* Activation is uint8 (QNN post-Cast, = signed_act + 128). HMX reads it
     * as ub; -128 un-shift deferred to combine via col_sum_w. */
#ifdef __hexagon__
    /* HVX: per phys_row, build [s0(32B) | s1(32B) | pad(64B)] → vzxt
     * bytes→halfwords (low half = [s0 hwords(32) | s1 hwords(32)]) →
     * halfword asl 8 (each hword becomes byte [0, value]) → self-shuff
     * halfword interleaves → output bytes [0, s0[k], 0, s1[k], ...].
     * Since every byte of output tile is written, no memset needed. */
    for (int kt = 0; kt < Ktiles; kt++) {
        uint8_t *tile = vt + VTCM_ACT(kt);
        for (int phys_row = 0; phys_row < 16; phys_row++) {
            const uint8_t *s0 = &au[(m0 + phys_row)      * K + kt * 32];
            const uint8_t *s1 = &au[(m0 + phys_row + 16) * K + kt * 32];
            /* 4-way byte transpose via 2× Q6_Vb_vshuff_Vb (same trick as
             * pack_weight_32x32). Input layout [r0=0|r1=s0|r2=0|r3=s1]
             * produces output bytes at 4k+i = r_i[k] → [0, s0[k], 0, s1[k]]. */
            uint8_t staging[128] __attribute__((aligned(128))) = {0};
            memcpy(&staging[32], s0, 32);   /* r1 = s0 (bytes go to pos 4k+1) */
            memcpy(&staging[96], s1, 32);   /* r3 = s1 (bytes go to pos 4k+3) */
            HVX_Vector v_in, v1, v2;
            memcpy(&v_in, staging, sizeof(HVX_Vector));
            v1 = Q6_Vb_vshuff_Vb(v_in);
            v2 = Q6_Vb_vshuff_Vb(v1);
            memcpy(tile + 128 * phys_row, &v2, sizeof(HVX_Vector));
        }
    }
#else
    for (int kt = 0; kt < Ktiles; kt++) {
        uint8_t *tile = vt + VTCM_ACT(kt);
        for (int i = 0; i < 2048; i++) tile[i] = 0;
        for (int phys_row = 0; phys_row < 16; phys_row++) {
            uint32_t *__restrict__ dst = (uint32_t *)(tile + 128 * phys_row);
            const uint8_t *s0 = &au[(m0 + phys_row)      * K + kt * 32];
            const uint8_t *s1 = &au[(m0 + phys_row + 16) * K + kt * 32];
            for (int K32 = 0; K32 < 32; K32++) {
                uint8_t a0 = s0[K32];
                uint8_t a1 = s1[K32];
                dst[K32] = ((uint32_t)a1 << 24) | ((uint32_t)a0 << 8);
            }
        }
    }
#endif
}

void hmx_int4xint8_matmul_mn(
    int32_t       *__restrict__ out,
    const int8_t  *__restrict__ w,
    int                          K,
    void          *__restrict__ vtcm_base,
    const int32_t *              col_sum_w_in)
{
    uint8_t  *vt       = (uint8_t *)vtcm_base;
    int8_t   *wt_tile  = (int8_t *)( vt + VTCM_OFF_WT_TILE);
    uint16_t *bias_lo  = (uint16_t *)(vt + VTCM_OFF_BIAS_LO);
    uint16_t *bias_hi  = (uint16_t *)(vt + VTCM_OFF_BIAS_HI);
    uint16_t *out_lo   = (uint16_t *)(vt + VTCM_OFF_OUT_LO);
    uint16_t *out_hi   = (uint16_t *)(vt + VTCM_OFF_OUT_HI);

    fill_bias_scale(bias_lo, 0x4000);
    fill_bias_scale(bias_hi, 0x2000);

    /* col_sum_w — T1d hoist support: use caller-provided if available. */
    const int32_t *col_sum_w;
    if (col_sum_w_in) {
        col_sum_w = col_sum_w_in;
    } else {
        for (int j = 0; j < 32; j++) sg_col_sum_w[j] = 0;
        for (int k = 0; k < K; k++)
            for (int j = 0; j < 32; j++)
                sg_col_sum_w[j] += (int32_t)w[k * 32 + j];
        col_sum_w = sg_col_sum_w;
    }

    const int Ktiles = K / 32;
    static int32_t sg_P[32 * 32];

    /* Single-pass K-accumulated u8·i8 MAC. */
    hmx_load_bias_i(bias_lo);
    hmx_clracc_i();
    for (int kt = 0; kt < Ktiles; kt++) {
        pack_weight_32x32(wt_tile, &w[kt * 32 * 32]);
        hmx_load_pair_u8_i8(vt + VTCM_ACT(kt), wt_tile);
    }
    hmx_store_acc_uh_2x1_retain(out_lo);
    hmx_load_bias_i(bias_hi);
    hmx_store_acc_uh_2x1(out_hi);

    /* Reconstruct int24 partial from dual-scale readback. */
    for (int ir = 0; ir < 32; ir++) {
        int phys_row = ir & 15, stream = ir >> 4;
        for (int jc = 0; jc < 32; jc++) {
            uint16_t lo = out_lo[phys_row * 64 + 2 * jc + stream];
            uint16_t hi = out_hi[phys_row * 64 + 2 * jc + stream];
            sg_P[ir * 32 + jc] = ((int32_t)(int16_t)hi << 8) | ((int32_t)lo & 0xFF);
        }
    }

    /* Combine: out = acc - 128·col_sum_w.
     * (acc = sum_k au[m,k]·w[k,n],  where au = signed_act + 128.
     *  true = sum_k (au-128)·w = acc - 128·col_sum_w.) */
#ifdef __hexagon__
    HVX_Vector v_col_shifted;
    memcpy(&v_col_shifted, col_sum_w, sizeof(HVX_Vector));
    v_col_shifted = Q6_Vw_vasl_VwR(v_col_shifted, 7);
    for (int i = 0; i < 32; i++) {
        HVX_Vector v_p, v_out;
        memcpy(&v_p, &sg_P[i * 32], sizeof(HVX_Vector));
        v_out = Q6_Vw_vsub_VwVw(v_p, v_col_shifted);
        memcpy(&out[i * 32], &v_out, sizeof(HVX_Vector));
    }
#else
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++)
            out[i * 32 + j] = sg_P[i * 32 + j] - (col_sum_w[j] << 7);
#endif
}
