/*
 * hmx_int16x16_matmul.c — int16×int16 matmul on HMX, K-accumulated, int32 out.
 *
 * Based on int16_matmul_hmx.c's 4-term u8·i8 decomp. Adapted for K>32 by
 * running the 4 partial-product MACs K-accumulated (one mxclracc per
 * partial, K/32 mxmem pairs inside).
 *
 * Per 32×32 output: 4 partials × (2 + K/32) HMX issues + dual-scale readback.
 */

#include "hmx_int16x16_matmul.h"
#include <string.h>

/* ============== HMX wrappers ============================================= */
static inline __attribute__((always_inline)) void hmx_clracc_i(void)
{ asm volatile("mxclracc" ::: "memory"); }

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

/* ============== Tile packers ============================================= *
 * Activation tile (2 KB): byte at 128·phys + 4·K + (stream?3:1).
 * Weight tile   (1 KB): byte at 128·(K>>2) + 4·col + (K&3). */
static void pack_activation_u8_full(uint8_t *tile, const uint8_t *a_32x32)
{
    memset(tile, 0, 2048);
    for (int ir = 0; ir < 32; ir++) {
        int phys = ir & 15, stream = ir >> 4;
        int byte_off = stream ? 3 : 1;
        for (int K = 0; K < 32; K++)
            tile[128*phys + 4*K + byte_off] = a_32x32[ir*32 + K];
    }
}

static void pack_weight_i8_full(int8_t *tile, const int8_t *w_32x32)
{
    memset(tile, 0, 1024);
    for (int K = 0; K < 32; K++)
        for (int col = 0; col < 32; col++)
            tile[128*(K>>2) + 4*col + (K&3)] = w_32x32[K*32 + col];
}

/* ============== K-accumulated partial product ============================ */
/* Computes: partial[i][j] = Σ_k A_slice[i,k] × W_slice[k,j]
 * where A,W each a K-strip of the full matrix extracted per iter.
 *
 * a_byte_stream: for each K-tile, a 32×32 u8 strip (from hi or lo of the int16
 *                activation slice).
 * w_byte_stream: same pattern but for weight.
 *
 * Uses HMX u8·i8 MAC (treating u8 weight byte as i8 re-interp for the
 * M2/M4 partials — the caller applies topbit corrections externally). */
static void partial_dual_scale(
    int32_t       *__restrict__ dst,
    const uint8_t *const *a_tiles,    /* K/32 pointers to 32×32 u8 strips */
    const int8_t  *const *w_tiles,
    int                    Ktiles,
    uint8_t       *act_tile,
    int8_t        *wt_tile,
    uint16_t      *bias_lo,
    uint16_t      *bias_hi,
    uint16_t      *out_lo,
    uint16_t      *out_hi)
{
    hmx_load_bias_i(bias_lo);
    hmx_clracc_i();
    for (int kt = 0; kt < Ktiles; kt++) {
        pack_activation_u8_full(act_tile, a_tiles[kt]);
        pack_weight_i8_full(wt_tile, w_tiles[kt]);
        hmx_load_pair_u8_i8(act_tile, wt_tile);
    }
    hmx_store_acc_uh_2x1_retain(out_lo);
    hmx_load_bias_i(bias_hi);
    hmx_store_acc_uh_2x1(out_hi);

    for (int ir = 0; ir < 32; ir++) {
        int phys = ir & 15, stream = ir >> 4;
        for (int jc = 0; jc < 32; jc++) {
            uint16_t lo = out_lo[phys*64 + 2*jc + stream];
            uint16_t hi = out_hi[phys*64 + 2*jc + stream];
            dst[ir*32 + jc] = ((int32_t)(int16_t)hi << 8) | ((int32_t)lo & 0xFF);
        }
    }
}

/* ============== Persistent scratch (off-stack) =========================== */
#define MAX_KTILES 128    /* supports K up to 4096 */
static uint8_t  sg_Ah[MAX_KTILES][32*32];
static uint8_t  sg_Al[MAX_KTILES][32*32];
static int8_t   sg_Wh[MAX_KTILES][32*32];
static uint8_t  sg_Wl[MAX_KTILES][32*32];
static int32_t  sg_col_sum_w[32];
static int32_t  sg_row_sum_a[32];
static int32_t  sg_M1[32*32], sg_M2[32*32], sg_M3[32*32], sg_M4[32*32];
static int32_t  sg_M2c[32*32], sg_M4c[32*32];
static const uint8_t *sg_aptrs[MAX_KTILES];
static const int8_t  *sg_wptrs[MAX_KTILES];

/* VTCM layout (16 KiB fixed, 2KB-aligned out tiles). */
#define VTCM_OFF_ACT_TILE   (0 * 1024)
#define VTCM_OFF_WT_TILE    (2 * 1024)
#define VTCM_OFF_BIAS_LO    (3 * 1024)
#define VTCM_OFF_BIAS_HI    (3 * 1024 + 256)
#define VTCM_OFF_OUT_LO     (4 * 1024)   /* 2KB-aligned */
#define VTCM_OFF_OUT_HI     (6 * 1024)   /* 2KB-aligned */

void hmx_int16x16_matmul_mn(
    int32_t        *__restrict__ out,
    const uint16_t *__restrict__ au,
    const uint16_t *__restrict__ wu,
    int                          M_full,
    int                          K,
    int                          N_full,
    int                          m0,
    int                          n0,
    void           *__restrict__ vtcm_base)
{
    (void)M_full;
    uint8_t  *vt      = (uint8_t *)vtcm_base;
    uint8_t  *act_tile = vt + VTCM_OFF_ACT_TILE;
    int8_t   *wt_tile  = (int8_t *)(vt + VTCM_OFF_WT_TILE);
    uint16_t *bias_lo  = (uint16_t *)(vt + VTCM_OFF_BIAS_LO);
    uint16_t *bias_hi  = (uint16_t *)(vt + VTCM_OFF_BIAS_HI);
    uint16_t *out_lo   = (uint16_t *)(vt + VTCM_OFF_OUT_LO);
    uint16_t *out_hi   = (uint16_t *)(vt + VTCM_OFF_OUT_HI);

    fill_bias_scale(bias_lo, 0x4000);
    fill_bias_scale(bias_hi, 0x2000);

    const int Ktiles = K / 32;
    /* Decompose + gather into K-tile array of 32×32 uint8/int8 strips. */
    for (int j = 0; j < 32; j++) sg_col_sum_w[j] = 0;
    for (int i = 0; i < 32; i++) sg_row_sum_a[i] = 0;

    for (int kt = 0; kt < Ktiles; kt++) {
        int k0 = kt * 32;
        for (int i = 0; i < 32; i++) {
            for (int k = 0; k < 32; k++) {
                uint16_t au_val = au[(m0 + i) * K + (k0 + k)];
                int16_t  aq     = (int16_t)((int32_t)au_val - 32768);
                uint16_t au_u   = (uint16_t)(aq + 32768);
                sg_Ah[kt][i*32 + k] = (uint8_t)(au_u >> 8);
                sg_Al[kt][i*32 + k] = (uint8_t)(au_u & 0xFF);
                sg_row_sum_a[i] += (int32_t)aq;
            }
        }
        for (int k = 0; k < 32; k++) {
            for (int j = 0; j < 32; j++) {
                uint16_t wu_val = wu[(k0 + k) * N_full + (n0 + j)];
                int16_t  wq     = (int16_t)((int32_t)wu_val - 32768);
                sg_Wh[kt][k*32 + j] = (int8_t)(wq >> 8);
                sg_Wl[kt][k*32 + j] = (uint8_t)(wq & 0xFF);
                sg_col_sum_w[j] += (int32_t)wq;
            }
        }
    }

    /* ---- Build pointer arrays for partial_dual_scale's a_tiles/w_tiles. ---- */
    /* M1 = Σ A_h · W_h */
    for (int kt = 0; kt < Ktiles; kt++) { sg_aptrs[kt] = sg_Ah[kt]; sg_wptrs[kt] = sg_Wh[kt]; }
    partial_dual_scale(sg_M1, sg_aptrs, sg_wptrs, Ktiles,
                       act_tile, wt_tile, bias_lo, bias_hi, out_lo, out_hi);

    /* M3 = Σ A_l · W_h */
    for (int kt = 0; kt < Ktiles; kt++) { sg_aptrs[kt] = sg_Al[kt]; sg_wptrs[kt] = sg_Wh[kt]; }
    partial_dual_scale(sg_M3, sg_aptrs, sg_wptrs, Ktiles,
                       act_tile, wt_tile, bias_lo, bias_hi, out_lo, out_hi);

    /* M2_hmx = Σ A_h · W_l (treated as i8 reinterp; corrected below) */
    for (int kt = 0; kt < Ktiles; kt++) { sg_aptrs[kt] = sg_Ah[kt]; sg_wptrs[kt] = (const int8_t *)sg_Wl[kt]; }
    partial_dual_scale(sg_M2, sg_aptrs, sg_wptrs, Ktiles,
                       act_tile, wt_tile, bias_lo, bias_hi, out_lo, out_hi);

    /* M4_hmx = Σ A_l · W_l */
    for (int kt = 0; kt < Ktiles; kt++) { sg_aptrs[kt] = sg_Al[kt]; sg_wptrs[kt] = (const int8_t *)sg_Wl[kt]; }
    partial_dual_scale(sg_M4, sg_aptrs, sg_wptrs, Ktiles,
                       act_tile, wt_tile, bias_lo, bias_hi, out_lo, out_hi);

    /* ---- Topbit correction for u8·u8 done via u8·(i8 reinterp). ---- */
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++) {
            int32_t c2 = 0, c4 = 0;
            for (int kt = 0; kt < Ktiles; kt++) {
                for (int k = 0; k < 32; k++) {
                    uint8_t tb = (uint8_t)(sg_Wl[kt][k*32 + j] >> 7);
                    c2 += (int32_t)sg_Ah[kt][i*32 + k] * tb;
                    c4 += (int32_t)sg_Al[kt][i*32 + k] * tb;
                }
            }
            sg_M2c[i*32 + j] = c2;
            sg_M4c[i*32 + j] = c4;
        }

    /* ---- Combine into int32, then apply offset corrections.
     *
     * First: reconstruct Σ au·wu from the 4 partials (au, wu post-Cast u16).
     *   au·wu = 65536·(A_h·W_h_signed + signed-offset) + 256·(A_h·W_l + A_l·W_h) + A_l·W_l
     * where int16 wq = (W_h<<8)|W_l reinterpretation requires the subtraction
     * of col_sum_w·32768 term carried in the int16_matmul_hmx.c formula.
     *
     * Output: the real int32 sum_k a·w, where a,w are the ORIGINAL signed
     * int16 values (pre-Cast): a = au - 32768, w = wu - 32768.
     * out = Σ a·w = Σ(au-32768)(wu-32768)
     *     = Σ au·wu - 32768·row_sum_au - 32768·col_sum_wu + 32768²·K
     *
     * sum_au_wu from 4 partials (signed int16 wq in decomp matches
     * int16_matmul_hmx.c exactly — see its line 230). */
    for (int i = 0; i < 32; i++) {
        int32_t rsa = sg_row_sum_a[i];  /* signed a row sum */
        for (int j = 0; j < 32; j++) {
            int64_t M2_true = (int64_t)sg_M2[i*32 + j] + 256LL * sg_M2c[i*32 + j];
            int64_t M4_true = (int64_t)sg_M4[i*32 + j] + 256LL * sg_M4c[i*32 + j];
            /* Σ au · wq (where wq is signed int16 reconstructed from W_h,W_l): */
            int64_t sum_au_wq =
                  ((int64_t)sg_M1[i*32+j] << 16)
                + (M2_true << 8)
                + ((int64_t)sg_M3[i*32+j] << 8)
                +  M4_true;
            /* wq is signed = w (pre-Cast). So sum_au_wq = Σ au · w.
             * We want Σ a · w = Σ (au - 32768) · w = sum_au_wq - 32768·col_sum_w. */
            int64_t sum_a_w = sum_au_wq - ((int64_t)sg_col_sum_w[j] << 15);
            /* But col_sum_w here is Σ_k wq = Σ w (signed). ✓ */
            out[i*32 + j] = (int32_t)sum_a_w;
            (void)rsa;  /* unused in this path */
        }
    }
}

