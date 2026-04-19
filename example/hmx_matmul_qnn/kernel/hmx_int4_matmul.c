/*
 * hmx_int4_matmul.c — K-accumulated int4×int16 matmul on HMX.
 *
 * Produces one 32×32 output tile per call, consuming the full K axis of the
 * (m_tile, n_tile) strip in one hot HMX accumulator. Two accumulation passes:
 *   P_hi = sum_k A_hi[m,k] · W[k,n]   (u8 · i8)
 *   P_lo = sum_k A_lo[m,k] · W[k,n]   (u8 · i8)
 *   out[m,n] = (P_hi << 8) + P_lo - 32768 · col_sum_w[n]
 *
 * Activation decomp:  a_u = a + 32768;  A_hi = a_u >> 8;  A_lo = a_u & 0xFF.
 * Weight path is u8·i8 HMX MAC (weight.b). int4 input range [-7,7] means
 * per-packet max magnitude 32·255·7 ≈ 57 K; even at K = 8192 we stay within
 * the 24-bit signed dual-scale readback range.
 *
 * Per (m_tile, n_tile) output: 2 × mxclracc + 2 × (K/32) MAC-load packets +
 * 4 convert-stores, vs the previous per-32³-tile kernel's
 * (K/32) × (2 × mxclracc + 2 MAC + 4 convert-stores) — i.e. we amortize all
 * per-K-tile overhead across the full K dim.
 */

#include "hmx_int4_matmul.h"
#include <string.h>

/* ============== HMX asm wrappers (shared with the previous per-tile kernel) */
static inline __attribute__((always_inline)) void hmx_clracc_i(void)
{ asm volatile("mxclracc" ::: "memory"); }

static inline __attribute__((always_inline))
void hmx_load_bias_i(const void *p)
{ asm volatile("bias = mxmem(%0)" :: "r"(p) : "memory"); }

/* Rt value controls some aspect of HMX mxmem tile geometry. Empirical values:
 *   0x07FF (2047)  — standard 2 KB tile, what our int16 kernel has used
 *   0x7FFF (32767) — ch03 uses this for activation.hf
 *   0x0780 (1920)  — ch03 uses this for weight.hf
 *   0x00FF         — shorter mask
 *   0xFFFF         — wider mask
 * Parameterized here so we can sweep by rebuilding with -DHMX_RT_ACT=N. */
#ifndef HMX_RT_ACT
#define HMX_RT_ACT 2047
#endif
#ifndef HMX_RT_WT
#define HMX_RT_WT  2047
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

/* ============== Tile packers (32×32 slice into HMX layout) ================ *
 * 4-byte packed writes — the HMX tile layout naturally groups 4 bytes per
 * stride-4 cell. Instead of 1024 scalar byte writes + memset, we do 512 u32
 * writes (activation) or 256 u32 writes (weight) with no read-modify-write.
 */
/* Row-strided version — a_rows points to 32 rows × row_stride bytes, and we
 * pack the first 32 columns of each row into the HMX tile.
 *
 * Kept scalar (u32-packed writes): HVX shuffle rewrite was tried but produced
 * wrong HMX tile content (vshuffe_b semantics de-interleave EVEN bytes only,
 * dropping odd-indexed bytes). Since this function is called once per
 * m_tile (amortized <1% of total runtime after P3c), HVX is not a win here.
 */
static void pack_activation_32x32_rs(uint8_t *tile, const uint8_t *a_rows, int row_stride)
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

static void pack_weight_32x32(int8_t *tile, const int8_t *w_32x32)
{
    /* Output per K-group (128 bytes): 32 cells, each a u32 = [K0,K1,K2,K3] */
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

/* ============== Persistent scratch (kept off stack per debugging notes) === */
#define HMX_INT4_MAX_K   8192
static uint8_t  sg_A_h[32 * HMX_INT4_MAX_K];
static uint8_t  sg_A_l[32 * HMX_INT4_MAX_K];
static int32_t  sg_col_sum_w[32];

/* VTCM layout constants.
 *  [0, 12 KiB)                                     fixed (bias, out_lo/hi, transient tile)
 *  [12 KiB, 12 KiB + Ktiles*2048)                  act_prepack_hi (K/32 tiles × 2 KB)
 *  [12 KiB + Ktiles*2048, 12 KiB + 2*Ktiles*2048)  act_prepack_lo
 *  [above, above + Ktiles*1024)                    wt_prepack   (K/32 tiles × 1 KB)
 *  [above, above + 128)                            col_sum_w    (32 int32)
 */
#define VTCM_OFF_WT_TILE    (2 * 1024)           /* 1 KB transient wt tile (unused in prepacked path) */
#define VTCM_OFF_BIAS_LO    (3 * 1024)           /* 256 B */
#define VTCM_OFF_BIAS_HI    (3 * 1024 + 256)     /* 256 B */
#define VTCM_OFF_OUT_LO     (4 * 1024)           /* 2 KB */
#define VTCM_OFF_OUT_HI     (6 * 1024)           /* 2 KB */
#define VTCM_OFF_PREPACK    (12 * 1024)
/* Activation tiles: [12K, 12K + 2*Ktiles*2048) — hi then lo */
#define VTCM_ACT_HI(k32)                   (VTCM_OFF_PREPACK + (k32) * 2048)
#define VTCM_ACT_LO(k32, Ktiles)           (VTCM_OFF_PREPACK + ((Ktiles) + (k32)) * 2048)
/* Weight tiles: right after activation prepack, 1 KB each. */
#define VTCM_WT_REGION_BASE(Ktiles)        (VTCM_OFF_PREPACK + 2 * (Ktiles) * 2048)
#define VTCM_WT(k32, Ktiles)               (VTCM_WT_REGION_BASE(Ktiles) + (k32) * 1024)
#define VTCM_OFF_COLSUM(Ktiles)            (VTCM_WT_REGION_BASE(Ktiles) + (Ktiles) * 1024)

/* ============== Public kernels ============================================ */

void hmx_int4_prepack_activation(
    const int16_t *__restrict__ a,
    int                         K,
    void          *__restrict__ vtcm_base)
{
    /* Legacy (reference) path: decompose to DDR scratch, then pack. Kept for
     * correctness-bisection only. New call sites use
     * hmx_int4_prepack_activation_fused which avoids the DDR intermediate. */
    const int total = 32 * K;
    for (int i = 0; i < total; i++) {
        uint32_t au = (uint32_t)((int32_t)a[i] + 32768);
        sg_A_h[i] = (uint8_t)(au >> 8);
        sg_A_l[i] = (uint8_t)(au & 0xFF);
    }
    const int Ktiles = K / 32;
    uint8_t *vt = (uint8_t *)vtcm_base;
    for (int kt = 0; kt < Ktiles; kt++) {
        pack_activation_32x32_rs(vt + VTCM_ACT_HI(kt),         &sg_A_h[kt * 32], K);
        pack_activation_32x32_rs(vt + VTCM_ACT_LO(kt, Ktiles), &sg_A_l[kt * 32], K);
    }
}

/* Fused: decompose + pack in one pass. Reads uint16 activation directly from
 * VTCM input tensor, writes packed HMX tiles directly to VTCM. No DDR
 * intermediate (eliminates sg_A_h / sg_A_l round-trip). */
void hmx_int4_prepack_activation_fused(
    const uint16_t *__restrict__ au,
    int                          M_full,
    int                          K,
    int                          m0,
    void           *__restrict__ vtcm_base)
{
    (void)M_full;
    uint8_t *vt = (uint8_t *)vtcm_base;
    const int Ktiles = K / 32;

    /* For each K-tile, pack 32 rows × 32 K-cols of (hi,lo) into two HMX tiles. */
    for (int kt = 0; kt < Ktiles; kt++) {
        uint8_t *tile_hi = vt + VTCM_ACT_HI(kt);
        uint8_t *tile_lo = vt + VTCM_ACT_LO(kt, Ktiles);
        int k0 = kt * 32;

        /* For each phys_row (16 physical rows × 2 streams = 32 logical rows). */
        for (int phys_row = 0; phys_row < 16; phys_row++) {
            uint32_t *__restrict__ dst_hi = (uint32_t *)(tile_hi + 128 * phys_row);
            uint32_t *__restrict__ dst_lo = (uint32_t *)(tile_lo + 128 * phys_row);
            /* Logical rows phys_row and phys_row+16 (the two streams). */
            const uint16_t *s0 = &au[(m0 + phys_row)      * K + k0];
            const uint16_t *s1 = &au[(m0 + phys_row + 16) * K + k0];
            for (int K32 = 0; K32 < 32; K32++) {
                uint16_t a0 = s0[K32];   /* already (signed + 32768) post-Cast */
                uint16_t a1 = s1[K32];
                /* Hi byte → stream 0 slot 1, stream 1 slot 3 */
                uint8_t a0_hi = (uint8_t)(a0 >> 8);
                uint8_t a1_hi = (uint8_t)(a1 >> 8);
                uint8_t a0_lo = (uint8_t)(a0 & 0xFF);
                uint8_t a1_lo = (uint8_t)(a1 & 0xFF);
                dst_hi[K32] = ((uint32_t)a1_hi << 24) | ((uint32_t)a0_hi << 8);
                dst_lo[K32] = ((uint32_t)a1_lo << 24) | ((uint32_t)a0_lo << 8);
            }
        }
    }
}

void hmx_int4_matmul_mn_using_prepacked_act(
    int32_t       *__restrict__ out,
    const int8_t  *__restrict__ w,
    int                         K,
    void          *__restrict__ vtcm_base)
{
    uint8_t  *vt       = (uint8_t *)vtcm_base;
    int8_t   *wt_tile  = (int8_t *)( vt + VTCM_OFF_WT_TILE);
    uint16_t *bias_lo  = (uint16_t *)(vt + VTCM_OFF_BIAS_LO);
    uint16_t *bias_hi  = (uint16_t *)(vt + VTCM_OFF_BIAS_HI);
    uint16_t *out_lo   = (uint16_t *)(vt + VTCM_OFF_OUT_LO);
    uint16_t *out_hi   = (uint16_t *)(vt + VTCM_OFF_OUT_HI);

    fill_bias_scale(bias_lo, 0x4000);
    fill_bias_scale(bias_hi, 0x2000);

    for (int j = 0; j < 32; j++) sg_col_sum_w[j] = 0;
    for (int k = 0; k < K; k++)
        for (int j = 0; j < 32; j++)
            sg_col_sum_w[j] += (int32_t)w[k * 32 + j];

    const int Ktiles = K / 32;
    static int32_t sg_P_hi[32 * 32];
    static int32_t sg_P_lo[32 * 32];

    /* P_hi = sum_k A_hi · W — activation from VTCM prepack, weight packed
     * scalar per K-iter. The scalar pack naturally interleaves VTCM MAC
     * reads, hiding load latency. */
    hmx_load_bias_i(bias_lo);
    hmx_clracc_i();
    for (int kt = 0; kt < Ktiles; kt++) {
        pack_weight_32x32(wt_tile, &w[kt * 32 * 32]);
        hmx_load_pair_u8_i8(vt + VTCM_ACT_HI(kt), wt_tile);
    }
    hmx_store_acc_uh_2x1_retain(out_lo);
    hmx_load_bias_i(bias_hi);
    hmx_store_acc_uh_2x1(out_hi);
    for (int ir = 0; ir < 32; ir++) {
        int phys_row = ir & 15, stream = ir >> 4;
        for (int jc = 0; jc < 32; jc++) {
            uint16_t lo = out_lo[phys_row * 64 + 2 * jc + stream];
            uint16_t hi = out_hi[phys_row * 64 + 2 * jc + stream];
            sg_P_hi[ir * 32 + jc] = ((int32_t)(int16_t)hi << 8) | ((int32_t)lo & 0xFF);
        }
    }

    /* P_lo = sum_k A_lo · W */
    hmx_load_bias_i(bias_lo);
    hmx_clracc_i();
    for (int kt = 0; kt < Ktiles; kt++) {
        pack_weight_32x32(wt_tile, &w[kt * 32 * 32]);
        hmx_load_pair_u8_i8(vt + VTCM_ACT_LO(kt, Ktiles), wt_tile);
    }
    hmx_store_acc_uh_2x1_retain(out_lo);
    hmx_load_bias_i(bias_hi);
    hmx_store_acc_uh_2x1(out_hi);
    for (int ir = 0; ir < 32; ir++) {
        int phys_row = ir & 15, stream = ir >> 4;
        for (int jc = 0; jc < 32; jc++) {
            uint16_t lo = out_lo[phys_row * 64 + 2 * jc + stream];
            uint16_t hi = out_hi[phys_row * 64 + 2 * jc + stream];
            sg_P_lo[ir * 32 + jc] = ((int32_t)(int16_t)hi << 8) | ((int32_t)lo & 0xFF);
        }
    }

    /* Combine. */
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++)
            out[i * 32 + j] = (sg_P_hi[i * 32 + j] << 8)
                            +  sg_P_lo[i * 32 + j]
                            - (sg_col_sum_w[j] << 15);
}

/* Fused weight variant: reads raw uint8 wu[K_full × N_full] post-Cast, does
 * -128 offset shift inline during pack and col_sum computation. Eliminates
 * the gather_w_col DDR-intermediate round-trip. */
void hmx_int4_matmul_mn_fused_weight(
    int32_t        *__restrict__ out,
    const uint8_t  *__restrict__ wu,
    int                          K,
    int                          N_full,
    int                          n0,
    void           *__restrict__ vtcm_base)
{
    uint8_t  *vt       = (uint8_t *)vtcm_base;
    int8_t   *wt_tile  = (int8_t *)( vt + VTCM_OFF_WT_TILE);
    uint16_t *bias_lo  = (uint16_t *)(vt + VTCM_OFF_BIAS_LO);
    uint16_t *bias_hi  = (uint16_t *)(vt + VTCM_OFF_BIAS_HI);
    uint16_t *out_lo   = (uint16_t *)(vt + VTCM_OFF_OUT_LO);
    uint16_t *out_hi   = (uint16_t *)(vt + VTCM_OFF_OUT_HI);

    fill_bias_scale(bias_lo, 0x4000);
    fill_bias_scale(bias_hi, 0x2000);

    /* Column sums of SIGNED weight (subtracting 128 inline). */
    for (int j = 0; j < 32; j++) sg_col_sum_w[j] = 0;
    for (int k = 0; k < K; k++) {
        const uint8_t *wu_row = &wu[k * N_full + n0];
        for (int j = 0; j < 32; j++)
            sg_col_sum_w[j] += (int32_t)wu_row[j] - 128;
    }

    const int Ktiles = K / 32;
    static int32_t sg_P_hi[32 * 32];
    static int32_t sg_P_lo[32 * 32];

    /* Inline weight pack (from raw wu, signed via -128 per element). */
    #define PACK_WT_FUSED(kt)                                                 \
        do {                                                                  \
            int _k0 = (kt) * 32;                                              \
            for (int kg = 0; kg < 8; kg++) {                                  \
                uint32_t *__restrict__ dst = (uint32_t *)(wt_tile + 128 * kg);\
                const uint8_t *r0 = &wu[(_k0 + kg*4 + 0) * N_full + n0];      \
                const uint8_t *r1 = &wu[(_k0 + kg*4 + 1) * N_full + n0];      \
                const uint8_t *r2 = &wu[(_k0 + kg*4 + 2) * N_full + n0];      \
                const uint8_t *r3 = &wu[(_k0 + kg*4 + 3) * N_full + n0];      \
                for (int col = 0; col < 32; col++) {                          \
                    uint32_t b0 = (uint32_t)(uint8_t)((int32_t)r0[col] - 128);\
                    uint32_t b1 = (uint32_t)(uint8_t)((int32_t)r1[col] - 128);\
                    uint32_t b2 = (uint32_t)(uint8_t)((int32_t)r2[col] - 128);\
                    uint32_t b3 = (uint32_t)(uint8_t)((int32_t)r3[col] - 128);\
                    dst[col] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);      \
                }                                                             \
            }                                                                 \
        } while (0)

    /* P_hi = sum_k A_hi · W */
    hmx_load_bias_i(bias_lo);
    hmx_clracc_i();
    for (int kt = 0; kt < Ktiles; kt++) {
        PACK_WT_FUSED(kt);
        hmx_load_pair_u8_i8(vt + VTCM_ACT_HI(kt), wt_tile);
    }
    hmx_store_acc_uh_2x1_retain(out_lo);
    hmx_load_bias_i(bias_hi);
    hmx_store_acc_uh_2x1(out_hi);
    for (int ir = 0; ir < 32; ir++) {
        int phys_row = ir & 15, stream = ir >> 4;
        for (int jc = 0; jc < 32; jc++) {
            uint16_t lo = out_lo[phys_row * 64 + 2 * jc + stream];
            uint16_t hi = out_hi[phys_row * 64 + 2 * jc + stream];
            sg_P_hi[ir * 32 + jc] = ((int32_t)(int16_t)hi << 8) | ((int32_t)lo & 0xFF);
        }
    }

    /* P_lo = sum_k A_lo · W */
    hmx_load_bias_i(bias_lo);
    hmx_clracc_i();
    for (int kt = 0; kt < Ktiles; kt++) {
        PACK_WT_FUSED(kt);
        hmx_load_pair_u8_i8(vt + VTCM_ACT_LO(kt, Ktiles), wt_tile);
    }
    hmx_store_acc_uh_2x1_retain(out_lo);
    hmx_load_bias_i(bias_hi);
    hmx_store_acc_uh_2x1(out_hi);
    for (int ir = 0; ir < 32; ir++) {
        int phys_row = ir & 15, stream = ir >> 4;
        for (int jc = 0; jc < 32; jc++) {
            uint16_t lo = out_lo[phys_row * 64 + 2 * jc + stream];
            uint16_t hi = out_hi[phys_row * 64 + 2 * jc + stream];
            sg_P_lo[ir * 32 + jc] = ((int32_t)(int16_t)hi << 8) | ((int32_t)lo & 0xFF);
        }
    }

    #undef PACK_WT_FUSED

    /* Combine. */
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++)
            out[i * 32 + j] = (sg_P_hi[i * 32 + j] << 8)
                            +  sg_P_lo[i * 32 + j]
                            - (sg_col_sum_w[j] << 15);
}

/* Pre-pack ALL weight K-tiles upfront so inner MAC loop has no scalar work
 * between HMX issues — enables back-to-back HMX MAC pipelining. */
void hmx_int4_prepack_weight_tiles(
    const int8_t  *__restrict__ w,
    int                         K,
    void          *__restrict__ vtcm_base)
{
    uint8_t *vt = (uint8_t *)vtcm_base;
    const int Ktiles = K / 32;
    /* col_sum_w in one pass. */
    int32_t *cs = (int32_t *)(vt + VTCM_OFF_COLSUM(Ktiles));
    for (int j = 0; j < 32; j++) cs[j] = 0;
    for (int kt = 0; kt < Ktiles; kt++) {
        pack_weight_32x32((int8_t *)(vt + VTCM_WT(kt, Ktiles)), &w[kt * 32 * 32]);
        for (int kk = 0; kk < 32; kk++)
            for (int j = 0; j < 32; j++)
                cs[j] += (int32_t)w[(kt * 32 + kk) * 32 + j];
    }
}

void hmx_int4_matmul_mn_all_prepacked(
    int32_t       *__restrict__ out,
    int                         K,
    void          *__restrict__ vtcm_base)
{
    uint8_t  *vt       = (uint8_t *)vtcm_base;
    uint16_t *bias_lo  = (uint16_t *)(vt + VTCM_OFF_BIAS_LO);
    uint16_t *bias_hi  = (uint16_t *)(vt + VTCM_OFF_BIAS_HI);
    uint16_t *out_lo   = (uint16_t *)(vt + VTCM_OFF_OUT_LO);
    uint16_t *out_hi   = (uint16_t *)(vt + VTCM_OFF_OUT_HI);

    fill_bias_scale(bias_lo, 0x4000);
    fill_bias_scale(bias_hi, 0x2000);

    const int Ktiles = K / 32;
    const int32_t *cs = (const int32_t *)(vt + VTCM_OFF_COLSUM(Ktiles));
    static int32_t sg_P_hi[32 * 32];
    static int32_t sg_P_lo[32 * 32];

    /* P_hi: back-to-back HMX issues, no scalar in between — pipeline enabled. */
    hmx_load_bias_i(bias_lo);
    hmx_clracc_i();
    for (int kt = 0; kt < Ktiles; kt++) {
        hmx_load_pair_u8_i8(vt + VTCM_ACT_HI(kt), vt + VTCM_WT(kt, Ktiles));
    }
    hmx_store_acc_uh_2x1_retain(out_lo);
    hmx_load_bias_i(bias_hi);
    hmx_store_acc_uh_2x1(out_hi);
    for (int ir = 0; ir < 32; ir++) {
        int phys_row = ir & 15, stream = ir >> 4;
        for (int jc = 0; jc < 32; jc++) {
            uint16_t lo = out_lo[phys_row * 64 + 2 * jc + stream];
            uint16_t hi = out_hi[phys_row * 64 + 2 * jc + stream];
            sg_P_hi[ir * 32 + jc] = ((int32_t)(int16_t)hi << 8) | ((int32_t)lo & 0xFF);
        }
    }

    /* P_lo */
    hmx_load_bias_i(bias_lo);
    hmx_clracc_i();
    for (int kt = 0; kt < Ktiles; kt++) {
        hmx_load_pair_u8_i8(vt + VTCM_ACT_LO(kt, Ktiles), vt + VTCM_WT(kt, Ktiles));
    }
    hmx_store_acc_uh_2x1_retain(out_lo);
    hmx_load_bias_i(bias_hi);
    hmx_store_acc_uh_2x1(out_hi);
    for (int ir = 0; ir < 32; ir++) {
        int phys_row = ir & 15, stream = ir >> 4;
        for (int jc = 0; jc < 32; jc++) {
            uint16_t lo = out_lo[phys_row * 64 + 2 * jc + stream];
            uint16_t hi = out_hi[phys_row * 64 + 2 * jc + stream];
            sg_P_lo[ir * 32 + jc] = ((int32_t)(int16_t)hi << 8) | ((int32_t)lo & 0xFF);
        }
    }

    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++)
            out[i * 32 + j] = (sg_P_hi[i * 32 + j] << 8)
                            +  sg_P_lo[i * 32 + j]
                            - (cs[j] << 15);
}

/* Legacy entry kept for API compat. */
void hmx_int4_matmul_mn_tile(
    int32_t       *__restrict__ out,
    const int16_t *__restrict__ a,
    const int8_t  *__restrict__ w,
    int                         K,
    void          *__restrict__ vtcm_base)
{
    hmx_int4_prepack_activation(a, K, vtcm_base);
    hmx_int4_matmul_mn_using_prepacked_act(out, w, K, vtcm_base);
}

#if 0  /* inlined original kept for reference diff */
{
    /* VTCM layout (fixed 12 KiB regardless of K; per-K-tile packing reuses
     * the same 2 KiB act_tile and 1 KiB wt_tile):
     *   [0, 2 KiB)     act_tile
     *   [2, 3 KiB)     wt_tile
     *   [3, 3.25 KiB)  bias_lo (f16 scale 1.0)
     *   [3.25, 3.5)    bias_hi (f16 scale 2^-8)
     *   [4, 6 KiB)     out_lo
     *   [6, 8 KiB)     out_hi
     *   [8, 12 KiB)    reserved for P2+ (double-buffer / prefetch)
     */
    uint8_t  *vt       = (uint8_t *)vtcm_base;
    uint8_t  *act_tile =             vt + 0 * 1024;
    int8_t   *wt_tile  = (int8_t *)( vt + 2 * 1024);
    uint16_t *bias_lo  = (uint16_t *)(vt + 3 * 1024);
    uint16_t *bias_hi  = (uint16_t *)(vt + 3 * 1024 + 256);
    uint16_t *out_lo   = (uint16_t *)(vt + 4 * 1024);
    uint16_t *out_hi   = (uint16_t *)(vt + 6 * 1024);

    fill_bias_scale(bias_lo, 0x4000);
    fill_bias_scale(bias_hi, 0x2000);

    /* ---- Decompose full 32×K activation into hi/lo byte streams ---- */
    /* Row-major: A_h[row*K + k] holds high byte of (a[row*K + k] + 32768). */
    const int total = 32 * K;
    for (int i = 0; i < total; i++) {
        uint32_t au = (uint32_t)((int32_t)a[i] + 32768);
        sg_A_h[i] = (uint8_t)(au >> 8);
        sg_A_l[i] = (uint8_t)(au & 0xFF);
    }

    /* ---- Column sums for the -32768·W correction ---- */
    for (int j = 0; j < 32; j++) sg_col_sum_w[j] = 0;
    for (int k = 0; k < K; k++)
        for (int j = 0; j < 32; j++)
            sg_col_sum_w[j] += (int32_t)w[k * 32 + j];

    /* ---- Helper: one K-accumulated HMX partial with dual-scale readback --- */
    /*
     * Runs:  clracc → for each k_tile {pack + activation.ub + weight.b} →
     *        dual-scale readback → writes 32×32 int32 into dst.
     * Accumulator stays hot across the entire inner K loop.
     */
#define K_ACCUM_PARTIAL(dst_i32, A_src)                                       \
    do {                                                                      \
        hmx_load_bias_i(bias_lo);                                             \
        hmx_clracc_i();                                                       \
        for (int k0 = 0; k0 < K; k0 += 32) {                                  \
            /* Row-strided pack: no sub-gather, read directly from           \
             * [32 × K] row-major A with stride K. */                         \
            pack_activation_32x32_rs(act_tile, &(A_src)[k0], K);              \
            pack_weight_32x32(wt_tile, &w[k0 * 32]);                          \
            hmx_load_pair_u8_i8(act_tile, wt_tile);                           \
        }                                                                     \
        hmx_store_acc_uh_2x1_retain(out_lo);                                  \
        hmx_load_bias_i(bias_hi);                                             \
        hmx_store_acc_uh_2x1(out_hi);                                         \
        for (int ir = 0; ir < 32; ir++) {                                     \
            int phys_row = ir & 15;                                           \
            int stream   = ir >> 4;                                           \
            for (int jc = 0; jc < 32; jc++) {                                 \
                uint16_t lo = out_lo[phys_row * 64 + 2 * jc + stream];        \
                uint16_t hi = out_hi[phys_row * 64 + 2 * jc + stream];        \
                (dst_i32)[ir * 32 + jc] =                                     \
                    ((int32_t)(int16_t)hi << 8) | ((int32_t)lo & 0xFF);       \
            }                                                                 \
        }                                                                     \
    } while (0)

    static int32_t sg_P_hi[32 * 32];
    static int32_t sg_P_lo[32 * 32];

    K_ACCUM_PARTIAL(sg_P_hi, sg_A_h);
    K_ACCUM_PARTIAL(sg_P_lo, sg_A_l);

#undef K_ACCUM_PARTIAL

    /* ---- Combine: out = (P_hi << 8) + P_lo - 32768 · col_sum_w ---- */
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            out[i * 32 + j] = (sg_P_hi[i * 32 + j] << 8)
                            +  sg_P_lo[i * 32 + j]
                            - (sg_col_sum_w[j] << 15);
        }
    }
}
#endif /* 0 */
