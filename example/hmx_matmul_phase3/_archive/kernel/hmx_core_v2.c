/*
 * hmx_core_v2.c — HMX MatMul kernel using `:cm` + row-major activation.
 * Implements hmx_core_v2.h API.
 */
#include "hmx_core_v2.h"
#include <string.h>

#ifdef __hexagon__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

#ifndef HMX_RT_ACT_CM
/* r7 | 0x1c required; 2047 already covers 0x1c (= 0x7ff), so 2047 works. */
#define HMX_RT_ACT_CM (2047 | 0x1C)
#endif
#ifndef HMX_RT_WT
#define HMX_RT_WT  0x3FF
#endif

#if defined(__hexagon__)

static inline __attribute__((always_inline)) void hmx_clracc_i(void)
{ __asm__ volatile("mxclracc" ::: "memory"); }

static inline __attribute__((always_inline))
void hmx_load_bias_i(const void *p)
{ __asm__ volatile("bias = mxmem(%0)" :: "r"(p) : "memory"); }

/* `:cm` activation + plain weight — Agent A probe P=C (7.92 cyc/MAC). */
static inline __attribute__((always_inline))
void hmx_load_pair_cm(const void *act, const void *wt)
{
    __asm__ volatile(
        "{ activation.ub = mxmem(%0, %1):cm\n"
        "  weight.b      = mxmem(%2, %3) }\n"
        :: "r"(act), "r"(HMX_RT_ACT_CM),
           "r"(wt),  "r"(HMX_RT_WT)
        : "memory");
}

/* Phase-2 plain mxmem pair (2-stream activation, packed weight). */
static inline __attribute__((always_inline))
void hmx_load_pair_plain(const void *act, const void *wt)
{
    __asm__ volatile(
        "{ activation.ub = mxmem(%0, %1)\n"
        "  weight.b      = mxmem(%2, %3) }\n"
        :: "r"(act), "r"(2047),
           "r"(wt),  "r"(HMX_RT_WT)
        : "memory");
}

static inline __attribute__((always_inline))
void hmx_store_acc_uh_2x1_retain(void *out)
{ __asm__ volatile("mxmem(%0, %1):after:retain.uh = acc:2x1\n"
               :: "r"(out), "r"(0) : "memory"); }

static inline __attribute__((always_inline))
void hmx_store_acc_uh_2x1(void *out)
{ __asm__ volatile("mxmem(%0, %1):after.uh = acc:2x1\n"
               :: "r"(out), "r"(0) : "memory"); }

#endif /* __hexagon__ */


void hmx_core_v2_fill_bias(void *bias_vtcm)
{
    uint16_t *blo = (uint16_t *)bias_vtcm;
    uint16_t *bhi = blo + 128;
    for (int i = 0; i < 128; i++) { blo[i] = 0x4000; bhi[i] = 0x2000; }
}

/* Phase 2 activation pack: 2-stream interleaved (bytes [0, s0, 0, s1]
 * per 4B cell). 32 logical rows × 32 K-cols → 16 phys_rows × 128 B tile
 * = 2 KiB. Scalar here; could be HVX'd in a separate upstream op. */
void hmx_core_v2_gather_act_tile(
    uint8_t       *tile_vtcm,
    const uint8_t *au,
    uint32_t       K_full,
    uint32_t       m0,
    uint32_t       k0)
{
    memset(tile_vtcm, 0, 2048);
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        uint32_t *dst = (uint32_t *)(tile_vtcm + 128 * phys_row);
        const uint8_t *s0 = &au[(m0 + phys_row)      * K_full + k0];
        const uint8_t *s1 = &au[(m0 + phys_row + 16) * K_full + k0];
        for (int k = 0; k < 32; k++)
            dst[k] = ((uint32_t)s1[k] << 24) | ((uint32_t)s0[k] << 8);
    }
}

/* Gather 32 K-rows × 32 N-cols of weight into VTCM tile.
 *
 * Key layout decision: Agent A's silicon probe Scenario C used a
 * "row-major weight" implicitly (all-1 bytes in 1 KiB), same answer as
 * pre-Phase 3 weight packed format. But the PRM says HMX weight.b
 * expects a specific per-K-group packing (4 K-rows × 32 cols per 128 B).
 *
 * For this first Phase 3B cut, use Phase 2's pack_weight_32x32 layout:
 *   128 bytes per K-group (4 consecutive K-rows × 32 N-cols)
 *   8 K-groups per tile → 1024 B
 * This is the SAME as hmx_int4_matmul.c pack_weight_32x32 output.
 * Scalar loop here; HVX-ify via the upstream PackWeightToHmxTile op
 * in a later phase if profile shows it matters. */
void hmx_core_v2_gather_wt_tile(
    uint8_t       *tile_vtcm,
    const int8_t  *wu,
    uint32_t       N_full,
    uint32_t       k0,
    uint32_t       n0)
{
    /* Phase 2 packed format: 8 K-groups × 32 cells × 4 bytes each.
     * Each cell encodes 4 consecutive K-rows at one N col. Empirically
     * required for `weight.b = mxmem` even with `:cm` on activation
     * (Agent A probe used uniform all-1 weight so didn't discriminate). */
    for (int kg = 0; kg < 8; kg++) {
        uint32_t *dst = (uint32_t *)(tile_vtcm + 128 * kg);
        const uint8_t *r0 = (const uint8_t *)&wu[(k0 + kg * 4 + 0) * N_full + n0];
        const uint8_t *r1 = (const uint8_t *)&wu[(k0 + kg * 4 + 1) * N_full + n0];
        const uint8_t *r2 = (const uint8_t *)&wu[(k0 + kg * 4 + 2) * N_full + n0];
        const uint8_t *r3 = (const uint8_t *)&wu[(k0 + kg * 4 + 3) * N_full + n0];
        for (int col = 0; col < 32; col++) {
            dst[col] =  (uint32_t)r0[col]
                     | ((uint32_t)r1[col] << 8)
                     | ((uint32_t)r2[col] << 16)
                     | ((uint32_t)r3[col] << 24);
        }
    }
}

void hmx_matmul_v2_decode_scatter_hvx(
    int32_t       *out_tile_base,
    uint32_t       N,
    const uint16_t *lo_rb,
    const uint16_t *hi_rb)
{
#if defined(__hexagon__)
    /* rb layout per 128 B phys_row slot: 64 u16 = 32 (s0_jc, s1_jc) u16
     * pairs. Viewed as 32 u32 lanes, each lane = (s0_jc in low 16,
     * s1_jc in high 16). No halfword-deal needed — just mask/shift.
     *
     *   v_lo_in u32 lane jc = (lo_rb[2*jc+1] << 16) | lo_rb[2*jc+0]
     *                      = (s1_lo[jc] << 16) | s0_lo[jc]
     *
     * Isolate streams:
     *   s0_lo = low 16 bits (AND 0xFFFF)
     *   s1_lo = high 16 bits (logical shift right 16)
     *   s0_hi = sign-extend low 16 bits  (asl 16 then asr 16)
     *   s1_hi = sign-extend high 16 bits (asr 16)
     *
     * Then combine per V3 formula: out = (hi << 8) + (lo & 0xFF). */
    const HVX_Vector v_mask16 = Q6_V_vsplat_R(0x0000FFFF);
    const HVX_Vector v_maskFF = Q6_V_vsplat_R(0x000000FF);
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        HVX_Vector v_lo_in, v_hi_in;
        memcpy(&v_lo_in, (const uint8_t *)lo_rb + 128 * phys_row, sizeof(HVX_Vector));
        memcpy(&v_hi_in, (const uint8_t *)hi_rb + 128 * phys_row, sizeof(HVX_Vector));

        HVX_Vector v_s0_lo = Q6_V_vand_VV(v_lo_in, v_mask16);
        HVX_Vector v_s1_lo = Q6_Vuw_vlsr_VuwR(v_lo_in, 16);
        HVX_Vector v_s0_hi = Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(v_hi_in, 16), 16);
        HVX_Vector v_s1_hi = Q6_Vw_vasr_VwR(v_hi_in, 16);

        HVX_Vector v_s0_hi_shl  = Q6_Vw_vasl_VwR(v_s0_hi, 8);
        HVX_Vector v_s1_hi_shl  = Q6_Vw_vasl_VwR(v_s1_hi, 8);
        HVX_Vector v_s0_lo_mask = Q6_V_vand_VV(v_s0_lo, v_maskFF);
        HVX_Vector v_s1_lo_mask = Q6_V_vand_VV(v_s1_lo, v_maskFF);

        HVX_Vector v_s0_out = Q6_Vw_vadd_VwVw(v_s0_hi_shl, v_s0_lo_mask);
        HVX_Vector v_s1_out = Q6_Vw_vadd_VwVw(v_s1_hi_shl, v_s1_lo_mask);

        memcpy(&out_tile_base[phys_row        * N], &v_s0_out, sizeof(HVX_Vector));
        memcpy(&out_tile_base[(phys_row + 16) * N], &v_s1_out, sizeof(HVX_Vector));
    }
#else
    /* Scalar fallback — matches inline loop in V3 pre-HVX decode. */
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        for (int jc = 0; jc < 32; jc++) {
            int idx = phys_row * 64 + 2 * jc;
            uint16_t lo0 = lo_rb[idx + 0], hi0 = hi_rb[idx + 0];
            uint16_t lo1 = lo_rb[idx + 1], hi1 = hi_rb[idx + 1];
            out_tile_base[phys_row        * N + jc] =
                ((int32_t)(int16_t)hi0 << 8) | ((int32_t)lo0 & 0xFF);
            out_tile_base[(phys_row + 16) * N + jc] =
                ((int32_t)(int16_t)hi1 << 8) | ((int32_t)lo1 & 0xFF);
        }
    }
#endif
}

/* Scalar helper — TFLite saturating rounding doubling high mul. */
static inline int32_t srdhm(int32_t x, int32_t y)
{
    int64_t prod = (int64_t)x * (int64_t)y;
    /* Round to bit 31: add 2^30 then arithmetic shift right 31. */
    int64_t nudge = (prod >= 0) ? (1LL << 30) : -(1LL << 30);
    int32_t result = (int32_t)((prod + nudge) >> 31);
    /* Saturate: the only overflow case is x=INT32_MIN, y=INT32_MIN. */
    if (x == (int32_t)0x80000000 && y == (int32_t)0x80000000)
        result = 0x7fffffff;
    return result;
}

static inline int32_t rounding_shift_right(int32_t x, int shift)
{
    if (shift <= 0) return x << (-shift);
    int32_t mask = (1 << shift) - 1;
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    int32_t result = (x >> shift) + (remainder > threshold ? 1 : 0);
    return result;
}

static inline int8_t sat_i8(int32_t x)
{
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

void hmx_matmul_v2_requant_scatter_i8_hvx(
    int8_t        *out_tile_base,
    uint32_t       N,
    const uint16_t *lo_rb,
    const uint16_t *hi_rb,
    const int32_t *multiplier_tile,
    int32_t        shift)
{
#if defined(__hexagon__)
    /* HVX pipeline per phys_row (optimized):
     *   1. Decode rb into 2× int32 vectors (s0, s1)
     *   2. SRDHM via vmpye + vmpyoacc_s1_rnd_sat_shift
     *   3. Final vasr by `shift`
     *   4. Fuse s0/s1 pack: one vpack_VwVw_sat(s1, s0) → 64 i16 in one vec
     *      then one vpack_VhVh_sat(zero, h) → 64 i8 in low half of vec
     *   5. Split 64-byte result into 32B s0 (row phys_row) + 32B s1 (row +16)
     * Key: pack instructions take 2 full input vectors each, so combining
     * s0 and s1 into a single pack call halves the pack cost vs passing
     * zero as the unused half. */
    const HVX_Vector v_mask16 = Q6_V_vsplat_R(0x0000FFFF);
    const HVX_Vector v_maskFF = Q6_V_vsplat_R(0x000000FF);
    const HVX_Vector v_zero   = Q6_V_vzero();

    HVX_Vector v_mult;
    memcpy(&v_mult, multiplier_tile, sizeof(HVX_Vector));

    for (int phys_row = 0; phys_row < 16; phys_row++) {
        HVX_Vector v_lo_in, v_hi_in;
        memcpy(&v_lo_in, (const uint8_t *)lo_rb + 128 * phys_row, sizeof(HVX_Vector));
        memcpy(&v_hi_in, (const uint8_t *)hi_rb + 128 * phys_row, sizeof(HVX_Vector));

        /* Decode int24 acc into two int32 vectors. */
        HVX_Vector v_s0_lo = Q6_V_vand_VV(v_lo_in, v_mask16);
        HVX_Vector v_s1_lo = Q6_Vuw_vlsr_VuwR(v_lo_in, 16);
        HVX_Vector v_s0_hi = Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(v_hi_in, 16), 16);
        HVX_Vector v_s1_hi = Q6_Vw_vasr_VwR(v_hi_in, 16);
        HVX_Vector v_s0_acc = Q6_Vw_vadd_VwVw(
            Q6_Vw_vasl_VwR(v_s0_hi, 8),
            Q6_V_vand_VV(v_s0_lo, v_maskFF));
        HVX_Vector v_s1_acc = Q6_Vw_vadd_VwVw(
            Q6_Vw_vasl_VwR(v_s1_hi, 8),
            Q6_V_vand_VV(v_s1_lo, v_maskFF));

        /* SRDHM */
        HVX_Vector v_s0_mul = Q6_Vw_vmpye_VwVuh(v_s0_acc, v_mult);
        v_s0_mul            = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(v_s0_mul, v_s0_acc, v_mult);
        HVX_Vector v_s1_mul = Q6_Vw_vmpye_VwVuh(v_s1_acc, v_mult);
        v_s1_mul            = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(v_s1_mul, v_s1_acc, v_mult);

        /* External shift */
        HVX_Vector v_s0_shifted = Q6_Vw_vasr_VwR(v_s0_mul, shift);
        HVX_Vector v_s1_shifted = Q6_Vw_vasr_VwR(v_s1_mul, shift);

        /* FUSED PACK: s0 and s1 into a single 128-B vector.
         * vpack_VwVw_sat(Vu, Vv) → output halfwords [Vv[0..31], Vu[0..31]].
         * So pack(s1_shifted, s0_shifted) gives:
         *   hw[0..31]  = sat_h(s0[0..31])
         *   hw[32..63] = sat_h(s1[0..31])
         * Then vpack_VhVh_sat(Vu, Vv) → output bytes [Vv[0..63], Vu[0..63]]:
         *   b[0..31]   = sat_b(s0[0..31])
         *   b[32..63]  = sat_b(s1[0..31])
         *   b[64..127] = sat_b(v_zero's halfwords) = 0 */
        HVX_Vector v_h = Q6_Vh_vpack_VwVw_sat(v_s1_shifted, v_s0_shifted);
        HVX_Vector v_b = Q6_Vb_vpack_VhVh_sat(v_zero, v_h);

        /* Store low 32B → row phys_row (s0); next 32B → row phys_row+16 (s1).
         * Explicit 4× u64 stores instead of memcpy(32): memcpy may trampoline
         * through libc for non-compile-time-known small sizes, and scalar
         * byte-by-byte copy is ~5× slower than direct 64-bit stores. */
#ifdef V6_NO_STORE
        (void)v_b;
#else
        {
            const uint64_t *src = (const uint64_t *)&v_b;
            uint64_t *p0 = (uint64_t *)&out_tile_base[phys_row        * N];
            uint64_t *p1 = (uint64_t *)&out_tile_base[(phys_row + 16) * N];
            p0[0] = src[0]; p0[1] = src[1]; p0[2] = src[2]; p0[3] = src[3];
            p1[0] = src[4]; p1[1] = src[5]; p1[2] = src[6]; p1[3] = src[7];
        }
#endif
    }
#else
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        for (int jc = 0; jc < 32; jc++) {
            int idx_s0 = phys_row * 64 + 2 * jc + 0;
            int idx_s1 = phys_row * 64 + 2 * jc + 1;
            int32_t acc_s0 = ((int32_t)(int16_t)hi_rb[idx_s0] << 8)
                           | ((int32_t)lo_rb[idx_s0] & 0xFF);
            int32_t acc_s1 = ((int32_t)(int16_t)hi_rb[idx_s1] << 8)
                           | ((int32_t)lo_rb[idx_s1] & 0xFF);
            int32_t m = multiplier_tile[jc];
            int32_t q_s0 = rounding_shift_right(srdhm(acc_s0, m), shift);
            int32_t q_s1 = rounding_shift_right(srdhm(acc_s1, m), shift);
            out_tile_base[phys_row        * N + jc] = sat_i8(q_s0);
            out_tile_base[(phys_row + 16) * N + jc] = sat_i8(q_s1);
        }
    }
#endif
}

/* Staging variant of requant+scatter: writes 32-B of i8 output to a
 * specific 32-B slice of a row in `staging_rows`. For 4-way nt batching:
 * caller invokes this 4 times (col_offset = 0, 32, 64, 96), then does a
 * single 128-B aligned DDR store per row. */
void hmx_matmul_v2_requant_to_staging_i8_hvx(
    int8_t        *staging_rows,
    int            col_offset,
    const uint16_t *lo_rb,
    const uint16_t *hi_rb,
    const int32_t *multiplier_tile,
    int32_t        shift)
{
#if defined(__hexagon__)
    const HVX_Vector v_mask16 = Q6_V_vsplat_R(0x0000FFFF);
    const HVX_Vector v_maskFF = Q6_V_vsplat_R(0x000000FF);
    const HVX_Vector v_zero   = Q6_V_vzero();

#ifndef V6_PROBE_NO_STORE
    /* Build predicate: bits [col_offset..col_offset+31] true.
     *   pred_hi = bits [0..col_offset+31]  (vsetq2 handles R=128 as all-true,
     *                                       unlike vsetq which masks R mod 128)
     *   pred_lo = bits [0..col_offset-1]
     *   pred_at_col = pred_hi AND NOT pred_lo
     * Probe showed scalar u64 extract+store costs ~1890 cyc/tile; HVX
     * merge-in-place avoids the HVX→scalar spill. */
    HVX_VectorPred pred_lo = Q6_Q_vsetq_R(col_offset);
    HVX_VectorPred pred_hi = Q6_Q_vsetq2_R(col_offset + 32);
    HVX_VectorPred pred_at_col = Q6_Q_and_QQn(pred_hi, pred_lo);
#endif

    HVX_Vector v_mult;
    memcpy(&v_mult, multiplier_tile, sizeof(HVX_Vector));

    for (int phys_row = 0; phys_row < 16; phys_row++) {
        HVX_Vector v_lo_in, v_hi_in;
        memcpy(&v_lo_in, (const uint8_t *)lo_rb + 128 * phys_row, sizeof(HVX_Vector));
        memcpy(&v_hi_in, (const uint8_t *)hi_rb + 128 * phys_row, sizeof(HVX_Vector));

        HVX_Vector v_s0_lo = Q6_V_vand_VV(v_lo_in, v_mask16);
        HVX_Vector v_s1_lo = Q6_Vuw_vlsr_VuwR(v_lo_in, 16);
        HVX_Vector v_s0_hi = Q6_Vw_vasr_VwR(Q6_Vw_vasl_VwR(v_hi_in, 16), 16);
        HVX_Vector v_s1_hi = Q6_Vw_vasr_VwR(v_hi_in, 16);
        HVX_Vector v_s0_acc = Q6_Vw_vadd_VwVw(
            Q6_Vw_vasl_VwR(v_s0_hi, 8),
            Q6_V_vand_VV(v_s0_lo, v_maskFF));
        HVX_Vector v_s1_acc = Q6_Vw_vadd_VwVw(
            Q6_Vw_vasl_VwR(v_s1_hi, 8),
            Q6_V_vand_VV(v_s1_lo, v_maskFF));

        HVX_Vector v_s0_mul = Q6_Vw_vmpye_VwVuh(v_s0_acc, v_mult);
        v_s0_mul            = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(v_s0_mul, v_s0_acc, v_mult);
        HVX_Vector v_s1_mul = Q6_Vw_vmpye_VwVuh(v_s1_acc, v_mult);
        v_s1_mul            = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(v_s1_mul, v_s1_acc, v_mult);

        HVX_Vector v_s0_shifted = Q6_Vw_vasr_VwR(v_s0_mul, shift);
        HVX_Vector v_s1_shifted = Q6_Vw_vasr_VwR(v_s1_mul, shift);

        HVX_Vector v_h = Q6_Vh_vpack_VwVw_sat(v_s1_shifted, v_s0_shifted);
        HVX_Vector v_b = Q6_Vb_vpack_VhVh_sat(v_zero, v_h);

#ifdef V6_PROBE_NO_STORE
        (void)v_b;
        (void)staging_rows;
        (void)col_offset;
#else
        /* HVX merge-in-place: rotate v_b so s0 (bytes [0..31]) lands at
         * output bytes [col_offset..col_offset+31], and s1 (v_b bytes
         * [32..63]) lands at same offset in the s1 row. vror per HVX PRM:
         * output[k] = v[(k+Rt) mod 128]. To map input[i] -> output[i+d]
         * use Rt = -d mod 128. Hardware masks Rt mod 128 so negatives OK.
         *   For s0 (d = col_offset):      Rt = -col_offset
         *   For s1 (d = col_offset - 32): Rt = 32 - col_offset
         *
         * Aligned vmem_a on rows — staging_rows is 128-B aligned per the
         * V6Op `static ... aligned(128)` declaration. */
        HVX_Vector v_s0_rot = Q6_V_vror_VR(v_b, -col_offset);
        HVX_Vector v_s1_rot = Q6_V_vror_VR(v_b, 32 - col_offset);

        HVX_Vector *p0 = (HVX_Vector *)(staging_rows + phys_row        * 128);
        HVX_Vector *p1 = (HVX_Vector *)(staging_rows + (phys_row + 16) * 128);
        *p0 = Q6_V_vmux_QVV(pred_at_col, v_s0_rot, *p0);
        *p1 = Q6_V_vmux_QVV(pred_at_col, v_s1_rot, *p1);
#endif
    }
#else
    /* Scalar fallback: compute + store to staging. */
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        for (int jc = 0; jc < 32; jc++) {
            int idx_s0 = phys_row * 64 + 2 * jc + 0;
            int idx_s1 = phys_row * 64 + 2 * jc + 1;
            int32_t acc_s0 = ((int32_t)(int16_t)hi_rb[idx_s0] << 8)
                           | ((int32_t)lo_rb[idx_s0] & 0xFF);
            int32_t acc_s1 = ((int32_t)(int16_t)hi_rb[idx_s1] << 8)
                           | ((int32_t)lo_rb[idx_s1] & 0xFF);
            int32_t m = multiplier_tile[jc];
            int32_t q_s0 = rounding_shift_right(srdhm(acc_s0, m), shift);
            int32_t q_s1 = rounding_shift_right(srdhm(acc_s1, m), shift);
            staging_rows[phys_row        * 128 + col_offset + jc] = sat_i8(q_s0);
            staging_rows[(phys_row + 16) * 128 + col_offset + jc] = sat_i8(q_s1);
        }
    }
#endif
}

void hmx_matmul_v2_core_mn_cm(
    const uint8_t *act_tiles,
    const uint8_t *wt_tiles,
    uint32_t       K_tiles,
    void          *bias_vtcm,
    uint16_t      *out_top_lo,
    uint16_t      *out_top_hi,
    uint16_t      *out_bot_lo,
    uint16_t      *out_bot_hi)
{
#if defined(__hexagon__)
    uint16_t *blo = (uint16_t *)bias_vtcm;
    uint16_t *bhi = blo + 128;
    (void)out_bot_lo; (void)out_bot_hi;

    hmx_load_bias_i(blo);
    hmx_clracc_i();
    for (uint32_t kt = 0; kt < K_tiles; kt++) {
        /* `:cm` + row-major: act tile stride = 1 KiB; wt tile = 1 KiB. */
        hmx_load_pair_cm(act_tiles + kt * 1024,
                         wt_tiles  + kt * 1024);
    }
    hmx_store_acc_uh_2x1_retain(out_top_lo);
    hmx_load_bias_i(bhi);
    hmx_store_acc_uh_2x1(out_top_hi);
#else
    (void)act_tiles; (void)wt_tiles; (void)K_tiles; (void)bias_vtcm;
    (void)out_top_lo; (void)out_top_hi; (void)out_bot_lo; (void)out_bot_hi;
#endif
}

void hmx_core_v2_flat_pack_act_tile(
    uint8_t       *packed_tile,
    const uint8_t *act_raw,
    uint32_t       K_full,
    uint32_t       mt,
    uint32_t       kt)
{
    /* Tile covers rows [mt*32, mt*32+32) and K-bytes [kt*32, kt*32+32). */
    const uint8_t *a_base = &act_raw[mt * 32 * K_full + kt * 32];

#if defined(__hexagon__)
    /* Same HVX pack as pack_act_u8_hvx.c (see that file for correctness
     * proof). Per-phys_row: s0 = row r, s1 = row r+16. Target layout
     * after two byte-vshuff per 128-B chunk:
     *   bytes  0..31  = 0
     *   bytes 32..63  = s0[0..31]
     *   bytes 64..95  = 0
     *   bytes 96..127 = s1[0..31]
     * 2× Q6_Vb_vshuff_Vb produces `[0, s0[0], 0, s1[0], 0, s0[1], ...]`. */
    const HVX_Vector     v_zero   = Q6_V_vzero();
    const HVX_VectorPred pred_32  = Q6_Q_vsetq_R(32);
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        const uint8_t *s0_src = &a_base[phys_row        * K_full];
        const uint8_t *s1_src = &a_base[(phys_row + 16) * K_full];

        HVX_Vector v_s0_raw, v_s1_raw;
        memcpy(&v_s0_raw, s0_src, sizeof(HVX_Vector));
        memcpy(&v_s1_raw, s1_src, sizeof(HVX_Vector));

        HVX_Vector v_s0 = Q6_V_vmux_QVV(pred_32, v_s0_raw, v_zero);
        HVX_Vector v_s1 = Q6_V_vmux_QVV(pred_32, v_s1_raw, v_zero);

        HVX_Vector v_s0_pos = Q6_V_vror_VR(v_s0, 32);
        HVX_Vector v_s1_pos = Q6_V_vror_VR(v_s1, 96);
        HVX_Vector v_in     = Q6_V_vor_VV(v_s0_pos, v_s1_pos);

        HVX_Vector step1  = Q6_Vb_vshuff_Vb(v_in);
        HVX_Vector final_ = Q6_Vb_vshuff_Vb(step1);

        *((HVX_Vector *)(packed_tile + 128 * phys_row)) = final_;
    }
#else
    memset(packed_tile, 0, 2048);
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        const uint8_t *s0 = &a_base[phys_row        * K_full];
        const uint8_t *s1 = &a_base[(phys_row + 16) * K_full];
        uint32_t *dst = (uint32_t *)(packed_tile + 128 * phys_row);
        for (int k = 0; k < 32; k++)
            dst[k] = ((uint32_t)s1[k] << 24) | ((uint32_t)s0[k] << 8);
    }
#endif
}


void hmx_matmul_v2_core_mn(
    const uint8_t *act_tiles,
    const uint8_t *wt_tiles,
    uint32_t       K_tiles,
    void          *bias_vtcm,
    uint16_t      *out_top_lo,
    uint16_t      *out_top_hi,
    uint16_t      *out_bot_lo,
    uint16_t      *out_bot_hi)
{
#if defined(__hexagon__)
    uint16_t *blo = (uint16_t *)bias_vtcm;
    uint16_t *bhi = blo + 128;

    /* Phase 2 plain mxmem path: single MAC sequence covers 32 logical
     * rows via 2-stream activation tile; dual-scale readback gives full
     * int24 range. out_bot buffers unused in this path. */
    (void)out_bot_lo; (void)out_bot_hi;

    hmx_load_bias_i(blo);
    hmx_clracc_i();
    for (uint32_t kt = 0; kt < K_tiles; kt++) {
        /* act_tile stride = 2 KiB (2-stream pack); wt_tile = 1 KiB. */
        hmx_load_pair_plain(act_tiles + kt * 2048,
                            wt_tiles  + kt * 1024);
    }
    hmx_store_acc_uh_2x1_retain(out_top_lo);
    hmx_load_bias_i(bhi);
    hmx_store_acc_uh_2x1(out_top_hi);
#else
    (void)act_tiles; (void)wt_tiles; (void)K_tiles; (void)bias_vtcm;
    (void)out_top_lo; (void)out_top_hi; (void)out_bot_lo; (void)out_bot_hi;
#endif
}
