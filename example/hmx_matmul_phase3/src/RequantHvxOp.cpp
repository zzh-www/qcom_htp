/*
 * RequantHvxOp.cpp — standalone HVX requantization op (Phase 3D.3).
 *
 * Consumes V7_hmx's dual-scale readback (u16 lo + u16 hi tiles) plus
 * per-channel int32 multiplier + shared shift, produces i8 [M, N] output.
 *
 * Declared: QHPI_RESOURCE_HVX, multithreaded=true. QNN's scheduler
 * slices this across 4 HVX worker threads, and schedules it in parallel
 * with the next graph's MatMulV7 tile (HMX resource) when possible.
 *
 * Per (mt, nt) output tile decode pipeline (same math as V6's inline
 * requant, but extracted to a standalone op so it's not bound to the
 * HMX thread):
 *   acc_s0 = (hi_s0 << 8) | (lo_s0 & 0xFF)   [int24 -> int32]
 *   stage1 = SaturatingRoundingDoublingHighMul(acc, multiplier[n])
 *   stage2 = RoundingShiftRight(stage1, shift)
 *   out    = saturate_i8(stage2)
 *
 * Signatures:
 *   Input 0: rb_lo       [1, M_tiles, N_tiles, 1024] u16  TCM_Only
 *   Input 1: rb_hi       [1, M_tiles, N_tiles, 1024] u16  TCM_Only
 *   Input 2: multiplier  [N]                         i32  DDR_OR_TCM
 *   Input 3: shift       [1]                         i32  DDR_OR_TCM
 *   Output:  out         [1, 1, M, N]                i8   DDR_OR_TCM
 */

#include "HTP/core/qhpi.h"
#include <cstdint>
#include <cstring>

#ifdef __hexagon__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

extern "C" {
#include "../kernel/hmx_core_v2.h"
}

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

/* Scalar reference helpers (also used on aarch64 CPU fallback). */
static inline int32_t rq_srdhm(int32_t x, int32_t y)
{
    int64_t prod = (int64_t)x * (int64_t)y;
    int64_t nudge = (prod >= 0) ? (1LL << 30) : -(1LL << 30);
    int32_t result = (int32_t)((prod + nudge) >> 31);
    if (x == (int32_t)0x80000000 && y == (int32_t)0x80000000)
        result = 0x7fffffff;
    return result;
}

static inline int32_t rq_rshr(int32_t x, int shift)
{
    if (shift <= 0) return x << (-shift);
    int32_t mask = (1 << shift) - 1;
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    int32_t result = (x >> shift) + (remainder > threshold ? 1 : 0);
    return result;
}

static inline int8_t rq_sat_i8(int32_t x)
{
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

static inline uint32_t dim_at_rq(const QHPI_Shape &s, uint32_t i)
{
    return i < s.rank ? s.dims[i] : 1;
}

#ifdef __hexagon__
/* Process one (mt, nt) tile: 32 × 32 i8 output. Uses the same HVX kernel
 * body as hmx_core_v2.c::hmx_matmul_v2_requant_scatter_i8_hvx, duplicated
 * here (rather than calling across object files) to let the compiler
 * inline freely. */
static inline void requant_one_tile_hvx(
    int8_t        *out_tile_base,   /* &out[mt*32 * N + nt*32], N-stride rows */
    uint32_t       N,
    const uint16_t *lo_rb,          /* 2 KiB dual-scale lo tile */
    const uint16_t *hi_rb,          /* 2 KiB dual-scale hi tile */
    const int32_t *multiplier_tile, /* &multiplier[nt*32], 32 × i32 */
    int32_t        shift)
{
    const HVX_Vector v_mask16 = Q6_V_vsplat_R(0x0000FFFF);
    const HVX_Vector v_maskFF = Q6_V_vsplat_R(0x000000FF);
    const HVX_Vector v_zero   = Q6_V_vzero();

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
            Q6_Vw_vasl_VwR(v_s0_hi, 8), Q6_V_vand_VV(v_s0_lo, v_maskFF));
        HVX_Vector v_s1_acc = Q6_Vw_vadd_VwVw(
            Q6_Vw_vasl_VwR(v_s1_hi, 8), Q6_V_vand_VV(v_s1_lo, v_maskFF));

        HVX_Vector v_s0_mul = Q6_Vw_vmpye_VwVuh(v_s0_acc, v_mult);
        v_s0_mul            = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(v_s0_mul, v_s0_acc, v_mult);
        HVX_Vector v_s1_mul = Q6_Vw_vmpye_VwVuh(v_s1_acc, v_mult);
        v_s1_mul            = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(v_s1_mul, v_s1_acc, v_mult);

        HVX_Vector v_s0_shifted = Q6_Vw_vasr_VwR(v_s0_mul, shift);
        HVX_Vector v_s1_shifted = Q6_Vw_vasr_VwR(v_s1_mul, shift);

        HVX_Vector v_h = Q6_Vh_vpack_VwVw_sat(v_s1_shifted, v_s0_shifted);
        HVX_Vector v_b = Q6_Vb_vpack_VhVh_sat(v_zero, v_h);

        /* 32 bytes s0 → row phys_row, 32 bytes s1 → row phys_row+16.
         * Scalar u64 extract is unavoidable here without the merge-in-
         * place optimization (which requires 4-way nt batching; keep
         * simple here since MT=true gives us 4× parallelism). */
        const uint64_t *src = (const uint64_t *)&v_b;
        uint64_t *p0 = (uint64_t *)&out_tile_base[phys_row        * N];
        uint64_t *p1 = (uint64_t *)&out_tile_base[(phys_row + 16) * N];
        p0[0] = src[0]; p0[1] = src[1]; p0[2] = src[2]; p0[3] = src[3];
        p1[0] = src[4]; p1[1] = src[5]; p1[2] = src[6]; p1[3] = src[7];
    }
}
#endif

static uint32_t requant_hvx_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
    (void)num_outputs; (void)num_inputs;

    const uint16_t *rb_lo      = (const uint16_t *)qhpi_tensor_raw_data(inputs[0]);
    const uint16_t *rb_hi      = (const uint16_t *)qhpi_tensor_raw_data(inputs[1]);
    const int32_t  *multiplier = (const int32_t  *)qhpi_tensor_raw_data(inputs[2]);
    const int32_t  *shift_ptr  = (const int32_t  *)qhpi_tensor_raw_data(inputs[3]);
    int8_t         *out        = (int8_t *)qhpi_tensor_raw_data(outputs[0]);

    QHPI_Shape ls = qhpi_tensor_shape(inputs[0]);
    QHPI_Shape os = qhpi_tensor_shape(outputs[0]);
    const uint32_t M_tiles = dim_at_rq(ls, 1);
    const uint32_t N_tiles = dim_at_rq(ls, 2);
    const uint32_t N       = dim_at_rq(os, os.rank - 1);
    const int32_t  shift   = shift_ptr[0];

    /* Self-slice across M_tiles: each of `num_slices` HVX threads gets
     * a disjoint subset of M-rows. Slice i processes mt in
     * [start_mt .. end_mt). This is the MT=true self-slicing contract:
     * QNN calls us N times with slice indices 0..N-1. Without this, all
     * threads would stomp the same output bytes → race + slowdown. */
    const uint32_t num_slices = qhpi_num_slices(handle);
    const uint32_t slice_idx  = qhpi_slice_number(handle);
    const uint32_t start_mt   = (M_tiles * slice_idx)     / num_slices;
    const uint32_t end_mt     = (M_tiles * (slice_idx+1)) / num_slices;

#ifdef __hexagon__
    /* 4-way nt batched merge-in-place on staging row (same trick as V6
     * after Phase 3D.1 — see Agent project_v6_matmul_perf_2026-04-24.md).
     * Processes 4 consecutive nt tiles per mt before flushing their 32
     * output rows as 128-B aligned vmem stores. Eliminates the u64
     * extract cost that was ~1894 cyc/tile. */
    if ((N_tiles % 4) == 0) {
        /* Static staging in .bss — 4 KiB per slice thread. With MT=4 this
         * is 16 KiB total but each thread writes only its own slice rows
         * so no write contention. */
        static __thread int8_t row_staging[32][128] __attribute__((aligned(128)));

        for (uint32_t mt = start_mt; mt < end_mt; mt++) {
            for (uint32_t nt_base = 0; nt_base < N_tiles; nt_base += 4) {
                for (uint32_t b = 0; b < 4; b++) {
                    const uint32_t nt = nt_base + b;
                    const uint16_t *lo_tile = rb_lo + (mt * N_tiles + nt) * 1024;
                    const uint16_t *hi_tile = rb_hi + (mt * N_tiles + nt) * 1024;
                    const int32_t  *mult_tile = &multiplier[nt * 32];
                    /* Reuse V6's in-place merge helper. */
                    hmx_matmul_v2_requant_to_staging_i8_hvx(
                        (int8_t *)row_staging, b * 32,
                        lo_tile, hi_tile, mult_tile, shift);
                }
                /* Flush 32 rows × 128-B aligned stores to DDR output. */
                int8_t *out_row0 = &out[(mt * 32) * N + nt_base * 32];
                for (uint32_t r = 0; r < 32; r++) {
                    *((HVX_Vector *)&out_row0[r * N]) =
                        *((HVX_Vector *)&row_staging[r][0]);
                }
            }
        }
    } else {
        /* Fallback for odd N_tiles — per-tile scatter with u64 stores. */
        for (uint32_t mt = start_mt; mt < end_mt; mt++) {
            for (uint32_t nt = 0; nt < N_tiles; nt++) {
                const uint16_t *lo_tile = rb_lo + (mt * N_tiles + nt) * 1024;
                const uint16_t *hi_tile = rb_hi + (mt * N_tiles + nt) * 1024;
                const int32_t  *mult_tile = &multiplier[nt * 32];
                int8_t *out_tile = &out[mt * 32 * N + nt * 32];
                requant_one_tile_hvx(out_tile, N, lo_tile, hi_tile,
                                     mult_tile, shift);
            }
        }
    }
#else
    for (uint32_t mt = start_mt; mt < end_mt; mt++) {
        for (uint32_t nt = 0; nt < N_tiles; nt++) {
            const uint16_t *lo_tile = rb_lo + (mt * N_tiles + nt) * 1024;
            const uint16_t *hi_tile = rb_hi + (mt * N_tiles + nt) * 1024;
            const int32_t  *mult_tile = &multiplier[nt * 32];
            int8_t *out_tile = &out[mt * 32 * N + nt * 32];
            for (uint32_t phys_row = 0; phys_row < 16; phys_row++) {
                for (uint32_t jc = 0; jc < 32; jc++) {
                    int idx_s0 = phys_row * 64 + 2 * jc + 0;
                    int idx_s1 = phys_row * 64 + 2 * jc + 1;
                    int32_t acc_s0 = ((int32_t)(int16_t)hi_tile[idx_s0] << 8)
                                   | ((int32_t)lo_tile[idx_s0] & 0xFF);
                    int32_t acc_s1 = ((int32_t)(int16_t)hi_tile[idx_s1] << 8)
                                   | ((int32_t)lo_tile[idx_s1] & 0xFF);
                    int32_t m = mult_tile[jc];
                    int32_t q_s0 = rq_rshr(rq_srdhm(acc_s0, m), shift);
                    int32_t q_s1 = rq_rshr(rq_srdhm(acc_s1, m), shift);
                    out_tile[phys_row        * N + jc] = rq_sat_i8(q_s0);
                    out_tile[(phys_row + 16) * N + jc] = rq_sat_i8(q_s1);
                }
            }
        }
    }
#endif
    return QHPI_Success;
}

static QHPI_Tensor_Signature_v1 sig_inputs_rq[] = {
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},     /* rb_lo */
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},     /* rb_hi */
    {QHPI_Int32,   QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},   /* multiplier[N] */
    {QHPI_Int32,   QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},   /* shift */
};
static QHPI_Tensor_Signature_v1 sig_outputs_rq[] = {
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},   /* out i8 */
};

static QHPI_Kernel_v1 sg_kernels_rq[] = {
    {
        THIS_PKG_NAME_STR "::requant_hvx",
        requant_hvx_kernel,
        QHPI_RESOURCE_HVX,
        /* source_destructive */ false,
        /* multithreaded       */ true,   /* QNN slices across 4 HVX threads */
        /* variable_inputs     */ false,
        /* variable_outputs    */ false,
        4, sig_inputs_rq,
        1, sig_outputs_rq,
        nullptr, 0, 0, nullptr, nullptr, nullptr,
    },
};

static QHPI_OpInfo_v1 sg_ops_rq[] = {
    {
        THIS_PKG_NAME_STR "::RequantHvx",
        1, sg_kernels_rq,
        nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    },
};

extern "C" void register_requant_hvx_op() {
    qhpi_register_ops_v1(sizeof(sg_ops_rq) / sizeof(sg_ops_rq[0]),
                         sg_ops_rq, THIS_PKG_NAME_STR);
}
