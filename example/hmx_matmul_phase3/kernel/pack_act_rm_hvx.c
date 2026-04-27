/*
 * pack_act_rm_hvx.c — PackActivationU8RowMajor (Phase 3D.4 for V8).
 *
 * Produces 1 KiB row-major activation tiles consumed by V8's
 * `:cm` HMX kernel. tile[r*32 + k] = act[m_tile*32 + r][k_tile*32 + k].
 *
 * Simpler than pack_act_u8_hvx (which does 2-stream interleave for
 * plain mxmem). `:cm` reads row-major directly — no interleaving.
 *
 * Tensor contract:
 *   Input  0: uint8 act      [1, 1, M, K]           Flat4 + Direct
 *   Output 0: uint8 tile     [1, M/32, K/32, 1024]  Flat4 + Direct
 *
 * Declared multithreaded=true; QNN slices along M_tiles.
 */

#include "HTP/core/qhpi.h"
#include <stdint.h>
#include <string.h>

#ifdef __hexagon__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

static inline void pack_one_rm_tile(
    uint8_t       *tile,
    const uint8_t *a_base,
    uint32_t       K_full)
{
#if defined(__hexagon__)
    for (int r = 0; r < 32; r += 4) {
        const uint64_t *s0 = (const uint64_t *)&a_base[(r + 0) * K_full];
        const uint64_t *s1 = (const uint64_t *)&a_base[(r + 1) * K_full];
        const uint64_t *s2 = (const uint64_t *)&a_base[(r + 2) * K_full];
        const uint64_t *s3 = (const uint64_t *)&a_base[(r + 3) * K_full];
        uint64_t *d = (uint64_t *)&tile[r * 32];
        d[0] = s0[0]; d[1] = s0[1]; d[2] = s0[2]; d[3] = s0[3];
        d[4] = s1[0]; d[5] = s1[1]; d[6] = s1[2]; d[7] = s1[3];
        d[8] = s2[0]; d[9] = s2[1]; d[10] = s2[2]; d[11] = s2[3];
        d[12] = s3[0]; d[13] = s3[1]; d[14] = s3[2]; d[15] = s3[3];
    }
#else
    for (int r = 0; r < 32; r++)
        memcpy(&tile[r * 32], &a_base[r * K_full], 32);
#endif
}

void pack_act_rm_hvx_kernel_body(
    const uint8_t *a,
    uint8_t       *out,
    int M, int K,
    uint32_t mt_start, uint32_t mt_end)
{
    const int K_tiles = K / 32;
    (void)M;
#if defined(__hexagon__)
    if ((K % 128) == 0) {
        /* Fast path: 2-pass vshuff with single-bit Rt (32, then 64), the
         * same topology used by pack_wt_v3. Per row-major 4-row × 128-col
         * chunk, produces 4 output vectors (one per 32-col K-tile lane),
         * each 128 B = 4 rows × 32 cols already in row-major layout. 4
         * vmemu loads + 4 vshuff + 4 aligned vmem stores per 4 K-tiles
         * (≈3 HVX ops/tile vs old vmux+vror+vor's 12 ops/tile → ~4×
         * fewer ops for the same output).
         *
         *   Pass 1: vshuff(V1=row1, V0=row0, Rt=32) → P01.lo, P01.hi
         *           vshuff(V3=row3, V2=row2, Rt=32) → P23.lo, P23.hi
         *     P01.lo = [r0_0..31, r1_0..31, r0_64..95, r1_64..95]
         *     P01.hi = [r0_32..63, r1_32..63, r0_96..127, r1_96..127]
         *     P23.lo similar but for r2/r3
         *   Pass 2: vshuff(P23.lo, P01.lo, Rt=64) → T02
         *           vshuff(P23.hi, P01.hi, Rt=64) → T13
         *     T02.lo = [r0_0..31, r1_0..31, r2_0..31, r3_0..31]   ← K-tile 0 (cols 0..31)
         *     T13.lo = [r0_32..63, r1_32..63, r2_32..63, r3_32..63]  ← K-tile 1 (cols 32..63)
         *     T02.hi = [r0_64..95, r1_64..95, r2_64..95, r3_64..95]  ← K-tile 2 (cols 64..95)
         *     T13.hi = [r0_96..127, r1_96..127, r2_96..127, r3_96..127] ← K-tile 3 (cols 96..127) */
        for (uint32_t mt = mt_start; mt < mt_end; mt++) {
            const uint8_t *a_m = &a[mt * 32 * K];
            uint8_t *tiles_base = out + (mt * K_tiles) * 1024;
            for (int r0 = 0; r0 < 32; r0 += 4) {
                const int h_in_mt = r0 / 4;   /* 0..7 within mt */
                const uint8_t *s0 = &a_m[(r0 + 0) * K];
                const uint8_t *s1 = &a_m[(r0 + 1) * K];
                const uint8_t *s2 = &a_m[(r0 + 2) * K];
                const uint8_t *s3 = &a_m[(r0 + 3) * K];
                for (int kk = 0; kk < K; kk += 128) {
                    HVX_Vector V0, V1, V2, V3;
                    memcpy(&V0, &s0[kk], sizeof(HVX_Vector));
                    memcpy(&V1, &s1[kk], sizeof(HVX_Vector));
                    memcpy(&V2, &s2[kk], sizeof(HVX_Vector));
                    memcpy(&V3, &s3[kk], sizeof(HVX_Vector));

                    HVX_VectorPair P01 = Q6_W_vshuff_VVR(V1, V0, 32);
                    HVX_VectorPair P23 = Q6_W_vshuff_VVR(V3, V2, 32);

                    HVX_Vector p01lo = Q6_V_lo_W(P01);
                    HVX_Vector p01hi = Q6_V_hi_W(P01);
                    HVX_Vector p23lo = Q6_V_lo_W(P23);
                    HVX_Vector p23hi = Q6_V_hi_W(P23);

                    HVX_VectorPair T02 = Q6_W_vshuff_VVR(p23lo, p01lo, 64);
                    HVX_VectorPair T13 = Q6_W_vshuff_VVR(p23hi, p01hi, 64);

                    HVX_Vector t02lo = Q6_V_lo_W(T02);  /* K-tile g_base+0 */
                    HVX_Vector t02hi = Q6_V_hi_W(T02);  /* K-tile g_base+2 */
                    HVX_Vector t13lo = Q6_V_lo_W(T13);  /* K-tile g_base+1 */
                    HVX_Vector t13hi = Q6_V_hi_W(T13);  /* K-tile g_base+3 */

                    const int g_base = kk / 32;
                    uint8_t *t0 = tiles_base + (g_base + 0) * 1024 + h_in_mt * 128;
                    uint8_t *t1 = tiles_base + (g_base + 1) * 1024 + h_in_mt * 128;
                    uint8_t *t2 = tiles_base + (g_base + 2) * 1024 + h_in_mt * 128;
                    uint8_t *t3 = tiles_base + (g_base + 3) * 1024 + h_in_mt * 128;
                    *((HVX_Vector *)t0) = t02lo;
                    *((HVX_Vector *)t1) = t13lo;
                    *((HVX_Vector *)t2) = t02hi;
                    *((HVX_Vector *)t3) = t13hi;
                }
            }
        }
    } else {
        /* Slow fallback for K not multiple of 128. */
        for (uint32_t mt = mt_start; mt < mt_end; mt++) {
            for (int kt = 0; kt < K_tiles; kt++) {
                uint8_t *tile = out + (mt * K_tiles + kt) * 1024;
                for (int r = 0; r < 32; r++)
                    memcpy(&tile[r * 32],
                           &a[(mt * 32 + r) * K + kt * 32], 32);
            }
        }
    }
#else
    for (uint32_t mt = mt_start; mt < mt_end; mt++) {
        for (int kt = 0; kt < K_tiles; kt++) {
            uint8_t *tile = out + (mt * K_tiles + kt) * 1024;
            for (int r = 0; r < 32; r++)
                memcpy(&tile[r * 32],
                       &a[(mt * 32 + r) * K + kt * 32], 32);
        }
    }
#endif
}

static uint32_t pack_act_rm_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs,  const QHPI_Tensor *const *inputs)
{
    (void)num_outputs; (void)num_inputs;

    const uint8_t *a   = (const uint8_t *)qhpi_tensor_raw_data(inputs[0]);
    uint8_t       *out = (uint8_t *)qhpi_tensor_raw_data(outputs[0]);

    QHPI_Shape as = qhpi_tensor_shape(inputs[0]);
    const int M = (int)as.dims[as.rank - 2];
    const int K = (int)as.dims[as.rank - 1];
    const int M_tiles = M / 32;

    const uint32_t num_slices = qhpi_num_slices(handle);
    const uint32_t slice_idx  = qhpi_slice_number(handle);
    const uint32_t mt_start   = (uint32_t)((uint64_t)M_tiles * slice_idx)     / num_slices;
    const uint32_t mt_end     = (uint32_t)((uint64_t)M_tiles * (slice_idx+1)) / num_slices;

    pack_act_rm_hvx_kernel_body(a, out, M, K, mt_start, mt_end);
    return QHPI_Success;
}

static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};

static QHPI_Kernel_v1 sg_kernels[] = {
    {
        THIS_PKG_NAME_STR "::pack_act_rm_hvx",
        pack_act_rm_kernel,
        QHPI_RESOURCE_HVX,
        false, true, false, false,
        1, sig_inputs,
        1, sig_outputs,
        nullptr, 0, 0, nullptr, nullptr, nullptr,
    },
};

static QHPI_OpInfo_v1 sg_ops[] = {
    {
        THIS_PKG_NAME_STR "::PackActivationU8RowMajor",
        1, sg_kernels,
        nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    },
};

extern "C" void register_pack_act_rm_op(void)
{
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
