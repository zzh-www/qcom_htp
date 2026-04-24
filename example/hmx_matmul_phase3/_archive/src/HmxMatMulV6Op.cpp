/*
 * HmxMatMulV6Op.cpp — QUANTIZED MatMul: u8 × i8 → i8.
 *
 * Symmetric quantization (zp_a = zp_w = zp_out = 0):
 *   activation: u8, per-tensor quantized (scale_a)
 *   weight    : i8, per-output-channel quantized (scale_w[n])
 *   output    : i8, per-tensor quantized (scale_out)
 *
 * Host prep folds all three into per-N int32 multiplier + shared shift:
 *   scale_combined[n] = scale_a × scale_w[n] / scale_out
 *   multiplier[n]     = round(scale_combined[n] × 2^31)   // Q0.31 fixed-point
 *   shift             = common right-shift to normalize across channels
 *
 * Runtime per output cell:
 *   raw_acc    = Σₖ u8_act × i8_wt                              (HMX MAC, int24)
 *   stage1     = SaturatingRoundingDoublingHighMul(raw_acc, mult[n])
 *   stage2     = RoundingShiftRight(stage1, shift)
 *   out[m][n]  = saturate_i8(stage2)
 *
 * Signatures:
 *   Input 0: act_raw     [1, 1, M, K]                u8  Flat4 + Direct
 *                        (Phase 3D.0: act consumed as raw row-major; the
 *                        2-stream pack happens inside this op to remove
 *                        the separate pack_act graph node. We tried
 *                        Crouton_8 + Indirect briefly — QNN's
 *                        ForceFormat_Crouton_b was 5× slower than our
 *                        pack_act because Crouton_8 is a 12.5%-density
 *                        layout designed for conv, not matmul.)
 *   Input 1: packed_wt   [1, N_tiles, K_tiles, 1024] u8  TCM_Only (same as V3)
 *   Input 2: scratch     VTCM bias + readback + per-mt packed-act buf
 *   Input 3: multiplier  [N]                         i32 DDR_OR_TCM
 *   Param  0: shift      int32
 *   Output:  out         [1, 1, M, N]                i8  DDR_OR_TCM
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

static inline uint32_t dim_at_v6(const QHPI_Shape &s, uint32_t i)
{
    return i < s.rank ? s.dims[i] : 1;
}

static uint32_t hmx_matmul_v6_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
    (void)num_outputs; (void)num_inputs; (void)handle;

#if !defined(__hexagon__) || defined(SCALAR_ONLY)
    int8_t *out = (int8_t *)qhpi_tensor_raw_data(outputs[0]);
    QHPI_Shape os = qhpi_tensor_shape(outputs[0]);
    uint32_t total = 1;
    for (uint32_t i = 0; i < os.rank; i++) total *= os.dims[i];
    memset(out, 0, total);
    return QHPI_Success;
#else
    /* Input 0 is Flat4 + Direct — row-major [M, K] u8 in DDR. */
    const uint8_t *act_raw    = (const uint8_t *)qhpi_tensor_raw_data(inputs[0]);
    const uint8_t *packed_wt  = (const uint8_t *)qhpi_tensor_raw_data(inputs[1]);
    void          *vtcm       = qhpi_tensor_raw_data(inputs[2]);
    const int32_t *multiplier = (const int32_t *)qhpi_tensor_raw_data(inputs[3]);
    int8_t        *out        = (int8_t *)qhpi_tensor_raw_data(outputs[0]);

    /* Shift is passed via scalar param. QHPI exposes params through the
     * handle; for now read it from input tensor 4 as a 1-element i32 to
     * avoid the param plumbing. Host harness will create a [1]-tensor
     * holding the shift value. */
    int32_t shift = 0;
    if (num_inputs > 4) {
        const int32_t *shift_ptr = (const int32_t *)qhpi_tensor_raw_data(inputs[4]);
        shift = shift_ptr[0];
    }

    QHPI_Shape as = qhpi_tensor_shape(inputs[0]);      /* [1, 1, M, K] */
    QHPI_Shape ws = qhpi_tensor_shape(inputs[1]);      /* [1, N_tiles, K_tiles, 1024] */
    QHPI_Shape os = qhpi_tensor_shape(outputs[0]);
    const uint32_t M       = dim_at_v6(as, as.rank - 2);
    const uint32_t K       = dim_at_v6(as, as.rank - 1);
    const uint32_t M_tiles = M / 32;
    const uint32_t K_tiles = K / 32;
    const uint32_t N_tiles = dim_at_v6(ws, 1);
    const uint32_t N = dim_at_v6(os, os.rank - 1);

    /* VTCM scratch layout (128 KB at prepare — see run_matmul_v6_graph.cpp):
     *   0x00000..0x007FF  bias         (2 KiB)
     *   0x00800..0x00FFF  out_top_lo   (2 KiB)
     *   0x01000..0x017FF  out_top_hi   (2 KiB)
     *   0x01800..0x01FFF  out_bot_lo   (2 KiB, unused by current path)
     *   0x02000..0x027FF  out_bot_hi   (2 KiB, unused)
     *   0x02800..0x037FF  row_staging  (4 KiB, 32×128 HVX merge-in-place)
     *   0x03800..........  act_pack_mt (K_tiles × 2 KiB, per-mt pre-packed
     *                                   activation tiles; max 64 KiB at K=1024)
     */
    uint8_t *vt = (uint8_t *)vtcm;
    void    *bias_vtcm = vt;
    hmx_core_v2_fill_bias(bias_vtcm);
    uint16_t *out_top_lo = (uint16_t *)(vt + 0x00800);
    uint16_t *out_top_hi = (uint16_t *)(vt + 0x01000);
    uint16_t *out_bot_lo = (uint16_t *)(vt + 0x01800);
    uint16_t *out_bot_hi = (uint16_t *)(vt + 0x02000);
    int8_t  *row_staging = (int8_t  *)(vt + 0x02800);
    uint8_t *act_pack_mt = (uint8_t *)(vt + 0x03800);   /* K_tiles * 2048 B */

    /* 4-way nt batch with VTCM staging of output. Compute 4 consecutive
     * nt tiles, stage their i8 results into 32×128 B row buf, flush via
     * 32 aligned 128-B vmem stores to DDR. All-VTCM staging stores are
     * bank-fast; only final 32 aligned DDR writes pay cache-line cost. */
    if ((N_tiles % 4) == 0) {
        for (uint32_t mt = 0; mt < M_tiles; mt++) {
            /* Gather + pack all K_tiles of this mt into VTCM once, reuse
             * across the entire nt sweep. Replaces the separate pack_act
             * graph op — ~200 cyc/tile × K_tiles, total ~same work as
             * before but eliminates packed_act intermediate tensor. */
            for (uint32_t kt = 0; kt < K_tiles; kt++) {
                hmx_core_v2_flat_pack_act_tile(
                    act_pack_mt + kt * 2048,
                    act_raw, K, mt, kt);
            }
            for (uint32_t nt_base = 0; nt_base < N_tiles; nt_base += 4) {
                for (int b = 0; b < 4; b++) {
                    const uint32_t nt = nt_base + b;
                    const uint8_t *act_tiles = act_pack_mt;
                    const uint8_t *wt_tiles  = packed_wt  + (nt * K_tiles) * 1024;

                    hmx_matmul_v2_core_mn(act_tiles, wt_tiles, K_tiles, bias_vtcm,
                                           out_top_lo, out_top_hi,
                                           out_bot_lo, out_bot_hi);

#ifndef V6_PROBE_NO_REQUANT
                    const int32_t *mult_tile = &multiplier[nt * 32];
                    hmx_matmul_v2_requant_to_staging_i8_hvx(
                        row_staging, b * 32,
                        out_top_lo, out_top_hi, mult_tile, shift);
#endif
                }

#ifndef V6_PROBE_NO_FLUSH
                /* Flush 32 rows of 128-B aligned stores to DDR. */
                int8_t *out_row0 = &out[(mt * 32) * N + nt_base * 32];
                for (int r = 0; r < 32; r++) {
                    *((HVX_Vector *)&out_row0[r * N]) =
                        *((HVX_Vector *)&row_staging[r * 128]);
                }
#endif
            }
        }
    } else {
        /* Fallback for non-4-divisible N_tiles. */
        for (uint32_t mt = 0; mt < M_tiles; mt++) {
            for (uint32_t kt = 0; kt < K_tiles; kt++) {
                hmx_core_v2_flat_pack_act_tile(
                    act_pack_mt + kt * 2048,
                    act_raw, K, mt, kt);
            }
            for (uint32_t nt = 0; nt < N_tiles; nt++) {
                const uint8_t *act_tiles = act_pack_mt;
                const uint8_t *wt_tiles  = packed_wt  + (nt * K_tiles) * 1024;
                hmx_matmul_v2_core_mn(act_tiles, wt_tiles, K_tiles, bias_vtcm,
                                       out_top_lo, out_top_hi,
                                       out_bot_lo, out_bot_hi);
                int8_t *out_tile_base = &out[(mt * 32) * N + nt * 32];
                const int32_t *mult_tile = &multiplier[nt * 32];
                hmx_matmul_v2_requant_scatter_i8_hvx(out_tile_base, N,
                                                      out_top_lo, out_top_hi,
                                                      mult_tile, shift);
            }
        }
    }
    return QHPI_Success;
#endif
}

static QHPI_Tensor_Signature_v1 sig_inputs_v6[] = {
    /* Phase 3D.0: act consumed as QNN-native Crouton_8 blocks (QNN auto-
     * inserts ForceFormat_Crouton_b producer before this op). Kernel
     * accesses via qhpi_tensor_block_table(). */
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},      /* act_raw */
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},     /* packed_wt */
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},     /* scratch */
    {QHPI_Int32,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},   /* multiplier[N] */
    {QHPI_Int32,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},   /* shift [1] */
};
/* Output placed in VTCM: V6 writes 32-B unaligned (relative to DDR
 * cache-line) per row × 32 rows per tile, triggering partial-line
 * flushes if going directly to DDR. Forcing TCM_Only lets the 32-B
 * stores hit VTCM fast, and QNN will DMA the final tensor to DDR
 * (bulk aligned) at graph exit. */
static QHPI_Tensor_Signature_v1 sig_outputs_v6[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};

static QHPI_Kernel_v1 sg_kernels_v6[] = {
    {
        THIS_PKG_NAME_STR "::hmx_matmul_v6",
        hmx_matmul_v6_kernel,
        QHPI_RESOURCE_HMX,
        false, false, false, false,
        5, sig_inputs_v6,
        1, sig_outputs_v6,
        nullptr, 0, 0, nullptr, nullptr, nullptr,
    },
};

static QHPI_OpInfo_v1 sg_ops_v6[] = {
    {
        THIS_PKG_NAME_STR "::MatMulV6",
        1, sg_kernels_v6,
        nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    },
};

extern "C" void register_hmx_matmul_v6_op() {
    qhpi_register_ops_v1(sizeof(sg_ops_v6) / sizeof(sg_ops_v6[0]),
                         sg_ops_v6, THIS_PKG_NAME_STR);
}
