/*
 * pack_act_hvx.c — PackActivationToHmxTile op (Phase 3B, Path W).
 *
 * Splits the activation-prepack work previously fused inside the Phase 2
 * MatMul op (hmx_int4_prepack_activation_fused in
 * example/hmx_matmul_qnn/kernel/hmx_int4_matmul.c). This op runs on HVX
 * threads only and is marked multithreaded=true so QHPI's self-slicer can
 * distribute the M_tile index across the four HVX worker threads.
 *
 * Algorithm (verbatim from Phase 2):
 *   uint16 a → a_u = a + 32768 (interpret as uint16 already since
 *              activations are stored as uint16 post-Cast in the QNN graph,
 *              which applies the +32768 offset when feeding int16).
 *   A_hi = a_u >> 8
 *   A_lo = a_u & 0xFF
 *   Both streams packed into 2 KB HMX activation tile layout:
 *     tile[128 * phys_row + 4 * K + stream] = a_stream[phys_row + 16*stream][K]
 *   The intrinsic shortcut: mask + halfword-shuffle (hi), asl + halfword-
 *   shuffle (lo), producing one HVX vector (= one phys_row) per 32 K cols.
 *
 * Tensor contract:
 *   Input  0:  uint16 act      [1, 1, M, K]           Flat4 + Direct
 *   Output 0:  uint8  tile_hi  [1, M/32, K/32, 2048]  Flat4 + Direct
 *   Output 1:  uint8  tile_lo  [1, M/32, K/32, 2048]  Flat4 + Direct
 *
 * Each 2 KB output entry is one HMX activation tile ready for
 *   `activation.ub = mxmem(ptr, 0x7FF)`.
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

/* -------------------------------------------------------------------
 * Core HVX packer — identical body to Phase 2 prepack_activation_fused,
 * but writes into two contiguous 2 KB tiles (hi, lo) addressed by
 * per-(m_tile, k_tile) offsets.
 * ------------------------------------------------------------------- */
static inline void pack_one_tile(
    uint8_t *__restrict__ tile_hi,  /* 2048 B */
    uint8_t *__restrict__ tile_lo,  /* 2048 B */
    const uint16_t *__restrict__ au_base, /* pointer to this (m_tile, k_tile) start */
    int K_stride)                   /* full K dimension (row stride in elements) */
{
#ifdef __hexagon__
    const HVX_Vector v_mask_FF00 = Q6_V_vsplat_R(0xFF00FF00);
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        const uint16_t *s0 = &au_base[phys_row        * K_stride];
        const uint16_t *s1 = &au_base[(phys_row + 16) * K_stride];
        union { HVX_Vector v; uint8_t b[128]; } u;
        memcpy(&u.b[0],  s0, 64);   /* 32 halfwords from row phys_row */
        memcpy(&u.b[64], s1, 64);   /* 32 halfwords from row phys_row+16 */

        HVX_Vector v_hi_prep = Q6_V_vand_VV(u.v, v_mask_FF00);
        HVX_Vector v_hi_out  = Q6_Vh_vshuff_Vh(v_hi_prep);
        memcpy(tile_hi + 128 * phys_row, &v_hi_out, sizeof(HVX_Vector));

        HVX_Vector v_lo_prep = Q6_Vh_vasl_VhR(u.v, 8);
        HVX_Vector v_lo_out  = Q6_Vh_vshuff_Vh(v_lo_prep);
        memcpy(tile_lo + 128 * phys_row, &v_lo_out, sizeof(HVX_Vector));
    }
#else
    /* Scalar reference for CPU build / sim bisection. */
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        uint32_t *__restrict__ dst_hi = (uint32_t *)(tile_hi + 128 * phys_row);
        uint32_t *__restrict__ dst_lo = (uint32_t *)(tile_lo + 128 * phys_row);
        const uint16_t *s0 = &au_base[phys_row        * K_stride];
        const uint16_t *s1 = &au_base[(phys_row + 16) * K_stride];
        for (int K = 0; K < 32; K++) {
            uint16_t a0 = s0[K];
            uint16_t a1 = s1[K];
            uint8_t a0_hi = (uint8_t)(a0 >> 8);
            uint8_t a1_hi = (uint8_t)(a1 >> 8);
            uint8_t a0_lo = (uint8_t)(a0 & 0xFF);
            uint8_t a1_lo = (uint8_t)(a1 & 0xFF);
            dst_hi[K] = ((uint32_t)a1_hi << 24) | ((uint32_t)a0_hi << 8);
            dst_lo[K] = ((uint32_t)a1_lo << 24) | ((uint32_t)a0_lo << 8);
        }
    }
#endif
}

/* Public standalone entry — callable from the sim harness to verify. */
void pack_act_hvx_kernel_body(
    const uint16_t *au,   /* [M, K] row-major */
    uint8_t *out_hi,      /* [M/32, K/32, 2048] flat */
    uint8_t *out_lo,      /* [M/32, K/32, 2048] flat */
    int M, int K)
{
    const int M_tiles = M / 32;
    const int K_tiles = K / 32;
    for (int mt = 0; mt < M_tiles; mt++) {
        for (int kt = 0; kt < K_tiles; kt++) {
            const uint16_t *au_base = &au[mt * 32 * K + kt * 32];
            uint8_t *tile_hi = out_hi + (mt * K_tiles + kt) * 2048;
            uint8_t *tile_lo = out_lo + (mt * K_tiles + kt) * 2048;
            pack_one_tile(tile_hi, tile_lo, au_base, K);
        }
    }
}

/* -------------------------------------------------------------------
 * QHPI kernel entry.
 * ------------------------------------------------------------------- */
static uint32_t pack_act_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs,  const QHPI_Tensor *const *inputs)
{
    (void)handle; (void)num_outputs; (void)num_inputs;

    const uint16_t *au = (const uint16_t *)qhpi_tensor_raw_data(inputs[0]);
    uint8_t *out_hi = (uint8_t *)qhpi_tensor_raw_data(outputs[0]);
    uint8_t *out_lo = (uint8_t *)qhpi_tensor_raw_data(outputs[1]);

    QHPI_Shape as = qhpi_tensor_shape(inputs[0]);
    const int M = (int)as.dims[as.rank - 2];
    const int K = (int)as.dims[as.rank - 1];

    pack_act_hvx_kernel_body(au, out_hi, out_lo, M, K);
    return QHPI_Success;
}

/* -------------------------------------------------------------------
 * QHPI registration.
 * ------------------------------------------------------------------- */
static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};

static QHPI_Kernel_v1 sg_kernels[] = {
    {
        /* .function_name      */ THIS_PKG_NAME_STR "::pack_act_hvx",
        /* .function           */ pack_act_kernel,
        /* .resources          */ QHPI_RESOURCE_HVX,
        /* .source_destructive */ false,
        /* .multithreaded      */ true,
        /* .variable_inputs    */ false,
        /* .variable_outputs   */ false,
        /* .min_inputs         */ 1,
        /* .input_signature    */ sig_inputs,
        /* .min_outputs        */ 2,
        /* .output_signature   */ sig_outputs,
        /* .cost_function      */ NULL,
        /* .sync_block_size    */ 0,
        /* .precomputed_data_size */ 0,
        /* .do_precomputation_function */ NULL,
        /* .function_with_precomputed_data */ NULL,
        /* .predicate          */ NULL,
    },
};

static QHPI_OpInfo_v1 sg_ops[] = {
    {
        /* .name              */ THIS_PKG_NAME_STR "::PackActivationToHmxTile",
        /* .num_kernels       */ 1,
        /* .kernels           */ sg_kernels,
        /* .early_rewrite     */ NULL,
        /* .shape_required    */ NULL,
        /* .shape_legalized   */ NULL,
        /* .tile_output       */ 0,
        /* .build_tile        */ NULL,
        /* .late_rewrite      */ NULL,
    },
};

extern "C" void register_pack_act_op(void)
{
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
