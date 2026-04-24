/*
 * pack_act_u8_hvx.c — PackActivationU8ToHmxTile op (Phase 3C, Path W).
 *
 * Variant of pack_act_hvx for u8 activations (V3 path). Produces ONE
 * 2 KB activation tile per (m_tile, k_tile) in Phase 2 2-stream format:
 *
 *   tile[128*phys_row + 4*K + 0] = 0
 *   tile[128*phys_row + 4*K + 1] = act[m_tile*32 + phys_row     ][k_tile*32 + K]
 *   tile[128*phys_row + 4*K + 2] = 0
 *   tile[128*phys_row + 4*K + 3] = act[m_tile*32 + phys_row + 16][k_tile*32 + K]
 *
 * Packed for `activation.ub = mxmem(ptr, 2047)` plain-mxmem consumer,
 * matching the layout hmx_core_v2_gather_act_tile produces.
 *
 * Tensor contract:
 *   Input  0: uint8 act      [1, 1, M, K]           Flat4 + Direct
 *   Output 0: uint8 tile     [1, M/32, K/32, 2048]  Flat4 + Direct
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

/* Pack one 32×32 u8 tile into 2 KB 2-stream format. */
static inline void pack_one_tile_u8(
    uint8_t *__restrict__ tile,           /* 2048 B */
    const uint8_t *__restrict__ a_base,   /* pointer to (m_tile*32, k_tile*32) start */
    int K_stride)                          /* full K dimension (row stride in bytes) */
{
#if defined(__hexagon__) && !defined(PACK_ACT_U8_SCALAR_FORCE)
    /* HVX pack (optimized): load each activation row once via unaligned
     * vmem (128 bytes = one cache line), mask to first 32 bytes, rotate
     * into target position, OR, then 2× Q6_Vb_vshuff_Vb. Replaces the
     * earlier memset + 2× memcpy(32B) scalar pattern which was forcing
     * the compiler through stack memory.
     *
     * Target input layout for the 2× vshuff:
     *   bytes  0..31  = 0
     *   bytes 32..63  = s0[0..31]
     *   bytes 64..95  = 0
     *   bytes 96..127 = s1[0..31]
     *
     * 2× Q6_Vb_vshuff_Vb produces [0, s0[0], 0, s1[0], 0, s0[1], 0, s1[1], ...] */
    const HVX_Vector     v_zero   = Q6_V_vzero();
    const HVX_VectorPred pred_32  = Q6_Q_vsetq_R(32);  /* bytes 0..31 = active */
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        const uint8_t *s0 = &a_base[phys_row        * K_stride];
        const uint8_t *s1 = &a_base[(phys_row + 16) * K_stride];

        /* Unaligned 128-B load (vmemu) — reads one full cache line.
         * Only first 32 bytes are semantically needed; rest is masked. */
        HVX_Vector v_s0_raw, v_s1_raw;
        memcpy(&v_s0_raw, s0, sizeof(HVX_Vector));
        memcpy(&v_s1_raw, s1, sizeof(HVX_Vector));

        /* Keep only lanes 0..31 of each. */
        HVX_Vector v_s0 = Q6_V_vmux_QVV(pred_32, v_s0_raw, v_zero);
        HVX_Vector v_s1 = Q6_V_vmux_QVV(pred_32, v_s1_raw, v_zero);

        /* Rotate right: s0 lanes 0..31 → 32..63, s1 lanes 0..31 → 96..127.
         * Masked-out lanes stay zero. */
        HVX_Vector v_s0_pos = Q6_V_vror_VR(v_s0, 32);
        HVX_Vector v_s1_pos = Q6_V_vror_VR(v_s1, 96);

        /* Combine (zero + value + zero + value). */
        HVX_Vector v_input  = Q6_V_vor_VV(v_s0_pos, v_s1_pos);

        HVX_Vector step1  = Q6_Vb_vshuff_Vb(v_input);
        HVX_Vector final_ = Q6_Vb_vshuff_Vb(step1);

        /* Aligned store (tile is 128-B aligned in VTCM). */
        *((HVX_Vector *)(tile + 128 * phys_row)) = final_;
    }
#else
    memset(tile, 0, 2048);
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        uint32_t *__restrict__ dst = (uint32_t *)(tile + 128 * phys_row);
        const uint8_t *s0 = &a_base[phys_row        * K_stride];
        const uint8_t *s1 = &a_base[(phys_row + 16) * K_stride];
        for (int k = 0; k < 32; k++)
            dst[k] = ((uint32_t)s1[k] << 24) | ((uint32_t)s0[k] << 8);
    }
#endif
}

void pack_act_u8_hvx_kernel_body(
    const uint8_t *a,   /* [M, K] row-major */
    uint8_t *out,       /* [M/32, K/32, 2048] flat */
    int M, int K)
{
    const int M_tiles = M / 32;
    const int K_tiles = K / 32;
    for (int mt = 0; mt < M_tiles; mt++) {
        for (int kt = 0; kt < K_tiles; kt++) {
            const uint8_t *a_base = &a[mt * 32 * K + kt * 32];
            uint8_t *tile = out + (mt * K_tiles + kt) * 2048;
            pack_one_tile_u8(tile, a_base, K);
        }
    }
}

static uint32_t pack_act_u8_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs,  const QHPI_Tensor *const *inputs)
{
    (void)handle; (void)num_outputs; (void)num_inputs;

    const uint8_t *a   = (const uint8_t *)qhpi_tensor_raw_data(inputs[0]);
    uint8_t       *out = (uint8_t *)qhpi_tensor_raw_data(outputs[0]);

    QHPI_Shape as = qhpi_tensor_shape(inputs[0]);
    const int M = (int)as.dims[as.rank - 2];
    const int K = (int)as.dims[as.rank - 1];

    pack_act_u8_hvx_kernel_body(a, out, M, K);
    return QHPI_Success;
}

static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};
/* Output must live in VTCM so the downstream MatMulV3 (TCM_Only) can
 * consume it via HMX mxmem. */
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};

static QHPI_Kernel_v1 sg_kernels[] = {
    {
        /* .function_name      */ THIS_PKG_NAME_STR "::pack_act_u8_hvx",
        /* .function           */ pack_act_u8_kernel,
        /* .resources          */ QHPI_RESOURCE_HVX,
        /* .source_destructive */ false,
        /* .multithreaded      */ false,
        /* .variable_inputs    */ false,
        /* .variable_outputs   */ false,
        /* .min_inputs         */ 1,
        /* .input_signature    */ sig_inputs,
        /* .min_outputs        */ 1,
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
        /* .name              */ THIS_PKG_NAME_STR "::PackActivationU8ToHmxTile",
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

extern "C" void register_pack_act_u8_op(void)
{
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
