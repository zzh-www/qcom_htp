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
    uint8_t       *tile,     /* 1 KiB destination, aligned 128 */
    const uint8_t *a_base,   /* &act[m_tile*32][k_tile*32] */
    uint32_t       K_full)
{
    /* Copy 32 rows × 32 bytes contiguous to tile.
     * Source rows are K_full-strided; dest is 32-contiguous.
     * HVX vmemu (128-B unaligned load) reads one row (32 B meaningful
     * + 96 B trailing into next row territory) per load — we mask. */
#if defined(__hexagon__)
    const HVX_VectorPred pred_32 = Q6_Q_vsetq_R(32);
    const HVX_Vector     v_zero  = Q6_V_vzero();
    for (int r = 0; r < 32; r++) {
        HVX_Vector v_raw;
        memcpy(&v_raw, &a_base[r * K_full], sizeof(HVX_Vector));
        /* Keep only lanes 0..31, zero rest. Tile destination is
         * packed 32 B per row so only first 32 B matter. Using
         * memcpy(32) avoids needing to store a masked vector. */
        (void)pred_32; (void)v_zero;
        memcpy(&tile[r * 32], &v_raw, 32);
    }
#else
    for (int r = 0; r < 32; r++)
        memcpy(&tile[r * 32], &a_base[r * K_full], 32);
#endif
}

void pack_act_rm_hvx_kernel_body(
    const uint8_t *a,     /* [M, K] row-major */
    uint8_t       *out,   /* [M/32, K/32, 1024] flat */
    int M, int K,
    uint32_t mt_start, uint32_t mt_end)
{
    const int K_tiles = K / 32;
    for (uint32_t mt = mt_start; mt < mt_end; mt++) {
        for (int kt = 0; kt < K_tiles; kt++) {
            const uint8_t *a_base = &a[mt * 32 * K + kt * 32];
            uint8_t *tile = out + (mt * K_tiles + kt) * 1024;
            pack_one_rm_tile(tile, a_base, (uint32_t)K);
        }
    }
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
        /* source_destructive */ false,
        /* multithreaded       */ true,
        /* variable_inputs     */ false,
        /* variable_outputs    */ false,
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
