/*
 * HmxMatMulV3Op.cpp — PURE HMX matmul. Inputs must be pre-packed HMX
 * tiles (host-side pre-pack for now; future: wire Agent B's upstream
 * HVX ops). Kernel body is mxclracc + mxmem + mxmac + readback only.
 *
 * Signatures:
 *   Input 0: packed_act — [1, M_tiles, K_tiles, 2048] uint8
 *            Phase-2 2-stream format per 32×32 act tile
 *   Input 1: packed_wt  — [1, N_tiles, K_tiles, 1024] uint8
 *            Phase-2 4-K-row × 32-col packed per 32×32 wt tile
 *   Input 2: scratch    — VTCM for bias + readback buffers
 *   Output:  int32 [1, 1, M, N]
 *
 * NOT-HMX work here intentionally left outside this op:
 *   - Gather act[m,k] into 2-stream tile → PackAct HVX op (Agent B) or host
 *   - Gather wt[k,n] into 4-row tile → PackWt HVX op (Agent B) or host
 *   - The tiny dual-scale readback decode is ~1024 int32 ops per tile;
 *     kept here as inline scalar arithmetic (no HVX). Moving it out to a
 *     downstream Combine op is a cleanup option but not a perf win.
 */

#include "HTP/core/qhpi.h"
#include <cstdint>
#include <cstring>

extern "C" {
#include "../kernel/hmx_core_v2.h"
}

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

static inline uint32_t dim_at(const QHPI_Shape &s, uint32_t i)
{
    return i < s.rank ? s.dims[i] : 1;
}

static uint32_t hmx_matmul_v3_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
    (void)num_outputs; (void)num_inputs; (void)handle;

#if !defined(__hexagon__) || defined(SCALAR_ONLY)
    /* Fallback not meaningful for pre-packed inputs; just zero output. */
    int32_t *out = (int32_t *)qhpi_tensor_raw_data(outputs[0]);
    QHPI_Shape os = qhpi_tensor_shape(outputs[0]);
    uint32_t total = 1;
    for (uint32_t i = 0; i < os.rank; i++) total *= os.dims[i];
    memset(out, 0, total * sizeof(int32_t));
    return QHPI_Success;
#else
    const uint8_t *packed_act = (const uint8_t *)qhpi_tensor_raw_data(inputs[0]);
    const uint8_t *packed_wt  = (const uint8_t *)qhpi_tensor_raw_data(inputs[1]);
    void          *vtcm       = qhpi_tensor_raw_data(inputs[2]);
    int32_t       *out        = (int32_t *)qhpi_tensor_raw_data(outputs[0]);

    QHPI_Shape as = qhpi_tensor_shape(inputs[0]);
    QHPI_Shape ws = qhpi_tensor_shape(inputs[1]);
    QHPI_Shape os = qhpi_tensor_shape(outputs[0]);
    /* packed_act shape: [1, M_tiles, K_tiles, 2048] */
    const uint32_t M_tiles = dim_at(as, 1);
    const uint32_t K_tiles = dim_at(as, 2);
    /* packed_wt shape: [1, N_tiles, K_tiles, 1024] */
    const uint32_t N_tiles = dim_at(ws, 1);
    const uint32_t M = dim_at(os, os.rank - 2);
    const uint32_t N = dim_at(os, os.rank - 1);

    /* VTCM layout: bias (512 B) + 4 × 2 KiB readback buffers. */
    uint8_t *vt = (uint8_t *)vtcm;
    void    *bias_vtcm = vt;
    hmx_core_v2_fill_bias(bias_vtcm);
    uint16_t *out_top_lo = (uint16_t *)(vt + 2048);
    uint16_t *out_top_hi = (uint16_t *)(vt + 4096);
    uint16_t *out_bot_lo = (uint16_t *)(vt + 6144);   /* unused (plain-mxmem path) */
    uint16_t *out_bot_hi = (uint16_t *)(vt + 8192);

    /* Core loop. NO gather, NO pack, NO scalar data movement.
     * Inputs are already HMX-tile-format bytes from upstream pack.
     * Dual-scale decode + scatter fused into one HVX pass per tile — no
     * scalar intermediate buffer. */
    for (uint32_t mt = 0; mt < M_tiles; mt++) {
        for (uint32_t nt = 0; nt < N_tiles; nt++) {
            const uint8_t *act_tiles = packed_act + (mt * K_tiles) * 2048;
            const uint8_t *wt_tiles  = packed_wt  + (nt * K_tiles) * 1024;

            hmx_matmul_v2_core_mn(act_tiles, wt_tiles, K_tiles, bias_vtcm,
                                   out_top_lo, out_top_hi,
                                   out_bot_lo, out_bot_hi);

            int32_t *out_tile_base = &out[(mt * 32) * N + nt * 32];
            hmx_matmul_v2_decode_scatter_hvx(out_tile_base, N,
                                              out_top_lo, out_top_hi);
        }
    }
    return QHPI_Success;
#endif
}

/* Inputs are pre-packed bytes — signature as opaque uint8 tensors at the
 * expected packed shape. TCM_Only so QNN places them in VTCM where HMX
 * mxmem can reach. */
static QHPI_Tensor_Signature_v1 sig_inputs_v3[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};
static QHPI_Tensor_Signature_v1 sig_outputs_v3[] = {
    {QHPI_Int32,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};

static QHPI_Kernel_v1 sg_kernels_v3[] = {
    {
        THIS_PKG_NAME_STR "::hmx_matmul_v3",
        hmx_matmul_v3_kernel,
        QHPI_RESOURCE_HMX,
        false, false, false, false,
        3, sig_inputs_v3,
        1, sig_outputs_v3,
        nullptr, 0, 0, nullptr, nullptr, nullptr,
    },
};

static QHPI_OpInfo_v1 sg_ops_v3[] = {
    {
        THIS_PKG_NAME_STR "::MatMulV3",
        1, sg_kernels_v3,
        nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    },
};

extern "C" void register_hmx_matmul_v3_op() {
    qhpi_register_ops_v1(sizeof(sg_ops_v3) / sizeof(sg_ops_v3[0]),
                         sg_ops_v3, THIS_PKG_NAME_STR);
}
