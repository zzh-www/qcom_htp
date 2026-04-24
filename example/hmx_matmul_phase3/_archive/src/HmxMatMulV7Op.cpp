/*
 * HmxMatMulV7Op.cpp — minimal HMX-only matmul kernel (Phase 3D.3).
 *
 * Principle: this op does ONLY HMX MAC work (bias + clracc + K pairs of
 * mxmem + dual-scale readback). NO HVX, NO gather, NO requant, NO pack.
 * Everything else lives in separate ops so QNN's scheduler can run them
 * on 4 HVX threads in parallel with HMX:
 *
 *   act_raw ──► PackActivationU8ToHmxTile [HVX, MT=4] ──► packed_act ┐
 *   wt_raw  ──► PackWeightToHmxTileV3     [HVX, MT=4] ──► packed_wt  │
 *                                                                    ▼
 *                                           MatMulV7 [HMX, MT=false]
 *                                                    │  │
 *                                            rb_lo   │  │ rb_hi
 *                                                    ▼  ▼
 *                                    RequantHvx [HVX, MT=4]
 *                                             │
 *                                             ▼
 *                                            out (i8)
 *
 * Signatures:
 *   Input 0: packed_act  [1, M_tiles, K_tiles, 2048] u8   TCM_Only (2-stream tile)
 *   Input 1: packed_wt   [1, N_tiles, K_tiles, 1024] u8   TCM_Only (P2 packed)
 *   Input 2: scratch     VTCM bias (512 B used) + per-nt alternation
 *   Output 0: rb_lo      [1, M_tiles, N_tiles, 2048] u16  TCM_Only (dual-scale lo)
 *   Output 1: rb_hi      [1, M_tiles, N_tiles, 2048] u16  TCM_Only (dual-scale hi)
 */

#include "HTP/core/qhpi.h"
#include <cstdint>
#include <cstring>

#ifdef __hexagon__
#include <hexagon_types.h>
#endif

extern "C" {
#include "../kernel/hmx_core_v2.h"
}

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

static inline uint32_t dim_at_v7(const QHPI_Shape &s, uint32_t i)
{
    return i < s.rank ? s.dims[i] : 1;
}

static uint32_t hmx_matmul_v7_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
    (void)num_outputs; (void)num_inputs; (void)handle;

#if !defined(__hexagon__) || defined(SCALAR_ONLY)
    /* CPU fallback: zero the readback tensors. */
    for (uint32_t o = 0; o < num_outputs; o++) {
        uint16_t *p = (uint16_t *)qhpi_tensor_raw_data(outputs[o]);
        QHPI_Shape s = qhpi_tensor_shape(outputs[o]);
        uint32_t total = 1;
        for (uint32_t i = 0; i < s.rank; i++) total *= s.dims[i];
        memset(p, 0, total * sizeof(uint16_t));
    }
    return QHPI_Success;
#else
    const uint8_t *packed_act = (const uint8_t *)qhpi_tensor_raw_data(inputs[0]);
    const uint8_t *packed_wt  = (const uint8_t *)qhpi_tensor_raw_data(inputs[1]);
    void          *vtcm       = qhpi_tensor_raw_data(inputs[2]);

    uint16_t *rb_lo = (uint16_t *)qhpi_tensor_raw_data(outputs[0]);
    uint16_t *rb_hi = (uint16_t *)qhpi_tensor_raw_data(outputs[1]);

    QHPI_Shape as = qhpi_tensor_shape(inputs[0]);
    QHPI_Shape ws = qhpi_tensor_shape(inputs[1]);
    const uint32_t M_tiles = dim_at_v7(as, 1);
    const uint32_t K_tiles = dim_at_v7(as, 2);
    const uint32_t N_tiles = dim_at_v7(ws, 1);

    /* Scratch layout: only bias needed (fill once). */
    uint8_t *vt = (uint8_t *)vtcm;
    void    *bias_vtcm = vt;
    hmx_core_v2_fill_bias(bias_vtcm);

    /* Per-tile dual-scale readback offsets in rb_lo / rb_hi tensors.
     * Each (mt, nt) tile's readback is 2 KiB (dual-stream 16 rows × 128 B).
     * Tile offset = (mt * N_tiles + nt) * 2048 bytes = * 1024 u16. */
    for (uint32_t mt = 0; mt < M_tiles; mt++) {
        for (uint32_t nt = 0; nt < N_tiles; nt++) {
            const uint8_t *act_tiles = packed_act + (mt * K_tiles) * 2048;
            const uint8_t *wt_tiles  = packed_wt  + (nt * K_tiles) * 1024;

            uint16_t *out_top_lo = rb_lo + (mt * N_tiles + nt) * 1024;
            uint16_t *out_top_hi = rb_hi + (mt * N_tiles + nt) * 1024;

            /* core_mn: bias + clracc + K_tiles × plain mxmem pair +
             * dual-scale readback (lo bias 0x4000, hi bias 0x2000).
             * Pure HMX inline asm; no HVX instructions used. */
            hmx_matmul_v2_core_mn(act_tiles, wt_tiles, K_tiles, bias_vtcm,
                                   out_top_lo, out_top_hi,
                                   /* out_bot_{lo,hi} unused */ NULL, NULL);
        }
    }
    return QHPI_Success;
#endif
}

/* Signatures — readbacks are u16 but QHPI element type closest fit is
 * QUInt16. QNN HTP places them in TCM so RequantHvx (next op) reads fast. */
static QHPI_Tensor_Signature_v1 sig_inputs_v7[] = {
    /* packed_act declared DDR_OR_TCM: allows QNN to place in DDR when
     * VTCM is tight (1024³ has 7 MB of TCM-only tensors vs 8 MB budget).
     * HMX mxmem for activation historically required VTCM, but QNN's
     * placement may DMA from DDR automatically. If it doesn't work
     * downstream we'll see graph finalize errors, not silent corruption. */
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM}, /* packed_act */
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},   /* packed_wt (HMX weight needs VTCM) */
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},   /* scratch (bias) */
};
static QHPI_Tensor_Signature_v1 sig_outputs_v7[] = {
    /* HMX mxmem store requires VTCM destination. At M=N=K=1024 the
     * packed act (2 MB) + packed wt (1 MB) + rb_lo (2 MB) + rb_hi (2 MB)
     * all-TCM requirement = 7 MB, crowding VTCM's 8 MB. Works to ≤512³;
     * 1024³ runs but outputs are corrupt (allocator overflow silent
     * fail). Future work: chunk along M. */
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},   /* rb_lo */
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},   /* rb_hi */
};

static QHPI_Kernel_v1 sg_kernels_v7[] = {
    {
        THIS_PKG_NAME_STR "::hmx_matmul_v7",
        hmx_matmul_v7_kernel,
        QHPI_RESOURCE_HMX,
        false, false, false, false,
        3, sig_inputs_v7,
        2, sig_outputs_v7,
        nullptr, 0, 0, nullptr, nullptr, nullptr,
    },
};

static QHPI_OpInfo_v1 sg_ops_v7[] = {
    {
        THIS_PKG_NAME_STR "::MatMulV7",
        1, sg_kernels_v7,
        nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    },
};

extern "C" void register_hmx_matmul_v7_op() {
    qhpi_register_ops_v1(sizeof(sg_ops_v7) / sizeof(sg_ops_v7[0]),
                         sg_ops_v7, THIS_PKG_NAME_STR);
}
