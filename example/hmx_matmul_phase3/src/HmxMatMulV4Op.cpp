/*
 * HmxMatMulV4Op.cpp — PURE HMX matmul using `:cm` + row-major activation.
 *
 * **BROKEN — DO NOT USE FOR PRODUCTION**
 *
 * The probe in Agent/cm_readback_re.md (2026-04-23) established that
 * `activation.ub = mxmem(p, ...):cm` only writes 16 of the 32 output
 * rows — specifically the rows whose M index is ODD. Even-M rows are
 * not present in any readback we found. To compute all 32 rows under
 * `:cm` would require two `:cm` MACs per K-step (one with the tile
 * shifted by one row), doubling HMX work and erasing any per-MAC win.
 * V3's 2-stream plain path (0.143 cyc/MAC @ 512³) remains the working
 * baseline.
 *
 * V4 is kept as a scaffold: if a future probe finds a `:cm` MAC variant
 * that produces all 32 rows (e.g., via different `Rt_a` encoding,
 * different `:after.uh` sub-mode, or a `:cm + :above` pair), the
 * structure here can be revived. As written, V4 produces zero for
 * even-M rows and `lo - hi` (per the new dual-scale formula derived
 * from the probe) for odd-M rows.
 *
 * Same structure as V3 but:
 *   - Activation tile is 1 KiB row-major (32 rows × 32 K-bytes contiguous)
 *   - MAC issues `activation.ub = mxmem(p, 2047|0x1c):cm`
 *   - Weight tile stays 1 KiB Phase-2 4-K-row packed (unchanged)
 *   - Decode uses `lo - hi` (NOT V3's `(hi<<8)|(lo&0xFF)`)
 *
 * Signatures:
 *   Input 0: packed_act_rm — [1, M_tiles, K_tiles, 1024] uint8
 *   Input 1: packed_wt     — [1, N_tiles, K_tiles, 1024] uint8
 *   Input 2: scratch       — VTCM bias + readback buffers
 *   Output:  int32 [1, 1, M, N]  (even-M rows will be zero)
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

static inline uint32_t dim_at_v4(const QHPI_Shape &s, uint32_t i)
{
    return i < s.rank ? s.dims[i] : 1;
}

static uint32_t hmx_matmul_v4_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
    (void)num_outputs; (void)num_inputs; (void)handle;

#if !defined(__hexagon__) || defined(SCALAR_ONLY)
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
    /* packed_act shape: [1, M_tiles, K_tiles, 1024] (row-major, half of V3). */
    const uint32_t M_tiles = dim_at_v4(as, 1);
    const uint32_t K_tiles = dim_at_v4(as, 2);
    const uint32_t N_tiles = dim_at_v4(ws, 1);
    const uint32_t N = dim_at_v4(os, os.rank - 1);

    uint8_t *vt = (uint8_t *)vtcm;
    void    *bias_vtcm = vt;
    hmx_core_v2_fill_bias(bias_vtcm);
    uint16_t *out_top_lo = (uint16_t *)(vt + 2048);
    uint16_t *out_top_hi = (uint16_t *)(vt + 4096);
    uint16_t *out_bot_lo = (uint16_t *)(vt + 6144);   /* unused in 2x1 path */
    uint16_t *out_bot_hi = (uint16_t *)(vt + 8192);
    static int32_t mn_tile[32 * 32];

    for (uint32_t mt = 0; mt < M_tiles; mt++) {
        for (uint32_t nt = 0; nt < N_tiles; nt++) {
            /* act tile stride = 1 KiB (row-major); wt tile = 1 KiB (P2). */
            const uint8_t *act_tiles = packed_act + (mt * K_tiles) * 1024;
            const uint8_t *wt_tiles  = packed_wt  + (nt * K_tiles) * 1024;

            hmx_matmul_v2_core_mn_cm(act_tiles, wt_tiles, K_tiles, bias_vtcm,
                                      out_top_lo, out_top_hi,
                                      out_bot_lo, out_bot_hi);

            for (int ir = 0; ir < 32; ir++) {
                int phys_row = ir & 15, stream = ir >> 4;
                for (int jc = 0; jc < 32; jc++) {
                    int idx = phys_row * 64 + 2 * jc + stream;
                    uint16_t lo = out_top_lo[idx], hi = out_top_hi[idx];
                    mn_tile[ir * 32 + jc] =
                        ((int32_t)(int16_t)hi << 8) | ((int32_t)lo & 0xFF);
                }
            }
            for (uint32_t i = 0; i < 32; i++) {
                int32_t *d = &out[(mt * 32 + i) * N + nt * 32];
                const int32_t *s = &mn_tile[i * 32];
                for (uint32_t j = 0; j < 32; j++) d[j] = s[j];
            }
        }
    }
    return QHPI_Success;
#endif
}

static QHPI_Tensor_Signature_v1 sig_inputs_v4[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};
static QHPI_Tensor_Signature_v1 sig_outputs_v4[] = {
    {QHPI_Int32,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};

static QHPI_Kernel_v1 sg_kernels_v4[] = {
    {
        THIS_PKG_NAME_STR "::hmx_matmul_v4",
        hmx_matmul_v4_kernel,
        QHPI_RESOURCE_HMX,
        false, false, false, false,
        3, sig_inputs_v4,
        1, sig_outputs_v4,
        nullptr, 0, 0, nullptr, nullptr, nullptr,
    },
};

static QHPI_OpInfo_v1 sg_ops_v4[] = {
    {
        THIS_PKG_NAME_STR "::MatMulV4",
        1, sg_kernels_v4,
        nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    },
};

extern "C" void register_hmx_matmul_v4_op() {
    qhpi_register_ops_v1(sizeof(sg_ops_v4) / sizeof(sg_ops_v4[0]),
                         sg_ops_v4, THIS_PKG_NAME_STR);
}
