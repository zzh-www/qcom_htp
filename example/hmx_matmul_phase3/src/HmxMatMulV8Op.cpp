/*
 * HmxMatMulV8Op.cpp — Phase 3D.4: pure-HMX replica of QNN's q::ConvLayer_s1.opt.
 *
 * One-shot u8×i8→u8 matmul using HMX's :after:cm:sat.ub store.
 * Hardware formula (silicon-verified by probe_sat_ub.c):
 *   out[m][n] = saturate_u8( round(acc[m][n] × bias_fp16[n] / 512) + 128 )
 * HMX folds the quantization scale into the per-column fp16 bias:
 *   bias_fp16[n] = 512 × scale_quant[n]
 * so requant (SRDHM + shift + clip + +128 zero-offset) happens entirely in
 * hardware during the readback. No separate HVX requant op needed.
 *
 * Signatures:
 *   Input 0: packed_act  [1, M_tiles, K_tiles, 1024] u8   TCM_Only
 *            (row-major 32×32 tile for :cm activation)
 *   Input 1: packed_wt   [1, N_tiles, K_tiles, 1024] u8   TCM_Only
 *            (Phase 2 P2 4-K-row × 32-col pack)
 *   Input 2: bias_scale  [1, 1, N_tiles, 32] u16/fp16     TCM_Only
 *            (per-channel scale folded = 512 × scale_out[n])
 *   Output 0: out        [1, 1, M, N] u8                  DDR_OR_TCM
 *            (u8 directly, zero_offset=128)
 *
 * Kernel body = 4 HMX inline-asm instructions per output tile, no HVX.
 */

#include "HTP/core/qhpi.h"
#include <cstdint>
#include <cstring>

#ifdef __hexagon__
#include <hexagon_types.h>
#endif

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

#if defined(__hexagon__)
#define HMX_RT_ACT_CM  (2047 | 0x1C)   /* :cm activation Rt per Agent/cm_row_major_re.md */
#define HMX_RT_WT      0x3FF           /* plain weight Rt */

static inline __attribute__((always_inline))
void hmx_v8_mac_convert(
    const void *act_tile,              /* 1 KiB row-major */
    const void *wt_tile,               /* 1 KiB P2 packed */
    const void *bias_scale,            /* 128 fp16 values */
    void       *out_tile)              /* 1 KiB u8 destination */
{
    asm volatile("bias = mxmem(%0)" :: "r"(bias_scale) : "memory");
    asm volatile("mxclracc" ::: "memory");
    asm volatile(
        "{ activation.ub = mxmem(%0, %1):cm\n"
        "  weight.b      = mxmem(%2, %3) }"
        :: "r"(act_tile), "r"(HMX_RT_ACT_CM),
           "r"(wt_tile),  "r"(HMX_RT_WT)
        : "memory");
    asm volatile("mxmem(%0, %1):after:cm:sat.ub = acc"
                 :: "r"(out_tile), "r"(0) : "memory");
}

static inline __attribute__((always_inline))
void hmx_v8_mac_accumulate(
    const void *act_tile,
    const void *wt_tile)
{
    asm volatile(
        "{ activation.ub = mxmem(%0, %1):cm\n"
        "  weight.b      = mxmem(%2, %3) }"
        :: "r"(act_tile), "r"(HMX_RT_ACT_CM),
           "r"(wt_tile),  "r"(HMX_RT_WT)
        : "memory");
}
#endif

static inline uint32_t dim_at_v8(const QHPI_Shape &s, uint32_t i)
{
    return i < s.rank ? s.dims[i] : 1;
}

static uint32_t hmx_matmul_v8_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
    (void)handle; (void)num_outputs; (void)num_inputs;

#if !defined(__hexagon__) || defined(SCALAR_ONLY)
    uint8_t *out = (uint8_t *)qhpi_tensor_raw_data(outputs[0]);
    QHPI_Shape os = qhpi_tensor_shape(outputs[0]);
    uint32_t total = 1;
    for (uint32_t i = 0; i < os.rank; i++) total *= os.dims[i];
    memset(out, 128, total);
    return QHPI_Success;
#else
    const uint8_t  *packed_act = (const uint8_t  *)qhpi_tensor_raw_data(inputs[0]);
    const uint8_t  *packed_wt  = (const uint8_t  *)qhpi_tensor_raw_data(inputs[1]);
    const uint16_t *bias_all   = (const uint16_t *)qhpi_tensor_raw_data(inputs[2]);
    uint8_t        *vtcm_stg   = (uint8_t *)qhpi_tensor_raw_data(inputs[3]);
    uint8_t        *out        = (uint8_t *)qhpi_tensor_raw_data(outputs[0]);

    QHPI_Shape as = qhpi_tensor_shape(inputs[0]);
    QHPI_Shape ws = qhpi_tensor_shape(inputs[1]);
    QHPI_Shape os = qhpi_tensor_shape(outputs[0]);
    const uint32_t M_tiles = dim_at_v8(as, 1);
    const uint32_t K_tiles = dim_at_v8(as, 2);
    const uint32_t N_tiles = dim_at_v8(ws, 1);
    const uint32_t N       = dim_at_v8(os, os.rank - 1);

    /* HMX :sat.ub store writes 1 KiB contiguous (32×32) to out_ptr. To
     * place into row-major [M,N] output with stride N, we write each
     * tile to VTCM scratch (1 KiB aligned), then scatter 32 rows × 32 B
     * to DDR output at the right stride. Scatter uses scalar memcpy —
     * 1 KiB per tile, negligible vs HMX MAC. */
    for (uint32_t mt = 0; mt < M_tiles; mt++) {
        for (uint32_t nt = 0; nt < N_tiles; nt++) {
            const uint8_t  *act_tiles = packed_act + (mt * K_tiles) * 1024;
            const uint8_t  *wt_tiles  = packed_wt  + (nt * K_tiles) * 1024;
            const uint16_t *bias_n    = bias_all   + nt * 128;
            uint8_t        *out_tile  = out + (mt * 32) * N + nt * 32;

            asm volatile("bias = mxmem(%0)" :: "r"(bias_n) : "memory");
            asm volatile("mxclracc" ::: "memory");
            for (uint32_t kt = 0; kt < K_tiles; kt++) {
                hmx_v8_mac_accumulate(
                    act_tiles + kt * 1024,
                    wt_tiles  + kt * 1024);
            }
            /* HMX writes 1 KiB contiguous to vtcm_stg. */
            asm volatile("mxmem(%0, %1):after:cm:sat.ub = acc"
                         :: "r"(vtcm_stg), "r"(0) : "memory");

            /* Scatter 32 rows × 32 B to row-major output. */
            for (uint32_t r = 0; r < 32; r++) {
                memcpy(&out_tile[r * N], &vtcm_stg[r * 32], 32);
            }
        }
    }
    return QHPI_Success;
#endif
}

static QHPI_Tensor_Signature_v1 sig_inputs_v8[] = {
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},   /* packed_act (row-major) */
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},   /* packed_wt (P2) */
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},   /* bias fp16 */
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},   /* vtcm staging (1 KiB aligned) */
};
static QHPI_Tensor_Signature_v1 sig_outputs_v8[] = {
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM}, /* u8 out */
};

static QHPI_Kernel_v1 sg_kernels_v8[] = {
    {
        THIS_PKG_NAME_STR "::hmx_matmul_v8",
        hmx_matmul_v8_kernel,
        QHPI_RESOURCE_HMX,
        false, false, false, false,
        4, sig_inputs_v8,
        1, sig_outputs_v8,
        nullptr, 0, 0, nullptr, nullptr, nullptr,
    },
};

static QHPI_OpInfo_v1 sg_ops_v8[] = {
    {
        THIS_PKG_NAME_STR "::MatMulV8",
        1, sg_kernels_v8,
        nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    },
};

extern "C" void register_hmx_matmul_v8_op() {
    qhpi_register_ops_v1(sizeof(sg_ops_v8) / sizeof(sg_ops_v8[0]),
                         sg_ops_v8, THIS_PKG_NAME_STR);
}
