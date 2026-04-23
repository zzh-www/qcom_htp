/*
 * HmxMatMulV2Op.cpp — Path X MatMul op (int8 × int8, :cm + row-major).
 *
 * QNN-native minimal kernel. Declares Flat4 + Direct signature so we
 * get raw row-major tensors. Kernel gathers 32x32 act sub-tile + packs
 * weight tile (both scalar mem-moves, fast — <1% vs MAC), then runs
 * HMX MAC with :cm + Rt_a|0x1c + Rt_w=0x3FF.
 *
 * Distinct from existing `MatMulInt8xInt8Crouton` (phase3 probe op) —
 * this one is the functional Path X kernel.
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

static uint32_t hmx_matmul_v2_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
    (void)num_outputs; (void)num_inputs; (void)handle;

#if !defined(__hexagon__) || defined(SCALAR_ONLY)
    const uint8_t *au = (const uint8_t *)qhpi_tensor_raw_data(inputs[0]);
    const int8_t  *wu = (const int8_t  *)qhpi_tensor_raw_data(inputs[1]);
    int32_t       *out = (int32_t *)qhpi_tensor_raw_data(outputs[0]);
    QHPI_Shape as = qhpi_tensor_shape(inputs[0]);
    QHPI_Shape ws = qhpi_tensor_shape(inputs[1]);
    uint32_t M = dim_at(as, as.rank - 2);
    uint32_t K = dim_at(as, as.rank - 1);
    uint32_t N = dim_at(ws, ws.rank - 1);
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = 0; j < N; j++) {
            int32_t s = 0;
            for (uint32_t k = 0; k < K; k++)
                s += (int32_t)au[i * K + k] * (int32_t)wu[k * N + j];
            out[i * N + j] = s;
        }
    return QHPI_Success;
#else
    const uint8_t *au    = (const uint8_t *)qhpi_tensor_raw_data(inputs[0]);
    const int8_t  *wu    = (const int8_t  *)qhpi_tensor_raw_data(inputs[1]);
    void          *vtcm  =                  qhpi_tensor_raw_data(inputs[2]);
    int32_t       *out   = (int32_t *)qhpi_tensor_raw_data(outputs[0]);

    QHPI_Shape as = qhpi_tensor_shape(inputs[0]);
    QHPI_Shape ws = qhpi_tensor_shape(inputs[1]);
    const uint32_t M = dim_at(as, as.rank - 2);
    const uint32_t K = dim_at(as, as.rank - 1);
    const uint32_t N = dim_at(ws, ws.rank - 1);
    const uint32_t K_tiles = K / 32;
    const uint32_t M_tiles = (M + 31) / 32;
    const uint32_t N_tiles = (N + 31) / 32;

    /* VTCM layout:
     *   [0, 512)                      bias (filled once)
     *   [512, 512+K_tiles*1024)        act_tiles (row-major, per m-tile)
     *   [above, + K_tiles*1024)        wt_tiles (per n-tile, amortize across m)
     *   align to 2K boundary:
     *   [OUT_LO, +2K)                  out_lo
     *   [OUT_HI, +2K)                  out_hi */
    /* HMX 2-KiB aligned VTCM layout:
     *   [0, 2K)                   reserved (bias at start)
     *   [2K, 2K + K*1024)         act_tiles
     *   [rounded up to 2K, + K*1024)  wt_tiles
     *   [rounded, + 2K)           out_top_lo
     *   [+ 2K)                    out_top_hi
     *   [+ 2K)                    out_bot_lo
     *   [+ 2K)                    out_bot_hi   */
    uint8_t *vt = (uint8_t *)vtcm;
    void     *bias_vtcm = vt;
    hmx_core_v2_fill_bias(bias_vtcm);

    uint8_t *act_tiles_vtcm = vt + 2048;
    /* Act tile = 2 KiB each (Phase 2 2-stream pack). */
    uintptr_t after_act = (uintptr_t)(act_tiles_vtcm + K_tiles * 2048);
    after_act = (after_act + 2047u) & ~(uintptr_t)2047u;
    uint8_t *wt_tiles_vtcm  = (uint8_t *)after_act;
    /* Weight tile = 1 KiB each (Phase 2 packed). */
    uintptr_t after_wt = (uintptr_t)(wt_tiles_vtcm + K_tiles * 1024);
    after_wt = (after_wt + 2047u) & ~(uintptr_t)2047u;
    uint16_t *out_top_lo = (uint16_t *)(after_wt + 0 * 2048);
    uint16_t *out_top_hi = (uint16_t *)(after_wt + 1 * 2048);
    uint16_t *out_bot_lo = (uint16_t *)(after_wt + 2 * 2048);
    uint16_t *out_bot_hi = (uint16_t *)(after_wt + 3 * 2048);

    /* Staging: int32 tile buffer for readback decode. 4 KiB. */
    static int32_t mn_tile[32 * 32];

    for (uint32_t mt = 0; mt < M_tiles; mt++) {
        /* Gather all K_tiles of activation for this m-tile (2 KiB each). */
        for (uint32_t kt = 0; kt < K_tiles; kt++) {
            hmx_core_v2_gather_act_tile(act_tiles_vtcm + kt * 2048,
                                         au, K, mt * 32, kt * 32);
        }

        for (uint32_t nt = 0; nt < N_tiles; nt++) {
            /* Gather weight K×N column tiles for this n-tile.
             * Re-gathered per (mt, nt) pair — easy to hoist out of m-loop
             * with a cache like Phase 2 T0 if this shows up in profile. */
            for (uint32_t kt = 0; kt < K_tiles; kt++) {
                hmx_core_v2_gather_wt_tile(wt_tiles_vtcm + kt * 1024,
                                            wu, N, kt * 32, nt * 32);
            }

            /* HMX 2-pass :cm MAC + dual-scale readback. */
            hmx_matmul_v2_core_mn(act_tiles_vtcm, wt_tiles_vtcm,
                                   K_tiles, bias_vtcm,
                                   out_top_lo, out_top_hi,
                                   out_bot_lo, out_bot_hi);

            /* Phase 2 2-stream dual-scale decode: 32 logical rows in one
             * readback buffer pair. */
            (void)out_bot_lo; (void)out_bot_hi;
            for (int ir = 0; ir < 32; ir++) {
                int phys_row = ir & 15, stream = ir >> 4;
                for (int jc = 0; jc < 32; jc++) {
                    int idx = phys_row * 64 + 2 * jc + stream;
                    uint16_t lo = out_top_lo[idx], hi = out_top_hi[idx];
                    mn_tile[ir * 32 + jc] =
                        ((int32_t)(int16_t)hi << 8) | ((int32_t)lo & 0xFF);
                }
            }

            /* Scatter to output (M, N) */
            for (uint32_t i = 0; i < 32; i++) {
                int32_t *d = &out[(mt * 32 + i) * N + nt * 32];
                for (uint32_t j = 0; j < 32; j++) d[j] = mn_tile[i * 32 + j];
            }
        }
    }
    return QHPI_Success;
#endif
}

/* Signature: Flat4 + Direct, raw tensor access.
 * Input 2 (scratch) is VTCM for gather buffers + bias + readback.
 * Per-(mt,nt) VTCM: 512 + 2 * K_tiles * 1024 + 4096 ≈ 4 KiB + K*64 B.
 * At K=512: 512 + 32 KiB + 4 KiB ≈ 37 KiB. At K=4096: 256 KiB. */
static QHPI_Tensor_Signature_v1 sig_inputs_v2[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},  /* wt: QNN Cast makes uint8 */
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};
static QHPI_Tensor_Signature_v1 sig_outputs_v2[] = {
    {QHPI_Int32,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};

static QHPI_Kernel_v1 sg_kernels_v2[] = {
    {
        /* .function_name      */ THIS_PKG_NAME_STR "::hmx_matmul_v2",
        /* .function           */ hmx_matmul_v2_kernel,
        /* .resources          */ QHPI_RESOURCE_HMX,
        /* .source_destructive */ false,
        /* .multithreaded      */ false,
        /* .variable_inputs    */ false,
        /* .variable_outputs   */ false,
        /* .min_inputs         */ 3,
        /* .input_signature    */ sig_inputs_v2,
        /* .min_outputs        */ 1,
        /* .output_signature   */ sig_outputs_v2,
        /* .cost_function      */ nullptr,
        /* .sync_block_size    */ 0,
        /* .precomputed_data_size */ 0,
        /* .do_precomputation_function */ nullptr,
        /* .function_with_precomputed_data */ nullptr,
        /* .predicate          */ nullptr,
    },
};

static QHPI_OpInfo_v1 sg_ops_v2[] = {
    {
        /* .name */ THIS_PKG_NAME_STR "::MatMulV2",
        /* .num_kernels */ 1,
        /* .kernels */ sg_kernels_v2,
        /* .early_rewrite */ nullptr,
        /* .shape_required */ nullptr,
        /* .shape_legalized */ nullptr,
        /* .tile_output */ 0,
        /* .build_tile */ nullptr,
        /* .late_rewrite */ nullptr,
    },
};

extern "C" void register_hmx_matmul_v2_op() {
    qhpi_register_ops_v1(sizeof(sg_ops_v2) / sizeof(sg_ops_v2[0]),
                         sg_ops_v2, THIS_PKG_NAME_STR);
}
