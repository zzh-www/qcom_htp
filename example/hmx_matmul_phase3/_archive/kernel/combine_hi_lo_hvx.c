/*
 * combine_hi_lo_hvx.c — CombineHiLo op (Phase 3B, Path W).
 *
 * Final reconstruction stage for the int4×int16 (actually int4×uint16
 * post-Cast) matmul:
 *   out[m,n] = (P_hi[m,n] << 8) + P_lo[m,n] - (col_sum_w[n] << 15)
 *
 * All three inputs and the output are int32 [1,1,M,N]. col_sum_w is the
 * per-N-column sum of the int8 weight (broadcast across M). This body is
 * verbatim from the HVX combine loop at the end of
 * hmx_int4_matmul_mn_dualacc (example/hmx_matmul_qnn/kernel/hmx_int4_matmul.c
 * around lines 574-595).
 *
 * HVX lane math: 32 int32/row × 4-byte = 1 vector per (row, n_tile) where
 * n_tile is 32 columns. For N > 32 we simply iterate through consecutive
 * HVX vectors per row — each iteration uses a freshly-loaded col_sum slice
 * (the N dimension cycles through aligned 32-col groups).
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

/* Core body — operates on 32-col chunks. Works in-place on unaligned data
 * because memcpy + HVX_Vector locals let the compiler emit vldu/vstu when
 * needed; aligned fast-path kicks in automatically when inputs happen to
 * be vector-aligned (common for VTCM-resident outputs from MatMul). */
void combine_hi_lo_hvx_kernel_body(
    const int32_t *P_hi,     /* [M, N] */
    const int32_t *P_lo,     /* [M, N] */
    const int32_t *col_sum,  /* [N] */
    int32_t *out,            /* [M, N] */
    int M, int N)
{
#ifdef __hexagon__
    /* Pre-shift col_sum by 15 into a row of vectors (N/32 vectors). */
    const int N_vec = N / 32;
    /* Use a small stack buffer for the shifted col_sum — avoids mutating
     * caller data. Guarantees 128-byte alignment for aligned vstore. */
    for (int m = 0; m < M; m++) {
        for (int nv = 0; nv < N_vec; nv++) {
            HVX_Vector v_col_shifted, v_hi, v_lo, v_hi_shl, v_sum, v_out;
            memcpy(&v_col_shifted, &col_sum[nv * 32], sizeof(HVX_Vector));
            v_col_shifted = Q6_Vw_vasl_VwR(v_col_shifted, 15);
            memcpy(&v_hi, &P_hi[m * N + nv * 32], sizeof(HVX_Vector));
            memcpy(&v_lo, &P_lo[m * N + nv * 32], sizeof(HVX_Vector));
            v_hi_shl = Q6_Vw_vasl_VwR(v_hi, 8);
            v_sum    = Q6_Vw_vadd_VwVw(v_hi_shl, v_lo);
            v_out    = Q6_Vw_vsub_VwVw(v_sum, v_col_shifted);
            memcpy(&out[m * N + nv * 32], &v_out, sizeof(HVX_Vector));
        }
    }
#else
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            out[m * N + n] = (P_hi[m * N + n] << 8)
                           +  P_lo[m * N + n]
                           - (col_sum[n] << 15);
        }
    }
#endif
}

/* -------------------------------------------------------------------
 * QHPI kernel entry.
 * ------------------------------------------------------------------- */
static uint32_t combine_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs,  const QHPI_Tensor *const *inputs)
{
    (void)handle; (void)num_outputs; (void)num_inputs;

    const int32_t *P_hi    = (const int32_t *)qhpi_tensor_raw_data(inputs[0]);
    const int32_t *P_lo    = (const int32_t *)qhpi_tensor_raw_data(inputs[1]);
    const int32_t *col_sum = (const int32_t *)qhpi_tensor_raw_data(inputs[2]);
    int32_t       *out     = (int32_t *)qhpi_tensor_raw_data(outputs[0]);

    QHPI_Shape os = qhpi_tensor_shape(outputs[0]);
    const int M = (int)os.dims[os.rank - 2];
    const int N = (int)os.dims[os.rank - 1];

    combine_hi_lo_hvx_kernel_body(P_hi, P_lo, col_sum, out, M, N);
    return QHPI_Success;
}

/* -------------------------------------------------------------------
 * QHPI registration.
 * ------------------------------------------------------------------- */
static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_Int32, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
    {QHPI_Int32, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
    {QHPI_Int32, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_Int32, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};

static QHPI_Kernel_v1 sg_kernels[] = {
    {
        /* .function_name      */ THIS_PKG_NAME_STR "::combine_hi_lo_hvx",
        /* .function           */ combine_kernel,
        /* .resources          */ QHPI_RESOURCE_HVX,
        /* .source_destructive */ false,
        /* .multithreaded      */ true,
        /* .variable_inputs    */ false,
        /* .variable_outputs   */ false,
        /* .min_inputs         */ 3,
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
        /* .name              */ THIS_PKG_NAME_STR "::CombineHiLo",
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

extern "C" void register_combine_op(void)
{
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
