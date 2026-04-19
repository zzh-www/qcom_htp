/*
 * HmxInt4MatMulOp.cpp — QHPI custom op: MatMulInt4xInt16.
 *
 * Inputs  (all Flat4, 4-D shape [1,1,M,K] / [1,1,K,N] / [1,1,1, vtcm_bytes]):
 *   0: activation  int16        (QHPI_QInt16 or Int raw)  M·K elements
 *   1: weight      int8         (sign-extended int4)      K·N elements
 *   2: scratch     uint8        TCM_Only                  >= HMX_INT4_VTCM_BYTES
 * Output (Flat4, [1,1,M,N]):
 *   0: result      int32                                  M·N elements
 *
 * Tiles M,K,N in strides of 32. Over the K loop, an int32 accumulator is
 * maintained in the output buffer (zero-initialized on the first K tile).
 *
 * Hexagon build path: dispatches to hmx_int4_matmul_tile() per 32x32x32 tile.
 * CPU build path: naive scalar matmul for QNN prepare-time validation.
 */

#include "HTP/core/qhpi.h"
#include <cstdint>
#include <cstring>

extern "C" {
#include "../kernel/hmx_int4_matmul.h"
}

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

static inline uint32_t dim_at(const QHPI_Shape &s, uint32_t i)
{
    return i < s.rank ? s.dims[i] : 1;
}

/* ------------------------------------------------------------------
 * Tile copy helpers — gather a 32x32 sub-block from a row-major matrix.
 * ------------------------------------------------------------------ */
/* Un-shift Cast offsets while gathering — the data in memory is
 * (signed_value + zero_offset) per the Cast; recover signed values for the
 * HMX kernel which expects the ordinary int16/int8 interpretation. */
static void gather_a_tile(int16_t tile[32 * 32], const uint16_t *au,
                          uint32_t M, uint32_t K, uint32_t m0, uint32_t k0)
{
    const uint32_t rows = (m0 + 32 <= M) ? 32 : (M - m0);
    const uint32_t cols = (k0 + 32 <= K) ? 32 : (K - k0);
    memset(tile, 0, sizeof(int16_t) * 32 * 32);
    for (uint32_t i = 0; i < rows; i++)
        for (uint32_t j = 0; j < cols; j++)
            tile[i * 32 + j] = (int16_t)((int32_t)au[(m0 + i) * K + (k0 + j)] - 32768);
}

static void gather_w_tile(int8_t tile[32 * 32], const uint8_t *wu,
                          uint32_t K, uint32_t N, uint32_t k0, uint32_t n0)
{
    const uint32_t rows = (k0 + 32 <= K) ? 32 : (K - k0);
    const uint32_t cols = (n0 + 32 <= N) ? 32 : (N - n0);
    memset(tile, 0, sizeof(int8_t) * 32 * 32);
    for (uint32_t i = 0; i < rows; i++)
        for (uint32_t j = 0; j < cols; j++)
            tile[i * 32 + j] = (int8_t)((int32_t)wu[(k0 + i) * N + (n0 + j)] - 128);
}

static void scatter_accum_tile(int32_t *out, const int32_t tile[32 * 32],
                               uint32_t M, uint32_t N, uint32_t m0, uint32_t n0,
                               bool first_k)
{
    const uint32_t rows = (m0 + 32 <= M) ? 32 : (M - m0);
    const uint32_t cols = (n0 + 32 <= N) ? 32 : (N - n0);
    for (uint32_t i = 0; i < rows; i++) {
        int32_t *dst = &out[(m0 + i) * N + n0];
        const int32_t *src = &tile[i * 32];
        if (first_k) {
            for (uint32_t j = 0; j < cols; j++) dst[j] = src[j];
        } else {
            for (uint32_t j = 0; j < cols; j++) dst[j] += src[j];
        }
    }
}

/* ------------------------------------------------------------------
 * Kernel entry — called by QHPI runtime.
 * ------------------------------------------------------------------ */
static uint32_t hmx_int4_matmul_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
    (void)handle; (void)num_outputs; (void)num_inputs;

    /* QNN supplies post-Cast data: uint16 = (signed_int16 + 32768) and
     * uint8 = (signed_int8 + 128). gather_*_tile undoes that shift. */
    const uint16_t *au    = (const uint16_t *)qhpi_tensor_raw_data(inputs[0]);
    const uint8_t  *wu    = (const uint8_t  *)qhpi_tensor_raw_data(inputs[1]);
    void           *vtcm  =                   qhpi_tensor_raw_data(inputs[2]);
    int32_t        *out   = (int32_t *)qhpi_tensor_raw_data(outputs[0]);

    QHPI_Shape as = qhpi_tensor_shape(inputs[0]);
    QHPI_Shape ws = qhpi_tensor_shape(inputs[1]);

    /* Accept shapes [.., M, K] for act and [.., K, N] for wt.
     * Rank up to QHPI_MAX_RANK; reduce dims beyond last-2 as multiplied
     * batch, but for MVP we assume batch=1 and just read last-2. */
    const uint32_t M = dim_at(as, as.rank - 2);
    const uint32_t K = dim_at(as, as.rank - 1);
    const uint32_t N = dim_at(ws, ws.rank - 1);

#if defined(SCALAR_ONLY) || !defined(__hexagon__)
    (void)vtcm;
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            int32_t s = 0;
            for (uint32_t k = 0; k < K; k++) {
                int32_t av = (int32_t)au[i * K + k] - 32768;
                int32_t wv = (int32_t)wu[k * N + j] - 128;
                s += av * wv;
            }
            out[i * N + j] = s;
        }
    }
#else
    /* Static tile buffers — QHPI threads have small stacks; the kernel is
     * single-slice (multithreaded=false) so sequential invocations don't
     * race on these. */
    static int16_t a_tile[32 * 32];
    static int8_t  w_tile[32 * 32];
    static int32_t p_tile[32 * 32];
    for (uint32_t m0 = 0; m0 < M; m0 += 32) {
        for (uint32_t n0 = 0; n0 < N; n0 += 32) {
            bool first = true;
            for (uint32_t k0 = 0; k0 < K; k0 += 32) {
                gather_a_tile(a_tile, au, M, K, m0, k0);
                gather_w_tile(w_tile, wu, K, N, k0, n0);
                hmx_int4_matmul_tile(p_tile, a_tile, w_tile, vtcm);
                scatter_accum_tile(out, p_tile, M, N, m0, n0, first);
                first = false;
            }
        }
    }
#endif
    return QHPI_Success;
}

/* ------------------------------------------------------------------
 * Tensor signatures + op registration.
 * ------------------------------------------------------------------ */
/*
 * Type labels match what QNN actually supplies after graph optimization:
 * INT_16/INT_8 host tensors get coerced to QUInt16/QUInt8 with TCM placement.
 * The kernel reads raw bytes and reinterprets as int16/int8 internally — the
 * only thing that matters at match time is the bit-width and placement.
 */
static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    /* activation: int16 bits (labeled QUInt16 by QNN). */
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
    /* weight: int8 bits, sign-extended int4 (labeled QUInt8 by QNN). */
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
    /* scratch: uint8, TCM only. */
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_Int32,   QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};

static QHPI_Kernel_v1 sg_kernels[] = {
    {
        /* .function_name      */ THIS_PKG_NAME_STR "::hmx_int4_matmul",
        /* .function           */ hmx_int4_matmul_kernel,
        /* .resources          */ QHPI_RESOURCE_HMX,
        /* .source_destructive */ false,
        /* .multithreaded      */ false,
        /* .variable_inputs    */ false,
        /* .variable_outputs   */ false,
        /* .min_inputs         */ 3,
        /* .input_signature    */ sig_inputs,
        /* .min_outputs        */ 1,
        /* .output_signature   */ sig_outputs,
        /* .cost_function      */ nullptr,
        /* .sync_block_size    */ 0,
        /* .precomputed_data_size */ 0,
        /* .do_precomputation_function */ nullptr,
        /* .function_with_precomputed_data */ nullptr,
        /* .predicate          */ nullptr,
    },
};

static QHPI_OpInfo_v1 sg_ops[] = {
    {
        /* .name              */ THIS_PKG_NAME_STR "::MatMulInt4xInt16",
        /* .num_kernels       */ 1,
        /* .kernels           */ sg_kernels,
        /* .early_rewrite     */ nullptr,
        /* .shape_required    */ nullptr,
        /* .shape_legalized   */ nullptr,
        /* .tile_output       */ 0,
        /* .build_tile        */ nullptr,
        /* .late_rewrite      */ nullptr,
    },
};

void register_hmx_int4_matmul_ops() {
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
