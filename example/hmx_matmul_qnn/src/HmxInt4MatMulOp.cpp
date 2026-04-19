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
 * HMX kernel which expects the ordinary int16/int8 interpretation.
 *
 * Gather the M=32 × K strip of activation and the K × N=32 column of weight
 * from the user tensors into contiguous row-major int16/int8 buffers that
 * hmx_int4_matmul_mn_tile consumes. */
static void gather_a_strip(int16_t *__restrict__ a_strip,
                           const uint16_t *au,
                           uint32_t M, uint32_t K, uint32_t m0)
{
    const uint32_t rows = (m0 + 32 <= M) ? 32 : (M - m0);
    memset(a_strip, 0, sizeof(int16_t) * 32 * K);
    for (uint32_t i = 0; i < rows; i++)
        for (uint32_t k = 0; k < K; k++)
            a_strip[i * K + k] = (int16_t)((int32_t)au[(m0 + i) * K + k] - 32768);
}

static void gather_w_col(int8_t *__restrict__ w_col,
                         const uint8_t *wu,
                         uint32_t K, uint32_t N, uint32_t n0)
{
    const uint32_t cols = (n0 + 32 <= N) ? 32 : (N - n0);
    memset(w_col, 0, sizeof(int8_t) * K * 32);
    for (uint32_t k = 0; k < K; k++)
        for (uint32_t j = 0; j < cols; j++)
            w_col[k * 32 + j] = (int8_t)((int32_t)wu[k * N + (n0 + j)] - 128);
}

static void scatter_mn_tile(int32_t *out, const int32_t tile[32 * 32],
                            uint32_t M, uint32_t N, uint32_t m0, uint32_t n0)
{
    const uint32_t rows = (m0 + 32 <= M) ? 32 : (M - m0);
    const uint32_t cols = (n0 + 32 <= N) ? 32 : (N - n0);
    for (uint32_t i = 0; i < rows; i++) {
        int32_t *dst = &out[(m0 + i) * N + n0];
        const int32_t *src = &tile[i * 32];
        for (uint32_t j = 0; j < cols; j++) dst[j] = src[j];
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
    (void)num_outputs; (void)num_inputs;

    /* QNN supplies post-Cast data: uint16 = (signed_int16 + 32768) and
     * uint8 = (signed_int8 + 128). gather_*_tile undoes that shift. */
    const uint16_t *au    = (const uint16_t *)qhpi_tensor_raw_data(inputs[0]);
    const uint8_t  *wu    = (const uint8_t  *)qhpi_tensor_raw_data(inputs[1]);
    void           *vtcm  =                   qhpi_tensor_raw_data(inputs[2]);
    int32_t        *out   = (int32_t *)qhpi_tensor_raw_data(outputs[0]);

    /* Self-slicing across the M dimension (P2). Each slice handles a
     * contiguous band of m_tiles; no cross-slice sync needed since each
     * thread writes a distinct output region. */
    const uint32_t slice = qhpi_slice_number(handle);
    const uint32_t nslc  = qhpi_num_slices(handle);

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
    /* Per-slice static scratch. Up to 4 slices supported; each slice owns a
     * disjoint section. Sizing: 32 × 4096 bytes for activation strip +
     * 4096 × 32 for weight + 4KiB for output = ~260 KiB per slice. */
    #define MAX_SLICES 4
    static int16_t a_strip_pool[MAX_SLICES][32 * 4096];
    static int8_t  w_col_pool  [MAX_SLICES][4096 * 32];
    static int32_t mn_tile_pool[MAX_SLICES][32 * 32];
    const uint32_t si = (slice < MAX_SLICES) ? slice : 0;
    int16_t *a_strip = a_strip_pool[si];
    int8_t  *w_col   = w_col_pool[si];
    int32_t *mn_tile = mn_tile_pool[si];

    /* Partition M-tiles across slices. Each tile is 32 rows.
     * Total m_tile count = ceil(M / 32). Slice s handles
     * [s * ceil_tiles / nslc, (s+1) * ceil_tiles / nslc). */
    const uint32_t m_tiles = (M + 31) / 32;
    const uint32_t my_begin = (slice     * m_tiles) / nslc;
    const uint32_t my_end   = ((slice+1) * m_tiles) / nslc;

    /* Each slice needs its own VTCM scratch region. Size is
     * HMX_INT4_VTCM_BYTES_FOR_K(K); carve the scratch tensor into
     * MAX_SLICES equal slots. */
    const uint32_t per_slice_vtcm = HMX_INT4_VTCM_BYTES_FOR_K(K);
    uint8_t *my_vtcm = (uint8_t *)vtcm + si * per_slice_vtcm;

    for (uint32_t mt = my_begin; mt < my_end; mt++) {
        uint32_t m0 = mt * 32;
        gather_a_strip(a_strip, au, M, K, m0);
        /* Pre-pack activation once per m_tile (reused across all n_tiles). */
        hmx_int4_prepack_activation(a_strip, (int)K, my_vtcm);
        for (uint32_t n0 = 0; n0 < N; n0 += 32) {
            gather_w_col(w_col, wu, K, N, n0);
            /* Weight still packed per K-iter — the scalar pack interleaved
             * between HMX MACs hides VTCM read latency; all-prepacked was
             * slower due to back-to-back VTCM-bank contention. */
            hmx_int4_matmul_mn_using_prepacked_act(mn_tile, w_col, (int)K, my_vtcm);
            scatter_mn_tile(out, mn_tile, M, N, m0, n0);
        }
    }
#endif
    return QHPI_Success;
}

/* ------------------------------------------------------------------
 * Tensor signatures + op registration.
 * ------------------------------------------------------------------ */
/*
 * Signature uses Crouton layout for activation + weight. QNN's graph
 * optimizer auto-inserts ForceFormat_Crouton before our op, which is the
 * HVX-vectorized pack used by all the built-in HMX ops (see
 * ConvLayer_s1.opt disassembly: no HVX pack in the hot loop — data is
 * pre-laid-out by a separate framework op).
 *
 * Crouton_16 = 8×8×32 chunks of 16-bit data (4 KB per chunk);
 * Crouton_8  = 8×8×32 chunks of 8-bit data  (2 KB per chunk).
 */
/* For now, stay on Flat4 + Direct — kernel body still does its own pack.
 * Next step (P6): swap to Crouton_16 / Crouton_8 + Indirect and rewrite
 * kernel body to consume block-table data directly. Verified working:
 * declaring Crouton signatures causes QNN to auto-insert
 * ForceFormat_Crouton_f2c@{CH.FH, CB.FB} nodes upstream of our op. */
static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
    {QHPI_QUInt8,  QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
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
        /* .multithreaded      */ false,      /* self-slicing is HVX-only in QNN */
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
