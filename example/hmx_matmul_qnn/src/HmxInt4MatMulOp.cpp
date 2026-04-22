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
    /* Skip memset when cols==32 — every byte is written below. */
    if (cols < 32) memset(w_col, 0, sizeof(int8_t) * K * 32);
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
    static int8_t  w_col_pool  [MAX_SLICES][4096 * 32];
    static int32_t mn_tile_pool[MAX_SLICES][32 * 32];
    const uint32_t si = (slice < MAX_SLICES) ? slice : 0;
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

    /* P4: gather_w_col depends only on n0, not m0. Cache per-n_tile w_col
     * in a small working buffer and detect m-loop-iter via mt index to
     * skip redundant gathers on subsequent m iterations.
     *
     * Simpler alternative that doesn't need a big cache: the first m
     * iteration performs all gathers (populating w_col per n_tile into a
     * shared pool), later iterations reuse them. But sizing is 16 n_tiles
     * × K*32 = 256KB at K=N=512 — too big for stack/BSS ×4 slices.
     *
     * MVP implementation: re-gather on every iteration but the hope is
     * the DDR cache keeps it L2-resident across m iterations. Revisit if
     * profile shows gather_w_col is >10% of total.
     */
    for (uint32_t mt = my_begin; mt < my_end; mt++) {
        uint32_t m0 = mt * 32;
        hmx_int4_prepack_activation_fused(au, (int)M, (int)K, (int)m0, my_vtcm);
        for (uint32_t n0 = 0; n0 < N; n0 += 32) {
            gather_w_col(w_col, wu, K, N, n0);
            hmx_int4_matmul_mn_dualacc(mn_tile, w_col, (int)K, my_vtcm);
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

/* ------------------------------------------------------------------
 * P8 — Auto-tile callbacks.
 * Declare that M (dim 2) and N (dim 3) of the output must be multiples
 * of 32 (HMX tile grain). QNN's central tiling system will carve the
 * output [1,1,M,N] into 32-aligned sub-tiles based on its own VTCM / HVX
 * parallelism heuristics, creating N_sub_tiles independent invocations
 * of our op each computing one [M_tile, N_tile] region of the output.
 * Those independent ops land on separate HVX threads → 2–4× parallelism.
 * ------------------------------------------------------------------ */
/* Auto-tile is registered but empirically does NOT help in isolation:
 *   shape_required=32:  16-way tiling at 512³ → 51% REGRESSION (per-op
 *                       setup dominates; lost K-accumulation amortization)
 *   shape_required=256: 4-way tiling at 512³  →  5% regression (2.22 vs
 *                       single-op 2.12; only 1 HMX unit, sub-ops serialize)
 *   shape_required=1024: no tiling for ≤1024³ — identical to single-op
 *
 * For auto-tile to win, the kernel needs substantial HVX-side work that
 * can execute concurrently on separate HVX threads while HMX is issuing.
 * Without HVX-vectorized pack/unpack/combine (P7, deferred), all sub-ops
 * serialize on the single HMX compute unit and the single scalar main
 * thread. Setting shape_required to a large value effectively disables
 * tiling at our test scales while leaving the machinery ready for a
 * future session that combines auto-tile with HVX parallelism. */
static QHPI_Shape matmul_shape_required(const QHPI_Op *op)
{
    (void)op;
    QHPI_Shape s;
    s.rank = 4;
    s.dims[0] = 1;
    s.dims[1] = 1;
    s.dims[2] = 2048;
    s.dims[3] = 2048;
    return s;
}

static QHPI_Shape matmul_shape_legalized(const QHPI_Op *op, const QHPI_Shape *proposed)
{
    (void)op;
    QHPI_Shape s = *proposed;
    if (s.rank >= 4) {
        s.dims[0] = 1;
        s.dims[1] = 1;
        s.dims[2] = (s.dims[2] + 2047u) & ~2047u;
        s.dims[3] = (s.dims[3] + 2047u) & ~2047u;
    }
    return s;
}

/* Build the sub-graph computing one output tile [start..start+extent).
 *
 * Inputs:
 *   0: activation [1, 1, M_full, K_full]
 *   1: weight     [1, 1, K_full, N_full]
 *   2: scratch    [1, 1, 1, vtcm_bytes]  (shared; unsliced)
 *
 * Output tile:  [1, 1, extent_m, extent_n] @ (0, 0, start_m, start_n)
 *
 * We slice activation along M (dim 2) and weight along N (dim 3), each
 * keeping K_full. The scratch is reused as-is (the kernel writes/reads
 * only within its own invocation, so sharing across sub-ops that run
 * on different threads would race — here we accept the risk and verify
 * empirically; a proper fix is to carve scratch by thread/op). */
static const QHPI_Op *matmul_build_tile(const QHPI_Op *op,
                                        const QHPI_Shape *start,
                                        const QHPI_Shape *extent)
{
    QHPI_OpRef act_in = qhpi_op_input(op, 0);
    QHPI_OpRef wt_in  = qhpi_op_input(op, 1);
    QHPI_OpRef sc_in  = qhpi_op_input(op, 2);

    /* Full shapes of inputs. */
    QHPI_OutputDef act_def = qhpi_op_output(act_in.op, act_in.output_number);
    QHPI_OutputDef wt_def  = qhpi_op_output(wt_in.op,  wt_in.output_number);

    /* Activation slice: (0, 0, start_m, 0) .. (.., .., extent_m, K_full) */
    QHPI_Shape a_start = {4, {0, 0, start->dims[2], 0}};
    QHPI_Shape a_ext;
    a_ext.rank = 4;
    a_ext.dims[0] = 1;
    a_ext.dims[1] = 1;
    a_ext.dims[2] = extent->dims[2];
    a_ext.dims[3] = act_def.shape.dims[3];    /* K_full */

    /* Weight slice: (0, 0, 0, start_n) .. (.., .., K_full, extent_n) */
    QHPI_Shape w_start = {4, {0, 0, 0, start->dims[3]}};
    QHPI_Shape w_ext;
    w_ext.rank = 4;
    w_ext.dims[0] = 1;
    w_ext.dims[1] = 1;
    w_ext.dims[2] = wt_def.shape.dims[2];     /* K_full */
    w_ext.dims[3] = extent->dims[3];

    QHPI_OpRef a_sliced = qhpi_op_slice(act_in, &a_start, &a_ext);
    QHPI_OpRef w_sliced = qhpi_op_slice(wt_in,  &w_start, &w_ext);

    QHPI_OpRef inputs[3] = { a_sliced, w_sliced, sc_in };
    /* Output def for the sub-op — same quant/type as the original, but with the
     * sliced extent. */
    QHPI_OutputDef out_def = qhpi_op_output(op, 0);
    out_def.shape = *extent;

    const char *op_name = qhpi_op_name(op);
    return qhpi_op_create(op, op_name ? op_name : THIS_PKG_NAME_STR "::MatMulInt4xInt16",
                         3, inputs, 1, &out_def);
}

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
        /* .shape_required    */ matmul_shape_required,
        /* .shape_legalized   */ matmul_shape_legalized,
        /* .tile_output       */ 0,
        /* .build_tile        */ matmul_build_tile,
        /* .late_rewrite      */ nullptr,
    },
};

void register_hmx_int4_matmul_ops() {
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
