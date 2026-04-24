/*
 * HmxMatMulPhase3Op.cpp — Phase 3A probe: int8×int8 matmul via Crouton_8
 * + Indirect, HMX-only kernel.
 *
 * Goal: verify QNN's auto-inserted ForceFormat_Crouton_b produces blocks
 * that HMX mxmem can consume directly. No HVX pack, no scalar layout
 * munging inside this op.
 *
 * Kernel body:
 *   for each (m_tile, n_tile):
 *     mxclracc
 *     for each k_tile:
 *       activation.ub = mxmem(act_block_addr, Rt_act|0x1C):cm
 *       weight.b      = mxmem(wt_block_addr, 0x3FF)
 *     readback to output (dual-scale)
 *
 * Block_table walk: for now, assume Crouton_8 tensor produces one block
 * per (8·8·32 = 2048 element) chunk. Shape [1,1,M,K] → block count
 * = ceil(M/8) × ceil(K/32). We iterate and pass block pointers.
 */

#include "HTP/core/qhpi.h"
#include <cstdint>
#include <cstring>
#include <cstdio>

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

/* --------------------------------------------------------------------
 * HMX asm wrappers (same as Phase 2 kernels).
 * -------------------------------------------------------------------- */
#if defined(__hexagon__)
static inline __attribute__((always_inline)) void hmx_clracc_i(void)
{ asm volatile("mxclracc" ::: "memory"); }

static inline __attribute__((always_inline)) void hmx_load_bias_i(const void *p)
{ asm volatile("bias = mxmem(%0)" :: "r"(p) : "memory"); }

static inline __attribute__((always_inline))
void hmx_load_pair_u8_i8(const void *act, const void *wt)
{
    const int HMX_RT_ACT = 2047;
    const int HMX_RT_WT  = 0x3FF;
    asm volatile(
        "{ activation.ub = mxmem(%0, %1)\n"
        "  weight.b      = mxmem(%2, %3) }\n"
        :: "r"(act), "r"(HMX_RT_ACT),
           "r"(wt),  "r"(HMX_RT_WT)
        : "memory");
}

static inline __attribute__((always_inline))
void hmx_store_acc_uh_2x1_retain(void *out)
{ asm volatile("mxmem(%0, %1):after:retain.uh = acc:2x1\n"
               :: "r"(out), "r"(0) : "memory"); }

static inline __attribute__((always_inline))
void hmx_store_acc_uh_2x1(void *out)
{ asm volatile("mxmem(%0, %1):after.uh = acc:2x1\n"
               :: "r"(out), "r"(0) : "memory"); }
#endif

/* --------------------------------------------------------------------
 * Kernel entry.
 * -------------------------------------------------------------------- */
static uint32_t phase3_hmx_matmul_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
    (void)num_outputs; (void)num_inputs;
    (void)handle;

#if !defined(__hexagon__) || defined(SCALAR_ONLY)
    /* Scalar fallback: straight matmul from raw-data pointers (Direct
     * storage assumed in this build). Meant for CPU prepare-time
     * validation; not used on-device. */
    const int8_t *a = (const int8_t *)qhpi_tensor_raw_data(inputs[0]);
    const int8_t *w = (const int8_t *)qhpi_tensor_raw_data(inputs[1]);
    int32_t *out = (int32_t *)qhpi_tensor_raw_data(outputs[0]);
    QHPI_Shape as = qhpi_tensor_shape(inputs[0]);
    QHPI_Shape ws = qhpi_tensor_shape(inputs[1]);
    const uint32_t M = as.dims[as.rank - 2];
    const uint32_t K = as.dims[as.rank - 1];
    const uint32_t N = ws.dims[ws.rank - 1];
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            int32_t s = 0;
            for (uint32_t k = 0; k < K; k++)
                s += (int32_t)a[i * K + k] * (int32_t)w[k * N + j];
            out[i * N + j] = s;
        }
    }
    return QHPI_Success;
#else
    /* Phase 3A: just probe — read block_table pointers, do one tile of
     * HMX mxmem, compare to scalar. Single 32×32×32 tile only. */

    /* Query block tables. */
    void **act_blocks = qhpi_tensor_block_table(inputs[0]);
    void **wt_blocks  = qhpi_tensor_block_table(inputs[1]);
    uint32_t act_nblk = qhpi_tensor_block_table_length(inputs[0]);
    uint32_t wt_nblk  = qhpi_tensor_block_table_length(inputs[1]);
    QHPI_Shape act_block_shape = qhpi_tensor_block_shape(inputs[0]);
    QHPI_Shape wt_block_shape  = qhpi_tensor_block_shape(inputs[1]);

    int32_t *out = (int32_t *)qhpi_tensor_raw_data(outputs[0]);
    QHPI_Shape os = qhpi_tensor_shape(outputs[0]);
    const uint32_t M = os.dims[os.rank - 2];
    const uint32_t N = os.dims[os.rank - 1];

    /* Probe: write diagnostics into out[] itself as ints so host can read.
     * Layout:
     *   out[0]      = magic 0xDEADBEEF (sanity marker)
     *   out[1]      = act_nblk
     *   out[2]      = wt_nblk
     *   out[3..6]   = act_block_shape dims[0..3]
     *   out[7..10]  = wt_block_shape dims[0..3]
     *   out[11..14] = out_shape dims[0..3]
     *   out[15..30] = act block 0 bytes 0..63 packed 4 bytes per int32
     *   out[31..46] = wt  block 0 bytes 0..63 packed 4 bytes per int32
     */
    out[0]  = (int32_t)0xDEADBEEF;
    out[1]  = (int32_t)act_nblk;
    out[2]  = (int32_t)wt_nblk;
    out[3]  = (int32_t)act_block_shape.dims[0];
    out[4]  = (int32_t)act_block_shape.dims[1];
    out[5]  = (int32_t)act_block_shape.dims[2];
    out[6]  = (int32_t)act_block_shape.dims[3];
    out[7]  = (int32_t)wt_block_shape.dims[0];
    out[8]  = (int32_t)wt_block_shape.dims[1];
    out[9]  = (int32_t)wt_block_shape.dims[2];
    out[10] = (int32_t)wt_block_shape.dims[3];
    out[11] = (int32_t)os.dims[0];
    out[12] = (int32_t)os.dims[1];
    out[13] = (int32_t)os.dims[2];
    out[14] = (int32_t)os.dims[3];
    if (act_nblk > 0) {
        const uint32_t *ab0 = (const uint32_t *)act_blocks[0];
        for (int i = 0; i < 16; i++) out[15 + i] = (int32_t)ab0[i];
    }
    if (wt_nblk > 0) {
        const uint32_t *wb0 = (const uint32_t *)wt_blocks[0];
        for (int i = 0; i < 16; i++) out[31 + i] = (int32_t)wb0[i];
    }
    for (uint32_t i = 47; i < M * N; i++) out[i] = 0;

    return QHPI_Success;
#endif
}

/* --------------------------------------------------------------------
 * Signatures — Crouton_8 + Indirect for inputs so QNN auto-inserts
 * ForceFormat_Crouton_b. Output Direct int32 for now (host reads raw).
 * -------------------------------------------------------------------- */
static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Crouton_8, QHPI_Storage_Indirect, QHPI_MemLoc_TCM_Only},
    {QHPI_QUInt8, QHPI_Layout_Crouton_8, QHPI_Storage_Indirect, QHPI_MemLoc_TCM_Only},
};
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_Int32,  QHPI_Layout_Flat4,     QHPI_Storage_Direct,   QHPI_MemLoc_DDR_OR_TCM},
};

static QHPI_Kernel_v1 sg_kernels[] = {
    {
        /* .function_name      */ THIS_PKG_NAME_STR "::phase3_hmx_matmul",
        /* .function           */ phase3_hmx_matmul_kernel,
        /* .resources          */ QHPI_RESOURCE_HMX,
        /* .source_destructive */ false,
        /* .multithreaded      */ false,
        /* .variable_inputs    */ false,
        /* .variable_outputs   */ false,
        /* .min_inputs         */ 2,
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
        /* .name              */ THIS_PKG_NAME_STR "::MatMulInt8xInt8Crouton",
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

void register_phase3_ops() {
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
