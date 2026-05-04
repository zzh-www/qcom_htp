/*
 * HmxU8I8ToU8MatMulOp.cpp
 *
 * Single production custom MatMul path:
 *   bias[0], wt[1], act[2], scratch[3] -> out[0]
 *
 * The runtime body is the owned V73DEEP Conv1x1 kernel replica.  The default
 * QHPI path precomputes QNN tensor/block metadata at graph load, then the hot
 * callback only stitches the small native descriptor ABI expected by that
 * kernel.
 *
 * High-level data flow:
 *
 *   QNN/QHPI tensor form
 *       |
 *       |  QHPI precompute records metadata; hot callback stitches descriptors
 *       v
 *   native V73DEEP descriptor form
 *       |
 *       |  owned 1132-byte HMX inline-asm body consumes the descriptors
 *       v
 *   output Crouton_8 TCM blocks
 */

#include "HTP/core/qhpi.h"
#include <cstdint>
#include <cstring>

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

/*
 * These descriptor structs are the native skel ABI, not a new C++ API.
 * Their field order and offsets were chosen to match what the replicated
 * hmx_v73_convbbb1x1deep_stride1 body loads from r0/r1/r4.  Treat the comments
 * below as part of the ABI contract: changing a field order or unit silently
 * changes what the inline asm sees.
 *
 * Unit convention:
 *   - pointer-table strides are in 32-bit words because the kernel advances
 *     through tables of Hexagon pointers.
 *   - K byte counts are in bytes because the HMX weight stream is byte-packed.
 *   - tile counts are 32x32 logical tiles unless explicitly stated otherwise.
 */
struct hmx_conv_out_desc_t {
    int32_t *out_tile_ptr_table;      /* +0x00 */
    uint32_t out_table_stride_dwords; /* +0x04 */
    uint32_t out_y_stride_words;      /* +0x08 */
    uint32_t n_tiles_pow2;            /* +0x0c */
    int32_t m_total_minus_step;       /* +0x10 */
    uint32_t k_total_bytes;           /* +0x14 */
};

struct hmx_conv_act_desc_t {
    int32_t *act_ptr_pairs;            /* +0x00 */
    uint32_t n_act_pairs;              /* +0x04 */
    uint32_t act_table_y_stride_words; /* +0x08 */
};

struct hmx_conv_mask_desc_t {
    int32_t out_check;       /* +0x00 */
    uint32_t out_rt_mask;    /* +0x04 */
    int32_t act_check;       /* +0x08 */
    uint32_t act_rt_base;    /* +0x0c */
    uint32_t filter_x_stride;/* +0x10 */
    uint32_t _pad14;         /* +0x14 */
    uint32_t alt_rt;         /* +0x18 */
};

#if defined(__hexagon__)
#include <hexagon_types.h>

/*
 * This helper is provided by the QNN skel binary.  We do not reimplement its
 * bit packing here because the mask layout is internal to QNN's HMX conv
 * wrapper.  The arguments below are the decoded production tuple for the
 * V73DEEP 1x1 path we are cloning.
 */
extern "C" void _Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm(
    void *out_params,
    uint32_t arg1,
    uint32_t arg2,
    uint32_t arg3,
    uint32_t arg4,
    uint32_t arg5);
#define set_hmx_params_conv1x1 _Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm

#include "v73deep_conv1x1_kernel.h"
#endif

/*
 * QNN's Crouton_8 activation arrives as an indirect table of TCM blocks.
 * For this controlled replica flow the graph generator emits square shapes,
 * so the number of activation blocks is enough to recover S=M=K=N.
 *
 * This is intentionally a whitelist instead of a formula.  It prevents the
 * wrapper from attempting to run shapes whose Crouton tiling we have not
 * validated against the native V73DEEP descriptor rules.
 */
static uint32_t square_size_from_crouton_blocks(uint32_t blocks)
{
    switch (blocks) {
    case 4: return 32;
    case 8: return 64;
    case 16: return 128;
    case 32: return 256;
    case 128: return 512;
    case 512: return 1024;
    case 2048: return 2048;
    case 8192: return 4096;
    default: return 0;
    }
}

#if defined(__hexagon__)
static inline void store_le32(uint8_t *dst, uint32_t offset, uint32_t value)
{
    dst[offset + 0] = (uint8_t)(value & 0xffu);
    dst[offset + 1] = (uint8_t)((value >> 8) & 0xffu);
    dst[offset + 2] = (uint8_t)((value >> 16) & 0xffu);
    dst[offset + 3] = (uint8_t)((value >> 24) & 0xffu);
}

static void init_mask_desc(uint32_t *mask_buf)
{
    for (uint32_t i = 0; i < 16; ++i) mask_buf[i] = 0;
    set_hmx_params_conv1x1(mask_buf, 0x700, 0, 0, 0, 0x20);
}

static const hmx_conv_mask_desc_t *get_mask_desc(uint32_t *mask_buf)
{
    /*
     * The mask is shape-independent for the current production path, so we
     * initialize the stable part once.  The per-invocation extra_param pointer
     * is patched later because it points to stack storage in the current call.
     */
    static int initialized = 0;
    if (!initialized) {
        init_mask_desc(mask_buf);
        initialized = 1;
    }
    return reinterpret_cast<const hmx_conv_mask_desc_t *>(mask_buf);
}

#endif

#if defined(HMX_U8I8_ENABLE_QHPI_PRECOMPUTE)
static constexpr uint32_t kHmxU8I8PrecomputedDataSize = 56;
#endif

#if defined(__hexagon__) && !defined(SCALAR_ONLY) && defined(HMX_U8I8_ENABLE_QHPI_PRECOMPUTE)
/*
 * Default QHPI precompute path.
 *
 * Native QNN MatMul does not charge all setup work to the final ConvLayer
 * kernel event: constant movement, descriptor construction, and DMA
 * synchronization show up as sidecar HTP events such as weights_to_vtcm and
 * bias_to_vtcm.  QHPI exposes the same idea through do_precomputation_function:
 * it is called once when the graph is loaded, before inference executions.
 *
 * This struct is the custom-op equivalent of that native prepared state.  It
 * captures everything the hot callback used to rebuild every invocation:
 *
 *   QHPI tensors/block tables
 *        |
 *        | graph-load precompute
 *        v
 *   hmx_u8i8_precomputed_t
 *     - direct bias/weight pointers
 *     - activation/output Crouton block-table pointers
 *     - tile counts and descriptor constants
 *        |
 *        | inference hot path
 *        v
 *   our_v73deep_kernel(...)
 *
 * The large payload blocks and pointer tables are still owned by QNN.  The hot
 * callback no longer asks QHPI for tensor/block metadata; it only stitches the
 * tiny native descriptor ABI on stack and enters the HMX body.
 */
static constexpr uint32_t kHmxU8I8PrecomputeMagic = 0x48385850u; /* H8XP */

struct hmx_u8i8_precomputed_t {
    uint32_t magic;
    uint32_t S;
    uint32_t M_t;
    uint32_t N_t;
    uint32_t K_t;
    uint32_t mt_per_block;
    uint32_t mt_groups;
    uint32_t act_entries;
    uint32_t out_entries;
    const uint8_t *bias_bytes;
    const uint8_t *wt_pack;
    uint8_t *out_first_block;
    const int32_t *act_qhpi_table;
    const int32_t *out_qhpi_table;
};

static_assert(sizeof(hmx_u8i8_precomputed_t) == kHmxU8I8PrecomputedDataSize,
              "QHPI precompute ABI size must match the x86 registration stub");

static uint32_t hmx_u8i8_precompute(
    QHPI_RuntimeHandle *handle,
    void *data,
    uint32_t num_outputs,
    QHPI_Tensor **outputs,
    uint32_t num_inputs,
    const QHPI_Tensor *const *inputs)
{
    (void)handle;
    (void)num_outputs;
    (void)num_inputs;

    if (!data || !outputs || !outputs[0] || !inputs || num_inputs < 4) return QHPI_Success;

    hmx_u8i8_precomputed_t *pc = reinterpret_cast<hmx_u8i8_precomputed_t *>(data);
    std::memset(pc, 0, sizeof(*pc));

    const uint8_t *bias_bytes =
        reinterpret_cast<const uint8_t *>(qhpi_tensor_raw_data(inputs[0]));
    const uint8_t *wt_pack =
        reinterpret_cast<const uint8_t *>(qhpi_tensor_raw_data(inputs[1]));
    void **act_blocks = qhpi_tensor_block_table(inputs[2]);
    void **out_blocks = qhpi_tensor_block_table(outputs[0]);
    const uint32_t blocks = qhpi_tensor_block_table_length(inputs[2]);
    if (!bias_bytes || !wt_pack || !act_blocks || !out_blocks) return QHPI_Success;

    const uint32_t S = square_size_from_crouton_blocks(blocks);
    if (S < 128) return QHPI_Success;

    const uint32_t M_t = S / 32;
    const uint32_t N_t = S / 32;
    const uint32_t K_t = S / 32;
    const uint32_t block_rows = (S / 4) < 64 ? (S / 4) : 64;
    const uint32_t mt_per_block = block_rows / 32;
    if (mt_per_block == 0) return QHPI_Success;

    const uint32_t mt_groups = (mt_per_block == 2) ? (M_t >> 1) : M_t;
    const uint32_t act_entries = mt_groups * K_t;
    const uint32_t out_entries = mt_groups * N_t;
    if (act_entries > 1024 || out_entries > 1024) {
        return QHPI_Success;
    }

    const int32_t *act_src = reinterpret_cast<const int32_t *>(act_blocks);
    const int32_t *out_src = reinterpret_cast<const int32_t *>(out_blocks);

    pc->bias_bytes = bias_bytes;
    pc->wt_pack = wt_pack;
    pc->out_first_block = reinterpret_cast<uint8_t *>(out_blocks[0]);
    pc->act_qhpi_table = act_src;
    pc->out_qhpi_table = out_src;
    pc->S = S;
    pc->M_t = M_t;
    pc->N_t = N_t;
    pc->K_t = K_t;
    pc->mt_per_block = mt_per_block;
    pc->mt_groups = mt_groups;
    pc->act_entries = act_entries;
    pc->out_entries = out_entries;
    pc->magic = kHmxU8I8PrecomputeMagic;
    return QHPI_Success;
}

static uint32_t hmx_u8i8_to_u8_matmul_precomputed_kernel(
    QHPI_RuntimeHandle *handle,
    const void *precomputed_data)
{
    (void)handle;

    const hmx_u8i8_precomputed_t *pc =
        reinterpret_cast<const hmx_u8i8_precomputed_t *>(precomputed_data);
    if (!pc || pc->magic != kHmxU8I8PrecomputeMagic) return QHPI_Success;

#if defined(HMX_U8I8_PROBE_CYCLES)
    uint64_t cyc_start = 0;
    asm volatile("%0 = C15:14" : "=r"(cyc_start));
#endif

    /*
     * Hot path after QHPI precompute:
     *
     *   prepared state: bias/weight raw pointers, QNN Crouton block tables,
     *                   tile counts recovered once at graph load
     *   per invoke:     stitch three tiny native descriptors on stack
     *   compute:        jump into the owned V73DEEP body
     *
     * The pointer tables themselves are not copied here.  QNN already prepared
     * them as part of tensor materialization, and the V73DEEP table order
     * matches the Crouton block-table order for this C8 path.
     */
    int32_t *act_tbl_ptr = const_cast<int32_t *>(pc->act_qhpi_table);
    int32_t *out_tbl_ptr = const_cast<int32_t *>(pc->out_qhpi_table);

    static uint32_t mask_buf[16] __attribute__((aligned(16)));
    uint32_t extra_param_local[16] __attribute__((aligned(16))) = {1u, 0u};
    const hmx_conv_mask_desc_t *mask_desc = get_mask_desc(mask_buf);
    mask_buf[0x38 / 4] =
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(extra_param_local));

    hmx_conv_out_desc_t out_desc_local = {
        out_tbl_ptr,
        pc->N_t,
        pc->M_t * 4,
        pc->M_t * 4,
        8,
        pc->N_t * 32,
    };
    hmx_conv_act_desc_t act_desc_local = {
        act_tbl_ptr,
        pc->K_t,
        pc->M_t * 4,
    };
    const hmx_conv_out_desc_t *out_desc = &out_desc_local;
    const hmx_conv_act_desc_t *act_desc = &act_desc_local;
    const uint32_t *extra_param = extra_param_local;

#if defined(HMX_U8I8_DESC_DUMP)
    if (pc->out_first_block) {
        uint8_t *dst = pc->out_first_block;
        for (uint32_t i = 0; i < 128; ++i) dst[i] = 0;
        store_le32(dst, 0, 0x48385844u); /* H8XD */
        store_le32(dst, 4, pc->S);
        store_le32(dst, 8, pc->M_t);
        store_le32(dst, 12, pc->N_t);
        store_le32(dst, 16, pc->K_t);
        store_le32(dst, 20, pc->mt_per_block);
        store_le32(dst, 24, reinterpret_cast<uintptr_t>(act_desc->act_ptr_pairs));
        store_le32(dst, 28, reinterpret_cast<uintptr_t>(out_desc->out_tile_ptr_table));
        store_le32(dst, 32, out_desc->out_table_stride_dwords);
        store_le32(dst, 36, out_desc->out_y_stride_words);
        store_le32(dst, 40, out_desc->n_tiles_pow2);
        store_le32(dst, 44, static_cast<uint32_t>(out_desc->m_total_minus_step));
        store_le32(dst, 48, out_desc->k_total_bytes);
        store_le32(dst, 52, act_desc->n_act_pairs);
        store_le32(dst, 56, act_desc->act_table_y_stride_words);
        const uint32_t *mask_words = reinterpret_cast<const uint32_t *>(mask_desc);
        for (uint32_t i = 0; i < 16; ++i) store_le32(dst, 64 + i * 4, mask_words[i]);
    }
    return QHPI_Success;
#endif

#if defined(HMX_U8I8_SKIP_KERNEL)
    if (pc->out_first_block) {
        uint8_t *dst = pc->out_first_block;
        for (uint32_t i = 0; i < 16; ++i) dst[i] = 0;
        store_le32(dst, 0, 0x48385853u); /* H8XS */
        store_le32(dst, 4, pc->S);
    }
    return QHPI_Success;
#endif

#if defined(HMX_U8I8_PROBE_CYCLES)
    uint64_t cyc_before_kernel = 0;
    asm volatile("%0 = C15:14" : "=r"(cyc_before_kernel));
#endif

    /*
     * Precomputed hot path: QHPI tensor lookup, shape recovery, and pointer
     * table discovery happened at graph load.  The custom op event now starts
     * at the tiny native descriptor stitching boundary.
     */
    our_v73deep_kernel(
        out_desc,
        act_desc,
        pc->wt_pack,
        pc->bias_bytes,
        mask_desc,
        extra_param);

#if defined(HMX_U8I8_PROBE_CYCLES)
    uint64_t cyc_after_kernel = 0;
    asm volatile("%0 = C15:14" : "=r"(cyc_after_kernel));
    if (pc->out_first_block) {
        uint8_t *dst = pc->out_first_block;
        store_le32(dst, 0, static_cast<uint32_t>(cyc_after_kernel - cyc_before_kernel));
        store_le32(dst, 4, static_cast<uint32_t>(cyc_before_kernel - cyc_start));
        store_le32(dst, 8, 0);
        store_le32(dst, 12, 0);
    }
#endif

    return QHPI_Success;
}
#elif defined(HMX_U8I8_ENABLE_QHPI_PRECOMPUTE)
/*
 * The x86 context generator must see the same QHPI kernel registration as the
 * device package.  It does not execute the HMX body, so these are registration
 * stubs only; the real precompute/execute implementation is compiled into the
 * Hexagon HTP package above.
 */
static uint32_t hmx_u8i8_precompute(
    QHPI_RuntimeHandle *,
    void *,
    uint32_t,
    QHPI_Tensor **,
    uint32_t,
    const QHPI_Tensor *const *)
{
    return QHPI_Success;
}

static uint32_t hmx_u8i8_to_u8_matmul_precomputed_kernel(
    QHPI_RuntimeHandle *,
    const void *)
{
    return QHPI_Success;
}
#endif

/*
 * Runtime input-layout bridge.
 *
 * What QNN gives this custom op:
 *
 *   inputs[0] bias       Flat4 Direct
 *       -> qhpi_tensor_raw_data()
 *       -> bias_bytes
 *
 *   inputs[1] weight     Flat4 Direct, already K-major packed
 *       -> qhpi_tensor_raw_data()
 *       -> wt_pack
 *
 *   inputs[2] activation Crouton_8 Indirect
 *       -> qhpi_tensor_block_table()
 *       -> act_blocks: table of TCM block pointers
 *
 *   outputs[0] output    Crouton_8 Indirect
 *       -> qhpi_tensor_block_table()
 *       -> out_blocks: table of TCM block pointers
 *
 * The wrapper then builds native descriptors:
 *
 *   act_blocks void**       out_blocks void**
 *        |                       |
 *        | copy pointer table    | copy pointer table
 *        v                       v
 *   act_tbl_all int32[]     out_tbl_all int32[]
 *        |                       |
 *        v                       v
 *   act_desc                out_desc
 *        |                       |
 *        +-----------+-----------+
 *                    |
 *                    v
 *        our_v73deep_kernel(out_desc, act_desc, wt, bias, mask, extra)
 *
 * Important: activation/output payload blocks are not copied or repacked here.
 * Only their pointer tables are reshaped into the ABI expected by the native
 * V73DEEP Conv1x1 kernel. Weight packing and bias folding are already done by
 * the graph generator/converter flow.
 *
 * In other words, this function is mostly a metadata adapter:
 *
 *   direct payload pointers       pass through unchanged
 *   indirect block tables         copied into native pointer-table arrays
 *   descriptor constants          filled in the units expected by skel asm
 *   HMX computation               delegated to our_v73deep_kernel()
 */
static uint32_t hmx_u8i8_to_u8_matmul_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs,
    QHPI_Tensor **outputs,
    uint32_t num_inputs,
    const QHPI_Tensor *const *inputs)
{
    (void)handle;
    (void)num_outputs;
    (void)num_inputs;

#if !defined(__hexagon__) || defined(SCALAR_ONLY)
    return QHPI_Success;
#else
#if defined(HMX_U8I8_PROBE_CYCLES)
    uint64_t cyc_start = 0;
    asm volatile("%0 = C15:14" : "=r"(cyc_start));
#endif

    if (!outputs || !outputs[0] || !inputs || num_inputs < 4) return QHPI_Success;

    /*
     * Extract the two direct buffers and the two indirect Crouton tables.
     *
     * Direct tensors:
     *
     *   bias tensor     weight tensor
     *       |               |
     *       v               v
     *   bias_bytes      wt_pack
     *
     * Indirect tensors:
     *
     *   activation tensor             output tensor
     *       |                            |
     *       v                            v
     *   act_blocks void**             out_blocks void**
     *   +-----+-----+-----+---        +-----+-----+-----+---
     *   | p0  | p1  | p2  | ...       | p0  | p1  | p2  | ...
     *   +--+--+--+--+--+--+---        +--+--+--+--+--+--+---
     *      |     |     |                 |     |     |
     *      v     v     v                 v     v     v
     *     TCM   TCM   TCM               TCM   TCM   TCM
     *    block block block             block block block
     */
    const uint8_t *bias_bytes =
        reinterpret_cast<const uint8_t *>(qhpi_tensor_raw_data(inputs[0]));
    const uint8_t *wt_pack =
        reinterpret_cast<const uint8_t *>(qhpi_tensor_raw_data(inputs[1]));
    void **act_blocks = qhpi_tensor_block_table(inputs[2]);
    void **out_blocks = qhpi_tensor_block_table(outputs[0]);
    const uint32_t blocks = qhpi_tensor_block_table_length(inputs[2]);
    if (!bias_bytes || !wt_pack || !act_blocks || !out_blocks) return QHPI_Success;

    /*
     * Shape recovery.
     *
     * QHPI does expose tensor shapes, but the native kernel path is driven by
     * Crouton tiling, so the activation block table length is the most direct
     * signal for this wrapper.  Once S is known, all kernel loop quantities are
     * expressed as counts of 32-wide tiles:
     *
     *   M_t = output row tiles
     *   N_t = output column / weight-N tiles
     *   K_t = reduction / activation-K tiles
     *
     * The current generated flow is square-only, so all three are S/32.
     */
    const uint32_t S = square_size_from_crouton_blocks(blocks);
    if (S < 128) return QHPI_Success;

    const uint32_t M_t = S / 32;
    const uint32_t N_t = S / 32;
    const uint32_t K_t = S / 32;
    const uint32_t block_rows = (S / 4) < 64 ? (S / 4) : 64;
    const uint32_t mt_per_block = block_rows / 32;
    if (mt_per_block == 0) return QHPI_Success;

    /*
     * QNN's Crouton block table groups up to two 32-row tiles in one physical
     * block for the C8 path.  The native V73DEEP descriptor wants one pointer
     * table row per such group, so 256^3 becomes:
     *
     *   M_t=8, mt_per_block=2 -> mt_groups=4
     *   act entries = mt_groups * K_t = 4 * 8 = 32
     *   out entries = mt_groups * N_t = 4 * 8 = 32
     */
    const uint32_t mt_groups = (mt_per_block == 2) ? (M_t >> 1) : M_t;
    const uint32_t act_entries = mt_groups * K_t;
    const uint32_t out_entries = mt_groups * N_t;
    if (act_entries > 1024 || out_entries > 1024) return QHPI_Success;

#if defined(HMX_U8I8_PROBE_CYCLES)
    uint64_t cyc_after_qhpi = 0;
    asm volatile("%0 = C15:14" : "=r"(cyc_after_qhpi));
#endif

    int32_t act_tbl_all[1024] __attribute__((aligned(64)));
    int32_t out_tbl_all[1024] __attribute__((aligned(64)));
    const int32_t *act_src = reinterpret_cast<const int32_t *>(act_blocks);
    const int32_t *out_src = reinterpret_cast<const int32_t *>(out_blocks);

    /*
     * Rebuild only the native pointer tables.
     *
     * QNN table as seen by the custom op:
     *
     *   act_blocks / out_blocks
     *   +------+------+------+------+-----+
     *   | ptr0 | ptr1 | ptr2 | ptr3 | ... |
     *   +------+------+------+------+-----+
     *
     * Native table passed through the descriptor:
     *
     *   act_tbl_all / out_tbl_all
     *   +------+------+------+------+-----+
     *   | ptr0 | ptr1 | ptr2 | ptr3 | ... |
     *   +------+------+------+------+-----+
     *
     * Same pointer values, different owner/location. On Hexagon these are
     * 32-bit pointers, hence the int32_t tables. For the canonical 256^3 case:
     *
     *   S=256, M_t=N_t=K_t=8
     *   entries per table = mt_groups * K_t = 4 * 8 = 32
     *   32 pointers * 4 bytes = 128 bytes = one HVX_Vector
     */
#if defined(__hexagon__)
    if (mt_per_block == 2 && K_t == 8 && N_t == 8) {
        // Keep the canonical 256^3 C8 path as a pair of 128B HVX copies.
        // Scalarizing these table copies costs about 80 extra hot-op packets.
        HVX_Vector act_vec;
        HVX_Vector out_vec;
        std::memcpy(&act_vec, act_blocks, sizeof(HVX_Vector));
        std::memcpy(&out_vec, out_blocks, sizeof(HVX_Vector));
        std::memcpy(act_tbl_all, &act_vec, sizeof(HVX_Vector));
        std::memcpy(out_tbl_all, &out_vec, sizeof(HVX_Vector));
    } else
#endif
    if (mt_per_block == 2) {
        /*
         * Generic C8 grouped fallback.  It preserves the same table order as
         * QNN gives us, but copies group-by-group so non-256 square sizes still
         * produce a contiguous native table.
         */
        for (uint32_t rg = 0; rg < mt_groups; ++rg) {
            const int32_t *__restrict a_src = act_src + rg * K_t;
            const int32_t *__restrict o_src = out_src + rg * N_t;
            int32_t *__restrict a_dst = act_tbl_all + rg * K_t;
            int32_t *__restrict o_dst = out_tbl_all + rg * N_t;
            for (uint32_t kt = 0; kt < K_t; ++kt) a_dst[kt] = a_src[kt];
            for (uint32_t nt = 0; nt < N_t; ++nt) o_dst[nt] = o_src[nt];
        }
    } else {
        /*
         * Smaller shapes can have only one 32-row tile per block.  In that case
         * the native table is indexed directly by M tile instead of M-pair group.
         */
        for (uint32_t rg = 0; rg < M_t; ++rg) {
            const int32_t *__restrict a_src = act_src + rg * K_t;
            const int32_t *__restrict o_src = out_src + rg * N_t;
            int32_t *__restrict a_dst = act_tbl_all + rg * K_t;
            int32_t *__restrict o_dst = out_tbl_all + rg * N_t;
            for (uint32_t kt = 0; kt < K_t; ++kt) a_dst[kt] = a_src[kt];
            for (uint32_t nt = 0; nt < N_t; ++nt) o_dst[nt] = o_src[nt];
        }
    }

#if defined(HMX_U8I8_PROBE_CYCLES)
    uint64_t cyc_after_tables = 0;
    asm volatile("%0 = C15:14" : "=r"(cyc_after_tables));
#endif

    /*
     * Build the native descriptor ABI consumed by the embedded V73DEEP body.
     *
     *   act_tbl_all
     *       |
     *       v
     *   act_desc
     *     +0x00 act_ptr_pairs            -> act_tbl_all
     *     +0x04 n_act_pairs              =  K_t
     *     +0x08 act_table_y_stride_words =  M_t * 4
     *
     *   out_tbl_all
     *       |
     *       v
     *   out_desc
     *     +0x00 out_tile_ptr_table       -> out_tbl_all
     *     +0x04 out_table_stride_dwords  =  N_t
     *     +0x08 out_y_stride_words       =  M_t * 4
     *     +0x0c n_tiles_pow2             =  M_t * 4
     *     +0x10 m_total_minus_step       =  8
     *     +0x14 k_total_bytes            =  N_t * 32
     *
     * Final register ABI after the C call lowers on Hexagon:
     *
     *   r0 -> out_desc       r1 -> act_desc
     *   r2 -> wt_pack        r3 -> bias_bytes
     *   r4 -> mask_desc      r5 -> extra_param
     *
     * The constants below are copied from the decoded native Conv1x1 path:
     *
     *   out_table_stride_dwords = N_t       next N tile pointer in same M group
     *   out_y_stride_words      = M_t * 4   y stride after native <<2 scaling
     *   n_tiles_pow2            = M_t * 4   native loop/count selector
     *   m_total_minus_step      = 8         decoded fixed step for this path
     *   k_total_bytes           = N_t * 32  byte span expected by kernel setup
     *   n_act_pairs             = K_t       number of activation pointer pairs
     */
    static uint32_t mask_buf[16] __attribute__((aligned(16)));
    uint32_t extra_param[16] __attribute__((aligned(16))) = {1u, 0u};
    const hmx_conv_mask_desc_t *mask_desc = get_mask_desc(mask_buf);
    mask_buf[0x38 / 4] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(extra_param));

    hmx_conv_out_desc_t out_desc = {
        out_tbl_all,
        N_t,
        M_t * 4,
        M_t * 4,
        8,
        N_t * 32,
    };
    hmx_conv_act_desc_t act_desc = {
        act_tbl_all,
        K_t,
        M_t * 4,
    };

#if defined(HMX_U8I8_DESC_DUMP)
    /*
     * Descriptor dump mode writes the derived ABI state into the first output
     * block and returns before HMX compute.  Use this when validating that QNN's
     * tensor/block metadata was translated into the expected native descriptor.
     */
    if (out_blocks[0]) {
        uint8_t *dst = reinterpret_cast<uint8_t *>(out_blocks[0]);
        for (uint32_t i = 0; i < 128; ++i) dst[i] = 0;
        store_le32(dst, 0, 0x48385844u); /* H8XD */
        store_le32(dst, 4, S);
        store_le32(dst, 8, M_t);
        store_le32(dst, 12, N_t);
        store_le32(dst, 16, K_t);
        store_le32(dst, 20, mt_per_block);
        store_le32(dst, 24, reinterpret_cast<uintptr_t>(act_tbl_all));
        store_le32(dst, 28, reinterpret_cast<uintptr_t>(out_tbl_all));
        store_le32(dst, 32, out_desc.out_table_stride_dwords);
        store_le32(dst, 36, out_desc.out_y_stride_words);
        store_le32(dst, 40, out_desc.n_tiles_pow2);
        store_le32(dst, 44, static_cast<uint32_t>(out_desc.m_total_minus_step));
        store_le32(dst, 48, out_desc.k_total_bytes);
        store_le32(dst, 52, act_desc.n_act_pairs);
        store_le32(dst, 56, act_desc.act_table_y_stride_words);
        for (uint32_t i = 0; i < 16; ++i) store_le32(dst, 64 + i * 4, mask_buf[i]);
    }
    return QHPI_Success;
#endif

#if defined(HMX_U8I8_SKIP_KERNEL)
    /*
     * Skip mode proves the custom-op callback, tensor extraction, and output
     * write path are alive while avoiding the HMX body.  A successful device run
     * in this mode isolates failures to descriptor/kernel state rather than QNN
     * package registration.
     */
    if (out_blocks[0]) {
        uint8_t *dst = reinterpret_cast<uint8_t *>(out_blocks[0]);
        for (uint32_t i = 0; i < 16; ++i) dst[i] = 0;
        store_le32(dst, 0, 0x48385853u); /* H8XS */
        store_le32(dst, 4, S);
    }
    return QHPI_Success;
#endif

#if defined(HMX_U8I8_PROBE_CYCLES)
    uint64_t cyc_before_kernel = 0;
    asm volatile("%0 = C15:14" : "=r"(cyc_before_kernel));
#endif

    /*
     * Boundary between wrapper and compute:
     *
     * Everything above prepares the native ABI.  Everything below is the owned
     * V73DEEP kernel body; the C++ wrapper does not inspect accumulators or
     * implement matmul arithmetic.
     */
    our_v73deep_kernel(&out_desc, &act_desc, wt_pack, bias_bytes, mask_desc, extra_param);

#if defined(HMX_U8I8_PROBE_CYCLES)
    uint64_t cyc_after_kernel = 0;
    asm volatile("%0 = C15:14" : "=r"(cyc_after_kernel));
    if (out_blocks[0]) {
        uint8_t *dst = reinterpret_cast<uint8_t *>(out_blocks[0]);
        store_le32(dst, 0, static_cast<uint32_t>(cyc_after_kernel - cyc_before_kernel));
        store_le32(dst, 4, static_cast<uint32_t>(cyc_before_kernel - cyc_after_tables));
        store_le32(dst, 8, static_cast<uint32_t>(cyc_after_tables - cyc_after_qhpi));
        store_le32(dst, 12, static_cast<uint32_t>(cyc_after_qhpi - cyc_start));
    }
#endif

    return QHPI_Success;
#endif
}

static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_Int32,  QHPI_Layout_Flat4,     QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},
    {QHPI_QUInt8, QHPI_Layout_Flat4,     QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},
    {QHPI_QUInt8, QHPI_Layout_Crouton_8, QHPI_Storage_Indirect, QHPI_MemLoc_TCM_Only},
    {QHPI_QUInt8, QHPI_Layout_Flat4,     QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},
};

static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Crouton_8, QHPI_Storage_Indirect, QHPI_MemLoc_TCM_Only},
};

static float hmx_u8i8_cost_function(uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
    if (!inputs || num_inputs < 3 || !inputs[2]) return 1.0f;
    QHPI_Shape s = qhpi_tensor_shape(inputs[2]);
    uint32_t M = 256;
    uint32_t K = 256;
    if (s.rank == 4) {
        M = s.dims[1] * s.dims[2];
        K = s.dims[3];
    } else if (s.rank == 3) {
        M = s.dims[1];
        K = s.dims[2];
    }
    const uint32_t M_t = (M + 31) / 32;
    const uint32_t K_t = (K + 31) / 32;
    const uint32_t N_t = K_t;
    return static_cast<float>(M_t * N_t * K_t) * 16.0f;
}

static QHPI_Shape hmx_u8i8_shape_required(const QHPI_Op *op)
{
    (void)op;
    QHPI_Shape req = {0};
    req.rank = 4;
    req.dims[0] = 1;
    req.dims[1] = 1;
    req.dims[2] = 32;
    req.dims[3] = 32;
    return req;
}

static QHPI_Shape hmx_u8i8_shape_legalized(const QHPI_Op *op, const QHPI_Shape *proposed)
{
    (void)op;
    QHPI_Shape s = *proposed;
    if (s.rank >= 4) {
        if (s.dims[1] < 1) s.dims[1] = 1;
        if (s.dims[2] < 32) s.dims[2] = 32;
        if (s.dims[3] < 32) s.dims[3] = 32;
        if (s.dims[3] % 32) s.dims[3] = ((s.dims[3] + 31) / 32) * 32;
    }
    return s;
}

static QHPI_Kernel_v1 sg_kernels[] = {
    {
#if defined(HMX_U8I8_ENABLE_QHPI_PRECOMPUTE)
        THIS_PKG_NAME_STR "::hmx_u8i8_to_u8_matmul_precomputed_kernel",
        nullptr,
#else
        THIS_PKG_NAME_STR "::hmx_u8i8_to_u8_matmul_kernel",
        hmx_u8i8_to_u8_matmul_kernel,
#endif
        QHPI_RESOURCE_HMX,
        false,
        false,
        false, false,
        4, sig_inputs,
        1, sig_outputs,
        hmx_u8i8_cost_function,
        0,
#if defined(HMX_U8I8_ENABLE_QHPI_PRECOMPUTE)
        kHmxU8I8PrecomputedDataSize,
        hmx_u8i8_precompute,
        hmx_u8i8_to_u8_matmul_precomputed_kernel,
#else
        0,
        nullptr,
        nullptr,
#endif
        nullptr,
    },
};

static QHPI_OpInfo_v1 sg_ops[] = {
    {
        THIS_PKG_NAME_STR "::HmxU8I8ToU8MatMul",
        1, sg_kernels,
        nullptr,
        hmx_u8i8_shape_required,
        hmx_u8i8_shape_legalized,
        0,
        nullptr,
        nullptr,
    },
};

extern "C" void register_hmx_u8i8_to_u8_matmul_op(void)
{
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
