/*
 * HmxU8I4ToU8MatMulOp.cpp
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

#ifndef HMX_W4A8_MASK_ARG3
#define HMX_W4A8_MASK_ARG3 0u
#endif
#ifndef HMX_W4A8_MASK_ARG4
#define HMX_W4A8_MASK_ARG4 0u
#endif
#ifndef HMX_W4A8_MASK_ARG5
#define HMX_W4A8_MASK_ARG5 0u
#endif
#ifndef HMX_W4A8_MASK_ARG6
#define HMX_W4A8_MASK_ARG6 0x20u
#endif

static inline uint32_t hmx_w4a8_desc_m_tiles(uint32_t m_t, uint32_t mt_groups)
{
#if defined(HMX_W4A8_DESC_M_TILES)
    (void)m_t;
    (void)mt_groups;
    return HMX_W4A8_DESC_M_TILES;
#elif defined(HMX_W4A8_DESC_USE_MT_GROUPS)
    (void)m_t;
    return mt_groups;
#else
    (void)mt_groups;
    return m_t * 4u;
#endif
}

static constexpr uintptr_t kHmxW4A8CroutonMGroupStrideBytes = 256u;

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
extern "C" void _Z25set_hmx_params_convw4b1x1P10hmx_paramsmmmmmm(
    void *out_params,
    uint32_t arg1,
    uint32_t arg2,
    uint32_t arg3,
    uint32_t arg4,
    uint32_t arg5,
    uint32_t arg6);
#define set_hmx_params_convw4b1x1 _Z25set_hmx_params_convw4b1x1P10hmx_paramsmmmmmm

extern "C" void hmx_v73_convbnb1x1_stride1(
    const hmx_conv_out_desc_t *od,
    const hmx_conv_act_desc_t *ad,
    const uint8_t *wt,
    const uint8_t *bias,
    const hmx_conv_mask_desc_t *mask,
    const uint32_t *extra_param);

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

static inline uint8_t hmx_w4a8_block_pattern_byte(uint32_t block, uint32_t offset)
{
#if defined(HMX_W4A8_BLOCK_PATTERN_DUMP_HIGH)
    (void)block;
    return static_cast<uint8_t>((offset >> 5) & 0xffu);
#elif defined(HMX_W4A8_BLOCK_PATTERN_DUMP_BLOCK)
    (void)offset;
    return static_cast<uint8_t>(block & 0xffu);
#else
    return static_cast<uint8_t>(((block & 7u) << 5) | (offset & 31u));
#endif
}

static inline uint32_t hmx_w4a8_crouton_row8_blocks(uint32_t m16_tiles)
{
    return ((m16_tiles + 7u) >> 3) * 2u;
}

static inline uint32_t hmx_w4a8_crouton_row8_block_index(
    uint32_t row8_tile,
    uint32_t kn_tile,
    uint32_t kn_tiles)
{
    const uint32_t m16_tile = row8_tile >> 1;
    const uint32_t row8_half = row8_tile & 1u;
    const uint32_t m16_chunk = m16_tile >> 3;
    return (m16_chunk * 2u + row8_half) * kn_tiles + kn_tile;
}

static inline int32_t hmx_w4a8_crouton_row8_ptr(
    const int32_t *qhpi_table,
    uint32_t row8_tile,
    uint32_t kn_tile,
    uint32_t kn_tiles)
{
    const uint32_t m16_tile = row8_tile >> 1;
    const uint32_t m16_in_chunk = m16_tile & 7u;
    const uint32_t block_index =
        hmx_w4a8_crouton_row8_block_index(row8_tile, kn_tile, kn_tiles);
    const uintptr_t ptr = static_cast<uint32_t>(qhpi_table[block_index]);
    return static_cast<int32_t>(
        ptr + m16_in_chunk * kHmxW4A8CroutonMGroupStrideBytes);
}

static uint32_t g_hmx_w4a8_mask_buf[16] __attribute__((aligned(16)));

static void init_mask_desc(uint32_t *mask_buf, uint32_t k_total)
{
    for (uint32_t i = 0; i < 16; ++i) mask_buf[i] = 0;
    set_hmx_params_convw4b1x1(
        mask_buf,
        0x700,
        k_total,
        HMX_W4A8_MASK_ARG3,
        HMX_W4A8_MASK_ARG4,
        HMX_W4A8_MASK_ARG5,
        HMX_W4A8_MASK_ARG6);
}

static const hmx_conv_mask_desc_t *get_mask_desc(uint32_t k_total)
{
    /*
     * The mask is shape-independent for the current production path.  Native
     * QNN sets bit 5 in the conv1x1 mask tuple and then the wrapper tail-calls
     * the V73DEEP body.  For that direct deep entry, r5 already carries the
     * extra_param pointer, so the old mask[0x38] patch is not needed.
     */
    init_mask_desc(g_hmx_w4a8_mask_buf, k_total);
    return reinterpret_cast<const hmx_conv_mask_desc_t *>(g_hmx_w4a8_mask_buf);
}

static const hmx_conv_mask_desc_t *get_precomputed_mask_desc()
{
    return reinterpret_cast<const hmx_conv_mask_desc_t *>(g_hmx_w4a8_mask_buf);
}

#endif

#if defined(HMX_W4A8_ENABLE_QHPI_PRECOMPUTE)
static constexpr uint32_t kHmxW4A8MaxCopiedTableEntries = 512;
static constexpr uint32_t kHmxW4A8PrecomputedDataSize =
    56 + 2 * kHmxW4A8MaxCopiedTableEntries * sizeof(int32_t);
#endif

#if defined(__hexagon__) && !defined(SCALAR_ONLY) && defined(HMX_W4A8_ENABLE_QHPI_PRECOMPUTE)
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
 *   hmx_w4a8_precomputed_t
 *     - direct bias/weight pointers
 *     - activation/output Crouton block-table pointers
 *     - tile counts and descriptor constants
 *        |
 *        | inference hot path
 *        v
 *   our_v73deep_kernel(...)
 *
 * The large payload blocks are still owned by QNN.  For validated small shapes
 * the pointer-table values are copied into this precomputed record at graph
 * load, so the hot callback no longer asks QHPI for tensor/block metadata or
 * touches the original QNN table storage.  It only stitches the tiny native
 * descriptor ABI on stack and enters the HMX body.
 */
static constexpr uint32_t kHmxW4A8PrecomputeMagic = 0x48385850u; /* H8XP */

struct hmx_w4a8_precomputed_t {
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
    int32_t act_table_copy[kHmxW4A8MaxCopiedTableEntries];
    int32_t out_table_copy[kHmxW4A8MaxCopiedTableEntries];
};

static_assert(sizeof(hmx_w4a8_precomputed_t) == kHmxW4A8PrecomputedDataSize,
              "QHPI precompute ABI size must match the x86 registration stub");

static uint32_t hmx_w4a8_precompute(
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

    hmx_w4a8_precomputed_t *pc = reinterpret_cast<hmx_w4a8_precomputed_t *>(data);
    std::memset(pc, 0, sizeof(*pc));

    const uint8_t *bias_bytes =
        reinterpret_cast<const uint8_t *>(qhpi_tensor_raw_data(inputs[0]));
    const uint8_t *wt_pack =
        reinterpret_cast<const uint8_t *>(qhpi_tensor_raw_data(inputs[1]));
    void **act_blocks = qhpi_tensor_block_table(inputs[2]);
    void **out_blocks = qhpi_tensor_block_table(outputs[0]);
    const uint32_t act_block_entries = qhpi_tensor_block_table_length(inputs[2]);
    const uint32_t out_block_entries = qhpi_tensor_block_table_length(outputs[0]);
    if (!bias_bytes || !wt_pack || !act_blocks || !out_blocks) return QHPI_Success;

    const QHPI_Shape act_shape = qhpi_tensor_shape(inputs[2]);
    const QHPI_Shape out_shape = qhpi_tensor_shape(outputs[0]);
    if (act_shape.rank != 4 || out_shape.rank != 4) return QHPI_Success;

    const uint32_t M = act_shape.dims[1] * act_shape.dims[2];
    const uint32_t K = act_shape.dims[3];
    const uint32_t N = out_shape.dims[3];
    if ((act_shape.dims[2] != 16 && act_shape.dims[2] != 32) ||
        M < 128 || K < 32 || N < 32) {
        return QHPI_Success;
    }
    if ((M % 16) || (K % 32) || (N % 32)) return QHPI_Success;
    init_mask_desc(g_hmx_w4a8_mask_buf, K);

    const uint32_t M_t = M / 16;
    const uint32_t N_t = N / 32;
    const uint32_t K_t = K / 32;
    const uint32_t mt_per_block = 8;
    const uint32_t row8_groups = M_t * 2u;
    const uint32_t crouton_row8_blocks = hmx_w4a8_crouton_row8_blocks(M_t);
    const uint32_t mt_groups = row8_groups;
    const uint32_t physical_act_entries = crouton_row8_blocks * K_t;
    const uint32_t physical_out_entries = crouton_row8_blocks * N_t;
    const uint32_t act_entries = row8_groups * K_t;
    const uint32_t out_entries = row8_groups * N_t;
    if (act_block_entries < physical_act_entries || out_block_entries < physical_out_entries) {
        return QHPI_Success;
    }
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
    if (act_entries <= kHmxW4A8MaxCopiedTableEntries &&
        out_entries <= kHmxW4A8MaxCopiedTableEntries) {
        for (uint32_t row8 = 0; row8 < row8_groups; ++row8) {
            for (uint32_t kt = 0; kt < K_t; ++kt) {
                pc->act_table_copy[row8 * K_t + kt] =
                    hmx_w4a8_crouton_row8_ptr(act_src, row8, kt, K_t);
            }
            for (uint32_t nt = 0; nt < N_t; ++nt) {
                pc->out_table_copy[row8 * N_t + nt] =
                    hmx_w4a8_crouton_row8_ptr(out_src, row8, nt, N_t);
            }
        }
        pc->act_qhpi_table = pc->act_table_copy;
        pc->out_qhpi_table = pc->out_table_copy;
    }
    pc->S = M;
    pc->M_t = M_t;
    pc->N_t = N_t;
    pc->K_t = K_t;
    pc->mt_per_block = mt_per_block;
    pc->mt_groups = mt_groups;
    pc->act_entries = act_entries;
    pc->out_entries = out_entries;
    pc->magic = kHmxW4A8PrecomputeMagic;
    return QHPI_Success;
}

static uint32_t hmx_w4a8_to_u8_matmul_precomputed_kernel(
    QHPI_RuntimeHandle *handle,
    const void *precomputed_data)
{
    (void)handle;

    const hmx_w4a8_precomputed_t *pc =
        reinterpret_cast<const hmx_w4a8_precomputed_t *>(precomputed_data);

#if defined(HMX_W4A8_PROBE_CYCLES)
    uint64_t cyc_start = 0;
    asm volatile("%0 = C15:14" : "=r"(cyc_start));
#endif

    /*
     * Hot path after QHPI precompute:
     *
     *   prepared state: bias/weight raw pointers, local Crouton block-table
     *                   copies, tile counts recovered once at graph load
     *   per invoke:     stitch three tiny native descriptors on stack
     *   compute:        jump into the owned V73DEEP body
     *
     * For the canonical 256^3 shape these tables are copied into
     * hmx_w4a8_precomputed_t at graph load.  That removes the last hot-path
     * QNN table locality miss without spending extra dcfetch packets in the
     * profiled callback.  Larger fallback shapes keep pointing at QNN-owned
     * block tables.
     */
    int32_t *act_tbl_ptr = const_cast<int32_t *>(pc->act_qhpi_table);
    int32_t *out_tbl_ptr = const_cast<int32_t *>(pc->out_qhpi_table);

    uint32_t extra_param[2] __attribute__((aligned(16))) = {1u, 0u};
    const hmx_conv_mask_desc_t *mask_desc = get_precomputed_mask_desc();
    const uint32_t desc_m_t = hmx_w4a8_desc_m_tiles(pc->M_t, pc->mt_groups);

    hmx_conv_out_desc_t out_desc_local __attribute__((aligned(64))) = {
        out_tbl_ptr,
        pc->N_t,
        desc_m_t * 4,
        desc_m_t * 4,
        8,
        pc->N_t * 32,
    };
    hmx_conv_act_desc_t act_desc_local __attribute__((aligned(64))) = {
        act_tbl_ptr,
        pc->K_t,
        desc_m_t * 4,
    };
    const hmx_conv_out_desc_t *out_desc = &out_desc_local;
    const hmx_conv_act_desc_t *act_desc = &act_desc_local;

#if defined(HMX_W4A8_DESC_DUMP)
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

#if defined(HMX_W4A8_BLOCK_PATTERN_DUMP)
    if (pc->out_first_block) {
        uint8_t *dst = pc->out_first_block;
        for (uint32_t i = 0; i < 2048; ++i) dst[i] = static_cast<uint8_t>(i);
    }
    return QHPI_Success;
#endif

#if defined(HMX_W4A8_SKIP_KERNEL)
    if (pc->out_first_block) {
        uint8_t *dst = pc->out_first_block;
        for (uint32_t i = 0; i < 16; ++i) dst[i] = 0;
        store_le32(dst, 0, 0x48385853u); /* H8XS */
        store_le32(dst, 4, pc->S);
    }
    return QHPI_Success;
#endif

#if defined(HMX_W4A8_PROBE_CYCLES)
    uint64_t cyc_before_kernel = 0;
    asm volatile("%0 = C15:14" : "=r"(cyc_before_kernel));
#endif

    /*
     * Precomputed hot path: QHPI tensor lookup, shape recovery, and pointer
     * table discovery happened at graph load.  The custom op event now starts
     * at the tiny native descriptor stitching boundary.
     */
#if defined(HMX_W4A8_USE_SKEL_KERNEL)
    hmx_v73_convbnb1x1_stride1(
#else
    our_v73deep_kernel(
#endif
        out_desc,
        act_desc,
        pc->wt_pack,
        pc->bias_bytes,
        mask_desc,
        extra_param);

#if defined(HMX_W4A8_PROBE_CYCLES)
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
#elif defined(HMX_W4A8_ENABLE_QHPI_PRECOMPUTE)
/*
 * The x86 context generator must see the same QHPI kernel registration as the
 * device package.  It does not execute the HMX body, so these are registration
 * stubs only; the real precompute/execute implementation is compiled into the
 * Hexagon HTP package above.
 */
static uint32_t hmx_w4a8_precompute(
    QHPI_RuntimeHandle *,
    void *,
    uint32_t,
    QHPI_Tensor **,
    uint32_t,
    const QHPI_Tensor *const *)
{
    return QHPI_Success;
}

static uint32_t hmx_w4a8_to_u8_matmul_precomputed_kernel(
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
static uint32_t hmx_w4a8_to_u8_matmul_kernel(
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
#if defined(HMX_W4A8_PROBE_CYCLES)
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
    const uint32_t act_block_entries = qhpi_tensor_block_table_length(inputs[2]);
    const uint32_t out_block_entries = qhpi_tensor_block_table_length(outputs[0]);
    if (!bias_bytes || !wt_pack || !act_blocks || !out_blocks) return QHPI_Success;

    /*
     * Shape recovery.
     *
     * The W4 native MatMul/FC path uses Crouton row tiles of 16 for activations
     * and outputs.  The native descriptor still counts K/N in 32-wide tiles:
     *
     *   M_t = output 16-row groups
     *   N_t = output column / weight-N tiles
     *   K_t = reduction / activation-K tiles
     */
    const QHPI_Shape act_shape = qhpi_tensor_shape(inputs[2]);
    const QHPI_Shape out_shape = qhpi_tensor_shape(outputs[0]);
    if (act_shape.rank != 4 || out_shape.rank != 4) return QHPI_Success;

    const uint32_t M = act_shape.dims[1] * act_shape.dims[2];
    const uint32_t K = act_shape.dims[3];
    const uint32_t N = out_shape.dims[3];
    if ((act_shape.dims[2] != 16 && act_shape.dims[2] != 32) ||
        M < 128 || K < 32 || N < 32) {
        return QHPI_Success;
    }
    if ((M % 16) || (K % 32) || (N % 32)) return QHPI_Success;

#if defined(HMX_W4A8_BLOCK_INFO_DUMP)
    if (out_blocks[0]) {
        uint8_t *dst = reinterpret_cast<uint8_t *>(out_blocks[0]);
        for (uint32_t i = 0; i < 64; ++i) dst[i] = 0;
        store_le32(dst, 0, 0x48384942u); /* H8IB */
        store_le32(dst, 4, act_block_entries);
        store_le32(dst, 8, out_block_entries);
        store_le32(dst, 12, M / 16);
        store_le32(dst, 16, K / 32);
        store_le32(dst, 20, N / 32);
    }
    return QHPI_Success;
#endif

#if defined(HMX_W4A8_BLOCK_PATTERN_DUMP)
    for (uint32_t b = 0; b < out_block_entries; ++b) {
        if (!out_blocks[b]) continue;
        uint8_t *dst = reinterpret_cast<uint8_t *>(out_blocks[b]);
        for (uint32_t i = 0; i < 2048; ++i) {
            dst[i] = hmx_w4a8_block_pattern_byte(b, i);
        }
    }
    return QHPI_Success;
#endif

    const uint32_t M_t = M / 16;
    const uint32_t N_t = N / 32;
    const uint32_t K_t = K / 32;
    const uint32_t mt_per_block = 8;
    const uint32_t row8_groups = M_t * 2u;
    const uint32_t crouton_row8_blocks = hmx_w4a8_crouton_row8_blocks(M_t);
    const uint32_t mt_groups = row8_groups;
    const uint32_t physical_act_entries = crouton_row8_blocks * K_t;
    const uint32_t physical_out_entries = crouton_row8_blocks * N_t;
    const uint32_t act_entries = row8_groups * K_t;
    const uint32_t out_entries = row8_groups * N_t;
    if (act_block_entries < physical_act_entries || out_block_entries < physical_out_entries) {
        return QHPI_Success;
    }
    if (act_entries > 1024 || out_entries > 1024) return QHPI_Success;

#if defined(HMX_W4A8_PROBE_CYCLES)
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
     * 32-bit pointers, hence the int32_t tables.
     */
    for (uint32_t row8 = 0; row8 < row8_groups; ++row8) {
        int32_t *__restrict a_dst = act_tbl_all + row8 * K_t;
        int32_t *__restrict o_dst = out_tbl_all + row8 * N_t;
        for (uint32_t kt = 0; kt < K_t; ++kt) {
            a_dst[kt] = hmx_w4a8_crouton_row8_ptr(act_src, row8, kt, K_t);
        }
        for (uint32_t nt = 0; nt < N_t; ++nt) {
            o_dst[nt] = hmx_w4a8_crouton_row8_ptr(out_src, row8, nt, N_t);
        }
    }

#if defined(HMX_W4A8_PROBE_CYCLES)
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
    uint32_t extra_param[2] __attribute__((aligned(16))) = {1u, 0u};
    const hmx_conv_mask_desc_t *mask_desc = get_mask_desc(K);
    const uint32_t desc_m_t = hmx_w4a8_desc_m_tiles(M_t, mt_groups);

    hmx_conv_out_desc_t out_desc = {
        out_tbl_all,
        N_t,
        desc_m_t * 4,
        desc_m_t * 4,
        8,
        N_t * 32,
    };
    hmx_conv_act_desc_t act_desc = {
        act_tbl_all,
        K_t,
        desc_m_t * 4,
    };

#if defined(HMX_W4A8_DESC_DUMP)
    /*
     * Descriptor dump mode writes the derived ABI state into the first output
     * block and returns before HMX compute.  Use this when validating that QNN's
     * tensor/block metadata was translated into the expected native descriptor.
     */
    if (out_blocks[0]) {
        uint8_t *dst = reinterpret_cast<uint8_t *>(out_blocks[0]);
        for (uint32_t i = 0; i < 128; ++i) dst[i] = 0;
        store_le32(dst, 0, 0x48385844u); /* H8XD */
        store_le32(dst, 4, M);
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
        const uint32_t *mask_words = reinterpret_cast<const uint32_t *>(mask_desc);
        for (uint32_t i = 0; i < 16; ++i) store_le32(dst, 64 + i * 4, mask_words[i]);
    }
    return QHPI_Success;
#endif

#if defined(HMX_W4A8_BLOCK_PATTERN_DUMP)
    for (uint32_t b = 0; b < out_block_entries; ++b) {
        if (!out_blocks[b]) continue;
        uint8_t *dst = reinterpret_cast<uint8_t *>(out_blocks[b]);
        for (uint32_t i = 0; i < 2048; ++i) {
            dst[i] = hmx_w4a8_block_pattern_byte(b, i);
        }
    }
    return QHPI_Success;
#endif

#if defined(HMX_W4A8_SKIP_KERNEL)
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
        store_le32(dst, 4, M);
    }
    return QHPI_Success;
#endif

#if defined(HMX_W4A8_PROBE_CYCLES)
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
#if defined(HMX_W4A8_USE_SKEL_KERNEL)
    hmx_v73_convbnb1x1_stride1(&out_desc, &act_desc, wt_pack, bias_bytes, mask_desc, extra_param);
#else
    our_v73deep_kernel(&out_desc, &act_desc, wt_pack, bias_bytes, mask_desc, extra_param);
#endif

#if defined(HMX_W4A8_PROBE_CYCLES)
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
    {QHPI_Int32,            QHPI_Layout_Flat4,     QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},
    {QHPI_Any_Element_Type, QHPI_Layout_Flat4,     QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},
    {QHPI_QUInt8,           QHPI_Layout_Crouton_8, QHPI_Storage_Indirect, QHPI_MemLoc_TCM_Only},
    {QHPI_QUInt8,           QHPI_Layout_Flat4,     QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},
};

static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QUInt8,           QHPI_Layout_Crouton_8, QHPI_Storage_Indirect, QHPI_MemLoc_TCM_Only},
};

static float hmx_w4a8_cost_function(uint32_t num_inputs, const QHPI_Tensor *const *inputs)
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
    const uint32_t M_t = (M + 15) / 16;
    const uint32_t K_t = (K + 31) / 32;
    const uint32_t N_t = K_t;
    return static_cast<float>(M_t * N_t * K_t) * 16.0f;
}

static QHPI_Shape hmx_w4a8_shape_required(const QHPI_Op *op)
{
    (void)op;
    QHPI_Shape req = {0};
    req.rank = 4;
    req.dims[0] = 1;
    req.dims[1] = 1;
    req.dims[2] = 16;
    req.dims[3] = 32;
    return req;
}

static QHPI_Shape hmx_w4a8_shape_legalized(const QHPI_Op *op, const QHPI_Shape *proposed)
{
    (void)op;
    QHPI_Shape s = *proposed;
    if (s.rank >= 4) {
        if (s.dims[1] < 1) s.dims[1] = 1;
        s.dims[2] = 16;
        if (s.dims[3] < 32) s.dims[3] = 32;
        if (s.dims[3] % 32) s.dims[3] = ((s.dims[3] + 31) / 32) * 32;
    }
    return s;
}

static QHPI_Kernel_v1 sg_kernels[] = {
    {
#if defined(HMX_W4A8_ENABLE_QHPI_PRECOMPUTE)
        THIS_PKG_NAME_STR "::hmx_w4a8_to_u8_matmul_precomputed_kernel",
        nullptr,
#else
        THIS_PKG_NAME_STR "::hmx_w4a8_to_u8_matmul_kernel",
        hmx_w4a8_to_u8_matmul_kernel,
#endif
        QHPI_RESOURCE_HMX,
        false,
        false,
        false, false,
        4, sig_inputs,
        1, sig_outputs,
        hmx_w4a8_cost_function,
        0,
#if defined(HMX_W4A8_ENABLE_QHPI_PRECOMPUTE)
        kHmxW4A8PrecomputedDataSize,
        hmx_w4a8_precompute,
        hmx_w4a8_to_u8_matmul_precomputed_kernel,
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
        THIS_PKG_NAME_STR "::HmxU8I4ToU8MatMul",
        1, sg_kernels,
        nullptr,
        hmx_w4a8_shape_required,
        hmx_w4a8_shape_legalized,
        0,
        nullptr,
        nullptr,
    },
};

extern "C" void register_hmx_w4a8_to_u8_matmul_op(void)
{
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
