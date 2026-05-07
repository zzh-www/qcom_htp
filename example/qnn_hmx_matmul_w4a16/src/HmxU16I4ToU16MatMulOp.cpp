/*
 * HmxU16I4ToU16MatMulOp.cpp
 *
 * Single production custom MatMul path:
 *   bias[0], wt[1], act[2], control[3] -> out[0]
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

#if defined(__hexagon__) && !defined(HMX_W4A16_SKIP_KERNEL) && \
    !defined(HMX_W4A16_ALLOW_UNVALIDATED_KERNEL)
#error "w4a16 HMX body is not validated yet; keep HMX_W4A16_SKIP_KERNEL until the family-specific body is wired."
#endif

#ifndef HMX_W4A16_MASK_ARG1
#define HMX_W4A16_MASK_ARG1 0x70b
#endif
#ifndef HMX_W4A16_MASK_ARG3
#define HMX_W4A16_MASK_ARG3 0u
#endif
#ifndef HMX_W4A16_MASK_ARG4
#define HMX_W4A16_MASK_ARG4 0u
#endif
#ifndef HMX_W4A16_MASK_ARG5
#define HMX_W4A16_MASK_ARG5 0u
#endif
#ifndef HMX_W4A16_MASK_ARG6
#define HMX_W4A16_MASK_ARG6 0x20u
#endif
#ifndef HMX_W4A16_EXTRA_PARAM0
#define HMX_W4A16_EXTRA_PARAM0 1u
#endif
#ifndef HMX_W4A16_EXTRA_PARAM1
#define HMX_W4A16_EXTRA_PARAM1 1025u
#endif
#ifndef HMX_W4A16_EXTRA_PARAM2
#define HMX_W4A16_EXTRA_PARAM2 524u
#endif
#ifndef HMX_W4A16_USE_ROW4_TABLES
#define HMX_W4A16_USE_ROW4_TABLES 1
#endif
#ifndef HMX_W4A16_WEIGHT_PTR_OFFSET
#define HMX_W4A16_WEIGHT_PTR_OFFSET 0
#endif
#ifndef HMX_W4A16_BIAS_PTR_OFFSET
#define HMX_W4A16_BIAS_PTR_OFFSET 0
#endif

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

extern "C" void hmx_v73_convhnh1x1_stride1(
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

static bool hmx_w4a16_tile_counts_from_shapes(
    const QHPI_Tensor *activation,
    const QHPI_Tensor *weight,
    const QHPI_Tensor *output,
    uint32_t act_blocks,
    uint32_t *M_t,
    uint32_t *N_t,
    uint32_t *K_t)
{
    uint32_t M = 0;
    uint32_t K = 0;
    uint32_t N = 0;
    if (activation) {
        const QHPI_Shape a = qhpi_tensor_shape(activation);
        if (a.rank == 4) {
            M = a.dims[1] * a.dims[2];
            K = a.dims[3];
        } else if (a.rank == 3) {
            M = a.dims[1];
            K = a.dims[2];
        }
    }
    if (output) {
        const QHPI_Shape o = qhpi_tensor_shape(output);
        if (o.rank == 4) {
            if (M == 0) M = o.dims[1] * o.dims[2];
            N = o.dims[3];
        } else if (o.rank == 3) {
            if (M == 0) M = o.dims[1];
            N = o.dims[2];
        }
    }
    if (weight) {
        const QHPI_Shape w = qhpi_tensor_shape(weight);
        if (w.rank == 4) {
            if (K == 0) {
                if (w.dims[2] * 2u == N && N != 0) {
                    K = w.dims[2] * 2u;
                } else {
                    K = w.dims[2];
                }
            }
            if (N == 0) {
                if (K != 0 && w.dims[2] * 2u == K) {
                    N = w.dims[3];
                } else {
                    N = w.dims[3] * 2u;
                }
            }
        }
    }
    if (M == 0 || K == 0 || N == 0) {
        const uint32_t S = square_size_from_crouton_blocks(act_blocks);
        M = S;
        K = S;
        N = S;
    }
    if (M < 128 || K < 128 || N < 128 || (M % 32) || (K % 32) || (N % 32)) {
        return false;
    }
    *M_t = M / 32;
    *K_t = K / 32;
    *N_t = N / 32;
    return true;
}

static inline uint32_t hmx_w4a16_desc_m_tiles(uint32_t m_t, uint32_t row4_groups)
{
#if defined(HMX_W4A16_DESC_M_TILES_OVERRIDE)
    (void)m_t;
    (void)row4_groups;
    return HMX_W4A16_DESC_M_TILES_OVERRIDE;
#elif defined(HMX_W4A16_DESC_USE_ROW4_GROUPS)
    (void)m_t;
    return row4_groups;
#else
    (void)m_t;
    return row4_groups * 4u;
#endif
}

static inline uint32_t hmx_w4a16_act_table_storage_stride(uint32_t k_t)
{
#if defined(HMX_W4A16_ACT_TABLE_STRIDE_OVERRIDE)
    (void)k_t;
    return HMX_W4A16_ACT_TABLE_STRIDE_OVERRIDE;
#else
    uint32_t stride = k_t;
#if defined(HMX_W4A16_ACT_TABLE_Y_STRIDE_WORDS_OVERRIDE)
    if (stride < HMX_W4A16_ACT_TABLE_Y_STRIDE_WORDS_OVERRIDE) {
        stride = HMX_W4A16_ACT_TABLE_Y_STRIDE_WORDS_OVERRIDE;
    }
#endif
    return stride;
#endif
}

static inline uint32_t hmx_w4a16_out_table_storage_stride(uint32_t n_t)
{
#if defined(HMX_W4A16_OUT_TABLE_STRIDE_OVERRIDE)
    (void)n_t;
    return HMX_W4A16_OUT_TABLE_STRIDE_OVERRIDE;
#else
    uint32_t stride = n_t;
#if defined(HMX_W4A16_OUT_TABLE_STRIDE_DWORDS_OVERRIDE)
    if (stride < HMX_W4A16_OUT_TABLE_STRIDE_DWORDS_OVERRIDE) {
        stride = HMX_W4A16_OUT_TABLE_STRIDE_DWORDS_OVERRIDE;
    }
#endif
#if defined(HMX_W4A16_OUT_Y_STRIDE_WORDS_OVERRIDE)
    if (stride < HMX_W4A16_OUT_Y_STRIDE_WORDS_OVERRIDE) {
        stride = HMX_W4A16_OUT_Y_STRIDE_WORDS_OVERRIDE;
    }
#endif
    return stride;
#endif
}

static inline uint32_t hmx_w4a16_act_desc_n_pairs(uint32_t k_t, uint32_t table_storage_stride)
{
#if defined(HMX_W4A16_ACT_N_PAIRS_OVERRIDE)
    (void)k_t;
    (void)table_storage_stride;
    return HMX_W4A16_ACT_N_PAIRS_OVERRIDE;
#else
    (void)k_t;
    return table_storage_stride;
#endif
}

static inline uint32_t hmx_w4a16_act_desc_y_stride_words(
    uint32_t k_t,
    uint32_t table_storage_stride,
    uint32_t desc_m_tiles)
{
#if defined(HMX_W4A16_ACT_TABLE_Y_STRIDE_WORDS_OVERRIDE)
    (void)k_t;
    (void)table_storage_stride;
    (void)desc_m_tiles;
    return HMX_W4A16_ACT_TABLE_Y_STRIDE_WORDS_OVERRIDE;
#else
    (void)k_t;
    (void)table_storage_stride;
    return desc_m_tiles;
#endif
}

static inline uint32_t hmx_w4a16_out_desc_table_stride_dwords(
    uint32_t n_t,
    uint32_t table_storage_stride)
{
#if defined(HMX_W4A16_OUT_TABLE_STRIDE_DWORDS_OVERRIDE)
    (void)n_t;
    (void)table_storage_stride;
    return HMX_W4A16_OUT_TABLE_STRIDE_DWORDS_OVERRIDE;
#else
    (void)n_t;
    return table_storage_stride;
#endif
}

static inline uint32_t hmx_w4a16_out_desc_y_stride_words(
    uint32_t n_t,
    uint32_t table_storage_stride,
    uint32_t desc_m_tiles)
{
#if defined(HMX_W4A16_OUT_Y_STRIDE_WORDS_OVERRIDE)
    (void)n_t;
    (void)table_storage_stride;
    (void)desc_m_tiles;
    return HMX_W4A16_OUT_Y_STRIDE_WORDS_OVERRIDE;
#else
    (void)n_t;
    (void)table_storage_stride;
    return desc_m_tiles;
#endif
}

static inline int32_t hmx_w4a16_desc_m_total_minus_step(uint32_t row4_groups)
{
#if defined(HMX_W4A16_DESC_M_TOTAL_MINUS_STEP_OVERRIDE)
    (void)row4_groups;
    return HMX_W4A16_DESC_M_TOTAL_MINUS_STEP_OVERRIDE;
#else
    (void)row4_groups;
    return 8;
#endif
}

static inline uint32_t hmx_w4a16_desc_k_total_bytes(uint32_t n_t)
{
#if defined(HMX_W4A16_DESC_K_TOTAL_BYTES_OVERRIDE)
    (void)n_t;
    return HMX_W4A16_DESC_K_TOTAL_BYTES_OVERRIDE;
#else
    return n_t * 32u;
#endif
}

static inline uint32_t hmx_w4a16_required_table_entries(
    uint32_t row4_groups,
    uint32_t logical_tiles,
    uint32_t table_storage_stride)
{
    if (row4_groups == 0 || logical_tiles == 0 || table_storage_stride < logical_tiles) {
        return 0;
    }
    return (row4_groups - 1u) * table_storage_stride + logical_tiles;
}

static inline const uint8_t *hmx_w4a16_ptr_with_offset(const uint8_t *ptr, intptr_t offset)
{
    return reinterpret_cast<const uint8_t *>(reinterpret_cast<uintptr_t>(ptr) + offset);
}

static inline uint32_t hmx_w4a16_crouton_row4_groups(uint32_t m_t)
{
    return m_t * 8u;
}

static inline uint32_t hmx_w4a16_crouton_row4_block_index(
    uint32_t row4_tile,
    uint32_t kn_tile,
    uint32_t kn_tiles)
{
    return (row4_tile & 7u) * kn_tiles + kn_tile;
}

static inline int32_t hmx_w4a16_crouton_row4_ptr(
    const int32_t *block_table,
    uint32_t row4_tile,
    uint32_t kn_tile,
    uint32_t kn_tiles)
{
    const uint32_t block_index =
        hmx_w4a16_crouton_row4_block_index(row4_tile, kn_tile, kn_tiles);
    const uintptr_t base = static_cast<uintptr_t>(static_cast<uint32_t>(block_table[block_index]));
    const uintptr_t offset_bytes = static_cast<uintptr_t>(row4_tile >> 3) * 256u;
    return static_cast<int32_t>(base + offset_bytes);
}

static inline int32_t hmx_w4a16_crouton_row4_physical_ptr(
    const int32_t *block_table,
    uint32_t row4_tile,
    uint32_t kn_tile,
    uint32_t kn_tiles)
{
    return block_table[hmx_w4a16_crouton_row4_block_index(row4_tile, kn_tile, kn_tiles)];
}

static inline int32_t hmx_w4a16_crouton_logical_or_compact_ptr(
    const int32_t *block_table,
    uint32_t block_entries,
    uint32_t row4_groups,
    uint32_t row4_tile,
    uint32_t kn_tile,
    uint32_t kn_tiles)
{
    if (block_entries >= row4_groups * kn_tiles) {
        return block_table[row4_tile * kn_tiles + kn_tile];
    }
#if defined(HMX_W4A16_ROW4_BLOCK_ORDER_MOD8)
    return hmx_w4a16_crouton_row4_ptr(block_table, row4_tile, kn_tile, kn_tiles);
#else
    const uint32_t compact_m_groups = block_entries / kn_tiles;
    if (compact_m_groups == 0) return 0;
    uint32_t row4_per_block = row4_groups / compact_m_groups;
    if (row4_per_block == 0) row4_per_block = 1;
    const uint32_t block_index = (row4_tile / row4_per_block) * kn_tiles + kn_tile;
    const uintptr_t base = static_cast<uintptr_t>(static_cast<uint32_t>(block_table[block_index]));
    const uintptr_t offset_bytes = static_cast<uintptr_t>(row4_tile % row4_per_block) * 256u;
    return static_cast<int32_t>(base + offset_bytes);
#endif
}

#if defined(__hexagon__)
static inline void store_le32(uint8_t *dst, uint32_t offset, uint32_t value)
{
    dst[offset + 0] = (uint8_t)(value & 0xffu);
    dst[offset + 1] = (uint8_t)((value >> 8) & 0xffu);
    dst[offset + 2] = (uint8_t)((value >> 16) & 0xffu);
    dst[offset + 3] = (uint8_t)((value >> 24) & 0xffu);
}

static inline uint32_t load_le32(const uint8_t *src, uint32_t offset)
{
    return (uint32_t)src[offset + 0] |
           ((uint32_t)src[offset + 1] << 8) |
           ((uint32_t)src[offset + 2] << 16) |
           ((uint32_t)src[offset + 3] << 24);
}

static uint32_t g_hmx_w4a16_mask_buf[16] __attribute__((aligned(16)));

static void init_mask_desc(uint32_t *mask_buf, uint32_t k_total)
{
    for (uint32_t i = 0; i < 16; ++i) mask_buf[i] = 0;
    set_hmx_params_convw4b1x1(
        mask_buf,
        HMX_W4A16_MASK_ARG1,
#if defined(HMX_W4A16_MASK_ARG2)
        HMX_W4A16_MASK_ARG2,
#else
        k_total,
#endif
        HMX_W4A16_MASK_ARG3,
        HMX_W4A16_MASK_ARG4,
        HMX_W4A16_MASK_ARG5,
        HMX_W4A16_MASK_ARG6);
}

static const hmx_conv_mask_desc_t *get_mask_desc(uint32_t k_total)
{
    /*
     * The mask is shape-independent for the current production path.  Native
     * QNN sets bit 5 in the conv1x1 mask tuple and then the wrapper tail-calls
     * the V73DEEP body.  For that direct deep entry, r5 already carries the
     * extra_param pointer, so the old mask[0x38] patch is not needed.
     */
    init_mask_desc(g_hmx_w4a16_mask_buf, k_total);
    return reinterpret_cast<const hmx_conv_mask_desc_t *>(g_hmx_w4a16_mask_buf);
}

static const hmx_conv_mask_desc_t *get_precomputed_mask_desc()
{
    return reinterpret_cast<const hmx_conv_mask_desc_t *>(g_hmx_w4a16_mask_buf);
}

#endif

#if defined(HMX_W4A16_MAX_TABLE_ENTRIES)
static constexpr uint32_t kHmxW4A16MaxRuntimeTableEntries = HMX_W4A16_MAX_TABLE_ENTRIES;
#elif defined(HMX_W4A16_ACT_TABLE_STRIDE_OVERRIDE) || \
    defined(HMX_W4A16_OUT_TABLE_STRIDE_OVERRIDE) || \
    defined(HMX_W4A16_ACT_TABLE_Y_STRIDE_WORDS_OVERRIDE) || \
    defined(HMX_W4A16_OUT_TABLE_STRIDE_DWORDS_OVERRIDE) || \
    defined(HMX_W4A16_OUT_Y_STRIDE_WORDS_OVERRIDE)
static constexpr uint32_t kHmxW4A16MaxRuntimeTableEntries = 4096;
#else
static constexpr uint32_t kHmxW4A16MaxRuntimeTableEntries = 1024;
#endif

#if defined(__hexagon__) && !defined(SCALAR_ONLY)
static inline void hmx_w4a16_enter_kernel(
    const hmx_conv_out_desc_t *out_desc,
    const hmx_conv_act_desc_t *act_desc,
    const uint8_t *wt_pack,
    const uint8_t *bias_bytes,
    const hmx_conv_mask_desc_t *mask_desc,
    const uint32_t *extra_param)
{
#if defined(HMX_W4A16_USE_SKEL_KERNEL)
    hmx_v73_convhnh1x1_stride1(
#else
    our_v73deep_kernel(
#endif
        out_desc,
        act_desc,
        wt_pack,
        bias_bytes,
        mask_desc,
        extra_param);
}
#endif

#if defined(HMX_W4A16_ENABLE_QHPI_PRECOMPUTE)
#if defined(HMX_W4A16_MAX_COPIED_TABLE_ENTRIES)
static constexpr uint32_t kHmxW4A16MaxCopiedTableEntries =
    HMX_W4A16_MAX_COPIED_TABLE_ENTRIES;
#elif defined(HMX_W4A16_ACT_TABLE_STRIDE_OVERRIDE) || \
    defined(HMX_W4A16_OUT_TABLE_STRIDE_OVERRIDE) || \
    defined(HMX_W4A16_ACT_TABLE_Y_STRIDE_WORDS_OVERRIDE) || \
    defined(HMX_W4A16_OUT_TABLE_STRIDE_DWORDS_OVERRIDE) || \
    defined(HMX_W4A16_OUT_Y_STRIDE_WORDS_OVERRIDE)
static constexpr uint32_t kHmxW4A16MaxCopiedTableEntries = 4096;
#else
static constexpr uint32_t kHmxW4A16MaxCopiedTableEntries = 512;
#endif
static constexpr uint32_t kHmxW4A16PrecomputedDataSize =
    56 + 2 * kHmxW4A16MaxCopiedTableEntries * sizeof(int32_t);
#endif

#if defined(__hexagon__) && !defined(SCALAR_ONLY) && defined(HMX_W4A16_ENABLE_QHPI_PRECOMPUTE)
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
 *   hmx_w4a16_precomputed_t
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
static constexpr uint32_t kHmxW4A16PrecomputeMagic = 0x48385850u; /* H8XP */

struct hmx_w4a16_precomputed_t {
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
    int32_t act_table_copy[kHmxW4A16MaxCopiedTableEntries];
    int32_t out_table_copy[kHmxW4A16MaxCopiedTableEntries];
};

static_assert(sizeof(hmx_w4a16_precomputed_t) == kHmxW4A16PrecomputedDataSize,
              "QHPI precompute ABI size must match the x86 registration stub");

static uint32_t hmx_w4a16_precompute(
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

    hmx_w4a16_precomputed_t *pc = reinterpret_cast<hmx_w4a16_precomputed_t *>(data);
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

    uint32_t M_t = 0;
    uint32_t N_t = 0;
    uint32_t K_t = 0;
    if (!hmx_w4a16_tile_counts_from_shapes(
            inputs[2],
            inputs[1],
            outputs[0],
            act_block_entries,
            &M_t,
            &N_t,
            &K_t)) {
        return QHPI_Success;
    }
    init_mask_desc(g_hmx_w4a16_mask_buf, K_t * 32u);

    const uint32_t mt_per_block = 2;
#if HMX_W4A16_USE_ROW4_TABLES
    const uint32_t mt_groups = hmx_w4a16_crouton_row4_groups(M_t);
    const uint32_t act_table_stride = hmx_w4a16_act_table_storage_stride(K_t);
    const uint32_t out_table_stride = hmx_w4a16_out_table_storage_stride(N_t);
    const uint32_t act_entries =
        hmx_w4a16_required_table_entries(mt_groups, K_t, act_table_stride);
    const uint32_t out_entries =
        hmx_w4a16_required_table_entries(mt_groups, N_t, out_table_stride);
#else
    const uint32_t mt_groups = M_t >> 1;
    const uint32_t act_table_stride = K_t;
    const uint32_t out_table_stride = N_t;
    const uint32_t act_entries = mt_groups * K_t;
    const uint32_t out_entries = mt_groups * N_t;
#endif
    if (mt_groups == 0 ||
        act_entries == 0 ||
        out_entries == 0 ||
        act_entries > kHmxW4A16MaxCopiedTableEntries ||
        out_entries > kHmxW4A16MaxCopiedTableEntries ||
        act_block_entries < (M_t >> 1) * K_t ||
        out_block_entries < (M_t >> 1) * N_t) {
        return QHPI_Success;
    }

    const int32_t *act_src = reinterpret_cast<const int32_t *>(act_blocks);
    const int32_t *out_src = reinterpret_cast<const int32_t *>(out_blocks);

    pc->bias_bytes = bias_bytes;
    pc->wt_pack = wt_pack;
    pc->out_first_block = reinterpret_cast<uint8_t *>(out_blocks[0]);
    pc->act_qhpi_table = act_src;
    pc->out_qhpi_table = out_src;
#if HMX_W4A16_USE_ROW4_TABLES
    for (uint32_t rg = 0; rg < mt_groups; ++rg) {
        for (uint32_t kt = 0; kt < K_t; ++kt) {
            pc->act_table_copy[rg * act_table_stride + kt] =
#if defined(HMX_W4A16_ACT_PHYSICAL_ONLY)
                hmx_w4a16_crouton_row4_physical_ptr(act_src, rg, kt, K_t);
#else
                hmx_w4a16_crouton_logical_or_compact_ptr(
                    act_src, act_block_entries, mt_groups, rg, kt, K_t);
#endif
        }
        for (uint32_t nt = 0; nt < N_t; ++nt) {
            pc->out_table_copy[rg * out_table_stride + nt] =
#if defined(HMX_W4A16_OUT_PHYSICAL_ONLY)
                hmx_w4a16_crouton_row4_physical_ptr(out_src, rg, nt, N_t);
#else
                hmx_w4a16_crouton_logical_or_compact_ptr(
                    out_src, out_block_entries, mt_groups, rg, nt, N_t);
#endif
        }
    }
#else
    for (uint32_t i = 0; i < act_entries; ++i) pc->act_table_copy[i] = act_src[i];
    for (uint32_t i = 0; i < out_entries; ++i) pc->out_table_copy[i] = out_src[i];
#endif
    pc->act_qhpi_table = pc->act_table_copy;
    pc->out_qhpi_table = pc->out_table_copy;
    pc->S = M_t * 32u;
    pc->M_t = M_t;
    pc->N_t = N_t;
    pc->K_t = K_t;
    pc->mt_per_block = mt_per_block;
    pc->mt_groups = mt_groups;
    pc->act_entries = act_entries;
    pc->out_entries = out_entries;
    pc->magic = kHmxW4A16PrecomputeMagic;
    return QHPI_Success;
}

static uint32_t hmx_w4a16_to_u16_matmul_precomputed_kernel(
    QHPI_RuntimeHandle *handle,
    const void *precomputed_data)
{
    (void)handle;

    const hmx_w4a16_precomputed_t *pc =
        reinterpret_cast<const hmx_w4a16_precomputed_t *>(precomputed_data);
    if (!pc || pc->magic != kHmxW4A16PrecomputeMagic) return QHPI_Success;

#if defined(HMX_W4A16_PROBE_CYCLES)
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
     * hmx_w4a16_precomputed_t at graph load.  That removes the last hot-path
     * QNN table locality miss without spending extra dcfetch packets in the
     * profiled callback.  Larger fallback shapes keep pointing at QNN-owned
     * block tables.
     */
    int32_t *act_tbl_ptr = const_cast<int32_t *>(pc->act_qhpi_table);
    int32_t *out_tbl_ptr = const_cast<int32_t *>(pc->out_qhpi_table);

    uint32_t extra_param[3] __attribute__((aligned(16))) = {
        HMX_W4A16_EXTRA_PARAM0,
        HMX_W4A16_EXTRA_PARAM1,
        HMX_W4A16_EXTRA_PARAM2,
    };
    const hmx_conv_mask_desc_t *mask_desc = get_precomputed_mask_desc();
    const uint32_t desc_m_tiles =
#if HMX_W4A16_USE_ROW4_TABLES
        hmx_w4a16_desc_m_tiles(pc->M_t, pc->mt_groups);
#else
        pc->M_t * 4u;
#endif
    const uint32_t act_table_stride = hmx_w4a16_act_table_storage_stride(pc->K_t);
    const uint32_t out_table_stride = hmx_w4a16_out_table_storage_stride(pc->N_t);

    hmx_conv_out_desc_t out_desc_local __attribute__((aligned(64))) = {
        out_tbl_ptr,
        hmx_w4a16_out_desc_table_stride_dwords(pc->N_t, out_table_stride),
        hmx_w4a16_out_desc_y_stride_words(pc->N_t, out_table_stride, desc_m_tiles),
        desc_m_tiles,
        hmx_w4a16_desc_m_total_minus_step(pc->mt_groups),
        hmx_w4a16_desc_k_total_bytes(pc->N_t),
    };
    hmx_conv_act_desc_t act_desc_local __attribute__((aligned(64))) = {
        act_tbl_ptr,
        hmx_w4a16_act_desc_n_pairs(pc->K_t, act_table_stride),
        hmx_w4a16_act_desc_y_stride_words(pc->K_t, act_table_stride, desc_m_tiles),
    };
    const hmx_conv_out_desc_t *out_desc = &out_desc_local;
    const hmx_conv_act_desc_t *act_desc = &act_desc_local;
    const uint8_t *effective_wt_pack =
        hmx_w4a16_ptr_with_offset(pc->wt_pack, HMX_W4A16_WEIGHT_PTR_OFFSET);
    const uint8_t *effective_bias_bytes =
        hmx_w4a16_ptr_with_offset(pc->bias_bytes, HMX_W4A16_BIAS_PTR_OFFSET);

#if defined(HMX_W4A16_DESC_DUMP)
    if (pc->out_first_block) {
        uint8_t *dst = pc->out_first_block;
        for (uint32_t i = 0; i < 192; ++i) dst[i] = 0;
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
        store_le32(dst, 60, pc->mt_groups);
        store_le32(dst, 64, act_table_stride);
        store_le32(dst, 68, out_table_stride);
        store_le32(dst, 72, pc->act_entries);
        store_le32(dst, 76, pc->out_entries);
        store_le32(dst, 80, 0); /* source QHPI block-table length is not kept here */
        store_le32(dst, 84, 0);
        store_le32(dst, 88, reinterpret_cast<const uint32_t *>(act_tbl_ptr)[0]);
        store_le32(dst, 92, reinterpret_cast<const uint32_t *>(act_tbl_ptr)[1]);
        store_le32(dst, 96, reinterpret_cast<const uint32_t *>(out_tbl_ptr)[0]);
        store_le32(dst, 100, reinterpret_cast<const uint32_t *>(out_tbl_ptr)[1]);
        store_le32(dst, 104, load_le32(effective_wt_pack, 0));
        store_le32(dst, 108, load_le32(effective_wt_pack, 4));
        store_le32(dst, 112, load_le32(effective_bias_bytes, 0));
        store_le32(dst, 116, load_le32(effective_bias_bytes, 4));
        store_le32(dst, 120, extra_param[0]);
        store_le32(dst, 124, extra_param[1]);
        const uint32_t *mask_words = reinterpret_cast<const uint32_t *>(mask_desc);
        for (uint32_t i = 0; i < 16; ++i) store_le32(dst, 128 + i * 4, mask_words[i]);
    }
    return QHPI_Success;
#endif

#if defined(HMX_W4A16_SKIP_KERNEL)
    if (pc->out_first_block) {
        uint8_t *dst = pc->out_first_block;
        for (uint32_t i = 0; i < 16; ++i) dst[i] = 0;
        store_le32(dst, 0, 0x48385853u); /* H8XS */
        store_le32(dst, 4, pc->S);
    }
    return QHPI_Success;
#endif

#if defined(HMX_W4A16_PROBE_CYCLES)
    uint64_t cyc_before_kernel = 0;
    asm volatile("%0 = C15:14" : "=r"(cyc_before_kernel));
#endif

    /*
     * Precomputed hot path: QHPI tensor lookup, shape recovery, and pointer
     * table discovery happened at graph load.  The custom op event now starts
     * at the tiny native descriptor stitching boundary.
     */
    hmx_w4a16_enter_kernel(
        out_desc,
        act_desc,
        effective_wt_pack,
        effective_bias_bytes,
        mask_desc,
        extra_param);

#if defined(HMX_W4A16_PROBE_CYCLES)
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
#elif defined(HMX_W4A16_ENABLE_QHPI_PRECOMPUTE)
/*
 * The x86 context generator must see the same QHPI kernel registration as the
 * device package.  It does not execute the HMX body, so these are registration
 * stubs only; the real precompute/execute implementation is compiled into the
 * Hexagon HTP package above.
 */
static uint32_t hmx_w4a16_precompute(
    QHPI_RuntimeHandle *,
    void *,
    uint32_t,
    QHPI_Tensor **,
    uint32_t,
    const QHPI_Tensor *const *)
{
    return QHPI_Success;
}

static uint32_t hmx_w4a16_to_u16_matmul_precomputed_kernel(
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
static uint32_t hmx_w4a16_to_u16_matmul_kernel(
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
#if defined(HMX_W4A16_PROBE_CYCLES)
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

    uint32_t M_t = 0;
    uint32_t N_t = 0;
    uint32_t K_t = 0;
    if (!hmx_w4a16_tile_counts_from_shapes(
            inputs[2],
            inputs[1],
            outputs[0],
            act_block_entries,
            &M_t,
            &N_t,
            &K_t)) {
        return QHPI_Success;
    }

    const uint32_t mt_per_block = 2;
#if HMX_W4A16_USE_ROW4_TABLES
    const uint32_t mt_groups = hmx_w4a16_crouton_row4_groups(M_t);
    const uint32_t act_table_stride = hmx_w4a16_act_table_storage_stride(K_t);
    const uint32_t out_table_stride = hmx_w4a16_out_table_storage_stride(N_t);
    const uint32_t act_entries =
        hmx_w4a16_required_table_entries(mt_groups, K_t, act_table_stride);
    const uint32_t out_entries =
        hmx_w4a16_required_table_entries(mt_groups, N_t, out_table_stride);
#else
    const uint32_t mt_groups = M_t >> 1;
    const uint32_t act_table_stride = K_t;
    const uint32_t out_table_stride = N_t;
    const uint32_t act_entries = mt_groups * K_t;
    const uint32_t out_entries = mt_groups * N_t;
#endif
    if (mt_groups == 0 || act_entries == 0 || out_entries == 0 ||
        act_entries > kHmxW4A16MaxRuntimeTableEntries ||
        out_entries > kHmxW4A16MaxRuntimeTableEntries ||
        act_block_entries < (M_t >> 1) * K_t ||
        out_block_entries < (M_t >> 1) * N_t) {
        return QHPI_Success;
    }

#if defined(HMX_W4A16_PROBE_CYCLES)
    uint64_t cyc_after_qhpi = 0;
    asm volatile("%0 = C15:14" : "=r"(cyc_after_qhpi));
#endif

    int32_t act_tbl_all[kHmxW4A16MaxRuntimeTableEntries] __attribute__((aligned(64)));
    int32_t out_tbl_all[kHmxW4A16MaxRuntimeTableEntries] __attribute__((aligned(64)));
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
#if HMX_W4A16_USE_ROW4_TABLES
    for (uint32_t rg = 0; rg < mt_groups; ++rg) {
        for (uint32_t kt = 0; kt < K_t; ++kt) {
            act_tbl_all[rg * act_table_stride + kt] =
#if defined(HMX_W4A16_ACT_PHYSICAL_ONLY)
                hmx_w4a16_crouton_row4_physical_ptr(act_src, rg, kt, K_t);
#else
                hmx_w4a16_crouton_logical_or_compact_ptr(
                    act_src, act_block_entries, mt_groups, rg, kt, K_t);
#endif
        }
        for (uint32_t nt = 0; nt < N_t; ++nt) {
            out_tbl_all[rg * out_table_stride + nt] =
#if defined(HMX_W4A16_OUT_PHYSICAL_ONLY)
                hmx_w4a16_crouton_row4_physical_ptr(out_src, rg, nt, N_t);
#else
                hmx_w4a16_crouton_logical_or_compact_ptr(
                    out_src, out_block_entries, mt_groups, rg, nt, N_t);
#endif
        }
    }
#else
    for (uint32_t rg = 0; rg < mt_groups; ++rg) {
        const int32_t *__restrict a_src = act_src + rg * K_t;
        const int32_t *__restrict o_src = out_src + rg * N_t;
        int32_t *__restrict a_dst = act_tbl_all + rg * K_t;
        int32_t *__restrict o_dst = out_tbl_all + rg * N_t;
        for (uint32_t kt = 0; kt < K_t; ++kt) a_dst[kt] = a_src[kt];
        for (uint32_t nt = 0; nt < N_t; ++nt) o_dst[nt] = o_src[nt];
    }
#endif

#if defined(HMX_W4A16_PROBE_CYCLES)
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
    uint32_t extra_param[3] __attribute__((aligned(16))) = {
        HMX_W4A16_EXTRA_PARAM0,
        HMX_W4A16_EXTRA_PARAM1,
        HMX_W4A16_EXTRA_PARAM2,
    };
    const hmx_conv_mask_desc_t *mask_desc = get_mask_desc(K_t * 32u);
    const uint32_t desc_m_tiles =
#if HMX_W4A16_USE_ROW4_TABLES
        hmx_w4a16_desc_m_tiles(M_t, mt_groups);
#else
        M_t * 4u;
#endif

    hmx_conv_out_desc_t out_desc = {
        out_tbl_all,
        hmx_w4a16_out_desc_table_stride_dwords(N_t, out_table_stride),
        hmx_w4a16_out_desc_y_stride_words(N_t, out_table_stride, desc_m_tiles),
        desc_m_tiles,
        hmx_w4a16_desc_m_total_minus_step(mt_groups),
        hmx_w4a16_desc_k_total_bytes(N_t),
    };
    hmx_conv_act_desc_t act_desc = {
        act_tbl_all,
        hmx_w4a16_act_desc_n_pairs(K_t, act_table_stride),
        hmx_w4a16_act_desc_y_stride_words(K_t, act_table_stride, desc_m_tiles),
    };
    const uint8_t *effective_wt_pack =
        hmx_w4a16_ptr_with_offset(wt_pack, HMX_W4A16_WEIGHT_PTR_OFFSET);
    const uint8_t *effective_bias_bytes =
        hmx_w4a16_ptr_with_offset(bias_bytes, HMX_W4A16_BIAS_PTR_OFFSET);

#if defined(HMX_W4A16_DESC_DUMP)
    /*
     * Descriptor dump mode writes the derived ABI state into the first output
     * block and returns before HMX compute.  Use this when validating that QNN's
     * tensor/block metadata was translated into the expected native descriptor.
     */
    if (out_blocks[0]) {
        uint8_t *dst = reinterpret_cast<uint8_t *>(out_blocks[0]);
        for (uint32_t i = 0; i < 192; ++i) dst[i] = 0;
        store_le32(dst, 0, 0x48385844u); /* H8XD */
        store_le32(dst, 4, M_t * 32u);
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
        store_le32(dst, 60, mt_groups);
        store_le32(dst, 64, act_table_stride);
        store_le32(dst, 68, out_table_stride);
        store_le32(dst, 72, act_entries);
        store_le32(dst, 76, out_entries);
        store_le32(dst, 80, act_block_entries);
        store_le32(dst, 84, out_block_entries);
        store_le32(dst, 88, reinterpret_cast<const uint32_t *>(act_tbl_all)[0]);
        store_le32(dst, 92, reinterpret_cast<const uint32_t *>(act_tbl_all)[1]);
        store_le32(dst, 96, reinterpret_cast<const uint32_t *>(out_tbl_all)[0]);
        store_le32(dst, 100, reinterpret_cast<const uint32_t *>(out_tbl_all)[1]);
        store_le32(dst, 104, load_le32(effective_wt_pack, 0));
        store_le32(dst, 108, load_le32(effective_wt_pack, 4));
        store_le32(dst, 112, load_le32(effective_bias_bytes, 0));
        store_le32(dst, 116, load_le32(effective_bias_bytes, 4));
        store_le32(dst, 120, extra_param[0]);
        store_le32(dst, 124, extra_param[1]);
        const uint32_t *mask_words = reinterpret_cast<const uint32_t *>(mask_desc);
        for (uint32_t i = 0; i < 16; ++i) store_le32(dst, 128 + i * 4, mask_words[i]);
    }
    return QHPI_Success;
#endif

#if defined(HMX_W4A16_SKIP_KERNEL)
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
        store_le32(dst, 4, M_t * 32u);
    }
    return QHPI_Success;
#endif

#if defined(HMX_W4A16_PROBE_CYCLES)
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
    hmx_w4a16_enter_kernel(
        &out_desc,
        &act_desc,
        effective_wt_pack,
        effective_bias_bytes,
        mask_desc,
        extra_param);

#if defined(HMX_W4A16_PROBE_CYCLES)
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
    {QHPI_Int32,            QHPI_Layout_Flat4,      QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},
#if defined(HMX_W4A16_QHPI_SIGNED_WEIGHT)
    {QHPI_QInt8,            QHPI_Layout_Flat4,      QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},
#else
    {QHPI_Any_Element_Type, QHPI_Layout_Flat4,      QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},
#endif
    {QHPI_QUInt16,          QHPI_Layout_Crouton_16, QHPI_Storage_Indirect, QHPI_MemLoc_TCM_Only},
    {QHPI_Int32,            QHPI_Layout_Flat4,      QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},
};

static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QUInt16,          QHPI_Layout_Crouton_16, QHPI_Storage_Indirect, QHPI_MemLoc_TCM_Only},
};

static float hmx_w4a16_cost_function(uint32_t num_inputs, const QHPI_Tensor *const *inputs)
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

static QHPI_Shape hmx_w4a16_shape_required(const QHPI_Op *op)
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

static QHPI_Shape hmx_w4a16_shape_legalized(const QHPI_Op *op, const QHPI_Shape *proposed)
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
#if defined(HMX_W4A16_ENABLE_QHPI_PRECOMPUTE)
        THIS_PKG_NAME_STR "::hmx_w4a16_to_u16_matmul_precomputed_kernel",
        nullptr,
#else
        THIS_PKG_NAME_STR "::hmx_w4a16_to_u16_matmul_kernel",
        hmx_w4a16_to_u16_matmul_kernel,
#endif
        QHPI_RESOURCE_HMX,
        false,
        false,
        false, false,
        4, sig_inputs,
        1, sig_outputs,
        hmx_w4a16_cost_function,
        0,
#if defined(HMX_W4A16_ENABLE_QHPI_PRECOMPUTE)
        kHmxW4A16PrecomputedDataSize,
        hmx_w4a16_precompute,
        hmx_w4a16_to_u16_matmul_precomputed_kernel,
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
        THIS_PKG_NAME_STR "::HmxU16I4ToU16MatMul",
        1, sg_kernels,
        nullptr,
        hmx_w4a16_shape_required,
        hmx_w4a16_shape_legalized,
        0,
        nullptr,
        nullptr,
    },
};

extern "C" void register_hmx_w4a16_to_u16_matmul_op(void)
{
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
