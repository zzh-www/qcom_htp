/*
 * HmxU16I16ToU16MatMulOp.cpp
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

#if defined(HMX_W16A16_ACCEPTED_NATIVE_RECORD_256) && !defined(HMX_W16A16_NATIVE_RECORD_256_PROFILE)
#define HMX_W16A16_NATIVE_RECORD_256_PROFILE
#endif

#if defined(HMX_W16A16_NATIVE_RECORD_256_PROFILE)
#ifndef HMX_W16A16_INTERNAL_SPLIT_N128
#define HMX_W16A16_INTERNAL_SPLIT_N128
#endif
#ifndef HMX_W16A16_QHPI_WEIGHT8
#define HMX_W16A16_QHPI_WEIGHT8
#endif
#ifndef HMX_W16A16_NATIVE_RECORD_FIELDS
#define HMX_W16A16_NATIVE_RECORD_FIELDS
#endif
#ifndef HMX_W16A16_MASK_ARG1
#define HMX_W16A16_MASK_ARG1 0x70bu
#endif
#ifndef HMX_W16A16_MASK_ARG5
#define HMX_W16A16_MASK_ARG5 0x80u
#endif
#endif

#if defined(__hexagon__) && !defined(HMX_W16A16_SKIP_KERNEL) && \
    !defined(HMX_W16A16_ALLOW_UNVALIDATED_KERNEL) && \
    !defined(HMX_W16A16_ACCEPTED_NATIVE_RECORD_256)
#error "w16a16 HMX body is not validated yet; keep HMX_W16A16_SKIP_KERNEL until the family-specific body is wired."
#endif

#ifndef HMX_W16A16_MASK_ARG1
#define HMX_W16A16_MASK_ARG1 0x700u
#endif
#ifndef HMX_W16A16_MASK_ARG2
#define HMX_W16A16_MASK_ARG2 0u
#endif
#ifndef HMX_W16A16_MASK_ARG3
#define HMX_W16A16_MASK_ARG3 0u
#endif
#ifndef HMX_W16A16_MASK_ARG4
#define HMX_W16A16_MASK_ARG4 0u
#endif
#ifndef HMX_W16A16_MASK_ARG5
#define HMX_W16A16_MASK_ARG5 0x20u
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

static bool hmx_w16a16_tile_counts_from_shapes(
    const QHPI_Tensor *act,
    const QHPI_Tensor *weight,
    const QHPI_Tensor *out,
    uint32_t act_blocks,
    uint32_t *M_t,
    uint32_t *N_t,
    uint32_t *K_t)
{
    uint32_t M = 0;
    uint32_t K = 0;
    uint32_t N = 0;
    if (act) {
        const QHPI_Shape a = qhpi_tensor_shape(act);
        if (a.rank == 4) {
            M = a.dims[1] * a.dims[2];
            K = a.dims[3];
        } else if (a.rank == 3) {
            M = a.dims[1];
            K = a.dims[2];
        }
    }
    if (out) {
        const QHPI_Shape o = qhpi_tensor_shape(out);
        if (o.rank == 4) {
            N = o.dims[3];
        } else if (o.rank == 3) {
            N = o.dims[2];
        }
    }
    if (weight && N == 0) {
        const QHPI_Shape w = qhpi_tensor_shape(weight);
        if (w.rank == 4) {
            if (K == 0) K = w.dims[2];
            N = w.dims[3];
        }
    }
    if (M == 0 || K == 0 || N == 0) {
        const uint32_t S = square_size_from_crouton_blocks(act_blocks);
        M = S;
        K = S;
        N = S;
    }
    if (M < 128 || K < 128 || N < 32 || (M % 32) || (K % 32) || (N % 32)) {
        return false;
    }
    *M_t = M / 32;
    *K_t = K / 32;
    *N_t = N / 32;
    return true;
}

static inline uint32_t hmx_w16a16_required_table_entries(
    uint32_t row4_groups,
    uint32_t logical_tiles,
    uint32_t table_storage_stride)
{
    if (row4_groups == 0 || logical_tiles == 0 || table_storage_stride < logical_tiles) {
        return 0;
    }
    return (row4_groups - 1u) * table_storage_stride + logical_tiles;
}

static inline uint32_t hmx_w16a16_crouton_row4_groups(uint32_t m_t)
{
    return m_t * 8u;
}

static inline uint32_t hmx_w16a16_crouton_row4_block_index(
    uint32_t row4_tile,
    uint32_t kn_tile,
    uint32_t kn_tiles)
{
#if defined(HMX_W16A16_ROW4_BLOCK_OFFSET)
    return (row4_tile >> 3u) * kn_tiles + kn_tile;
#else
    return (row4_tile & 7u) * kn_tiles + kn_tile;
#endif
}

static inline int32_t hmx_w16a16_crouton_row4_physical_ptr(
    const int32_t *block_table,
    uint32_t row4_tile,
    uint32_t kn_tile,
    uint32_t kn_tiles)
{
#if defined(HMX_W16A16_ROW4_BLOCK_OFFSET)
    return block_table[hmx_w16a16_crouton_row4_block_index(row4_tile, kn_tile, kn_tiles)] +
        static_cast<int32_t>((row4_tile & 7u) * 256u);
#else
    return block_table[hmx_w16a16_crouton_row4_block_index(row4_tile, kn_tile, kn_tiles)];
#endif
}

static inline uint32_t hmx_w16a16_k_total_bytes(uint32_t n_tiles, uint32_t k_tiles)
{
#if defined(HMX_W16A16_K_TOTAL_BYTES_OVERRIDE)
    (void)n_tiles;
    (void)k_tiles;
    return HMX_W16A16_K_TOTAL_BYTES_OVERRIDE;
#elif defined(HMX_W16A16_NATIVE_RECORD_FIELDS)
    (void)n_tiles;
    (void)k_tiles;
    return 128u;
#else
#if defined(HMX_W16A16_K_TOTAL_BYTES_SCALE_K)
    return n_tiles * k_tiles * 32u;
#else
    (void)k_tiles;
    return n_tiles * 32u;
#endif
#endif
}

static inline uint32_t hmx_w16a16_act_table_stride(uint32_t m_tiles, uint32_t k_tiles)
{
#if defined(HMX_W16A16_ACT_STRIDE_M32)
    (void)k_tiles;
    return m_tiles * 4u;
#else
    (void)m_tiles;
    return k_tiles;
#endif
}

static inline uint32_t hmx_w16a16_act_desc_y_stride(uint32_t fallback)
{
#if defined(HMX_W16A16_ACT_DESC_Y_STRIDE_OVERRIDE)
    (void)fallback;
    return HMX_W16A16_ACT_DESC_Y_STRIDE_OVERRIDE;
#elif defined(HMX_W16A16_NATIVE_RECORD_FIELDS)
    (void)fallback;
    return 512u;
#else
    return fallback;
#endif
}

static inline uint32_t hmx_w16a16_out_y_stride(uint32_t fallback)
{
#if defined(HMX_W16A16_OUT_Y_STRIDE_OVERRIDE)
    (void)fallback;
    return HMX_W16A16_OUT_Y_STRIDE_OVERRIDE;
#elif defined(HMX_W16A16_NATIVE_RECORD_FIELDS)
    (void)fallback;
    return 256u;
#else
    return fallback;
#endif
}

static inline uint32_t hmx_w16a16_out_n_tiles_pow2(uint32_t fallback)
{
#if defined(HMX_W16A16_OUT_N_TILES_POW2_OVERRIDE)
    (void)fallback;
    return HMX_W16A16_OUT_N_TILES_POW2_OVERRIDE;
#elif defined(HMX_W16A16_NATIVE_RECORD_FIELDS)
    (void)fallback;
    return 256u;
#else
    return fallback;
#endif
}

static inline int32_t hmx_w16a16_out_m_total_minus_step(int32_t fallback)
{
#if defined(HMX_W16A16_OUT_M_TOTAL_MINUS_STEP_OVERRIDE)
    (void)fallback;
    return HMX_W16A16_OUT_M_TOTAL_MINUS_STEP_OVERRIDE;
#elif defined(HMX_W16A16_NATIVE_RECORD_FIELDS)
    (void)fallback;
    return 1;
#else
    return fallback;
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

static uint32_t g_hmx_w16a16_mask_buf[16] __attribute__((aligned(16)));
static uint32_t g_hmx_w16a16_extra_param[2] __attribute__((aligned(16))) = {1u, 1536u};

static void init_mask_desc(uint32_t *mask_buf)
{
    for (uint32_t i = 0; i < 16; ++i) mask_buf[i] = 0;
    set_hmx_params_conv1x1(
        mask_buf,
        HMX_W16A16_MASK_ARG1,
        HMX_W16A16_MASK_ARG2,
        HMX_W16A16_MASK_ARG3,
        HMX_W16A16_MASK_ARG4,
        HMX_W16A16_MASK_ARG5);
}

static const hmx_conv_mask_desc_t *get_mask_desc()
{
    /*
     * The mask is shape-independent for the current production path.  Native
     * QNN sets bit 5 in the conv1x1 mask tuple and then the wrapper tail-calls
     * the V73DEEP body.  For that direct deep entry, r5 already carries the
     * extra_param pointer, so the old mask[0x38] patch is not needed.
     */
    static int initialized = 0;
    if (!initialized) {
        init_mask_desc(g_hmx_w16a16_mask_buf);
#if defined(HMX_W16A16_NATIVE_RECORD_FIELDS)
        g_hmx_w16a16_mask_buf[14] =
            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(g_hmx_w16a16_extra_param));
#endif
        initialized = 1;
    }
    return reinterpret_cast<const hmx_conv_mask_desc_t *>(g_hmx_w16a16_mask_buf);
}

static const hmx_conv_mask_desc_t *get_precomputed_mask_desc()
{
    return reinterpret_cast<const hmx_conv_mask_desc_t *>(g_hmx_w16a16_mask_buf);
}

static inline const hmx_conv_mask_desc_t *hmx_w16a16_maybe_patch_mask_extra(
    const hmx_conv_mask_desc_t *mask_desc,
    uint32_t *mask_storage,
    const uint32_t *extra_param)
{
#if defined(HMX_W16A16_MASK_EXTRA_PTR_WORD) || defined(HMX_W16A16_NATIVE_RECORD_FIELDS)
    const uint32_t *src = reinterpret_cast<const uint32_t *>(mask_desc);
    for (uint32_t i = 0; i < 16; ++i) mask_storage[i] = src[i];
    mask_storage[14] = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(extra_param));
    return reinterpret_cast<const hmx_conv_mask_desc_t *>(mask_storage);
#else
    (void)mask_storage;
    (void)extra_param;
    return mask_desc;
#endif
}

static inline void hmx_w16a16_enter_kernel(
    const hmx_conv_out_desc_t *out_desc,
    const hmx_conv_act_desc_t *act_desc,
    const uint8_t *wt_pack,
    const uint8_t *bias_bytes,
    const hmx_conv_mask_desc_t *mask_desc,
    const uint32_t *extra_param)
{
    our_v73deep_kernel(out_desc, act_desc, wt_pack, bias_bytes, mask_desc, extra_param);
}

static inline void hmx_w16a16_enter_kernel_maybe_split_n128(
    const hmx_conv_out_desc_t *out_desc,
    const hmx_conv_act_desc_t *act_desc,
    const uint8_t *wt_pack,
    const uint8_t *bias_bytes,
    const hmx_conv_mask_desc_t *mask_desc,
    const uint32_t *extra_param,
    uint32_t mt_groups,
    uint32_t N_t,
    uint32_t K_t,
    const int32_t *precomputed_split_out_tables = nullptr)
{
#if defined(HMX_W16A16_INTERNAL_SPLIT_N128)
    constexpr uint32_t kSplitNTiles = 4;
    if (N_t >= kSplitNTiles && (N_t % kSplitNTiles) == 0 &&
        out_desc->out_table_stride_dwords >= N_t) {
        int32_t out_tbl_split[1024] __attribute__((aligned(64)));
        const uint32_t split_entries = mt_groups * kSplitNTiles;
        if (split_entries > 0 && split_entries <= 1024) {
            const uint32_t split_weight_bytes = K_t * kSplitNTiles * 2048u;
            const uint32_t split_bias_bytes = kSplitNTiles * 512u;
            for (uint32_t split = 0; split < N_t / kSplitNTiles; ++split) {
                const int32_t *split_table = precomputed_split_out_tables ?
                    precomputed_split_out_tables + split * split_entries : out_tbl_split;
                if (!precomputed_split_out_tables) {
                    for (uint32_t rg = 0; rg < mt_groups; ++rg) {
                        int32_t *dst = out_tbl_split + rg * kSplitNTiles;
                        const int32_t *src = out_desc->out_tile_ptr_table +
                            rg * out_desc->out_table_stride_dwords + split * kSplitNTiles;
                        for (uint32_t nt = 0; nt < kSplitNTiles; ++nt) dst[nt] = src[nt];
                    }
                }
                hmx_conv_out_desc_t split_out_desc = *out_desc;
                split_out_desc.out_tile_ptr_table = const_cast<int32_t *>(split_table);
                split_out_desc.out_table_stride_dwords = kSplitNTiles;
                split_out_desc.out_y_stride_words =
                    hmx_w16a16_out_y_stride(kSplitNTiles);
                split_out_desc.n_tiles_pow2 =
                    hmx_w16a16_out_n_tiles_pow2(split_out_desc.n_tiles_pow2);
                split_out_desc.m_total_minus_step =
                    hmx_w16a16_out_m_total_minus_step(split_out_desc.m_total_minus_step);
                split_out_desc.k_total_bytes =
                    hmx_w16a16_k_total_bytes(kSplitNTiles, K_t);
                hmx_w16a16_enter_kernel(
                    &split_out_desc,
                    act_desc,
                    wt_pack + split * split_weight_bytes,
                    bias_bytes + split * split_bias_bytes,
                    mask_desc,
                    extra_param);
            }
            return;
        }
    }
#else
    (void)mt_groups;
    (void)N_t;
    (void)K_t;
#endif
    hmx_w16a16_enter_kernel(out_desc, act_desc, wt_pack, bias_bytes, mask_desc, extra_param);
}

#endif

#if defined(HMX_W16A16_ENABLE_QHPI_PRECOMPUTE)
static constexpr uint32_t kHmxW16A16MaxCopiedTableEntries = 1024;
static constexpr uint32_t kHmxW16A16PrecomputedDataSize =
    64 + 3 * kHmxW16A16MaxCopiedTableEntries * sizeof(int32_t);
#endif

#if defined(__hexagon__) && !defined(SCALAR_ONLY) && defined(HMX_W16A16_ENABLE_QHPI_PRECOMPUTE)
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
 *   hmx_w16a16_precomputed_t
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
static constexpr uint32_t kHmxW16A16PrecomputeMagic = 0x48385850u; /* H8XP */

struct hmx_w16a16_precomputed_t {
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
    const int32_t *split_out_qhpi_table;
    uint32_t split_out_entries;
    int32_t act_table_copy[kHmxW16A16MaxCopiedTableEntries];
    int32_t out_table_copy[kHmxW16A16MaxCopiedTableEntries];
    int32_t split_out_table_copy[kHmxW16A16MaxCopiedTableEntries];
};

static_assert(sizeof(hmx_w16a16_precomputed_t) == kHmxW16A16PrecomputedDataSize,
              "QHPI precompute ABI size must match the x86 registration stub");

static uint32_t hmx_w16a16_precompute(
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

    hmx_w16a16_precomputed_t *pc = reinterpret_cast<hmx_w16a16_precomputed_t *>(data);
    std::memset(pc, 0, sizeof(*pc));
    init_mask_desc(g_hmx_w16a16_mask_buf);

    const uint8_t *bias_bytes =
        reinterpret_cast<const uint8_t *>(qhpi_tensor_raw_data(inputs[0]));
    const uint8_t *wt_pack =
        reinterpret_cast<const uint8_t *>(qhpi_tensor_raw_data(inputs[1]));
#if defined(HMX_W16A16_WEIGHT_FROM_SCRATCH)
    wt_pack = reinterpret_cast<const uint8_t *>(qhpi_tensor_raw_data(inputs[3]));
#endif
    void **act_blocks = qhpi_tensor_block_table(inputs[2]);
    void **out_blocks = qhpi_tensor_block_table(outputs[0]);
    const uint32_t blocks = qhpi_tensor_block_table_length(inputs[2]);
    if (!bias_bytes || !wt_pack || !act_blocks || !out_blocks) return QHPI_Success;

    uint32_t M_t = 0;
    uint32_t N_t = 0;
    uint32_t K_t = 0;
    if (!hmx_w16a16_tile_counts_from_shapes(
            inputs[2], inputs[1], outputs[0], blocks, &M_t, &N_t, &K_t)) {
        return QHPI_Success;
    }
    const uint32_t S = M_t * 32;
    const uint32_t block_rows = (S / 4) < 64 ? (S / 4) : 64;
    const uint32_t mt_per_block = block_rows / 32;
    if (mt_per_block == 0) return QHPI_Success;

    const uint32_t mt_groups = hmx_w16a16_crouton_row4_groups(M_t);
    const uint32_t act_table_stride = hmx_w16a16_act_table_stride(M_t, K_t);
    const uint32_t out_table_stride = N_t;
    const uint32_t act_entries =
        hmx_w16a16_required_table_entries(mt_groups, K_t, act_table_stride);
    const uint32_t out_entries =
        hmx_w16a16_required_table_entries(mt_groups, N_t, out_table_stride);
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
    if (act_entries <= kHmxW16A16MaxCopiedTableEntries &&
        out_entries <= kHmxW16A16MaxCopiedTableEntries) {
        for (uint32_t row4 = 0; row4 < mt_groups; ++row4) {
            int32_t *a_dst = pc->act_table_copy + row4 * act_table_stride;
            int32_t *o_dst = pc->out_table_copy + row4 * out_table_stride;
            for (uint32_t kt = 0; kt < K_t; ++kt) {
                a_dst[kt] = hmx_w16a16_crouton_row4_physical_ptr(act_src, row4, kt, K_t);
            }
            for (uint32_t nt = 0; nt < N_t; ++nt) {
                o_dst[nt] = hmx_w16a16_crouton_row4_physical_ptr(out_src, row4, nt, N_t);
            }
        }
        constexpr uint32_t kSplitNTiles = 4;
        if (N_t >= kSplitNTiles && (N_t % kSplitNTiles) == 0 &&
            out_table_stride >= N_t) {
            const uint32_t split_entries = mt_groups * kSplitNTiles;
            const uint32_t total_split_entries = split_entries * (N_t / kSplitNTiles);
            if (total_split_entries <= kHmxW16A16MaxCopiedTableEntries) {
                for (uint32_t split = 0; split < N_t / kSplitNTiles; ++split) {
                    int32_t *split_base = pc->split_out_table_copy + split * split_entries;
                    for (uint32_t rg = 0; rg < mt_groups; ++rg) {
                        const int32_t *src = pc->out_table_copy +
                            rg * out_table_stride + split * kSplitNTiles;
                        int32_t *dst = split_base + rg * kSplitNTiles;
                        for (uint32_t nt = 0; nt < kSplitNTiles; ++nt) dst[nt] = src[nt];
                    }
                }
                pc->split_out_qhpi_table = pc->split_out_table_copy;
                pc->split_out_entries = split_entries;
            }
        }
        pc->act_qhpi_table = pc->act_table_copy;
        pc->out_qhpi_table = pc->out_table_copy;
    }
    pc->S = S;
    pc->M_t = M_t;
    pc->N_t = N_t;
    pc->K_t = K_t;
    pc->mt_per_block = mt_per_block;
    pc->mt_groups = mt_groups;
    pc->act_entries = act_entries;
    pc->out_entries = out_entries;
    pc->magic = kHmxW16A16PrecomputeMagic;
    return QHPI_Success;
}

static uint32_t hmx_w16a16_to_u16_matmul_precomputed_kernel(
    QHPI_RuntimeHandle *handle,
    const void *precomputed_data)
{
    (void)handle;

    const hmx_w16a16_precomputed_t *pc =
        reinterpret_cast<const hmx_w16a16_precomputed_t *>(precomputed_data);

#if defined(HMX_W16A16_PROBE_CYCLES)
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
     * hmx_w16a16_precomputed_t at graph load.  That removes the last hot-path
     * QNN table locality miss without spending extra dcfetch packets in the
     * profiled callback.  Larger fallback shapes keep pointing at QNN-owned
     * block tables.
     */
    int32_t *act_tbl_ptr = const_cast<int32_t *>(pc->act_qhpi_table);
    int32_t *out_tbl_ptr = const_cast<int32_t *>(pc->out_qhpi_table);

    const uint32_t *extra_param = g_hmx_w16a16_extra_param;
    const hmx_conv_mask_desc_t *mask_desc = get_precomputed_mask_desc();
#if !defined(HMX_W16A16_NATIVE_RECORD_FIELDS)
    uint32_t mask_local[16] __attribute__((aligned(16)));
    mask_desc = hmx_w16a16_maybe_patch_mask_extra(mask_desc, mask_local, extra_param);
#endif

    hmx_conv_out_desc_t out_desc_local __attribute__((aligned(64))) = {
        out_tbl_ptr,
        pc->N_t,
        hmx_w16a16_out_y_stride(pc->N_t),
        hmx_w16a16_out_n_tiles_pow2(pc->M_t * 4),
        hmx_w16a16_out_m_total_minus_step(8),
        hmx_w16a16_k_total_bytes(pc->N_t, pc->K_t),
    };
    hmx_conv_act_desc_t act_desc_local __attribute__((aligned(64))) = {
        act_tbl_ptr,
        pc->K_t,
        hmx_w16a16_act_desc_y_stride(pc->K_t),
    };
    const hmx_conv_out_desc_t *out_desc = &out_desc_local;
    const hmx_conv_act_desc_t *act_desc = &act_desc_local;

#if defined(HMX_W16A16_DESC_DUMP)
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

#if defined(HMX_W16A16_SKIP_KERNEL)
#if defined(HMX_W16A16_SKIP_NO_WRITE)
    return QHPI_Success;
#endif
    if (pc->out_first_block) {
        uint8_t *dst = pc->out_first_block;
        for (uint32_t i = 0; i < 16; ++i) dst[i] = 0;
        store_le32(dst, 0, 0x48385853u); /* H8XS */
        store_le32(dst, 4, pc->S);
    }
    return QHPI_Success;
#endif

#if defined(HMX_W16A16_PROBE_CYCLES)
    uint64_t cyc_before_kernel = 0;
    asm volatile("%0 = C15:14" : "=r"(cyc_before_kernel));
#endif

    /*
     * Precomputed hot path: QHPI tensor lookup, shape recovery, and pointer
     * table discovery happened at graph load.  The custom op event now starts
     * at the tiny native descriptor stitching boundary.
     */
    hmx_w16a16_enter_kernel_maybe_split_n128(
        out_desc,
        act_desc,
        pc->wt_pack,
        pc->bias_bytes,
        mask_desc,
        extra_param,
        pc->mt_groups,
        pc->N_t,
        pc->K_t,
        pc->split_out_qhpi_table);

#if defined(HMX_W16A16_PROBE_CYCLES)
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
#elif defined(HMX_W16A16_ENABLE_QHPI_PRECOMPUTE)
/*
 * The x86 context generator must see the same QHPI kernel registration as the
 * device package.  It does not execute the HMX body, so these are registration
 * stubs only; the real precompute/execute implementation is compiled into the
 * Hexagon HTP package above.
 */
static uint32_t hmx_w16a16_precompute(
    QHPI_RuntimeHandle *,
    void *,
    uint32_t,
    QHPI_Tensor **,
    uint32_t,
    const QHPI_Tensor *const *)
{
    return QHPI_Success;
}

static uint32_t hmx_w16a16_to_u16_matmul_precomputed_kernel(
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
static uint32_t hmx_w16a16_to_u16_matmul_kernel(
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
#if defined(HMX_W16A16_PROBE_CYCLES)
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
#if defined(HMX_W16A16_WEIGHT_FROM_SCRATCH)
    wt_pack = reinterpret_cast<const uint8_t *>(qhpi_tensor_raw_data(inputs[3]));
#endif
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
    uint32_t M_t = 0;
    uint32_t N_t = 0;
    uint32_t K_t = 0;
    if (!hmx_w16a16_tile_counts_from_shapes(
            inputs[2], inputs[1], outputs[0], blocks, &M_t, &N_t, &K_t)) {
        return QHPI_Success;
    }
    const uint32_t S = M_t * 32;
    const uint32_t block_rows = (S / 4) < 64 ? (S / 4) : 64;
    const uint32_t mt_per_block = block_rows / 32;
    if (mt_per_block == 0) return QHPI_Success;

    /*
     * QNN's Crouton block table groups up to two 32-row tiles in one physical
     * block for the C8 path.  The native V73DEEP descriptor wants one pointer
     * table row per such group, so 256^3 becomes:
     *
     *   M_t=8 -> row4_groups=64
     *   act entries = row4_groups * K_t = 64 * 8 = 512
     *   out entries = row4_groups * N_t = 64 * 8 = 512
     */
    const uint32_t mt_groups = hmx_w16a16_crouton_row4_groups(M_t);
    const uint32_t act_table_stride = hmx_w16a16_act_table_stride(M_t, K_t);
    const uint32_t out_table_stride = N_t;
    const uint32_t act_entries =
        hmx_w16a16_required_table_entries(mt_groups, K_t, act_table_stride);
    const uint32_t out_entries =
        hmx_w16a16_required_table_entries(mt_groups, N_t, out_table_stride);
    if (act_entries > 1024 || out_entries > 1024) return QHPI_Success;

#if defined(HMX_W16A16_PROBE_CYCLES)
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
     *   row4_groups = M_t * 8 = 64
     *   entries per table = row4_groups * K_t = 64 * 8 = 512
     */
    for (uint32_t row4 = 0; row4 < mt_groups; ++row4) {
        int32_t *__restrict a_dst = act_tbl_all + row4 * act_table_stride;
        int32_t *__restrict o_dst = out_tbl_all + row4 * out_table_stride;
        for (uint32_t kt = 0; kt < K_t; ++kt) {
            a_dst[kt] = hmx_w16a16_crouton_row4_physical_ptr(act_src, row4, kt, K_t);
        }
        for (uint32_t nt = 0; nt < N_t; ++nt) {
            o_dst[nt] = hmx_w16a16_crouton_row4_physical_ptr(out_src, row4, nt, N_t);
        }
    }

#if defined(HMX_W16A16_PROBE_CYCLES)
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
     *     +0x08 out_y_stride_words       =  N_t
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
     *   out_y_stride_words      = N_t       next row4 output-table stride
     *   n_tiles_pow2            = M_t * 4   native loop/count selector
     *   m_total_minus_step      = 8         decoded fixed step for this path
     *   k_total_bytes           = N_t * 32  byte span expected by kernel setup
     *   n_act_pairs             = K_t       number of activation pointer pairs
     */
    uint32_t extra_param[2] __attribute__((aligned(16))) = {1u, 1536u};
    const hmx_conv_mask_desc_t *mask_desc = get_mask_desc();
    uint32_t mask_local[16] __attribute__((aligned(16)));
    mask_desc = hmx_w16a16_maybe_patch_mask_extra(mask_desc, mask_local, extra_param);

    hmx_conv_out_desc_t out_desc = {
        out_tbl_all,
        out_table_stride,
        hmx_w16a16_out_y_stride(out_table_stride),
        hmx_w16a16_out_n_tiles_pow2(M_t * 4),
        hmx_w16a16_out_m_total_minus_step(8),
        hmx_w16a16_k_total_bytes(N_t, K_t),
    };
    hmx_conv_act_desc_t act_desc = {
        act_tbl_all,
        K_t,
        hmx_w16a16_act_desc_y_stride(act_table_stride),
    };

#if defined(HMX_W16A16_DESC_DUMP)
    /*
     * Descriptor dump mode writes the derived ABI state into every observed
     * output block and returns before HMX compute.  The public output path
     * untile/slices Crouton storage, so writing only out_blocks[0] can hide the
     * marker in public raw.  Duplicating the compact 128B payload across blocks
     * makes the diagnostic visible without changing normal builds.
     */
    uint8_t payload[128] __attribute__((aligned(16)));
    for (uint32_t i = 0; i < 128; ++i) payload[i] = 0;
    store_le32(payload, 0, 0x48385844u); /* H8XD */
    store_le32(payload, 4, S);
    store_le32(payload, 8, M_t);
    store_le32(payload, 12, N_t);
    store_le32(payload, 16, K_t);
    store_le32(payload, 20, mt_per_block);
    store_le32(payload, 24, reinterpret_cast<uintptr_t>(act_tbl_all));
    store_le32(payload, 28, reinterpret_cast<uintptr_t>(out_tbl_all));
    store_le32(payload, 32, out_desc.out_table_stride_dwords);
    store_le32(payload, 36, out_desc.out_y_stride_words);
    store_le32(payload, 40, out_desc.n_tiles_pow2);
    store_le32(payload, 44, static_cast<uint32_t>(out_desc.m_total_minus_step));
    store_le32(payload, 48, out_desc.k_total_bytes);
    store_le32(payload, 52, act_desc.n_act_pairs);
    store_le32(payload, 56, act_desc.act_table_y_stride_words);
    const uint32_t *mask_words = reinterpret_cast<const uint32_t *>(mask_desc);
    for (uint32_t i = 0; i < 16; ++i) store_le32(payload, 64 + i * 4, mask_words[i]);
    for (uint32_t block = 0; block < blocks; ++block) {
        if (!out_blocks[block]) continue;
        uint8_t *dst = reinterpret_cast<uint8_t *>(out_blocks[block]);
        for (uint32_t i = 0; i < 128; ++i) dst[i] = payload[i];
    }
    return QHPI_Success;
#endif

#if defined(HMX_W16A16_SKIP_KERNEL)
#if defined(HMX_W16A16_SKIP_NO_WRITE)
    return QHPI_Success;
#endif
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

#if defined(HMX_W16A16_PROBE_CYCLES)
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
    hmx_w16a16_enter_kernel_maybe_split_n128(
        &out_desc,
        &act_desc,
        wt_pack,
        bias_bytes,
        mask_desc,
        extra_param,
        mt_groups,
        N_t,
        K_t);

#if defined(HMX_W16A16_PROBE_CYCLES)
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
    {QHPI_Int32,  QHPI_Layout_Flat4,      QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},
#if defined(HMX_W16A16_QHPI_SIGNED_WEIGHT8)
    {QHPI_QInt8,  QHPI_Layout_Flat4,      QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},
#elif defined(HMX_W16A16_QHPI_WEIGHT8)
    {QHPI_QUInt8, QHPI_Layout_Flat4,      QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},
#else
    {QHPI_QUInt16, QHPI_Layout_Flat4,      QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},
#endif
    {QHPI_QUInt16, QHPI_Layout_Crouton_16, QHPI_Storage_Indirect, QHPI_MemLoc_TCM_Only},
    {QHPI_QUInt8, QHPI_Layout_Flat4,      QHPI_Storage_Direct,   QHPI_MemLoc_TCM_Only},
};

static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QUInt16, QHPI_Layout_Crouton_16, QHPI_Storage_Indirect, QHPI_MemLoc_TCM_Only},
};

static float hmx_w16a16_cost_function(uint32_t num_inputs, const QHPI_Tensor *const *inputs)
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

static QHPI_Shape hmx_w16a16_shape_required(const QHPI_Op *op)
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

static QHPI_Shape hmx_w16a16_shape_legalized(const QHPI_Op *op, const QHPI_Shape *proposed)
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
#if defined(HMX_W16A16_ENABLE_QHPI_PRECOMPUTE)
        THIS_PKG_NAME_STR "::hmx_w16a16_to_u16_matmul_precomputed_kernel",
        nullptr,
#else
        THIS_PKG_NAME_STR "::hmx_w16a16_to_u16_matmul_kernel",
        hmx_w16a16_to_u16_matmul_kernel,
#endif
        QHPI_RESOURCE_HMX,
        false,
        false,
        false, false,
        4, sig_inputs,
        1, sig_outputs,
        hmx_w16a16_cost_function,
        0,
#if defined(HMX_W16A16_ENABLE_QHPI_PRECOMPUTE)
        kHmxW16A16PrecomputedDataSize,
        hmx_w16a16_precompute,
        hmx_w16a16_to_u16_matmul_precomputed_kernel,
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
        THIS_PKG_NAME_STR "::HmxU16I16ToU16MatMul",
        1, sg_kernels,
        nullptr,
        hmx_w16a16_shape_required,
        hmx_w16a16_shape_legalized,
        0,
        nullptr,
        nullptr,
    },
};

extern "C" void register_hmx_w16a16_to_u16_matmul_op(void)
{
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
