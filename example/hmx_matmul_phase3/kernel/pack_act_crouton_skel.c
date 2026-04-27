/*
 * pack_act_crouton_skel.c — PackActCrouton (V10 Route 1).
 *
 * Wraps libQnnHtpV75Skel.so::convert_to_crouton_b (offset 0x237700) to
 * produce QNN-native Crouton-byte tile layout from a flat row-major
 * activation. Bit-exact output proven against pure-C reference at
 * 32×128 (Agent/dlsym_spike_PASS_2026-04-25.md).
 *
 * Tensor contract:
 *   Input  0: u8 act      [1, 1, M, K]                  Flat4 + Direct DDR/TCM
 *               (M %  4 == 0, K % 128 == 0)
 *   Output 0: u8 crouton  [1, K/32, M/4, 128]           Flat4 + Direct TCM
 *               block (g, h) at offset (g*M_grp + h)*128 holds 4-spatial × 32-depth
 *               where M_grp = M/4
 *
 * Multithreaded=true; QNN slices along M_grp (h dimension).
 */

#include "HTP/core/qhpi.h"
#include <stdint.h>
#include <string.h>

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

#if defined(__hexagon__)
/* libQnnHtpV75Skel.so internal — resolved by DSP loader at op-pkg load. */
typedef struct {
    void*    block_offset_table;
    int32_t  outer_step;
    int32_t  row_stride;
    uint32_t channel_groups;
    uint32_t height_tiles;
    uint32_t depth;
} convert_to_crouton_params_t;

extern "C" void convert_to_crouton_b(const convert_to_crouton_params_t *p,
                                      uintptr_t aux,
                                      const void *flat_in);
#endif

/* Pure-C reference Crouton-byte pack — generalized form of
 * crouton_pack_b_reference() in crouton_pack_spike_hvx.c. Output bit-exact
 * with the device skel-call (validated at 13 shapes,
 * Agent/pack_act_crouton_skel_2026-04-25.md). Used on x86 host so QNN's
 * const-prop pass can fold static-weight PackActCrouton at compile time. */
static void crouton_pack_b_reference_full(
    uint8_t *__restrict__ out, const uint8_t *__restrict__ a,
    int M, int K, uint32_t h_start, uint32_t h_end)
{
    const int M_grp = M / 4;
    const int K_grp = K / 32;
    for (uint32_t h = h_start; h < h_end; h++) {
        for (int g = 0; g < K_grp; g++) {
            uint8_t *blk = out + ((uint32_t)g * (uint32_t)M_grp + h) * 128u;
            for (int r = 0; r < 4; r++) {
                const uint8_t *src_row = a + ((h * 4u + (uint32_t)r) * (uint32_t)K) + (uint32_t)g * 32u;
                for (int c = 0; c < 32; c++) {
                    blk[r * 32 + c] = src_row[c];
                }
            }
        }
    }
}

static uint32_t pack_act_crouton_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs,  const QHPI_Tensor *const *inputs)
{
    (void)num_outputs; (void)num_inputs;

    const uint8_t *a   = (const uint8_t *)qhpi_tensor_raw_data(inputs[0]);
    uint8_t       *out = (uint8_t *)qhpi_tensor_raw_data(outputs[0]);

    QHPI_Shape as = qhpi_tensor_shape(inputs[0]);
    const int M = (int)as.dims[as.rank - 2];
    const int K = (int)as.dims[as.rank - 1];

    /* Slicing: along M_grp = M/4 (height groups of 4 rows) */
    const int M_grp = M / 4;
    const int K_grp = K / 32;

#if defined(__hexagon__)
    const uint32_t num_slices = qhpi_num_slices(handle);
    const uint32_t slice_idx  = qhpi_slice_number(handle);
#else
    /* x86 const-prop pass invokes us with no slicing — handle entire range. */
    (void)handle;
    const uint32_t num_slices = 1;
    const uint32_t slice_idx  = 0;
#endif
    const uint32_t h_start = (uint32_t)((uint64_t)M_grp * slice_idx)     / num_slices;
    const uint32_t h_end   = (uint32_t)((uint64_t)M_grp * (slice_idx+1)) / num_slices;
    if (h_end <= h_start) return QHPI_Success;

#if !defined(__hexagon__)
    /* x86 host path: pure-C reference. QNN's GraphPrepare const-prop pass
     * runs us on the host to fold static-input PackActCrouton at compile
     * time. Calling the device dlsym `convert_to_crouton_b` here would
     * dispatch to libQnnHtp.so's x86 simulator stub and segfault. */
    crouton_pack_b_reference_full(out, a, M, K, h_start, h_end);
    return QHPI_Success;
#else
    /* Skel function constraint: with aux=16 (ct0=4), the per-h-iter r20
     * advance becomes non-zero for h>=16 and walks beyond a small block table.
     * → split the slice into sub-calls of at most CHUNK_H=16 height-iters each.
     * For each sub-call, we pass the table pre-offset by chunk_h0*128 so that
     * the function's r25 = (sub_h)*128 lands at the correct flat-output byte. */
    enum { CHUNK_H = 16 };
    int32_t scatter_table[256] __attribute__((aligned(64)));
    if (K_grp > (int)(sizeof(scatter_table)/sizeof(scatter_table[0]))) {
        return QHPI_ErrorFatal;
    }

    for (uint32_t chunk_h0 = h_start; chunk_h0 < h_end; chunk_h0 += CHUNK_H) {
        const uint32_t chunk_h1 = chunk_h0 + CHUNK_H < h_end ? chunk_h0 + CHUNK_H : h_end;
        const uint32_t chunk_h_count = chunk_h1 - chunk_h0;

        for (int g = 0; g < K_grp; g++) {
            scatter_table[g] = (int32_t)(uintptr_t)(
                out + (uint32_t)g * (uint32_t)M_grp * 128u + chunk_h0 * 128u);
        }

        convert_to_crouton_params_t p;
        p.block_offset_table = (void *)scatter_table;
        p.outer_step         = 0;
        p.row_stride         = (int32_t)K;
        /* 4 blocks/inner-iter (vshuff(-32) topology); middle loop walks K_grp/4
         * sets of 4 entries from the table, advancing r26 by 16 each iter. */
        p.channel_groups     = 4;
        p.height_tiles       = chunk_h_count;
        p.depth              = (uint32_t)K;

        const uint8_t *a_chunk = a + (size_t)chunk_h0 * 4u * (size_t)K;
        /* aux=16 → r25 stride = h*128 within sub-call; valid only for chunk h<16. */
        convert_to_crouton_b(&p, 16, a_chunk);
    }

    return QHPI_Success;
#endif
}

static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};

static QHPI_Kernel_v1 sg_kernels[] = {
    {
        THIS_PKG_NAME_STR "::pack_act_crouton_skel",
        pack_act_crouton_kernel,
        QHPI_RESOURCE_HVX,
        false, false, false, false,  /* multithreaded=false (debugging) */
        1, sig_inputs,
        1, sig_outputs,
        nullptr, 0, 0, nullptr, nullptr, nullptr,
    },
};

static QHPI_OpInfo_v1 sg_ops[] = {
    {
        THIS_PKG_NAME_STR "::PackActCrouton",
        1, sg_kernels,
        nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    },
};

extern "C" void register_pack_act_crouton_op(void)
{
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
