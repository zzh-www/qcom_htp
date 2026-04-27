/*
 * crouton_pack_spike_hvx.c — Phase 1 dlsym spike for Route 1.
 *
 * Goal: prove that an op-pkg .so can call internal exports of
 * libQnnHtpV75Skel.so loaded into the same FastRPC process. We anchor
 * `convert_to_crouton_b` (a GLOBAL DEFAULT export at 0x237700, RE'd in
 * Agent/sig_convert_to_crouton_b_2026-04-25.md), invoke it on a fixed
 * 32×128 u8 tile, and bit-compare against a pure-C reference.
 *
 * Tensor contract:
 *   Input  0 : uint8 act       [1, 1, 32, 128]   Flat4 + Direct DDR/TCM
 *   Output 0 : uint8 crouton   [1, 1, 4, 1024]   Flat4 + Direct DDR/TCM
 *                              4 Crouton blocks × 1024 B (depth lanes)
 *   Output 1 : uint8 stats     [1, 1, 1, 4]      Flat4 + Direct
 *                              [skel_done, ref_done, num_diffs(sat255), max_diff]
 *
 * SUCCESS criteria for the spike: stats == [1, 1, 0, 0].
 *   stats[0] == 1 → convert_to_crouton_b symbol resolved AND call returned.
 *   stats[1] == 1 → reference impl ran.
 *   stats[2] == 0 ∧ stats[3] == 0 → outputs bit-exact.
 */

#include "HTP/core/qhpi.h"
#include <stdint.h>
#include <string.h>

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

/* ---------- libQnnHtpV75Skel.so internal symbol (resolved by DSP loader) ---------- */
typedef struct {
    void*    block_offset_table;   /* +0x00 int32_t[] of dst byte-offsets */
    int32_t  outer_step;           /* +0x04 ignored when single-pass     */
    int32_t  row_stride;           /* +0x08 input row stride in bytes    */
    uint32_t channel_groups;       /* +0x0c #depth-32 lanes              */
    uint32_t height_tiles;         /* +0x10 #4-row spatial groups        */
    uint32_t depth;                /* +0x14 depth in BYTES               */
} convert_to_crouton_params_t;

extern "C" void convert_to_crouton_b(const convert_to_crouton_params_t *p,
                                      uintptr_t aux,
                                      const void *flat_in);

/* ---------- C reference (Agent/forceformat_crouton_re.md §4.2) ---------- */
static void crouton_pack_b_reference(uint8_t *out, const uint8_t *in,
                                      int M, int K)
{
    /* For M=32 K=128: 4 depth blocks × 1024 B each.
     * Block g covers depth [g*32, g*32+31]; within block, h-th 4-row group
     * lives at offset h*128, with row r at offset r*32. */
    const int n_height_groups = M / 4;   /* 8 */
    const int n_depth_lanes   = K / 32;  /* 4 */
    for (int h = 0; h < n_height_groups; h++) {
        for (int g = 0; g < n_depth_lanes; g++) {
            uint8_t *blk = out + g * (M * 32) + h * 128;
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 32; c++) {
                    blk[r * 32 + c] = in[(h * 4 + r) * K + g * 32 + c];
                }
            }
        }
    }
}

static uint32_t crouton_pack_spike_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs,  const QHPI_Tensor *const *inputs)
{
    (void)handle; (void)num_outputs; (void)num_inputs;

    const uint8_t *in       = (const uint8_t *)qhpi_tensor_raw_data(inputs[0]);
    uint8_t       *crouton  = (uint8_t *)qhpi_tensor_raw_data(outputs[0]);
    uint8_t       *stats    = (uint8_t *)qhpi_tensor_raw_data(outputs[1]);

    /* Hardcoded: 32×128 in → 4×1024 out. */
    enum { M = 32, K = 128, OUT_BYTES = 4 * 1024 };

    stats[0] = 0; stats[1] = 0; stats[2] = 0; stats[3] = 0;

    /* (1) Reference */
    static uint8_t ref_buf[OUT_BYTES] __attribute__((aligned(128)));
    memset(ref_buf, 0, OUT_BYTES);
    crouton_pack_b_reference(ref_buf, in, M, K);
    stats[1] = 1;

    /* (2) Skel call. Scatter table = absolute pointers to each 1KB block. */
    static int32_t scatter_table[4] __attribute__((aligned(16)));
    for (int i = 0; i < 4; i++) {
        scatter_table[i] = (int32_t)(uintptr_t)(crouton + i * 1024);
    }
    convert_to_crouton_params_t p;
    p.block_offset_table = (void *)scatter_table;
    p.outer_step         = 0;
    p.row_stride         = K;        /* 128 */
    p.channel_groups     = 4;        /* K/32 */
    p.height_tiles       = M / 4;    /* 8 */
    p.depth              = K;        /* 128, in BYTES */

    memset(crouton, 0, OUT_BYTES);
    /* aux=16: ct0(16)=4 → r25 stride = (1 << (4-4)) << 7 = 128 per height iter
     * (per Agent/sig_convert_to_crouton_b_2026-04-25.md §8 outer loop RE) */
    convert_to_crouton_b(&p, 16, in);
    stats[0] = 1;   /* survived without crash */

    /* (3) Compare */
    uint32_t n_diff = 0;
    uint8_t  max_d = 0;
    for (int i = 0; i < OUT_BYTES; i++) {
        int d = (int)crouton[i] - (int)ref_buf[i];
        if (d != 0) {
            n_diff++;
            uint8_t ad = (d < 0) ? (uint8_t)(-d) : (uint8_t)d;
            if (ad > max_d) max_d = ad;
        }
    }
    stats[2] = (n_diff > 255) ? 255 : (uint8_t)n_diff;
    stats[3] = max_d;
    return QHPI_Success;
}

static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};

static QHPI_Kernel_v1 sg_kernels[] = {
    {
        THIS_PKG_NAME_STR "::crouton_pack_spike",
        crouton_pack_spike_kernel,
        QHPI_RESOURCE_HVX,
        false, false, false, false,
        1, sig_inputs,
        2, sig_outputs,
        nullptr, 0, 0, nullptr, nullptr, nullptr,
    },
};

static QHPI_OpInfo_v1 sg_ops[] = {
    {
        THIS_PKG_NAME_STR "::CroutonPackSpike",
        1, sg_kernels,
        nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    },
};

extern "C" void register_crouton_pack_spike_op(void)
{
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
