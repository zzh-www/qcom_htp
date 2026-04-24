/*
 * tcm_dram_copy_hvx.c — bulk VTCM → DDR copy for V8's tile-layout output.
 *
 * Matches what QNN would effectively do at a graph output boundary when
 * the hot op produces tile-layout: a single contiguous memcpy of
 * M_tiles * N_tiles * 1024 bytes from VTCM to DDR — no scatter, no
 * layout change.  This is the path that hits full HVX write bandwidth
 * (aligned 128-B vmem bursts), bypassing the per-tile DDR cache-line
 * churn that the row-major scatter suffered.
 *
 * Input  : [1, M_tiles, N_tiles, 1024] u8 VTCM
 * Output : [1, M_tiles, N_tiles, 1024] u8 DDR (same shape)
 */

#include "HTP/core/qhpi.h"
#include <stdint.h>
#include <string.h>

#ifdef __hexagon__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

static uint32_t tcm_dram_copy_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs,  const QHPI_Tensor *const *inputs)
{
    (void)num_outputs; (void)num_inputs;
    const uint8_t *src = (const uint8_t *)qhpi_tensor_raw_data(inputs[0]);
    uint8_t       *dst = (uint8_t *)qhpi_tensor_raw_data(outputs[0]);

    QHPI_Shape os = qhpi_tensor_shape(outputs[0]);
    uint32_t total = 1;
    for (uint32_t i = 0; i < os.rank; i++) total *= os.dims[i];

    const uint32_t num_slices = qhpi_num_slices(handle);
    const uint32_t slice_idx  = qhpi_slice_number(handle);
    /* Round chunk boundaries to 128 B so HVX aligned stores stay aligned. */
    const uint32_t block = 128;
    const uint32_t nblocks = total / block;
    const uint32_t b_start = (uint32_t)((uint64_t)nblocks * slice_idx)     / num_slices;
    const uint32_t b_end   = (uint32_t)((uint64_t)nblocks * (slice_idx+1)) / num_slices;

#if defined(__hexagon__)
    /* Explicit HVX vmem load/store, 128 B per iter.  VTCM source is
     * 128-aligned; DDR destination address must also be 128-aligned
     * for vmem.  tile-layout tensor starts at a tile boundary (1 KiB
     * aligned), so this holds. */
    const uint8_t *s = src + b_start * block;
    uint8_t       *d = dst + b_start * block;
    const uint32_t nb = b_end - b_start;
    for (uint32_t i = 0; i < nb; i++) {
        HVX_Vector v;
        memcpy(&v, s + i * 128, 128);
        *(HVX_Vector *)(d + i * 128) = v;
    }
#else
    memcpy(dst + b_start * block, src + b_start * block,
           (size_t)(b_end - b_start) * block);
#endif
    /* Handle tail (< 128 B) on last slice. */
    if (slice_idx == num_slices - 1 && total % block != 0) {
        uint32_t tail_off = nblocks * block;
        memcpy(dst + tail_off, src + tail_off, total - tail_off);
    }
    return QHPI_Success;
}

static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};

static QHPI_Kernel_v1 sg_kernels[] = {
    {
        THIS_PKG_NAME_STR "::tcm_dram_copy_hvx",
        tcm_dram_copy_kernel,
        QHPI_RESOURCE_HVX,
        false, true, false, false,     /* multithreaded */
        1, sig_inputs,
        1, sig_outputs,
        NULL, 0, 0, NULL, NULL, NULL,
    },
};

static QHPI_OpInfo_v1 sg_ops[] = {
    {
        THIS_PKG_NAME_STR "::TcmDramCopy",
        1, sg_kernels,
        NULL, NULL, NULL, 0, NULL, NULL,
    },
};

extern "C" void register_tcm_dram_copy_op(void)
{
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
