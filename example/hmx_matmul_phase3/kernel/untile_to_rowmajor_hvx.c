/*
 * untile_to_rowmajor_hvx.c — UntileToRowMajor op for V8.
 *
 * Input  : [1, M_tiles, N_tiles, 1024] u8 VTCM tile-layout
 *          (each tile is 32 rows × 32 cols stored contiguously 1 KiB).
 * Output : [1, 1, M, N] u8 DDR row-major.
 *
 *   out[(mt*32 + r) * N + nt*32 + c] = tile[(mt*N_tiles + nt)*1024 + r*32 + c]
 *
 * Fast path: pack 4 tile rows (128 B contiguous in VTCM) into one HVX
 * vector, then 4× u64 stores distribute it to 4 DDR rows at stride N.
 *
 * This is the mirror of pack_act_rm_hvx and corresponds to QNN's built-in
 * tile-layout → DDR conversion at graph output boundaries.
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

void untile_to_rowmajor_hvx_kernel_body(
    const uint8_t *in,    /* [M_tiles, N_tiles, 1024] */
    uint8_t       *out,   /* [M, N] row-major */
    int M, int N,
    uint32_t mt_start, uint32_t mt_end)
{
    (void)M;
    const int N_tiles = N / 32;
    /* Tile-first: sequential VTCM read, strided 32-B DDR writes.
     * Bottleneck is DDR partial-write cache traffic at ~1.6M cyc @ 512³.
     * Tried row-first (VTCM bank conflicts: 2.3M) and per-row staging
     * with full-row memcpy (2.0M) — neither beat tile-first.  The DDR
     * bandwidth for N-strided 32-B writes is the hard limit here. */
    for (uint32_t mt = mt_start; mt < mt_end; mt++) {
        const uint8_t *tiles_base = in + (mt * N_tiles) * 1024;
        for (int nt = 0; nt < N_tiles; nt++) {
            const uint8_t *tile = tiles_base + nt * 1024;
            uint8_t *out_base = out + (mt * 32) * N + nt * 32;
            for (int r = 0; r < 32; r++) {
                const uint64_t *s = (const uint64_t *)&tile[r * 32];
                uint64_t *d = (uint64_t *)&out_base[r * N];
                d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
            }
        }
    }
}

static uint32_t untile_to_rowmajor_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs,  const QHPI_Tensor *const *inputs)
{
    (void)num_outputs; (void)num_inputs;
    const uint8_t *in  = (const uint8_t *)qhpi_tensor_raw_data(inputs[0]);
    uint8_t       *out = (uint8_t *)qhpi_tensor_raw_data(outputs[0]);

    QHPI_Shape os = qhpi_tensor_shape(outputs[0]);
    const int M = (int)os.dims[os.rank - 2];
    const int N = (int)os.dims[os.rank - 1];
    const int M_tiles = M / 32;

    const uint32_t num_slices = qhpi_num_slices(handle);
    const uint32_t slice_idx  = qhpi_slice_number(handle);
    const uint32_t mt_start   = (uint32_t)((uint64_t)M_tiles * slice_idx)     / num_slices;
    const uint32_t mt_end     = (uint32_t)((uint64_t)M_tiles * (slice_idx+1)) / num_slices;

    untile_to_rowmajor_hvx_kernel_body(in, out, M, N, mt_start, mt_end);
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
        THIS_PKG_NAME_STR "::untile_to_rowmajor_hvx",
        untile_to_rowmajor_kernel,
        QHPI_RESOURCE_HVX,
        false, true, false, false,       /* multithreaded = true */
        1, sig_inputs,
        1, sig_outputs,
        NULL, 0, 0, NULL, NULL, NULL,
    },
};

static QHPI_OpInfo_v1 sg_ops[] = {
    {
        THIS_PKG_NAME_STR "::UntileToRowMajor",
        1, sg_kernels,
        NULL, NULL, NULL, 0, NULL, NULL,
    },
};

extern "C" void register_untile_to_rowmajor_op(void)
{
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
