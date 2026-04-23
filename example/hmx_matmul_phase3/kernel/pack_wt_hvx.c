/*
 * pack_wt_hvx.c — PackWeightToHmxTile op (Phase 3B, Path W).
 *
 * Produces HMX-tile-format weight blocks from a flat [1,1,K,N] int8 tensor.
 * One 1 KB tile per (k_tile, n_tile) == 32 rows × 32 cols.
 *
 * Algorithm: reuse Phase 2's `pack_weight_32x32` which is a 4-row × 32-col
 * byte transpose implemented as two back-to-back `Q6_Vb_vshuff_Vb` per
 * 128-byte kg chunk (8 kg per tile). See Agent/hvx_4way_byte_transpose_re.md
 * for the correctness proof (rotate-right-by-2 on a 7-bit byte index).
 *
 * Per (kt, nt): input is a 32×32 byte sub-matrix sliced from w[K×N] at
 * row-range [kt*32, kt*32+32) and col-range [nt*32, nt*32+32). Because N
 * may not equal 32, we must gather the 32×32 slice with a per-kg stride
 * before feeding pack_weight_32x32. We inline the gather to avoid a DDR
 * round-trip: grab 32 contiguous bytes per K-row (N-stride), write into a
 * 1 KB scratch, then apply the 2×vshuff transpose.
 *
 * Tensor contract:
 *   Input  0:  int8 w       [1, 1, K, N]             Flat4 + Direct
 *   Output 0:  int8 tiles   [1, K/32, N/32, 1024]    Flat4 + Direct
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

/* -------------------------------------------------------------------
 * Gather 32×32 byte block from w[K, N] at (k0, n0), then 2×vshuff
 * transpose into 1 KB HMX weight tile.
 *
 * Scratch is on-stack (1 KB) — matches Phase 2's transient wt_tile usage.
 * ------------------------------------------------------------------- */
static inline void pack_one_wt_tile(
    int8_t *__restrict__ tile,           /* 1024 B */
    const int8_t *__restrict__ w,        /* [K, N] row-major */
    int k0, int n0, int N_stride)
{
    /* Gather 32 K-rows × 32 N-cols into a contiguous 32×32 buffer. */
    int8_t buf[32 * 32] __attribute__((aligned(128)));
    for (int kk = 0; kk < 32; kk++) {
        const int8_t *src = &w[(k0 + kk) * N_stride + n0];
        memcpy(&buf[kk * 32], src, 32);
    }

#ifdef __hexagon__
    /* 2×vshuff per kg. 8 kg total per tile. */
    for (int kg = 0; kg < 8; kg++) {
        HVX_Vector v, s1, s2;
        memcpy(&v, &buf[kg * 128], sizeof(HVX_Vector));
        s1 = Q6_Vb_vshuff_Vb(v);
        s2 = Q6_Vb_vshuff_Vb(s1);
        memcpy(tile + kg * 128, &s2, sizeof(HVX_Vector));
    }
#else
    /* Scalar reference (matches the non-hexagon branch of
     * pack_weight_32x32 in Phase 2 code). */
    for (int kg = 0; kg < 8; kg++) {
        uint32_t *__restrict__ dst = (uint32_t *)(tile + 128 * kg);
        const uint8_t *r0 = (const uint8_t *)&buf[(kg * 4 + 0) * 32];
        const uint8_t *r1 = (const uint8_t *)&buf[(kg * 4 + 1) * 32];
        const uint8_t *r2 = (const uint8_t *)&buf[(kg * 4 + 2) * 32];
        const uint8_t *r3 = (const uint8_t *)&buf[(kg * 4 + 3) * 32];
        for (int col = 0; col < 32; col++) {
            dst[col] =  (uint32_t)r0[col]
                     | ((uint32_t)r1[col] << 8)
                     | ((uint32_t)r2[col] << 16)
                     | ((uint32_t)r3[col] << 24);
        }
    }
#endif
}

/* Public entry for sim harness. */
void pack_wt_hvx_kernel_body(
    const int8_t *w,  /* [K, N] row-major */
    int8_t *out,      /* [K/32, N/32, 1024] flat */
    int K, int N)
{
    const int K_tiles = K / 32;
    const int N_tiles = N / 32;
    for (int kt = 0; kt < K_tiles; kt++) {
        for (int nt = 0; nt < N_tiles; nt++) {
            int8_t *tile = out + (kt * N_tiles + nt) * 1024;
            pack_one_wt_tile(tile, w, kt * 32, nt * 32, N);
        }
    }
}

/* -------------------------------------------------------------------
 * QHPI kernel entry.
 * ------------------------------------------------------------------- */
static uint32_t pack_wt_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs,  const QHPI_Tensor *const *inputs)
{
    (void)handle; (void)num_outputs; (void)num_inputs;

    const int8_t *w   = (const int8_t *)qhpi_tensor_raw_data(inputs[0]);
    int8_t       *out = (int8_t *)qhpi_tensor_raw_data(outputs[0]);

    QHPI_Shape ws = qhpi_tensor_shape(inputs[0]);
    const int K = (int)ws.dims[ws.rank - 2];
    const int N = (int)ws.dims[ws.rank - 1];

    pack_wt_hvx_kernel_body(w, out, K, N);
    return QHPI_Success;
}

/* -------------------------------------------------------------------
 * QHPI registration.
 * ------------------------------------------------------------------- */
static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_QInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};

static QHPI_Kernel_v1 sg_kernels[] = {
    {
        /* .function_name      */ THIS_PKG_NAME_STR "::pack_wt_hvx",
        /* .function           */ pack_wt_kernel,
        /* .resources          */ QHPI_RESOURCE_HVX,
        /* .source_destructive */ false,
        /* .multithreaded      */ true,
        /* .variable_inputs    */ false,
        /* .variable_outputs   */ false,
        /* .min_inputs         */ 1,
        /* .input_signature    */ sig_inputs,
        /* .min_outputs        */ 1,
        /* .output_signature   */ sig_outputs,
        /* .cost_function      */ NULL,
        /* .sync_block_size    */ 0,
        /* .precomputed_data_size */ 0,
        /* .do_precomputation_function */ NULL,
        /* .function_with_precomputed_data */ NULL,
        /* .predicate          */ NULL,
    },
};

static QHPI_OpInfo_v1 sg_ops[] = {
    {
        /* .name              */ THIS_PKG_NAME_STR "::PackWeightToHmxTile",
        /* .num_kernels       */ 1,
        /* .kernels           */ sg_kernels,
        /* .early_rewrite     */ NULL,
        /* .shape_required    */ NULL,
        /* .shape_legalized   */ NULL,
        /* .tile_output       */ 0,
        /* .build_tile        */ NULL,
        /* .late_rewrite      */ NULL,
    },
};

extern "C" void register_pack_wt_op(void)
{
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
