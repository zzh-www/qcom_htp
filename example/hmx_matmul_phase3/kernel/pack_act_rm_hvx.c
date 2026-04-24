/*
 * pack_act_rm_hvx.c — PackActivationU8RowMajor (Phase 3D.4 for V8).
 *
 * Produces 1 KiB row-major activation tiles consumed by V8's
 * `:cm` HMX kernel. tile[r*32 + k] = act[m_tile*32 + r][k_tile*32 + k].
 *
 * Simpler than pack_act_u8_hvx (which does 2-stream interleave for
 * plain mxmem). `:cm` reads row-major directly — no interleaving.
 *
 * Tensor contract:
 *   Input  0: uint8 act      [1, 1, M, K]           Flat4 + Direct
 *   Output 0: uint8 tile     [1, M/32, K/32, 1024]  Flat4 + Direct
 *
 * Declared multithreaded=true; QNN slices along M_tiles.
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

static inline void pack_one_rm_tile(
    uint8_t       *tile,     /* 1 KiB destination, aligned 128 */
    const uint8_t *a_base,   /* &act[m_tile*32][k_tile*32] */
    uint32_t       K_full)
{
    /* Copy 32 rows × 32 bytes to tile (1 KiB contiguous).
     * Pack 4 source rows (32 B each) into one 128 B HVX vector; do 8
     * aligned vmem stores to tile. Source rows are 32-byte aligned
     * because K is a multiple of 32 and K_full = K (a_base at 32-byte
     * stride offsets). */
#if defined(__hexagon__)
    /* Read 32 B from row via scalar u64×4, assemble into a 128 B vec
     * via intrinsic. Simpler: do 8 iterations × 4 rows; each iteration
     * reads 4×32 B with `memcpy(tile+kg*128, ..., 128)` one row at a
     * time via u64 stores.
     * Proven fast: 4 × u64 loads + 4 × u64 stores per row, 32 rows,
     * everything on HVX-adjacent hardware. Compiler inlines u64 moves
     * into 2-cycle scalar loops. */
    for (int r = 0; r < 32; r += 4) {
        const uint64_t *s0 = (const uint64_t *)&a_base[(r + 0) * K_full];
        const uint64_t *s1 = (const uint64_t *)&a_base[(r + 1) * K_full];
        const uint64_t *s2 = (const uint64_t *)&a_base[(r + 2) * K_full];
        const uint64_t *s3 = (const uint64_t *)&a_base[(r + 3) * K_full];
        uint64_t *d = (uint64_t *)&tile[r * 32];
        d[0] = s0[0]; d[1] = s0[1]; d[2] = s0[2]; d[3] = s0[3];
        d[4] = s1[0]; d[5] = s1[1]; d[6] = s1[2]; d[7] = s1[3];
        d[8] = s2[0]; d[9] = s2[1]; d[10] = s2[2]; d[11] = s2[3];
        d[12] = s3[0]; d[13] = s3[1]; d[14] = s3[2]; d[15] = s3[3];
    }
#else
    for (int r = 0; r < 32; r++)
        memcpy(&tile[r * 32], &a_base[r * K_full], 32);
#endif
}

void pack_act_rm_hvx_kernel_body(
    const uint8_t *a,     /* [M, K] row-major */
    uint8_t       *out,   /* [M/32, K/32, 1024] flat */
    int M, int K,
    uint32_t mt_start, uint32_t mt_end)
{
    const int K_tiles = K / 32;
    (void)M;
#if defined(__hexagon__)
    /* HVX-packed path: for each (mt, r_group of 4 rows, kt) produce one
     * 128 B vector [row0 32B | row1 32B | row2 32B | row3 32B] and store
     * with one aligned vmem to tile[kt][r_group*32]. 4 vmemu loads + 4
     * vmux masks + 3 vror shifts + 3 vor combines + 1 vmem store per
     * output vector.  ~12 HVX ops × 8 r-groups × K_tiles per mt. */
    const HVX_VectorPred p32 = Q6_Q_vsetq_R(32);
    const HVX_Vector     vz  = Q6_V_vzero();
    for (uint32_t mt = mt_start; mt < mt_end; mt++) {
        uint8_t *tiles_base = out + (mt * K_tiles) * 1024;
        const uint8_t *a_m = &a[mt * 32 * K];
        for (int r0 = 0; r0 < 32; r0 += 4) {
            const uint8_t *s0 = &a_m[(r0 + 0) * K];
            const uint8_t *s1 = &a_m[(r0 + 1) * K];
            const uint8_t *s2 = &a_m[(r0 + 2) * K];
            const uint8_t *s3 = &a_m[(r0 + 3) * K];
            for (int kt = 0; kt < K_tiles; kt++) {
                HVX_Vector v0, v1, v2, v3;
                memcpy(&v0, &s0[kt * 32], sizeof(HVX_Vector));  /* vmemu */
                memcpy(&v1, &s1[kt * 32], sizeof(HVX_Vector));
                memcpy(&v2, &s2[kt * 32], sizeof(HVX_Vector));
                memcpy(&v3, &s3[kt * 32], sizeof(HVX_Vector));
                /* Zero all but first 32 B of each vector. */
                v0 = Q6_V_vmux_QVV(p32, v0, vz);
                v1 = Q6_V_vmux_QVV(p32, v1, vz);
                v2 = Q6_V_vmux_QVV(p32, v2, vz);
                v3 = Q6_V_vmux_QVV(p32, v3, vz);
                /* Rotate each into its destination slot.  vror right by R
                 * means byte at position (i) moves to position (i-R) mod 128.
                 * To put v1's first 32 B at bytes 32..63, we rotate RIGHT by
                 * (128-32)=96, so position 0 → position 32. */
                HVX_Vector v1p = Q6_V_vror_VR(v1, 96);
                HVX_Vector v2p = Q6_V_vror_VR(v2, 64);
                HVX_Vector v3p = Q6_V_vror_VR(v3, 32);
                HVX_Vector combined =
                    Q6_V_vor_VV(Q6_V_vor_VV(v0, v1p),
                                Q6_V_vor_VV(v2p, v3p));
                *((HVX_Vector *)&tiles_base[kt * 1024 + r0 * 32]) = combined;
            }
        }
    }
#else
    for (uint32_t mt = mt_start; mt < mt_end; mt++) {
        for (int kt = 0; kt < K_tiles; kt++) {
            uint8_t *tile = out + (mt * K_tiles + kt) * 1024;
            for (int r = 0; r < 32; r++)
                memcpy(&tile[r * 32],
                       &a[(mt * 32 + r) * K + kt * 32], 32);
        }
    }
#endif
}

static uint32_t pack_act_rm_kernel(
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
    const int M_tiles = M / 32;

    const uint32_t num_slices = qhpi_num_slices(handle);
    const uint32_t slice_idx  = qhpi_slice_number(handle);
    const uint32_t mt_start   = (uint32_t)((uint64_t)M_tiles * slice_idx)     / num_slices;
    const uint32_t mt_end     = (uint32_t)((uint64_t)M_tiles * (slice_idx+1)) / num_slices;

    pack_act_rm_hvx_kernel_body(a, out, M, K, mt_start, mt_end);
    return QHPI_Success;
}

static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};

static QHPI_Kernel_v1 sg_kernels[] = {
    {
        THIS_PKG_NAME_STR "::pack_act_rm_hvx",
        pack_act_rm_kernel,
        QHPI_RESOURCE_HVX,
        /* source_destructive */ false,
        /* multithreaded       */ true,
        /* variable_inputs     */ false,
        /* variable_outputs    */ false,
        1, sig_inputs,
        1, sig_outputs,
        nullptr, 0, 0, nullptr, nullptr, nullptr,
    },
};

static QHPI_OpInfo_v1 sg_ops[] = {
    {
        THIS_PKG_NAME_STR "::PackActivationU8RowMajor",
        1, sg_kernels,
        nullptr, nullptr, nullptr, 0, nullptr, nullptr,
    },
};

extern "C" void register_pack_act_rm_op(void)
{
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
