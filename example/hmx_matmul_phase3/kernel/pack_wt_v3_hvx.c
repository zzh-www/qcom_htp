/*
 * pack_wt_v3_hvx.c — PackWeightToHmxTileV3 op (Phase 3C, Path W).
 *
 * Variant of pack_wt_hvx that iterates (nt, kt) instead of (kt, nt) and
 * emits uint8 tiles to match V3's expected packed_wt layout:
 *   [1, N/32, K/32, 1024]
 *
 * Per (nt, kt): Phase 2 4-K-row × 32-col pack (same per-tile layout as
 * pack_wt_hvx; only the outer iteration order differs). V3 consumes the
 * byte stream as uint8; the mathematical signed interpretation happens
 * in HMX via `weight.b = mxmem(...)`.
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

static inline void pack_one_wt_tile_v3(
    uint8_t *__restrict__ tile,           /* 1024 B */
    const int8_t *__restrict__ w,         /* [K, N] row-major */
    int k0, int n0, int N_stride)
{
    /* Gather 32 K-rows × 32 N-cols into stack buf via unaligned vmem. */
    int8_t buf[32 * 32] __attribute__((aligned(128)));
#if defined(__hexagon__)
    for (int kk = 0; kk < 32; kk++) {
        const int8_t *src = &w[(k0 + kk) * N_stride + n0];
        HVX_Vector v_row;
        memcpy(&v_row, src, sizeof(HVX_Vector));
        memcpy(&buf[kk * 32], &v_row, 32);
    }
#else
    for (int kk = 0; kk < 32; kk++) {
        const int8_t *src = &w[(k0 + kk) * N_stride + n0];
        memcpy(&buf[kk * 32], src, 32);
    }
#endif

#if defined(__hexagon__) && !defined(PACK_WT_V3_SCALAR_FORCE)
    for (int kg = 0; kg < 8; kg++) {
        HVX_Vector v, s1, s2;
        memcpy(&v, &buf[kg * 128], sizeof(HVX_Vector));
        s1 = Q6_Vb_vshuff_Vb(v);
        s2 = Q6_Vb_vshuff_Vb(s1);
        memcpy(tile + kg * 128, &s2, sizeof(HVX_Vector));
    }
#else
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

#if defined(__hexagon__) && !defined(PACK_WT_V3_SCALAR_FORCE)
/* Batched variant — process 4 consecutive nt tiles per row load, fully HVX.
 *
 * Each weight row at (k, n0) spans 128 B = 32 B × 4 nt-sub-tiles. Rather
 * than scalar-memcpy-redistributing each row into 4 staging bufs, we
 * regroup by tile with 2 layers of Q6_W_vshuff_VVR, in chunks of 4 K-rows.
 *
 * Layout per row:  V_i = [T_{i,0}, T_{i,1}, T_{i,2}, T_{i,3}]  (32 B each)
 * Goal per kg group (4 K-rows, k=kg*4..kg*4+3), per tile j:
 *   pre-vshuff 128 B = [T_{0,j}, T_{1,j}, T_{2,j}, T_{3,j}]
 *
 * Semantics of `Vdd = vshuff(Vu, Vv, Rt)` per v75 HVX PRM:
 *   Vdd.v[0] = Vv;  Vdd.v[1] = Vu
 *   for each bit-offset set in Rt (ascending): for k where (k & offset)==0,
 *     swap Vdd.v[1].ub[k] ↔ Vdd.v[0].ub[k+offset].
 *
 * Single-bit Rt=32: .lo = [Vv0..31, Vu0..31, Vv64..95, Vu64..95]
 *                   .hi = [Vv32..63, Vu32..63, Vv96..127, Vu96..127]
 *
 * shuff(V1, V0, 32): .lo = [T00, T10, T02, T12]  .hi = [T01, T11, T03, T13]
 * shuff(V3, V2, 32): .lo = [T20, T30, T22, T32]  .hi = [T21, T31, T23, T33]
 *
 * Single-bit Rt=64: .lo = Vv[0..63] ++ Vu[0..63]
 *                   .hi = Vv[64..127] ++ Vu[64..127]
 *
 * shuff(P23.lo, P01.lo, 64): .lo = [T00,T10,T20,T30] (tile 0)
 *                            .hi = [T02,T12,T22,T32] (tile 2)
 * shuff(P23.hi, P01.hi, 64): .lo = [T01,T11,T21,T31] (tile 1)
 *                            .hi = [T03,T13,T23,T33] (tile 3)
 *
 * Then 2× Q6_Vb_vshuff_Vb on each 128 B (standard Phase 2 4-row × 32-col
 * byte transpose). Finally 4 aligned vmem stores to the 4 tiles' kg slot.
 */
static inline void pack_wt_v3_batch4(
    uint8_t *tile_base,            /* &out[(nt_base * K_tiles + kt) * 1024] */
    int K_tiles,
    const int8_t *w,
    int k0, int n0, int N_stride)
{
    const int tile_stride = K_tiles * 1024;  /* nt stride in bytes */

    for (int kg = 0; kg < 8; kg++) {
        const int kk = kg * 4;
        HVX_Vector V0, V1, V2, V3;
        memcpy(&V0, &w[(k0 + kk + 0) * N_stride + n0], sizeof(HVX_Vector));
        memcpy(&V1, &w[(k0 + kk + 1) * N_stride + n0], sizeof(HVX_Vector));
        memcpy(&V2, &w[(k0 + kk + 2) * N_stride + n0], sizeof(HVX_Vector));
        memcpy(&V3, &w[(k0 + kk + 3) * N_stride + n0], sizeof(HVX_Vector));

        HVX_VectorPair P01 = Q6_W_vshuff_VVR(V1, V0, 32);
        HVX_VectorPair P23 = Q6_W_vshuff_VVR(V3, V2, 32);

        HVX_Vector p01lo = Q6_V_lo_W(P01);
        HVX_Vector p01hi = Q6_V_hi_W(P01);
        HVX_Vector p23lo = Q6_V_lo_W(P23);
        HVX_Vector p23hi = Q6_V_hi_W(P23);

        HVX_VectorPair T02 = Q6_W_vshuff_VVR(p23lo, p01lo, 64);
        HVX_VectorPair T13 = Q6_W_vshuff_VVR(p23hi, p01hi, 64);

        HVX_Vector t0 = Q6_V_lo_W(T02);   /* tile 0 */
        HVX_Vector t2 = Q6_V_hi_W(T02);   /* tile 2 */
        HVX_Vector t1 = Q6_V_lo_W(T13);   /* tile 1 */
        HVX_Vector t3 = Q6_V_hi_W(T13);   /* tile 3 */

        /* Phase 2 byte transpose: 4 rows × 32 cols -> packed form. */
        t0 = Q6_Vb_vshuff_Vb(Q6_Vb_vshuff_Vb(t0));
        t1 = Q6_Vb_vshuff_Vb(Q6_Vb_vshuff_Vb(t1));
        t2 = Q6_Vb_vshuff_Vb(Q6_Vb_vshuff_Vb(t2));
        t3 = Q6_Vb_vshuff_Vb(Q6_Vb_vshuff_Vb(t3));

        uint8_t *tile0 = tile_base + 0 * tile_stride + kg * 128;
        uint8_t *tile1 = tile_base + 1 * tile_stride + kg * 128;
        uint8_t *tile2 = tile_base + 2 * tile_stride + kg * 128;
        uint8_t *tile3 = tile_base + 3 * tile_stride + kg * 128;
        memcpy(tile0, &t0, sizeof(HVX_Vector));
        memcpy(tile1, &t1, sizeof(HVX_Vector));
        memcpy(tile2, &t2, sizeof(HVX_Vector));
        memcpy(tile3, &t3, sizeof(HVX_Vector));
    }
}
#endif

void pack_wt_v3_hvx_kernel_body(
    const int8_t *w,  /* [K, N] row-major */
    uint8_t *out,     /* [N/32, K/32, 1024] flat */
    int K, int N)
{
    const int K_tiles = K / 32;
    const int N_tiles = N / 32;
#if defined(__hexagon__) && !defined(PACK_WT_V3_SCALAR_FORCE)
    /* Batch 4 nt tiles per row-load if N_tiles % 4 == 0 (standard shapes). */
    if ((N_tiles % 4) == 0) {
        for (int kt = 0; kt < K_tiles; kt++) {
            for (int nt_base = 0; nt_base < N_tiles; nt_base += 4) {
                uint8_t *tile_base = out + (nt_base * K_tiles + kt) * 1024;
                pack_wt_v3_batch4(tile_base, K_tiles, w,
                                   kt * 32, nt_base * 32, N);
            }
        }
        return;
    }
#endif
    for (int nt = 0; nt < N_tiles; nt++) {
        for (int kt = 0; kt < K_tiles; kt++) {
            uint8_t *tile = out + (nt * K_tiles + kt) * 1024;
            pack_one_wt_tile_v3(tile, w, kt * 32, nt * 32, N);
        }
    }
}

/* Phase 3D.1: wu-pointer keyed cache. First inference packs; subsequent
 * inferences detect unchanged wu pointer (STATIC tensor address is stable
 * across executions) and skip repacking entirely. Phase 2 Path B pattern.
 * .bss persists across inference invocations → steady-state cost ≈ 0. */
static const int8_t *g_pack_wt_cached_wu     = NULL;
static int           g_pack_wt_cached_K      = 0;
static int           g_pack_wt_cached_N      = 0;
static uint8_t      *g_pack_wt_cached_out    = NULL;  /* last output tensor */

static uint32_t pack_wt_v3_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs,  const QHPI_Tensor *const *inputs)
{
    (void)handle; (void)num_outputs; (void)num_inputs;

    const int8_t *w   = (const int8_t *)qhpi_tensor_raw_data(inputs[0]);
    uint8_t      *out = (uint8_t *)qhpi_tensor_raw_data(outputs[0]);

    QHPI_Shape ws = qhpi_tensor_shape(inputs[0]);
    const int K = (int)ws.dims[ws.rank - 2];
    const int N = (int)ws.dims[ws.rank - 1];

    /* Cache hit: same wu pointer + shape + output tensor → skip work.
     * Output tensor is NATIVE/persistent so data stays valid across
     * inferences when wu is unchanged. */
    if (w == g_pack_wt_cached_wu
        && K == g_pack_wt_cached_K
        && N == g_pack_wt_cached_N
        && out == g_pack_wt_cached_out) {
        return QHPI_Success;
    }

    pack_wt_v3_hvx_kernel_body(w, out, K, N);

    g_pack_wt_cached_wu  = w;
    g_pack_wt_cached_K   = K;
    g_pack_wt_cached_N   = N;
    g_pack_wt_cached_out = out;
    return QHPI_Success;
}

/* QNN inserts a Cast int8→uint8 (offset +128) on int8 graph inputs, so
 * accept QUInt8 here. TCM_Only so weights are VTCM-resident — avoids DDR
 * cache misses per row read (weights reused across all m_tiles). */
static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};
/* Output must live in VTCM so MatMulV3 can read it via HMX mxmem. */
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};

static QHPI_Kernel_v1 sg_kernels[] = {
    {
        /* .function_name      */ THIS_PKG_NAME_STR "::pack_wt_v3_hvx",
        /* .function           */ pack_wt_v3_kernel,
        /* .resources          */ QHPI_RESOURCE_HVX,
        /* .source_destructive */ false,
        /* .multithreaded      */ false,
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
        /* .name              */ THIS_PKG_NAME_STR "::PackWeightToHmxTileV3",
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

extern "C" void register_pack_wt_v3_op(void)
{
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
