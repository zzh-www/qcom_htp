/*
 * GdnSolveOp.cpp — QNN/QHPI custom op "GdnSolve": T = (I - A)^-1 for a strictly-lower A.
 *
 *   input[0]  A : QInt16  [B,H,C,C]  (per-tensor scale sA = quant_parameters.stepsize)
 *   output[0] T : QInt16  [B,H,C,C]  (per-tensor scale sT)
 *
 * Why a custom op: QNN has no native matrix-inverse, so the pure-graph route fakes it with an
 * int8-matmul Neumann chain (iterated Ap@Ap, requant every step) → T's value degrades to ~4.4e-3.
 * This op does the inverse the KDA way (fla/ops/kda/chunk_intra.py): per-BL(=16) diagonal-block
 * forward substitution (int16 codes, int32 accumulation, requant once per row) + block-triangular
 * merge.  Internal arithmetic is int16/int32; we control where requant happens (NOT every matmul),
 * so T comes out near-exact (host bit-model scripts/gdn_solve_int16_model.py: T relerr 3.6e-5,
 * int32 accumulator peak 5.4e8 < 2^31).  Bit-faithful to that golden + gdn_solve_ref.c.
 *
 * Scope: solve only (A->T).  Upstream A build + downstream U/W consume stay in the QNN graph;
 * downstream casts T to int8 as needed.
 */
#include "HTP/core/qhpi.h"
#include "gdn_solve_core.h"   /* scalar gdn_solve_head_q() — host-validated golden + x86 fallback */

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

#define GDN_MAX_SLICES 8     /* per-slice scratch slots (>= HVX threads); avoids races + DSP stack */

#if defined(__hexagon__)
#include <hexagon_types.h>
#include <hexagon_protos.h>
/*
 * HVX path: full C=64 forward substitution T[i,:] = e_i + sum_{k<i} A[i,k]*T[k,:], vectorized over the
 * 64 columns (2 IEEE-fp32 HVX vectors per row), int accumulation in fp32 (the precision is set by the
 * int16-quantized A and int16 output T — the accumulator type is immaterial for these bounded values,
 * and fp32 avoids the int16 even/odd width-double shuffle).  One Tf scratch per slice (thread).
 */
static float __attribute__((aligned(128))) g_Tf[GDN_MAX_SLICES][GDN_CMAX * GDN_CMAX];
static float __attribute__((aligned(128))) g_Af[GDN_MAX_SLICES][GDN_CMAX * GDN_CMAX];

static inline HVX_Vector splat_f(float f) { int w; __builtin_memcpy(&w, &f, 4); return Q6_V_vsplat_R(w); }
static inline int rnd_i(float x) { return (int)(x >= 0.0f ? x + 0.5f : x - 0.5f); }  /* libcall-free round */

/* HVX kernel: AXPY over 64 cols (2 fp32 vecs/row, qf32) with 2 accumulators to break the acc latency
 * chain.  Dequant + output quant are scalar but libcall-free (no lroundf).  No HVX pack/unpack (those
 * even/odd-split and scramble column order) — keeps the layout trivially correct. */
static void gdn_solve_head_hvx(const uint16_t *Au, int C, int zpA, float sA,
                               float sT, int zpT, uint16_t *Tu, int slot) {
    float *Tf = g_Tf[slot];
    float *Af = g_Af[slot];
    const int NV = (C + 31) / 32;
    for (int i = 0; i < C*C; ++i) Af[i] = (float)((int)Au[i] - zpA) * sA;   /* dequant A (libcall-free) */

    for (int i = 0; i < C; ++i) {
        HVX_Vector a0[2], a1[2];                          /* 2 accumulators per col-vector (ILP) */
        for (int v = 0; v < NV; ++v) { a0[v] = Q6_V_vzero(); a1[v] = Q6_V_vzero(); }
        int k = 0;
        for (; k + 1 < i; k += 2) {
            HVX_Vector s0 = splat_f(Af[i*C + k]), s1 = splat_f(Af[i*C + k + 1]);
            for (int v = 0; v < NV; ++v) {
                a0[v] = Q6_Vqf32_vadd_Vqf32Vqf32(a0[v], Q6_Vqf32_vmpy_VsfVsf(s0, *(HVX_Vector *)(Tf + k*C + v*32)));
                a1[v] = Q6_Vqf32_vadd_Vqf32Vqf32(a1[v], Q6_Vqf32_vmpy_VsfVsf(s1, *(HVX_Vector *)(Tf + (k+1)*C + v*32)));
            }
        }
        for (; k < i; ++k) {
            HVX_Vector s0 = splat_f(Af[i*C + k]);
            for (int v = 0; v < NV; ++v)
                a0[v] = Q6_Vqf32_vadd_Vqf32Vqf32(a0[v], Q6_Vqf32_vmpy_VsfVsf(s0, *(HVX_Vector *)(Tf + k*C + v*32)));
        }
        for (int v = 0; v < NV; ++v)
            *(HVX_Vector *)(Tf + i*C + v*32) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(a0[v], a1[v]));
        Tf[i*C + i] += 1.0f;                              /* + e_i */
    }

    const float invsT = 1.0f / sT;                        /* output: uint16-midpoint codes @ sT */
    for (int r = 0; r < C; ++r)
        for (int c = 0; c < C; ++c) {
            long q = ((c > r) ? 0 : rnd_i(Tf[r*C + c] * invsT)) + zpT;
            if (q < 0) q = 0; if (q > 65535) q = 65535;
            Tu[r*C + c] = (uint16_t)q;
        }
}
#endif  /* __hexagon__ */

static void gdn_solve_head_scalar(const uint16_t *Au, int C, int zpA, float sA,
                                  float sT, int zpT, uint16_t *Tu, int slot) {
    static int16_t g_As[GDN_MAX_SLICES][GDN_CMAX * GDN_CMAX];
    static int16_t g_Ts[GDN_MAX_SLICES][GDN_CMAX * GDN_CMAX];
    int16_t *As = g_As[slot], *Ts = g_Ts[slot];
    for (int i = 0; i < C*C; ++i) As[i] = (int16_t)((int32_t)Au[i] - zpA);
    gdn_solve_head_q<int16_t>(As, C, sA, sT, 32767.0f, Ts);
    for (int i = 0; i < C*C; ++i) Tu[i] = (uint16_t)((int32_t)Ts[i] + zpT);
}

static uint32_t gdn_solve_kernel(
        QHPI_RuntimeHandle *handle,
        uint32_t num_outputs, QHPI_Tensor **outputs,
        uint32_t num_inputs, const QHPI_Tensor *const *inputs) {
    (void)num_outputs; (void)num_inputs;
    if (!outputs || !outputs[0] || !inputs || !inputs[0]) return QHPI_Success;

    /* A and T are int16 acts carried as uint16-midpoint on HTP (zero_offset ~= 32768). */
    const uint16_t *Au = (const uint16_t *)qhpi_tensor_raw_data(inputs[0]);
    uint16_t *Tu = (uint16_t *)qhpi_tensor_raw_data(outputs[0]);
    if (!Au || !Tu) return QHPI_Success;

    const QHPI_Quant_Parameters qa = qhpi_tensor_quant_parameters(inputs[0]);
    const QHPI_Quant_Parameters qt = qhpi_tensor_quant_parameters(outputs[0]);
    const float sA = qa.stepsize, sT = qt.stepsize;
    const int32_t zpA = qa.zero_offset, zpT = qt.zero_offset;

    QHPI_Shape s = qhpi_tensor_shape(inputs[0]);   /* [B,H,C,C] */
    int C = (s.rank >= 1) ? (int)s.dims[s.rank - 1] : GDN_CMAX;
    if (C <= 0 || C > GDN_CMAX || (C % GDN_BL)) return QHPI_Success;
    uint32_t heads = 1;
    for (uint32_t d = 0; d + 2 < s.rank; ++d) heads *= s.dims[d];

    /* self-slice across HVX threads: this thread handles heads [h0, h1) */
    uint32_t ns = qhpi_num_slices(handle), sl = qhpi_slice_number(handle);
    if (ns == 0) ns = 1;
    if (sl >= GDN_MAX_SLICES) sl = GDN_MAX_SLICES - 1;
    uint32_t h0 = (uint64_t)heads * sl / ns, h1 = (uint64_t)heads * (sl + 1) / ns;
    for (uint32_t h = h0; h < h1; ++h) {
        const uint16_t *Ah = Au + (size_t)h*C*C;
        uint16_t *Th = Tu + (size_t)h*C*C;
#if defined(__hexagon__)
        gdn_solve_head_hvx(Ah, C, zpA, sA, sT, zpT, Th, (int)sl);
#else
        gdn_solve_head_scalar(Ah, C, zpA, sA, sT, zpT, Th, (int)sl);
#endif
    }
    return QHPI_Success;
}

static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QUInt16, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};

static float gdn_solve_cost(uint32_t num_inputs, const QHPI_Tensor *const *inputs) {
    if (!inputs || num_inputs < 1 || !inputs[0]) return 1.0f;
    QHPI_Shape s = qhpi_tensor_shape(inputs[0]);
    float n = 1.0f; for (uint32_t d = 0; d < s.rank; ++d) n *= (float)s.dims[d];
    return n;                                       /* ~ elements; refine after HVX */
}

static QHPI_Kernel_v1 sg_kernels[] = {
    {
        THIS_PKG_NAME_STR "::gdn_solve_kernel",
        gdn_solve_kernel,
        QHPI_RESOURCE_HVX,                        /* HVX-vectorized */
        false, true,  false, false,                /* multithreaded=true (engages when tiler can split, e.g. isolated) *
                                                   * ~11M; QNN won't self-slice a custom op whose output feeds the
                                                   * non-tiled Transpose/matmul). Real multi-HVX needs manual qurt. */
        1, sig_inputs,
        1, sig_outputs,
        gdn_solve_cost,
        0,
        0, nullptr, nullptr,
        nullptr,
    },
};

/* Tiling: opt into the central tiler so GdnSolve is split over the head dim and its tile-ops run on
 * multiple HVX threads.  C,C must NOT tile (each tile is a full 64x64 solve over a head-subset). */
static QHPI_Shape gdn_shape_required(const QHPI_Op *op) {
    (void)op;
    QHPI_Shape req; req.rank = 4;
    req.dims[0] = 1;                 /* B */
    req.dims[1] = 8;                 /* H: force tiles of 8 heads -> 4 parallel tile-ops */
    req.dims[2] = QHPI_DO_NOT_TILE;  /* C rows: keep whole */
    req.dims[3] = QHPI_DO_NOT_TILE;  /* C cols: keep whole */
    return req;
}

static const QHPI_Op *gdn_build_tile(const QHPI_Op *op, const QHPI_Shape *out_start,
                                     const QHPI_Shape *out_extent) {
    /* input A and output T share shape; the tile's input slice == output slice */
    QHPI_OpRef in = qhpi_op_input(op, 0);
    QHPI_Shape s = *out_start, e = *out_extent;
    QHPI_OpRef in_slice = qhpi_op_slice(in, &s, &e);
    QHPI_OpRef inputs[] = { in_slice };
    QHPI_OutputDef o0 = qhpi_op_output(op, 0);
    QHPI_OutputDef outputs[] = { { o0.type, o0.quant_parameters, *out_extent } };
    return qhpi_op_create(op, qhpi_op_name(op), 1, inputs, 1, outputs);
}

static QHPI_OpInfo_v1 sg_ops[] = {
    {
        THIS_PKG_NAME_STR "::GdnSolve",
        1, sg_kernels,
        nullptr,                 /* early_rewrite */
        gdn_shape_required,      /* shape_required */
        nullptr,                 /* shape_legalized */
        0,                       /* tile_output index */
        gdn_build_tile,          /* build_tile */
        nullptr,                 /* late_rewrite */
    },
};

extern "C" void register_gdn_solve_op(void) {
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
