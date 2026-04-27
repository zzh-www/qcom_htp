/*
 * pack_act_crouton_hvx.c — PackActCrouton op.
 *
 * HVX implementation of QNN's `convert_to_crouton_b` byte pack — a
 * standalone backup that does NOT dlsym into libQnnHtpV75Skel.so. Same
 * output bytes (modulo our flat output layout, see below), implemented
 * via two passes of `Q6_Wb_vshuff_VbVbR_128B(Vu,Vv,Rt=-32)` exactly as
 * RE'd in `Agent/forceformat_crouton_re.md` §2.1+§4.2.
 *
 * Tensor contract:
 *   Input  0: uint8 act      [1, 1, M, K]            row-major flat
 *               M % 4 == 0,  K % 128 == 0  (byte path needs K-step 128).
 *   Output 0: uint8 crouton  [1, K/32, M/4, 128]     u8, M*K bytes total.
 *
 * Output layout (flat, contiguous in memory):
 *   Let G       = K / 32          (depth lanes)
 *       H_grps  = M / 4           (height groups, 4 rows each)
 *   For depth lane g  in [0..G), height group h in [0..H_grps):
 *     out[g][h] is a 128-byte block at offset (g*H_grps + h)*128.
 *     The block holds:
 *       row(4h+0)[g*32..g*32+31] | row(4h+1)[same]
 *       | row(4h+2)[same]        | row(4h+3)[same]
 *
 * This matches the natural Crouton "depth-32 lane × 4-row spatial group"
 * layout but with the per-lane scatter offsets collapsed to a contiguous
 * `[G, H_grps, 128]` block layout — the consumer of this op decides how
 * to interpret it (V8's MatMul expects a row-major tile, so this op is
 * NOT a drop-in for pack_act_rm; it produces the *Crouton-byte* layout
 * that ConvLayer_s1.opt natively consumes).
 *
 * Algorithm (per height group h, per K-chunk of 128 cols):
 *   1. vmemu-load 4 input rows × 128 cols into v3..v6.
 *   2. Pass 1:
 *        Wd_AB = vshuff(v4 /*B*/, v3 /*A*/, -32)   -> v14 = lo, v15 = hi
 *        Wd_CD = vshuff(v6 /*D*/, v5 /*C*/, -32)   -> v16 = lo, v17 = hi
 *   3. Pass 2:
 *        Wd_lo = vshuff(v16, v14, -32)             -> v18 = lo (depth 0)
 *                                                  -> v19 = hi (depth 2)
 *        Wd_hi = vshuff(v17, v15, -32)             -> v20 = lo (depth 1)
 *                                                  -> v21 = hi (depth 3)
 *      v18..v21 are 4 Crouton blocks for cols [0..31, 32..63, 64..95, 96..127].
 *   4. Aligned vmem store each block to its (g, h)-indexed offset.
 *
 * Multithreaded=true; we slice along H_grps so each thread owns a band
 * of height groups and writes to disjoint (g, h) cells.
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

/* Set to 1 to compute a scalar reference in parallel and assert
 * bit-exactness against the HVX result inside the kernel. Kept off by
 * default — the reference runs the entire transform a second time, so
 * it doubles the kernel cost.  */
#ifndef DEBUG_PACK_ACT_CROUTON
#define DEBUG_PACK_ACT_CROUTON 0
#endif

/* ----------------------------------------------------------------- */
/* Pure-C reference — also used by non-Hexagon (x86) builds.         */
/* ----------------------------------------------------------------- */

static inline void pack_act_crouton_ref(
    const uint8_t *a,    /* [M, K] row-major */
    uint8_t       *out,  /* [G, H_grps, 128] flat */
    int M, int K,
    uint32_t h_start, uint32_t h_end)
{
    const int G      = K / 32;
    const int H_grps = M / 4;
    for (uint32_t h = h_start; h < h_end; h++) {
        for (int g = 0; g < G; g++) {
            uint8_t *block = out + ((uint32_t)g * H_grps + h) * 128;
            for (int r = 0; r < 4; r++) {
                const uint8_t *src = &a[((int)h * 4 + r) * K + g * 32];
                /* 32 source bytes -> dest[r*32 .. r*32+31] */
                memcpy(&block[r * 32], src, 32);
            }
        }
    }
}

/* ----------------------------------------------------------------- */
/* HVX path: two-pass vshuff(-32), 4 rows × 128 cols at a time.       */
/* ----------------------------------------------------------------- */

#if defined(__hexagon__)
static inline void pack_act_crouton_hvx_band(
    const uint8_t *a,    /* [M, K] row-major */
    uint8_t       *out,  /* [G, H_grps, 128] flat */
    int M, int K,
    uint32_t h_start, uint32_t h_end)
{
    (void)M;
    const int G          = K / 32;
    const int H_grps     = M / 4;
    /* per-lane stride: H_grps * 128 bytes between lane-g cells. */
    const int lane_stride = H_grps * 128;

    for (uint32_t h = h_start; h < h_end; h++) {
        const uint8_t *r0_base = &a[((int)h * 4 + 0) * K];
        const uint8_t *r1_base = &a[((int)h * 4 + 1) * K];
        const uint8_t *r2_base = &a[((int)h * 4 + 2) * K];
        const uint8_t *r3_base = &a[((int)h * 4 + 3) * K];

        /* Each inner iteration consumes 128 cols (4 depth-lanes worth). */
        for (int kk = 0; kk < K; kk += 128) {
            HVX_Vector v3, v4, v5, v6;
            /* Unaligned vmem loads; idiom is `memcpy` of HVX_Vector size. */
            memcpy(&v3, &r0_base[kk], sizeof(HVX_Vector));
            memcpy(&v4, &r1_base[kk], sizeof(HVX_Vector));
            memcpy(&v5, &r2_base[kk], sizeof(HVX_Vector));
            memcpy(&v6, &r3_base[kk], sizeof(HVX_Vector));

            /* Pass 1: byte-stride-32 deinterleave on (rowA,rowB) and (rowC,rowD).
             *   Q6_Wb_vshuff_VbVbR(Vu, Vv, R) -- mnemonic order Vu,Vv to match
             *   `Vdd = vshuff(Vu, Vv, R)` in disasm. Pair: lo = Q6_V_lo_W,
             *   hi = Q6_V_hi_W. */
            HVX_VectorPair WAB = Q6_Wb_vshuff_VbVbR_128B(v4, v3, -32);
            HVX_VectorPair WCD = Q6_Wb_vshuff_VbVbR_128B(v6, v5, -32);
            HVX_Vector v14 = Q6_V_lo_W(WAB);
            HVX_Vector v15 = Q6_V_hi_W(WAB);
            HVX_Vector v16 = Q6_V_lo_W(WCD);
            HVX_Vector v17 = Q6_V_hi_W(WCD);

            /* Pass 2: combine (AB,CD) so all 4 rows of the same depth-lane
             * land in one vector. Per Agent/forceformat_crouton_re.md §4.2:
             *   v18 = blocks for cols  0..31  (depth lane g0)
             *   v19 = blocks for cols 32..63  (depth lane g1)
             *   v20 = blocks for cols 64..95  (depth lane g2)
             *   v21 = blocks for cols 96..127 (depth lane g3) */
            HVX_VectorPair Wlo = Q6_Wb_vshuff_VbVbR_128B(v16, v14, -32);
            HVX_VectorPair Whi = Q6_Wb_vshuff_VbVbR_128B(v17, v15, -32);
            HVX_Vector v18 = Q6_V_lo_W(Wlo);  /* depth lane g0 */
            HVX_Vector v19 = Q6_V_hi_W(Wlo);  /* depth lane g2 */
            HVX_Vector v20 = Q6_V_lo_W(Whi);  /* depth lane g1 */
            HVX_Vector v21 = Q6_V_hi_W(Whi);  /* depth lane g3 */

            /* Map back to absolute depth-lane index in the output. The
             * 128-col chunk covers lanes [kk/32 .. kk/32+3]. The two-pass
             * deinterleave produces them in (g0, g2, g1, g3) source order
             * (cols 0..31 / 64..95 / 32..63 / 96..127); we rearrange via
             * the destination pointer. */
            const int g_base = kk / 32;
            uint8_t *dst_g0 = out + ((g_base + 0) * H_grps + (int)h) * 128;
            uint8_t *dst_g1 = out + ((g_base + 1) * H_grps + (int)h) * 128;
            uint8_t *dst_g2 = out + ((g_base + 2) * H_grps + (int)h) * 128;
            uint8_t *dst_g3 = out + ((g_base + 3) * H_grps + (int)h) * 128;
            (void)lane_stride;

            /* Aligned 128-byte stores: dst pointers all 128-B aligned
             * because H_grps*128 is a multiple of 128 and `out` is
             * VTCM-aligned. */
            *((HVX_Vector *)dst_g0) = v18;
            *((HVX_Vector *)dst_g1) = v20;
            *((HVX_Vector *)dst_g2) = v19;
            *((HVX_Vector *)dst_g3) = v21;
        }
    }
}
#endif  /* __hexagon__ */

/* ----------------------------------------------------------------- */
/* Kernel body — selects HVX or scalar reference path.                */
/* ----------------------------------------------------------------- */

void pack_act_crouton_kernel_body(
    const uint8_t *a,
    uint8_t       *out,
    int M, int K,
    uint32_t h_start, uint32_t h_end)
{
#if defined(__hexagon__)
    pack_act_crouton_hvx_band(a, out, M, K, h_start, h_end);

  #if DEBUG_PACK_ACT_CROUTON
    /* In-place self-check: re-run reference into a scratch buffer and
     * bit-compare. NOTE: this allocates M*K bytes on the stack — keep
     * DEBUG flag off for non-tiny shapes. */
    {
        const int G      = K / 32;
        const int H_grps = M / 4;
        const uint32_t band_bytes = (h_end - h_start) * 128;
        /* Allocate per-lane temp via static buffer (single-thread debug
         * only; gates DEBUG_PACK_ACT_CROUTON to multithreaded=false too). */
        static uint8_t scratch[1 << 20] __attribute__((aligned(128)));
        if ((size_t)G * H_grps * 128 <= sizeof(scratch)) {
            memset(scratch, 0, sizeof(scratch));
            pack_act_crouton_ref(a, scratch, M, K, h_start, h_end);
            for (int g = 0; g < G; g++) {
                const uint8_t *exp = scratch + ((uint32_t)g * H_grps + h_start) * 128;
                const uint8_t *got = out     + ((uint32_t)g * H_grps + h_start) * 128;
                for (uint32_t b = 0; b < band_bytes; b++) {
                    if (exp[b] != got[b]) {
                        /* Mark mismatch: write 0xCC marker into the byte. */
                        ((uint8_t *)out)[0] = 0xCC;
                        return;
                    }
                }
            }
        }
    }
  #endif
#else
    pack_act_crouton_ref(a, out, M, K, h_start, h_end);
#endif
}

/* ----------------------------------------------------------------- */
/* QHPI op shell.                                                     */
/* ----------------------------------------------------------------- */

static uint32_t pack_act_crouton_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs,  const QHPI_Tensor *const *inputs)
{
    (void)num_outputs; (void)num_inputs;

    const uint8_t *a   = (const uint8_t *)qhpi_tensor_raw_data(inputs[0]);
    uint8_t       *out = (uint8_t *)      qhpi_tensor_raw_data(outputs[0]);

    QHPI_Shape as = qhpi_tensor_shape(inputs[0]);
    const int M = (int)as.dims[as.rank - 2];
    const int K = (int)as.dims[as.rank - 1];

    /* Preconditions for byte path (caller ensures via shape_required hook
     * or graph construction). */
    if ((M % 4) != 0)   return QHPI_Failure;
    if ((K % 128) != 0) return QHPI_Failure;

    const int H_grps = M / 4;

    const uint32_t num_slices = qhpi_num_slices(handle);
    const uint32_t slice_idx  = qhpi_slice_number(handle);
    const uint32_t h_start =
        (uint32_t)((uint64_t)H_grps * slice_idx)     / num_slices;
    const uint32_t h_end   =
        (uint32_t)((uint64_t)H_grps * (slice_idx+1)) / num_slices;

    pack_act_crouton_kernel_body(a, out, M, K, h_start, h_end);
    return QHPI_Success;
}

/* Match pack_act_rm_hvx convention: u8 input from DDR/TCM, u8 output
 * resident in TCM (downstream consumer is HMX/HVX in VTCM). */
static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
};

static QHPI_Kernel_v1 sg_kernels[] = {
    {
        THIS_PKG_NAME_STR "::pack_act_crouton_hvx",
        pack_act_crouton_kernel,
        QHPI_RESOURCE_HVX,
        /* source_destructive */ false,
#if DEBUG_PACK_ACT_CROUTON
        /* multithreaded       */ false,  /* scratch buf is shared */
#else
        /* multithreaded       */ true,
#endif
        /* variable_inputs     */ false,
        /* variable_outputs    */ false,
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
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]),
                          sg_ops, THIS_PKG_NAME_STR);
}
