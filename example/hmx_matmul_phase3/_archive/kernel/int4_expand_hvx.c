/*
 * int4_expand_hvx.c — Int4Expand op (Phase 3B, Path W).
 *
 * Expands a uint8-packed int4 weight tensor (2 nibbles per byte,
 * little-nibble first i.e. low-nibble is logical column 2*k, high-nibble
 * is logical column 2*k+1) into an int8 tensor with sign-extended nibbles.
 *
 *   sign_extend(n) = (n < 8) ? n : n - 16   for n in [0, 15]
 *
 * Tensor contract:
 *   Input  0:  uint8 packed   [1, 1, K, N/2]   Flat4 + Direct
 *   Output 0:  int8  expanded [1, 1, K, N]     Flat4 + Direct
 *
 * HVX algorithm: given a byte b with lo=b&0xF, hi=(b>>4)&0xF, the two
 * output bytes should be sign_extend(lo), sign_extend(hi). We produce
 * this by:
 *   1. v_lo = (v_in << 4) (byte shift into HIGH nibble) then arith-shift-
 *      right by 4 via `Q6_Vb_vlut32_VbVbI`-free alternative: mask low
 *      nibble, compare >= 8, conditionally subtract 16.
 *   2. v_hi = (v_in >> 4) (byte-wise logical shift) also sign-extended.
 *   3. vshuffoe_b interleaves {v_lo, v_hi} to produce the expanded output
 *      where lo bytes occupy even positions (col 2*k) and hi bytes occupy
 *      odd positions (col 2*k+1).
 *
 * Implementation notes:
 *  - HVX doesn't have a direct byte-wise arithmetic shift-right. We fake
 *    sign-extend of a 4-bit value by:
 *      extracted_nibble = byte & 0x0F        (0..15)
 *      sign_bit_set     = extracted_nibble >= 8  (i.e. bit3 set)
 *      signed_value     = extracted_nibble - (sign_bit_set ? 16 : 0)
 *    Using `Q6_Vb_vsub_VbVb` with a mask built from
 *    `Q6_Q_vcmp_gt_VubVub` or the bit-test equivalent.
 *  - We do byte-wise splat of constants via `Q6_V_vsplat_R` with a byte
 *    pattern (0x0F0F0F0F, 0x10101010, etc).
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

#ifdef __hexagon__
/* Sign-extend one HVX_Vector whose bytes are in [0, 15] (nibbles). */
static inline HVX_Vector sext_nibble_v(HVX_Vector v_nibble)
{
    /* mask = (v_nibble >= 8) ? 0xFF : 0x00 per byte */
    const HVX_Vector v_eight = Q6_V_vsplat_R(0x08080808);
    const HVX_Vector v_sixteen = Q6_V_vsplat_R(0x10101010);
    HVX_VectorPred q_ge8 = Q6_Q_vcmp_gt_VubVub(v_nibble, Q6_V_vsplat_R(0x07070707));
    /* signed = nibble - (q_ge8 ? 16 : 0) == nibble - vmux(q_ge8, 16, 0) */
    HVX_Vector v_sub = Q6_V_vmux_QVV(q_ge8, v_sixteen, Q6_V_vzero());
    (void)v_eight;
    return Q6_Vb_vsub_VbVb(v_nibble, v_sub);
}

static inline void int4_expand_one_vec(
    int8_t *__restrict__ out_bytes,   /* 256 bytes (two vectors) */
    const uint8_t *__restrict__ in_bytes) /* 128 bytes (one vector) */
{
    const HVX_Vector v_0F = Q6_V_vsplat_R(0x0F0F0F0F);
    HVX_Vector v_in;
    memcpy(&v_in, in_bytes, sizeof(HVX_Vector));

    /* Low nibbles: v_lo = v_in & 0x0F */
    HVX_Vector v_lo_nib = Q6_V_vand_VV(v_in, v_0F);
    /* High nibbles: v_hi = (v_in >> 4) & 0x0F — use halfword shift then mask.
     * Q6_Vuh_vlsr_VuhR shifts halfword lanes right logically; the byte 0xAB
     * inside a halfword becomes 0x0AB0 >> 4 = 0x00AB in the upper half of
     * the halfword. But because bytes are packed 2 per halfword, the shift
     * also pulls bits from the neighboring byte's low nibble. We instead
     * use a word shift + mask, which is equally wrong for the same reason.
     *
     * Correct portable approach: use Q6_Vb_vlsr_VbR — but there's no such
     * single-instruction byte shift. Workaround: use Q6_Vh_vlsr_VhR on the
     * full vector then mask 0x0F0F. This works because:
     *   Let halfword = [byte_lo | byte_hi] (bits 0..7 = lo, 8..15 = hi).
     *   halfword >> 4 = bits [4..15] of halfword, bits 0..3 of new lo come
     *   from bits 4..7 of old lo (= original high nibble of byte_lo). OK.
     *   But the new high byte of this halfword contains bits 12..15 of old
     *   halfword (= high nibble of byte_hi) in its low 4 bits, and bits
     *   0..3 of the new byte_hi come from bits 8..11 of old halfword (=
     *   low nibble of byte_hi). So after masking with 0x0F0F we get:
     *     new_byte_lo = high nibble of old byte_lo     ✓
     *     new_byte_hi = low nibble of old byte_hi      ✗ (wrong!)
     * So this does NOT give us per-byte >> 4.
     *
     * Instead, do what Phase 2 u4×a8 does — apply TWO halfword-lane ops:
     * (A) mask out everything but the high nibble of each byte via
     *     v & 0xF0F0, then shift halfword right by 4. This leaves the hi
     *     nibble of each byte in the low 4 bits of that byte, with garbage
     *     in the high 4 bits of the odd byte. Then mask 0x0F0F.
     * Actually even simpler: shift whole vector right by 4 as halfwords,
     * then mask with 0x0F0F — but per the analysis above, the hi byte's
     * low nibble was pulled from the neighbor. The clean solution: use
     * vshuffoe on the bit pattern to separate before shifting. We instead
     * do the arithmetic:
     *   v_hi = (Q6_Vuh_vlsr_VuhR(Q6_V_vand_VV(v_in, 0xF0F0F0F0), 4))
     *          & 0x0F0F0F0F
     * Because we first masked low nibbles to zero, the halfword right-
     * shift now has zeros in the low-nibble positions, and the hi-nibble
     * of each byte ends up correctly in the low-nibble of that byte.
     */
    const HVX_Vector v_F0 = Q6_V_vsplat_R(0xF0F0F0F0);
    HVX_Vector v_hi_only = Q6_V_vand_VV(v_in, v_F0);
    HVX_Vector v_hi_shifted = Q6_Vuh_vlsr_VuhR(v_hi_only, 4);
    HVX_Vector v_hi_nib = Q6_V_vand_VV(v_hi_shifted, v_0F);

    HVX_Vector v_lo_sext = sext_nibble_v(v_lo_nib);
    HVX_Vector v_hi_sext = sext_nibble_v(v_hi_nib);

    /* Interleave: output column 2k = lo nibble of byte k, column 2k+1 = hi
     * nibble of byte k. Q6_W_vshuff_VVR zips two vectors lane-wise; we
     * want byte-zip so use Q6_Vb_vshuffe_Vb + Q6_Vb_vshuffo_Vb pairing.
     *
     * Actually the semantically-correct single intrinsic is
     *   HVX_VectorPair Q6_Wb_vshuffoe_VbVb(HVX_Vector Vu, HVX_Vector Vv)
     * which produces {shuffle_even, shuffle_odd} as a pair. For our case
     * we want an interleave producing:
     *   low-half:  lo[0], hi[0], lo[1], hi[1], ... (first 128 bytes)
     *   high-half: lo[64], hi[64], ..., lo[127], hi[127]
     * That's exactly `Q6_W_vshuff_VVR(v_hi, v_lo, -1)` or we can use
     * `Q6_Wb_vshuffoe_VbVb` which handily returns the pair we need.
     *
     * The intrinsic Q6_Wb_vshuffoe_VbVb(Vu, Vv) semantics:
     *   lo-vector = even-position bytes of zip(Vu, Vv) = Vv bytes
     *   hi-vector = odd-position bytes of zip(Vu, Vv)  = Vu bytes
     * which means to interleave lo_nib (even cols) with hi_nib (odd cols)
     * we call Q6_W_vshuff_VVR(hi_sext, lo_sext, 1).
     *
     * Simpler and portable: use the generic vshuff intrinsic
     *   Q6_W_vshuff_VVR(v_hi_sext, v_lo_sext, -1)  — scale -1 selects
     * byte-granularity shuffle: result.b[2i] = v_lo_sext.b[i], result.b[2i+1]
     * = v_hi_sext.b[i], spanning both halves of the pair (256 bytes total).
     */
    HVX_VectorPair vp = Q6_W_vshuff_VVR(v_hi_sext, v_lo_sext, -1);
    HVX_Vector v_out_lo = Q6_V_lo_W(vp);
    HVX_Vector v_out_hi = Q6_V_hi_W(vp);
    memcpy(out_bytes,       &v_out_lo, sizeof(HVX_Vector));
    memcpy(out_bytes + 128, &v_out_hi, sizeof(HVX_Vector));
}
#endif /* __hexagon__ */

/* Public entry for sim harness. Processes full [K, N/2] → [K, N] tensor.
 * Handles tail elements scalar if N/2 % 128 != 0. */
void int4_expand_hvx_kernel_body(
    const uint8_t *in,   /* [K, N/2] */
    int8_t *out,         /* [K, N]  */
    int K, int N)
{
    const int Nh = N / 2;  /* packed columns */
    const int total_bytes = K * Nh;

#ifdef __hexagon__
    int i = 0;
    /* Row-wise: since K rows of Nh bytes each may have Nh not a multiple
     * of 128, we process per-row HVX chunks with scalar tail per row. */
    for (int k = 0; k < K; k++) {
        const uint8_t *src = &in[k * Nh];
        int8_t *dst = &out[k * N];
        int j = 0;
        for (; j + 128 <= Nh; j += 128) {
            int4_expand_one_vec(dst + 2 * j, src + j);
        }
        for (; j < Nh; j++) {
            uint8_t b = src[j];
            int lo = b & 0x0F;
            int hi = (b >> 4) & 0x0F;
            dst[2 * j]     = (int8_t)(lo >= 8 ? lo - 16 : lo);
            dst[2 * j + 1] = (int8_t)(hi >= 8 ? hi - 16 : hi);
        }
    }
    (void)i; (void)total_bytes;
#else
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < Nh; j++) {
            uint8_t b = in[k * Nh + j];
            int lo = b & 0x0F;
            int hi = (b >> 4) & 0x0F;
            out[k * N + 2 * j]     = (int8_t)(lo >= 8 ? lo - 16 : lo);
            out[k * N + 2 * j + 1] = (int8_t)(hi >= 8 ? hi - 16 : hi);
        }
    }
#endif
}

/* -------------------------------------------------------------------
 * QHPI kernel entry.
 * ------------------------------------------------------------------- */
static uint32_t int4_expand_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs, QHPI_Tensor **outputs,
    uint32_t num_inputs,  const QHPI_Tensor *const *inputs)
{
    (void)handle; (void)num_outputs; (void)num_inputs;

    const uint8_t *in  = (const uint8_t *)qhpi_tensor_raw_data(inputs[0]);
    int8_t        *out = (int8_t *)qhpi_tensor_raw_data(outputs[0]);

    QHPI_Shape os = qhpi_tensor_shape(outputs[0]);
    const int K = (int)os.dims[os.rank - 2];
    const int N = (int)os.dims[os.rank - 1];

    int4_expand_hvx_kernel_body(in, out, K, N);
    return QHPI_Success;
}

/* -------------------------------------------------------------------
 * QHPI registration.
 * ------------------------------------------------------------------- */
static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};

static QHPI_Kernel_v1 sg_kernels[] = {
    {
        /* .function_name      */ THIS_PKG_NAME_STR "::int4_expand_hvx",
        /* .function           */ int4_expand_kernel,
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
        /* .name              */ THIS_PKG_NAME_STR "::Int4Expand",
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

extern "C" void register_int4_expand_op(void)
{
    qhpi_register_ops_v1(sizeof(sg_ops) / sizeof(sg_ops[0]), sg_ops, THIS_PKG_NAME_STR);
}
