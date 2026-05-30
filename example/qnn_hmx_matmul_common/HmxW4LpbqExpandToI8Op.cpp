/*
 * HmxW4LpbqExpandToI8Op.cpp
 *
 * Custom LPBQ pre-chain op:
 *   weight_lpbq[0]       [1, 1, K/2, N] uint8, native K-pair W4 bytes
 *   per_block_scale[1]   [1, 1, 1, N * (K/32)] uint8
 *       ->
 *   weight_i8[0]         [1, 1, K, N] uint8 carrier, bytes are signed int8
 *                        values packed in the K-major 32x32 HMX W8 stream.
 *
 * This mirrors QNN native LPBQ's expand_block_quant_to_pc_int8_weights stage:
 * compressed signed int4 weights are multiplied by integer per-block scales
 * along K, yielding a per-channel int8 weight payload for the downstream HMX
 * W8 compute op.  The output is intentionally packed for the existing custom
 * HMX W8 kernels; it is not row-major logical [K, N] storage.
 */

#include "HTP/core/qhpi.h"
#include <cstdint>
#include <vector>

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

static inline int8_t hmx_lpbq_int4_twos(uint8_t nibble)
{
    nibble &= 0x0f;
    return static_cast<int8_t>(nibble < 8 ? nibble : nibble - 16);
}

static inline int8_t hmx_lpbq_native_kpair_weight(
    const uint8_t *src,
    uint32_t N,
    uint32_t k,
    uint32_t n)
{
    const uint8_t packed = src[(k >> 1) * N + n];
    return hmx_lpbq_int4_twos((k & 1u) == 0 ? packed : packed >> 4);
}

static inline int8_t hmx_lpbq_saturate_i8(int32_t v)
{
    if (v < -128) return -128;
    if (v > 127) return 127;
    return static_cast<int8_t>(v);
}

static uint32_t hmx_w4_lpbq_expand_to_i8_kernel(
    QHPI_RuntimeHandle *handle,
    uint32_t num_outputs,
    QHPI_Tensor **outputs,
    uint32_t num_inputs,
    const QHPI_Tensor *const *inputs)
{
    (void)handle;
    if (!outputs || num_outputs < 1 || !outputs[0] ||
        !inputs || num_inputs < 2 || !inputs[0] || !inputs[1]) {
        return QHPI_Success;
    }

    const uint8_t *compressed =
        reinterpret_cast<const uint8_t *>(qhpi_tensor_raw_data(inputs[0]));
    const uint8_t *block_scales =
        reinterpret_cast<const uint8_t *>(qhpi_tensor_raw_data(inputs[1]));
    uint8_t *expanded =
        reinterpret_cast<uint8_t *>(qhpi_tensor_raw_data(outputs[0]));
    if (!compressed || !block_scales || !expanded) return QHPI_Success;

    const QHPI_Shape in_shape = qhpi_tensor_shape(inputs[0]);
    const QHPI_Shape out_shape = qhpi_tensor_shape(outputs[0]);
    if (in_shape.rank != 4 || out_shape.rank != 4) return QHPI_Success;
    if (in_shape.dims[0] != 1 || in_shape.dims[1] != 1 ||
        out_shape.dims[0] != 1 || out_shape.dims[1] != 1) {
        return QHPI_Success;
    }

    const uint32_t K = in_shape.dims[2] * 2u;
    const uint32_t N = in_shape.dims[3];
    if (out_shape.dims[2] != K || out_shape.dims[3] != N) return QHPI_Success;
    if ((K % 32u) != 0 || (N % 32u) != 0) return QHPI_Success;

    const uint32_t K_t = K / 32u;
    const uint32_t N_t = N / 32u;
    for (uint32_t kt = 0; kt < K_t; ++kt) {
        const uint32_t k_base = kt * 32u;
        for (uint32_t nt = 0; nt < N_t; ++nt) {
            const uint32_t n_base = nt * 32u;
            uint8_t *tile = expanded + (kt * N_t + nt) * 1024u;
            for (uint32_t r = 0; r < 32u; ++r) {
                const uint32_t k = k_base + r;
                for (uint32_t c = 0; c < 32u; ++c) {
                    const uint32_t n = n_base + c;
                    const int32_t q4 =
                        hmx_lpbq_native_kpair_weight(compressed, N, k, n);
                    const int32_t scale = block_scales[n * K_t + kt];
                    const int8_t q8 = hmx_lpbq_saturate_i8(q4 * scale);
                    const uint32_t dst = (r / 4u) * 128u + c * 4u + (r & 3u);
                    tile[dst] = static_cast<uint8_t>(q8);
                }
            }
        }
    }
    return QHPI_Success;
}

static const QHPI_Op *hmx_w4_lpbq_expand_rewrite(const QHPI_Op *op)
{
    if (!op || qhpi_op_num_inputs(op) < 2 || qhpi_op_num_outputs(op) < 1) {
        return op;
    }
    const QHPI_OpRef weight_ref = qhpi_op_input(op, 0);
    const QHPI_OpRef scale_ref = qhpi_op_input(op, 1);
    if (!weight_ref.op || !scale_ref.op ||
        !qhpi_op_is_constant(weight_ref.op) ||
        !qhpi_op_is_constant(scale_ref.op)) {
        return op;
    }

    const QHPI_OutputDef weight_def = qhpi_op_output(weight_ref.op, weight_ref.output_number);
    const QHPI_OutputDef output_def = qhpi_op_output(op, 0);
    if (weight_def.shape.rank != 4 || output_def.shape.rank != 4) return op;
    const uint32_t K = output_def.shape.dims[2];
    const uint32_t N = output_def.shape.dims[3];
    if (weight_def.shape.dims[2] * 2u != K || weight_def.shape.dims[3] != N ||
        (K % 32u) != 0 || (N % 32u) != 0) {
        return op;
    }

    const uint8_t *compressed =
        reinterpret_cast<const uint8_t *>(qhpi_op_constant_data(weight_ref.op));
    const uint8_t *block_scales =
        reinterpret_cast<const uint8_t *>(qhpi_op_constant_data(scale_ref.op));
    if (!compressed || !block_scales) return op;

    const uint32_t K_t = K / 32u;
    const uint32_t N_t = N / 32u;
    std::vector<uint8_t> expanded(static_cast<size_t>(K) * N);
    for (uint32_t kt = 0; kt < K_t; ++kt) {
        const uint32_t k_base = kt * 32u;
        for (uint32_t nt = 0; nt < N_t; ++nt) {
            const uint32_t n_base = nt * 32u;
            uint8_t *tile = expanded.data() + (kt * N_t + nt) * 1024u;
            for (uint32_t r = 0; r < 32u; ++r) {
                const uint32_t k = k_base + r;
                for (uint32_t c = 0; c < 32u; ++c) {
                    const uint32_t n = n_base + c;
                    const int32_t q4 =
                        hmx_lpbq_native_kpair_weight(compressed, N, k, n);
                    const int32_t scale = block_scales[n * K_t + kt];
                    const int8_t q8 = hmx_lpbq_saturate_i8(q4 * scale);
                    const uint32_t dst = (r / 4u) * 128u + c * 4u + (r & 3u);
                    tile[dst] = static_cast<uint8_t>(q8);
                }
            }
        }
    }

    return qhpi_op_create_constant(
        op,
        &output_def,
        static_cast<uint32_t>(expanded.size()),
        expanded.data());
}

static QHPI_Tensor_Signature_v1 expand_sig_inputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM},
};

static QHPI_Tensor_Signature_v1 expand_sig_outputs[] = {
#if defined(HMX_LPBQ_EXPAND_SIGNED_OUTPUT)
    {QHPI_QInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
#else
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only},
#endif
};

static float hmx_w4_lpbq_expand_cost(uint32_t num_inputs, const QHPI_Tensor *const *inputs)
{
    if (!inputs || num_inputs < 1 || !inputs[0]) return 1.0f;
    const QHPI_Shape s = qhpi_tensor_shape(inputs[0]);
    if (s.rank != 4) return 1.0f;
    return static_cast<float>(s.dims[2] * 2u * s.dims[3]);
}

static QHPI_Shape hmx_w4_lpbq_expand_shape_required(const QHPI_Op *op)
{
    (void)op;
    QHPI_Shape req = {0};
    req.rank = 4;
    req.dims[0] = 1;
    req.dims[1] = 1;
    req.dims[2] = 32;
    req.dims[3] = 32;
    return req;
}

static QHPI_Shape hmx_w4_lpbq_expand_shape_legalized(
    const QHPI_Op *op,
    const QHPI_Shape *proposed)
{
    (void)op;
    QHPI_Shape s = *proposed;
    if (s.rank >= 4) {
        s.dims[0] = 1;
        s.dims[1] = 1;
        if (s.dims[2] < 32) s.dims[2] = 32;
        if (s.dims[2] % 32) s.dims[2] = ((s.dims[2] + 31) / 32) * 32;
        if (s.dims[3] < 32) s.dims[3] = 32;
        if (s.dims[3] % 32) s.dims[3] = ((s.dims[3] + 31) / 32) * 32;
    }
    return s;
}

static QHPI_Kernel_v1 expand_kernels[] = {
    {
        THIS_PKG_NAME_STR "::hmx_w4_lpbq_expand_to_i8_kernel",
        hmx_w4_lpbq_expand_to_i8_kernel,
        QHPI_RESOURCE_HVX,
        false,
        false,
        false, false,
        2, expand_sig_inputs,
        1, expand_sig_outputs,
        hmx_w4_lpbq_expand_cost,
        0,
        0,
        nullptr,
        nullptr,
        nullptr,
    },
};

static QHPI_OpInfo_v1 expand_ops[] = {
    {
        THIS_PKG_NAME_STR "::HmxW4LpbqExpandToI8",
        1, expand_kernels,
        hmx_w4_lpbq_expand_rewrite,
        hmx_w4_lpbq_expand_shape_required,
        hmx_w4_lpbq_expand_shape_legalized,
        0,
        nullptr,
        nullptr,
    },
};

extern "C" void register_hmx_w4_lpbq_expand_to_i8_op(void)
{
    qhpi_register_ops_v1(1, expand_ops, THIS_PKG_NAME_STR);
}
