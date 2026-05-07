// Converter-side shape/type inference for QnnHmxMatMulW4A16Package.

#include <cstdint>
#include "QnnOpPackage.h"
#include "QnnTypes.h"

#ifndef EXPORT_API
#define EXPORT_API __attribute__((visibility("default")))
#endif

static inline uint32_t *tensor_dims(Qnn_Tensor_t *t)
{
    switch (t->version) {
    case QNN_TENSOR_VERSION_1: return t->v1.dimensions;
    case QNN_TENSOR_VERSION_2: return t->v2.dimensions;
    default: return nullptr;
    }
}

static inline uint32_t tensor_rank(const Qnn_Tensor_t *t)
{
    switch (t->version) {
    case QNN_TENSOR_VERSION_1: return t->v1.rank;
    case QNN_TENSOR_VERSION_2: return t->v2.rank;
    default: return 0;
    }
}

static inline void set_tensor_dtype(Qnn_Tensor_t *t, Qnn_DataType_t dt)
{
    switch (t->version) {
    case QNN_TENSOR_VERSION_1: t->v1.dataType = dt; break;
    case QNN_TENSOR_VERSION_2: t->v2.dataType = dt; break;
    default: break;
    }
}

extern "C" {

EXPORT_API Qnn_ErrorHandle_t HmxU16I4ToU16MatMulShapeInference(Qnn_OpConfig_t *op)
{
    if (!op || op->v1.numOfInputs < 4 || op->v1.numOfOutputs < 1) {
        return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    }
    if (tensor_rank(&op->v1.inputTensors[1]) != 4 ||
        tensor_rank(&op->v1.inputTensors[2]) != 4) {
        return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    }

    uint32_t *w = tensor_dims(&op->v1.inputTensors[1]);
    uint32_t *a = tensor_dims(&op->v1.inputTensors[2]);
    uint32_t *o = tensor_dims(&op->v1.outputTensors[0]);
    if (!w || !a || !o) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;

    o[0] = 1;
    o[1] = a[1];
    o[2] = a[2];
    // Native W4A16 follows QNN Conv1x1's packed sidecar shape [1,1,K/2,N].
    // Diagnostic graphs may also feed unpacked [1,1,K,N] W4 code bytes or the
    // older [1,1,K,N/2] carrier.  Keep all forms accepted for layout probes.
    if (w[2] == a[3]) {
        o[3] = w[3];
    } else if (w[2] * 2 == a[3]) {
        o[3] = w[3];
    } else {
        o[3] = w[3] * 2;
    }
    return QNN_SUCCESS;
}

EXPORT_API Qnn_ErrorHandle_t HmxU16I4ToU16MatMulDataTypeInference(Qnn_OpConfig_t *op)
{
    if (!op || op->v1.numOfOutputs < 1) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    if (op->v1.numOfInputs >= 4) {
        set_tensor_dtype(&op->v1.inputTensors[0], QNN_DATATYPE_SFIXED_POINT_32);
        set_tensor_dtype(&op->v1.inputTensors[1], QNN_DATATYPE_SFIXED_POINT_8);
        set_tensor_dtype(&op->v1.inputTensors[2], QNN_DATATYPE_UFIXED_POINT_16);
        set_tensor_dtype(&op->v1.inputTensors[3], QNN_DATATYPE_INT_32);
    }
    set_tensor_dtype(&op->v1.outputTensors[0], QNN_DATATYPE_UFIXED_POINT_16);
    return QNN_SUCCESS;
}

EXPORT_API Qnn_ErrorHandle_t HmxW4A16TensorDumpShapeInference(Qnn_OpConfig_t *op)
{
    if (!op || op->v1.numOfInputs < 1 || op->v1.numOfOutputs < 1) {
        return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    }
    const uint32_t rank = tensor_rank(&op->v1.inputTensors[0]);
    if (rank == 0 || rank > 8) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    uint32_t *in = tensor_dims(&op->v1.inputTensors[0]);
    uint32_t *out = tensor_dims(&op->v1.outputTensors[0]);
    if (!in || !out) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    for (uint32_t i = 0; i < rank; ++i) out[i] = in[i];
    return QNN_SUCCESS;
}

EXPORT_API Qnn_ErrorHandle_t HmxW4A16TensorDumpDataTypeInference(Qnn_OpConfig_t *op)
{
    if (!op || op->v1.numOfInputs < 1 || op->v1.numOfOutputs < 1) {
        return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    }
    set_tensor_dtype(&op->v1.inputTensors[0], QNN_DATATYPE_UFIXED_POINT_16);
    set_tensor_dtype(&op->v1.outputTensors[0], QNN_DATATYPE_UFIXED_POINT_16);
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t (*HmxU16I4ToU16MatMulOutputInfoInferencePtr)(Qnn_OpConfig_t *) =
    &HmxU16I4ToU16MatMulShapeInference;
Qnn_ErrorHandle_t (*HmxU16I4ToU16MatMulDataTypeInferencePtr)(Qnn_OpConfig_t *) =
    &HmxU16I4ToU16MatMulDataTypeInference;
Qnn_ErrorHandle_t (*HmxW4A16TensorDumpOutputInfoInferencePtr)(Qnn_OpConfig_t *) =
    &HmxW4A16TensorDumpShapeInference;
Qnn_ErrorHandle_t (*HmxW4A16TensorDumpDataTypeInferencePtr)(Qnn_OpConfig_t *) =
    &HmxW4A16TensorDumpDataTypeInference;

}
