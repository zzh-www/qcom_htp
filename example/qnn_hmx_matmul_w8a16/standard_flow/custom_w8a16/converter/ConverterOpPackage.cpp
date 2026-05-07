// Converter-side shape/type inference for QnnHmxMatMulW8A16Package.

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

EXPORT_API Qnn_ErrorHandle_t HmxU16I8ToU16MatMulShapeInference(Qnn_OpConfig_t *op)
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
    o[3] = w[3];
    return QNN_SUCCESS;
}

EXPORT_API Qnn_ErrorHandle_t HmxU16I8ToU16MatMulDataTypeInference(Qnn_OpConfig_t *op)
{
    if (!op || op->v1.numOfOutputs < 1) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    if (op->v1.numOfInputs >= 4) {
        set_tensor_dtype(&op->v1.inputTensors[0], QNN_DATATYPE_SFIXED_POINT_32);
        set_tensor_dtype(&op->v1.inputTensors[1], QNN_DATATYPE_SFIXED_POINT_8);
        set_tensor_dtype(&op->v1.inputTensors[2], QNN_DATATYPE_UFIXED_POINT_16);
        set_tensor_dtype(&op->v1.inputTensors[3], QNN_DATATYPE_UFIXED_POINT_8);
    }
    set_tensor_dtype(&op->v1.outputTensors[0], QNN_DATATYPE_UFIXED_POINT_16);
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t (*HmxU16I8ToU16MatMulOutputInfoInferencePtr)(Qnn_OpConfig_t *) =
    &HmxU16I8ToU16MatMulShapeInference;
Qnn_ErrorHandle_t (*HmxU16I8ToU16MatMulDataTypeInferencePtr)(Qnn_OpConfig_t *) =
    &HmxU16I8ToU16MatMulDataTypeInference;

}
