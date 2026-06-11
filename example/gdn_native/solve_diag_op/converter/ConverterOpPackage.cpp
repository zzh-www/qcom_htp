// Converter-side shape/type inference for GdnSolveDiagPackage::GdnSolveDiag.
// Output T has the same shape as input A; output dtype is unsigned fixed-point 16.

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

EXPORT_API Qnn_ErrorHandle_t GdnSolveDiagShapeInference(Qnn_OpConfig_t *op)
{
    if (!op || op->v1.numOfInputs < 1 || op->v1.numOfOutputs < 1)
        return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    const uint32_t rank = tensor_rank(&op->v1.inputTensors[0]);
    uint32_t *a = tensor_dims(&op->v1.inputTensors[0]);
    uint32_t *o = tensor_dims(&op->v1.outputTensors[0]);
    if (!a || !o || rank == 0) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    for (uint32_t d = 0; d < rank; ++d) o[d] = a[d];
    /* output 1 (Hd handoff) shares the [B,H,C,C] shape. */
    if (op->v1.numOfOutputs >= 2) {
        uint32_t *o1 = tensor_dims(&op->v1.outputTensors[1]);
        if (o1) for (uint32_t d = 0; d < rank; ++d) o1[d] = a[d];
    }
    return QNN_SUCCESS;
}

EXPORT_API Qnn_ErrorHandle_t GdnSolveDiagDataTypeInference(Qnn_OpConfig_t *op)
{
    if (!op || op->v1.numOfOutputs < 1) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    set_tensor_dtype(&op->v1.outputTensors[0], QNN_DATATYPE_UFIXED_POINT_16);
    if (op->v1.numOfOutputs >= 2)
        set_tensor_dtype(&op->v1.outputTensors[1], QNN_DATATYPE_UFIXED_POINT_8);
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t (*GdnSolveDiagOutputInfoInferencePtr)(Qnn_OpConfig_t *) = &GdnSolveDiagShapeInference;
Qnn_ErrorHandle_t (*GdnSolveDiagDataTypeInferencePtr)(Qnn_OpConfig_t *) = &GdnSolveDiagDataTypeInference;

}
