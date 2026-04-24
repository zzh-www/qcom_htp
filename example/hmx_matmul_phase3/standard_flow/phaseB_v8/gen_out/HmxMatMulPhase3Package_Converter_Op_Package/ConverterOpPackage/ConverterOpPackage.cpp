//=============================================================================
// Converter Op Package for V8 (HmxMatMulPhase3Package)
//
// Each op's *ShapeInference() is called by qairt-converter / qnn-onnx-converter
// at graph-build time to derive output dimensions from input tensors.
// Output dim storage is pre-allocated by the caller (we just fill the values).
//=============================================================================
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "QnnOpPackage.h"
#include "QnnTypes.h"

#ifndef EXPORT_API
#define EXPORT_API __attribute__((visibility("default")))
#endif

static inline uint32_t *tensor_dims(Qnn_Tensor_t *t) {
    switch (t->version) {
        case QNN_TENSOR_VERSION_1: return t->v1.dimensions;
        case QNN_TENSOR_VERSION_2: return t->v2.dimensions;
        default:                   return nullptr;
    }
}
static inline uint32_t tensor_rank(const Qnn_Tensor_t *t) {
    switch (t->version) {
        case QNN_TENSOR_VERSION_1: return t->v1.rank;
        case QNN_TENSOR_VERSION_2: return t->v2.rank;
        default:                   return 0;
    }
}
static inline void set_tensor_dtype(Qnn_Tensor_t *t, Qnn_DataType_t dt) {
    switch (t->version) {
        case QNN_TENSOR_VERSION_1: t->v1.dataType = dt; break;
        case QNN_TENSOR_VERSION_2: t->v2.dataType = dt; break;
        default: break;
    }
}

extern "C" {

/* ---------- PackActivationU8RowMajor ----------
   in[0]  = [1, 1, M, K]  →  out[0] = [1, M/32, K/32, 1024] */
EXPORT_API Qnn_ErrorHandle_t PackActivationU8RowMajorShapeInference(Qnn_OpConfig_t *op) {
    if (op->v1.numOfInputs < 1 || op->v1.numOfOutputs < 1)
        return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    uint32_t *in  = tensor_dims(&op->v1.inputTensors[0]);
    uint32_t *out = tensor_dims(&op->v1.outputTensors[0]);
    if (tensor_rank(&op->v1.inputTensors[0]) != 4) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    uint32_t M = in[2], K = in[3];
    if (M % 32 || K % 32) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    out[0] = 1; out[1] = M / 32; out[2] = K / 32; out[3] = 1024;
    return QNN_SUCCESS;
}
EXPORT_API Qnn_ErrorHandle_t PackActivationU8RowMajorDataTypeInference(Qnn_OpConfig_t *op) {
    set_tensor_dtype(&op->v1.outputTensors[0], QNN_DATATYPE_UINT_8);
    return QNN_SUCCESS;
}

/* ---------- PackWeightToHmxTileV3 ----------
   in[0]  = [1, 1, K, N]  →  out[0] = [1, N/32, K/32, 1024] */
EXPORT_API Qnn_ErrorHandle_t PackWeightToHmxTileV3ShapeInference(Qnn_OpConfig_t *op) {
    if (op->v1.numOfInputs < 1 || op->v1.numOfOutputs < 1)
        return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    uint32_t *in  = tensor_dims(&op->v1.inputTensors[0]);
    uint32_t *out = tensor_dims(&op->v1.outputTensors[0]);
    if (tensor_rank(&op->v1.inputTensors[0]) != 4) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    uint32_t K = in[2], N = in[3];
    if (K % 32 || N % 32) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    out[0] = 1; out[1] = N / 32; out[2] = K / 32; out[3] = 1024;
    return QNN_SUCCESS;
}
EXPORT_API Qnn_ErrorHandle_t PackWeightToHmxTileV3DataTypeInference(Qnn_OpConfig_t *op) {
    set_tensor_dtype(&op->v1.outputTensors[0], QNN_DATATYPE_UINT_8);
    return QNN_SUCCESS;
}

/* ---------- MatMulV8 ----------
   in[0] = [1, M_t, K_t, 1024]   packed_act
   in[1] = [1, N_t, K_t, 1024]   packed_wt
   in[2] = [1, 1,   N_t, 128]    bias_fp16 (uint16)
   in[3] = [1, 1,   1,   2048]   scratch
   out   = [1, M_t, N_t, 1024]                                             */
EXPORT_API Qnn_ErrorHandle_t MatMulV8ShapeInference(Qnn_OpConfig_t *op) {
    if (op->v1.numOfInputs < 2 || op->v1.numOfOutputs < 1)
        return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    uint32_t *a = tensor_dims(&op->v1.inputTensors[0]);
    uint32_t *w = tensor_dims(&op->v1.inputTensors[1]);
    uint32_t *o = tensor_dims(&op->v1.outputTensors[0]);
    if (tensor_rank(&op->v1.inputTensors[0]) != 4 ||
        tensor_rank(&op->v1.inputTensors[1]) != 4) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    /* a[1]=M_t, a[2]=K_t ; w[1]=N_t, w[2]=K_t (check match) */
    if (a[2] != w[2]) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    o[0] = 1; o[1] = a[1]; o[2] = w[1]; o[3] = 1024;
    return QNN_SUCCESS;
}
EXPORT_API Qnn_ErrorHandle_t MatMulV8DataTypeInference(Qnn_OpConfig_t *op) {
    set_tensor_dtype(&op->v1.outputTensors[0], QNN_DATATYPE_UINT_8);
    return QNN_SUCCESS;
}

/* ---------- TcmDramCopy ---------- in[0] and out[0] identical shape. */
EXPORT_API Qnn_ErrorHandle_t TcmDramCopyShapeInference(Qnn_OpConfig_t *op) {
    if (op->v1.numOfInputs < 1 || op->v1.numOfOutputs < 1)
        return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    uint32_t *in  = tensor_dims(&op->v1.inputTensors[0]);
    uint32_t *out = tensor_dims(&op->v1.outputTensors[0]);
    uint32_t r = tensor_rank(&op->v1.inputTensors[0]);
    for (uint32_t i = 0; i < r; i++) out[i] = in[i];
    return QNN_SUCCESS;
}
EXPORT_API Qnn_ErrorHandle_t TcmDramCopyDataTypeInference(Qnn_OpConfig_t *op) {
    set_tensor_dtype(&op->v1.outputTensors[0], QNN_DATATYPE_UINT_8);
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t (*PackActivationU8RowMajorOutputInfoInferencePtr)(Qnn_OpConfig_t *) = &PackActivationU8RowMajorShapeInference;
Qnn_ErrorHandle_t (*PackWeightToHmxTileV3OutputInfoInferencePtr)(Qnn_OpConfig_t *)   = &PackWeightToHmxTileV3ShapeInference;
Qnn_ErrorHandle_t (*MatMulV8OutputInfoInferencePtr)(Qnn_OpConfig_t *)               = &MatMulV8ShapeInference;
Qnn_ErrorHandle_t (*TcmDramCopyOutputInfoInferencePtr)(Qnn_OpConfig_t *)            = &TcmDramCopyShapeInference;

Qnn_ErrorHandle_t (*PackActivationU8RowMajorDataTypeInferencePtr)(Qnn_OpConfig_t *) = &PackActivationU8RowMajorDataTypeInference;
Qnn_ErrorHandle_t (*PackWeightToHmxTileV3DataTypeInferencePtr)(Qnn_OpConfig_t *)    = &PackWeightToHmxTileV3DataTypeInference;
Qnn_ErrorHandle_t (*MatMulV8DataTypeInferencePtr)(Qnn_OpConfig_t *)                 = &MatMulV8DataTypeInference;
Qnn_ErrorHandle_t (*TcmDramCopyDataTypeInferencePtr)(Qnn_OpConfig_t *)              = &TcmDramCopyDataTypeInference;

} // extern "C"
