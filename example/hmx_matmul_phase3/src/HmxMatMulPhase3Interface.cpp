/*
 * HmxMatMulPhase3Interface.cpp — QNN OpPackage boilerplate.
 * Phase 3A probe: int8×int8 matmul, Crouton_8 + Indirect signatures,
 * kernel is pure mxmem + MAC (no HVX, no scalar pack).
 */

#include "HTP/QnnHtpCommon.h"
#include "HTP/core/qhpi.h"
#include "QnnOpPackage.h"
#include <array>
#include <string>

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

extern void register_phase3_ops();
extern "C" void register_hmx_matmul_v2_op();
extern "C" void register_hmx_matmul_v3_op();
extern "C" void register_hmx_matmul_v4_op();
extern "C" void register_hmx_matmul_v6_op();
extern "C" void register_hmx_matmul_v7_op();
extern "C" void register_hmx_matmul_v8_op();
extern "C" void register_requant_hvx_op();
extern "C" {
void register_pack_act_op();
void register_pack_wt_op();
void register_pack_act_u8_op();
void register_pack_act_rm_op();
void register_pack_wt_v3_op();
void register_combine_op();
void register_int4_expand_op();
}

static constexpr auto sg_packageName = THIS_PKG_NAME_STR;
static constexpr auto sg_opName = "MatMulInt8xInt8Crouton";
static constexpr auto sg_opNameV2      = "MatMulV2";
static constexpr auto sg_opNameV3      = "MatMulV3";
static constexpr auto sg_opNameV4      = "MatMulV4";
static constexpr auto sg_opNameV6      = "MatMulV6";
static constexpr auto sg_opNameV7      = "MatMulV7";
static constexpr auto sg_opNameV8      = "MatMulV8";
static constexpr auto sg_opNameRequant = "RequantHvx";
static constexpr auto sg_opNamePackActRM = "PackActivationU8RowMajor";
static constexpr auto sg_opNamePackAct = "PackActivationToHmxTile";
static constexpr auto sg_opNamePackWt  = "PackWeightToHmxTile";
static constexpr auto sg_opNameCombine = "CombineHiLo";
static constexpr auto sg_opNameInt4Exp = "Int4Expand";
static constexpr auto sg_opNamePackActU8 = "PackActivationU8ToHmxTile";
static constexpr auto sg_opNamePackWtV3  = "PackWeightToHmxTileV3";
static std::array<const char *, 15> sg_opNames{{
    sg_opName, sg_opNameV2, sg_opNameV3, sg_opNameV4, sg_opNameV6,
    sg_opNameV7, sg_opNameV8, sg_opNameRequant, sg_opNamePackActRM,
    sg_opNamePackAct, sg_opNamePackWt, sg_opNameCombine, sg_opNameInt4Exp,
    sg_opNamePackActU8, sg_opNamePackWtV3
}};

static Qnn_ApiVersion_t sg_sdkApiVersion = QNN_HTP_API_VERSION_INIT;
static Qnn_Version_t sg_opsetVersion = {1, 0, 0};
static QnnOpPackage_Info_t sg_packageInfo = {
    sg_packageName,
    sg_opNames.data(),
    nullptr,
    sg_opNames.size(),
    nullptr,
    0,
    "hmx_matmul_phase3_qhpi",
    &sg_sdkApiVersion,
    nullptr,
    &sg_opsetVersion,
    {0}};

static QnnOpPackage_GlobalInfrastructure_t sg_globalInfra = nullptr;
static bool sg_initialized = false;
static QnnLog_Callback_t sg_logCallback = nullptr;
static QnnLog_Level_t sg_maxLogLevel = (QnnLog_Level_t)0;
static bool sg_logInitialized = false;

static Qnn_ErrorHandle_t pkgInit(QnnOpPackage_GlobalInfrastructure_t infra) {
    if (sg_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
    sg_globalInfra = infra;
    sg_initialized = true;
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t pkgGetInfo(const QnnOpPackage_Info_t **info) {
    if (!sg_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    if (!info) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    *info = &sg_packageInfo;
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t pkgValidateOpConfig(Qnn_OpConfig_t opConfig) {
    if (std::string(sg_packageName) != opConfig.v1.packageName)
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    const std::string tn = opConfig.v1.typeName;
    bool match = (tn == sg_opName)
              || (tn == sg_opNameV2)
              || (tn == sg_opNameV3)
              || (tn == sg_opNameV4)
              || (tn == sg_opNameV6)
              || (tn == sg_opNameV7)
              || (tn == sg_opNameV8)
              || (tn == sg_opNameRequant)
              || (tn == sg_opNamePackActRM)
              || (tn == sg_opNamePackAct)
              || (tn == sg_opNamePackWt)
              || (tn == sg_opNameCombine)
              || (tn == sg_opNameInt4Exp)
              || (tn == sg_opNamePackActU8)
              || (tn == sg_opNamePackWtV3);
    if (!match) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    (void)opConfig;
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t pkgLogInit(QnnLog_Callback_t cb, QnnLog_Level_t level) {
    if (!cb) return QNN_LOG_ERROR_INVALID_ARGUMENT;
    sg_logCallback = cb;
    sg_maxLogLevel = level;
    sg_logInitialized = true;
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t pkgLogSetLevel(QnnLog_Level_t level) {
    sg_maxLogLevel = level;
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t pkgLogTerminate() {
    sg_logCallback = nullptr;
    sg_logInitialized = false;
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t pkgCreateOpImpl(
    QnnOpPackage_GraphInfrastructure_t, QnnOpPackage_Node_t, QnnOpPackage_OpImpl_t*) {
    return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
}

static Qnn_ErrorHandle_t pkgFreeOpImpl(QnnOpPackage_OpImpl_t) {
    return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE;
}

static Qnn_ErrorHandle_t pkgTerminate() {
    if (!sg_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    sg_globalInfra = nullptr;
    sg_initialized = false;
    return QNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

Qnn_ErrorHandle_t HmxMatMulPhase3InterfaceProvider(QnnOpPackage_Interface_t *interface) {
    if (!interface) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    interface->interfaceVersion = {1, 4, 0};
    interface->v1_4.init = pkgInit;
    interface->v1_4.terminate = pkgTerminate;
    interface->v1_4.getInfo = pkgGetInfo;
    interface->v1_4.validateOpConfig = pkgValidateOpConfig;
    interface->v1_4.createOpImpl = pkgCreateOpImpl;
    interface->v1_4.freeOpImpl = pkgFreeOpImpl;
    interface->v1_4.logInitialize = pkgLogInit;
    interface->v1_4.logSetLevel = pkgLogSetLevel;
    interface->v1_4.logTerminate = pkgLogTerminate;
    return QNN_SUCCESS;
}

const char *qhpi_init() {
    register_phase3_ops();
    register_hmx_matmul_v2_op();
    register_hmx_matmul_v3_op();
    register_hmx_matmul_v4_op();
    register_hmx_matmul_v6_op();
    register_hmx_matmul_v7_op();
    register_hmx_matmul_v8_op();
    register_requant_hvx_op();
    register_pack_act_rm_op();
    register_pack_act_op();
    register_pack_wt_op();
    register_pack_act_u8_op();
    register_pack_wt_v3_op();
    register_combine_op();
    register_int4_expand_op();
    return THIS_PKG_NAME_STR;
}

#ifdef __cplusplus
}
#endif
