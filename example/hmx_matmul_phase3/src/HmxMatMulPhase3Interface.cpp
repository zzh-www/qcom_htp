/*
 * HmxMatMulPhase3Interface.cpp — QHPI OpPackage interface provider.
 * V8-only after 2026-04-25 cleanup.
 *
 * Package name:  HmxMatMulPhase3Package
 * Ops exposed:   MatMulV8, PackActivationU8RowMajor, PackWeightToHmxTileV3,
 *                TcmDramCopy, UntileToRowMajor
 *
 * See docs/qnn_custom_op_sop.md for the end-to-end flow.
 */

#include "HTP/QnnHtpCommon.h"
#include "HTP/core/qhpi.h"
#include "QnnOpPackage.h"
#include <array>
#include <string>

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

extern "C" void register_hmx_matmul_v8_op();
extern "C" {
void register_pack_act_rm_op();
void register_pack_wt_v3_op();
void register_untile_to_rowmajor_op();
void register_tcm_dram_copy_op();
}

static constexpr auto sg_packageName     = THIS_PKG_NAME_STR;
static constexpr auto sg_opNameV8        = "MatMulV8";
static constexpr auto sg_opNamePackActRM = "PackActivationU8RowMajor";
static constexpr auto sg_opNamePackWtV3  = "PackWeightToHmxTileV3";
static constexpr auto sg_opNameTcmDram   = "TcmDramCopy";
static constexpr auto sg_opNameUntile    = "UntileToRowMajor";
static std::array<const char *, 5> sg_opNames{{
    sg_opNameV8, sg_opNamePackActRM, sg_opNamePackWtV3,
    sg_opNameTcmDram, sg_opNameUntile,
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
    bool match = (tn == sg_opNameV8)
              || (tn == sg_opNamePackActRM)
              || (tn == sg_opNamePackWtV3)
              || (tn == sg_opNameTcmDram)
              || (tn == sg_opNameUntile);
    if (!match) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
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
    register_hmx_matmul_v8_op();
    register_pack_act_rm_op();
    register_pack_wt_v3_op();
    register_untile_to_rowmajor_op();
    register_tcm_dram_copy_op();
    return THIS_PKG_NAME_STR;
}

#ifdef __cplusplus
}
#endif
