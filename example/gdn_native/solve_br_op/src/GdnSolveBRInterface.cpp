/*
 * GdnSolveBRInterface.cpp — QNN OpPackage interface for the GdnSolveBR custom op
 * (T = (I-A)^-1, C=128 block-recursive, HVX diagonals + HMX merge).
 * QHPI registration boilerplate (cloned from GdnSolvePackage).
 */
#include "HTP/QnnHtpCommon.h"
#include "HTP/core/qhpi.h"
#include "QnnOpPackage.h"
#include <array>
#include <string>

#define STRINGIZE_DETAIL(X) #X
#define STRINGIZE(X) STRINGIZE_DETAIL(X)
#define THIS_PKG_NAME_STR STRINGIZE(THIS_PKG_NAME)

extern "C" void register_gdn_solve_br_op();

static constexpr auto sg_packageName = THIS_PKG_NAME_STR;
static constexpr auto sg_opName = "GdnSolveBR";
static std::array<const char *, 1> sg_opNames{{sg_opName}};

static Qnn_ApiVersion_t sg_sdkApiVersion = QNN_HTP_API_VERSION_INIT;
static Qnn_Version_t sg_opsetVersion = {1, 0, 0};
static QnnOpPackage_Info_t sg_packageInfo = {
    sg_packageName,
    sg_opNames.data(),
    nullptr,
    sg_opNames.size(),
    nullptr,
    0,
    "gdn_solve_br_qhpi",
    &sg_sdkApiVersion,
    nullptr,
    &sg_opsetVersion,
    {0},
};

static QnnOpPackage_GlobalInfrastructure_t sg_globalInfra = nullptr;
static bool sg_initialized = false;
static QnnLog_Callback_t sg_logCallback = nullptr;
static QnnLog_Level_t sg_maxLogLevel = (QnnLog_Level_t)0;

static Qnn_ErrorHandle_t pkgInit(QnnOpPackage_GlobalInfrastructure_t infra)
{
    if (sg_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_ALREADY_INITIALIZED;
    sg_globalInfra = infra;
    sg_initialized = true;
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t pkgGetInfo(const QnnOpPackage_Info_t **info)
{
    if (!sg_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    if (!info) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    *info = &sg_packageInfo;
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t pkgValidateOpConfig(Qnn_OpConfig_t opConfig)
{
    if (std::string(sg_packageName) != opConfig.v1.packageName) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if (std::string(sg_opName) != opConfig.v1.typeName) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t pkgLogInit(QnnLog_Callback_t cb, QnnLog_Level_t level)
{
    if (!cb) return QNN_LOG_ERROR_INVALID_ARGUMENT;
    sg_logCallback = cb;
    sg_maxLogLevel = level;
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t pkgLogSetLevel(QnnLog_Level_t level) { sg_maxLogLevel = level; return QNN_SUCCESS; }
static Qnn_ErrorHandle_t pkgLogTerminate() { sg_logCallback = nullptr; return QNN_SUCCESS; }

static Qnn_ErrorHandle_t pkgCreateOpImpl(QnnOpPackage_GraphInfrastructure_t, QnnOpPackage_Node_t,
                                         QnnOpPackage_OpImpl_t *) { return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE; }
static Qnn_ErrorHandle_t pkgFreeOpImpl(QnnOpPackage_OpImpl_t) { return QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE; }

static Qnn_ErrorHandle_t pkgTerminate()
{
    if (!sg_initialized) return QNN_OP_PACKAGE_ERROR_LIBRARY_NOT_INITIALIZED;
    sg_globalInfra = nullptr;
    sg_initialized = false;
    return QNN_SUCCESS;
}

extern "C" Qnn_ErrorHandle_t
GdnSolveBRPackageInterfaceProvider(QnnOpPackage_Interface_t *interface)
{
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

extern "C" const char *qhpi_init()
{
    register_gdn_solve_br_op();
    return THIS_PKG_NAME_STR;
}
