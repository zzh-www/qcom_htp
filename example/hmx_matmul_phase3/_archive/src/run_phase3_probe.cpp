/*
 * run_phase3_probe.cpp — host harness for Phase 3A probe.
 *
 * Builds QNN graph with one MatMulInt8xInt8Crouton node. Inputs declared
 * as Crouton_8/Indirect in the OpPackage signature so QNN auto-inserts
 * ForceFormat_Crouton_b upstream. Kernel logs block-table diagnostics
 * via FARF and zeros output. Purpose: confirm signature triggers auto-
 * insert and see actual Crouton byte layout for 32x32 int8 data.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <dlfcn.h>
#include <vector>

#include "QnnInterface.h"
#include "QnnTypes.h"
#include "QnnCommon.h"
#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnProfile.h"
#include "HTP/QnnHtpGraph.h"

using QnnInterfaceGetProvidersFn = Qnn_ErrorHandle_t (*)(
    const QnnInterface_t ***providerList, uint32_t *numProviders);

static void *g_lib = nullptr;
static QNN_INTERFACE_VER_TYPE g_qnn;

#define QCHECK(expr) do {                                                  \
    Qnn_ErrorHandle_t _e = (expr);                                         \
    if (_e != QNN_SUCCESS) {                                               \
        std::fprintf(stderr, "QNN error %lu at %s:%d: %s\n",               \
                     (unsigned long)_e, __FILE__, __LINE__, #expr);        \
        return 1;                                                          \
    }                                                                      \
} while (0)

static int load_backend(const char *path) {
    g_lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!g_lib) { std::fprintf(stderr, "dlopen %s: %s\n", path, dlerror()); return -1; }
    auto fn = (QnnInterfaceGetProvidersFn)dlsym(g_lib, "QnnInterface_getProviders");
    if (!fn) return -1;
    const QnnInterface_t **list = nullptr;
    uint32_t n = 0;
    if (fn(&list, &n) != QNN_SUCCESS || n == 0) return -1;
    g_qnn = list[0]->QNN_INTERFACE_VER_NAME;
    return 0;
}

static Qnn_Tensor_t mk_tensor(const char *name, Qnn_TensorType_t type,
                              Qnn_DataType_t dt, uint32_t rank, uint32_t *dims) {
    Qnn_Tensor_t t{};
    t.version = QNN_TENSOR_VERSION_1;
    t.v1.name = name;
    t.v1.type = type;
    t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v1.dataType = dt;
    t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
    t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    t.v1.quantizeParams.scaleOffsetEncoding = {1.0f, 0};
    t.v1.rank = rank;
    t.v1.dimensions = dims;
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    return t;
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? std::atoi(argv[1]) : 32;
    int K = (argc > 2) ? std::atoi(argv[2]) : 32;
    int N = (argc > 3) ? std::atoi(argv[3]) : 32;

    std::printf("== Phase3A Probe: %dx%dx%d ==\n", M, K, N);
    if (load_backend("./libQnnHtp.so") != 0) return 1;

    Qnn_BackendHandle_t backend = nullptr;
    const QnnBackend_Config_t *beCfgs[] = {nullptr};
    QCHECK(g_qnn.backendCreate(nullptr, beCfgs, &backend));

    QCHECK(g_qnn.backendRegisterOpPackage(backend,
        "./libQnnHmxMatMulPhase3_cpu.so", "HmxMatMulPhase3InterfaceProvider", "CPU"));
    QCHECK(g_qnn.backendRegisterOpPackage(backend,
        "./libQnnHmxMatMulPhase3_htp.so", "HmxMatMulPhase3InterfaceProvider", "HTP"));

    Qnn_ContextHandle_t context = nullptr;
    const QnnContext_Config_t *ctxCfgs[] = {nullptr};
    QCHECK(g_qnn.contextCreate(backend, nullptr, ctxCfgs, &context));

    Qnn_GraphHandle_t graph = nullptr;
    const QnnGraph_Config_t *grCfgs[] = {nullptr};
    QCHECK(g_qnn.graphCreate(context, "phase3_probe", grCfgs, &graph));

    uint32_t aDims[] = {1, 1, (uint32_t)M, (uint32_t)K};
    uint32_t wDims[] = {1, 1, (uint32_t)K, (uint32_t)N};
    uint32_t oDims[] = {1, 1, (uint32_t)M, (uint32_t)N};

    auto aT = mk_tensor("act", QNN_TENSOR_TYPE_APP_WRITE,
                        QNN_DATATYPE_UINT_8, 4, aDims);
    auto wT = mk_tensor("wt",  QNN_TENSOR_TYPE_APP_WRITE,
                        QNN_DATATYPE_UINT_8, 4, wDims);
    auto oT = mk_tensor("out", QNN_TENSOR_TYPE_APP_READ,
                        QNN_DATATYPE_INT_32, 4, oDims);

    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &aT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &wT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &oT));

    Qnn_Tensor_t ins[] = {aT, wT};
    Qnn_OpConfig_t op{};
    op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name = "mm0";
    op.v1.packageName = "HmxMatMulPhase3Package";
    op.v1.typeName = "MatMulInt8xInt8Crouton";
    op.v1.numOfParams = 0;
    op.v1.params = nullptr;
    op.v1.numOfInputs = 2;
    op.v1.inputTensors = ins;
    op.v1.numOfOutputs = 1;
    op.v1.outputTensors = &oT;
    QCHECK(g_qnn.graphAddNode(graph, op));

    Qnn_ProfileHandle_t profile = nullptr;
    QCHECK(g_qnn.profileCreate(backend, QNN_PROFILE_LEVEL_DETAILED, &profile));
    QCHECK(g_qnn.graphFinalize(graph, profile, nullptr));
    std::printf("[Graph] finalized\n");

    std::vector<uint8_t> aBuf(M * K);
    std::vector<uint8_t> wBuf(K * N);
    std::vector<int32_t> oBuf(M * N);
    /* Deterministic pattern: activation = (i*37)&0xFF, weight = (i*13)%15 */
    for (int i = 0; i < M * K; i++) aBuf[i] = (uint8_t)((i * 37) & 0xFF);
    for (int i = 0; i < K * N; i++) wBuf[i] = (uint8_t)((i * 13) % 15);

    Qnn_Tensor_t eIn[2], eOut[1];
    std::memcpy(&eIn[0], &aT, sizeof(Qnn_Tensor_t));
    eIn[0].v1.clientBuf = {aBuf.data(), (uint32_t)(aBuf.size())};
    std::memcpy(&eIn[1], &wT, sizeof(Qnn_Tensor_t));
    eIn[1].v1.clientBuf = {wBuf.data(), (uint32_t)(wBuf.size())};
    std::memcpy(&eOut[0], &oT, sizeof(Qnn_Tensor_t));
    eOut[0].v1.clientBuf = {oBuf.data(), (uint32_t)(oBuf.size() * sizeof(int32_t))};

    /* Single invocation. Kernel encodes block-table diagnostics into oBuf. */
    QCHECK(g_qnn.graphExecute(graph, eIn, 2, eOut, 1, profile, nullptr));
    std::printf("[Exec] complete. Decoding Phase3A probe output...\n");

    if ((uint32_t)oBuf[0] != 0xDEADBEEF) {
        std::printf("  [FAIL] magic marker missing! oBuf[0]=0x%08x (expected 0xDEADBEEF)\n",
                    (uint32_t)oBuf[0]);
        std::printf("  Likely: kernel not called, or HMX path bypassed (SCALAR_ONLY fallback?)\n");
        std::printf("  oBuf[0..7] = %08x %08x %08x %08x %08x %08x %08x %08x\n",
                    (uint32_t)oBuf[0], (uint32_t)oBuf[1], (uint32_t)oBuf[2], (uint32_t)oBuf[3],
                    (uint32_t)oBuf[4], (uint32_t)oBuf[5], (uint32_t)oBuf[6], (uint32_t)oBuf[7]);
    } else {
        std::printf("  [OK] magic matches, kernel ran on HTP\n");
        std::printf("  act_nblk = %d\n", oBuf[1]);
        std::printf("  wt_nblk  = %d\n", oBuf[2]);
        std::printf("  act_block_shape = [%d, %d, %d, %d]\n",
                    oBuf[3], oBuf[4], oBuf[5], oBuf[6]);
        std::printf("  wt_block_shape  = [%d, %d, %d, %d]\n",
                    oBuf[7], oBuf[8], oBuf[9], oBuf[10]);
        std::printf("  out_shape       = [%d, %d, %d, %d]\n",
                    oBuf[11], oBuf[12], oBuf[13], oBuf[14]);
        std::printf("  act block 0 first 64 bytes:\n   ");
        for (int i = 0; i < 16; i++) {
            uint32_t w = (uint32_t)oBuf[15 + i];
            std::printf(" %02x %02x %02x %02x",
                        w & 0xff, (w >> 8) & 0xff, (w >> 16) & 0xff, (w >> 24) & 0xff);
            if ((i & 3) == 3) std::printf("\n   ");
        }
        std::printf("\n  wt  block 0 first 64 bytes:\n   ");
        for (int i = 0; i < 16; i++) {
            uint32_t w = (uint32_t)oBuf[31 + i];
            std::printf(" %02x %02x %02x %02x",
                        w & 0xff, (w >> 8) & 0xff, (w >> 16) & 0xff, (w >> 24) & 0xff);
            if ((i & 3) == 3) std::printf("\n   ");
        }
        std::printf("\n");
    }

    g_qnn.profileFree(profile);
    g_qnn.contextFree(context, nullptr);
    g_qnn.backendFree(backend);
    return 0;
}
