/*
 * run_matmul_graph.cpp — host harness for the Phase 3C "Path W" graph:
 *
 *   raw uint8 act [M,K] ──► PackActivationU8ToHmxTile ──► packed_act ┐
 *   raw int8  wt  [K,N] ──► PackWeightToHmxTileV3    ──► packed_wt  ┴─► MatMulV3 → int32 [M,N]
 *
 * Goal: let QNN HTP scheduler dispatch the two HVX pack nodes onto HVX
 * worker threads while the HMX matmul runs. Baseline V3 (host pre-pack)
 * is 0.143 cyc/MAC @ 512³; if graph scheduling overlaps pack+MAC we
 * expect the steady-state accelerator cycle count to stay near that.
 * If it rises roughly by the pack cost, scheduling is serial.
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

static void ref_matmul_u8_i8(int32_t *out, const uint8_t *au, const int8_t *w,
                              int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++)
                s += (int32_t)au[i * K + k] * (int32_t)w[k * N + j];
            out[i * N + j] = s;
        }
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? std::atoi(argv[1]) : 32;
    int K = (argc > 2) ? std::atoi(argv[2]) : 32;
    int N = (argc > 3) ? std::atoi(argv[3]) : 32;
    int ITERS = (argc > 4) ? std::atoi(argv[4]) : 5;
    if (M % 32 || K % 32 || N % 32) {
        std::fprintf(stderr, "M, K, N must all be multiples of 32\n");
        return 2;
    }
    const int M_tiles = M / 32;
    const int K_tiles = K / 32;
    const int N_tiles = N / 32;

    std::printf("== MatMulGraph (PackActU8 + PackWtV3 + MatMulV3): %dx%dx%d ==\n", M, K, N);
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
    QCHECK(g_qnn.graphCreate(context, "matmul_graph", grCfgs, &graph));

    /* Tensor shapes. */
    uint32_t aDims[]  = {1, 1, (uint32_t)M, (uint32_t)K};         /* raw act */
    uint32_t wDims[]  = {1, 1, (uint32_t)K, (uint32_t)N};         /* raw wt  */
    uint32_t paDims[] = {1, (uint32_t)M_tiles, (uint32_t)K_tiles, 2048}; /* packed_act */
    uint32_t pwDims[] = {1, (uint32_t)N_tiles, (uint32_t)K_tiles, 1024}; /* packed_wt  */
    uint32_t oDims[]  = {1, 1, (uint32_t)M, (uint32_t)N};
    const uint32_t vtcm_bytes = 16 * 1024;
    uint32_t sDims[]  = {1, 1, 1, vtcm_bytes};

    auto aT  = mk_tensor("act_raw",    QNN_TENSOR_TYPE_APP_WRITE,
                         QNN_DATATYPE_UINT_8, 4, aDims);
    /* Use UINT_8 (not INT_8) for raw weight so QNN doesn't insert a
     * Cast int8→uint8 that shifts bytes by +128. V3 interprets the
     * packed byte stream as signed int8 via the HMX `weight.b` consumer,
     * so we want raw bytes to flow through untouched. */
    auto wT  = mk_tensor("wt_raw",     QNN_TENSOR_TYPE_APP_WRITE,
                         QNN_DATATYPE_UINT_8, 4, wDims);
    auto paT = mk_tensor("packed_act", QNN_TENSOR_TYPE_NATIVE,
                         QNN_DATATYPE_UINT_8, 4, paDims);
    auto pwT = mk_tensor("packed_wt",  QNN_TENSOR_TYPE_NATIVE,
                         QNN_DATATYPE_UINT_8, 4, pwDims);
    auto sT  = mk_tensor("scratch",    QNN_TENSOR_TYPE_APP_WRITE,
                         QNN_DATATYPE_UINT_8, 4, sDims);
    auto oT  = mk_tensor("out",        QNN_TENSOR_TYPE_APP_READ,
                         QNN_DATATYPE_INT_32, 4, oDims);

    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &aT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &wT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &paT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &pwT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &sT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &oT));

    /* Node 1: PackActivationU8ToHmxTile (act_raw → packed_act). */
    {
        Qnn_Tensor_t ins[]  = {aT};
        Qnn_Tensor_t outs[] = {paT};
        Qnn_OpConfig_t op{};
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = "pack_act";
        op.v1.packageName = "HmxMatMulPhase3Package";
        op.v1.typeName = "PackActivationU8ToHmxTile";
        op.v1.numOfParams = 0;
        op.v1.params = nullptr;
        op.v1.numOfInputs = 1;
        op.v1.inputTensors = ins;
        op.v1.numOfOutputs = 1;
        op.v1.outputTensors = outs;
        QCHECK(g_qnn.graphAddNode(graph, op));
    }

    /* Node 2: PackWeightToHmxTileV3 (wt_raw → packed_wt). */
    {
        Qnn_Tensor_t ins[]  = {wT};
        Qnn_Tensor_t outs[] = {pwT};
        Qnn_OpConfig_t op{};
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = "pack_wt";
        op.v1.packageName = "HmxMatMulPhase3Package";
        op.v1.typeName = "PackWeightToHmxTileV3";
        op.v1.numOfParams = 0;
        op.v1.params = nullptr;
        op.v1.numOfInputs = 1;
        op.v1.inputTensors = ins;
        op.v1.numOfOutputs = 1;
        op.v1.outputTensors = outs;
        QCHECK(g_qnn.graphAddNode(graph, op));
    }

    /* Node 3: MatMulV3 (packed_act + packed_wt + scratch → out). */
    {
        Qnn_Tensor_t ins[] = {paT, pwT, sT};
        Qnn_OpConfig_t op{};
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = "mmv3";
        op.v1.packageName = "HmxMatMulPhase3Package";
        op.v1.typeName = "MatMulV3";
        op.v1.numOfParams = 0;
        op.v1.params = nullptr;
        op.v1.numOfInputs = 3;
        op.v1.inputTensors = ins;
        op.v1.numOfOutputs = 1;
        op.v1.outputTensors = &oT;
        QCHECK(g_qnn.graphAddNode(graph, op));
    }

    Qnn_ProfileHandle_t profile = nullptr;
    QCHECK(g_qnn.profileCreate(backend, QNN_PROFILE_LEVEL_DETAILED, &profile));
    QCHECK(g_qnn.graphFinalize(graph, profile, nullptr));
    std::printf("[Graph] finalized (3-node Path W: PackActU8 + PackWtV3 + MatMulV3)\n");

    /* Client-side raw tensors. */
    std::vector<uint8_t> aRaw(M * K);
    std::vector<int8_t>  wRaw(K * N);
    for (int i = 0; i < M * K; i++) aRaw[i] = (uint8_t)((i * 37) & 0xFF);
    for (int i = 0; i < K * N; i++) wRaw[i] = (int8_t)(((i * 13) % 15) - 7);

    std::vector<uint8_t> sBuf(vtcm_bytes);
    std::vector<int32_t> oBuf(M * N), oRef(M * N);

    Qnn_Tensor_t eIn[3], eOut[1];
    std::memcpy(&eIn[0], &aT, sizeof(Qnn_Tensor_t));
    eIn[0].v1.clientBuf = {aRaw.data(), (uint32_t)aRaw.size()};
    std::memcpy(&eIn[1], &wT, sizeof(Qnn_Tensor_t));
    eIn[1].v1.clientBuf = {wRaw.data(), (uint32_t)wRaw.size()};
    std::memcpy(&eIn[2], &sT, sizeof(Qnn_Tensor_t));
    eIn[2].v1.clientBuf = {sBuf.data(), (uint32_t)sBuf.size()};
    std::memcpy(&eOut[0], &oT, sizeof(Qnn_Tensor_t));
    eOut[0].v1.clientBuf = {oBuf.data(), (uint32_t)(oBuf.size() * sizeof(int32_t))};

    QCHECK(g_qnn.graphExecute(graph, eIn, 3, eOut, 1, profile, nullptr));  /* warmup */

    auto cycles_from_profile = [](Qnn_ProfileHandle_t p) -> uint64_t {
        const QnnProfile_EventId_t *evs = nullptr; uint32_t ne = 0;
        if (g_qnn.profileGetEvents(p, &evs, &ne) != QNN_SUCCESS) return 0;
        for (uint32_t i = 0; i < ne; i++) {
            QnnProfile_EventData_t d{};
            if (g_qnn.profileGetEventData(evs[i], &d) == QNN_SUCCESS
                && d.identifier
                && std::strstr(d.identifier, "Accelerator (execute) time (cycles)"))
                return d.value;
        }
        return 0;
    };

    auto dump_profile = [](Qnn_ProfileHandle_t p) {
        const QnnProfile_EventId_t *evs = nullptr; uint32_t ne = 0;
        if (g_qnn.profileGetEvents(p, &evs, &ne) != QNN_SUCCESS) return;
        std::printf("[Profile] %u top-level events:\n", ne);
        for (uint32_t i = 0; i < ne; i++) {
            QnnProfile_EventData_t d{};
            if (g_qnn.profileGetEventData(evs[i], &d) == QNN_SUCCESS) {
                std::printf("  [%02u] val=%llu unit=%d id='%s'\n",
                            i, (unsigned long long)d.value,
                            (int)d.unit,
                            d.identifier ? d.identifier : "(null)");
            }
            const QnnProfile_EventId_t *subs = nullptr; uint32_t ns = 0;
            if (g_qnn.profileGetSubEvents(evs[i], &subs, &ns) == QNN_SUCCESS) {
                for (uint32_t j = 0; j < ns; j++) {
                    QnnProfile_EventData_t ds{};
                    if (g_qnn.profileGetEventData(subs[j], &ds) == QNN_SUCCESS) {
                        std::printf("    [%02u.%02u] val=%llu unit=%d id='%s'\n",
                                    i, j, (unsigned long long)ds.value,
                                    (int)ds.unit,
                                    ds.identifier ? ds.identifier : "(null)");
                    }
                }
            }
        }
    };

    uint64_t total = 0, mn_ = UINT64_MAX, mx = 0;
    for (int it = 0; it < ITERS; it++) {
        QCHECK(g_qnn.graphExecute(graph, eIn, 3, eOut, 1, profile, nullptr));
        uint64_t c = cycles_from_profile(profile);
        total += c; if (c < mn_) mn_ = c; if (c > mx) mx = c;
    }
    uint64_t avg = total / ITERS;

    /* Dump last-execute profile events for per-op breakdown. */
    dump_profile(profile);

    ref_matmul_u8_i8(oRef.data(), aRaw.data(), wRaw.data(), M, K, N);
    int bad = 0; int64_t merr = 0;
    for (int i = 0; i < M * N; i++) {
        int64_t d = (int64_t)oBuf[i] - (int64_t)oRef[i];
        if (d) { bad++; if (std::abs(d) > merr) merr = std::abs(d); }
    }

    uint64_t macs = (uint64_t)M * K * N;
    std::printf("[Steady] cycles: avg=%llu min=%llu max=%llu\n",
                (unsigned long long)avg, (unsigned long long)mn_, (unsigned long long)mx);
    std::printf("[Steady] MACs=%llu  cycles_per_MAC=%.3f\n",
                (unsigned long long)macs, (double)avg / (double)macs);
    std::printf("[Check] mismatches=%d/%d max_abs_err=%lld\n", bad, M*N, (long long)merr);
    std::printf("  oBuf[0..3]=%d %d %d %d  oRef[0..3]=%d %d %d %d\n",
                oBuf[0], oBuf[1], oBuf[2], oBuf[3],
                oRef[0], oRef[1], oRef[2], oRef[3]);

    g_qnn.profileFree(profile);
    g_qnn.contextFree(context, nullptr);
    g_qnn.backendFree(backend);
    return bad == 0 ? 0 : 1;
}
