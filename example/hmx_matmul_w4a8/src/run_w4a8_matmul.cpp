/*
 * run_w4a8_matmul.cpp — host harness: build a QNN graph with a single
 * HmxW4A8MatMulPackage::MatMulInt4xInt8 node, execute on HTP, verify
 * bit-exact against a scalar reference, and emit per-op profiling.
 *
 * Usage (on-device):
 *   ./run_int4_matmul [M] [K] [N]        # default 32 32 32
 *
 * The harness bypasses qairt-converter — QNN's ONNX converter doesn't yet
 * know how to route MatMul→int4 to our custom op. We construct the graph
 * via the QNN API directly (same pattern as ch03's qnn_hvx_hmx_test.c).
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
                              Qnn_DataType_t dt, uint32_t rank, uint32_t *dims,
                              int32_t zero_offset = 0, float scale = 1.0f) {
    Qnn_Tensor_t t{};
    t.version = QNN_TENSOR_VERSION_1;
    t.v1.name = name;
    t.v1.type = type;
    t.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v1.dataType = dt;
    /* With zero_offset=0, scale=1.0: QNN's Cast is a pass-through (no offset
     * shift), so the kernel reads the raw bits we wrote. Lets us treat the
     * uint16/uint8 data stream as signed int16/int8 via reinterpret. */
    t.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
    t.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    t.v1.quantizeParams.scaleOffsetEncoding = {scale, zero_offset};
    t.v1.rank = rank;
    t.v1.dimensions = dims;
    t.v1.memType = QNN_TENSORMEMTYPE_RAW;
    return t;
}

static void ref_matmul(int32_t *out, const int8_t *a, const int8_t *w,
                       int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++)
                s += (int32_t)a[i * K + k] * (int32_t)w[k * N + j];
            out[i * N + j] = s;
        }
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? std::atoi(argv[1]) : 32;
    int K = (argc > 2) ? std::atoi(argv[2]) : 32;
    int N = (argc > 3) ? std::atoi(argv[3]) : 32;
    int ITERS = (argc > 4) ? std::atoi(argv[4]) : 5;    /* steady-state iters (after warmup) */
    if (M % 32 || K % 32 || N % 32) {
        std::fprintf(stderr, "M, K, N must all be multiples of 32\n");
        return 2;
    }

    std::printf("== HmxW4A8MatMul: %dx%dx%d ==\n", M, K, N);
    if (load_backend("./libQnnHtp.so") != 0) return 1;

    Qnn_BackendHandle_t backend = nullptr;
    const QnnBackend_Config_t *beCfgs[] = {nullptr};
    QCHECK(g_qnn.backendCreate(nullptr, beCfgs, &backend));

    QCHECK(g_qnn.backendRegisterOpPackage(backend,
        "./libQnnHmxW4A8MatMul_cpu.so", "HmxW4A8MatMulInterfaceProvider", "CPU"));
    QCHECK(g_qnn.backendRegisterOpPackage(backend,
        "./libQnnHmxW4A8MatMul_htp.so", "HmxW4A8MatMulInterfaceProvider", "HTP"));

    Qnn_ContextHandle_t context = nullptr;
    const QnnContext_Config_t *ctxCfgs[] = {nullptr};
    QCHECK(g_qnn.contextCreate(backend, nullptr, ctxCfgs, &context));

    Qnn_GraphHandle_t graph = nullptr;
    const QnnGraph_Config_t *grCfgs[] = {nullptr};
    QCHECK(g_qnn.graphCreate(context, "w4a8_matmul_graph", grCfgs, &graph));

    /* Tensor dims: [1, 1, *, *] for Flat4 compatibility. */
    uint32_t aDims[] = {1, 1, (uint32_t)M, (uint32_t)K};
    uint32_t wDims[] = {1, 1, (uint32_t)K, (uint32_t)N};
    uint32_t oDims[] = {1, 1, (uint32_t)M, (uint32_t)N};
    /* Per-slice VTCM: fixed 8 KiB + pre-packed act (K*128 B). */
    const uint32_t per_slice_vtcm = 8 * 1024 + K * 128;
    const uint32_t vtcm_bytes = per_slice_vtcm * 4;
    uint32_t sDims[] = {1, 1, 1, vtcm_bytes};

    auto aT = mk_tensor("activation", QNN_TENSOR_TYPE_APP_WRITE,
                        QNN_DATATYPE_INT_8, 4, aDims);
    auto wT = mk_tensor("weight", QNN_TENSOR_TYPE_APP_WRITE,
                        QNN_DATATYPE_INT_8, 4, wDims);
    auto sT = mk_tensor("scratch", QNN_TENSOR_TYPE_APP_WRITE,
                        QNN_DATATYPE_UINT_8, 4, sDims);
    auto oT = mk_tensor("result", QNN_TENSOR_TYPE_APP_READ,
                        QNN_DATATYPE_INT_32, 4, oDims);

    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &aT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &wT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &sT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &oT));

    Qnn_Tensor_t ins[] = {aT, wT, sT};
    Qnn_OpConfig_t op{};
    op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name = "mm0";
    op.v1.packageName = "HmxW4A8MatMulPackage";
    op.v1.typeName = "MatMulInt4xInt8";
    op.v1.numOfParams = 0;
    op.v1.params = nullptr;
    op.v1.numOfInputs = 3;
    op.v1.inputTensors = ins;
    op.v1.numOfOutputs = 1;
    op.v1.outputTensors = &oT;
    QCHECK(g_qnn.graphAddNode(graph, op));

    /* The HTP backend ties the profile handle to the graph at finalize
     * time; subsequent graphExecute calls must pass the same handle (or
     * nullptr). Passing a fresh handle per iter returns err 6006. */
    Qnn_ProfileHandle_t profile = nullptr;
    QCHECK(g_qnn.profileCreate(backend, QNN_PROFILE_LEVEL_DETAILED, &profile));
    QCHECK(g_qnn.graphFinalize(graph, profile, nullptr));
    std::printf("[Graph] finalized\n");

    /* Fill inputs deterministically. */
    std::vector<int8_t>  aBuf(M * K);
    std::vector<int8_t>  wBuf(K * N);
    std::vector<uint8_t> sBuf(vtcm_bytes);
    std::vector<int32_t> oBuf(M * N), oRef(M * N);
    for (int i = 0; i < M * K; i++) aBuf[i] = (int8_t)(((i * 37) & 0xFF) - 128);
    for (int i = 0; i < K * N; i++) wBuf[i] = (int8_t)(((i * 13) % 15) - 7);

    Qnn_Tensor_t eIn[3], eOut[1];
    std::memcpy(&eIn[0], &aT, sizeof(Qnn_Tensor_t));
    eIn[0].v1.clientBuf = {aBuf.data(), (uint32_t)(aBuf.size() * sizeof(int8_t))};
    std::memcpy(&eIn[1], &wT, sizeof(Qnn_Tensor_t));
    eIn[1].v1.clientBuf = {wBuf.data(), (uint32_t)(wBuf.size() * sizeof(int8_t))};
    std::memcpy(&eIn[2], &sT, sizeof(Qnn_Tensor_t));
    eIn[2].v1.clientBuf = {sBuf.data(), (uint32_t)(sBuf.size() * sizeof(uint8_t))};
    std::memcpy(&eOut[0], &oT, sizeof(Qnn_Tensor_t));
    eOut[0].v1.clientBuf = {oBuf.data(), (uint32_t)(oBuf.size() * sizeof(int32_t))};

    /* Warmup — first invocation pays VTCM acquire + HMX power-on. */
    QCHECK(g_qnn.graphExecute(graph, eIn, 3, eOut, 1, profile, nullptr));

    auto cycles_from_profile = [](Qnn_ProfileHandle_t p) -> uint64_t {
        const QnnProfile_EventId_t *evs = nullptr;
        uint32_t ne = 0;
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

    /* Each graphExecute replaces the profile's events — `cycles_from_profile`
     * reads the per-iter value directly. */
    uint64_t total_cycles = 0;
    uint64_t min_cycles = UINT64_MAX, max_cycles = 0;
    for (int it = 0; it < ITERS; it++) {
        QCHECK(g_qnn.graphExecute(graph, eIn, 3, eOut, 1, profile, nullptr));
        uint64_t cyc = cycles_from_profile(profile);
        total_cycles += cyc;
        if (cyc < min_cycles) min_cycles = cyc;
        if (cyc > max_cycles) max_cycles = cyc;
    }
    uint64_t avg_cycles = total_cycles / ITERS;

    /* Reference + diff. */
    ref_matmul(oRef.data(), aBuf.data(), wBuf.data(), M, K, N);
    /* ref_matmul reuses aBuf as int8_t (declared so above). */
    int mismatches = 0;
    int64_t max_abs_err = 0;
    for (int i = 0; i < M * N; i++) {
        int64_t diff = (int64_t)oBuf[i] - (int64_t)oRef[i];
        if (diff != 0) {
            mismatches++;
            if (std::abs(diff) > max_abs_err) max_abs_err = std::abs(diff);
        }
    }
    std::printf("[Steady] cycles: avg=%llu  min=%llu  max=%llu  (n=%d)\n",
                (unsigned long long)avg_cycles,
                (unsigned long long)min_cycles,
                (unsigned long long)max_cycles, ITERS);
    /* cycles per MAC (handy reference vs the HMX theoretical floor). */
    uint64_t macs = (uint64_t)M * K * N;
    std::printf("[Steady] MACs=%llu  cycles_per_MAC=%.2f\n",
                (unsigned long long)macs, (double)avg_cycles / (double)macs);
    std::printf("[Check] mismatches=%d/%d max_abs_err=%lld\n",
                mismatches, M * N, (long long)max_abs_err);
    std::printf("  oBuf[0..3]=%d %d %d %d  oRef[0..3]=%d %d %d %d\n",
                oBuf[0], oBuf[1], oBuf[2], oBuf[3],
                oRef[0], oRef[1], oRef[2], oRef[3]);
    /* Diagnostic: show first 8 mismatching (row, col, diff). */
    /* Per-row mismatch count summary. */
    int row_bad[64] = {0};
    for (int i = 0; i < M * N; i++)
        if (oBuf[i] != oRef[i]) row_bad[(i/N) & 63]++;
    std::printf("  Row mismatch counts (first 32 rows):\n  ");
    for (int r = 0; r < 32 && r < M; r++) std::printf(" %d", row_bad[r]);
    std::printf("\n");

    g_qnn.profileFree(profile);
    g_qnn.contextFree(context, nullptr);
    g_qnn.backendFree(backend);
    return mismatches == 0 ? 0 : 1;
}
