/*
 * run_matmul_v7_graph.cpp — host harness for Phase 3D.3 split graph.
 *
 *   raw u8 act ──► PackActivationU8ToHmxTile [HVX, MT=4] ──► packed_act ┐
 *   raw i8 wt  ──► PackWeightToHmxTileV3     [HVX, MT=4] ──► packed_wt  │
 *                                                                       ▼
 *                                              MatMulV7 [HMX, MT=false]
 *                                                     │  │
 *                                             rb_lo   │  │ rb_hi
 *                                                     ▼  ▼
 *                                    RequantHvx [HVX, MT=4]
 *                                       (mult, shift) ──┘
 *                                                     │
 *                                                     ▼
 *                                                    out (i8)
 *
 * Design principle: our custom code does ONLY HMX MAC work (MatMulV7).
 * All HVX work (act pack, wt pack, requant) is in MT=true ops so QNN's
 * scheduler parallelizes them across 4 HVX threads and can overlap them
 * with the HMX resource (exclusive to MatMulV7).
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
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

/* Scalar reference quantized matmul (matches V6's semantics). */
static int32_t srdhm(int32_t x, int32_t y) {
    int64_t prod = (int64_t)x * (int64_t)y;
    int64_t nudge = (prod >= 0) ? (1LL << 30) : -(1LL << 30);
    int32_t result = (int32_t)((prod + nudge) >> 31);
    if (x == (int32_t)0x80000000 && y == (int32_t)0x80000000) result = 0x7fffffff;
    return result;
}

static int32_t rounding_shift_right(int32_t x, int shift) {
    if (shift <= 0) return x << (-shift);
    int32_t mask = (1 << shift) - 1;
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> shift) + (remainder > threshold ? 1 : 0);
}

static int8_t sat_i8(int32_t x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

static void ref_matmul_quant(
    int8_t *out, const uint8_t *au, const int8_t *w,
    const int32_t *multiplier, int32_t shift,
    int M, int K, int N)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++)
                s += (int32_t)au[i * K + k] * (int32_t)w[k * N + j];
            int32_t stage1 = srdhm(s, multiplier[j]);
            int32_t stage2 = rounding_shift_right(stage1, shift);
            out[i * N + j] = sat_i8(stage2);
        }
}

static void quantize_multiplier(double scale, int32_t *out_mult, int32_t *out_shift)
{
    if (scale <= 0.0) { *out_mult = 0; *out_shift = 0; return; }
    int exp_shift = 0;
    double frac = std::frexp(scale, &exp_shift);
    int64_t mult_q31 = (int64_t)std::round(frac * (double)(1LL << 31));
    if (mult_q31 == (1LL << 31)) { mult_q31 >>= 1; exp_shift += 1; }
    *out_mult  = (int32_t)mult_q31;
    *out_shift = -exp_shift;
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? std::atoi(argv[1]) : 32;
    int K = (argc > 2) ? std::atoi(argv[2]) : 32;
    int N = (argc > 3) ? std::atoi(argv[3]) : 32;
    int ITERS = (argc > 4) ? std::atoi(argv[4]) : 5;
    if (M % 32 || K % 32 || N % 32) {
        std::fprintf(stderr, "M, K, N must be multiples of 32\n");
        return 2;
    }
    const int M_tiles = M / 32;
    const int K_tiles = K / 32;
    const int N_tiles = N / 32;

    std::printf("== MatMulV7Graph (PackActU8 + PackWtV3 + V7 [HMX-only] + RequantHvx): %dx%dx%d ==\n", M, K, N);
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
    QCHECK(g_qnn.graphCreate(context, "matmul_v7_graph", grCfgs, &graph));

    uint32_t aDims[]     = {1, 1, (uint32_t)M, (uint32_t)K};
    uint32_t wDims[]     = {1, 1, (uint32_t)K, (uint32_t)N};
    uint32_t paDims[]    = {1, (uint32_t)M_tiles, (uint32_t)K_tiles, 2048};  /* 2-stream packed */
    uint32_t pwDims[]    = {1, (uint32_t)N_tiles, (uint32_t)K_tiles, 1024};  /* P2 packed */
    uint32_t rbDims[]    = {1, (uint32_t)M_tiles, (uint32_t)N_tiles, 1024};  /* u16 dual-scale */
    uint32_t multDims[]  = {1, 1, 1, (uint32_t)N};
    uint32_t shiftDims[] = {1, 1, 1, 1};
    uint32_t oDims[]     = {1, 1, (uint32_t)M, (uint32_t)N};
    /* V7 scratch: only bias (512 B used, round up to 2 KiB). */
    const uint32_t vtcm_bytes = 2 * 1024;
    uint32_t sDims[]     = {1, 1, 1, vtcm_bytes};

    /* Phase 3D.1 retained: weight is STATIC so pack_wt's own wu-keyed
     * cache (in pack_wt_v3_hvx.c) hits on inference 2+ — steady-state
     * cost is O(1). */
    std::vector<int8_t> wRaw(K * N);
    for (int i = 0; i < K * N; i++) wRaw[i] = (int8_t)(((i * 13) % 15) - 7);

    auto aT   = mk_tensor("act_raw",    QNN_TENSOR_TYPE_APP_WRITE,
                          QNN_DATATYPE_UINT_8, 4, aDims);
    auto wT   = mk_tensor("wt_raw",     QNN_TENSOR_TYPE_STATIC,
                          QNN_DATATYPE_UINT_8, 4, wDims);
    wT.v1.clientBuf = {wRaw.data(), (uint32_t)wRaw.size()};
    auto paT  = mk_tensor("packed_act", QNN_TENSOR_TYPE_NATIVE,
                          QNN_DATATYPE_UINT_8, 4, paDims);
    auto pwT  = mk_tensor("packed_wt",  QNN_TENSOR_TYPE_NATIVE,
                          QNN_DATATYPE_UINT_8, 4, pwDims);
    auto rbLoT= mk_tensor("rb_lo",      QNN_TENSOR_TYPE_NATIVE,
                          QNN_DATATYPE_UINT_16, 4, rbDims);
    auto rbHiT= mk_tensor("rb_hi",      QNN_TENSOR_TYPE_NATIVE,
                          QNN_DATATYPE_UINT_16, 4, rbDims);
    auto sT   = mk_tensor("scratch",    QNN_TENSOR_TYPE_APP_WRITE,
                          QNN_DATATYPE_UINT_8, 4, sDims);
    auto mT   = mk_tensor("multiplier", QNN_TENSOR_TYPE_APP_WRITE,
                          QNN_DATATYPE_INT_32, 4, multDims);
    auto shT  = mk_tensor("shift",      QNN_TENSOR_TYPE_APP_WRITE,
                          QNN_DATATYPE_INT_32, 4, shiftDims);
    auto oT   = mk_tensor("out",        QNN_TENSOR_TYPE_APP_READ,
                          QNN_DATATYPE_UINT_8, 4, oDims);   /* i8 bytes */

    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &aT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &wT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &paT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &pwT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &rbLoT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &rbHiT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &sT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &mT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &shT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &oT));

    /* Node 1: PackActivationU8ToHmxTile (HVX, MT=true). */
    {
        Qnn_Tensor_t ins[]  = {aT};
        Qnn_Tensor_t outs[] = {paT};
        Qnn_OpConfig_t op{};
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = "pack_act"; op.v1.packageName = "HmxMatMulPhase3Package";
        op.v1.typeName = "PackActivationU8ToHmxTile";
        op.v1.numOfParams = 0; op.v1.params = nullptr;
        op.v1.numOfInputs = 1; op.v1.inputTensors = ins;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = outs;
        QCHECK(g_qnn.graphAddNode(graph, op));
    }
    /* Node 2: PackWeightToHmxTileV3 (HVX, MT=true; cached via wu pointer). */
    {
        Qnn_Tensor_t ins[]  = {wT};
        Qnn_Tensor_t outs[] = {pwT};
        Qnn_OpConfig_t op{};
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = "pack_wt"; op.v1.packageName = "HmxMatMulPhase3Package";
        op.v1.typeName = "PackWeightToHmxTileV3";
        op.v1.numOfParams = 0; op.v1.params = nullptr;
        op.v1.numOfInputs = 1; op.v1.inputTensors = ins;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = outs;
        QCHECK(g_qnn.graphAddNode(graph, op));
    }
    /* Node 3: MatMulV7 — HMX-only MAC + dual-scale readback.
     * Outputs u16 lo/hi tensors for RequantHvx to consume. */
    {
        Qnn_Tensor_t ins[]  = {paT, pwT, sT};
        Qnn_Tensor_t outs[] = {rbLoT, rbHiT};
        Qnn_OpConfig_t op{};
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = "mmv7"; op.v1.packageName = "HmxMatMulPhase3Package";
        op.v1.typeName = "MatMulV7";
        op.v1.numOfParams = 0; op.v1.params = nullptr;
        op.v1.numOfInputs = 3; op.v1.inputTensors = ins;
        op.v1.numOfOutputs = 2; op.v1.outputTensors = outs;
        QCHECK(g_qnn.graphAddNode(graph, op));
    }
    /* Node 4: RequantHvx — HVX MT=4. Decode dual-scale, SRDHM + shift + clip. */
    {
        Qnn_Tensor_t ins[] = {rbLoT, rbHiT, mT, shT};
        Qnn_OpConfig_t op{};
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = "requant"; op.v1.packageName = "HmxMatMulPhase3Package";
        op.v1.typeName = "RequantHvx";
        op.v1.numOfParams = 0; op.v1.params = nullptr;
        op.v1.numOfInputs = 4; op.v1.inputTensors = ins;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = &oT;
        QCHECK(g_qnn.graphAddNode(graph, op));
    }

    Qnn_ProfileHandle_t profile = nullptr;
    QCHECK(g_qnn.profileCreate(backend, QNN_PROFILE_LEVEL_DETAILED, &profile));
    QCHECK(g_qnn.graphFinalize(graph, profile, nullptr));
    std::printf("[Graph] finalized (V7: pack_act + pack_wt + V7[HMX] + requant_hvx)\n");

    /* Client-side data. (wRaw already initialized above for STATIC wT.) */
    std::vector<uint8_t> aRaw(M * K);
    for (int i = 0; i < M * K; i++) aRaw[i] = (uint8_t)((i * 37) & 0xFF);

    /* Build quantization parameters (same recipe as run_matmul_v6.cpp). */
    std::vector<int32_t> multArr(N);
    int32_t shift_val = 0;
    {
        int32_t mults[32], shifts[32];
        for (int n = 0; n < 32; n++) {
            double scale_n = 1.0 / ((double)K * (1.0 + 0.1 * (n % 7)));
            quantize_multiplier(scale_n, &mults[n], &shifts[n]);
        }
        int32_t max_shift = shifts[0];
        for (int n = 1; n < 32; n++) if (shifts[n] > max_shift) max_shift = shifts[n];
        shift_val = max_shift;
        for (int n = 0; n < N; n++) {
            int pn = n % 32;
            int32_t m = mults[pn];
            int32_t sd = max_shift - shifts[pn];
            if (sd > 0) m = m >> sd;
            multArr[n] = m;
        }
    }
    std::printf("[Quant] mult[0..3]=%d %d %d %d  shift=%d\n",
                multArr[0], multArr[1], multArr[2], multArr[3], shift_val);

    std::vector<uint8_t> sBuf(vtcm_bytes);
    std::vector<int8_t>  oBuf(M * N), oRef(M * N);

    /* Phase 3D.1: wT is STATIC, provided at graph-create time — NOT in
     * eIn. Runtime inputs are now 4: act, scratch, multiplier, shift. */
    Qnn_Tensor_t eIn[4], eOut[1];
    std::memcpy(&eIn[0], &aT, sizeof(Qnn_Tensor_t));
    eIn[0].v1.clientBuf = {aRaw.data(), (uint32_t)aRaw.size()};
    std::memcpy(&eIn[1], &sT, sizeof(Qnn_Tensor_t));
    eIn[1].v1.clientBuf = {sBuf.data(), (uint32_t)sBuf.size()};
    std::memcpy(&eIn[2], &mT, sizeof(Qnn_Tensor_t));
    eIn[2].v1.clientBuf = {multArr.data(), (uint32_t)(multArr.size() * sizeof(int32_t))};
    std::memcpy(&eIn[3], &shT, sizeof(Qnn_Tensor_t));
    eIn[3].v1.clientBuf = {&shift_val, (uint32_t)sizeof(shift_val)};
    std::memcpy(&eOut[0], &oT, sizeof(Qnn_Tensor_t));
    eOut[0].v1.clientBuf = {oBuf.data(), (uint32_t)oBuf.size()};

    QCHECK(g_qnn.graphExecute(graph, eIn, 4, eOut, 1, profile, nullptr));  /* warmup */

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
        for (uint32_t i = 0; i < ne; i++) {
            QnnProfile_EventData_t d{};
            if (g_qnn.profileGetEventData(evs[i], &d) != QNN_SUCCESS) continue;
            if (d.identifier && std::strstr(d.identifier, "Accelerator (execute) time (cycles)")) {
                std::printf("[Profile] top: %llu cycles ('%s')\n",
                    (unsigned long long)d.value, d.identifier);
                const QnnProfile_EventId_t *subs = nullptr; uint32_t ns = 0;
                if (g_qnn.profileGetSubEvents(evs[i], &subs, &ns) == QNN_SUCCESS) {
                    for (uint32_t j = 0; j < ns; j++) {
                        QnnProfile_EventData_t ds{};
                        if (g_qnn.profileGetEventData(subs[j], &ds) == QNN_SUCCESS)
                            std::printf("  %s: %llu\n",
                                ds.identifier ? ds.identifier : "(null)",
                                (unsigned long long)ds.value);
                    }
                }
            }
        }
    };

    uint64_t total = 0, mn_ = UINT64_MAX, mx = 0;
    for (int it = 0; it < ITERS; it++) {
        QCHECK(g_qnn.graphExecute(graph, eIn, 4, eOut, 1, profile, nullptr));
        uint64_t c = cycles_from_profile(profile);
        total += c; if (c < mn_) mn_ = c; if (c > mx) mx = c;
    }
    uint64_t avg = total / ITERS;
    dump_profile(profile);

    ref_matmul_quant(oRef.data(), aRaw.data(), wRaw.data(),
                     multArr.data(), shift_val, M, K, N);

    int bad = 0; int32_t merr = 0;
    for (int i = 0; i < M * N; i++) {
        int32_t d = (int32_t)oBuf[i] - (int32_t)oRef[i];
        if (d) { bad++; if (std::abs(d) > merr) merr = std::abs(d); }
    }

    uint64_t macs = (uint64_t)M * K * N;
    std::printf("[Steady] cycles: avg=%llu min=%llu max=%llu\n",
                (unsigned long long)avg, (unsigned long long)mn_, (unsigned long long)mx);
    std::printf("[Steady] MACs=%llu  cycles_per_MAC=%.4f\n",
                (unsigned long long)macs, (double)avg / (double)macs);
    std::printf("[Check] mismatches=%d/%d max_abs_err=%d\n", bad, M*N, merr);
    std::printf("  oBuf[0..7]=%d %d %d %d %d %d %d %d\n",
                oBuf[0], oBuf[1], oBuf[2], oBuf[3], oBuf[4], oBuf[5], oBuf[6], oBuf[7]);
    std::printf("  oRef[0..7]=%d %d %d %d %d %d %d %d\n",
                oRef[0], oRef[1], oRef[2], oRef[3], oRef[4], oRef[5], oRef[6], oRef[7]);

    g_qnn.profileFree(profile);
    g_qnn.contextFree(context, nullptr);
    g_qnn.backendFree(backend);
    return 0;
}
