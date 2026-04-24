/*
 * run_matmul_v6.cpp — host harness for MatMulV6 (quantized u8 × i8 → i8).
 *
 * Symmetric quantization: zp_a = zp_w = zp_out = 0.
 * Per-tensor activation scale + per-channel weight scale + per-tensor out scale
 * are pre-folded into int32 Q0.31 multiplier[N] + shared shift.
 *
 * Host packs activation (2-stream) + weight (P2 4-K-row) — same layout as V3b.
 * Then calls V6 op which produces i8 output after requant.
 * Reference scalar implementation verifies bit-exactness.
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

static void pack_activation_32x32_u8(uint8_t *tile, const uint8_t *au,
                                      int M_full, int K_full,
                                      int m0, int k0) {
    memset(tile, 0, 2048);
    for (int phys_row = 0; phys_row < 16; phys_row++) {
        uint32_t *dst = (uint32_t *)(tile + 128 * phys_row);
        const uint8_t *s0 = &au[(m0 + phys_row)      * K_full + k0];
        const uint8_t *s1 = &au[(m0 + phys_row + 16) * K_full + k0];
        for (int k = 0; k < 32; k++)
            dst[k] = ((uint32_t)s1[k] << 24) | ((uint32_t)s0[k] << 8);
    }
}

static void pack_weight_32x32_i8(uint8_t *tile, const int8_t *wu,
                                  int K_full, int N_full,
                                  int k0, int n0) {
    for (int kg = 0; kg < 8; kg++) {
        uint32_t *dst = (uint32_t *)(tile + 128 * kg);
        const uint8_t *r0 = (const uint8_t *)&wu[(k0 + kg*4 + 0) * N_full + n0];
        const uint8_t *r1 = (const uint8_t *)&wu[(k0 + kg*4 + 1) * N_full + n0];
        const uint8_t *r2 = (const uint8_t *)&wu[(k0 + kg*4 + 2) * N_full + n0];
        const uint8_t *r3 = (const uint8_t *)&wu[(k0 + kg*4 + 3) * N_full + n0];
        for (int col = 0; col < 32; col++) {
            dst[col] =  (uint32_t)r0[col]
                     | ((uint32_t)r1[col] << 8)
                     | ((uint32_t)r2[col] << 16)
                     | ((uint32_t)r3[col] << 24);
        }
    }
}

/* ---- Reference: TFLite-style saturating rounding doubling high mul. ---- */
static int32_t srdhm(int32_t x, int32_t y) {
    int64_t prod = (int64_t)x * (int64_t)y;
    int64_t nudge = (prod >= 0) ? (1LL << 30) : -(1LL << 30);
    int32_t result = (int32_t)((prod + nudge) >> 31);
    if (x == (int32_t)0x80000000 && y == (int32_t)0x80000000)
        result = 0x7fffffff;
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

/* Reference quantized matmul: u8 act × i8 wt → int32 acc → per-channel
 * multiplier[n] (Q0.31) + common shift → saturate i8. */
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

/* Encode a float scale into int32 Q0.31 multiplier + shift.
 * Returns (multiplier, shift) such that
 *   float_scale ≈ multiplier × 2^(-31 - shift)
 * with shift in [0, 31]. */
static void quantize_multiplier(double scale, int32_t *out_mult, int32_t *out_shift)
{
    if (scale <= 0.0) { *out_mult = 0; *out_shift = 0; return; }
    int exp_shift = 0;
    double frac = std::frexp(scale, &exp_shift);   /* scale = frac * 2^exp_shift, frac ∈ [0.5, 1) */
    /* Represent frac as Q0.31 int32: frac × 2^31. frac∈[0.5,1) → [2^30, 2^31). */
    int64_t mult_q31 = (int64_t)std::round(frac * (double)(1LL << 31));
    if (mult_q31 == (1LL << 31)) {
        mult_q31 >>= 1;
        exp_shift += 1;
    }
    /* total_shift_bits = 31 - exp_shift (right shift after srdhm's 2^-31). */
    int32_t total_shift = 31 - exp_shift - 31;   /* after srdhm already divided by 2^31 */
    /* srdhm returns (x*y + 2^30) >> 31 which is x*y * 2^-31 (rounded).
     * So x*float_scale ≈ srdhm(x, mult_q31) << exp_shift  →  shift = -exp_shift.
     * With shift convention (positive = right shift), shift = -exp_shift. */
    *out_mult  = (int32_t)mult_q31;
    *out_shift = -exp_shift;
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

    std::printf("== MatMulV6 (quantized u8 × i8 → i8): %dx%dx%d ==\n", M, K, N);
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
    QCHECK(g_qnn.graphCreate(context, "matmul_v6", grCfgs, &graph));

    uint32_t paDims[] = {1, (uint32_t)M_tiles, (uint32_t)K_tiles, 2048};
    uint32_t pwDims[] = {1, (uint32_t)N_tiles, (uint32_t)K_tiles, 1024};
    uint32_t multDims[] = {1, 1, 1, (uint32_t)N};
    uint32_t shiftDims[] = {1, 1, 1, 1};
    uint32_t oDims[]  = {1, 1, (uint32_t)M, (uint32_t)N};
    const uint32_t vtcm_bytes = 16 * 1024;
    uint32_t sDims[] = {1, 1, 1, vtcm_bytes};

    auto paT  = mk_tensor("packed_act", QNN_TENSOR_TYPE_APP_WRITE,
                          QNN_DATATYPE_UINT_8, 4, paDims);
    auto pwT  = mk_tensor("packed_wt",  QNN_TENSOR_TYPE_APP_WRITE,
                          QNN_DATATYPE_UINT_8, 4, pwDims);
    auto sT   = mk_tensor("scratch",    QNN_TENSOR_TYPE_APP_WRITE,
                          QNN_DATATYPE_UINT_8, 4, sDims);
    auto mT   = mk_tensor("multiplier", QNN_TENSOR_TYPE_APP_WRITE,
                          QNN_DATATYPE_INT_32, 4, multDims);
    auto shT  = mk_tensor("shift",      QNN_TENSOR_TYPE_APP_WRITE,
                          QNN_DATATYPE_INT_32, 4, shiftDims);
    /* Use UINT_8 to match the QUInt8 internal rep QNN insists on — the
     * byte stream encodes int8 values directly (symmetric zp=0). */
    auto oT   = mk_tensor("out",        QNN_TENSOR_TYPE_APP_READ,
                          QNN_DATATYPE_UINT_8, 4, oDims);

    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &paT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &pwT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &sT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &mT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &shT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &oT));

    Qnn_Tensor_t ins[] = {paT, pwT, sT, mT, shT};
    Qnn_OpConfig_t op{};
    op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name = "mmv6";
    op.v1.packageName = "HmxMatMulPhase3Package";
    op.v1.typeName = "MatMulV6";
    op.v1.numOfParams = 0;
    op.v1.params = nullptr;
    op.v1.numOfInputs = 5;
    op.v1.inputTensors = ins;
    op.v1.numOfOutputs = 1;
    op.v1.outputTensors = &oT;
    QCHECK(g_qnn.graphAddNode(graph, op));

    Qnn_ProfileHandle_t profile = nullptr;
    QCHECK(g_qnn.profileCreate(backend, QNN_PROFILE_LEVEL_DETAILED, &profile));
    QCHECK(g_qnn.graphFinalize(graph, profile, nullptr));
    std::printf("[Graph] finalized (V6 quantized u8×i8→i8)\n");

    /* Test data: u8 act, i8 wt (range ±7). Quantization parameters:
     * scale_a = 1/255 (maps u8 [0,255] → float [0,1])
     * scale_w[n] = some per-channel factor (vary by n for test coverage)
     * scale_out = scale that keeps typical acc in i8 range after requant.
     *
     * For test purposes we just fix them to values that exercise both
     * small and large multipliers. */
    std::vector<uint8_t> aRaw(M * K);
    std::vector<int8_t>  wRaw(K * N);
    for (int i = 0; i < M * K; i++) aRaw[i] = (uint8_t)((i * 37) & 0xFF);
    for (int i = 0; i < K * N; i++) wRaw[i] = (int8_t)(((i * 13) % 15) - 7);

    /* Build multiplier[n]: target ~ 1/K so typical dot products end up
     * in i8 range. Vary by n to cover per-channel semantics. */
    std::vector<int32_t> multArr(N);
    int32_t shift_val = 0;
    {
        /* Pick scale_float[n] = 1.0 / (K * (1 + 0.1*(n%7))) so each channel
         * has slightly different rescale factor. */
        double scale_n[32];
        int32_t mults[32];
        int32_t shifts[32];
        for (int n = 0; n < 32; n++) {
            scale_n[n] = 1.0 / ((double)K * (1.0 + 0.1 * (n % 7)));
            quantize_multiplier(scale_n[n], &mults[n], &shifts[n]);
        }
        /* Use the max shift across all channels; re-scale mults to common shift. */
        int32_t max_shift = shifts[0];
        for (int n = 1; n < 32; n++) if (shifts[n] > max_shift) max_shift = shifts[n];
        shift_val = max_shift;
        for (int n = 0; n < N; n++) {
            int pat_n = n % 32;
            int32_t m_raw = mults[pat_n];
            int32_t sh_diff = max_shift - shifts[pat_n];
            if (sh_diff > 0) m_raw = m_raw >> sh_diff;  /* reduce precision to match common shift */
            multArr[n] = m_raw;
        }
    }
    std::printf("[Quant] multiplier sample: %d %d %d %d ...  shift=%d\n",
                multArr[0], multArr[1], multArr[2], multArr[3], shift_val);

    /* Host pre-pack. */
    std::vector<uint8_t> packedAct((size_t)M_tiles * K_tiles * 2048);
    std::vector<uint8_t> packedWt ((size_t)N_tiles * K_tiles * 1024);
    for (int mt = 0; mt < M_tiles; mt++)
        for (int kt = 0; kt < K_tiles; kt++)
            pack_activation_32x32_u8(
                &packedAct[(size_t)(mt * K_tiles + kt) * 2048],
                aRaw.data(), M, K, mt * 32, kt * 32);
    for (int nt = 0; nt < N_tiles; nt++)
        for (int kt = 0; kt < K_tiles; kt++)
            pack_weight_32x32_i8(
                &packedWt[(size_t)(nt * K_tiles + kt) * 1024],
                wRaw.data(), K, N, kt * 32, nt * 32);

    std::vector<uint8_t> sBuf(vtcm_bytes);
    std::vector<int8_t>  oBuf(M * N), oRef(M * N);

    Qnn_Tensor_t eIn[5], eOut[1];
    std::memcpy(&eIn[0], &paT, sizeof(Qnn_Tensor_t));
    eIn[0].v1.clientBuf = {packedAct.data(), (uint32_t)packedAct.size()};
    std::memcpy(&eIn[1], &pwT, sizeof(Qnn_Tensor_t));
    eIn[1].v1.clientBuf = {packedWt.data(), (uint32_t)packedWt.size()};
    std::memcpy(&eIn[2], &sT, sizeof(Qnn_Tensor_t));
    eIn[2].v1.clientBuf = {sBuf.data(), (uint32_t)sBuf.size()};
    std::memcpy(&eIn[3], &mT, sizeof(Qnn_Tensor_t));
    eIn[3].v1.clientBuf = {multArr.data(), (uint32_t)(multArr.size() * sizeof(int32_t))};
    std::memcpy(&eIn[4], &shT, sizeof(Qnn_Tensor_t));
    eIn[4].v1.clientBuf = {&shift_val, (uint32_t)sizeof(shift_val)};
    std::memcpy(&eOut[0], &oT, sizeof(Qnn_Tensor_t));
    eOut[0].v1.clientBuf = {oBuf.data(), (uint32_t)(oBuf.size() * sizeof(int8_t))};

    QCHECK(g_qnn.graphExecute(graph, eIn, 5, eOut, 1, profile, nullptr));

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

    uint64_t total = 0, mn_ = UINT64_MAX, mx = 0;
    for (int it = 0; it < ITERS; it++) {
        QCHECK(g_qnn.graphExecute(graph, eIn, 5, eOut, 1, profile, nullptr));
        uint64_t c = cycles_from_profile(profile);
        total += c; if (c < mn_) mn_ = c; if (c > mx) mx = c;
    }
    uint64_t avg = total / ITERS;

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
    return bad == 0 ? 0 : 1;
}
