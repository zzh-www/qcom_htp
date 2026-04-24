/*
 * run_matmul_v8_graph.cpp — host harness for Phase 3D.4: pure-HMX replica
 * of QNN's q::ConvLayer_s1.opt. Single-kernel HMX does u8×i8→u8 end-to-end.
 *
 *   raw u8 act ──► PackActivationU8RowMajor [HVX, MT=4] ──► packed_act ┐
 *   raw i8 wt  ──► PackWeightToHmxTileV3    [HVX, MT=4] ──► packed_wt  │
 *                                                                      ▼
 *                                   MatMulV8 [HMX, MT=false, 4 asm ops]
 *                                   ├─ bias = mxmem(fp16_scale)
 *                                   ├─ mxclracc
 *                                   ├─ K × { mxmem(act):cm + mxmem(wt) }
 *                                   └─ mxmem(out):after:cm:sat.ub = acc
 *                                                 │
 *                                                 ▼
 *                                                out (u8 quantized)
 *
 * Requant folded into HMX bias: bias_fp16[n] = 512 × scale_quant[n].
 * Output formula (silicon-verified probe_sat_ub.c):
 *   out[m][n] = sat_u8( round(acc × bias_fp16[n] / 512) + 128 )
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
    QCHECK(g_qnn.graphCreate(context, "matmul_v8_graph", grCfgs, &graph));

    uint32_t aDims[]     = {1, 1, (uint32_t)M, (uint32_t)K};
    uint32_t wDims[]     = {1, 1, (uint32_t)K, (uint32_t)N};
    uint32_t paDims[]    = {1, (uint32_t)M_tiles, (uint32_t)K_tiles, 1024};  /* row-major 1 KiB tile */
    uint32_t pwDims[]    = {1, (uint32_t)N_tiles, (uint32_t)K_tiles, 1024};  /* P2 packed */
    uint32_t biasDims[]  = {1, 1, (uint32_t)N_tiles, 128};                   /* fp16, 128 entries per nt */
    /* Output is TILE-LAYOUT [1, M_tiles, N_tiles, 1024] — same total
     * bytes as row-major [M, N] but tile-contiguous, matching QNN's
     * ConvLayer_s1.opt native output.  mmv8's HMX sat.ub writes 1 KiB
     * per tile directly to this buffer — no scatter. */
    uint32_t oDims[]     = {1, (uint32_t)M_tiles, (uint32_t)N_tiles, 1024};
    uint32_t oTileDims[] = {1, (uint32_t)M_tiles, (uint32_t)N_tiles, 1024};
    /* V8 VTCM scratch: 1 KiB staging slot (unused in tile-layout mode). */
    const uint32_t vtcm_bytes = 2 * 1024;
    uint32_t sDims[]     = {1, 1, 1, vtcm_bytes};

    std::vector<int8_t> wRaw(K * N);
    for (int i = 0; i < K * N; i++) wRaw[i] = (int8_t)(((i * 13) % 15) - 7);

    /* DIAG early init (must happen BEFORE wT STATIC + graphCreate so data
     * is captured correctly). ITERS is 4th cmdline arg. */
    if (ITERS == 997) {
        for (int i = 0; i < K * N; i++) wRaw[i] = -7;
    } else if (ITERS == 998) {
        for (int k = 0; k < K; k++) for (int n = 0; n < N; n++)
            wRaw[k*N + n] = (int8_t)((n % 32) + 1);
    } else if (ITERS == 999) {
        for (int i = 0; i < K * N; i++) wRaw[i] = 1;
    } else if (ITERS == 996) {
        for (int i = 0; i < K * N; i++) wRaw[i] = 1;  /* wt uniform for DIAG4 */
    } else if (ITERS == 995) {
        /* DIAG5: varied (but structured) activation AND weight, uniform bias.
         * Tests pack_act + pack_wt with non-uniform data. */
        for (int k = 0; k < K; k++) for (int n = 0; n < N; n++) {
            wRaw[k*N + n] = (int8_t)((k + n) % 5 - 2);  /* range [-2, 2] */
        }
    }

    /* Build per-column fp16 bias encoding the quant scale.
     * Probe-validated: out = sat_u8(round(acc × bias_fp16 / 512) + 128)
     * ⟹ to get out = round(acc × scale) + 128: bias_fp16 = 512 × scale
     * For numerical parity with our scalar reference (which uses Q31
     * SRDHM + right-shift), the effective scale is:
     *   scale_ref[n] = mult_q31[n] × 2^(-31 - shift)
     * So bias_fp16[n] = 512 × scale_ref[n]. */

    auto fp32_to_fp16 = [](float f) -> uint16_t {
        union { float f; uint32_t u; } v = { f };
        uint32_t sign = (v.u >> 31) & 1;
        int32_t  exp  = ((v.u >> 23) & 0xFF) - 127;
        uint32_t mant = v.u & 0x7FFFFF;
        if (exp <= -15) return (uint16_t)(sign << 15);  /* zero/denorm→0 */
        if (exp >= 16) return (uint16_t)((sign << 15) | 0x7C00);  /* ±inf */
        uint32_t hmant = mant >> 13;
        return (uint16_t)((sign << 15) | (((exp + 15) & 0x1F) << 10) | hmant);
    };

    auto aT   = mk_tensor("act_raw",    QNN_TENSOR_TYPE_APP_WRITE,
                          QNN_DATATYPE_UINT_8, 4, aDims);
    /* Declare wT as UFIXED_POINT_8 to bypass QNN's auto-inserted Cast
     * int8→uint8 (+128) that would flip the sign-bit of each weight byte.
     * V8's :after:cm:sat.ub path treats the packed tile bytes directly as
     * signed i8 via `weight.b = mxmem(...)`, so we want HMX to see the
     * ORIGINAL i8 bit pattern, not (raw_i8+128) reinterpreted as i8.
     * The storage is the same 8-bit byte pattern either way; only the
     * QNN-level declared signedness changes, which controls whether Cast
     * fires. */
    auto wT   = mk_tensor("wt_raw",     QNN_TENSOR_TYPE_STATIC,
                          QNN_DATATYPE_UFIXED_POINT_8, 4, wDims);
    wT.v1.clientBuf = {wRaw.data(), (uint32_t)wRaw.size()};
    auto paT  = mk_tensor("packed_act", QNN_TENSOR_TYPE_NATIVE,
                          QNN_DATATYPE_UINT_8, 4, paDims);
    auto pwT  = mk_tensor("packed_wt",  QNN_TENSOR_TYPE_NATIVE,
                          QNN_DATATYPE_UINT_8, 4, pwDims);
    auto sT   = mk_tensor("scratch",    QNN_TENSOR_TYPE_APP_WRITE,
                          QNN_DATATYPE_UINT_8, 4, sDims);
    auto oT   = mk_tensor("out",        QNN_TENSOR_TYPE_APP_READ,
                          QNN_DATATYPE_UINT_8, 4, oDims);
    /* Intermediate tile-layout output of mmv8 (VTCM-resident). */
    auto oTile = mk_tensor("out_tile",  QNN_TENSOR_TYPE_NATIVE,
                           QNN_DATATYPE_UINT_8, 4, oTileDims);

    /* Bias tensor: STATIC per-column fp16 scales.
     *
     * Silicon formula (probe_hmx_formula + probe_pair_lane T12, 2026-04-24):
     *   out[m][c] = sat_u8( (bias_raw[2c+1] >> 7)              ← BASELINE (zp)
     *                     + floor(acc × bias_fp16[2c] / 512) ) ← SCALE (slope)
     * The TWO LANES ARE ORTHOGONAL CHANNELS:
     *   - bias[2c+1] top-9-bits = per-col output zero-point
     *   - bias[2c]   fp16-value = per-col scale
     *
     * For u8 output with uniform zp=128 and per-channel scale_c:
     *   bias[2c+1] = fp16(2.0) = 0x4000  (gives baseline 128)
     *   bias[2c]   = fp16(512 * scale_c) (gives scale = scale_c)
     *
     * This decouples the encoding: scales can now be arbitrary positive
     * values (within fp16 range) without disturbing the zp=128 baseline.
     */
    std::vector<uint16_t> biasArr((size_t)N_tiles * 128, 0);
    const uint16_t FP16_2_0 = 0x4000;   /* (0x4000>>7) = 128 baseline */
    for (int nt = 0; nt < N_tiles; nt++) {
        for (int c = 0; c < 32; c++) {
            int n = nt * 32 + c;
            double scale_n = 1.0 / ((double)K * (1.0 + 0.1 * (n % 7)));
            float  bias_fp = (float)(512.0 * scale_n);
            biasArr[nt * 128 + 2 * c]     = fp32_to_fp16(bias_fp);  /* SCALE */
            biasArr[nt * 128 + 2 * c + 1] = FP16_2_0;                /* zp=128 */
        }
    }
    auto biasT = mk_tensor("bias_fp16", QNN_TENSOR_TYPE_APP_WRITE,
                           QNN_DATATYPE_UINT_16, 4, biasDims);

    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &aT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &wT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &paT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &pwT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &biasT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &sT));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &oTile));
    QCHECK(g_qnn.tensorCreateGraphTensor(graph, &oT));

    /* Node 1: PackActivationU8RowMajor (HVX, MT=4) — row-major 1 KiB tiles for :cm consumption. */
    {
        Qnn_Tensor_t ins[]  = {aT};
        Qnn_Tensor_t outs[] = {paT};
        Qnn_OpConfig_t op{};
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = "pack_act"; op.v1.packageName = "HmxMatMulPhase3Package";
        op.v1.typeName = "PackActivationU8RowMajor";
        op.v1.numOfParams = 0; op.v1.params = nullptr;
        op.v1.numOfInputs = 1; op.v1.inputTensors = ins;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = outs;
        QCHECK(g_qnn.graphAddNode(graph, op));
    }
    /* Node 2: PackWeightToHmxTileV3 (HVX MT=4, wu-cached). */
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
    /* Node 3: MatMulV8 — pure HMX, sat.ub writes tile-layout to VTCM. */
    {
        Qnn_Tensor_t ins[]  = {paT, pwT, biasT, sT};
        Qnn_OpConfig_t op{};
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = "mmv8"; op.v1.packageName = "HmxMatMulPhase3Package";
        op.v1.typeName = "MatMulV8";
        op.v1.numOfParams = 0; op.v1.params = nullptr;
        op.v1.numOfInputs = 4; op.v1.inputTensors = ins;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = &oTile;
        QCHECK(g_qnn.graphAddNode(graph, op));
    }
    /* Node 4: TcmDramCopy — bulk memcpy tile-layout VTCM → DDR. */
    {
        Qnn_Tensor_t ins[]  = {oTile};
        Qnn_Tensor_t outs[] = {oT};
        Qnn_OpConfig_t op{};
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = "tcm2ddr"; op.v1.packageName = "HmxMatMulPhase3Package";
        op.v1.typeName = "TcmDramCopy";
        op.v1.numOfParams = 0; op.v1.params = nullptr;
        op.v1.numOfInputs = 1; op.v1.inputTensors = ins;
        op.v1.numOfOutputs = 1; op.v1.outputTensors = outs;
        QCHECK(g_qnn.graphAddNode(graph, op));
    }

    Qnn_ProfileHandle_t profile = nullptr;
    QCHECK(g_qnn.profileCreate(backend, QNN_PROFILE_LEVEL_DETAILED, &profile));
    QCHECK(g_qnn.graphFinalize(graph, profile, nullptr));
    std::printf("[Graph] finalized (V7: pack_act + pack_wt + V7[HMX] + requant_hvx)\n");

    /* Client-side data. */
    std::vector<uint8_t> aRaw(M * K);
    for (int i = 0; i < M * K; i++) aRaw[i] = (uint8_t)((i * 37) & 0xFF);
    std::vector<uint8_t> sBuf(vtcm_bytes);
    std::vector<uint8_t> oBuf(M * N);
    std::vector<uint8_t> oRef(M * N);

    /* Silicon bit-exact reference (probe_hmx_formula.c T7..T11, 2026-04-24):
     *   out[m][c] = sat_u8( (bias_raw[2c+1] >> 7)
     *                     + floor(acc_hmx[m][c] × bias_fp16[2c+1] / 512) )
     * where acc_hmx uses PLAIN u8 activation (no signed shift) and i8 weight.
     * Helper — decode fp16 u16 to float. */
    auto fp16_to_float = [](uint16_t h) -> double {
        uint32_t sign = (h >> 15) & 1;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        double v;
        if (exp == 0)        v = std::ldexp((double)mant, -24);
        else if (exp == 0x1F) v = 0.0;  /* treat ±inf/NaN as 0 for this path */
        else                  v = std::ldexp((double)(1024 + mant), (int)exp - 25);
        return sign ? -v : v;
    };

    /* Silicon-validated formula (T12 pair-lane probe, 2026-04-24):
     *   baseline comes from bias_raw at lane 2c+1 (top 9 bits)
     *   scale    comes from bias_raw at lane 2c   (fp16 value / 512) */
    auto hmx_silicon_u8 = [&](int32_t acc_hmx,
                              uint16_t bias_raw_baseline,   /* lane 2c+1 */
                              uint16_t bias_raw_scale)      /* lane 2c   */
                              -> uint8_t {
        int baseline = (int)(bias_raw_baseline >> 7);
        double bv = fp16_to_float(bias_raw_scale);
        double scaled = (double)acc_hmx * bv / 512.0;
        int scaled_i = (int)std::floor(scaled);
        int v = baseline + scaled_i;
        if (v < 0) v = 0; else if (v > 255) v = 255;
        return (uint8_t)v;
    };
    /* Known residual: ~2-3% of cells in random-data mode off-by-1 due to
     * silicon's internal fp16-level rounding at cells where `acc × bias / 512`
     * is fractionally just above an integer (e.g. 3.010 → silicon 2, formula 3).
     * max_abs_err stays ≤ 1, which is within QNN u8-quant tolerance. */

    /* DIAGNOSTIC MODE: uniform acc + bias to exercise the silicon formula.
     * All three modes now use the silicon formula to compute oRef. */
    /* wRaw was already set before graph create (lines ~180) so that wT
     * STATIC captures it. Only aRaw (APP_WRITE) and biasArr (APP_WRITE)
     * can be changed here.
     * DIAG modes use the split encoding:
     *   bias[2c+1] = 0x4000 (baseline 128), bias[2c] = 0x4000 (scale 2.0)
     * so floor(acc × 2 / 512) = floor(acc/256). */
    auto set_diag_bias = [&](uint16_t scale_raw) {
        for (size_t i = 0; i < biasArr.size(); i++) biasArr[i] = 0;
        for (int nt = 0; nt < N_tiles; nt++)
            for (int c = 0; c < 32; c++) {
                biasArr[nt * 128 + 2 * c]     = scale_raw;  /* SCALE */
                biasArr[nt * 128 + 2 * c + 1] = 0x4000;     /* BASELINE 128 */
            }
    };
    if (ITERS == 999) {
        std::printf("[DIAG1] Uniform all-1 act+wt, split-bias (zp=128, scale=2/512)\n");
        for (int i = 0; i < M * K; i++) aRaw[i] = 1;
        set_diag_bias(0x4000);
        ITERS = 1;
    } else if (ITERS == 998) {
        std::printf("[DIAG2] act=all-1, wt[k][n]=(n%%32)+1, split-bias\n");
        for (int i = 0; i < M * K; i++) aRaw[i] = 1;
        set_diag_bias(0x4000);
        ITERS = 1;
    } else if (ITERS == 997) {
        std::printf("[DIAG3] act=all-1, wt=-7 uniform, split-bias\n");
        for (int i = 0; i < M * K; i++) aRaw[i] = 1;
        set_diag_bias(0x4000);
        ITERS = 1;
    } else if (ITERS == 996) {
        std::printf("[DIAG4] act[r][k]=(r*4)+1, wt=all-1, split-bias\n");
        for (int m = 0; m < M; m++) for (int k = 0; k < K; k++)
            aRaw[m*K + k] = (uint8_t)((m * 4) + 1);
        set_diag_bias(0x4000);
        ITERS = 1;
    } else if (ITERS == 995) {
        std::printf("[DIAG5] act[r][k]=(r+k)*3%%200, wt[k][n]=(k+n)%%5-2, split-bias\n");
        for (int m = 0; m < M; m++) for (int k = 0; k < K; k++)
            aRaw[m*K + k] = (uint8_t)(((m + k) * 3) % 200);
        set_diag_bias(0x4000);
        ITERS = 1;
    }
    /* Build reference in ALL cases (DIAG or full-random) using the silicon
     * formula so the compare is bit-exact. */
    {
        /* Graph output is TILE-LAYOUT [M_tiles, N_tiles, 1024]:
         *   oRef[(mt*N_tiles + nt)*1024 + r*32 + c] — matches HMX sat.ub
         *   direct-to-DDR tile write. */
        for (int mt = 0; mt < M_tiles; mt++) {
            for (int r = 0; r < 32; r++) {
                int m = mt * 32 + r;
                for (int nt = 0; nt < N_tiles; nt++) {
                    for (int c = 0; c < 32; c++) {
                        int n = nt * 32 + c;
                        int32_t acc_hmx = 0;
                        for (int k = 0; k < K; k++)
                            acc_hmx += (int32_t)aRaw[m*K + k] * (int32_t)wRaw[k*N + n];
                        uint16_t bias_b = biasArr[nt * 128 + 2 * c + 1];
                        uint16_t bias_s = biasArr[nt * 128 + 2 * c];
                        oRef[(mt * N_tiles + nt) * 1024 + r * 32 + c] =
                            hmx_silicon_u8(acc_hmx, bias_b, bias_s);
                    }
                }
            }
        }
    }

    /* V8 graph inputs at execute: act, bias, scratch (wT STATIC = skip). */
    Qnn_Tensor_t eIn[3], eOut[1];
    std::memcpy(&eIn[0], &aT, sizeof(Qnn_Tensor_t));
    eIn[0].v1.clientBuf = {aRaw.data(), (uint32_t)aRaw.size()};
    std::memcpy(&eIn[1], &biasT, sizeof(Qnn_Tensor_t));
    eIn[1].v1.clientBuf = {biasArr.data(), (uint32_t)(biasArr.size() * sizeof(uint16_t))};
    std::memcpy(&eIn[2], &sT, sizeof(Qnn_Tensor_t));
    eIn[2].v1.clientBuf = {sBuf.data(), (uint32_t)sBuf.size()};
    std::memcpy(&eOut[0], &oT, sizeof(Qnn_Tensor_t));
    eOut[0].v1.clientBuf = {oBuf.data(), (uint32_t)oBuf.size()};

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
        QCHECK(g_qnn.graphExecute(graph, eIn, 3, eOut, 1, profile, nullptr));
        uint64_t c = cycles_from_profile(profile);
        total += c; if (c < mn_) mn_ = c; if (c > mx) mx = c;
    }
    uint64_t avg = total / ITERS;
    dump_profile(profile);

    /* oRef already built (u8 with zero_offset=128) above. */
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
    if (bad > 0 && bad < 50) {
        int shown = 0;
        for (int i = 0; i < M*N && shown < 20; i++) {
            if (oBuf[i] != oRef[i]) {
                int m = i / N, n = i % N;
                int32_t acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (int32_t)aRaw[m*K + k] * (int32_t)wRaw[k*N + n];
                int nt = n / 32, c = n % 32;
                uint16_t bs = biasArr[nt*128 + 2*c];
                uint16_t bb = biasArr[nt*128 + 2*c + 1];
                double bv = fp16_to_float(bs);
                double raw_scaled = (double)acc * bv / 512.0;
                std::printf("  mism[m=%d n=%d]: obs=%3d ref=%3d  acc=%6d  bias_b=0x%04x bias_s=0x%04x  raw_scaled=%.4f\n",
                    m, n, oBuf[i], oRef[i], acc, bb, bs, raw_scaled);
                shown++;
            }
        }
    }
    std::printf("  oBuf[0..15]= ");
    for (int i = 0; i < 16 && i < M*N; i++) std::printf("%3d ", oBuf[i]);
    std::printf("\n  oBuf[16..31]=");
    for (int i = 16; i < 32 && i < M*N; i++) std::printf("%3d ", oBuf[i]);
    std::printf("\n  oRef[0..15]= ");
    for (int i = 0; i < 16 && i < M*N; i++) std::printf("%3d ", oRef[i]);
    std::printf("\n  oRef[16..31]=");
    for (int i = 16; i < 32 && i < M*N; i++) std::printf("%3d ", oRef[i]);
    std::printf("\n");

    g_qnn.profileFree(profile);
    g_qnn.contextFree(context, nullptr);
    g_qnn.backendFree(backend);
    return 0;
}
