# Agent Knowledge

This is the single active Agent document for `qcom_htp`.  Older topic/current/archive Markdown notes were folded here or deleted; keep only necessary non-Markdown RE evidence under `Agent/qnn_re/`.

## Current Target

The active custom MatMul line is `example/qnn_hmx_matmul_u8i8`.

Public names:

| Surface | Name |
|---|---|
| Package | `QnnHmxMatMulU8I8Package` |
| Provider | `QnnHmxMatMulU8I8InterfaceProvider` |
| Op | `HmxU8I8ToU8MatMul` |
| XML | `standard_flow/custom_u8i8/QnnHmxMatMulU8I8Package.xml` |
| HTP lib | `libQnnHmxMatMulU8I8_htp.so` |
| CPU lib | `libQnnHmxMatMulU8I8_cpu.so` |
| x86 lib | `libQnnHmxMatMulU8I8.so` |
| Perf tool | `scripts/perf_hmx_u8i8_matmul.py` |

The custom op signature is intentionally native-aligned:

```text
in[0] bias     int32 Flat4 Direct   TCM_Only  [1, N/32, 1, 64]
in[1] weight   u8    Flat4 Direct   TCM_Only  [1, 1, K, N], bytes are K-major HMX tiles
in[2] act      u8    Crouton_8 Indirect TCM_Only [1, M/32, 32, K]
in[3] scratch  u8    Flat4 Direct   TCM_Only  [1, 1, 1, 2048]
out[0] output  u8    Crouton_8 Indirect TCM_Only [1, M/32, 32, N]
```

## Design Principle

The current QNN MatMul mental model is simple:

```text
HMX only computes.
Everything else is prepared before the hot HMX body.
```

Native QNN does not ask the final HMX kernel to understand high-level MatMul tensors.  It first converts the problem into HMX-native Conv1x1 state:

```text
ordinary MatMul tensors
  -> QNN graph/context/runtime preparation
       - pack weight into K-major HMX tiles
       - fold bias into native bias blocks
       - format activation/output as Crouton_8 TCM blocks
       - move static data through weights_to_vtcm / bias_to_vtcm sidecars
       - prepare pointer tables, mask state, and tile counts
  -> hot compute event
       - stitch tiny native descriptors
       - enter V73DEEP HMX body
```

So the custom rule is: do not put input/layout preparation in the profiled hot callback.  The hot callback should not query tensor metadata, recover shapes, repack weights, copy QNN block tables, patch mask state, or handle broad generic cases.  Those belong in the generator, converter, QNN sidecar ops, or QHPI precompute.

Kernel design summary: this MatMul is implemented as a native HMX Conv1x1 compute body.  The C++ hot callback only translates prepared QHPI state into native descriptors; scale, zero point, and bias are folded into the prepared native bias record before `cvt.ub = acc(...)`.  Keep the detailed explanation in [`docs/qnn_custom_op_matmul_e2e.md`](../docs/qnn_custom_op_matmul_e2e.md).

## Current Conclusion

The replicated kernel body is not the main gap anymore.  The owned V73DEEP inline-asm body is byte-equivalent to the native `hmx_v73_convbbb1x1deep_stride1` body for the 1132-byte kernel slice, and the 256^3 chain result is bit-exact.

Latest verified hot-op numbers at 256^3 chain8 on device, 2026-05-05:

| Path | cycles | packets | cpp | correctness |
|---|---:|---:|---:|---|
| native QNN kernel-only `ConvLayer_s1.opt` | 1140 | 346 | 3.296 | under-counts QNN MatMul |
| native QNN-op aggregate `MatMul_*` | 1452 | 451 | 3.220 | includes sidecar HTP ops |
| custom `HmxU8I8ToU8MatMul` precompute default | 1165 | 337 | 3.458 | bit-exact |

Current gap statement: after QHPI precompute, graph-load pointer-table copies, mask pre-initialization, and 64B stack-descriptor alignment, the latest clean hot-op run is `1165 - 1140 = +25 cycles` and `337 - 346 = -9 packets` versus native kernel-only `ConvLayer_s1.opt`.  Versus native QNN-op aggregate, custom is `287 cycles / 114 packets` lower because the native aggregate includes sidecar HTP setup events.

Probe split: an intrusive `HMX_U8I8_PROBE_CYCLES` run reports `kernel=1074` and `desc=29`.  The owned inline-asm body is therefore already faster than the native kernel-only chrometrace event (`1074 < 1140`); the remaining `+25 cycles` is QHPI callback/profiling envelope plus tiny descriptor glue and issue/locality effects, not extra matmul work.

Cycle-gap experiments: explicit dcfetch of QNN tables reduced some cycles but cost too many packets; graph-load table copies are the retained version. Writing descriptor/mask/extra into the existing Direct-TCM scratch is bit-exact but much slower (`~1700 cycles`, high cpp); pre-writing those records in QHPI precompute fails at device context creation. Raising the owned kernel entry alignment above 64B, static extra/mask variants, and precomputed descriptor records all worsened cycles.

Historical gap statement: the old custom wrapper path measured `1794 cycles / 471 pkts` because it still did input/layout preparation inside the profiled callback.  Comparing that to native kernel-only produced an apparent `+125 pkts`, but native QNN accounted for much of that preparation in sidecar HTP ops (`bias_to_vtcm`, `weights_to_vtcm`, `DmaCheckpointSet`) and graph-load work.  The real lesson is architectural: HMX should only see prepared TCM/VTCM state and perform compute.

## How QNN Appears To Call This

QNN does not call a custom op like an internal `q::*` primitive.  The observed design is:

```text
ONNX custom node
  -> converter XML + converter op package shape/type hooks
  -> ctxgen resolves OpPackage provider and QHPI precompute registration
  -> runtime inserts surrounding built-ins for tensor movement/layout
  -> QHPI precompute records bias/weight pointers, copies small Crouton block tables, pre-initializes mask state
  -> hot callback stitches small native descriptors on stack
  -> owned V73DEEP inline-asm kernel runs
```

The important architectural boundary is the QHPI callback and accounting model.  Native QNN splits MatMul setup across sidecar HTP ops and graph-load preparation; the custom path must mirror that by using QHPI precompute so `HmxU8I8ToU8MatMul` is effectively a compute event.

## Perf Reading

Use wall-time/optrace first, then packets/cpp:

```bash
python scripts/perf_hmx_u8i8_matmul.py \
  example/qnn_hmx_matmul_u8i8/standard_flow/custom_u8i8/out/u8i8_chain_256
```

Meaning:

| Metric | Use |
|---|---|
| `dur` / cycles | user-visible hot-op time on HTP timeline |
| `pkts` | best proxy for extra executed Hexagon packet work |
| `cpp` | sanity check for stalls/issue quality; important after packet accounting is fixed |

Use `dur` as the final latency answer.  Use `pkts` to decide whether custom is
doing extra work.  Current precompute default is packet-better than native
kernel-only (`-9 pkts`), so the remaining `+25 cycles` is not caused by extra
matmul instructions.

Use QNN-op aggregate view for native comparisons.  Kernel-only `ConvLayer_s1.opt`
is useful for studying the HMX body, but it excludes native setup events that
the custom op pays inside its own callback.

Optional diagnostic build:

```bash
EXTRA_DEFS=-DHMX_U8I8_PROBE_CYCLES bash example/qnn_hmx_matmul_u8i8/build.sh
python scripts/perf_hmx_u8i8_matmul.py <out_dir> --probe-cycles
```

## Do Not Retry

Do not spend more time on these unless new evidence appears:

- Reintroducing old V2-V8, row-major, spike, pack/copy/untile kernels.
- Swapping back to dlsym/native-kernel call paths.
- VTCM pointer-table scratch as a gap closer; it did not remove the packet gap.
- PMU-heavy probes as a normal measurement path; they perturb packet counts.
- Parameter sweeps around old descriptor flags as production candidates.
- Compatibility aliases for old package/op names; regenerate DLC/context binaries instead.

## Evidence Kept

Non-Markdown RE evidence remains in `Agent/qnn_re/`:

```text
skel_text_full.S
hmx_v73_convbbb1x1deep_stride1_2ebe40.S
hmx_v73_convbbb1x1_stride1_2eadc0.S
set_hmx_params_conv1x1.S
wrapper_3dc2a8.S
descriptor_builder_3d7920.S
descriptor_builder_3d7920_full.S
descriptor_builder_full.S
descriptor_builder_pt2.S
hmx_convbbb1x1_stride1_2ea740.S
hmx_convbbb1x1_stride1_full.S
hmx_convbbb1x1_stride1_FULL_decoded.S
```

Use those only when re-checking ABI, descriptor fields, or wrapper control flow.

## Common Commands

```bash
bash example/qnn_hmx_matmul_u8i8/build.sh
bash example/qnn_hmx_matmul_u8i8/build_x86.sh

cd example/qnn_hmx_matmul_u8i8/standard_flow/custom_u8i8
SKIP_DEVICE=1 bash run_u8i8_chain.sh

python scripts/perf_hmx_u8i8_matmul.py \
  example/qnn_hmx_matmul_u8i8/standard_flow/custom_u8i8/out/u8i8_chain_256
```

## Documentation Rule

This file is the active Agent knowledge base.  Add future agent-facing status, rules, handoffs, and conclusions here unless they are raw non-Markdown evidence under `Agent/qnn_re/`.
