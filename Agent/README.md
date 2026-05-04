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

## Current Conclusion

The replicated kernel body is not the main gap anymore.  The owned V73DEEP inline-asm body is byte-equivalent to the native `hmx_v73_convbbb1x1deep_stride1` body for the 1132-byte kernel slice, and the 256^3 chain result is bit-exact.

Latest verified hot-op numbers at 256^3 chain8 on device, 2026-05-05:

| Path | cycles | packets | cpp | correctness |
|---|---:|---:|---:|---|
| native QNN `ConvLayer_s1.opt` | 1147 | 346 | 3.315 | baseline |
| custom `HmxU8I8ToU8MatMul` | 1679 | 471 | 3.565 | bit-exact |

Simple gap statement: custom runs about 125 more committed packets per hot op in the current verified build.  The old post-cleanup regression was 550 packets; that came from scalarizing the activation/output pointer-table copies.  Restoring the old Hexagon-only 128B HVX copy path brings the custom op back to 471 packets, essentially aligned with the earlier 468-packet state.  The remaining extra work is outside the HMX body itself: QNN calls the custom op through an opaque QHPI custom-op wrapper, then our wrapper rebuilds descriptor/table state before jumping into the same V73DEEP kernel shape.  Native QNN reaches the skel wrapper with more internal state already materialized and scheduled as a built-in op.  Cycles vary more than packets run-to-run, so use packet gap as the stable comparison.

## How QNN Appears To Call This

QNN does not call a custom op like an internal `q::*` primitive.  The observed design is:

```text
ONNX custom node
  -> converter XML + converter op package shape/type hooks
  -> ctxgen resolves OpPackage provider and QHPI registration
  -> runtime inserts surrounding built-ins for tensor movement/layout
  -> QHPI custom op callback receives tensor handles/block tables
  -> custom wrapper builds native HMX descriptors
  -> owned V73DEEP inline-asm kernel runs
```

The important architectural boundary is the QHPI callback.  It gives us enough access to run the same low-level kernel, but not the same built-in scheduler/wrapper integration as native `q::ConvLayer_s1.opt`.

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
| `cpp` | sanity check for stalls/issue quality, not the primary gap size |

Most accurate for the current question is `pkts`: the gap is a packet-count gap, not primarily a cpp/stall gap.

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
