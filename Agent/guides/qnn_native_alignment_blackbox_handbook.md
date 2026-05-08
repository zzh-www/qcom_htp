# QNN Native Alignment Blackbox Handbook

Use this handbook when aligning a custom HMX OP with QNN native behavior.  The
task is a blackbox reverse-engineering loop: do not start from an analytic
formula as the oracle.  Start from a trustworthy QNN native artifact, then close
the gap with lowered execution evidence, prepared payload bytes, and controlled
negative probes.

## Operating Rule

The acceptance oracle is QNN native output and QNN native performance artifacts.
Analytic math, Python reference outputs, and inferred formulas are only
diagnostics.  A result is aligned only when the custom OP matches the native raw
output for the same graph/input and the performance comparison comes from the
standard `optrace/` artifact set.

## Standard Artifact Gate

Before using any native or custom run as evidence, require:

- converted DLC generated with layout-preservation flags on every public graph
  input and output;
- context binary generated before device execution;
- device execution from `qnn-net-run --retrieve_context`, not a live DLC run;
- native runtime I/O recorded in `native_io.json`;
- `--use_native_input_files --use_native_output_files` for runtime comparison;
- decoded `optrace/` directory beside the run artifact;
- checker pass:

```bash
scripts/check_qnn_artifact_standard.py <out_dir> \
  --require-native-io --require-layout-flags --reject-float-io
```

Reject float runtime I/O, missing `native_io.json`, missing NONTRIVIAL layout
flags, and ad-hoc `/tmp/_optrace*` evidence as current correctness or
performance oracles.  Keep those artifacts only as historical diagnostics.

## Alignment Loop

1. **Define the native oracle.**
   Generate a clean QNN native reference for the exact shape, graph, input, and
   quantization contract.  Record raw input/output storage, context binary,
   bottom mapping, optrace summary, and the native kernel event to compare
   against.

2. **Separate graph surface from runtime contract.**
   QNN native graphs may expose float ONNX inputs/outputs while the runtime
   contract is quantized native raw.  Compare against `native_io.json` and
   pulled native output raw, not the public ONNX tensor dtype.

3. **Read lowered execution first.**
   Inspect bottom mapping and optrace before changing code.  Identify sidecar
   nodes such as `weights_to_vtcm`, `bias_to_vtcm`, `ForceFormat_Crouton`,
   `DmaCheckpointSet`, and the final HMX or `ConvLayer_s1.opt` event.  Compare
   full graph timeline and kernel-only event separately.

4. **Extract prepared payloads.**
   When weights, bias records, or control blocks are opaque, compare native
   prepared sidecar bytes with generated custom bytes.  Prefer byte-for-byte
   payload checks over pack-order guesses.

5. **Run one-hypothesis probes.**
   Change one variable at a time: pack order, source-table shape, descriptor
   scalar, mask word, bias layout, or runtime tensor layout.  Name each artifact
   after the hypothesis, and keep failures because they eliminate false paths.

6. **Use permutation diagnostics on failed outputs.**
   If exact compare fails, check value distributions, sorted equality, row/tile
   rolls, and block-local rotations.  A value-preserving permutation points to
   layout, table, or descriptor state; saturation or changed histograms point
   toward arithmetic, scale, bias, control, or weight interpretation.

7. **Probe native minimally.**
   Use skel/native probes only to confirm call path, register arguments, entry
   addresses, pointer tables, and descriptor words.  Do not rewrite input/output
   semantics or insert C/C++ logic into the probe path; prefer the supported
   assembly-instruction style so the probe perturbs the blackbox as little as
   possible.

8. **Exclude scalar-field theories explicitly.**
   Copy observed native descriptor fields into custom one at a time.  If the
   output or perf does not move, stop iterating on that scalar and move back to
   full wrapper state, table shape, payload order, or sidecar contract.

9. **Close with a fresh standard rerun.**
   After the implementation appears aligned, rerun the current HEAD through the
   normal runner into a new artifact directory.  Require standard checker pass,
   native-output exactness, and optrace performance evidence from that fresh
   artifact before marking the work complete.

## Performance Reading

Always report at least three numbers when available:

- custom main-op event cycles;
- native comparable kernel event cycles;
- native and custom full graph timeline spans.

Do not compare custom main-op cycles to native full graph timeline as if they
were the same scope.  Native QNN often includes public-surface transforms,
format conversions, transpose/slice/concat/output nodes, and sidecar setup that
the custom prepared path intentionally avoids.

## Useful Evidence Patterns

- Byte-identical native output means correctness is closed for that exact
  shape/input/contract, even when analytic reference differs.
- `sorted_equal=True` plus a best tile roll identifies layout/table ordering
  bugs rather than arithmetic bugs.
- A native sidecar import that fixes values but leaves a rotation points to
  activation/output table or descriptor state.
- A generated payload that matches an extracted native sidecar closes the pack
  order question.
- A native kernel event slower than custom can be legitimate if native carries a
  generic wrapper and custom uses a specialized prepared HMX path.

## Anti-Patterns

- Treating formulas as the final oracle.
- Trusting old artifacts without checking runtime I/O, context-binary use, and
  layout flags.
- Comparing float public ONNX tensors instead of native runtime raw.
- Reading only top-level OP names and ignoring lowered sidecar events.
- Running broad speculative probes that change multiple variables at once.
- Deleting failed artifacts before recording what hypothesis they disproved.
- Reporting only end-to-end time when the gap is actually in sidecar or graph
  transform work.
- Marking a family complete from a proxy signal such as ctxgen success, a green
  checker, or bit-exact analytic output without native raw comparison.

## Completion Checklist

Before handing off or marking an OP aligned, record:

- native oracle artifact path;
- custom aligned artifact path;
- exact generation/run commands;
- output compare against native raw;
- optrace summary for custom and native;
- bottom-mapping evidence for the final kernel path;
- known boundary mismatches that are harmless for the accepted contract;
- open scope outside the accepted contract, such as broader shapes, LPBQ, or
  per-group extensions.
