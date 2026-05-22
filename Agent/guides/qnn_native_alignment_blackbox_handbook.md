# QNN Native Alignment Blackbox Handbook

Status note: this handbook is historical/supporting methodology for future
QNN-native alignment tasks.  It is not the active W4A16 handwritten MatMul
route; that work now follows
`Agent/guides/handwritten_hmx_matmul_roadmap.md` and the direct-body
custom-baseline gate under `example/handwritten_hmx_matmul/`.

Use this handbook when aligning a custom HMX OP with QNN native behavior. Treat
alignment as blackbox reverse engineering: the native lowered graph is the
source of truth, and every formula, decoded comment, Python reference, or custom
probe is only a diagnostic until it agrees with native raw output and native
performance evidence.

The goal is not just bit-exact output. A kernel is aligned only when the custom
default path reproduces the native runtime contract: raw storage, lowered tensor
surface, descriptor/table contract, comparable kernel event class, packet count
class, timeline class, and documented boundary differences.

## Acceptance Oracle

The oracle is a clean QNN native artifact for the same family, shape, chain,
input, weights, bias/control contract, quantization contract, layout flags, and
runtime raw I/O.

A custom OP is aligned only when a fresh default run:

- matches native runtime raw output byte-for-byte;
- passes the standard artifact gate;
- uses the intended embedded native kernel body or a byte-verified rewrite;
- enters HTP with the same activation/output tensor surface as native, unless a
  boundary mismatch is explicitly accepted;
- compares custom main-op cycles to the comparable native kernel event;
- compares custom timeline to native timeline;
- reports packet counts or equivalent per-event evidence for the kernel body;
- records harmless boundary differences and open scope.

Analytic references are useful for diagnosis, but native raw wins when analytic
and native disagree. A green checker proves artifact hygiene, not alignment by
itself.

## Standard Artifact Gate

Before using a run as correctness or performance evidence, require:

- converted DLC generated with layout-preservation flags on every public graph
  input and output;
- context binary generated before device execution;
- device execution from `qnn-net-run --retrieve_context`, not live DLC;
- native runtime I/O recorded in `native_io.json`;
- `--use_native_input_files --use_native_output_files` when comparing runtime
  raw storage;
- decoded `optrace/` beside the artifact directory;
- bottom mapping saved from the same context binary;
- checker pass:

```bash
scripts/check_qnn_artifact_standard.py <out_dir> \
  --require-native-io --require-layout-flags --reject-float-io
```

Reject float runtime I/O, missing `native_io.json`, missing NONTRIVIAL layout
flags, missing context-binary execution, and ad-hoc `/tmp/_optrace*` evidence as
current oracles. Keep them only as historical diagnostics.

Also audit any derived analysis files inside the artifact, such as
`analysis/*.txt` or `analysis/*.json`. If they still report old shapes, cycles,
or compare results, refresh or mark them stale before handoff.

## Alignment State Machine

### 1. Define the native oracle

Generate the native reference first. Record the artifact path, generation
command, raw input/output paths, context binary, bottom mapping, optrace summary,
native comparable kernel event, native QNN-op aggregate event, timeline span,
and packet counts when available.

Do not start by fitting formulas to custom output. First answer: what did QNN
actually lower, and what raw bytes did it actually consume and emit?

### 2. Split public graph surface from runtime contract

Public ONNX tensors may be float, rank-friendly, or reshaped for export. Runtime
HTP tensors may be native raw, tiled, indirect, or transformed by input/output
ops. Compare `native_io.json`, pulled raw files, and bottom mapping rather than
public ONNX dtype or rank.

For every candidate, record the HTP entry activation, weight, bias/control, and
output tensors with dtype and dims. Bit-exact output on a different kernel-entry
surface is a diagnostic success, not final alignment.

### 3. Read lowered execution before changing code

Inspect bottom mapping and optrace before editing descriptors. Identify:

- input/output slices;
- `ForceFormat_Crouton` or other format transforms;
- `weights_to_vtcm`, `bias_to_vtcm`, and other sidecars;
- `DmaCheckpointSet`, preload, sync, or graph-service events;
- final HMX/custom event or native `ConvLayer_s1.opt` event;
- native QNN-op aggregate cycles when one logical op lowers into sidecars.

Keep kernel-only, QNN-op aggregate, and full timeline scopes separate.

### 4. Confirm the active native entry

Prove the native call path before copying fields into custom code. Use the
smallest native/skel patch or probe that can prove entry address, branch target,
return path, register arguments, and liveness. A symbol name or static
disassembly match is not enough.

If the custom package embeds native code, verify the body byte-for-byte against
the native `.so` slice. Once the body is verified, prioritize descriptor, table,
payload, and surface evidence over swapping wrappers blindly.

### 5. Recover output and probe mappings

Native kernels often write into an internal tiled output before QNN output ops
transform it into public `Y.raw`. Before parsing descriptor words or table dumps
from a public raw file, run a pattern probe and invert the output mapping for
that family, dtype, and surface.

If a probe output is only partially exact, inspect row/tile exactness, sorted
equality, value histograms, row rolls, tile rolls, and block-local rotations.
Half-row, every-other-block, or exact-prefix patterns usually point to table
coverage or loop state. Changed histograms or saturation usually point to
payload, arithmetic, scale, bias, or control interpretation.

### 6. Dump descriptor, record, and table windows

Use entry, base-record, descriptor, table, and record-window probes to observe:

- descriptor scalars and masks;
- pointer-table addresses;
- table stride and length;
- adjacent metadata around the native record;
- payload pointers and sidecar buffers;
- first and later kernel-node differences in a chain.

Prefer narrow assembly-instruction probes that perturb the native path as little
as possible. Record the exact artifact name for each hypothesis.

Do not infer table length from tensor shape alone. Native compact or physical
tables may be shorter than a custom public-QHPI expansion, and an expanded table
can be correct but non-native.

### 7. Extract prepared payloads

When weight, bias, control, or sidecar bytes are opaque, compare native prepared
bytes with custom-generated bytes. Prefer byte-for-byte payload checks over
pack-order guesses. Keep full-range and low-saturation probes separate; a
special low-pattern native lowering may not be a clean comparator for ordinary
full-range W8/W4 behavior.

### 8. Run one-hypothesis probes

Change one variable at a time:

- graph surface rank or layout;
- descriptor scalar;
- table pointer contract;
- table length or stride;
- payload pack order;
- mask/control word;
- bias layout;
- runtime tensor layout;
- output export path.

Name each artifact after the hypothesis and keep failed artifacts because they
rule out false paths. Do not combine descriptor, table, payload, and surface
changes in one broad probe unless all narrower probes are already exhausted.

### 9. Treat scalars as clues until table and packet evidence agree

A descriptor scalar can make cycles native-class while output remains partly
wrong. Another scalar can make output exact while packet counts explode. In both
cases, record the scalar as a clue, not the solution.

Promote a scalar only when it agrees with:

- native raw output;
- native entry tensor surface;
- native table or record-window evidence;
- native packet count class;
- native kernel-cycle and timeline class.

The W8A16 lesson generalizes: a row-expanded table with a larger loop count can
cover all rows and produce exact output, yet still be the wrong ABI because the
kernel pays many more packets than native. Packet count is a first-class ABI
signal.

### 10. Promote only the native contract

Once a probe closes correctness and performance, integrate the native contract
into the default path for the accepted surface. Keep older surfaces,
row-expanded tables, marker paths, skip-kernel paths, and scalar workarounds as
explicit diagnostics only.

Default build and default runner should exercise the accepted real-kernel path.
Guarded diagnostic macros may remain, but they must not silently compete with
the native-aligned default.

### 11. Close with a fresh standard rerun

Rerun current HEAD through the normal runner into the canonical artifact
directory. Require checker pass, raw compare against native, optrace summary,
bottom mapping, packet evidence, refreshed derived analysis files, and updated
documentation before marking the family aligned.

## Performance Reading

Always report these scopes when available:

- custom main-op event cycles;
- custom packet count per kernel event;
- native comparable kernel event cycles;
- native packet count per kernel event;
- native QNN-op aggregate cycles when one logical op lowers to sidecars;
- input/output slice and format-conversion cycles;
- native and custom full graph timeline spans.

Do not compare custom main-op cycles to native full timeline as if they were the
same scope. Native QNN may include public-surface transforms, format conversion,
slice/concat/output ops, preload, and sidecar setup that the custom prepared
path handles differently.

Use packet counts to distinguish "semantically correct but wrong contract" from
"native-class contract with normal noise." First kernel events may include setup
or cache effects, so compare both aggregate and per-node packet/cycle patterns
across the chain.

## Evidence Tiers

Strong evidence:

- native and custom runtime raw files are byte-identical, with SHA256 recorded;
- standard checker passes for both native and custom artifacts;
- bottom mapping shows the accepted kernel-entry dtype/dims;
- optrace compares the same scope and reports native-class packet counts;
- native record/window probes explain the descriptor and table contract;
- embedded kernel body or readable rewrite is byte-identical to the native
  slice.

Diagnostic evidence:

- analytic reference exactness;
- ctxgen success;
- marker-output exactness;
- custom-only probe output;
- static disassembly without live-entry proof;
- old artifacts that predate the current runner or layout flags;
- derived analysis summaries unless refreshed from the current artifact.

Failure evidence worth keeping:

- sorted equality with tile or row rolls;
- exact prefixes, half-row exactness, or every-other-block exactness;
- native-class cycles with partial output;
- exact output with non-native packet count;
- graph execution failure after only one graph-contract change.

## Anti-Patterns

- Treating formulas, Python references, or decoded assembly comments as the
  final oracle.
- Trusting old artifacts without rechecking native I/O, context-binary use,
  layout flags, optrace, and derived analysis summaries.
- Comparing float public ONNX tensors instead of native runtime raw.
- Reading only top-level op names and ignoring lowered sidecar events.
- Parsing probe dumps before recovering the output transform.
- Assuming table length or loop count from tensor shape instead of native
  record/window evidence.
- Promoting a correctness-only expanded table when packet counts are far from
  native.
- Running broad probes that change descriptor, table, payload, and surface in
  one step.
- Deleting failed artifacts before recording what hypothesis they disproved.
- Reporting only end-to-end time when the gap is inside the kernel event,
  sidecar setup, or graph transforms.
- Marking a family complete from ctxgen success, a green checker,
  probe-marker exactness, analytic bit-exactness, or custom/native raw equality
  on the wrong kernel-entry surface.

## Completion Checklist

Before handing off or marking an OP aligned, record:

- native oracle artifact path;
- custom aligned artifact path;
- exact generation and run commands;
- checker output for both native and custom artifacts;
- raw output compare against native, preferably SHA256 plus `cmp`;
- embedded-body byte verification, if the custom path embeds native code;
- optrace summary for custom and native with scope labels;
- packet counts for custom and native kernel events;
- bottom-mapping evidence for the final kernel path;
- native entry, output-map, descriptor, table, and record/window evidence used
  to choose the contract;
- known harmless boundary mismatches such as carrier dtype reporting, control
  tensor shape, or custom-only sidecar representation;
- refreshed artifact-local analysis files, or an explicit note that they are
  stale diagnostics;
- open scope outside the accepted contract, such as broader shapes, LPBQ,
  dynamic chain sizes, split-output public QHPI contracts, or per-group
  extensions.

## Handoff Template

Use this compact format in `Agent/handoffs/` or `Agent/current/`:

```text
Status:
- accepted shape/chain/contract:
- native oracle:
- custom artifact:
- raw compare:
- checker:
- kernel body:
- bottom mapping:
- optrace:
- packet evidence:
- promoted native contract:
- diagnostic paths kept:
- harmless differences:
- open scope:
- exact rerun command:
```
