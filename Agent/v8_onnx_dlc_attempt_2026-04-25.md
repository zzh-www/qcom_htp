# V8 → ONNX → DLC → context binary attempt (2026-04-25) — **SUPERSEDED**

> ⚠️ **Obsolete** as of later the same day. This note captures the stuck state
> while we were chasing what looked like an NHWC-layout dead end. It was
> actually solvable with **`--source_model_input_layout NAME NONTRIVIAL`** +
> **`--desired_input_layout NAME NONTRIVIAL`** on every input and output
> (4-flag combo), plus **`--use_native_input_files`** at `qnn-net-run` time.
>
> ✅ **See `Agent/v8_vs_native_optrace_2026-04-25.md`** for the working
> pipeline + the V8-vs-QNN trace comparison. SOP: `docs/qnn_custom_op_sop.md`.
>
> Keeping this note for the NHWC-transform diagnosis that explains *why* the
> NONTRIVIAL flags are necessary.

---

> (original note, for historical context:)
> Goal: get V8 through the standard **ONNX → DLC / model.so → ctx-binary →
> qnn-net-run optrace** pipeline and diff its trace against QNN's native MatMul.
> Status: **toolchain validated**; **V8 custom ops reach graph-finalize** on
> device; **execution fails** because the QNN converter forcibly re-permutes
> rank-4 tensors to NHWC and inserts Transpose nodes before every custom op.

## What the conversion pipeline actually does to V8

- **Default** (no layout flags): converter picks NHWC and writes rank-4 intermediates
  as transposed dims (e.g. `[1,16,16,1024]` → `[1,16,1024,16]`) and inserts `qti.aisw::Transpose` nodes.
- **`--input_layout NAME NONTRIVIAL`**: preserves IO dims, but still inserts
  `_nhwc` transpose nodes before each custom op consumer.
- **`<Layout>NONTRIVIAL</Layout>` in XML OpDef**: stops inserting explicit
  Transpose nodes, but the converter then permutes dims at graph boundaries
  (all tensors become `[1,H,W,C]` regardless).
- **XML Layout=NONTRIVIAL + `--preserve_io layout`**: IO dims preserved, but
  input-side Transpose nodes are still inserted so the inner graph sees NHWC.

None of these produce a graph where every tensor keeps its ONNX dims end-to-end.
The converter's NHWC logic is triggered by rank==4; dropping to rank 3 or lower
isn't compatible with our existing op signatures.

## What did work

| Item | Verified |
|---|---|
| XML OpDef for 4 custom ops + Converter Op Package .so compile | ✓ |
| Custom-domain ONNX parses with `--op_package_config` | ✓ |
| `qnn-onnx-converter` / `qairt-converter` produce valid .cpp/.bin / .dlc | ✓ |
| `qnn-model-lib-generator` → `libv8model.so` | ✓ |
| Runtime pkg interface advertises all 4 op names (added TcmDramCopy + UntileToRowMajor) | ✓ |
| qnn-net-run Composes + Finalizes on SM8650 | ✓ |
| Graph **execute** | ✗ (NHWC-permuted dims → kernel reads wrong strides / panics HTP) |

Bias + scratch promoted to APP_WRITE graph inputs (only wt_raw STATIC) → matches
`run_matmul_v8_graph.cpp` tensor-type layout. Did not fix execution because the
permuted dims also affect APP_WRITE tensors.

## Why this won't be fixed by a kernel patch

Patching `HmxMatMulV8Op.cpp` to accept DDR-backed bias/scratch doesn't help
because:

- `qhpi_tensor_shape()` returns the **converter-permuted** dims, not the dims
  the kernel expects. E.g. `packed_act` dims arrive as `[1,16,1024,16]` where
  kernel treats `dims[2]=K_tiles` and `dims[3]=channel=1024`. After transpose
  it reads `dims[2]=1024` as K_tiles → out-of-bounds loop iteration.
- Pre-Transpose node inserted by converter runs on HTP and actually permutes
  the VTCM bytes. By the time pack_act sees input, it's no longer row-major
  activation, it's a transposed layout that `pack_act_rm_hvx` can't parse.

You'd have to rewrite every V8 kernel to operate on NHWC-permuted input —
that defeats the purpose of V8 being a literal replica of QNN ConvLayer_s1.opt.

## Two real choices

### (a) Post-process the generated `.cpp` to strip Transpose nodes

After `qnn-onnx-converter` emits `v8_model_cpp.cpp`, run a Python pass that:

1. Deletes every `addNode_*_nhwc()` function definition.
2. Deletes corresponding `dimensions_*_nhwc*`, `*_nhwc_perm[]`, and intermediate
   tensor definitions.
3. Rewrites each consumer node's `inputs_*[]` to reference the un-transposed
   tensor name (e.g. `"act_raw"` instead of `"act_raw_nhwc"`).
4. Rewrites each consumer node's output tensor dims to match our ONNX
   declarations (e.g. `[1,16,16,1024]` instead of `[1,16,1024,16]`).
5. Deletes calls to `addNode_*_nhwc()` in `QnnModel_composeGraphs`.

This is ~100 lines of Python regex surgery. Brittle against converter version
bumps but recoverable any time.

### (b) Handcraft a `QnnModel.cpp` mirror of `run_matmul_v8_graph.cpp`

Write a single `v8_graph_model.cpp` that defines `QnnModel_composeGraphs()`
using `QnnModel::addTensor`/`addNode` for the exact 4-node graph (pack_act →
pack_wt → mmv8 → tcm2ddr) with dims matching the kernel. No ONNX, no converter,
no XML — but still produces a `libv8model.so` consumable by qnn-net-run, which
still produces a standard `qnn_htp_optrace.log` + chrometrace.

This is ~150 lines of straightforward wrapper-API code. Robust to SDK version.
**Deviates from the ONNX→DLC preference but gets the optrace you want.**

## My recommendation

**(b)**. Diagnosing the QNN converter's NHWC logic further is a bigger
tangent than the goal justifies. The output artifact (optrace) and comparison
story are identical. Once we have a V8 optrace via qnn-net-run, we can diff it
cleanly against `sweep_data_2026-04-19/s512/w8a8/chrometrace.json`.

If you still want (a), I'll do the regex surgery — quote me ~30 min.
