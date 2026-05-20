---
name: qnn-optrace-svg
description: Generate fixed-layout SVG diagrams from QNN HTP optrace artifacts in qcom_htp. Use when Codex needs to visualize chrometrace_htp.json node-flow graphs together with chrometrace.json timing, especially W4A16/QNN Native/custom-op HTP op graphs, without relying on Mermaid auto-layout.
---

# QNN Optrace SVG

Use this skill to design readable fixed-layout SVG diagrams from QNN HTP
optrace artifacts. It should provide principles, workflow, and examples only.
Do not turn the skill itself into an end-to-end renderer; implement any
task-specific drawing code in the target workspace only when the current task
needs it.

## Source Rules

1. Use `chrometrace_htp.json` as the HTP node-flow source of truth.
2. Use `chrometrace.json` only for timing annotations.
3. Prefer `args["Dominant Path Cycles"]` from same-ID `ph=X`, `pid=0` events
   for node cycle labels. Do not silently substitute `Duration (cycles)`.
4. Keep rendered HTP node and dependency-edge counts equal to the graph source
   unless the user explicitly asks for an abstracted summary.
5. For important compute nodes, inspect the node's `input_names` directly.
   Dependency groups are not enough: the diagram must show the compute node's
   direct input tensor count and order, even when several inputs come from
   nearby staging nodes.

## Layout Principles

- Use fixed coordinates for high-fan-in or repeated-lane graphs.
- Divide the graph into broad stage bands, but keep all actual nodes visible.
- Place repeated lanes on aligned grids so repeated structure is recognizable.
- Route dense fan-in/fan-out through side buses or explicit orthogonal paths.
- Make important dependency edges visually stronger with enough contrast,
  stroke width, and arrowheads.
- For compute nodes, draw explicit input ports when the node has multiple
  direct inputs. Label them in `input_names` order, such as `in0`, `in1`,
  `in2`, and route each producer edge to a distinct port.
- Do not allow dependency lines to pass through node text or hide node inputs.
- Do not let two logical inputs share the same visual entry point unless the
  source graph really has a merged input. Avoid routes that pass through
  unrelated staging nodes, because they make the input count ambiguous.
- Keep external tensor inputs visible as separate dashed/input badges when
  they are not produced by HTP graph nodes.
- Preserve op type labels except for dropping the leading `q::` namespace.
  For example, draw `ForceFormat_Crouton` and `SlicePad_shape_inplace` exactly,
  not `ForceFormat` or `SlicePad`.

## Workflow

1. Read the graph and collect:
   - graph node order and IDs;
   - producer-to-consumer edges from tensor names;
   - direct `input_names` for important compute nodes, preserving order;
   - output tensor shapes for node labels;
   - dominant-path cycle labels from `chrometrace.json`;
   - external graph inputs that feed important compute nodes.
2. Sketch the structural stages before drawing:
   - input slicing / concat;
   - layout conversion lanes;
   - weight/bias/activation staging;
   - compute op;
   - output layout conversion lanes.
3. Draw the SVG with fixed dimensions and explicit paths. Favor a little extra
   whitespace over ambiguous line crossings.
4. Add a legend that states the exact timing semantic, such as
   `dom cyc = chrometrace.json args["Dominant Path Cycles"]`.
5. Embed large outputs as external `.svg` files from Markdown instead of
   inlining the SVG body into docs.
6. Preview the rendered image before finalizing:
   - convert the SVG to PNG with an available renderer such as ImageMagick
     `convert` or `rsvg-convert`;
   - inspect the PNG visually at normal and full resolution;
   - check that arrows are visible, labels are not crossed, and critical
     inputs such as activation/weight/bias into compute nodes are unambiguous.
   - for each important compute node, count visible input ports against
     `input_names`.
7. Run text-level checks after visual preview:
   - count rendered node labels and edge paths;
   - grep for forbidden abbreviations such as `>ForceFormat<` or `>SlicePad<`;
   - verify required labels such as `ForceFormat_Crouton` are present.

## Examples

- For the W4A16 native chain1 layout pattern and verification checklist, see
  [examples/w4a16_native_chain1.md](examples/w4a16_native_chain1.md).

## Guardrails

- Do not use `chrometrace_htp_graph_before.json` as the HTP op-flow graph unless
  the user explicitly asks for that intermediate graph.
- Do not silently switch timing semantics from dominant-path cycles to duration
  cycles. If DPC is absent, mark the node `n/a dom cyc`.
- Do not accept an SVG where dependency lines pass through node text or where
  important inputs such as activation/weight/bias into `ConvLayer_s1.opt` are
  visually ambiguous.
- Do not leave stale `.mmd` or Mermaid-derived SVGs beside a fixed-layout SVG
  unless the document intentionally compares rendering approaches.
- Do not claim the graph is complete unless the rendered node and edge counts
  match the HTP graph source.
- Do not claim a compute node is drawn correctly until its visible input ports
  match the node's direct `input_names` count and order.
