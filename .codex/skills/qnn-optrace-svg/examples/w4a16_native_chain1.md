# W4A16 Chain1 SVG Examples

Use this as a layout example, not as reusable renderer code.

## Native Source Artifacts

- Graph source:
  `example/qnn_matmul_profile/output_w4a16_native_chain1_default_ci_e2e_256/optrace/chrometrace_htp.json`
- Timing source:
  `example/qnn_matmul_profile/output_w4a16_native_chain1_default_ci_e2e_256/optrace/chrometrace.json`
- Example output:
  `example/qnn_hmx_matmul_w4a16/w4a16_chain1_native_full_layout.svg`

## Native Graph Facts

- HTP graph nodes: 82
- HTP dependency edges: 95
- Timing label: `Dominant Path Cycles`
- Rendered labels keep full op type after removing only the leading `q::`.

## Native Layout Pattern

- Stage 1: 32 `InputSlice` nodes in a grid, routed into the input `Concat`
  through left/right side buses.
- Stage 2: 8 repeated A-layout lanes,
  `SlicePad_shape_inplace -> Transpose_impl`, then `Concat -> Reshape ->
  ForceFormat_Crouton`.
- Stage 3: `weights_to_vtcm`, `bias_to_vtcm`, activation `ForceFormat_Crouton`,
  and `ConvLayer_s1.opt`. The activation/weight/bias edges must be prominent.
- Stage 4: `ConvLayer.opt.activations_to_vtcm`, then 8 repeated Y-layout lanes,
  `SlicePad_shape_inplace -> Transpose.2D -> Transpose.2D`.

## Custom Graph Lesson

For the custom chain1 graph, do not summarize the kernel as merely
"activation plus constants". Inspect `HmxU16I4ToU16MatMul.input_names` and draw
four distinct direct input ports in source order:

- `in0`: `$Const_11` / `ConvLayer.opt.weights_to_vtcm`, `[1,8,1,128]`
- `in1`: `$Const_12` / `ConvLayer.opt.weights_to_vtcm`, `[1,1,128,256]`
- `in2`: formatted activation from `ForceFormat_Crouton`, `[1,8,32,256]`
- `in3`: `$Const_13` / `ConvLayer.opt.weights_to_vtcm`, `[1,1,1,1]`

The activation edge must route to its own port and should not pass through the
VTCM constant nodes. If two edges enter the same y-coordinate or the same side
of the compute box, the drawing can falsely imply fewer kernel inputs.

## Visual Checks

- The graph should show all 82 nodes, not a collapsed summary.
- Important compute inputs should be obvious:
  activation from `ForceFormat_Crouton`, plus weight and bias VTCM nodes.
- For custom kernels, count visible compute input ports against the compute
  node's `input_names`; do not infer the input count from grouped edge bundles.
- Non-HTP tensor inputs to the compute node should be shown separately, for
  example as dashed external-input badges.
- No edge should cross through the text inside `InputSlice`, `SlicePad`, or
  compute-stage nodes.
- Do not abbreviate `ForceFormat_Crouton`, `SlicePad_shape_inplace`, or
  `ConvLayer.opt.*` labels.

## Preview Flow

After drawing or regenerating the SVG:

1. Convert it to a PNG using an available SVG renderer.
2. Open the PNG and inspect the whole image for aspect ratio and routing.
3. Zoom into dense regions: input fan-in, A concat, compute fan-in, and Y fan-out.
4. Run text checks for node count, edge count, cycle labels, and forbidden
   abbreviated labels.
