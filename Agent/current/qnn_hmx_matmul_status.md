# QNN HMX MatMul Status

## Current Direction

Immediate QNN-kernel alignment work is now on the W4 LPBQ path.  The
per-channel custom/native gates are the accepted baseline and remain protected
by default correctness and performance CI; LPBQ is the active next promotion
target.

Current LPBQ gate:

- The LPBQ correctness gate has been redesigned to compare against real QNN
  Native LPBQ Conv, not the older per-channel Conv oracle.
- Generated LPBQ cases now use dedicated `w4a8_lpbq` and `w4a16_lpbq` families
  that first derive signed int4 per-group scales on K32 blocks, then represent
  those scales as LPBQ `per_channel_float_scale * per_block_int_scale`.  The
  native runner validates that the QNN graph lowers through
  `Conv2d_w_blk_exp_scale` and
  `q::ConvLayer.opt.expand_block_quant_to_pc_int8_weights`.
- Current device evidence promotes the real-native LPBQ correctness gates:
  - Two-op LPBQ implementation:
    `HmxW4LpbqExpandToI8` expands native K-pair W4 plus per-block integer
    scales into the K-major W8 HMX carrier, const-folds static weights during
    prepare, and then reuses `HmxU8I8ToU8MatMul`/`HmxU16I8ToU16MatMul`.
  - W4A8 real-native LPBQ full case suite:
    `/tmp/qcom_htp_lpbq_w4a8_full_pergroup`, all 7 generated cases,
    custom/native exact `65536/65536`, maxabs `0`, sidecar `2048/2048`.
  - W4A16 real-native LPBQ full case suite:
    `/tmp/qcom_htp_lpbq_w4a16_full_pergroup`, all 7 generated cases,
    custom/native exact `65536/65536`, maxabs `0`, sidecar `4096/4096`.
    W4A16 uses direct tiled A16 input for the custom same-hardware gate to avoid
    float/QDQ fallback while preserving the native-output comparison.
- The older `/tmp/qcom_htp_lpbq_full_ci/` evidence is now classified as a
  degenerate all-one LPBQ metadata/per-channel-oracle check.  It must not be
  used as proof that custom LPBQ matches QNN Native blockwise expansion.

The current per-channel baseline starts with `u8i8` because its custom-op flow
is already cleanly matched to QNN Native:

- custom graph: `HmxU8I8ToU8MatMul(bias, weight, activation, scratch)`;
- public input/output: `[1,1,M,K]` and `[1,1,M,N]`;
- custom activation/output inside QNN: `[1,M/32,32,K]` and `[1,M/32,32,N]`
  Crouton_8 after reshape;
- weight: logical `W[K,N]` is pre-packed by the generator into the V73DEEP
  K-major 32x32 tile stream;
- bias: activation zero point is folded offline into the native 256-byte bias
  records;
- matched native oracle: `example/qnn_matmul_profile/run_matched_native_a8_ref.sh`
  rebuilds QNN Native MatMul+Add from the custom artifact's exact runtime input,
  logical `W`, folded bias, and chain length.

Current retained evidence:

- `example/qnn_matmul_profile/output_u8i8_aligned_e2e_256`
- `example/qnn_matmul_profile/output_u8i8_native_ref_e2e_256`
- `analysis/matched_native_compare.json`: `65536/65536`, `maxdiff=0`

Fresh device smoke for the new correctness target:

- command:
  `BUILD_PACKAGES=0 KERNEL_E2E_OUT_ROOT=/tmp/qcom_htp_u8i8_native_match_ci256 U8I8_MATCH_CHAIN=1 scripts/run_qnn_kernel_e2e_ci.sh u8i8_native_match`
- custom self-reference: `65536/65536`
- matched native compare: `65536/65536`, `maxdiff=0`
- final custom/native validation: pass
- note: a temporary 128^3 probe did not self-match and is not promoted; keep
  the current gate on validated canonical 256^3 while aligning per-channel
  kernels.

Case-import u8i8 input comparison update:

- `zp_neutral` 256^3 generated case now runs through the custom op and matches
  QNN Native exactly: `65536/65536`, `maxdiff=0`.
- `normal_random` 256^3 generated case runs through the custom op.  Source
  inputs match: runtime activation, logical `W`, `bias_q`, `effective_i32`, and
  native ONNX `W/B` all compare equal to the generated case.
- The custom packed 64KB weight initializer appears byte-for-byte in the native
  context at offset `41728`, so prepared weight is not the current mismatch.
- The custom generated 2048B bias record does not appear in the native context.
  The native bias sidecar is visible at context offset `107264`.
- `scripts/analyze_u8i8_native_bias_record.py` now extracts and compares that
  record.  With `--native-dlc`, it also reads the quantized DLC static Conv
  `B` tensor through the QNN SDK `IrDlcReader`.  For the 256^3 `normal_random`
  case it reports:
  `scale match: 256/256`, `control match: 256/256`, generated/native bytes
  `2048/2048`, `DLC B vs generated bias_q: {'0': 256}`, and native-sidecar
  bias deltas against DLC `B` of `-1:7, 0:238, +1:11`.
- The recovered scale/control rule splits each positive fp16 scale interval
  into quarters: `[0,0.25)` and `[0.75,1)` use control `0x0040`, while the
  middle half uses `0x8040`.  The A8 overflow edge is also aligned: if the
  exact drain scale is outside finite fp16 range, the active
  `ConvLayer.opt.bias_scale_shuff` path divides by `2^32` before the same
  quarter encoding, so `w4a8_per_channel/negative_boundary` is byte-exact
  against the native sidecar.
- The `+/-1` bias deltas are not produced by qairt-quantizer: the quantized DLC
  `B` equals the generated `bias_q`.  The deltas first appear when HTP prepare
  lowers graph-before `Conv2d_w_scale(scale float32, B sFxp32)` into final
  `q::ConvLayer.opt.bias_to_vtcm` sidecar bytes.
- A ctxgen gdb probe on the zero-sum sweep confirms the prepare call path hits
  `dequantize_bias` and `find_bias_scale` under
  `GraphPrepare::run_optimize_passes_single_registry`; it does not hit
  `compute_scale`, `conv_round`, or libc `roundf`.  Treat
  `bias_to_vtcm` itself as static VTCM materialization, not the place where the
  numeric `+/-1` bias conversion is introduced.
- A narrower callsite/`Replacement::Op_inner` trace shows the graph-prep
  chain first constructs `scale_normalizing(Scale, Max_scale)` and then
  `requant_bias(Bias, scale_normalizing(...))`.  It also builds
  `ConvLayer.opt.convert_bias.simple` with `WeightScale`, `TotalScale`,
  `OutOff`, and `ConvCtrl`, followed by `ConvLayer.opt.adjust_bias`,
  `ConvLayer.v73.opt.convert_bias`, and `ConvLayer.opt.bias_scale_shuff` before
  `bias_to_vtcm`.
- The active numeric loop is now identified at `0x1ba8116` under
  `GraphPrepare::const_prop`.  It performs the `requant_bias` stage:
  `nearbyintf(input_f32 * encoding_mul + offset)`, clamp, then `cvttss2si`.
  Combined with `dequantize_bias` and `find_bias_scale`, this gives the exact
  QNN Native sidecar-bias rule:

  ```text
  global_bias_scale = float32(act_scale) * max(float32(weight_scale))
  dequant = float32(DLC_B * global_bias_scale)
  find_bias_scale = float32(max(abs(dequant)) * 16 / 2^32)
  expanded = nearbyintf(float32(dequant * float32(1 / find_bias_scale)))
  final_bias = trunc(float32(float32(expanded * find_bias_scale) / global_bias_scale))
  ```
- A targeted gdb int32 const scan excludes the currently instrumented generic
  int32 const path as the final bias serializer.  In the `normal_random`
  256^3 run, `dequantize_bias` still matches
  `DLC_B * act_scale * max(weight_scale)` for `256/256` words, but the int32
  const dumps have no 256-element DLC/final-bias vector; the notable int32 dump
  is a 25-int shape/control const beginning with `[4, 4, 1, 65536, ...]`.
- Final mapping gives the clean boundary: `bias_to_vtcm` consumes a const
  tensor `data_type=50 dims=[1,8,1,64]` and produces the VTCM sidecar consumed
  as input 2 by `q::ConvLayer_s1.opt`.  That const is exactly 512 int32 words,
  i.e. the 2048B sidecar record for `N=256`.  `bias_scale_shuff.int` is the
  arithmetic const-fold/evaluator; `bias_to_vtcm` is DMA materialization, not
  the arithmetic source.  Broad sidecar pattern scans have been removed from
  the gdb helper.
- Direct context-binary extraction now pins the boundary: for the zero-sum
  sweep, `case_native_ctx.bin` contains the selected native bias record at byte
  offset `58624`.  The 1280-byte slice at that offset is byte-identical to the
  extracted `context_binary_extracted_bias_to_vtcm.raw`; scale/control match
  for all 160 channels, but effective bias already differs from DLC `B` by
  `{-1: 16, 0: 128, +1: 16}`.  This proves the mismatch is introduced between
  quantized DLC `B` and serialized context binary, before device execution.
- Device optrace confirms the HTP node surface for the same zero-sum sweep:
  `q::ConvLayer.opt.bias_to_vtcm` appears as a runtime event for
  `qnn_op=conv1x1` before `q::ConvLayer_s1.opt` (`1999` and `3460` cycles,
  respectively).  Final mapping shows the sidecar tensor as
  `Int32 [1,5,1,64]` for `N=160`.
- A zero-sum-weight probe (`32x128x32`) keeps `folded_i32 = 0` and still shows
  DLC `B` exact with final sidecar deltas `-1:4, 0:25, +1:3`.  Changing output
  scale and per-channel weight scale does not change that probe's delta set,
  which confirms the rule belongs to HTP prepare's bias conversion/shuffle path,
  not weight-sum folding or the fp16 drain-scale selector.
- The bias issue now has a dedicated reproducible example:
  `example/qnn_hmx_matmul_u8i8/bias_prepare_probe/`.  It runs without device
  execution, generates zero-sum-weight probes, prepares QNN Native contexts,
  reads quantized DLC `B`, extracts final `bias_to_vtcm`, and writes
  `analysis/summary.json` / `summary.md`.
- Formula-generated custom sidecar now matches QNN Native without injecting the
  extracted native record.  Fresh device evidence:

  ```bash
  OUT_ROOT=/tmp/qcom_htp_u8i8_formula_gate USE_NATIVE_BIAS_RECORD=0 \
    BUILD_PACKAGES=0 DEVICE=oneplus CHAIN=1 CASE_NAME=normal_random \
    M=256 K=256 N=256 bash scripts/run_u8i8_python_case_custom_native_match.sh
  ```

  produced `native vs custom bytes: 2048/2048` and same-hardware custom/native
  output `65536/65536`, `maxdiff=0`.

Current per-channel bias status: the HTP prepare rule that converts quantized
DLC `B` plus Conv scale metadata into the final `bias_to_vtcm` effective-bias
field is recovered.  U8-output kernels (`u8i8`, `w4a8_per_channel`) use the
prepared bias directly; U16-output kernels (`w8a16`, `w4a16_per_channel`) use
`trunc(prepared_bias / 256)` in the 512-byte native record.  Do not attribute
this case to kernel code, qairt-quantizer bias quantization, activation layout,
or weight layout differences.

Durable write-up and shared implementation:

- [QNN HTP per-channel bias prepare](../guides/qnn_htp_perchannel_bias_prepare.md)
- [scripts/qnn_htp_bias_prepare.py](/home/zzh/work/qcom_htp/scripts/qnn_htp_bias_prepare.py)

Same-hardware comparisons remain exact-output gates.  The small tolerance used
for QNN Native HTP vs Python/AIMET-style oracle does not apply to custom/native
or handwritten/custom comparisons.

The handwritten MatMul route is QNN-free at runtime.  The implementation lives
under `example/handwritten_hmx_matmul/` and is validated by:

```bash
HANDWRITTEN_HMX_MATMUL_OUT_ROOT=/tmp/handwritten_hmx_matmul_gate_device_refresh2 \
  DEVICE=oneplus tests/qnn_kernel_e2e/correctness/test_handwritten_hmx_matmul_e2e.sh
```

Current gate status: pass.

The gate promotes all active families:

- `u8i8`
- `w4a8_per_channel`
- `w8a16`
- `w4a16_per_channel`

W16A16 remains retained reference material only.

## W4A16 Acceptance

W4A16 is accepted through the QNN custom-op baseline, not by re-entering the
QNN native Conv graph.

Current evidence:

- `device_body_w4a16_chain8_custom_baseline.json`: direct-HMX device body is
  `byte_exact_device_diff`, byte diffs `0`, checksum `0xfcdb7a52`.
- `w4a16_chain8_custom_baseline_native_bridge.json`: custom/native public
  layout bridge is `accepted_bridge=true`.
- The required public-layout transform is the existing Python
  `native_transpose_2d` rule.
- `native_transpose_2d` is `65536/65536` exact with byte diffs `0`.

The old tutorial chain1 wrapper has been removed from the active gate.  Its
useful conclusion is retained here: it proved prepared-state, call-ABI,
VTCM-offset, step-trace, and HNH-path visibility, but it remained a
checksum-mismatched debug route and was not the W4A16 acceptance oracle.

## Gate Artifacts

The last full device gate wrote:

- `/tmp/handwritten_hmx_matmul_gate_device_refresh2/promotion_evidence.json`
- `/tmp/handwritten_hmx_matmul_gate_device_refresh2/completion_checklist.json`
- `/tmp/handwritten_hmx_matmul_gate_device_refresh2/roadmap_audit.json`

Important machine-readable status:

- `promotion_evidence.json`: `m4_promoted_count=4`,
  `m4_promotable_count=4`, `unpromoted_families=[]`.
- `completion_checklist.json`: `roadmap_complete=true`, `pass=3`, `open=0`,
  `fail=0`.
- `roadmap_audit.json`: `w4a16_complete=true`, `pass=4`, `open=0`, `fail=0`.

## Historical Boundary

The old W4A16 QNN blackbox/native-entry investigation is closed for the current
implementation goal.  Its useful result is provenance: QNN Native/custom-op
artifacts define prepared bytes, raw output oracles, body slices, packet counts,
and native performance references.  They are not runtime dependencies.

Do not wire old descdump, ctxgen, descriptor mutation, selector sweep, or
residual/cvt microprobe matrices into the current gate.
