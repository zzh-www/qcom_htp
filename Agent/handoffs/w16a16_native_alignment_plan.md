# W16A16 Native Alignment Plan

Status: the canonical 256^3 W16A16 native-record contract is aligned through the
scoped `W16A16_KERNEL_PROFILE=accepted` path.  The current accepted artifact,
`output_w16a16_accepted_256/`, matches the native raw output exactly
(`65536/65536`, `maxdiff=0`), passes the standard native-artifact checker, and
passes `scripts/analyze_w16a16_custom_run.py` with
`alignment_gate.accepted=True`.  Default device builds still define
`HMX_W16A16_SKIP_KERNEL`; the real-kernel path is deliberately opt-in because
the validated native-record fields are scoped to the 256^3 native oracle and
the public custom boundary is one tiled QHPI op internally issuing two N128 body
calls instead of native's two `ConvLayer_s1.opt` graph nodes.

This plan follows
[`qnn_native_alignment_blackbox_handbook.md`](../guides/qnn_native_alignment_blackbox_handbook.md):
native runtime raw output and native lowered execution are the oracle; formulas,
decoded comments, and custom probes are diagnostics only.

## Current Native Oracle Candidate

Use this artifact as the current 256^3 candidate oracle:

```text
example/qnn_matmul_profile/output_w16a16_native_ref_e2e_256/
```

Verified locally:

- `scripts/check_qnn_artifact_standard.py ... --require-native-io --require-layout-flags --reject-float-io`: `ok`
- native input: `runtime_inputs_native/A.raw`, `uint16_le`, `131072` bytes,
  SHA256 `37a0c7cf9dc4868a04a1f00b4cdf0e622bfe062559969a7bf0a63eb366d913cd`
- native output: `device_out/Y.raw`, expected `uint16_le`, `131072` bytes,
  SHA256 `4a0c0f0250b836809320a4428ff8aa5b673d25c5056c5bb820c2b0aefcb5a588`

Refresh it before final acceptance or after any runner/QNN-SDK change:

```bash
FLAT_OUT=1 CONFIGS=w16a16 SHAPE=256,256,256 NUM_INFERENCES=3 \
OUT_DIR="$PWD/example/qnn_matmul_profile/output_w16a16_native_ref_e2e_256" \
bash example/qnn_matmul_profile/profile_all.sh

python3 scripts/check_qnn_artifact_standard.py \
  example/qnn_matmul_profile/output_w16a16_native_ref_e2e_256 \
  --require-native-io --require-layout-flags --reject-float-io
```

Current native lowering signals from the candidate artifact:

- public runtime raw I/O is `U16 [1,256,256]`;
- lowered graph uses `InputSlice`, `ForceFormat_Crouton`,
  two `q::ConvLayer_s1.opt` kernel events, `Concat`, and `OutputSlice`;
- the two HMX kernel outputs are each `UFixed16 [1,1,256,128]`, then concatenated
  into `UFixed16 [1,1,256,256]`;
- each `ConvLayer_s1.opt` consumes the Crouton activation plus its own prepared
  weight and bias/control sidecars;
- first-inference native kernel events are `39107` and `36326` cycles, both
  `8836` packets; native `matmul_1` aggregate is `82644` cycles and full
  timeline span is `124593` cycles.
- schematic prepared sidecar constants are now recorded by the analyzer:
  weight halves are `QInt8 [1,1,256,256]`, `65536` bytes each, content hashes
  `140d942f` and `ba8f0f70`; bias/control halves are `Int32 [1,4,1,128]`,
  `2048` bytes each, content hashes `d66bd58a` and `a24f0938`.

Generated baseline files:

```bash
uv run python scripts/analyze_w16a16_native_run.py \
  example/qnn_matmul_profile/output_w16a16_native_ref_e2e_256
```

This writes:

- `analysis/w16a16_native_summary.json`
- `analysis/w16a16_native_summary.txt`

## Current Probe Evidence

Current embedded body byte check:

```bash
uv run python .codex/skills/hmx-inline-asm/scripts/verify_hexagon_inline_asm.py \
  --inc example/qnn_hmx_matmul_w16a16/src/v73deep_conv1x1_kernel.inc \
  --so tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so \
  --vma 0x2fa740 --size 1800
```

Result: the checked-in `.inc` is byte-identical to
`libQnnHtpV75Skel.so@0x2fa740` for 1800 bytes.  This is the
`hmx_v73_convhhh1x1_stride1` slice.  It is the current performance-matching
diagnostic candidate only when paired with the `HMX_W16A16_K_TOTAL_BYTES_SCALE_K`
descriptor diagnostic.  This proves only the copied slice is exact; it does not
prove that native W16A16 enters this slice.

Current native-skel entry evidence:

- `Agent/qnn_re/skel_text_full.S` has five non-sparse call sites into
  `hmx_v73_convhhh1x1_stride1` at `0x3ddf18`, `0x3ded64`, `0x3df338`,
  `0x3df9d8`, and `0x3e0078`.
- The four large dispatcher sites use the same register setup pattern as the
  neighboring HMX conv kernels: `r0/r1` are emitted as `combine(r19,r20)`,
  `r2/r3` as `combine(r18,r17)`, and `r4/r5` as `combine(r16,r21)`.  This is a
  generic six-register HMX conv ABI, not proof that the W16A16 graph selected a
  specific branch.
- The compact wrapper at `0x3decc0..0x3ded6c` is more concrete.  It derives
  `r1 = base+0x10`, `r6 = base+0x28`, `r4 = base+0x48`, loads
  `r2 = memw(base+0x8)` and `r5 = memw(base+0x80)`, then calls
  `hmx_v73_convhhh1x1_stride1` with `r0 = r6` and `r3 = memw(base+0xc)`.
  Because the call packet does not use `.new` on the `r0` source for the load,
  `r3` is loaded from the pre-call wrapper base, not from `r6+0xc`.
- That compact wrapper therefore supports this effective entry ABI:
  `r0=out_desc`, `r1=act_desc`, `r2=weight stream`, `r3=bias/control stream`,
  `r4=mask/params descriptor`, `r5=extra params`.  The current custom wrapper
  uses this ordering, so the remaining real-body failure is more likely in
  descriptor field values, pointer-table layout, prepared sidecar semantics, or
  native branch selection than in a simple swapped-register ABI.
- Dynamic shared-body proof now exists for the current native W16A16 oracle.
  `scripts/build_w16a16_hhh_entry_probe.py --mode entry` patches
  `libQnnHtpV75Skel.so@0x2fa740` with a tiny marker writer/return probe, then
  an isolated `~/qnn_loader_probe_w16a16` native context run changes public
  `Y.raw` SHA256 from
  `4a0c0f0250b836809320a4428ff8aa5b673d25c5056c5bb820c2b0aefcb5a588` to
  `e5ea39913aad73b8da611946ffba9d48cccd348bc8a6d4178bb4f09d4c93399c`.
  The two native `q::ConvLayer_s1.opt` events collapse from `39107/36326`
  cycles and `8836/8836` packets to `941/228` cycles and `76/76` packets, with
  output exactness `0/65536` versus the oracle.  This proves the shared
  `0x2fa740` hhh body is reached by both native W16A16 N128 nodes.
- The `convhhh` call sites above are under a broad `pkWeightsF16_TCM` static
  context in the stripped disassembly.  They are no longer strong evidence for
  the current native W16A16 graph, whose lowered schematic reports
  `pkSWeights_TCM` signed/packed weights.  Treat the `convhbh` candidate as a
  diagnostic bridge, not as the proven native entry.  The shared-body proof
  still does not identify which wrapper call site or prebuilt record path
  reaches `0x2fa740`; that requires an `r31`/base-record probe.
- `scripts/build_w16a16_hhh_entry_probe.py --mode pattern` writes
  `0x484d0000+i` into the first 512 u32 words of the first internal output
  block.  Public `Y.raw` contains no intact u32 pattern words, but does expose
  split u16 evidence: `0x484d` appears 128 times and low pattern values appear
  at public u16 positions such as `0..31`, `128..159`, `512..543`, and
  `640..671`.  Therefore W16A16 native descriptor/base-record dumps need a
  W16-specific output-map parser; do not reuse the W4 `crouton512` u32 parser
  blindly.
- `scripts/build_w16a16_hhh_entry_probe.py --mode pattern16` writes direct u16
  values `0x8000+i` into the first 512 halfwords of the first internal output
  block.  The public raw contains 812 tagged halfword hits covering 366 unique
  values; the first visible structure is even values at public u16 positions
  `0..31` and `128..159`, and odd values at `256..287` and `384..415`.
  This is cleaner than the u32 pattern but still not a complete parser.  Use it
  next to build the W16 hhh record/base parser before interpreting pointer
  fields.
- `scripts/build_w16a16_hhh_entry_probe.py --mode record16` plus
  `scripts/parse_w16a16_native_probe.py` now decodes a compact live native
  record through the `pattern16` map.  Current native W16A16 hhh entry fields:
  `r31=0x031ded6c`, `out_desc=[0x02d98878,4,256,256,1,128]`,
  `act_desc=[0x02d97f2c,8,512]`, `weight=0x04610800`,
  `bias=0x04620800`, `mask=0x02d98800`, `extra=0xfdd01000`.
  Mask words are
  `[0,0x700,0,0x77c,0,0,0x3ff,0,0,0,0,0,0x80,0,0xfdd01000,0]`,
  and extra words are `{1,1536}`.

Fresh custom probe results:

| Artifact | Result | Durable conclusion |
|---|---|---|
| `output_w16a16_realbody_current_payload_256/` | `MODE=chain` fails ctxgen at `reshape_in`: HTP rejects `UINT16 -> UINT16` `Reshape`; no device run. | Do not use raw `chain` mode for 256 W16A16 probes until the public-surface reshape issue is fixed. |
| `output_w16a16_realbody_current_payload_direct_256/` | `MODE=direct`, real body, generated payload reaches device then `Graph Execution failure`; `device_out/out.raw` is empty. | Current body/descriptor/payload contract is not executable.  The artifact still records the custom boundary mismatch. |
| `output_w16a16_descdump_noprecompute_chain_qdq_256/` | `MODE=chain_qdq`, no precompute, `HMX_W16A16_DESC_DUMP`; graph executes and standard checker passes, but raw output is diagnostic garbage. | `chain_qdq` is a viable ctxgen/device diagnostic surface; custom boundary is not native. |
| `output_w16a16_descdump_noprecompute_allblocks_chain_qdq_256/` | all-block descdump variant also executes, but `H8XD` is still not recoverable from public raw. | Descriptor bytes written into QHPI output blocks are not directly observable after the public output transform; recover the output map or use a different dump carrier before trusting descriptor raw. |
| `/tmp/qcom_htp_w16a16_native_split_ctx2/` | `MODE=chain_qdq --op-input-layout native_split`; ctxgen succeeds with two custom N128 ops.  Activation, weight shape, bias shape, and output shape now match native except custom weight dtype remains `UFixed16` instead of native `SFixed8`. | Split-N shape can be represented at ctxgen, but this is not an executable alignment path yet. |
| `output_w16a16_splitn128_surface_skip_nowrite_separate_256/` | no-precompute, `HMX_W16A16_SKIP_NO_WRITE`, split outputs exported separately; device still fails with `Graph Execution failure` before valid optrace. | The split custom surface is not runtime-executable in the current QHPI custom-op form, even when the callback returns without touching output. |
| `/tmp/qcom_htp_w16a16_native_split_u8carrier_s8_ctx/` | converter/op-package diagnostic attempted signed 8-bit weight carrier; ctxgen fails because QNN prepares the input as `QUInt8` while the QHPI kernel signature requires `QInt8`. | Matching native `SFixed8` weight sidecar needs a different import/preparation path; simply forcing the custom op signature to QInt8 is insufficient. |
| `output_w16a16_native_n128_single_skip_nowrite_256x256x128/` | single custom op with public native-like `[1,1,256,128]` output, no-precompute, no output write; device still fails with `Graph Execution failure`. | The failure is not caused by having two split custom ops or by concat; native-layout custom graph outputs are not runtime-executable in this QHPI form. |
| `output_w16a16_native_n256_single_skip_nowrite_256/` | single custom op with public native-like `[1,1,256,256]` output, no-precompute, no output write; device still fails. | Public custom surface must stay on the known executable tiled `Crouton_16` contract.  Native-style splitting has to happen inside the wrapper, not by changing the graph output surface. |
| `output_w16a16_tiled_internal_split_realbody_256/` | tiled public surface plus `HMX_W16A16_INTERNAL_SPLIT_N128`, real body under `ALLOW_UNVALIDATED`; device still fails before valid optrace/output. | Internal split is the right shape direction, but the real body/payload/descriptor contract is still invalid. |
| `output_w16a16_tiled_u8weight_skip_256/` | tiled public surface, `HMX_W16A16_QHPI_WEIGHT8`, `--w16-weight-carrier-dtype uint8`; skip kernel executes on device, passes standard checker, and records a 33-packet custom event. | 8-bit raw weight carrier is a viable executable path for importing candidate native `QInt8` sidecar bytes without requiring a `QInt8` QHPI signature. |
| `output_w16a16_tiled_u8weight_internal_split_realbody_256/` | same 8-bit carrier plus `HMX_W16A16_INTERNAL_SPLIT_N128`, real body under `ALLOW_UNVALIDATED`; device still fails before output/optrace. | The carrier type mismatch is no longer the only blocker; descriptor/body/sidecar bytes still do not match native. |
| `output_w16a16_import_ctx_weight_sidecars_256/` | context-derived candidate weight sidecars imported from `matmul_w16a16_ctx.bin` offsets `0x9100` and `0x19100`; public custom boundary becomes `weight=QUInt8 [1,1,256,512]`; real body still fails with `Graph Execution failure`. | The public tiled graph can carry both native 64K weight halves, but imported weights alone do not make the native body executable. |
| `output_w16a16_import_ctx_weight_bias_sidecars_256/` | same candidate weights plus context-derived 2K+2K bias/control sidecars and native `extra={1,1536}`; real body still fails before output/optrace. | Weight, bias/control, and extra-param candidates are not sufficient; the remaining blocker is likely the HMX entry proof or descriptor/register ABI. |
| `output_w16a16_import_ctx_sidecars_skip_256/` | same imported weight/bias candidates, but default `HMX_W16A16_SKIP_KERNEL`; device execution succeeds, standard checker passes, custom event is 33 packets. | The 131072-byte U8 over-carrier and 4096-byte native bias sidecar are runtime-executable.  Real-body failure is isolated to descriptor/body entry, not graph surface or carrier capacity. |
| `output_w16a16_hbhdeep_import_ctx_sidecars_256/` | imported context weight/bias sidecars, native `extra={1,1536}`, and byte-verified `hmx_v75_convhbh1x1deep_stride1` still fail with `Graph Execution failure` under the older C8 grouped pointer table. | Switching from `convhhh` to `convhbh` alone is insufficient; the pointer-table contract was still invalid. |
| `output_w16a16_hbhdeep_row4_import_ctx_sidecars_256/` | same `convhbh` body plus Crouton16 row4 pointer tables executes on device and passes the standard artifact checker, but output is not native-aligned: native-exact `12853/65536`, analytic-exact `28117/65536`, maxdiff `65535`; custom event is `12261` cycles and `1256` packets versus native W16A16 `75433` kernel cycles and two `8836`-packet kernels. | Row4 tables fixed the crash, but the body/descriptor loop contract is still wrong.  Packet count is near a small hbh-style workload, not the native W16A16 workload, so the active native entry or loop/count fields remain unresolved. |
| `output_w16a16_v73hbhdeep_row4_import_ctx_sidecars_256/` | byte-verified `hmx_v73_convhbh1x1deep_stride1` at `0x2f4ec0` plus the same row4 tables and imported native sidecars executes, but remains wrong: native-exact `13315/65536`, analytic-exact `29855/65536`, maxdiff `65535`; custom event is `13329` cycles and `1370` packets. | The v73 hbh-deep body named closest to `conv_layer_quant_v73<QUint16Crouton_TCM, pkSWeights_TCM>` still does not reproduce native W16A16. |
| `output_w16a16_hnhdeep_row4_import_ctx_sidecars_256/` | byte-verified `hmx_v73_convhnh1x1deep_stride1` plus the same row4 tables and imported native sidecars also executes, but remains wrong: native-exact `13490/65536`, analytic-exact `30287/65536`, maxdiff `65535`; custom event is `10371` cycles and `1370` packets. | `convhnh` is not the missing W16A16 native contract either.  It behaves like another small-family diagnostic, not the native W16A16 `pkSWeights_TCM` lowering. |
| `output_w16a16_hhh_row4_import_ctx_sidecars_256/` | byte-verified `hmx_v73_convhhh1x1_stride1` with row4 tables and imported sidecars executes, but is still too small: native-exact `11728/65536`, analytic-exact `25501/65536`, maxdiff `65535`; custom event is `14419` cycles and `2874` packets. | Row4 alone fixes the previous hhh crash, but the descriptor span is still too short. |
| `output_w16a16_hhh_row4_ktotalK_import_ctx_sidecars_256/` | same hhh body with diagnostic `HMX_W16A16_K_TOTAL_BYTES_SCALE_K` changes `out_desc+0x14` from `N_t*32` to `K_t*N_t*32`; custom event jumps to `78887` cycles and `18669` packets, near native `75433` cycles and `8836+8836=17672` packets, but output remains wrong. | `out_desc+0x14` is a byte loop span for hhh, and the scaled value reaches the native workload size.  This is the first performance-matching contract, not an output-aligned one. |
| `output_w16a16_hhh_row4_ktotalK_import_sidecars_nativeA_256/` | same performance-matching hhh contract with custom runtime input overridden to the native oracle `A.raw`; input SHA matches `37a0c7cf...`, but output is still wrong: native-exact `8818/65536`, maxdiff `65535`, custom event `80382` cycles and `18669` packets. | Output mismatch is now a valid native-input comparison.  The remaining blocker is not just mismatched activation input; it is likely prepared-weight semantics, output/table mapping, or another descriptor field. |
| `output_w16a16_hhh_row4_ktotalK_scratch_weight_nativeA_256/` | same native input and hhh span, but weight sidecars are carried through the raw `UInt8` scratch input and used directly by the wrapper; output becomes worse, native-exact `0/65536`, custom event `83129` cycles and `18669` packets. | The exact context-extracted QInt8 bytes are not directly consumable by this hhh body as a raw weight stream.  The QNN-prepared U8 weight input is currently the better diagnostic path, even though it is not native-aligned. |
| `output_w16a16_hhh_row4offset_ktotalK_import_sidecars_nativeA_256/` | same hhh span and native input, but `HMX_W16A16_ROW4_BLOCK_OFFSET` changes row4 table addressing to block-plus-byte-offset; native-exact drops to `2242/65536`, custom cycles jump to `469106`, packets remain `18669`. | Row4 byte-offset addressing is worse for both correctness and cycles.  Keep it opt-in only; the default row4 mapping remains the faster table-index form. |
| `output_w16a16_hhh_row4_actstrideM32_ktotalK_import_sidecars_nativeA_256/` | same hhh span and native input, but `HMX_W16A16_ACT_STRIDE_M32` changes the activation descriptor stride; device fails with `Graph Execution failure` and no usable output/optrace. | Activation table stride is not simply `M_t*4` for this contract.  Keep this diagnostic opt-in only. |
| `output_w16a16_hhh_row4_mask70b_ktotalK_import_sidecars_nativeA_256/` | same hhh span and native input, but `HMX_W16A16_MASK_ARG1=0x70b`; native-exact is `8815/65536`, custom event is `69367` cycles and `18669` packets. | `arg1=0x70b` changes timing but does not improve output.  W8A16/W4A16-like `arg1` alone is not the missing W16A16 mask contract. |
| `output_w16a16_hhh_row4_mask702_ktotalK_import_sidecars_nativeA_256/` | same hhh span and native input, but `HMX_W16A16_MASK_ARG1=0x702`; native-exact is `8823/65536`, custom event is `83229` cycles and `18669` packets. | `arg1=0x702` also fails to move correctness.  The next mask work must recover the full native `set_hmx_params_conv1x1` tuple, not sweep `arg1` in isolation. |
| `output_w16a16_hhh_row4_mask70b_arg5a0_ktotalK_import_sidecars_nativeA_256/` | same hhh span and native input, but `HMX_W16A16_MASK_ARG1=0x70b,HMX_W16A16_MASK_ARG5=0xa0`; native-exact is `8805/65536`, analytic-exact `19602/65536`, custom event is `69438` cycles and `18669` packets. | Matching the W4-style mask word `[12]=0xa0` does not improve W16A16 correctness.  Treat it as another timing-only mask variant. |
| `output_w16a16_hhh_row4_mask70b_arg5704_ktotalK_import_sidecars_nativeA_256/` | same hhh span and native input, but `HMX_W16A16_MASK_ARG1=0x70b,HMX_W16A16_MASK_ARG5=0x704`, a static W16 bitfield candidate; native-exact is `8806/65536`, analytic-exact `19629/65536`, custom event is `70603` cycles and `18669` packets. | The high-bit `arg5` candidate also stays in the same wrong-output band.  Do not continue mask sweeps without runtime native mask/base-record evidence. |
| `output_w16a16_hhh_row4_native_record_fields_import_sidecars_nativeA_256/` | hhh body, internal N128 split, native input, imported context sidecars, and live record16 descriptor/mask fields: `out_y=256`, `n_tiles_pow2=256`, `m_total_minus_step=1`, `k_total_bytes=128`, `act_y=512`, `mask_arg1=0x70b`, `mask_arg5=0x80`, and mask word `[14]=extra_ptr`. | Correctness closes for the first time: native-exact `65536/65536`, maxdiff `0`.  Performance is still slower than native: custom event `94857` cycles and `19089` packets versus native kernel sum `75433` cycles and `8836+8836` packets. |
| `output_w16a16_hhh_row4_native_record_fields_precomp_split_import_sidecars_nativeA_256/` | same exact-output contract, but split N128 output tables are materialized during QHPI precompute instead of copied in the hot callback. | Correctness remains native-exact `65536/65536`; custom event improves to `92539` cycles and `18580` packets.  Remaining gap is likely custom wrapper/table-surface overhead versus native compact records, not arithmetic correctness. |
| `output_w16a16_native_record_profile_256/` | `W16A16_KERNEL_PROFILE=native_record_256` builds and runs the same contract from one profile switch.  The runner extracts native sidecars from the oracle context, injects native `A.raw`, verifies against native `Y.raw`, and writes `w16a16_run_profile.json`. | Reproducibility closes for the diagnostic profile: native-exact `65536/65536`, maxdiff `0`; custom event before global mask/extra cleanup was `94489` cycles and `18580` packets, timeline `149985`. |
| `output_w16a16_native_record_profile_precomp_desc_256/` | attempted to move split descriptors, act descriptor, extra, and mask into a larger QHPI precompute record. | Rejected: ctxgen succeeded, but device `Create From Binary` failed before execution.  Do not enlarge the precompute record for this path without first proving the QHPI precompute ABI size limit. |
| `output_w16a16_native_record_profile_globalmask_256/` | keeps the same exact-output profile, but uses a stable aligned global `extra_param` and patches mask word `[14]` once, avoiding the hot-path 16-word mask copy. | Diagnostic profile meets the native kernel budget: native-exact `65536/65536`, maxdiff `0`; custom event `71238` cycles and `17594` packets versus native `75433` cycles and `8836+8836` packets.  `alignment_gate.accepted` remains `False` only because the path is still diagnostic and the public custom boundary differs from native. |
| `output_w16a16_native_record_profile_aczero_256/` | same global-mask profile, but zeroes native weight sidecar lane sets `[0..3]` and `[8..11]` in every 16-byte group while preserving `[4..7]` and `[12..15]`. | Device executes, but output is no longer native exact: `29253/65536`, maxdiff `4060`.  The AC lanes are real compute payload, not ignorable metadata.  Sidecar generation must recover both lane groups. |
| `output_w16a16_native_record_profile_nativeac_genbd_256/` | preserves native AC lanes but replaces BD lanes with a linear `round(W*128)` K-major split128 stream. | Device executes, but output is worse: `13226/65536`, maxdiff `65535`.  BD is not a simple linear stream even though most native BD 16-byte chunks occur somewhere in that candidate. |
| `output_w16a16_native_record_profile_nativeac_genbd_formula_256/` | preserves native AC lanes and replaces BD lanes with the recovered split/tile/order formula, before the final q16 high-byte fix. | Output improves but is still wrong: native-exact `52370/65536`, maxdiff `1104`.  This isolated the last mismatch to the emitted high-byte representation, not the outer traversal. |
| `output_w16a16_native_record_profile_generated_weight_256/` | uses a generated weight sidecar byte-equivalent to the native sidecar while keeping the native bias sidecar. | Weight generation is closed for the 256^3 native ONNX `W`: native-exact `65536/65536`, custom event `68851` cycles and `17594` packets. |
| `output_w16a16_native_record_profile_generated_sidecars_256/` | generates both weight and bias/control sidecars from native ONNX `W`. | Sidecar import is no longer needed for this oracle: native-exact `65536/65536`, analytic-exact `14913/65536`, custom event `69853` cycles and `17594` packets versus native `75433` cycles and `8836+8836` packets.  Gate remains false only for diagnostic profile and boundary mismatch. |
| `output_w16a16_native_record_profile_auto_generated_sidecars_256/` | runner auto-generates `generated_sidecars/weights_qint8_2x65536.bin` and `generated_sidecars/bias_i32_2x2048.bin` from `W16A16_NATIVE_ORACLE_DIR/matmul.onnx` when `W16A16_SIDECAR_DIR` is unset. | Diagnostic reproducibility baseline: native-exact `65536/65536`, custom event `69385` cycles and `17594` packets, timeline `127315`.  Generated sidecar SHA256 values are `8c5a0896f9d94f735fb4443ca3c335f01c72b91bf24431659d2e3310596bebb2` for weights and `d3ba5eb13c478028ab1d9d60bb17cbcbbae56ab556b35d7462947fe3cbd1d75d` for bias. |
| `output_w16a16_accepted_256/` | first scoped accepted-profile run, built with `W16A16_KERNEL_PROFILE=accepted` instead of `ALLOW_UNVALIDATED`, auto-generated sidecars from native ONNX `W`, native input override, and native raw verification. | Accepted canonical 256^3 baseline: standard checker `ok`; native-exact `65536/65536`, maxdiff `0`; custom event `71283` cycles and `17594` packets, timeline `132916`; native kernel sum `75433` cycles and `8836+8836` packets.  `alignment_gate.accepted=True`; boundary check passes by explicit policy `single_custom_op_internal_split_n128` with one custom boundary and two native boundaries. |

Sidecar-origin diagnostic:

```bash
python3 scripts/analyze_w16a16_sidecar_origin.py \
  --sidecar-raw example/qnn_matmul_profile/output_w16a16_native_record_profile_globalmask_256/native_sidecars/weights_qint8_2x65536.bin \
  --custom-w-raw example/qnn_matmul_profile/output_w16a16_native_record_profile_globalmask_256/w16a16.onnx.wRaw_KN.npy \
  --native-onnx example/qnn_matmul_profile/output_w16a16_native_ref_e2e_256/matmul.onnx \
  -o example/qnn_matmul_profile/output_w16a16_native_record_profile_globalmask_256/analysis/w16a16_sidecar_origin.json
```

Earlier result: the simple candidates did not explain the imported native QInt8
sidecar.  The useful clue was that the BD lane set `[4..7,12..15]` exposed a
`round(W*128)`-like relationship in many 16-byte chunks, while AC-zero proved
the other lane set was real compute payload rather than padding.

Current result: `scripts/generate_w16a16_weight_sidecar.py` now reproduces the
native prepared weight and bias/control streams byte-for-byte for the canonical
256^3 native ONNX `W`.  The recovered weight formula is:

- `q16 = round(float64(W) * 32767)`, clipped to signed 16-bit;
- traversal is `N128 split -> N32 tile -> q16 low/high half -> K32 tile -> row group -> lane`;
- for each eight q16 values, emit `low[0:4]`, `rounded_high[0:4]`,
  `low[4:8]`, `rounded_high[4:8]`;
- `rounded_high = (q16 + 128) >> 8`.

The recovered bias/control formula is:

- control plane repeats int32 `[0x00404420, 0x40000000] * 16`;
- value planes are `((-sum(q16_col)) // 2)` alternating with zero for each N128
  split and 16-column group.

The generated combined streams are `131072` bytes for weights and `4096` bytes
for bias/control, with SHA256 values
`8c5a0896f9d94f735fb4443ca3c335f01c72b91bf24431659d2e3310596bebb2` and
`d3ba5eb13c478028ab1d9d60bb17cbcbbae56ab556b35d7462947fe3cbd1d75d`.

Custom/native analyzer command:

```bash
uv run python scripts/analyze_w16a16_custom_run.py <custom_out_dir> \
  --native-out-dir example/qnn_matmul_profile/output_w16a16_native_ref_e2e_256 \
  --native-raw example/qnn_matmul_profile/output_w16a16_native_ref_e2e_256/device_out/Y.raw
```

The current executable mismatch from
`output_w16a16_hhh_row4_ktotalK_import_sidecars_nativeA_256/`:

- custom boundary:
  `act=UFixed16 [1,8,32,256]`, `weight=QUInt8 [1,1,256,512]`,
  `bias=Int32 [1,8,1,128]`, `scratch=UInt8 [1,1,1,2048]`,
  `out=UFixed16 [1,8,32,256]`;
- native boundary, twice:
  `act=UFixed16 [1,1,256,256]`, `weight=SFixed8 [1,1,256,256]`,
  `bias=Int32 [1,4,1,128]`, `control=Int32 [1]`,
  `extra=Int32 [1,1,1,2]`, `out=UFixed16 [1,1,256,128]`.

The native sidecar byte streams are present exactly in the best custom context,
but at custom constant-layout offsets rather than the native context offsets:
`weight_0_qint8.bin` at `0xa000`, `weight_1_qint8.bin` at `0x1a000`,
combined weights at `0xa000`, `bias_0_i32.bin` at `0x9000`,
`bias_1_i32.bin` at `0x9800`, and combined bias at `0x9000`.  Do not run the
native fixed-offset extractor on the custom context and interpret the resulting
hash mismatch as a sidecar mismatch.

This boundary mismatch is still the primary structural blocker.  The hhh
native record fields now close output correctness even though the public custom
surface is still tiled `[1,8,32,256]` and native exposes two `[1,1,256,128]`
Conv nodes.  Do not continue broad body-family or mask sweeps.  Remaining work
is performance/surface cleanup: reduce the wrapper/table overhead and decide
whether the single-custom-op internal split is an acceptable boundary difference.

The closest ctxgen-only split-N diagnostic currently reports:

- custom boundary, twice:
  `act=UFixed16 [1,1,256,256]`, `weight=UFixed16 [1,1,256,256]`,
  `bias=Int32 [1,4,1,128]`, `scratch=UInt8 [1,1,1,2048]`,
  `out=UFixed16 [1,1,256,128]`;
- native boundary, twice:
  `act=UFixed16 [1,1,256,256]`, `weight=SFixed8 [1,1,256,256]`,
  `bias=Int32 [1,4,1,128]`, `control=Int32 [1]`,
  `extra=Int32 [1,1,1,2]`, `out=UFixed16 [1,1,256,128]`.

Do not promote that split-N custom graph: public native-layout custom outputs
fail on device even with `HMX_W16A16_SKIP_NO_WRITE`.  Keep the public custom
surface tiled and do any native-style N128 split inside the wrapper.  The native
signed-8 prepared sidecar bytes are now reproducible from the native ONNX `W`,
but the executable custom carrier remains `HMX_W16A16_QHPI_WEIGHT8` plus
`--w16-weight-carrier-dtype uint8`; `QInt8` QHPI signatures fail to match
because ctxgen prepares the custom input as `QUInt8`.

Saver backend was also checked against the native W16A16 DLC:

```bash
cd example/qnn_matmul_profile/output_w16a16_native_ref_e2e_256
LD_LIBRARY_PATH="$PWD/../../../tools/qnn-sdk/lib/x86_64-linux-clang:$LD_LIBRARY_PATH" \
../../../tools/qnn-sdk/bin/x86_64-linux-clang/qnn-net-run \
  --backend ../../../tools/qnn-sdk/lib/x86_64-linux-clang/libQnnSaver.so \
  --dlc_path matmul_encoded.dlc \
  --input_list runtime_input_list.txt \
  --output_dir /tmp/qcom_htp_w16a16_saver_dlc/out \
  --use_native_input_files --use_native_output_files \
  --num_inferences 1 --log_level verbose
```

The generated `params.bin` was `655360` bytes, exactly five `131072`-byte
records.  The first three records match the original 16-bit `W` payload SHA256
`bce8fea87bd5469173c665a930d1b7015b5ce8c3720d5bc431c1a9b5a558d8b9`; the
fourth matches runtime `A.raw`; the recorded C only contains a public QNN
`MatMul` node with `A`, `W`, and `Y`.  No native-lowered `ConvLayer_s1.opt`,
`65536`-byte `QInt8` weight half, or `2048`-byte bias/control sidecar is exposed.
Treat Saver as useful for public QNN API provenance only, not for HTP prepared
sidecar recovery.

## Acceptance Gate

The canonical 256^3 W16A16 native-record path is accepted when a fresh
`W16A16_KERNEL_PROFILE=accepted` custom run:

- compares byte-identical to `output_w16a16_native_ref_e2e_256/device_out/Y.raw`;
- passes the standard artifact checker with native I/O and layout flags;
- uses the live native W16A16 HMX body or a byte-verified readable rewrite;
- either enters the custom kernel at the same effective HTP tensor surface as
  native, preferably the native split-N `256x128 + 256x128` contract, or records
  a justified boundary difference for the single custom op internally issuing
  two native-style N128 calls;
- records bottom mapping, decoded optrace, packet counts, kernel-only cycles,
  native QNN-op aggregate cycles, and full timeline span;
- removes `HMX_W16A16_ALLOW_UNVALIDATED_KERNEL` and records
  `boundary_policy=single_custom_op_internal_split_n128` for the accepted
  profile;
- reaches native-class packet/cycle behavior or documents an explicitly accepted
  residual overhead budget.

## Remaining Work Plan

The black-box plan below is retained for provenance, but the canonical 256^3
accepted path is closed.  Remaining scope beyond that accepted path:

1. Keep default builds on `profile: skip` unless a broader shape contract is
   validated; the accepted real-kernel path is intentionally opt-in.
2. Refresh the native oracle after any QNN SDK, runner, or graph-generation
   change and rerun `output_w16a16_accepted_256/`.
3. Extend sidecar generation and native-record fields beyond the canonical 256^3
   shape before promoting general W16A16 coverage.
4. Revisit public native-layout QHPI outputs only with a new hypothesis; prior
   no-write probes failed on device, so the accepted boundary is the documented
   tiled custom op with internal N128 split.
5. Keep `scripts/analyze_w16a16_custom_run.py` as the machine gate for this
   work: `alignment_gate.accepted` must be `true`, not merely raw exact.

## Completed and Diagnostic Work Plan

### 1. Freeze the oracle and native surface

Keep the current native artifact as the working oracle, but do not call it final
until it has been freshly regenerated.  Extract a small `analysis/w16a16_native_summary.*`
from `native_io.json`, bottom mapping, and `optrace/summary.json` so every probe
can compare against the same tensor surface and event scopes.

Record these fields in the summary:

- public raw input/output storage and SHA256;
- all HTP tensors feeding each `ConvLayer_s1.opt` node, including dtype ids,
  dims, VTCM read/write bytes, and producer node;
- sidecar events and their cycle/packet counts;
- per-kernel `ConvLayer_s1.opt` cycles and packets;
- `matmul_1` aggregate cycles and full timeline span.

### 2. Prove the active native entry

Do not assume the embedded `v73deep_conv1x1_kernel.inc` is the right W16A16 body
just because the symbol name looks plausible.  Create an isolated W16A16 native
probe run, then patch or instrument the smallest native skel slice needed to
prove:

- which `ConvLayer_s1.opt` wrapper branch is taken for both N128 kernel nodes;
- the exact HMX entry VMA, symbol family, and byte size;
- the register ABI at the HMX entry: output descriptor, activation descriptor,
  weight pointer, bias/control pointer, mask/params pointer, and extra params;
- whether both N128 native nodes call the same body with different sidecars or
  two different call paths.

Only after this proof should the custom `.inc` be regenerated or renamed.

### 3. Byte-verify the kernel body

For the proven entry, regenerate the custom byte replica from the native skel and
verify it as a whole function:

```bash
python3 scripts/extract_hmx_kernel_bytes.py \
  --so tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so \
  --vma <proven_vma> --size <proven_size> \
  --out /tmp/w16a16_native_kernel.inc

python3 .codex/skills/hmx-inline-asm/scripts/verify_hexagon_inline_asm.py \
  --inc example/qnn_hmx_matmul_w16a16/src/v73deep_conv1x1_kernel.inc \
  --so tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so \
  --vma <proven_vma> --size <proven_size>
```

`HMX_W16A16_ALLOW_UNVALIDATED_KERNEL` may be used only for named diagnostic
artifacts.  It must not become the default acceptance path.

### 4. Recover descriptor, table, and output mapping

Add W16A16-specific probe parsers instead of using ad-hoc dumps.  The first
probe set should recover:

- output descriptor scalars for each N128 native node;
- activation descriptor scalars and table pointer layout;
- mask/control words and extra-param values;
- native table length and stride, not inferred from `256^3`;
- internal N128 output block ordering before native `Concat` and public output
  export.

Current static mask evidence: native callers around `0x3dd73c` and `0x3e07d4`
call `set_hmx_params_conv1x1` with live nonzero middle arguments, not just
`(mask, arg1, 0, 0, 0, arg5)`.  Adjacent branches load `arg1` candidates
`0x70b` and `0x702`, but custom probes that changed only `HMX_W16A16_MASK_ARG1`
did not improve output.  `scripts/emulate_hmx_conv1x1_params.py` can convert a
candidate tuple into the 16 helper-written mask words; for example
`0x70b,0,0,0,0xa0` yields
`[0,0x700,0,0x77c,0,0,0x7ff,0,0,0,0,0,0xa0,0,0,0]`, while
`0x70b,0,0,0,0x704` yields
`[0,0x700,0x64,0x718,0,0,0xdf,0,0,0,0,0,0x4,0,0,0]`.  Both custom probes fail.
Reconstruct the full tuple (`arg2..arg5`) from native instrumentation before
spending more runs on mask sweeps.

Run pattern probes before interpreting public `Y.raw` as a descriptor or table
dump.  A partially exact output must be classified by row/tile exactness,
sorted equality, histograms, N128 half behavior, and row/tile rolls.

### 5. Extract prepared payloads

Native W16A16 lowers through prepared weight and bias/control sidecars.  Compare
the prepared native bytes to custom-generated bytes before sweeping descriptor
scalars:

- dump both native weight sidecars feeding the two `ConvLayer_s1.opt` nodes
  from HTP-prepared artifacts or skel/runtime instrumentation, not Saver;
- dump both native bias/control sidecars and the small control tensors from the
  same HTP-prepared source;
- determine whether logical W16 is preserved as a 16-bit stream or lowered into
  a narrower/deep HMX weight stream;
- record SHA256, size, and first/last tile samples for each sidecar;
- add a generator option to import native sidecars for diagnostics.

Correctness with imported native sidecars but generated custom sidecars failing
means the blocker is packing/control generation, not the HMX body.

Current extraction status:

- `matmul_encoded.dlc` / Saver expose only the original `W16` parameter stream;
- context schematic records the native sidecar metadata and content hashes;
- `scripts/extract_w16a16_context_sidecars.py` now extracts candidate raw bytes
  from the current context binary:
  - `extra_0.bin`: offset `0x9000`, 8 bytes, SHA256
    `ef911f875d6535f213ed771ec64ff165efb970cd2ffbd161c7cc00bb0e520509`,
    values `{1,1536}`;
  - `weight_0_qint8.bin`: offset `0x9100`, 65536 bytes, SHA256
    `0200cc5214a2678270c63226ef5f00ee6c5295554c33407c3ba596e29cf3a060`;
  - `weight_1_qint8.bin`: offset `0x19100`, 65536 bytes, SHA256
    `7a296f50a45970cadd0c49f2e68307740a924dda48853e187d17595e786ec21e`;
  - `bias_0_i32.bin`: offset `0x29100`, 2048 bytes, SHA256
    `f52c1d0ec713fd62839e05a912a0c096f4b1a1239177049fca252a3c78ded8a2`;
  - `bias_1_i32.bin`: offset `0x29900`, 2048 bytes, SHA256
    `f8336f9369fe38ceab252e784cffd779a954cafb2785a632426ab2f14074dd8e`.
- imported context sidecars execute in skip mode but still fail in the real body,
  so the next high-value path is native-skel entry/ABI instrumentation, not more
  weight pack-order sweeps.

### 6. Run one-hypothesis custom probes

Use artifact names that encode the hypothesis, and change only one contract at
a time.  Suggested order:

1. `output_w16a16_marker_256`: current skip-kernel marker path, standard gate
   only.
2. `output_w16a16_descdump_*_256`: custom descriptor/table dump compared to the
   native descriptor summary.  Use `MODE=chain_qdq` or `MODE=direct`, not raw
   `MODE=chain`, for 256 until the UINT16 `Reshape` issue is fixed.
3. `output_w16a16_realbody_current_payload_*_256`: remove skip only under
   `ALLOW_UNVALIDATED`, using the current generated payload.
4. `output_w16a16_import_native_sidecars_256`: same descriptors, imported
   native payloads.
5. `output_w16a16_splitn128_surface_*_256`: keep using split-N only as a
   ctxgen/boundary diagnostic until the runtime `Graph Execution failure` is
   explained.
6. `output_w16a16_tiled_internal_split_*_256`: keep the public tiled custom
   surface executable and split output tables/weight/bias offsets inside the
   wrapper.  Do not remove `ALLOW_UNVALIDATED` until this path produces valid
   optrace/output.
7. `output_w16a16_u8weight_*_256`: use the executable unsigned-8 carrier to
   test candidate native prepared weight byte streams (`clip`, `hi8`, `lo8`,
   then imported native bytes via `--w16-weight-sidecar-raw` once recovered).
8. `output_w16a16_import_native_sidecars_256`: same descriptors, imported
   native signed-8 weight and bias/control sidecars.
9. `output_w16a16_splitn128_import_sidecars_256`: mirror the native two-N128
   body calls and output concat contract after the split surface is executable.
10. `output_w16a16_splitn128_generated_payload_256`: generated payload after
   byte-equivalent sidecar packing is implemented.

For every failed probe, keep the artifact and record what it ruled out.  Promote
neither exact output with wrong packets nor native-class cycles with partial
output.

### 7. Integrate only the native contract

After a probe closes correctness and performance, move only the native-backed
contract into the default path:

- default build removes `HMX_W16A16_SKIP_KERNEL` only after byte verification
  and standard native compare pass;
- diagnostic macros remain opt-in and named;
- default runner verifies native raw equality when
  `VERIFY_NATIVE_RAW=.../output_w16a16_native_ref_e2e_256/device_out/Y.raw` is
  provided;
- analysis output records custom/native boundary, raw compare, packet counts,
  and timeline scopes.

### 8. Final rerun and handoff

Close with fresh artifacts:

```bash
bash example/qnn_hmx_matmul_w16a16/build.sh
bash example/qnn_hmx_matmul_w16a16/build_x86.sh

VERIFY_NATIVE_RAW="$PWD/example/qnn_matmul_profile/output_w16a16_native_ref_e2e_256/device_out/Y.raw" \
OUT_DIR="$PWD/example/qnn_matmul_profile/output_w16a16_aligned_e2e_256" \
M=256 K=256 N=256 CHAIN=1 MODE=chain_qdq \
NATIVE_OUTPUT=1 STRICT_OPTRACE=1 \
bash example/qnn_hmx_matmul_w16a16/standard_flow/custom_w16a16/run_w16a16_chain.sh
```

Then update this file or replace it with a completed
`w16a16_native_alignment.md` handoff containing:

- native oracle and custom artifact paths;
- exact commands;
- checker output;
- raw compare and SHA256;
- kernel body byte verification;
- bottom mapping for final custom and native paths;
- optrace kernel/aggregate/timeline scopes;
- packet evidence;
- descriptor/table/payload evidence;
- known harmless boundary differences;
- open scope beyond canonical 256^3 chain8.
