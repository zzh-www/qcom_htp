# V8 vs QNN-native MatMul optrace comparison (2026-04-25, SM8650 v75)

Both traces captured via the **standard flow**: `qairt-converter ONNX→DLC →
qnn-context-binary-generator (with x86 op-pkg for V8 custom ops) → on-device
qnn-net-run --retrieve_context --profiling_option optrace →
qnn-profile-viewer + schematic → chrometrace.json`.

Shape: M=K=N=512, u8×i8→u8, single inference.

## Top-line HTP resource budget

| Resource | V8 custom-op graph | QNN native w8a8 MatMul |
|---|---|---|
| **HMX** timeline cycles | **317,009** (util 65.2%) | **66,513** (util 30.0%) |
| HMX cycles_used      | 206,652 | 19,958 |
| HMX DRAM read        | 530,432 B | 266,240 B |
| HMX VTCM read / write| 530,432 / 792,576 B | 1,056,768 / 528,384 B |
| HVX (main tid=512)   | 103,523 cyc util 32.7% | 9,517 cyc util 14.3% |
| HVX MT tids 513-515  | 291K, 290K, 296K cyc (~25% each) | 43K, -, 62K cyc (65%/81%) |

V8 is **4.8× slower on HMX** (317K vs 66K timeline cyc) and hits 3 HVX tids
doing pack_act work the QNN path never does (QNN uses HTP built-in
`InputSlicePad` + `ForceFormat_Crouton`).

## Per-op cycle breakdown (from chrometrace.json `dur` aggregated)

### V8 custom graph
| Op | duration (cyc) |
|---|---|
| `PackActivationU8RowMajor` | 520,132 |
| `MatMulV8`                 | 363,944 |
| `PackWeightToHmxTileV3`    | 138,524 |
| `TcmDramCopy`              |  31,308 |
| `InputSlice` + `OutputSlice` framework | 44K total |
| `weights_to_vtcm` (framework) | 4,130 |

### QNN native MatMul
| Op | duration (cyc) |
|---|---|
| `q::*InputSlicePad`        |  95,152 |
| `q::ForceFormat_Crouton`   |  78,092 |
| `q::*OutputSlice`          |  41,034 |
| `q::ConvLayer_s1.opt`      |  24,474  ← the HMX MatMul core |
| `q::ConvLayer.opt.weights_to_vtcm` |  8,654 |
| `q::ConvLayer.opt.bias_to_vtcm`    |  6,184 |

## Key findings

1. **Core HMX MatMul parity is close**. V8's `MatMulV8` at 364K cycles vs
   QNN's `ConvLayer_s1.opt` at 24K cycles looks like a 15× gap, but that's
   because QNN's `ConvLayer_s1.opt` is only the inner MAC loop; it depends
   on HVX-resident pre-activations (`ForceFormat_Crouton`) that V8 folds
   into its MatMulV8 op. Adding V8's MatMulV8 + PackAct + PackWt → ~1,022K
   vs QNN's MatMul+pack+crouton sum ~222K — **V8 is ~4.6× slower overall**.

2. **Pack-activation is the biggest gap**. V8's
   `PackActivationU8RowMajor` = 520K cycles, QNN's equivalent (InputSlicePad
   + ForceFormat_Crouton) = 173K. Factor 3× — V8's row-major HVX pack
   leaves performance on the table.

3. **HMX utilization low on V8 (65%)** vs QNN (30%) — misleading because
   QNN's HMX usage is shorter (66K timeline) so utilization is over a
   shorter window; absolute HMX-active cycles 207K (V8) vs 20K (QNN).

4. **VTCM traffic pattern differs**:
   - V8: more DRAM read into HMX tid (530K vs 266K) → HMX pulls packed_act
     tiles from VTCM but PackActivationU8RowMajor is writing them to VTCM
     via DRAM (read from DDR as u8, write to VTCM packed).
   - QNN: less HMX-side DRAM read because `ForceFormat_Crouton` pre-stages.

5. **QNN overlaps HVX with HMX aggressively**. Three HVX tids at 65-81%
   utilization in shorter timelines → QNN uses HVX to prefetch next tile
   while HMX MACs current. V8 does pack_act *fully before* HMX starts
   (serialized).

## Trace files

| File | Purpose |
|---|---|
| `standard_flow/phaseB_v8/device_out/chrometrace.json` | V8 flame-graph ready for Perfetto/chrome://tracing |
| `standard_flow/phaseB_v8/device_out/chrometrace_qnn_htp_analysis_summary.json` | V8 HTP resource summary |
| `standard_flow/phaseA_native/baseline_s512_w8a8_existing/chrometrace.json` | QNN-native baseline (existing) |

Load both into chrome://tracing → compare timelines side by side.

## Perf takeaways for V8

- `PackActivationU8RowMajor` is the #1 target: 520K → ~170K (QNN-level)
  would save ~350K cycles, bringing total closer to QNN's core.
- HVX↔HMX overlap: current V8 graph serializes (pack → MatMul → copy).
  Inlining pack_act into MatMulV8 (the way `ConvLayer_s1.opt` does via
  Crouton prefetch) would let HMX kick off before all tiles are packed.
- V8's MatMulV8 internal cycle is dominated by `:after:cm:sat.ub` latency
  (established in `Agent/v8_tile_layout_2026-04-24.md`); pure-HMX design
  can't add HVX-overlapped requant, so the 364K floor is near-optimal
  given the constraint.

## Reproduce (full command set)

```bash
cd example/hmx_matmul_phase3/standard_flow/phaseB_v8

# 1. build x86 op-package (one-time)
bash ../../build_x86.sh   # → build/x86_64-linux-clang/libQnnHmxMatMulPhase3.so

# 2. ONNX (custom-domain) + Converter Op Package .so
python gen_v8_onnx.py
(cd gen_out/HmxMatMulPhase3Package_Converter_Op_Package && make cpu)

# 3. qairt-converter → DLC (NONTRIVIAL layout so dims aren't permuted)
CPL=$(pwd)/gen_out/HmxMatMulPhase3Package_Converter_Op_Package/ConverterOpPackage/libConverterOpPackage.so
qairt-converter -i v8_model.onnx \
    --op_package_config MatMulV8Package.xml --converter_op_package_lib $CPL \
    --source_model_input_layout act_raw NONTRIVIAL \
    --source_model_input_layout bias_fp16 NONTRIVIAL \
    --source_model_input_layout scratch NONTRIVIAL \
    --source_model_output_layout out NONTRIVIAL \
    --desired_input_layout act_raw NONTRIVIAL \
    --desired_input_layout bias_fp16 NONTRIVIAL \
    --desired_input_layout scratch NONTRIVIAL \
    --desired_output_layout out NONTRIVIAL \
    -o v8_model.dlc

# 4. DLC → context binary + schematic (on host)
X86_PKG=$(cd ../../build/x86_64-linux-clang && pwd)/libQnnHmxMatMulPhase3.so
qnn-context-binary-generator \
    --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
    --dlc_path v8_model.dlc \
    --op_packages "$X86_PKG:HmxMatMulPhase3InterfaceProvider" \
    --binary_file v8_ctx --output_dir ctx_out \
    --config_file htp_config.json \
    --profiling_level detailed --profiling_option optrace \
    --save_backend_op_mapping
# Produces: ctx_out/v8_ctx.bin + v8_model_schematic.bin

# 5. Push + run on device (MUST use --use_native_input_files since our u8/u16
#    .raw files are already in native format, not fp32 that QNN would quantize).
ssh oneplus "cat > qnn_run/phaseB/v8_ctx.bin" < ctx_out/v8_ctx.bin
ssh oneplus "cat > qnn_run/phaseB/v8_model_schematic.bin" < v8_model_schematic.bin
ssh oneplus 'cd ~/qnn_run/phaseB && LD_LIBRARY_PATH=../:.:/vendor/lib64 ADSP_LIBRARY_PATH=../ \
    ../qnn-net-run --backend ../libQnnHtp.so --retrieve_context v8_ctx.bin \
      --op_packages ../libQnnHmxMatMulPhase3_cpu.so:HmxMatMulPhase3InterfaceProvider:CPU,../libQnnHmxMatMulPhase3_htp.so:HmxMatMulPhase3InterfaceProvider:HTP \
      --input_list input_list.txt --profiling_level detailed --profiling_option optrace \
      --config_file htp_config.json --output_dir out --use_native_input_files'

# 6. Pull + render chrometrace
ssh oneplus "cat qnn_run/phaseB/out/qnn-profiling-data_0.log" > device_out/profile.log
qnn-profile-viewer --config device_out/optrace_config.json \
    --reader $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so \
    --input_log device_out/profile.log \
    --schematic v8_model_schematic.bin \
    --output device_out/chrometrace.json
```

## Critical flags / gotchas learned

- **`--use_native_input_files`** is mandatory when feeding pre-quantized u8/u16
  .raw files. Without it, qnn-net-run tries to interpret them as fp32 and
  errors with "batch size = 2 expected 4" (size mismatch × 4 because fp32
  is 4× uint16).
- **NONTRIVIAL layout on every IO** (4 CLI flags: source+desired × input+output)
  stops the NHWC dim permutation that breaks rank-4 custom tensors.
- **x86 op-package** only needs `libnative` (Hexagon intrinsics stubs) + libc,
  no libQnnHtp/libHtpPrepare linkage. See `build_x86.sh`.
- **`--save_backend_op_mapping`** on ctxgen produces `<graph>_schematic.bin`
  required by qnn-profile-viewer's optrace reader.
- Bias + scratch must be `APP_WRITE` (not STATIC) to match V8 kernel's
  VTCM residency expectations; weight is STATIC.
