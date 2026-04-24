# V8 matmul — standard QNN flow validated (2026-04-25)

> **Status**: V8 goes through `ONNX → DLC → context-binary → qnn-net-run →
> chrometrace` end-to-end. Only V8 path is kept under active maintenance;
> V2-V7 / int4 / probe code moved to `example/hmx_matmul_phase3/_archive/`.
> Standard operating procedure persisted at **`docs/qnn_custom_op_sop.md`**.

## What exists now

### Reference impl
`example/hmx_matmul_phase3/`
- V8-only src (3 files), kernels (4 files), `build.sh` + `build_x86.sh`.
- `standard_flow/phaseB_v8/` — ONNX + ConverterOpPackage + DLC + ctx-binary
  + qnn-net-run harness; produces `device_out/chrometrace.json`.
- `standard_flow/phaseA_native/baseline_s512_w8a8_existing` → symlink to
  the QNN-native MatMul optrace in `example/qnn_matmul_profile/sweep_data_*`.

### SOP
`docs/qnn_custom_op_sop.md` — 14-section standard operating procedure for
building any future QNN custom op through this pipeline. FAQ table at §11
captures the 11 landmines we hit (NONTRIVIAL 4-flag combo, x86 op-pkg
libnative-only linkage, `--use_native_input_files`, …).

### Perf numbers (512³ u8×i8, SM8650 v75)

| Metric                      | V8 (this) | QNN native w8a8 | V8/QNN |
|-----------------------------|----------:|----------------:|-------:|
| HMX timeline cycles         | 317,009   |   66,513        | 4.8×   |
| HMX active cycles           | 206,652   |   19,958        | 10.4×  |
| pack_act                    | 520,132   | 173,244 (InputSlicePad+ForceFormat_Crouton) | 3.0× |
| mmv8 / ConvLayer_s1.opt core| 363,944   |   24,474        | 14.9×  |
| tcm_dram_copy               |  31,308   |     —           | —      |

V8's host-harness standalone path (`run_matmul_v8_graph`) at 512³: ~358K
total cycles — matches V6. Standard-flow path at 512³: ~940K total (more
framework overhead from `qnn-net-run` + DLC load; core kernel cycles same).

## Resume

### Fastest iteration — V8 host harness (bit-exact DIAG modes)
```bash
cd example/hmx_matmul_phase3
bash build.sh
bash run_v8_graph_on_device.sh --shape 512,512,512    # ~358K cyc
bash run_v8_graph_on_device.sh --shape 1024,1024,1024 # ~2.7M cyc

# DIAG at 32³ (all 0/1024):
for d in 999 998 997 996 995; do
  ssh oneplus "cd ~/qnn_run && LD_LIBRARY_PATH=.:/vendor/lib64 ADSP_LIBRARY_PATH=. \
     ./run_matmul_v8_graph 32 32 32 $d" | grep Check
done
```

### Standard flow — ONNX → DLC → ctx-binary → qnn-net-run optrace
```bash
cd example/hmx_matmul_phase3
bash build.sh && bash build_x86.sh

cd standard_flow/phaseB_v8
python gen_v8_onnx.py
(cd gen_out/HmxMatMulPhase3Package_Converter_Op_Package && make cpu)

# Full pipeline (see docs/qnn_custom_op_sop.md §7-§10 for exact commands;
# ctxgen + schematic + on-device qnn-net-run --use_native_input_files +
# qnn-profile-viewer --schematic … → chrometrace.json).
```

## Open items (next candidates)

- Close the 3× pack_act gap to QNN's `InputSlicePad + ForceFormat_Crouton`.
  Crouton is 12.5%-density crouton layout; staying row-major by design
  may cap at ~2× if we inline pack into the HMX op (see
  `Agent/qnn_hmx_pipelining.md` for overlap notes).
- 1024³: V8 stays 1.6× slower than V6 because V6 overlaps HVX requant with
  HMX MAC and V8 is pure-HMX. No easy win while keeping HMX-only invariant.
- Random-mode residual: 28/1024 @ 32³ max_err=1 (fp16 rounding edge), and
  K-scale drift at 1024³. DIAG uniform modes all 0 mismatches up to 512³.

## Reference docs

- `docs/qnn_custom_op_sop.md` — **canonical** SOP for custom-op flow
- `Agent/v8_vs_native_optrace_2026-04-25.md` — full V8 vs QNN trace diff + reproduce
- `Agent/qnn_vs_v8_root_cause_2026-04-24.md` — why V8 uses tile-layout output
- `Agent/hmx_sat_ub_semantics_2026-04-24.md` — `:cm:sat.ub` silicon formula
- `example/hmx_matmul_phase3/README.md` — dir-level entrypoint
- `example/hmx_matmul_phase3/_archive/README.md` — what's archived, how to restore
