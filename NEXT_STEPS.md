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

## Open items (next candidates) — see `Agent/matmul_blueprint_2026-04-25.md`

**Core finding**: QNN's HMX utilization is only 18.4% (12K cyc MAC in 66K
timeline). Speed comes from graph-level slicing + parallel HVX pack on
2 threads, NOT HMX-HVX overlap. V8 HMX kernel is already correct.

Ranked ROI:

1. **Slice ONNX graph M/N into halves → 4 MatMulV8 + 2 pack_act + 2 pack_wt
   + Concat** (Python-only, `gen_v8_onnx.py` edit). Expected ~3×, no kernel
   changes. Validates that QNN scheduler dispatches the 2 pack_act
   instances to separate HVX threads.
2. **HVX-rewrite `pack_act_rm_hvx.c` + `pack_wt_v3_hvx.c`** to use
   `V6_vshuffvdd(Vu,Vv,-32)` topology (4 rows × 128 cols per iter, ~8 HVX
   insns vs our ~128 scalar cyc/tile). Expected ~3× on pack cycles.
3. **Add weights-to-VTCM prefetch op on HMX resource** — runs during HVX
   pack window, hides DDR→VTCM latency.
4. **`TcmDramCopy` → `UntileToRowMajor`** fused Crouton-untile + DDR write
   (kernel already exists at `kernel/untile_to_rowmajor_hvx.c`).

Dead ends confirmed (don't retry):
- `:dilate` / `mxswapacc` / `:retain` modifiers on `:cm:sat.ub` — no effect.
- Multi-threaded HMX — only one HMX unit, `tid=256` is the only one.
- Row-major scatter in mmv8 — DDR-latency bound at ~1.3M cyc @ 512³.

## Reference docs

- `docs/qnn_custom_op_sop.md` — **canonical** SOP for custom-op flow
- `Agent/v8_vs_native_optrace_2026-04-25.md` — full V8 vs QNN trace diff + reproduce
- `Agent/qnn_vs_v8_root_cause_2026-04-24.md` — why V8 uses tile-layout output
- `Agent/hmx_sat_ub_semantics_2026-04-24.md` — `:cm:sat.ub` silicon formula
- `example/hmx_matmul_phase3/README.md` — dir-level entrypoint
- `example/hmx_matmul_phase3/_archive/README.md` — what's archived, how to restore
