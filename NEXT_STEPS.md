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

### Standard flow — shape-adaptive ONNX (V9)
```bash
cd example/hmx_matmul_phase3
bash build.sh && bash build_x86.sh

cd standard_flow/phaseB_v8
# Any shape:
python gen_v8_graph.py --M 4096 --K 4096 --N 4096
(cd gen_out/HmxMatMulPhase3Package_Converter_Op_Package && make cpu)
# Then convert → ctxgen → qnn-net-run (see docs/qnn_custom_op_sop.md §7-§10)
# Or one-shot sweep:
bash sweep_v9.sh                           # 512/1024/2048/4096
SHAPES="32 128 256" bash sweep_v9.sh       # override
```

## Current V9 perf vs QNN native (2026-04-25, see `Agent/v9_sweep_results_2026-04-25.md`)

| Shape | V9 cycles | QNN cycles | V9/QNN | V9 cyc/MAC | QNN cyc/MAC |
|-------|----------:|-----------:|-------:|-----------:|------------:|
| 512³  |    540K   |    67K     |   8.1× | 4.22e-3    | 5.2e-4      |
| 1024³ |   3.08M   |   182K     |  16.9× | 3.01e-3    | 1.8e-4      |
| 2048³ |   21.1M   |   1.42M    |  14.9× | 2.58e-3    | 1.7e-4      |
| 4096³ |   161M    |   28.9M    |   5.6× | 2.47e-3    | 4.4e-4      |

V9 cyc/MAC converges to ~2.5 (fixed overhead amortizes). QNN hits
best 0.17 at 1024-2048³, degrades at 4096³ due to spill/fill.

**VTCM overflow threshold**: between 1024³ (0 spill) and 2048³ (8.8 MB
spill). Compiler auto-inserts @Spill/@Fill for our custom ops when
the graph has enough tile instances. V9 at 4096³: 62 MB spill +
452 MB fill (hidden in the compiler's memory management).

## Open items (next candidates)

Ranked ROI (post-V9, see `Agent/matmul_blueprint_2026-04-25.md`,
`Agent/v9_sweep_results_2026-04-25.md`):

1. **HVX vshuff pack rewrite** — `pack_act_rm_hvx.c` at 600K cyc/call
   vs QNN's ForceFormat_Crouton ~5K cyc/call. Use `V6_vshuffvdd(Vu,Vv,-32)`
   topology (`Agent/forceformat_crouton_re.md` §4). Expected 3-5× total.
2. **Smaller N_TILE at large shape** — planner currently picks N_TILE=256
   everywhere; at 4096³ this forces 62 MB spill. Shrinking to 64 or 128
   (QNN uses 64 at 4096³) would reduce spill and bring us closer to QNN.
3. **Investigate mmv8 inner loop at large K** — per-MAC cost is 28× above
   QNN at 4096³ despite same silicon primitives. Likely bank-conflict or
   cache-line penalty from the 64-iter K loop; probe to confirm.
4. **Fix `--profiling_option optrace` on multi-instance V9 graphs**
   (execution fails; `detailed` works).

Dead ends confirmed (don't retry):
- `:dilate` / `mxswapacc` / `:retain` modifiers on `:cm:sat.ub` — no effect.

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
