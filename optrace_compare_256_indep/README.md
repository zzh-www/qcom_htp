# 256³ V8C8 vs Native — apples-to-apples op profiling (independent mode)

This is the **standard methodology** for op-level performance comparison:
- Build a graph with **N independent matmul instances** (no data dep)
- Each MatMul gets a **fresh model input** and a **fresh static weight**
- HMX is single-issue → instances serialise at the hardware level
- chrometrace shows N distinct events; first one carries iter/HMX-power-on
  setup, instances 2..N expose pure-kernel steady-state timing

This avoids three confounders that single-iter profile has:
1. **Iter 1 cold start dominates** chrometrace (the QNN reader always
   renders the first execute) — masks steady values
2. **Per-iter HMX setup overhead** is amortised across N instances
3. **Compiler-side folding** (e.g., MatMul(MatMul(A,W),W) → MatMul(A,W²))
   that single-instance can't expose

## Result at 256³ (M=K=N=256, w8a8, iter 3 steady)

| | V8C8 BbbKMajor | Native q::ConvLayer_s1.opt | ratio |
|---|---|---|---|
| Per-instance cyc (1..7 avg) | **7,981 cyc** | **6,466 cyc** | V8C8/Native = 1.23× |
| First-instance cyc          | 8,678 cyc      | 9,110 cyc                  | comparable |
| Total iter (8 inst + Input + Output)   | 170,176 cyc    | 109,134 cyc                | 1.56× |
| Wall (iter 3)               | 65 µs          | 41 µs                      | 1.59× |

Native HMX kernel has ~23% advantage in pure steady-state. The wall ratio
(1.59×) is wider because V8C8's per-instance Input/ForceFormat_Crouton
cost adds up across all 8 BbbKMajor instances (custom-op opaque to QNN
scheduler, no fusion fast path).

## Per-instance cyc breakdown (iter 3)

```
                   V8C8                        Native
inst 0 (cold)      8,678                       9,110
inst 1             7,925                       7,338
inst 2             7,981                       6,718
inst 3             8,288                       6,217
inst 4             8,019                       6,022
inst 5             7,838                       6,558
inst 6             7,827                       6,228
inst 7             7,993                       6,181
─────────────────────────────────────────────────
1..7 avg           7,981                       6,466
```

## Lowered graph

```
V8C8 (this bundle):                            Native (this bundle):
8 × HmxMatMulPhase3Package::BbbKMajor          8 × q::ConvLayer_s1.opt
1 × q::ConvLayer.opt.weights_to_vtcm           8 × q::ConvLayer.opt.weights_to_vtcm
1 × q::ConvLayer.opt.weights_to_vtcm@Fi.fi.    8 × q::ConvLayer.opt.bias_to_vtcm
8 × q::ForceFormat_Crouton                      8 × q::ForceFormat_Crouton
8 × q::*InputSlice                              8 × q::*InputSlice
8 × q::*OutputSlice                             8 × q::ForceFormat_Flat
                                                8 × q::Reshape (×2 = pre/post)
total: ~30+ nodes                              total: 64 nodes
```

V8C8 has fewer nodes per matmul because (a) wt and bias are SHARED static
across all 8 BbbKMajor (1 wt DMA, 1 bias DMA total) — but native has
per-instance wt + bias DMAs because each MatMul has its own W in the
ONNX graph. So V8C8's lowered graph is more compact, but the per-bbb
HMX kernel is slower.

If we made native **share W** across instances (--shared_w flag), the
graph would be more comparable (1 wt DMA total) but introduces compiler-
folding risk and changes the nature of the test. We test per-op W as
default for both.

## Methodology recipe — going forward

For any HMX op profiling:

1. Build an "independent" version of the op with N=8 instances:
   - Each instance has its own model input
   - Each instance has its own static weight (no sharing → no folding)
   - Outputs are independent (or concat'd into one final output)
2. Run with `--num_inferences 3 --profiling_level detailed --profiling_option optrace`
3. Decode profile_text — read iter 3 per-instance cyc
4. Average instances 1..N-1 (skip instance 0 = first-of-iter setup)
5. Decode chrometrace.json with --schematic for visual timeline review
6. Compare with native equivalent built same way

The independent design ensures HMX kernel cyc is genuine MAC throughput
under the QNN scheduler — no chain saturation, no data-dep timing
distortion, no folding.

## Files

```
v8c8/                                                   native/
├── v8c8.onnx                                          ├── model.onnx
├── v8c8.dlc                                           ├── model.dlc
├── quant_overrides.json                                ├── quant_overrides.json
├── gen_v8c8_chain.py / run_v8c8_chain.sh              ├── gen_matmul_onnx_chain.py / run_native_chain.sh
├── ctx/
│   ├── v8c8_ctx.bin                                   ├── ctx/
│   ├── v8c8_schematic.bin                             │   ├── matmul_native_ctx.bin
│   ├── v8c8_bottom_mapping.json     (lowered graph)   │   ├── schematic.bin
│   └── ...                                             │   └── model_bottom_mapping.json
├── device_out/                                         ├── device_out/
│   ├── qnn-profiling-data_runtime.log                 │   └── qnn-profiling-data_runtime.log
│   └── ...                                             │
└── decoded/                                            └── decoded/
    ├── profile_text.txt   (3 iters × 8 bbb cyc)          ├── profile_text.txt   (3 iters × 8 MatMul cyc)
    ├── chrometrace.json   (chrome://tracing)             ├── chrometrace.json
    ├── chrometrace_htp.json                              ├── chrometrace_htp.json
    ├── chrometrace_runtrace.json                         ├── chrometrace_runtrace.json
    ├── chrometrace_qnn_htp_analysis_summary.html         ├── chrometrace_qnn_htp_analysis_summary.html
    └── _optrace_config.json                              └── _optrace_config.json
```

## Reproduce

V8C8:
```sh
cd example/hmx_matmul_phase3
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_HMX" bash build.sh
EXTRA_DEFS="-DV9_C8_ALIGNMENT_TEST -DV9_KERNEL_HMX" bash build_x86.sh
cd standard_flow/phaseB_v8
M=256 K=256 N=256 CHAIN=8 MODE=independent OUT_DIR=phase1_validation/v8c8_indep8_256 \
  bash run_v8c8_chain.sh
```

Native:
```sh
cd example/hmx_matmul_phase3/standard_flow/phaseA_native
SIZE=256 CHAIN=8 MODE=independent SHARED_W=0 bash run_native_chain.sh
```

Build state at capture: `git rev-parse HEAD` = `68f81ce` (post Step 2).
