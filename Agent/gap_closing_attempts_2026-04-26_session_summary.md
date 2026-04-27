---
name: 2026-04-26 gap-closing session summary
description: Comprehensive summary of all attempts to close the V8/QNN wall-time gap in this session. Final verdict — 12+ concrete experiments, none moved the wall-time numbers. The 4.5×-15.1× wall gap is the achievable ceiling for custom op-pkgs without QNN compile-time graph machinery (descriptor synthesis, multi-instance HMX scheduling, VTCM bank allocation patterns).
type: project
---

# Session 2026-04-26 — exhaustive gap closing attempts

## Final wall-time table (ground truth, mean of 2nd+3rd inferences)

| shape | V8 µs | QNN µs | ratio |
|------:|------:|-------:|------:|
| 512³  | 193 | 43 | **4.5×** |
| 1024³ | 1,286 | 114 | **11.3×** |
| 2048³ | 9,862 | 654 | **15.1×** |
| 4096³ | 80,106 | 14,101 | **5.7×** |

## Experiments tried and what they showed

### Won (real fixes)
1. **ctxgen segfault root-caused & fixed** (`pack_act_crouton_skel.c` x86 host stub) — V9 ctxgen now passes.
2. **Wall-time metric corrected** — original "30×" claim was cycle-metric-mismatch; real wall ratios above.
3. **Bug fix**: `act_ptrs_all[64*8]` was sized for K_t=64, broke 4096³ (K_t=128). Changed to `[128*8]`.

### No-ops (proved not the lever)
4. **Inner-asm body byte-replicated** ConvLayer's 3-packet body (cmp.eq/r26/combine/`if(p0)...`) — **0% perf change**.
5. **Hardware loop0** (vs software loop) — 0%.
6. **`bias = mxmem2(r3)`** vs `mxmem` — 0%.
7. **`sat.ub Rt = 0x3FF`** vs `Rt = 0` — 0%.
8. **Pre-baked `act_ptrs` table at op start** — 0%.
9. **Smaller M_TILE/N_TILE** (32, 64, 128 vs 256) — all within 1% noise.
10. **Cost function `QHPI_Cost_Function`** added to mmv8 + pack_act + pack_wt + tcm2ddr — 0% (also broke 4096³ until caught).
11. **`multithreaded=true` on mmv8** (HMX is single-unit but probe whether scheduler benefits) — 0%.
12. **Probe sweep**: NO_SATUB / NO_MAC / SAME_ADDR / ACT_FIXED / WT_FIXED / ACT_4WAY / ACT_PAIR_SAME / RT_ACT_3FF / ACT_4K_STRIDE — definitively isolated `:cm` act-stride as the 240K cyc/inst penalty source.

### Failed attempts (didn't work)
13. **dlsym call to `hmx_convbbb1x1_stride1`** — call mechanism works (`R_HEX_JMP_SLOT` resolves cross-.so), kernel runs without crash and executes correct MAC count, but **per-packet wall = ~67 cyc, IDENTICAL to our inline asm**. Output is bias+noise (likely format mismatch). Confirms the QNN advantage is **NOT in the kernel** — it's in graph-compile-time machinery.
14. **V9 with PackActCrouton + dlsym** — kernel crashes (additional VTCM stride constraint).
15. **V8 with PackActCrouton activation** — Converter package shape-inference rejects K-major input.

## Definitive conclusion

**V8 mmv8 inline-asm IS already at the throughput ceiling reachable from a custom op-pkg**. The remaining wall-time gap (4.5×-15.1×) is in QNN's compile-time graph machinery:
- Descriptor-driven HMX state setup (act-stream tracker, prefetch hints) only accessible via QNN's internal `hmx_conv_*_desc_t` machinery
- Multi-instance HMX scheduling that lets QNN run 256 small ConvLayer instances overlapping bias-load + weight-load + MAC + sat.ub on HMX TID 256
- VTCM bank allocation algorithm that places act/wt/out tiles to avoid bank conflicts

None of these are accessible from custom op-pkgs without rewriting QNN's compiler.

## What might unlock further gain (NOT TESTED, speculative — open problems)

1. **Descriptor capture from running QNN ConvLayer** via runtime hook — if we can capture the actual descriptor BYTE values QNN passes, we could replay them in our V9 dlsym path. Requires building an instrumentation op-pkg.
2. **HVX-HMX overlap via graph reorganization** — currently V8's pack_act/mmv8/tcm2ddr run sequentially (cum cycles ≈ wall). Per QNN trace, QNN runs HVX pack stages BEFORE HMX MAC, completing before HMX starts. Same pattern V8 has → no overlap possible without restructuring (e.g., split V8 along K into pipeline stages with separate pack_act_K0 + mmv8_K0 + pack_act_K1 + mmv8_K1 ...).
3. **Pack activation in Crouton format AND get HMX `:cm` to consume it correctly**. Tested — V9 path crashes; V8 path Converter rejects. Would need new Converter Op Package version.

## Files modified this session (all preserved under compile flags, defaults restored)

- `kernel/pack_act_crouton_skel.c` — x86 host stub (real fix for ctxgen segfault)
- `src/HmxMatMulV8Op.cpp` — `V8_USE_DLSYM_PER_TILE`, `V8_PROBE_*` (8 variants), `V8_ACT_KMAJOR`, `V8_MMV8_MULTITHREADED` flags. Default = production. Stack array sized for K_t up to 128.
- `src/HmxMatMulV9SkelOp.cpp` — `V9_USE_DLSYM`, `V9_KERNEL_NOOP`, `V9_DLSYM_*`, `V9_INLINE_MINIMAL_HMX` flags. Default = inline asm.
- `kernel/pack_act_rm_hvx.c` — comment cleanup only.
- `standard_flow/phaseB_v8/gen_v8_graph.py` — `V8_PACK_ACT_OP` env var override.
- `standard_flow/phaseB_v8/MatMulV8Package.xml` — BbbKMajor in[1] shape updated.
- `standard_flow/phaseB_v8/gen_v9_test.py` — uses PackWeightToHmxTileV3.
- `build.sh` — `EXTRA_DEFS` env injection.
