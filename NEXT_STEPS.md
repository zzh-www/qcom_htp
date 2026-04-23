# Next session start — Phase 3B wrap, Phase 3C/D launch (2026-04-23)

> Commits today (most recent first): `9d75a35` → `3ef0434` → `146d8e5` →
> `e4ae8f7` → `5551504` → `861ccd3` → `a8137db`.
> Phase 2 baseline preserved at `05aacfb` (w4a16=0.41/w4a8=0.24 cyc/MAC).
> Phase 3B commit `9d75a35` bit-exact @ 0.23 cyc/MAC @ 512³.

## 0. Quick status

**Phase 3B = Phase 2 parity, in cleaner Phase-3 architecture.**
`example/hmx_matmul_phase3/` has 6 registered ops. Main `MatMulV2` op
bit-exact across 32³/128³/512³, 0.23 cyc/MAC (matches Phase 2's 0.22).

Architectural pieces ready but **not yet wired together**:
- 4 upstream HVX ops (`pack_act_hvx`, `pack_wt_hvx`, `combine_hi_lo_hvx`,
  `int4_expand_hvx`) built and registered via `qhpi_init()`.
  All declare `QHPI_RESOURCE_HVX + multithreaded=true` so QNN scheduler
  can run them on 4 HVX threads.
- `:cm` + row-major activation path silicon-validated at 7.92 cyc/MAC
  (Agent A probe, `Agent/cm_row_major_re.md`) — faster than Phase 2's
  9.03, but random-data bit-exact still pending weight layout probe.

## 1. Sanity commands to restart

```bash
source scripts/env.sh

# Verify Phase 2 baseline still builds + passes (regression check)
bash tests/test_hmx_matmul_w4a16.sh
bash tests/test_hmx_matmul_w4a8.sh

# Verify Phase 3B (MatMulV2) still bit-exact
cd example/hmx_matmul_phase3
bash build.sh
bash run_v2_on_device.sh --shape 512,512,512
# Expected: mismatches=0/262144, cyc/MAC≈0.23
```

## 2. Priority task — wire Agent B's upstream HVX ops (Path W)

This is the core Phase 3 architectural win: split pack+MAC into separate
graph nodes so QNN scheduler parallelizes HVX pack on 4 threads while
HMX MAC runs. **Highest ROI remaining Phase 3 work.**

### Concrete steps

1. **Study Agent B's op signatures** in `Agent/phase3b_path_w_impl.md`.
   Key question: what's the exact output tensor shape of `pack_act_hvx`?
   (Agent B planned `[1, M/32, K/32, 2048]` as packed HMX tiles —
   confirm actual signature in `example/hmx_matmul_phase3/kernel/pack_act_hvx.c`.)

2. **Modify `MatMulV2` op to consume pre-packed tiles**:
   - Change `sig_inputs_v2` from Flat4 raw tensors to the pre-packed
     tensor shape produced by upstream pack ops
   - Strip the `hmx_core_v2_gather_act_tile` / `gather_wt_tile` calls
     from `phase3_hmx_matmul_v2_kernel` — just read block pointers from
     the input tensor
   - Keep the HMX MAC + dual-scale readback + decode loops
   - Rename op `MatMulV2` → `MatMulV3Chained` so Phase 3B stays as
     regression baseline

3. **Update host harness** (`run_matmul_v2.cpp` or new `run_matmul_v3.cpp`):
   - Build graph: `raw_tensor → PackAct → {packed_act_hi, packed_act_lo}`
     → `MatMulV3` → `{partial_hi, partial_lo}` → `Combine` → `out`
   - For w8a8 variant first (no hi/lo split): simpler single chain
     `raw → PackAct → MatMulV3 → out`
   - Test bit-exact at 512³, compare chromatrace to confirm HVX ops
     actually run on separate threads

4. **Profile with chrometrace** (Phase 3E):
   - `qnn-profile-viewer` on the generated trace
   - Confirm: pack ops dispatch on different HVX thread IDs, HMX MAC
     op on HMX resource, overlap visible
   - Measure cyc/MAC @ 512³ — expect < Phase 2's 0.22 if parallelism works

### If Path W wiring fails

Fallback: add Phase 2's remaining optimizations into current V2 kernel
(in-place in single op, no graph split):
- T1b HVX `pack_weight_32x32` — direct port from
  `example/hmx_matmul_qnn/kernel/hmx_int4_matmul.c` lines 141-175
- T2c HVX `prepack_activation_fused` — port from same file lines 235-270
- T2a HVX combine loop — from `hmx_int4_matmul_mn_dualacc` end

Expected gain: 0.23 → ~0.17 cyc/MAC (not dramatic, but cleaner kernel).

## 3. Secondary tasks (pick up if time permits)

### 3a. `:cm` row-major layout final probe

Status: constant bit-exact, random fails. Silicon mechanism confirmed
faster than 2-stream. Last step is non-uniform weight probe.

Files ready: `Agent/phase3b_cm_readback_layout.md` has probe design.

Concrete probe to add to `example/hmx_matmul_device/`:
- Test 1: weight[0,0]=1, all else=0; activation all-1. Expected
  output[m, 0]=1 at m=0..31 if row ordering is right.
- Test 2: weight[k,0]=k+1 for k=0..31, others=0. Expected output
  per m = sum_{k=0..31} a[m,k] * (k+1).
- Compare actual HMX output to infer weight tile K-ordering.

If resolved: switch V2/V3 kernel to `:cm` path, gain ~15% over 2-stream.

### 3b. Phase 3D — w4a16 integration

Currently Phase 3B handles int8×int8 only. For w4a16:
- Activation is int16 (uint16 post-Cast). Need hi/lo byte split.
- Agent B's `pack_act_hvx.c` emits 2 output tensors (hi + lo).
- Matmul needs 2 passes or dual-acc (like Phase 2).
- `combine_hi_lo_hvx.c` applies `(P_hi<<8) + P_lo - 128·col_sum`.

Wire this as second Phase 3 graph variant once w8a8 path W works.

### 3c. Phase 3F — w4a8 end-to-end

Int4→int8 expansion via Agent B's `int4_expand_hvx.c` as upstream op.
Then w8a8-like chain downstream.

## 4. Key files (paths from repo root)

```
Agent/cm_row_major_re.md              -- :cm silicon RE (conclusive)
Agent/phase3a_crouton_probe_results.md -- Crouton block RE
Agent/phase3b_path_w_impl.md          -- Agent B's HVX ops design
Agent/phase3b_cm_readback_layout.md   -- :cm readback debug trail
Agent/phase3b_readback_status.md      -- debugging progression

example/hmx_matmul_phase3/
  build.sh                            -- builds all ops + 2 host binaries
  run_v2_on_device.sh                 -- push + run on oneplus (ssh)
  src/HmxMatMulPhase3Interface.cpp    -- OpPackage registration (6 ops)
  src/HmxMatMulPhase3Op.cpp           -- Phase 3A probe op (diagnostics only)
  src/HmxMatMulV2Op.cpp                -- MatMulV2 main (Phase 2 2-stream path, BIT-EXACT)
  src/run_matmul_v2.cpp                -- host harness for V2
  src/run_phase3_probe.cpp             -- Phase 3A Crouton probe host
  kernel/hmx_core_v2.{h,c}             -- HMX MAC kernel (used by V2)
  kernel/hmx_core.{h,c}                -- LEGACY (early 2-stream kernel; unused)
  kernel/pack_act_hvx.c                -- Agent B upstream HVX: activation pack
  kernel/pack_wt_hvx.c                 -- Agent B: weight pack
  kernel/combine_hi_lo_hvx.c           -- Agent B: int32 combine
  kernel/int4_expand_hvx.c             -- Agent B: int4→int8 expand
  test_core_sim.c                      -- sim test (written, not wired to build)
  test_ops_sim.c                       -- Agent B's sim harness

example/hmx_matmul_device/probe_cm_row_major.c -- Agent A :cm silicon probe
```

## 5. Numbers snapshot (state at commit 9d75a35)

```
Phase 2 (commit 05aacfb, baseline):
  w4a16: 0.41 cyc/MAC @ 512³  (from Phase 2 final optim stack)
  w4a8:  0.24 cyc/MAC @ 512³

Phase 3B (commit 9d75a35, current MatMulV2 = int8×int8):
  32³:   3.50 cyc/MAC, 0/1024 mismatches
  128³:  0.87 cyc/MAC, 0/16384 mismatches
  512³:  0.23 cyc/MAC, 0/262144 mismatches

QNN built-in baselines:
  w8a8 @ 512³:  4.96e-4 cyc/MAC (closest comparator for MatMulV2's int8×int8)
  w8a16:        8.09e-4 cyc/MAC
```

Gap MatMulV2 → QNN w8a8: 0.23 / 4.96e-4 = **464×**. Same as Phase 2 at 8.7×
baseline speedup. Phase 3B architecture clean but perf ceiling still there
until Path W graph parallelism active.

## 6. What blocks "3× within QNN"?

- `:cm` + row-major unlocks 1.1 cyc/MAC on silicon probe (Agent A). Port into
  kernel once weight layout RE'd.
- Path W graph parallelism: up to 4× from HVX-thread concurrency. Wiring pending.
- `weight.n` native int4 HMX: Qualcomm ISA docs not public. Hard ceiling.

Combined if all land: 464× → ~30-50× of QNN w8a8. That's "within QNN" range.

## 7. Open questions

- Q1: Agent B's pack ops output Crouton-like packed tiles. Does QNN allow
  custom-op-output → custom-op-input chaining at arbitrary shapes, or is
  there a Flat4/Crouton enforcement?
- Q2: Does QNN scheduler respect `multithreaded=true` on HVX ops when
  their output feeds into an HMX op downstream?
- Q3: Is dual-scale readback layout for `:cm` actually Phase 2's stride-2,
  or something else? (Non-uniform probe needed.)

These are empirical; the probes are either written or sketched in
referenced docs. Just run them.

## 8. Explicit "don't do" list

- Don't re-invent Phase 2 optimization ordering. T0/T1b/T1d/Path B/T2a/T2c
  have documented perf impact in `Agent/int4_matmul_optimization_log.md`.
  Port verbatim if needed, don't re-derive.
- Don't touch `example/hmx_matmul_qnn/` or `example/hmx_matmul_w4a8/`.
  Those are Phase 2 baseline at commit `05aacfb` — keep frozen.
- Don't delete Agent B's four HVX ops without wiring attempt first.
  They took significant effort to write; they're the Path W foundation.

## 9. Session success criteria (what to ship)

**Minimum**: bit-exact `MatMulV2` with Path W graph, whether or not perf wins.
This validates the architectural model.

**Target**: Path W-enabled graph < Phase 2's 0.22 cyc/MAC. Proves 4-way
HVX parallelism works as claimed.

**Stretch**: `:cm` path wired + bit-exact. Combined with Path W, aim for
<0.1 cyc/MAC — first order of magnitude toward QNN w8a8.

---

**Today's commit hash chain, for git bisect if issues arise:**
`a8137db` (plan) → `861ccd3` (3A Crouton) → `5551504` (3B Path X/W init)
→ `e4ae8f7` (:cm RE) → `146d8e5` (stride-2 iteration) → `3ef0434` (3B main)
→ `9d75a35` (3B @ Phase 2 parity) [CURRENT].
