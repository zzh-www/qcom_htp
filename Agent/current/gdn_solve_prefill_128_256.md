# GDN triangular inverse `T=(I−A)⁻¹` at prefill chunk size C=128/256

**Question.** Decode is fine at C=32/64. Prefill wants a LARGER chunk (C≥128/256): fewer chunks →
fewer inter-chunk serial recurrence steps + bigger HMX-shaped matmuls for the rest of GDN. The one
GDN stage that gets *worse* with larger C is the per-head C×C triangular inverse (the `solve_tril`
stage). Goal: make that inverse optimal at C=128/256 on the current v75 HTP.

**Status (2026-06-02).** Analysis + de-risking COMPLETE; ready to build #2.
- **Shipped:** C=128/256 enabled + bit-accurate on the GdnSolve HVX op (was capped at C≤64), int16-packed
  (1.6×), boundary `q::Cast` removed (uint16 override). Current steady cost C=256 = **70,201 cyc/head**.
- **Decided:** the HMX-offload-via-QNN-graph route loses (Crouton glue). The win path is a hand-written
  **Crouton-resident block-recursive HMX merge** + HVX diagonal solves, **HVX∥HMX pipelined**.
- **Validated:** pipelined ceiling **~8–10×** (C=256 → ~7–8.7K cyc/head); HVX∥HMX overlap measured at
  **98%** on device; int16 merge accuracy + overflow OK; all inputs measured.
- **Next:** build the #2 PoC — see "Implementation plan for #2" near the bottom. **This doc is the
  start-here for that work.**

## What shipped

`example/gdn_native/solve_op/src/` — the GdnSolve op now handles C>64, **int16-packed**:
- `gdn_solve_core.h`: `GDN_CMAX 64 → 256`.
- `GdnSolveOp.cpp`: added a C>64 **column-tiled, int16-packed** forward-substitution path (C≤64 path
  unchanged, so decode is not touched). Two ideas:
  1. **Column-tiling.** T's columns are independent in forward subst (each col j solves
     `(I−A)·T[:,j]=e_j`), so solve 64-column tiles. Tiling also **exploits triangularity** (tile
     `[c0,c0+64)` skips rows `<c0`) → ~2.1× fewer MACs than a naive full-width pass.
  2. **int16 packing (1.6× on top).** T codes fit int16 (|code|<2¹⁵), so `Q6_Ww_vmpyacc_WwVhRh` does
     **64 columns/instr** (vs 32 for the int32 `vmpyiacc`). The acc is an int32 even/odd PAIR;
     `Q6_Vh_vasr_VwVwR_rnd_sat(hi,lo,F)` narrows **and** re-interleaves even/odd → natural order in
     one op (no separate shuffle). Accuracy is bit-identical to the int32 path.
- Guard: C≤64 needs C%16==0 (decode 16/32/64); C>64 needs C%64==0 (prefill 128/256, aligned tiles).

The int32 accumulator does NOT overflow at C=256 (the worry was accumulation over up to 256 terms):
measured T relerr clean (C=256 4.77e-5), identical int32 vs int16-packed.

**Metric = steady-state compute CYCLES** (not µs, not cold-start, not power-on). The op runs on 4 HVX
threads, 8 heads/tile; take the warm tile, exclude the one cold-start tile (~1.57M at C=256) and ignore
the runtrace `Resource Power On`/`VTCM Acquire` phases. Full C=32/128/256 table below.

**Boundary `q::Cast` removed (op-I/O dtype fix).** The HTP backend stages 16-bit quantized acts as
**uint16-midpoint** in TCM, which is why the GdnSolve op signature is `QHPI_QUInt16`. Declaring the
graph A/T as **`int16` (SFX16)** in the override forced a signed→unsigned `q::Cast` (zp 0 → 32768) at
every op boundary (5.1% of C=32 graph work-volume). Fix: declare the override as **`uint16` (zp 32768)**
so the graph tensor already matches the op — the Cast vanishes, accuracy identical (relerr unchanged
4.50e-5 / 4.77e-5). NOTE: changing the op signature to `QHPI_QInt16` does NOT work — the backend
re-casts to QUInt16-in-TCM regardless (the staging form is fixed). `scripts/gdn_shape_probe.py` uses
uint16; apply the same to the production override (`scripts/gdn_v2_override.py`) for the GdnSolve I/O.

**Measured speedup of int16 packing (per-head compute, Δwall/Δhead):**

| C | int32 col-tiled | int16-packed | speedup |
|---|---|---|---|
| 128 | ~32 µs/head | **19.9 µs/head** | 1.6× |
| 256 | 170 µs/head  | **104 µs/head** | 1.63× |

(Whole-op wall C=256 H=32: 7088 → 4832 µs.) Not the full 2× — the requant + uint16 output + per-tile
zeroing dilute the 2× MAC win; ILP across rows could recover some.

## Measured cost — steady-state compute CYCLES (the canonical metric)

Standalone `solve_op/standalone/gdn_shape.sh`, A→GdnSolve→T, H=32, v75, 4 HVX threads, 8 heads/tile,
int16-packed + uint16 override (no boundary Cast). Steady tile = warm GdnSolve tile cycles (cold-start
tile excluded). Reproduce: `CS="32 128 256" ./gdn_shape.sh`, then take the median per-tile `dur` from
`out_s/optrace/chrometrace.json` (= cycles; `scripts/perfetto_qnn_optrace.py` ÷1000's trace_processor's value).

| C | T relerr | steady tile (8h) | **cyc/head** | cyc/elem (cyc/head ÷ C²) | Cast |
|---|---|---|---|---|---|
| 32  (decode)  | 4.50e-5 |   7,672 |    959 | 0.94 | 0 |
| 128 (prefill) | 4.13e-5 |  91,400 | 11,425 | 0.70 | 0 |
| 256 (prefill) | 4.77e-5 | 561,613 | 70,201 | 1.07 | 0 |

**Kernel scales ~O(C²), NOT O(C³).** cyc/elem is flat ~0.7–1.07 → cyc/head ∝ C^2.06 (8× C → 73×, not
the 512× of O(C³)). The int16-packing (64 cols/instr) + triangular column-tiling collapse the dense
triangular inverse to near-quadratic. (The earlier "C^2.4" was a wall-µs artifact — glue + fixed
overhead; steady cycles are the truth.) At C=32 the op is fold/output/setup-bound (959 cyc/head ≫ its
~512 MAC instrs), so the inverse compute is negligible for decode.

**Total prefill solve ∝ ~C (linear).** For a fixed sequence (n_chunks = L_seq/C, H=32, L_seq=2048):
cyc/head ∝ C² and n_chunks ∝ 1/C → total ∝ C^1.06.

| chunk C | chunks | total solve cycles | vs C=32 |
|---|---|---|---|
| 32  | 64 |  1.96M | 1.0× |
| 128 | 16 |  5.85M | 3.0× |
| 256 |  8 | 17.97M | 9.2× |

This is the crux: large prefill chunks help every GDN stage *except* the solve — the *total* inverse
gets ~linearly more expensive (C=256 ≈ 9× the C=32-equivalent). Making this 70K cyc/head O(C²) HVX
compute cheaper (→ HMX) is what #2 targets.

## Why HMX-offload via the QNN graph does NOT win (measured)

A dense `[1,B,C,C]@[1,B,C,C]` u8×i8 matmul DOES map to HMX, but each matmul **node** carries heavy
per-head glue that does **not** amortize at batch (the glue ops — `convert_weights_to_signed`,
`ForceFormat_Crouton`, `Slice_contig`, `SyncOp` — scale with head count):

| 64³ matmul node | wall compute | per-head |
|---|---|---|
| B=32  | (291k aggregate cyc) | 9.1k cyc/head |
| B=256 | **1373 µs** | **5.36 µs/head** |

A 64³ HMX matmul costs ~**5.4 µs/head** — *the same as a 64×64 HVX solve* (6.1 µs). The Crouton
format conversion (the HMX tiling tax) eats HMX's raw throughput because 64³ is too small to amortize
it. Reproduce: `CS=64 H=256 ./gdn_mm.sh`.

Consequence for the two decomposed-inverse routes (both build T from dense matmuls so they *could*
use HMX):
- **Block-recursive inverse** (4 diagonal-64 solves + 6 merge matmuls = 4×64³ + 2×128³, ~6–10 graph
  nodes): est. ~19 ms compute at C=256/B=256 vs ~40 ms for the HVX op — only **~2×**, plus int16
  requant points, plus int32-overflow care on the 128³ matmuls.
- **Squaring / Neumann** (log₂C full C×C matmuls): more compute, fewer nodes; full-C int16×int16
  overflows int32 at C≥128 (the C=64 squaring accumulator peak is already 2.08e9, just under 2³¹).

**Accuracy of the matmul form is NOT the blocker** (it was, with int8): host probe
`scripts/gdn_solve_squaring_probe.py` shows **int16-requant between dense matmuls is bit-equivalent to
forward-subst downstream** — squaring T relerr 1.1e-5, U/W identical to fwd-subst (6.6e-3 / 6.1e-3).
So the doc's old "graph-matmul can't beat the chain" verdict was an *int8* result; int16 dense matmuls
are accuracy-viable. And **block-recursive with K=64 matmuls** keeps the accumulator ~2e9 < 2³¹
(unlike full-C squaring) — so it is both accurate AND overflow-safe in int16. The blocker is purely
the **per-node Crouton glue**, not math.

## The optimum, and the frontier

**Today's pragmatic optimum = the int16-packed HVX custom op, scaled to C=128/256** (shipped). One
node (no dispatch), accurate (int16 internal, controlled requant), triangular-tiled, 1.6× from int16
packing. It is within ~2× of the gluey HMX-graph route and far simpler/safer. The forward-subst is now
near its HVX-efficient limit (triangularity exploited + 64 cols/instr); remaining HVX headroom is
small (ILP across rows to push 1.6×→~2×).

**The only path to a large (≫2×) win = a hand-written, Crouton-RESIDENT HMX merge** that does the
block-recursive inverse in ONE kernel, keeping the 64×64 blocks in HMX/Crouton layout across all merge
matmuls so the format conversion is paid once, not per node — the project's descriptor-driven
HMX approach (`reference_hmx_dsp_vs_descriptor`, the handwritten-HMX route), with HVX diagonal solves.
**Caveat (real, raises the risk):** HMX is natively 8-bit (u8×i8). Use a **fully-owned** kernel for the
merge — **u8i8** (native, cheapest) or **w8a16** (one int16 operand, 2× the 8-bit matmuls) — NOT w16a16
(~4× cost *and* still an unproven `.word` replica, see "#2 theoretical ceiling"). Validate a standalone
hand-written u8i8/w8a16 64³ HMX matmul vs HVX (accuracy + cyc) BEFORE committing to the full
Crouton-resident inverse. Major build, not yet attempted.

## #2 theoretical ceiling — estimated from measured data (C=256, per head, steady cycles)

Block-recursive inverse: nb diagonal BL×BL blocks inverted on HVX (forward subst) + the off-diagonal
merge as HMX matmuls on a hand-written Crouton-resident kernel (glue paid once).

**Merge precision = w8a16 or u8i8, NOT w16a16** (decided 2026-06-02). The deciding reason is *kernel
ownership*, not the cost multiplier: only **w8a16** (`kernels/w8a16/`, 0 raw words) and **u8i8**
(`kernels/u8i8/`, 0 raw words) are fully byte-verified hand-written inline asm we can freely reshape for
the Crouton-resident merge. The **w16a16** kernel (`kernels/w16a16/`) is still a *hybrid `.word` replica*
of the vendor `libQnnHtpV75Skel.so` slice — **54 raw `.word` packets not yet byte-proven** — so it is NOT
truly owned; hand-modifying it would first require finishing that byte-proving (a hidden sub-project).
Cost is a bonus: native HMX is u8×i8, so **u8i8 = 1×, w8a16 = 2×, w16a16 = ~4×** the 8-bit matmuls
(int16 decomposition, `reference_hmx_arch`). Strategy: **try u8i8 first** (cheapest + most mature = the
v73deep breakthrough); if both-operands-int8 exceeds the merge accuracy ceiling, escalate to **w8a16**
(int16 on the range-sensitive/accumulating operand, int8 on the other). Gate the choice on
`scripts/gdn_merge_precision_probe.py` (w16a16/w8a16/w8a8≡u8i8 host compare — needs C=128/256 golden;
the C=64 p00 golden is degenerate, only ~13 real tokens) + end-to-end oc on device.

Measured inputs: 64×64 HVX block inverse = **3,150 cyc/head**, 32×32 = **959** (`gdn_shape.sh` steady);
glue-free HMX 256³ = **9,429 cyc** at w16a16 (handwritten oracle `example/handwritten_hmx_matmul/oracles.json`,
chain-8 ÷8) → **5.6e-4 cyc/MAC** — w8a16 ≈½ that, u8i8 ≈¼ (so the merge gets *cheaper* than this estimate,
making the pipeline even more solidly HVX-bound → the win comes from shrinking BL, see below); matmuls
confirmed HMX-mapped (HMXutil 41–45%, `MM_DTYPE=w16a16 ./gdn_mm.sh`); merge K=64 int16 stays <2³¹
(`gdn_solve_squaring_probe.py` peak 2.08e9).

**HVX (4 threads) and HMX are INDEPENDENT units → pipeline them.** Decouple the data dependency across
the batch (prefill has many independent head-solves): while HMX merges head h's blocks, HVX inverts
head h+1's diagonals + requants head h-1. Steady-state throughput = **max(HVX_load, HMX_load)**, NOT
their sum. Both per-head loads below already include the 4-HVX-thread split (the measured numbers are
per-head throughput).

| BL | HVX load (diag+requant) | HMX load (merge) | sequential (sum) | **pipelined (max)** | speedup |
|---|---|---|---|---|---|
| 64 | 12,600 + 1,000 = 13,600 | ~4,000 | 17,600 | **13,600** (HVX-bound) | 5.2× |
| 32 | 7,672 + 1,000 = 8,672  | ~5,500 | 14,172 | **8,672** (HVX-bound) | 8.1× |
| **24** | ~5,800            | ~7,000 | 12,800 | **~7,000** (balanced) | **~10×** |
| 16 | 4,096 + 1,500 = 5,596  | ~9,000 | 14,596 | **9,000** (HMX-bound) | 7.8× |

**Pipelined ceiling ≈ ~7,000–8,700 cyc/head (~8–10×)** — roughly double the sequential ~4×. The lever:
diagonal HVX load ∝ C·BL (shrinks with BL), merge HMX load grows as BL shrinks (smaller, less-efficient
matmuls) → **balance at BL≈24–32 where HVX_load≈HMX_load** minimizes max(). Below that the bottleneck
is the irreducible diagonal forward-subst (HMX can't do a triangular solve).

**This is contingent on a cheap HVX↔HMX handoff** — the diagonal blocks must reach HMX (Crouton layout)
and the merge results reach HVX (requant) **without a DDR round-trip**. That is the entire point of a
hand-written **Crouton-resident** kernel: double-buffer the 64×64 blocks in VTCM so both units read/write
in place. If the handoff serializes or spills to DDR, the overlap collapses toward the sequential ~4×.
Validate the overlap before building (see "Overlap validation" below).

## Overlap validation — HVX ∥ HMX confirmed (98%)

Built a combined graph `A[1,B,C,C] → GdnSolve(HVX) → T → MatMul(T, V)(HMX) → P` (C=64, B=64, sized to
fit VTCM) and read the optrace timeline (`example/gdn_native/solve_op/standalone/gdn_overlap.sh`,
`scripts/gdn_overlap_probe.py`):

- **GdnSolve runs on tids 512–515 (4 HVX threads); the MatMul `q::ConvLayer_s1` runs on tid 256 (HMX)**
  — distinct units. Their timelines **overlap 98%**: the HMX matmul (span 461,178 cyc) starts mid-solve
  and runs concurrently with the HVX forward-subst (span 769,775 cyc).
- Wall (`Accelerator (execute) time`): solve-only **1329 µs**, combined **1360 µs** — the 282,264-cyc
  HMX matmul (~157 µs of HMX work) added only **+31 µs** to the wall (~80–98% hidden).

This confirms the pipeline premise: **HVX and HMX genuinely run concurrently**, so #2's throughput is
bounded by max(HVX, HMX), not the sum — the ~8–10× pipelined estimate is achievable.

**Handoff lesson (reinforces Crouton-resident):** the matmul-only reference, fed a *fresh uint16 T from
a DDR graph input*, hit a pathological path (2.99 **billion** cyc). The SAME matmul consuming the op's
output **in place** (combined graph) cost 282K cyc — normal. So the HVX→HMX handoff must keep T on-chip
(VTCM, Crouton layout) and feed HMX directly; a DDR round-trip of the intermediate is catastrophic.

## Implementation plan for #2 (START HERE next session)

**Goal:** a hand-written custom op that computes `T=(I−A)⁻¹` for C=128/256 via block-recursive inversion,
HVX diagonal solves ∥ Crouton-resident HMX merge, target **~7–8.7K cyc/head at C=256 (~8–10×)**.

**Design (fixed by the analysis above):**
- Block size **BL=32** (the HVX≈HMX balance point; revisit 24 if HVX-bound).
- Diagonal: invert the nb=C/BL diagonal BL×BL blocks on HVX — reuse the existing int16-packed forward-subst
  (`GdnSolveOp.cpp`). 32×32 = 959 cyc/head measured.
- Merge: off-diagonal blocks `T_ij = T_ii @ (Σ_{k=j}^{i-1} A_ik T_kj)` (i>j) as HMX matmuls, K=BL≤64
  (stays <2³¹). Reuse a **fully-owned** handwritten kernel — **u8i8** first (cheapest, native 8-bit;
  `kernels/u8i8/v73deep_conv1x1_kernel.inc`), escalate to **w8a16** if accuracy needs int16 on one
  operand (`kernels/w8a16/v73deep_conv1x1_kernel.inc`). **Do NOT use w16a16** — that kernel is still a
  hybrid `.word` replica (54 unproven raw packets), not byte-verified, so not safe to hand-reshape.
  Recipe in `hmx-inline-asm` skill + `project_v73deep_BREAKTHROUGH_2026-04-28`.
- **Crouton-resident VTCM:** keep the BL×BL blocks in VTCM in HMX/Crouton layout; HVX writes
  diag/requant results there, HMX reads in place. **Never DDR round-trip the intermediate** (measured:
  DDR round-trip of T = 2.99 billion cyc vs 282K in-place).
- **Software-pipeline across heads:** HMX merges head h while HVX inverts head h+1 + requants head h-1.
- I/O: uint16-midpoint (QInt16 sig does NOT work — backend forces QUInt16-in-TCM); override `uint16`.

**PoC milestones (each device-validated, metric = steady compute cyc):**
1. Standalone **u8i8 (then w8a16 if needed) HMX 64×64 (and 32×32) matmul**, glue-free, from the
   fully-owned handwritten kernel — confirm cyc/MAC (≤5.6e-4; cheaper than the w16a16 oracle) and that
   it reads/writes VTCM-resident operands. (De-risks the merge primitive.)
2. **C=128 inverse** (nb=4 @ BL=32, or nb=2 @ BL=64): diagonal HVX + merge HMX, single head, no pipeline
   — validate accuracy (relerr vs `np.linalg.inv`, target ~4e-5) + cyc.
3. Add the **HVX∥HMX pipeline** across heads (B>1); confirm wall ≈ max(HVX,HMX) (the 98%-overlap result).
4. Scale to **C=256**; tune BL; hit ~7–9K cyc/head.
5. Integrate: replace GdnSolve for prefill C in the real GDN graph (`scripts/gdn_insert_solve_op.py`),
   re-check oc end-to-end.

**Risks/unknowns to watch:** (a) ~~Crouton layout management for non-256³ block shapes~~ **LARGELY
RESOLVED (2026-06-02):** the owned **u8i8 kernel is fully descriptor-driven — NO hardcoded shape
immediates.** Every loop trip is loaded from the out/act descriptors at runtime: `r28=n_tiles_pow2>>1`
(K-MAC outer), `r13=ceil(K/32)`, `r20=ceil(M_t/8)`, `r12=M_t` (kernel P5–P11); `loop0/loop1` use those
registers. "256³-canonical" = where it was byte-verified, not a hardcoded limit. A 64×64×64 matmul
(N=64→n_tiles=2→`p0` true, `ep[0]=1`→`p2` true) drives the **same main K-MAC path**; 32³ (n_tiles=1)
takes the alt-A arm (present, less-tested). `prepare_owned_inputs.py::generated_descriptor_tables` already
computes all descriptor fields + offset tables from m/k/n parametrically, and for **u8i8 the mask/RT
control words are shape-INDEPENDENT constants** (`conv1x1_words(0x700,0,0,0,0x20)`, `extra=[1,0]`). So
M1 needs **no new kernel** — just a 64³/32³ artifact. Residual: confirm the alt-A (n_tiles=1) arm for
32³, and verify the activation-surface/output Crouton layout at small shapes (tile contracts: K%32, N%32,
M steps by 8). (b) driving HMX from inside a QHPI op (vs the standalone-kernel harness) — may need manual
qurt/HMX descriptor setup; (c) keeping the pipeline fed (enough independent heads) — fine at prefill scale.

**M1 groundwork done (2026-06-02).** Exec path validated end-to-end WITHOUT device: handwritten kernels
run in **hexagon-sim** via the bare-metal h2 harness (`scripts/check_handwritten_hmx_body_entry_sim.py`
builds `tools/body_entry_smoke.c` with `hexagon-clang -mv75 -mhvx -mhmx -moslib=h2` + booter; all 4
families enter-and-return OK). Shape-driven numeric runs go through `scripts/run_handwritten_artifact_body_sim.py
--family u8i8 --artifact <dir>` — it reads `prepared_state/{activation,packed_weight,folded_bias,
output_surface,mask_control,activation_table,output_table}.raw` + an `abi_manifest` + minimal `oracle`
(for the output compare), emits a C harness, runs in sim, diffs sim output vs `oracle.raw_output`. The
u8i8 requant reference is `scripts/reconstruct_hmx_u8_drain.py`: `drain_in = ΣactW + (−128·Σw + bias_q)`,
`out_u8 = clamp(trunc(drain_in·scale/512) + (baseline>>7),0,255)`.

**M1 DONE (2026-06-02) — u8i8 64³ matmul PROVEN bit-exact in hexagon-sim.** `scripts/gdn_hmx_matmul_sim.py`
(self-contained generator + bare-metal C harness + sim runner + numpy verifier). The owned u8i8 kernel
computes a correct **64×64×64 u8×i8 matmul in ONE call**, VTCM-resident, glue-free: **max_abs_diff=0,
0/4096 mismatches** on real-slice + random full-range u8/i8 (seeds 7,42) + ramp. Reproduce:
`uv run python scripts/gdn_hmx_matmul_sim.py --mode real` (or `--mode random --seed 42`).
- **cyc/MAC: ~4.2e-4 steady-state** (back-to-back amortized ~109 pcyc/call; cold single-call 417 pcyc =
  1.59e-3, dominated by the ~430-cyc fixed prologue/drain on this tiny shape). Steady **beats the w16a16
  oracle's 5.6e-4** → confirms u8i8 ≈¼ cost, the cheapest merge primitive. (sim pcycles, no DDR/cache.)
- **Authoritative 64³ descriptor** (feeds M2): `out_desc={out_table_stride_dwords:2, out_y_stride_words:8,
  n_tiles_pow2:8, m_total_minus_step:8, k_total_bytes:64}`, `act_desc={n_act_pairs:2,
  act_table_y_stride_words:8}`, `extra=[1,0]`, mask=`conv1x1_words(0x700,0,0,0,0x20)`. Single call covers
  all 64 M-rows + both K-tiles.
- **CRITICAL activation-layout finding (load-bearing for M2's VTCM layout).** The `prepare_owned_inputs.py:247`
  crouton8 packer interleaves `kt` INSIDE the row8 loop, which scatters K-tiles → kernel reads WRONG act
  columns for k≥32. **Correct = each K-tile a SEPARATE contiguous crouton8 tile of `m*32` bytes laid
  end-to-end** (kt0@0, kt1@`m*32`), `act_table[kt]=kt*m*32` (offsets [0,2048] for 64³). See
  `gdn_hmx_matmul_sim.py::pack_act_crouton8`.
- **Output crouton8 de-pack (closed form, validated):** `out[r,c]` at byte
  `nt*2048 + r8*512 + m32*256 + rsub*32 + cw*4 + bsub` (`nt=c//32, m32=r//32, r8=(r%32)//8, rsub=r%8,
  cw=(c%32)//4, bsub=c%4`).
- Gotchas: recompute `effective64[n]=−128·Σ_{k<64}W[:64,n]+bias_q[n]` for the K-slice (don't reuse 256-K
  effective); 256³ `out_ref_u8.npy` is chain-8 (not a single-matmul golden); shipped `A.raw` is degenerate
  (identical rows) — random data is the real gate.

**Remaining M1 tail (deferred):** 32³ HMX (alt-A arm) only needed if BL=32; M2 starts at **BL=64 (nb=2,
C=128)** on the validated 64³ path. w8a16 (M1b) deferred to merge-accuracy gating.

## Reproduce

```bash
cd example/gdn_native/solve_op
bash build.sh                                                   # builds C>64-capable op
cd standalone
CS="32 64 128 256" ./gdn_shape.sh                              # STEADY compute cyc (canonical metric) + accuracy
python ../../../../scripts/perfetto_qnn_optrace.py out_s/optrace # wall phases + per-op cyc + mem bw
MM_DTYPE=w16a16 CS="64 128 256" ./gdn_mm.sh                     # int16 HMX matmul (the #2 merge unit)
C=64 B=64 ./gdn_overlap.sh                                      # HVX∥HMX overlap (solve->matmul timeline)
GDN_NO_VSCALE=1 ../../../../.venv/bin/python ../../../../scripts/gdn_solve_squaring_probe.py
                                                               # int16-requant accuracy + overflow (host)
```
