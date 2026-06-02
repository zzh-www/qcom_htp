# GDN triangular inverse `T=(I−A)⁻¹` at prefill chunk size C=128/256

**Question.** Decode is fine at C=32/64. Prefill wants a LARGER chunk (C≥128/256): fewer chunks →
fewer inter-chunk serial recurrence steps + bigger HMX-shaped matmuls for the rest of GDN. The one
GDN stage that gets *worse* with larger C is the per-head C×C triangular inverse (the `solve_tril`
stage). Goal: make that inverse optimal at C=128/256 on the current v75 HTP.

**Status (2026-06-02). RESOLVED — the block-recursive HMX route does NOT beat the shipped HVX op at any C,
in EITHER form: the fused one-op (M1→M5) NOR the co-designed two-op SPLIT (M6). KEEP the shipped
int16-packed HVX `GdnSolve` op for prefill.** See "M4 DONE" and "M6 DONE" below.
- **M6 (two-op split, 2026-06-02):** built it.  `multithreaded=true` on a pure-HVX Op1 DID self-slice 4 HVX
  threads (diag 18.6K/head, ~4.8× recovery) and the opaque-VTCM int8-tile handoff DID avoid the ForceFormat
  tax (no adapter op needed).  But the two graph nodes run **0% overlapped** (serial T1→Op2 edge, no
  cross-head pipeline) and Op2 (HMX, can't self-slice) is single-thread 524K/head → total **740K/head,
  ~10.5× slower** than 70,201.  Confirms M5's blocker from the split side: the per-merge HVX glue is the
  irreducible cost; splitting relocates but doesn't shrink it, and adds a serial op boundary.
- **Shipped (the winner):** C=128/256 enabled + bit-accurate on the GdnSolve HVX op (was capped at C≤64),
  int16-packed (1.6×), boundary `q::Cast` removed (uint16 override). Steady cost C=256 = **70,201 cyc/head**.
- **`GdnSolveBR` (block-recursive HMX, M1→M5):** device-correct at C=128 (relerr 1.29e-2) and C=256
  (relerr 2.38e-2). Definitive measured floor (M5, steady op_dur/head): **441,570 cyc/head single-thread
  (tuned int16-packed diag), ~170K best-threaded** (2.5× HVX ceiling, HMX-on-main; worker-HMX faults). LOSES
  by **2.4×**. The binding constraint is the single-thread HVX *work volume* of the 16-merge decomposition
  (~427K HVX glue: quant/pack/depack/fold/requant) — it out-weighs the one O(C²) HVX solve the baseline does
  (70,201, already threaded), and HMX (free, 14K) can't absorb it (can't do a triangular solve, can't run on a
  worker). The earlier 8–10× / 2–4× projections were too optimistic. The HMX kernel IS free; the HVX merge
  glue is the irreducible killer. **See "M5 DONE" below for the full per-lever breakdown.**
- **Conclusion:** the win path imagined here does not materialize on v75. The shipped HVX op is the optimum
  for prefill C=128/256. `GdnSolveBR` stands as a validated artifact + a documented negative result.

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

**M2a accuracy result (2026-06-02) — measured on REAL C=128 data, merge non-degenerate.**
`scripts/gdn_blockrec_c128_probe.py` builds A at CHUNK=128 from real ≥128-token golden (192 such files;
||T21||/||T||≈0.10, merge matters) and runs the 2-block (BL=64) recursion with int16 diagonals + quantized
merge. Across 5 prompts (135–226 tok):
- **T relerr:** w16a16 7e-5 (=fwd-subst), w8a16 3e-3, u8i8 7e-3.
- **Downstream (T consumed at int8, the real metric):** **w8a16 ≈ or slightly better than the int16
  fwd-subst baseline** (W ~1.97–2.10e-2 vs fwd 2.05–2.12e-2); **u8i8 ~6–13% worse** (W ~2.20–2.39e-2).
  The 100× worse u8i8 T-relerr does NOT propagate — the int8 quant of T at consumption dominates.
- **Refined decision:** both are viable. Since at BL≈32 the pipeline is **HVX-bound** (merge HMX cost
  hidden, [[feedback_perf_wall_not_aggregate_cycles]]), **w8a16 is the safer DEFAULT** — int16-baseline
  accuracy at no wall cost when HVX-bound. u8i8 (1×) only wins when HMX-bound at very small BL. Build the
  merge parameterized for BOTH; let M5 real-oc pick. (Supersedes the earlier "u8i8 first" cost-only call.)

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
   — validate accuracy (relerr vs `np.linalg.inv`, target ~4e-5) + cyc. **[M2b DONE 2026-06-02 — see below.]**
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
M steps by 8). (b) ~~driving HMX from inside a QHPI op~~ **RESOLVED (2026-06-02): trivially possible + production
precedent.** A QHPI op drives HMX by declaring `QHPI_RESOURCE_HMX` (`tools/qnn-sdk/include/QNN/HTP/core/qhpi.h:871`,
bitmask in `QHPI_Kernel_v1::resources`); the QNN backend acquires/locks HMX at graph load (HAP), so the op
callback just runs HMX inline asm — NO in-op `h2_mxaccess_acquire`/`HAP_compute_res_hmx_lock` needed.
**Production ops in THIS repo already drive the same u8i8 v73deep kernel on real v75 from a QHPI op:**
`example/qnn_hmx_matmul_u8i8/src/HmxU8I8ToU8MatMulOp.cpp` (`.resources=QHPI_RESOURCE_HMX`), also w4a8/w8a16/w4a16
siblings, and a MIXED HVX+HMX op `docs/hexagon-tutorial/qnn-tutorial/ch03-qnn-custom-op/src/dsp/HvxHmxOp.cpp`.
So `GdnSolveBR` = GdnSolveOp's HVX/QHPI structure + that op's HMX-drive pattern + the M2b merge choreography,
declaring `QHPI_RESOURCE_HVX|QHPI_RESOURCE_HMX`. (c) keeping the pipeline fed (enough independent heads) —
fine at prefill scale.

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

**M2b DONE (2026-06-02) — C=128 block-recursive merge chain PROVEN in hexagon-sim.**
`scripts/gdn_blockrec_sim.py` (extends `gdn_hmx_matmul_sim.py`). Two chained signed-operand 64³ u8i8 HMX
merges + requant + assembly produce `T=(I−A)⁻¹` whose relerr vs `np.linalg.inv` tracks the host u8i8
ceiling on real C=128 data (golden p15_L00, 32 heads): **sim 8.33e-3 vs host-u8i8 6.19e-3 vs host-BR
7.17e-3**, requant **≤1 ULP**. HVX diagonal solve is fed from host `solve_int16` (already proven by the
GdnSolve op); M2b isolates+validates the NEW thing = the HMX merge chain. Reproduce:
`GDN_NO_VSCALE=1 uv run python scripts/gdn_blockrec_sim.py --verify-control` (drain encoding, 0/4096) and
`--heads 3` (chain).
- **LOAD-BEARING for M3/M4/M5 — the signed-merge HMX choreography:**
  - HMX conv1x1 drain: `out_u8 = clip( FLOOR(P_int·scale_f16/512) + (baseline_u16>>7), 0, 255)` where
    `P_int = Σ(act_u8−128)·wt_i8` is the signed int matmul (effective bias `−128·Σwt` cancels the act zp).
  - **The drain rounds with FLOOR (toward −∞), NOT trunc.** (`reconstruct_hmx_u8_drain.py`'s `np.trunc` is
    latent-wrong — only ever used at gain=1.0 where there's no fraction; harmless there, do not rely on it
    for fractional gains.)
  - **Control word per N32 tile = `(baseline_u16<<16) | f16_bits(scale_f16)`.** M1's `0x6000`=`f16(512)`
    ⇒ gain 1.0, baseline 0. Set `baseline_u16 = 128<<7 = 16384` to inject the +128 output zero-point so a
    SIGNED merge result fits u8 (recover `int8 = out_u8 − 128`).
  - Chaining: merge1 `M=A21@T11` (act=A21 zp128 / wt=T11 i8) → output `M_int8 = out_u8−128` at scale
    `s_M=max|M|/127`, `scale_f16=(s_A21·s_T11/s_M)·512`. **Re-pack M_int8 as the i8 WEIGHT** of merge2
    `T21=T22@M` (act=T22 zp128). Dequant `T21=(out_u8−128)·s_T21`, assemble `T=[[T11,0],[T21,T22]]`.
  - Residual: ≤1-ULP sim↔host requant diff on ~2% of merge-output elements (HMX f16 drain rounds the last
    bit slightly differently from the host floor model); sub-LSB, does NOT affect assembled-T relerr.
- cyc: 2× 64³ merges = 834 sim pcyc/head single-call (prologue-dominated); steady ~4.2e-4 cyc/MAC (M1).

**M_op DONE (2026-06-02) — `GdnSolveBR` QHPI op runs the C=128 block-recursive inverse on REAL v75 device.**
`example/gdn_native/solve_br_op/` (sibling of `solve_op/`): a deployable QHPI custom op = HVX int16 diagonal
forward-subst + **two HMX u8i8 64³ merges driven from inside the op** (runtime Crouton/weight/bias packing
in VTCM, no DDR round-trip). Validated on `ssh oneplus` vs `np.linalg.inv` (real C=128 p15_L00, 16 heads):
**whole-T relerr mean 1.11e-2 / max 2.0e-2; device == sim** (M2b sim on same heads = 1.114e-2). T11/T22
diagonals 1.1e-4 (HVX bit-faithful); T21 off-diag ~8.4e-2 (double-int8 merge noise, dominated by the
well-conditioned diagonals in whole-T). Reproduce: `cd example/gdn_native/solve_br_op/standalone &&
EXTRA_DEFS="" H=4 bash gdn_br.sh` (input/ref gen `scripts/gdn_solve_br_probe.py`; bring-up ladder
`-DGDN_BR_SKIP_KERNEL/-DGDN_BR_DIAG_ONLY/-DGDN_BR_DUMP_M`, `-DGDN_BR_PROBE_CYCLES`).
- **Device cyc breakdown (single graph, per head, PROBE_CYCLES) — KEY FOR M3/M4:** HVX diagonals 684K;
  **scalar scale-estimation (2 float 64³ matmuls for s_M/s_T21) + float→int8 packing = 2.27M ← DOMINANT;**
  HMX merges only 275K. The bottleneck is NOT intrinsic merge work — it's the scalar scale-estimation
  matmuls. **M3/M4 must kill these** (HVX-vectorize the pack/requant, and replace the scale-probe matmuls
  with an analytic range bound or fold the range into the drain). Current op is single-HMX-thread, no head
  pipeline, scalar packing → wall high (expected for M_op); ~7–9K target needs the M3 pipeline + this fix.
- **Device gotchas (load-bearing for the productization):**
  - **HMX operands MUST live in VTCM, not BSS.** Static aligned BSS in the HTP package lands in DDR; the
    `mxmem` kernel faults ("Graph Execution failure"). Fix: carve HMX surfaces from a scratch tensor
    declared `QHPI_MemLoc_TCM_Only` — add a 2nd op input `S` (a uint8 zero constant in the graph);
    `qhpi_tensor_raw_data(inputs[1])` returns its VTCM addr (same trick as `HvxHmxOp.cpp`).
  - **`QHPI_RESOURCE_HVX|HMX` (0x6) is REJECTED** at x86 prepare ("invalid resource flag 0x6"), and
    `multithreaded=true` on an HMX op → "Can't set self_slicing on non-HVX op". Fix: declare
    **`QHPI_RESOURCE_HMX` ALONE, `multithreaded=false`** — HVX intrinsics still work inside it. ⇒ **no
    central-tiler head self-slicing; M3 pipeline must use manual qurt threads.**
  - **VTCM buffer spacing matters:** tight packing corrupted the 2nd N-tile of the HMX output; space each
    surface 64 KB apart (the M1 sim layout). The kernel reads/writes with more slack than the 4 KB surface.
  - Converter XML must declare the scratch input (`<Input>S</Input>`) or `qairt-converter` throws
    IndexError; do NOT put a quant override on the uint8 constant `S` (quantizer rejects UINT_8).
  - All packers/descriptor/FLOOR-drain/control-word ported to C **byte-identical** to the M1/M2b sim.

**M3 DONE (2026-06-02) — optimized 13.9× + manual HVX∥HMX threads proven; BUT re-baselining shows the
approach LOSES to the shipped HVX op.** `GdnSolveBR` optimized: killed the scalar scale-estimation
(exact HVX int-matmul max-reduce), tuned-vectorized diagonals + packing. Device cyc/head C=128:
**3,228K (M_op) → 231K (13.9×)**, relerr held 1.11e-2, wall H=4 75.8ms→1.85ms. Breakdown: scale-est+pack
121K, HMX merges 66K (kernel only **1.4K** — free), diagonals 44K, int8 quant 45K. Manual qurt-thread
HVX∥HMX overlap **proven feasible on device** (`qurt_thread_create`+`qurt_hvx_lock` succeed inside the
HMX-resource callback, `-DGDN_BR_THREAD_TEST`) — retires the threading risk.

**Finding (corrected) — C=128 is the WORST case; the win lives at C=256+.** At C=128 `GdnSolveBR` is
231K single-thread (~50–90K pipelined) vs the shipped int16-packed HVX op's **11,425 cyc/head** → loses.
BUT C=128 is the worst shape for this route, and M3 left the two decisive levers undone (packing
elimination + threading). Why C scaling flips it: block-recursive keeps diagonal blocks at **64×64
regardless of C**, so the irreducible HVX diagonal work is `nb × (64×64 solve)` while the baseline HVX does
a full O(C²) solve:
- C=128 (nb=2): diag ≈ 2×3,150 = 6,300 (tuned, 4-thread) vs baseline 11,425 → ceiling only **~1.8×**.
- C=256 (nb=4): diag ≈ 4×3,150 = 12,600; merges (≈6–12× 64³, HMX kernel ~1.4K each, pipelined UNDER the
  HVX diagonals per the 98%-overlap result) → steady ≈ max(HVX≈12.6K, HMX≈17K) vs baseline **70,201** →
  ceiling **~2–4×, growing with C** (diag fraction shrinks: ½ at C=128, ¼ at C=256, ⅛ at C=512).
**Two hard prerequisites for the win (both undone in M3):** (1) **eliminate the runtime Crouton/k-major
packing** (~165K/head at C=128 — the actual dominant cost; the route is packing-bound, not compute-bound)
via a **Crouton-RESIDENT** design (HVX diagonal solve writes its output directly in HMX/Crouton int8 layout,
no per-merge re-gather); (2) **manual qurt-thread HVX∥HMX pipeline** (central tiler can't self-slice an HMX
op; M3 proved manual threads work). The earlier 7–9K/8–10× projection assumed both AND omitted packing — too
optimistic, but **~2–4× at C=256 looks reachable**. `GdnSolveBR` is a device-correct validated artifact
(M1→M_op→M3); M4 tests the actual win regime (C=256 + packing-resident + pipeline).

**M4 DONE (2026-06-02) — C=256 nb=4 block-recursive built, device-correct, fully packing-resident +
single-thread-optimized; VERDICT: does NOT beat the 70,201 HVX baseline (~5.8× slower single-thread,
~2.3× slower even with the achievable HVX∥HMX pipeline). The route loses at C=256.**
- **Algorithm (validated):** nb=4 (BL=64) lower-tri inverse `T_ij = T_ii @ (Σ_{k=j}^{i−1} A_ik @ T_kj)`,
  6 off-diag blocks in increasing i−j, 16 u8i8 64³ HMX merges/head + 4 HVX diagonal solves. Exact formula
  verified vs `np.linalg.inv` (host probe `scripts/gdn_blockrec_c256_probe.py`, relerr 3e-17 unquantized).
  Device whole-T relerr **2.4e-2 mean / 4.5e-2 max** (8 real heads, golden p29_L00) — in the target ~1–2e-2
  range; diagonals bit-faithful 1.1e-4; deep off-diag blocks noisy but negligible in whole-T.
- **Packing-elimination (Task 2) DONE — the route is no longer packing-bound.** Drove single-thread C=256
  from a **4.93M cyc/head** naive baseline → **408K cyc/head (12×)** by killing every float roundtrip and
  scalarism: (a) int-ONLY inner-sum accumulation in int32 codes (no deq-to-float); (b) **2-pass HMX
  scale-estimation** — pack once, run HMX with a cheap `K·max|act|·max|wt|` provisional gain, read max|P|
  from the u8 surface, re-run at the tight gain → replaces the 685K-cyc HVX `pint` 64³ matmul with ~2 free
  HMX passes (→36K); (c) HVX-vectorized k-major weight pack (290K→16K via 2 byte/halfword vshuffs),
  `effective` column-sum (138K→8K), int32→int8 narrow (vpack), and an order-preserving int8→int32 widen
  (4-stream vshuff weave). Final per-head breakdown: diag 124K, quant 90K, hmxpack 64K, pint-2pass 35K,
  depack 30K, hmxkern **14K (free)**, + ~50K fold/acc/requant residual.
- **Pipeline (Task 3) — HMX-from-worker FAULTS; only HVX threads, only ~2.5× → still loses.** Measured on
  device: a spawned qurt worker that calls the `mxmem` kernel triggers "Graph Execution failure" **even at
  NT=1** — the QNN backend grants HMX to the MAIN callback thread only; HMX cannot be driven from a worker.
  (HVX-only workers DO run: DIAG_ONLY 4-thread succeeded.) So the only viable pipeline is HVX-workers +
  HMX-marshalled-to-main. Measured HVX threading speedup is **~2.5× not 4×** (DIAG_ONLY 2.45M→999K
  aggregate; v75 HVX-context contention — the backend already holds HVX contexts). Projected best pipelined
  C=256 = HVX_work(~394K)/2.5 ≈ **158K/head**, HMX(~21K) hidden under it → **~158K, 2.3× SLOWER than 70,201**.
- **WHY it loses (the real finding):** block-recursive replaces the baseline's single clean O(C²) HVX
  forward-subst with **16 small 64³ matmuls whose per-merge HVX glue (quant + crouton/k-major pack +
  requant + depack + fold + int-accumulate, ~4096 elems each) costs MORE than the O(C²) solve it removes.**
  The HMX kernel itself is genuinely free (14K/head), but it's a rounding error next to ~280K of HVX merge
  glue + ~124K diagonals. Total HVX work (~394K) > the baseline's 4-thread-equiv (~280K), so no amount of
  the *achievable* (≤2.5×, HMX-on-main) threading closes the gap. The earlier ~2–4× projection assumed the
  merge glue would vanish into Crouton-residency and a full 4× HVX∥HMX pipeline; in practice the glue is
  irreducible per-merge HVX work and the pipeline is HMX-bound-to-main + HVX-context-limited.
- **Honest best achievable C=256:** ~150–160K cyc/head (HVX-worker pipeline w/ HMX on main, 2.5× HVX) vs
  baseline **70,201** → the shipped int16-packed HVX `GdnSolve` op **wins** at C=256. The block-recursive
  HMX route is NOT the prefill-solve win; **keep the shipped HVX op for prefill C=128/256.**
- **Artifact:** `example/gdn_native/solve_br_op/` now parametric in `GDN_BR_C` (128|256, nb=2|4),
  device-correct at both, all packing vectorized. Single-thread default (HMX on main); `-DGDN_BR_USE_THREADS`
  gates the (faulting) worker-HMX path; `-DGDN_BR_DIAG_ONLY -DGDN_BR_USE_THREADS` runs the working HVX-only
  thread pool. Reproduce: `cd example/gdn_native/solve_br_op/standalone && CB=256 H=8 bash gdn_br.sh`
  (relerr), `CB=256 H=16 EXTRA_DEFS=-DGDN_BR_PROBE_CYCLES bash gdn_br.sh` (per-stage cyc, decode head-0
  uint32 probe). Host formula/scale validation: `GDN_NO_VSCALE=1 uv run python scripts/gdn_blockrec_c256_probe.py --heads 8`.

**M5 DONE (2026-06-02) — the definitive Crouton-resident push.  Both levers driven to a real measured floor;
verdict UNCHANGED: `GdnSolveBR` loses to the 70,201 HVX baseline by ~2.4×.  This is the floor, not a tuning gap.**
Replaced the wall-µs/work-volume estimates with the canonical metric = the chrometrace **op `dur` (concurrent
wall cycles) ÷ heads** (the per-tile steady metric), and instrumented the FULL residual.

- **Full single-thread breakdown, C=256, measured op_dur/head (was estimated at 408K — true op_dur is higher):**
  diag 121K · quant 90K · hmxpack 64K (eff 9K + actpack 39K + wtpack 16K) · fold 63K (DDR-bound A read) ·
  pint-2pass 35K · requant 36K · depack 30K · hmxkern **14K (free)** · acc 12K · widen 4K · zero 4K
  → **op_dur 480,219 cyc/head**.  (Decode head-0 probe: codes = `clip(round(f/sT)+zpT,0,65535)` then
  `view(uint32)`; sT=6.1037e-5, zpT=32768; p[0]=diag…p[16]=zero, all ÷nheads = per-head work-volume.)

- **Lever B (threading) — definitive ceiling = 2.5×, measured on real device, and it is NOT the binding
  constraint.**  Worker-HMX **re-confirmed faulting** (full path `-DGDN_BR_USE_THREADS` NT=2 → RUNFAIL; the
  backend grants HMX to the main callback thread only).  HVX-only threading ceiling, measured as DIAG_ONLY
  **op_dur/head** vs NT (the real concurrent wall, not the summed `g_c_*` work-volume): NT=1 156,654 →
  NT=2 82,778 (1.89×) → NT=3 66,054 → NT=4 59,152 → **2.65× at NT=4** (tuned-diag build: 125K→49.9K = 2.51×).
  v75 HVX-context contention caps it ~2.5–2.65× because the backend already holds HVX contexts.  **Key point:
  even a HYPOTHETICAL full N-way head pipeline with HMX-on-workers (which faults) at 2.65× on the whole
  471K = 178K/head — still loses.**  The binding constraint is the single-thread HVX *work volume* (~457K of
  glue), not the thread count: the baseline does ONE O(C²) HVX solve (70,201, already 4-thread) where BR does
  457K of per-merge glue that out-weighs the threading win.

- **Lever A (glue fusion) — pushed the biggest reducible lever, measured the slope, confirmed it can't close
  2.4×.**  (1) **Diagonal tuned** to the shipped int16-packed forward-subst (ported `Q6_Ww_vmpyacc_WwVhRh`
  64-col/instr + int16 `Tc16` buffer + one `vasr_VwVwR_rnd_sat` narrow, widen to int32 once at the end):
  **diag 121K→92K**, full op_dur **480K→441,570 cyc/head**, relerr bit-identical (diag 1.077e-4, whole-T
  2.378e-2).  The diag floor is **DDR-read-bound on A** (the fold dominates, not the MAC) — the baseline reads
  A once and amortizes over O(C²); BR re-reads A blocks per merge.  (2) Operand-caching analysis (counted
  reuse: 10 inner merges over only **6 distinct** A-folds + **6 distinct** T-weights; 3 reused T_ii acts) ⇒
  max ~56K saveable → ~385K → /2.51 = **153K**, still 2.2× over.  Quant (90K, 32 calls) + pack (64K) + depack
  (30K) are intrinsic per-merge HVX glue, not redundancy.

- **DEFINITIVE NUMBERS (C=256, real v75, steady op_dur/head):**
  | config | single-thread | best-threaded (2.51×, HMX hidden) | vs 70,201 |
  |---|---|---|---|
  | M4 (as committed) | 480,219 | ~191K | 2.7× slower |
  | M5 tuned-diag (current) | **441,570** | **~170K** | **2.4× slower** |
  | M5 + full operand-caching (estimate) | ~385K | ~153K | ~2.2× slower |
  Baseline shipped int16-packed HVX op | — | **70,201** | 1.0× |

- **THE DEFINITIVE BLOCKER (quantified):** NOT the threading ceiling, NOT residual glue tuning, NOT diagonals —
  it is the **raw single-thread HVX work volume of the 16-merge decomposition (~427K HVX after tuning) vs the
  baseline's single O(C²) solve (70,201, already threaded).**  Block-recursion trades one clean O(C²) HVX
  forward-subst for 16 dense 64³ matmuls, and each matmul's *irreducible* HVX glue (quant act + quant wt +
  crouton-pack + k-major-pack + 2-pass scale + depack + widen + requant, ~4096 elems each) sums to far more
  HVX work than the solve it removes.  The HMX kernel is genuinely free (14K), but HMX cannot do a triangular
  solve and cannot run on a worker, so it cannot absorb the glue.  **No achievable combination of the two
  levers crosses 70,201.**

- **HONEST VERDICT:** best achievable C=256 ≈ **150–170K cyc/head** (tuned diag + full glue fusion + 2.5×
  HVX threading + HMX-on-main).  That is **2.2–2.4× SLOWER** than the shipped 70,201.  More is NOT extractable:
  the diag is DDR-bound, the merge glue is intrinsic, threading is HVX-context-capped, and HMX-on-worker faults.
  **This is the floor of the Crouton-resident path.  KEEP the shipped int16-packed HVX `GdnSolve` op for
  prefill C=128/256.**  `GdnSolveBR` is a fully-validated device-correct artifact + a now-quantified negative
  result; the tuned int16-packed diagonal (correctness-preserving) is left in `gdn_solve_diag64`.
  Reproduce M5: `cd example/gdn_native/solve_br_op/standalone && EXTRA_DEFS=-DGDN_BR_PROBE_CYCLES CB=256 H=16 bash gdn_br.sh`
  then decode head-0 probe (uint32 view, ÷16); op_dur/head from `out_s/optrace/chrometrace.json` GdnSolveBR
  `dur`÷16; threading sweep `EXTRA_DEFS="-DGDN_BR_DIAG_ONLY -DGDN_BR_USE_THREADS -DGDN_BR_NT=<1|2|4>"`.

## M6 DONE (2026-06-02) — co-designed two-op SPLIT (HVX-diag op ∥ HMX-merge op, opaque-VTCM handoff).
**Built + device-measured on real v75.  Two NEW findings nail the route; verdict UNCHANGED: the split
also LOSES to 70,201, and the reason is now fully attributed.  KEEP the shipped HVX op for prefill.**

The split = Op1 `GdnSolveDiag` (`example/gdn_native/solve_diag_op/`, pure-HVX, `multithreaded=true`) +
Op2 `GdnMergeHmx` (`example/gdn_native/merge_hmx_op/`, HMX) with the int8 32x32-tile handoff carried in
a 2nd Op1 OUTPUT tensor `Hd` (a real graph edge).  Reproduce:
`cd example/gdn_native/merge_hmx_op/standalone && CB=256 H=16 bash gdn_split.sh` (full-T relerr + per-op
cyc + overlap); `cd ../../solve_diag_op/standalone && CB=256 H=16 bash gdn_diag.sh` (Op1 alone). Add
`EXTRA_DEFS=-DGDN_BR_PROBE_CYCLES` for per-stage work-volume (head-0 uint32, ÷nheads).

**RAW DEVICE NUMBERS (C=256, H=16, real v75):**
- **FINDING 1 — `multithreaded=true` on a PURE-HVX op DOES self-slice across 4 HVX threads (tids 512–515).**
  The central tiler (tiles of 8 heads, `gdn_diag_shape_required`/`build_tile`) makes parallel tile-ops —
  the parallelism the fused HMX op structurally could NOT have (`QHPI_RESOURCE_HVX|HMX`=0x6 is rejected;
  an HMX op can't be `multithreaded`).  Diagonal forward-subst alone (PROBE) = **18,647 cyc/head** vs the
  fused single-thread diag **89–92K/head ⇒ ~4.8× threading recovery, confirmed.**  Op1 also does the
  handoff prep (quant diag→i8 57K + offdiag-A quant 64K + requant 24K + i8 tile-write), so Op1's busiest-
  thread WALL = **167K cyc/head** (not just the 18.6K solve).  Diag bit-faithful (relerr 1.08e-4 in T1).
- **FINDING 2 — the opaque-VTCM handoff AVOIDS QNN's ForceFormat tax (the M6-v1 ~2M killer is GONE).**
  Optrace node list between Op1→Op2 has **NO `ForceFormat_Crouton`/`convert_weights`** — only `@Fill`/
  `@Spill` (≈17K/head total) + `flat_from_vtcm` (98K aggr) + `Concat` (67K aggr).  Confirms the milestone
  premise: a 32x32 tile == row-major, so the int8-tile handoff needs no format conversion.  **No adapter
  op needed.**  BUT the handoff MUST be VTCM-resident: as a plain DDR tensor edge the un-tile READ cost
  **1.85M cyc/head** (the DDR-round-trip catastrophe); declaring `Hd`/Op2-input `QHPI_MemLoc_TCM_Only`
  + a vectorized un-tile dropped it to **234K → wall 20,829µs → 7,426µs**.
- **THE TWO KILLERS that still sink the split:**
  1. **NO Op1∥Op2 overlap (measured 0%).**  As two graph nodes joined by the `T1→Op2` data edge, QNN runs
     them strictly SERIAL (Op2 starts only after ALL of Op1).  The 98%-overlap result (`gdn_overlap.sh`)
     was for HVX+HMX *inside one op*; splitting into two nodes does NOT pipeline across heads — the
     central scheduler has no per-head software-pipeline.  So total ≈ Op1 + Op2 (sum), not max.
  2. **Op2 is single-HMX-thread (524K cyc/head).**  HMX ops can't self-slice and the worker-HMX path
     faults (M4/M5), so Op2 processes all 16 heads on tid 256 serially.  Its work is dominated by the
     per-merge HVX glue that MOVED but did NOT shrink: read/un-tile 234K + quant 33K + actpack 39K +
     hmxpack 24K + wtpack 16K + pint 13K + depack 11K + hmxkern only **5.3K (free)**.
- **TOTAL: 167K (Op1) + 524K (Op2) serial = 740K cyc/head WALL** (combined span /16) vs baseline
  **70,201 ⇒ ~10.5× SLOWER.**  Even crediting a hypothetical full Op1∥Op2 overlap (max not sum) = 524K
  (Op2-bound) ⇒ still ~7.5× slower; even Op2's 4-thread-equivalent fantasy (524/4≈131K) + Op1 overlapped
  ⇒ still ~1.9× slower.  Accuracy: C=128 single-merge full-T relerr **1.8e-2 (PASS, ~baseline)**; C=256
  diag 9.4e-3 (int8 handoff) but the multi-level d≥2 merge chain is noisy (deep blocks ≈0 in ref so harmless
  in abs/whole-T, but whole-T 7.3e-2 > fused 2.8e-2 — int8 diagonal act in the final merge is the regression).
- **WHY the split doesn't rescue the route (the definitive attribution):** the SPLIT correctly (a) recovers
  4× HVX threading for the diagonals and (b) eliminates the ForceFormat handoff tax — the two things M4/M5
  lacked.  It does NOT help because the irreducible cost was never the diagonals or the format tax: it is
  the **per-merge HVX glue volume (quant+pack+un-tile+depack+requant ≈ 380K/head across the 16 merges)**,
  which is single-thread-bound in the HMX op (Op2) and out-weighs the baseline's one O(C²) solve — exactly
  M5's "definitive blocker", now re-confirmed from the opposite (split) direction.  Splitting moved the
  glue's diagonal portion onto threads but left the merge glue serial in the HMX op, and added a serial
  Op1→Op2 boundary with no pipeline.  Artifacts (`solve_diag_op/`, `merge_hmx_op/`) are device-correct at
  C=128 and stand as the validated negative result for the co-designed-split idea.

## M7 (2026-06-02) — round-2 split squeeze: Op1 tile-write vectorized (−100K/head); 1-pass scale fails accuracy.
RAW device numbers (C=256, H=16, real v75, per head; busiest tile/8 for Op1, max_dur/H for the un-tiled Op2).

- **Op1 lever A — the `gdn_write_tiles_64` natural→(4×32×32-tile) re-pack was a SCALAR byte-scatter (4096
  byte-copies × 10 blocks/head).  Vectorized it with one `Q6_W_vdeal_VVR(v1,v0,-32)` per 4 rows (32-byte-
  granularity deal of a vector pair → lo=tileL rows, hi=tileR rows).  Bit-identical layout (split full-T
  relerr UNCHANGED 7.17e-2).**  Isolated Op1 (`gdn_diag.sh`, busiest tile/8): full **183,846 → 59,314**
  (skip-tilewrite ablation = 69,309 ⇒ scalar re-tile alone WAS ~114K/head; now ≈0).  This was the bulk of
  the task's "off-diag A-quant ~127K" — it was the re-tile, not the fold/quant (both already vectorized).
- **Split totals after lever A (2-pass scale, default, accuracy-neutral):**
  | C | Op1/head | Op2/head | serial total | relerr | vs baseline |
  |---|---|---|---|---|---|
  | 256 | ~59–66K | ~180K | **~238–246K** | 7.17e-2 | 3.4× slower (base 70,201) |
  | 128 | ~38–48K | ~24K  | **~62–72K**   | 1.48e-2 (PASS) | 5.5× slower (base 11,425) |
  Down from the task's round-1 362K (C=256).  The mover was entirely Op1's scalar re-tile.
- **Op2 breakdown (C=256, ablation):** DIAG_ONLY (T1→T copy only, no merges) = **26,587/head** (DDR-bound
  128KB int16 T1 read+write); the **16 merges = ~152K/head** (~9.5K each: i8→u8 recenter + crouton/k-major
  pack + 2-pass HMX scale + depack + accumulate + requant — all already HVX-vectorized, intrinsic per-merge
  glue).  Op2 is single-HMX-thread (can't self-slice, worker-HMX faults) so all 16 heads serial on tid 256.
- **Lever B (1-pass HMX scale via analytic bound) — MEASURED NEGATIVE, accuracy fails at BOTH C.**  Dropping
  pass-1 (provisional HMX run + surface max|P| scan) saves only ~18K/head on Op2 (180K→161K @C=256; the 2
  HMX runs are ~free, the saving is the surf scan + bias repack), but the analytic gain coarsens every
  merge's int8 output scale:
  | bound | scale slack vs true max\|P\| | C=256 whole-T relerr | C=128 whole-T relerr |
  |---|---|---|---|
  | loose `K·amax·wmax` | ~66× (~6 bits) | 1.77 | — |
  | tight `amax·max_c Σ_k\|wt\|` (added) | ~2.7× mean / 4.7× max (~1.5 bits) | 0.239 | 0.134 |
  | 2-pass true max\|P\| (kept) | 1.0× (full 127 levels) | **7.17e-2** | **1.48e-2** |
  Even the tight bound (HVX column-sum `gdn_pint_tightbound`/`_tiled`, always ≥ true so no saturation) breaks
  the merge chain: the d≥2 merges multiply already-quantized d=1 outputs, so ~1.5 bits of per-merge scale
  slack compounds catastrophically (C=256 d≥2 heads → 0.4–0.5).  Even C=128 (single d=1 level) goes
  1.48e-2 → 0.134.  The 2-pass per-merge TRUE-max scale is load-bearing.  Gated OFF by default
  (`-DGDN_MERGE_1PASS` opt-in for the record).
- **Standing verdict UNCHANGED:** the split at C=256 is **~3.4× slower** than the shipped 70,201 HVX op
  (was ~10.5× in M6-v1, ~5.2× at the task's round-1 362K).  The two architectural killers remain: NO Op1∥Op2
  overlap (serial graph edge) and Op2 single-HMX-thread (~180K serial); the ~152K of per-merge HVX glue is
  intrinsic, and lever B can't shrink it without losing accuracy.  KEEP the shipped int16-packed HVX
  `GdnSolve` op for prefill.  Op1's vectorized `gdn_write_tiles_64` + the `gdn_pint_tightbound[_tiled]`
  helpers are left in place (the former is a pure correctness-neutral win, the latter for the negative-result
  record).  Reproduce: `cd example/gdn_native/merge_hmx_op/standalone && CB=256 H=16 bash gdn_split.sh`
  (per-op cyc now printed); `cd ../../solve_diag_op/standalone && EXTRA_DEFS=-DGDN_DIAG_SKIP_TILEWRITE
  CB=256 H=16 bash gdn_diag.sh` (tile-write ablation); add `-DGDN_MERGE_1PASS` to gdn_split.sh for lever B.

## METHOD MANDATE (2026-06-02/03) — every perf step: (1) ASCII timeline AND (2) compiled op-graph vs expected
**(1)** Parse `chrometrace.json` and draw the complete per-thread ASCII timeline (HVX 512–515 / HMX 256 rows,
op spans + GAPS, + QNN boundary ops Spill/Fill/Concat/flat_from_vtcm). Aggregate per-stage cyc HID the facts
that decide #2; the timeline showed them at a glance.
**(2)** ALSO parse `chrometrace_htp.json` (`graph.nodes` input_names/output_names) into the actual
op-dependency DAG and DIFF it against the op-graph you EXPECTED (draw expected in ASCII first). The HTP
compiler fuses/inserts ops AND **dedups byte-identical constants into one shared tensor** — which silently
merged the M8 per-chunk `Hs_0`/`Hs_1` scratch into a shared `$Const_17` → false cross-chain dep →
serialization (the "no overlap" wrongly blamed on QNN). netron on the `.onnx` shows the PRE-compile graph
only; `chrometrace_htp.json` is the authority for what actually ran. Renderer/rationale in skill
`qnn-htp-profiling` + [[feedback_always_render_ascii_timeline_from_trace]].

## M8 (2026-06-03, in progress) — ROOT CAUSE found via the timeline: batching all heads through one Op1→Op2 boundary
The M6/M7 split's real trace (`merge_hmx_op/.../out_s/optrace/chrometrace.json`, C=256 H=16, total span
4,907,214 = 306,700/head) ASCII timeline:
```
Op1 GdnSolveDiag (HVX) ████████████                                  span 33k–1,538k  (~94K/head)
boundary Spill/Fill/Concat/flat_from_vtcm   ▓▓▓▓▓▓                    971k–1,925k      (~58K/head)
Op2 GdnMergeHmx (HMX)               ██████████████████████████████   1,925k–4,772k    (~178K/head)
overlap = 0%
```
**Both pathologies have ONE root cause (user insight): all H=16 independent heads go through a single
Op1→Op2 boundary.** ⇒ (a) Op2 waits for ALL of Op1 → **0% HVX∥HMX overlap**; (b) all 16 heads' handoff
intermediate is materialized at once → exceeds VTCM → **~545K @Spill/@Fill** (Spill 273K + Fill 271K) in
the gap. Heads are independent, so the fix is **per-head / small-chunk pipelining**: emit ceil(H/CK)
independent `GdnSolveDiag(chunk)→GdnMergeHmx(chunk)` chains so QNN overlaps chain-i's HMX Op2 with
chain-(i+1)'s HVX Op1, and only one chunk's (~40KB) intermediate is live (fits VTCM, no spill). Expected:
kill the ~58K/head boundary glue + overlap Op1(94K) under Op2(178K) → ~178K/head, then attack Op2's
single-thread merge glue.

**M8 DONE (2026-06-03) — per-chunk pipeline MEASURED on device (C=256, H=16). DDR-spill half CONFIRMED,
overlap half REFUTED.** `scripts/gdn_split_probe.py` (per-chunk via `GDN_CK`) + fast harness
`merge_hmx_op/standalone/gdn_split_sweep.sh` (build ONCE + loop CK; replaces the wasteful per-CK rebuild —
sweep now ~minutes) + `scripts/gdn_timeline.py` (the mandated ASCII-timeline renderer). Verified:

| CK | chains | boundary Spill/Fill | HVX∥HMX overlap | cyc/head | vs 70,201 |
|---|---|---|---|---|---|
| 16 | 1 | 56,041/head | 1% | 287,571 | 4.10× |
| **8** | **2** | **0 (vanished)** | **1%** | **239,501** | **3.41×** |
| 4 | 4 | — | — | **HANG** (DDR thrash) | — |
| 1 | 16 | — | — | **HANG** | — |

- **DDR-spill root cause CONFIRMED:** 1→2 chains drops boundary Spill/Fill from **56K→0/head** (smaller
  per-chunk handoff stays VTCM-resident) → 287K→239K. The user's insight was right.
- **Overlap REFUTED (decisive new finding):** QNN **serializes the independent chains — overlap stays 1%**
  even at 2 chains. QNN's scheduler does NOT stream/overlap a **custom** consumer op. (The earlier 98%
  overlap was `GdnSolve(HVX)→NATIVE MatMul(HMX)` — QNN fine-grain-streams a *native* consumer; a custom
  merge op reads its full input before starting → serial. So you can have a cheap custom merge OR overlap,
  not both: native MatMul overlaps but re-imposes the ~2M ForceFormat tax.)
- **≥4 chains DDR-thrash-HANG:** multiple chains' TCM_Only handoff + HMX scratch can't all stay VTCM-
  resident → un-tile falls into the billion-cycle DDR path; ctxgen's static `spill_bytes=0` doesn't predict
  it. Usable window = 1–2 chains.
- **Gotcha (cost a debug cycle):** a killed/interrupted `qnn-net-run` holds the HMX lock and wedges ALL
  subsequent graphs ("Graph Execution failure") until `pkill -9 -f qnn-net-run` on device.

**Net for #2 (all architectures now measured):** the route floors at **~3.4× over the 70,201 baseline**
(239K/head, CK=8). The 2× HVX∥HMX-overlap lever is **unreachable** — QNN won't overlap a custom HMX op,
manual worker-HMX faults, and the merge stays single-HMX-thread. The shipped int16-packed HVX `GdnSolve`
op (70,201) remains the prefill optimum. The block-recursive HMX route = device-correct validated artifact
+ rigorously-quantified negative result, redundancy fully squeezed (740K→239K), floor is structural.

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

# M6 two-op SPLIT (HVX-diag op ∥ HMX-merge op):
cd ../../merge_hmx_op/standalone
CB=256 H=16 bash gdn_split.sh                                  # full-T relerr + per-op cyc + Op1/Op2 overlap
EXTRA_DEFS=-DGDN_BR_PROBE_CYCLES CB=256 H=16 bash gdn_split.sh # per-stage work-volume (head-0 uint32 ÷16)
EXTRA_DEFS=-DGDN_BR_DIAG_ONLY    CB=256 H=16 bash gdn_split.sh # diag-only (handoff diag precision)
cd ../../solve_diag_op/standalone
CB=256 H=16 bash gdn_diag.sh                                   # Op1 alone: multithreaded diag, 4 HVX tids
```
