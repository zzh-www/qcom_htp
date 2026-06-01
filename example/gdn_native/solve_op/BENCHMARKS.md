# GdnSolve — compute-cycle benchmarks (v75 HVX)

Performance data for the GdnSolve triangular-solve op `T=(I-A)^-1`, A strictly-lower int16 `[1,H,C,C]`,
H=32 heads.

**Primary metric = DOMAIN CYCLE (real performance), not total cycle.** The hardware units run in parallel
(HMX ∥ the 4 HVX threads), so the *real time* of a solve is the busiest unit's cycles, NOT the sum. Total
cycle = sum over all units/threads = **work volume only**. We report per-domain (HMX, HVX) cycles for the
whole 32-head solve (4 HVX tiles run in parallel → one tile's warm cycles = the 32-head real time).
NOT wall — wall carries a ~1.25 ms fixed graph/dispatch/IO overhead that amortizes away in a real
multi-layer model. Warm (cold-start excluded); ~±10% measurement noise.

## Table 1 — forward-substitution kernel vs chunk size C

The solve as the HVX forward-substitution custom op (this kernel), at chunk sizes C=16/32/64 (the `C` is
the GDN chunk length). `per-token = H·(cyc/head)/C` is the deployment metric (chunk size sets tokens/chunk).

| chunk C | **HVX domain (real time, 32-head solve)** | HMX domain | total cyc (work) | per-token (real) |
|---|---|---|---|---|
| 16 | **~3,060** | 0 | ~12,500 | ~190 |
| 32 | **~7,940** | 0 | ~32,000 | ~250 |
| 64 | **~29,500** | 0 | ~118,000 | ~460 |

(domain = real time; total = work volume = sum over the 4 HVX threads ≈ 4× domain. per-token = HVX
domain / C tokens.) The solve is **HVX-only — HMX domain = 0** (idle).

- **cyc/elem floor ≈ 1.0** for C=32/64 — the AXPY is MAC-throughput bound (`vmpyiacc` uses both HVX
  multiply resources → 1 MAC/packet). C=16 stays ~1.5: small 16×16 matrices have proportionally more
  per-row/per-matrix overhead (fold, requant, de-interleave) that doesn't amortize, even after the 2-head
  lane-packing removed the lane waste (2.0→1.5).
- **per-token favours small chunks** (C=16 ~780 < C=32 ~1,000 < C=64 ~1,850): the solve is `H·C²/2` MACs
  per token, so larger chunks cost more per token even though they're more cyc/elem-efficient.

Reproduce: `CS="16 32 64" bash example/gdn_native/solve_op/standalone/gdn_shape.sh`

## Table 2 — forward-substitution vs iterative-multiply (u8×i8) at C=32, with domain split

The alternative: solve via a chain of matmuls (squaring `T=∏(I+A^(2^k))`, ~8 matmuls for C=32) at u8×i8
(uint8 act × int8 wt, the "fast" HMX primitive). Domain = HMX (matmul compute) vs HVX (lowering glue).
Op/step level, per head, graph I/O excluded for a like-for-like compare.

Domain cycle = real time (HMX ∥ HVX), 32 heads, 1 inference.

| C=32 approach | **real time = max(HMX,HVX)** | HMX domain | HVX domain | total cyc (work) |
|---|---|---|---|---|
| **forward-subst (this op)** | **~7,940** (HVX) | 0 | ~7,940 | ~32,000 |
| u8×i8 matmul — **1 step** | **~45,000** (HVX) | 31,972 | **44,984** | ~194,000 |
| u8×i8 iterative — **~8 steps** | **~360,000** | ~256,000 | ~360,000 | ~1,550,000 |

- **In real time (domain cycle): one u8×i8 matmul step (~45,000) ≈ 5.7× the entire forward-subst solve
  (~7,940); the full iterative solve (~8 steps) ≈ 45×.**
- **The matmul is HVX-glue bound, not HMX bound:** its HMX domain (31,972) is *less* than its HVX domain
  (44,984), so HMX∥HVX overlap doesn't help — the real time is set by HVX. The HVX work is QNN's matmul
  lowering glue (`convert_weights_to_signed` + `Crouton` + `bias_weight_update`, re-done every step
  because the operand is a dynamic activation), not real compute. So "u8×i8 is fast on HMX" is a mirage:
  the matmul barely benefits from HMX and is bottlenecked on HVX glue.
- The forward-subst op wins by having **HMX domain = 0 and zero matmul glue** — one self-contained HVX op
  emits T directly, so its HVX domain (~7,940) is the whole cost.

Reproduce: `CS=32 H=32 bash example/gdn_native/solve_op/standalone/gdn_mm.sh`

## Kernel optimization summary (this op, cyc/head)

| | C=16 | C=32 | C=64 |
|---|---|---|---|
| baseline (scalar dequant/quant, fp32) | 516 | 1,194 | 4,762 |
| **optimized (final)** | **~390** | **~1,000** | **~3,700** |

Optimizations applied (all device-validated, oc 1.23e-2 PASS on the full GDN graph):
1. **Pure-integer** fold: float scale → fixed-point `(M,S)` multiplier (computed once/call); AXPY is
   `int16×int16→int32` via `Q6_Vw_vmpyiacc_VwVwRh`; requant is an arithmetic shift (`scale folded into A`).
2. **C=16 2-head packing**: interleave 2 heads (even/odd lanes) so a 16-col row fills the full 32-lane
   vector, using `vmpyi_VwRh`'s both-halfword behaviour (low half = head A, high half = head B).
3. **Pre-replicated Afx** (fused into fold): removes the per-k `*0x10001` scalar multiply from the AXPY.
4. **Aligned Tc loads**: C=16 routes through the pair path, so the head path only runs C%32==0 → Tc rows
   are 128-aligned → aligned loads (the biggest single win, esp. C=64).
5. Latent **scratch race fixed** (claim/release slot per call) and **accumulator stack-spill fixed**
   (scalar locals, not arrays).
