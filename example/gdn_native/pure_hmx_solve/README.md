# pure_hmx_solve — isolated "pure HMX (all-w16a16)" GDN inverse experiment

Self-contained experiment that measures the **speed + per-thread timeline** of doing the GDN
C=256 triangular inverse the **pure-HMX** way: the diagonal 64-blocks via Taylor(p=3)+Newton(K=4)
= 11 w16a16 64³ matmuls/block (44/head) instead of HVX forward-subst, plus 16 off-diag merge
matmuls = **60 w16a16 64³ matmuls/head**, all on the single HMX consumer with 4 HVX producers.

Kept **out** of the shipping solve (`GdnSolveBR16.cpp` / `gdnbm_imp.cpp`); all logic lives in
`pure_hmx_solve.cpp`. The skel only gets a 3-line guarded hook (`-DGDNBM_PURE_HMX_SOLVE`).

## Result (real device, oneplus v75, 32-head TOTAL wall)

| | HMX consumer | 4× HVX producers | 32-head wall |
|---|---|---|---|
| **pure HMX (this)** | **busy 98%** (pinned) | **SPIN 96%** (idle) | **~16.9M cyc** |
| GDNSolveHVXMixHMX (shipping) | busy 7% | busy ~88% | 1.78M cyc |

→ **pure HMX is ~9.5× slower.** The timeline shows why: all 60 matmuls (44 of them the expensive
w16a16 diagonal) serialize on the one HMX unit (`CONS` row a solid `mmmm…`), so the 4 HVX units sit
96% idle (`SPIN`). It is the exact inverse of the shipping pipeline (HVX-bound, HMX 7%).

Floor cross-check: 1920 back-to-back w16a16 64³ matmuls = 15.4M (single-thread), full pipeline = 16.9M.

**Numerics are separately KNOWN-BROKEN** (int16 matrix-power inverse overflows ~15% of high-‖A‖₂
diagonal blocks — `scripts/gdn_solve_e2e_precision_probe.py`). This bench measures speed only
(matmul cycles are data-independent), so it runs with garbage operands.

## Reproduce

```bash
bash example/gdn_native/pure_hmx_solve/run.sh        # build + device + render timeline
```
