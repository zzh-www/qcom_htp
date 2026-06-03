# GDN triangular-inverse solve — locked baselines

`T = (I − A)⁻¹`, C=256, H=32 (one prefill chunk). Real device `ssh oneplus` (v75, 8 Gen 3).
Metric: aligned PCYCLE (QHAS `cycles` == C15:14, no conversion). PCYCLE/µs ≈ 1422 @ TURBO.
Times below = full 32-head workload (`= per-head × 32 ÷ 1422`).

Four implementations are kept and frozen here; everything else was cleaned/archived.

| name | engine / threads | algorithm | 4-thread wall | oc (gate ≤2.4e-2) | code / build |
|---|---|---|---|---|---|
| **qnn_hvx_int16** | QNN-tiled 4×HVX | int16 forward-subst | ~4.3 ms | (shipped ref) | `solve_op/` |
| **bm_hvx_int16** | bare-metal 4×HVX | block-recursive, int16 matmul (4-acc) | ~2.8 ms | ≤ 2.8e-3 (≤ int8; more precise) | `baremetal/` default |
| **bm_hvx_int8** | bare-metal 4×HVX | block-recursive, int8 `vrmpy` + fold-free | ~2.0 ms | **2.816e-3** | `baremetal/` `-DGDN_BR_MM_I8` |
| **bm_hmx_int8** | bare-metal / QNN, HMX | block-recursive, int8-HMX crouton merge | ~9 ms (1-thr) | 1.285e-2 | `solve_br_op/` default (HMX path) |

> oc measured on the hard golden p29_L00 (absmax 0.94). The HVX paths' oc MUST be measured with
> `-DGDN_BR_HVX_MERGE` (the QNN op default omits it → measures the HMX path; that mislabel is why
> "1.285e-2" was wrongly attributed to int8 earlier — int8-HVX is actually 2.816e-3, more accurate).

> `bm_hvx_int16` / `bm_hvx_int8` / `bm_hmx_int8` are all the SAME source
> `solve_br_op/src/GdnSolveBROp.cpp`, selected by compile flags. `qnn_hvx_int16` is the separate
> shipped op in `solve_op/`.

## Build / run each

```bash
# bm_hvx_int8  (fastest, the win)
cd example/gdn_native/baremetal
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDN_BR_MM_I8" bash build.sh
#   deploy build/{libgdnbm_skel.so,gdnbm} to $HOME/gdnbm_run on ssh oneplus, then:
#   ./gdnbm 4 A_u16_h32.raw T.raw 32 256 32768 32768 2.770166930875267e-05 6.103701895199438e-05

# bm_hvx_int16  (drop -DGDN_BR_MM_I8)
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT" bash build.sh

# bm_hmx_int8  (recursive + HMX scheme; the GDN_BR_HVX_MERGE-off path, kept for reference)
#   QNN op default build IS this path:
H=32 CB=256 bash example/gdn_native/solve_br_op/standalone/gdn_br.sh

# qnn_hvx_int16  (shipped GdnSolve)
H=32 CB=256 bash example/gdn_native/solve_op/standalone/gdn_shape.sh   # (or its bench harness)

# Perfetto trace for a bare-metal baseline (QNN-optrace schema):
EXTRA_DEFS="-DGDNBM_VTCM_RESIDENT -DGDN_BR_MM_I8 -DGDN_BR_TRACE" bash build.sh   # run 4-thread, fetch T.raw
python scripts/gdn_baremetal_trace.py T.raw <name>/chrometrace.json
```

## Per-baseline artifacts (this dir)
- `<name>/chrometrace.json` — Perfetto trace (QNN-optrace schema). Load at https://ui.perfetto.dev.
- `<name>/NUMBERS.txt` — measured wall (µs/ms), per-head cycles, oc/relerr.
- Input data: `baremetal/A_u16_h32.raw` (bare-metal), `solve_br_op/standalone/A.raw` (QNN op golden).

## Accuracy notes (corrected 2026-06-04)
- The QNN op `gdn_br.sh` runs WITHOUT `-DGDN_BR_HVX_MERGE` by default → it exercises the **HMX path**
  (`bm_hmx_int8`), not the HVX paths. To validate the HVX paths' oc on the hard golden you MUST pass
  `-DGDN_BR_HVX_MERGE [-DGDN_BR_MM_I8]`.
- Direct bare-metal raw-T relerr vs `np.linalg.inv` on `A_u16_h32.raw` (well-conditioned): int8 = 1.2e-3,
  int16 = 9.4e-5. int8 is same precision-class as the HMX path (int8 operands) which passes oc 1.285e-2.
