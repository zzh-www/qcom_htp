# V9 adaptive MatMul — shape sweep results (2026-04-25)

V9 = `gen_v8_graph.py` shape-adaptive generator. SM8650 v75, w8a8.

## 1. V9 vs QNN native across 4 shapes

| Shape | V9 cycles  | QNN cycles | V9/QNN | V9 cyc/MAC | QNN cyc/MAC |
|-------|-----------:|-----------:|-------:|-----------:|------------:|
| 512³  |    540,407 |     66,523 | **8.1×** | 4.22e-3  | 5.2e-4 |
| 1024³ |  3,077,876 |    182,208 | **16.9×** | 3.01e-3 | 1.8e-4 |
| 2048³ | 21,124,549 |  1,420,662 | **14.9×** | 2.58e-3 | 1.7e-4 |
| 4096³ | 161,528,782 | 28,875,162 | **5.6×** | 2.47e-3 | 4.4e-4 |

V9 efficiency (cyc/MAC) improves with shape (fixed overhead amortizes):
4.22 → 3.01 → 2.58 → 2.47. Plateau around 2.5 — that's our current floor.

QNN efficiency has a sweet spot at 1024-2048³ (0.17-0.18 cyc/MAC).
**QNN degrades at 4096³** (0.44 cyc/MAC, 2.6× worse than its best) because
spill/fill overhead kicks in. **Our gap shrinks from 17× to 5.6×** exactly
because QNN loses some advantage at scale.

## 2. V9 planner behavior across shapes

All 4 shapes use the same `M_TILE=256` (matches QNN). N_TILE stays 256
because VTCM budget isn't tight enough to force shrinking.

| Shape  | M_ROUNDS | N_ROUNDS | total MatMulV8 | Total nodes | spill_bytes | fill_bytes |
|--------|---------:|---------:|---------------:|------------:|------------:|-----------:|
| 512³   |        2 |        2 |              4 |          15 |           0 |          0 |
| 1024³  |        4 |        4 |             16 |          37 |           0 |          0 |
| 2048³  |        8 |        8 |             64 |         105 |   8,781,824 |  31,326,208 |
| 4096³  |       16 |       16 |            256 |         337 |  62,849,024 | 451,870,720 |

**VTCM overflow threshold**: between 1024³ and 2048³. At 1024³ the 16
packed_act/packed_wt intermediate tensors (~2 MB total) fit; at 2048³
they don't (~8 MB total at N_ROUNDS=8 simultaneous live tensors).

## 3. Takeaways

### V9 design works
- 4 shapes run end-to-end without crashes (monolithic V8 OOM'd at ≥4096³).
- @Spill/@Fill **does** get auto-inserted for custom ops when graph has
  enough instances (proves my earlier "QNN won't insert Spill/Fill for
  custom ops" claim was wrong — it WILL, given enough tile granularity).

### Remaining gap analysis
V9's 2.5 cyc/MAC floor × QNN's 0.17 best = **~15× per-MAC gap**. Split:
- **HMX MAC kernel**: per-mmv8-call V9 is 2.12 cyc/packet vs QNN 0.8
  (QNN hits silicon ceiling 7.89 and amortizes pack over longer N).
  ~2.5× gap from HMX kernel alone.
- **Pack/unpack**: V9's `pack_act` at 600K cyc/call vs QNN
  `ForceFormat_Crouton` ~5K cyc/call. 120× per call, but fewer calls
  due to MT=true auto-parallelism.
- **Framework/concat/spill overhead**: scales super-linearly at 4096³,
  partly explaining why QNN also slows down at that shape.

### Priorities
1. **HVX vshuff pack rewrite** (blueprint §5#4) — biggest single-lever
   target, should close 3-5× on total cycles.
2. **Smaller N_TILE at large shape** — 4096³ compiler had to do huge
   spill/fill; shrinking N_TILE to 64-128 would produce more granular
   instances and less simultaneous VTCM pressure. Copy QNN's strategy.
3. **Revisit mmv8 inner loop for large K** — at K=4096 each mmv8 call
   runs 64 K-tile iterations; may have cache-line or bank-conflict
   penalties that shorter K didn't expose.

### Files

- `example/hmx_matmul_phase3/standard_flow/phaseB_v8/gen_v8_graph.py`
  shape-adaptive generator
- `example/hmx_matmul_phase3/standard_flow/phaseB_v8/sweep_v9.sh`
  sweep driver (SHAPES=… bash sweep_v9.sh)
- `sweep_v9/s{512,1024,2048,4096}/` — per-shape ONNX, DLC, ctx, profile
- `example/qnn_matmul_profile/sweep_data_{1024,2048,4096}/w8a8/` — QNN
  baselines

## 4. Known issues

### `--profiling_option optrace` fails on multi-instance graphs
Monolithic V8 (1 MatMulV8 instance) worked with optrace. V9 (≥4 instances)
consistently fails at `Graph Execution failure` when optrace enabled;
`--profiling_level detailed` (without optrace) works fine. Likely an HTP
optrace collection incompatibility with custom-op graphs that have many
small instances + compiler-inserted spill/fill. **Workaround**: use
`--profiling_level detailed` for cycle counts; chrometrace not available.

## 5. Reproduce

```bash
cd example/hmx_matmul_phase3/standard_flow/phaseB_v8
bash sweep_v9.sh                            # defaults 512 1024 2048 4096
SHAPES="2048 4096 8192" bash sweep_v9.sh    # override
```
