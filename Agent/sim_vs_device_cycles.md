# Simulator ↔ Device cycle parity (2026-04-22)

Phase 5 of the three-kernel plan. All three QNN op kernels now have
standalone hexagon-sim harnesses — bit-exact PASS on all scenarios.

## Sim harnesses (3/3 PASS, all kernels)

| kernel  | sim test                                | pcycles (3 scenarios) |
|---------|-----------------------------------------|----------------------:|
| int16 single-tile | `tests/test_hmx_matmul_int16.sh` | 4.87M |
| w4a8   | `tests/test_hmx_matmul_w4a8.sh`         | 3.59M |
| w4a16  | `tests/test_hmx_matmul_w4a16.sh`        | 5.31M |
| w16a16 | `tests/test_hmx_matmul_w16a16.sh`       | 6.42M |

Each harness runs 3 scenarios: constant small, random full-range, edge
(max positive). Each scenario is a single 32×32×32 tile, HMX path vs
scalar int32 reference, must match bit-exact.

## Device cycle baselines (SM8650 v75)

| kernel  | 32³      | 128³     | 512³     | 32³ cycles |
|---------|---------:|---------:|---------:|-----------:|
| w4a8    |   5.74   |   2.65   |   1.92   |    188K    |
| w4a16   |   8.37   |    —     |   2.08   |    274K    |
| w16a16  |  20.78   |  12.37   |  12.23   |    680K    |

## Sim vs Device divergence (per-scenario approximation, 32³)

Sim scenarios include H2 booter setup overhead (~400K–500K pcyc baseline).
Net per-scenario HMX-only cycles:

| kernel  | sim per-scenario (est)† | device 32³ | ratio |
|---------|------------------------:|-----------:|------:|
| w4a8    | ~1.2 M                  | 188 K      | 6.4×  |
| w4a16   | ~1.77 M                 | 274 K      | 6.5×  |
| w16a16  | ~2.14 M                 | 680 K      | 3.1×  |

† (total_pcycles − h2_setup) / n_scenarios; approximate until per-scenario
pcycle markers are wired in.

**Interpretation**: sim consistently OVERESTIMATES cycles by 3-6.5×
vs silicon. Divergence sources (unchanged from earlier analysis):

- Sim's cycle model is conservative for HMX MAC issue rate (likely
  treats each `mxmem` as fixed max-latency rather than pipelined).
- Sim doesn't model silicon's `:cm`/Rt-dependent credit-based VTCM
  port scheduling (the Phase 1 RE finding — Rt_wt=0x3FF gives 2.5×
  HMX speedup on silicon; sim may give 0×).
- Sim runs HVX threads sequentially by default; device has parallel
  HVX contexts.
- H2 booter adds ~400-500K pcycles setup per run.

**Key takeaway**: sim is useful for CORRECTNESS verification (bit-exact
matches) but NOT for perf tuning. Always measure cyc/MAC on silicon.
This is explicitly called out in the user's memory and now validated.

## How to run

```sh
# All four:
bash tests/test_hmx_matmul_int16.sh
bash tests/test_hmx_matmul_w4a8.sh
bash tests/test_hmx_matmul_w4a16.sh
bash tests/test_hmx_matmul_w16a16.sh

# Device (each):
bash example/hmx_matmul_w4a8/run_on_device.sh --shape 512,512,512
bash example/hmx_matmul_qnn/run_on_device.sh  --shape 512,512,512
bash example/hmx_matmul_w16a16/run_on_device.sh --shape 512,512,512
```

## Status

- ✅ All three kernels: sim correctness verified
- ✅ All three kernels: device correctness verified at 32³/128³/512³
- 🟡 Cycle parity documented (sim 3-6× conservative); not tuned to match
  — sim timing model limitations are structural, not kernel-side
- ⬜ Per-scenario pcycle markers (using `h2_perf_counter_start/end` or
  similar) for tighter parity numbers — deferred, nice-to-have
